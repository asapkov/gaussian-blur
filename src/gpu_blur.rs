//! GPU-accelerated Gaussian Blur using wgpu with optimized multi-shader approach
//!
//! This module implements a multi-strategy Gaussian blur using WebGPU:
//! - Small sigmas (< 2.0): True Gaussian convolution
//! - Medium sigmas (2.0-32.0): 3-pass box blur approximation  
//! - Large sigmas (> 32.0): Downsample -> blur -> upsample pipeline
//!
//! All shaders assume a workgroup size of 16x16 threads.
//! All textures use Rgba8Unorm format for compatibility.

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, ComputePipeline,
    ComputePipelineDescriptor, Device, Instance, Queue, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, StorageTextureAccess, Texture, TextureDescriptor, TextureFormat, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};

#[cfg(feature = "gpu")]
use bytemuck;

use crate::Pixel;

// ============================================================================
// Configuration Constants
// ============================================================================

/// Sigma threshold for using true Gaussian convolution
const GAUSSIAN_THRESHOLD: f32 = 2.0;

/// Sigma threshold for using box blur approximation vs downsampling
const DOWNSAMPLE_THRESHOLD: f32 = 32.0;

/// Sigma threshold for using 8x vs 4x downsampling
const LARGE_SIGMA_THRESHOLD: f32 = 100.0;

/// Workgroup size in X dimension (assumed by all shaders)
const WORKGROUP_SIZE_X: u32 = 16;

/// Workgroup size in Y dimension (assumed by all shaders)
const WORKGROUP_SIZE_Y: u32 = 16;

/// Texture format used throughout the pipeline
const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// Row alignment for texture data transfers (bytes)
const ROW_ALIGNMENT: u32 = 256;

/// Bytes per pixel (RGBA)
const BYTES_PER_PIXEL: u32 = 4;

// ============================================================================
// Error Types
// ============================================================================

/// Error type for GPU blur operations
#[derive(Debug)]
pub enum BlurError {
    #[cfg(feature = "gpu")]
    GpuError(String),
    InvalidDimensions {
        width: usize,
        height: usize,
    },
    InvalidSigma(f32),
    GpuFeatureDisabled,
    BufferError(String),
}

impl std::fmt::Display for BlurError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "gpu")]
            BlurError::GpuError(msg) => write!(f, "GPU operation failed: {}", msg),
            BlurError::InvalidDimensions { width, height } => {
                write!(
                    f,
                    "Image dimensions invalid: width={}, height={}",
                    width, height
                )
            }
            BlurError::InvalidSigma(sigma) => write!(f, "Sigma must be positive, got {}", sigma),
            BlurError::GpuFeatureDisabled => {
                write!(f, "GPU feature not enabled. Build with --features gpu")
            }
            BlurError::BufferError(msg) => write!(f, "Buffer operation failed: {}", msg),
        }
    }
}

impl std::error::Error for BlurError {}

// ============================================================================
// Shader Parameter Structs
// ============================================================================

/// Parameters for downsample shader
#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _padding: [u32; 8],
}

/// Parameters for upsample shader
#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct UpsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _padding: [u32; 8],
}

/// Parameters for box blur shader
#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct BoxBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    _padding: [u32; 7],
}

/// Parameters for Gaussian blur shader
#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct GaussianBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    sigma: f32,
    _padding: [u32; 6],
}

/// Gaussian kernel weights packed into vec4<f32> arrays (256 vec4s = 1024 weights)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GaussianWeights {
    weights: [[f32; 4]; 256],
}

// Implement Pod/Zeroable for all parameter structs
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for DownsampleParams {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for DownsampleParams {}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for UpsampleParams {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for UpsampleParams {}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for BoxBlurParams {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for BoxBlurParams {}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for GaussianBlurParams {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for GaussianBlurParams {}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for GaussianWeights {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for GaussianWeights {}

// ============================================================================
// Downsample Configuration
// ============================================================================

/// Configuration for downsampling strategy
#[derive(Debug, Clone, Copy)]
struct DownsampleConfig {
    factor: u32,
    adjusted_sigma: f32,
}

/// Strategy for blur algorithm
#[derive(Debug)]
enum BlurStrategy {
    /// True Gaussian convolution for small sigmas (≤ 2.0)
    Gaussian,
    /// 3-pass box blur approximation for medium sigmas (2.0-32.0)
    Box3Pass,
    /// Downsample -> blur -> upsample for large sigmas (> 32.0)
    Downsample(DownsampleConfig),
}

// ============================================================================
// GPU Texture Wrapper
// ============================================================================

/// Represents a GPU texture with associated view
#[cfg(feature = "gpu")]
struct GpuTexture {
    texture: Texture,
    view: TextureView,
}

#[cfg(feature = "gpu")]
impl GpuTexture {
    /// Create a new GPU texture with specified usage flags
    fn new(
        device: &Device,
        width: u32,
        height: u32,
        label: &str,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage,
            view_formats: &[TEXTURE_FORMAT],
        });

        let view = texture.create_view(&TextureViewDescriptor::default());

        Self { texture, view }
    }

    /// Create a texture suitable for reading (sampling)
    fn new_readable(device: &Device, width: u32, height: u32, label: &str) -> Self {
        Self::new(
            device,
            width,
            height,
            label,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
    }

    /// Create a texture suitable for writing (storage)
    fn new_writable(device: &Device, width: u32, height: u32, label: &str) -> Self {
        Self::new(
            device,
            width,
            height,
            label,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        )
    }

    /// Create an intermediate texture for blur passes
    fn new_intermediate(device: &Device, width: u32, height: u32, label: &str) -> Self {
        Self::new(
            device,
            width,
            height,
            label,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        )
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Helper for calculating dispatch dimensions
fn calculate_dispatch(width: u32, height: u32) -> (u32, u32) {
    let dispatch_x = (width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
    let dispatch_y = (height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
    (dispatch_x, dispatch_y)
}

/// Format a label with a suffix
fn format_label(base: &str, suffix: &str) -> String {
    format!("{} {}", base, suffix)
}

/// Format a label for a pass
fn format_pass_label(label: &str, index: usize) -> String {
    format!("{} Pass {}", label, index)
}

/// Create a uniform buffer with the given label and data
#[cfg(feature = "gpu")]
fn create_uniform_buffer<T: bytemuck::Pod>(device: &Device, label: &str, data: &T) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// Calculate optimal box sizes for 3-pass Gaussian approximation
/// Based on the Central Limit Theorem approximation
fn boxes_for_gauss_3pass(sigma: f32) -> [usize; 3] {
    assert!(sigma > 0.0, "Sigma must be positive");

    let n = 3;
    let w_ideal = ((12.0 * sigma * sigma / n as f32) + 1.0).sqrt();

    let mut wl = w_ideal.floor() as usize;
    if wl % 2 == 0 {
        wl -= 1;
    }
    if wl < 1 {
        wl = 1;
    }

    let wu = wl + 2;

    let numerator =
        12.0 * sigma * sigma - (n * wl * wl) as f32 - (4 * n * wl) as f32 - (3 * n) as f32;

    let denominator = -4.0 * wl as f32 - 4.0;
    let m_ideal = numerator / denominator;

    let m = m_ideal.round().max(0.0).min(n as f32) as usize;

    let mut sizes = [wl; 3];
    for i in m..3 {
        sizes[i] = wu;
    }

    sizes
}

/// Calculate box radii from box sizes
fn calculate_box_radii(sigma: f32) -> [u32; 3] {
    boxes_for_gauss_3pass(sigma).map(|size| ((size as i32 - 1) / 2).max(0) as u32)
}

/// Calculate downsampled dimensions
fn calculate_downsampled_dimensions(width: usize, height: usize, factor: u32) -> (u32, u32) {
    let down_width = (width as u32 + factor - 1) / factor;
    let down_height = (height as u32 + factor - 1) / factor;
    (down_width, down_height)
}

/// Convert image to flat RGBA bytes
fn image_to_rgba_bytes(image: &[Vec<Pixel>], width: usize, height: usize) -> Vec<u8> {
    let mut rgba_data = Vec::with_capacity(width * height * 4);
    for row in image {
        for pixel in row {
            rgba_data.push(pixel.r);
            rgba_data.push(pixel.g);
            rgba_data.push(pixel.b);
            rgba_data.push(pixel.a);
        }
    }
    rgba_data
}

/// Calculate aligned row size for texture transfers
fn calculate_aligned_row_size(width: usize) -> u32 {
    let bytes_per_row_unaligned = BYTES_PER_PIXEL * width as u32;
    ((bytes_per_row_unaligned + ROW_ALIGNMENT - 1) / ROW_ALIGNMENT) * ROW_ALIGNMENT
}

// ============================================================================
// Main GPU Blur Processor
// ============================================================================

/// GPU Gaussian Blur processor with optimized multi-shader approach
pub struct GpuGaussianBlur {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    queue: Queue,

    #[cfg(feature = "gpu")]
    downsample_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    upsample_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    box_blur_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    gaussian_blur_pipeline: ComputePipeline,

    #[cfg(feature = "gpu")]
    downsample_bind_group_layout: BindGroupLayout,
    #[cfg(feature = "gpu")]
    upsample_bind_group_layout: BindGroupLayout,
    #[cfg(feature = "gpu")]
    box_blur_bind_group_layout: BindGroupLayout,
    #[cfg(feature = "gpu")]
    gaussian_blur_bind_group_layout: BindGroupLayout,

    sigma: f32,
    radius: i32,
    blur_alpha: bool,
    strategy: BlurStrategy,
}

impl GpuGaussianBlur {
    // ============================================================================
    // Public API
    // ============================================================================

    /// Create a new GPU Gaussian Blur processor
    pub async fn new(sigma: f32, radius: Option<i32>, blur_alpha: bool) -> Result<Self, BlurError> {
        #[cfg(not(feature = "gpu"))]
        return Err(BlurError::GpuFeatureDisabled);

        #[cfg(feature = "gpu")]
        {
            if sigma <= 0.0 {
                return Err(BlurError::InvalidSigma(sigma));
            }

            let radius = radius.unwrap_or_else(|| (3.0 * sigma).ceil() as i32);
            let strategy = Self::select_strategy(sigma);

            let (device, queue, pipelines, layouts) = Self::initialize_gpu()
                .await
                .map_err(|e| BlurError::GpuError(format!("Failed to initialize GPU: {}", e)))?;

            Ok(Self {
                device,
                queue,
                downsample_pipeline: pipelines.0, // First tuple element
                upsample_pipeline: pipelines.1,   // Second tuple element
                box_blur_pipeline: pipelines.2,   // Third tuple element
                gaussian_blur_pipeline: pipelines.3, // Fourth tuple element
                downsample_bind_group_layout: layouts.0,
                upsample_bind_group_layout: layouts.1,
                box_blur_bind_group_layout: layouts.2,
                gaussian_blur_bind_group_layout: layouts.3,
                sigma,
                radius,
                blur_alpha,
                strategy,
            })
        }
    }

    /// Apply blur to an image and return as 2D pixel array
    pub fn blur(&self, image: &[Vec<Pixel>]) -> Result<Vec<Vec<Pixel>>, String> {
        let (bytes, width, height) = self.blur_to_bytes(image)?;

        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(height);
        let mut offset = 0;

        for _ in 0..height {
            let mut row = Vec::with_capacity(width);
            for _ in 0..width {
                row.push(Pixel::new(
                    bytes[offset],
                    bytes[offset + 1],
                    bytes[offset + 2],
                    bytes[offset + 3],
                ));
                offset += 4;
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Apply blur to an image using GPU with optimal strategy
    pub fn blur_to_bytes(&self, image: &[Vec<Pixel>]) -> Result<(Vec<u8>, usize, usize), String> {
        #[cfg(not(feature = "gpu"))]
        return Err(BlurError::GpuFeatureDisabled.to_string());

        #[cfg(feature = "gpu")]
        {
            // Validate input
            if image.is_empty() || image[0].is_empty() {
                return Ok((Vec::new(), 0, 0));
            }

            let height = image.len();
            let width = image[0].len();

            // Check GPU limits
            Self::validate_image_dimensions(&self.device, width, height)
                .map_err(|e| e.to_string())?;

            // Upload image to GPU
            let input_texture = self
                .upload_image_to_gpu(image, width, height)
                .map_err(|e| e.to_string())?;

            // Execute selected strategy
            let output_texture = match &self.strategy {
                BlurStrategy::Gaussian => self
                    .apply_gaussian_blur(&input_texture.view, width, height)
                    .map_err(|e| e.to_string())?,
                BlurStrategy::Box3Pass => self
                    .apply_box_blur_3pass(&input_texture.view, width, height)
                    .map_err(|e| e.to_string())?,
                BlurStrategy::Downsample(config) => self
                    .apply_downsample_blur_upsample(
                        &input_texture.view,
                        width,
                        height,
                        config.factor,
                        config.adjusted_sigma,
                    )
                    .map_err(|e| e.to_string())?,
            };

            // Download result from GPU
            self.download_texture_to_cpu(&output_texture.texture, width, height)
                .map_err(|e| e.to_string())
        }
    }

    // ============================================================================
    // Strategy Selection
    // ============================================================================

    /// Select the optimal blur strategy based on sigma value
    fn select_strategy(sigma: f32) -> BlurStrategy {
        if sigma <= GAUSSIAN_THRESHOLD {
            BlurStrategy::Gaussian
        } else if sigma <= DOWNSAMPLE_THRESHOLD {
            BlurStrategy::Box3Pass
        } else {
            let factor = if sigma > LARGE_SIGMA_THRESHOLD {
                8
            } else if sigma > 64.0 {
                4
            } else {
                2
            };

            let adjusted_sigma = sigma / factor as f32;

            BlurStrategy::Downsample(DownsampleConfig {
                factor,
                adjusted_sigma,
            })
        }
    }

    // ============================================================================
    // GPU Initialization
    // ============================================================================

    #[cfg(feature = "gpu")]
    async fn initialize_gpu() -> Result<
        (
            Device,
            Queue,
            (
                ComputePipeline, // downsample
                ComputePipeline, // upsample
                ComputePipeline, // box_blur
                ComputePipeline, // gaussian
            ),
            (
                BindGroupLayout, // downsample
                BindGroupLayout, // upsample
                BindGroupLayout, // box_blur
                BindGroupLayout, // gaussian
            ),
        ),
        String,
    > {
        let instance = Instance::default();

        // request_adapter returns Result<Adapter, RequestAdapterError>
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| format!("Failed to find a suitable GPU adapter: {}", e))?;

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Gaussian Blur Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        // Load and compile all shaders
        let downsample_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Downsample Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/downsample.wgsl").into()),
        });

        let upsample_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Upsample Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/upsample.wgsl").into()),
        });

        let box_blur_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Box Blur Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/box_blur_separable.wgsl").into()),
        });

        let gaussian_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Gaussian Blur Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/gaussian_blur_separable.wgsl").into()),
        });

        // Create bind group layouts
        let common_entries = &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TEXTURE_FORMAT,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let downsample_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Downsample Bind Group Layout"),
            entries: common_entries,
        });

        let upsample_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Upsample Bind Group Layout"),
            entries: common_entries,
        });

        let box_blur_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Box Blur Bind Group Layout"),
            entries: common_entries,
        });

        let gaussian_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Gaussian Blur Bind Group Layout"),
            entries: &[
                common_entries[0].clone(),
                common_entries[1].clone(),
                common_entries[2].clone(),
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create compute pipelines
        let create_pipeline =
            |device: &Device, layout: &BindGroupLayout, module: &ShaderModule, label: &str| {
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format_label(label, "Pipeline Layout")),
                        bind_group_layouts: &[layout],
                        immediate_size: 0,
                    });

                device.create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(&format_label(label, "Pipeline")),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
            };

        let pipelines = (
            create_pipeline(
                &device,
                &downsample_layout,
                &downsample_shader,
                "Downsample",
            ),
            create_pipeline(&device, &upsample_layout, &upsample_shader, "Upsample"),
            create_pipeline(&device, &box_blur_layout, &box_blur_shader, "Box Blur"),
            create_pipeline(&device, &gaussian_layout, &gaussian_shader, "Gaussian"),
        );

        let layouts = (
            downsample_layout,
            upsample_layout,
            box_blur_layout,
            gaussian_layout,
        );

        Ok((device, queue, pipelines, layouts))
    }

    // ============================================================================
    // Data Transfer
    // ============================================================================

    #[cfg(feature = "gpu")]
    fn validate_image_dimensions(
        device: &Device,
        width: usize,
        height: usize,
    ) -> Result<(), BlurError> {
        let device_limits = device.limits();

        if width as u32 > device_limits.max_texture_dimension_2d {
            return Err(BlurError::InvalidDimensions { width, height });
        }

        if height as u32 > device_limits.max_texture_dimension_2d {
            return Err(BlurError::InvalidDimensions { width, height });
        }

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn upload_image_to_gpu(
        &self,
        image: &[Vec<Pixel>],
        width: usize,
        height: usize,
    ) -> Result<GpuTexture, BlurError> {
        // Convert image to flat RGBA bytes
        let rgba_data = image_to_rgba_bytes(image, width, height);

        // Create input texture
        let input_texture =
            GpuTexture::new_readable(&self.device, width as u32, height as u32, "Input Texture");

        // Write image data to texture
        let bytes_per_row_aligned = calculate_aligned_row_size(width);

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &input_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row_aligned),
                rows_per_image: Some(height as u32),
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        Ok(input_texture)
    }

    #[cfg(feature = "gpu")]
    fn download_texture_to_cpu(
        &self,
        texture: &Texture,
        width: usize,
        height: usize,
    ) -> Result<(Vec<u8>, usize, usize), BlurError> {
        let bytes_per_row_aligned = calculate_aligned_row_size(width);

        // Create staging buffer with exact size needed
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (width * height * BYTES_PER_PIXEL as usize) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });

        // Copy from texture to staging buffer with proper row alignment
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row_aligned),
                    rows_per_image: Some(height as u32),
                },
            },
            wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Read data directly from staging buffer
        Self::read_staging_buffer(&self.device, &staging_buffer, width, height)
            .map(|bytes| (bytes, width, height))
    }

    // ============================================================================
    // Blur Algorithms
    // ============================================================================

    #[cfg(feature = "gpu")]
    fn apply_gaussian_blur(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
    ) -> Result<GpuTexture, BlurError> {
        // Precompute Gaussian kernel weights
        let weights_data = self.precompute_gaussian_weights(self.radius as u32, self.sigma);
        let weights_buffer =
            create_uniform_buffer(&self.device, "Gaussian Weights Buffer", &weights_data);

        // Create intermediate texture for vertical pass
        let intermediate = GpuTexture::new_intermediate(
            &self.device,
            width as u32,
            height as u32,
            "Intermediate Texture",
        );

        // Create output texture
        let output = GpuTexture::new_writable(
            &self.device,
            width as u32,
            height as u32,
            "Gaussian Blur Output",
        );

        // Horizontal pass
        let horiz_params = GaussianBlurParams {
            width: width as u32,
            height: height as u32,
            radius: self.radius as u32,
            blur_alpha: self.blur_alpha as u32,
            direction: 0,
            sigma: self.sigma,
            _padding: [0; 6],
        };

        self.execute_gaussian_pass(
            "Horizontal Gaussian",
            width,
            height,
            input_view,
            &intermediate.view,
            &horiz_params,
            &weights_buffer,
        )?;

        // Vertical pass
        let vert_params = GaussianBlurParams {
            width: width as u32,
            height: height as u32,
            radius: self.radius as u32,
            blur_alpha: self.blur_alpha as u32,
            direction: 1,
            sigma: self.sigma,
            _padding: [0; 6],
        };

        self.execute_gaussian_pass(
            "Vertical Gaussian",
            width,
            height,
            &intermediate.view,
            &output.view,
            &vert_params,
            &weights_buffer,
        )?;

        Ok(output)
    }

    #[cfg(feature = "gpu")]
    fn apply_box_blur_3pass(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
    ) -> Result<GpuTexture, BlurError> {
        // Calculate box sizes for 3-pass approximation
        let box_radii = calculate_box_radii(self.sigma);

        // Create intermediate textures
        let texture1 = GpuTexture::new_writable(
            &self.device,
            width as u32,
            height as u32,
            "Intermediate Texture 1",
        );
        let texture2 = GpuTexture::new_intermediate(
            &self.device,
            width as u32,
            height as u32,
            "Intermediate Texture 2",
        );

        // Create output texture
        let output =
            GpuTexture::new_writable(&self.device, width as u32, height as u32, "Box Blur Output");

        // Apply 6 blur passes (3 passes × 2 directions)
        let blur_passes = [
            (input_view, &texture1.view, box_radii[0], 0u32),
            (&texture1.view, &texture2.view, box_radii[0], 1u32),
            (&texture2.view, &texture1.view, box_radii[1], 0u32),
            (&texture1.view, &texture2.view, box_radii[1], 1u32),
            (&texture2.view, &texture1.view, box_radii[2], 0u32),
            (&texture1.view, &output.view, box_radii[2], 1u32),
        ];

        for (i, (input_view, output_view, radius, direction)) in blur_passes.iter().enumerate() {
            let params = BoxBlurParams {
                width: width as u32,
                height: height as u32,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                direction: *direction,
                _padding: [0; 7],
            };

            self.execute_box_blur_pass(
                &format_pass_label("Box Blur", i + 1),
                width,
                height,
                input_view,
                output_view,
                &params,
            )?;
        }

        Ok(output)
    }

    #[cfg(feature = "gpu")]
    fn apply_downsample_blur_upsample(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
        factor: u32,
        adjusted_sigma: f32,
    ) -> Result<GpuTexture, BlurError> {
        // Calculate downsampled dimensions
        let (down_width, down_height) = calculate_downsampled_dimensions(width, height, factor);

        // Step 1: Downsample
        let downsampled =
            self.downsample_image(input_view, width, height, down_width, down_height)?;

        // Step 2: Apply blur on downsampled image
        let blurred_down = self.blur_downsampled_image(
            &downsampled.view,
            down_width,
            down_height,
            adjusted_sigma,
        )?;

        // Step 3: Upsample
        let final_output =
            self.upsample_image(&blurred_down.view, width, height, down_width, down_height)?;

        Ok(final_output)
    }

    // ============================================================================
    // Pipeline Operations
    // ============================================================================

    #[cfg(feature = "gpu")]
    fn execute_gaussian_pass(
        &self,
        label: &str,
        width: usize,
        height: usize,
        input_view: &TextureView,
        output_view: &TextureView,
        params: &GaussianBlurParams,
        weights_buffer: &Buffer,
    ) -> Result<(), BlurError> {
        let params_buffer =
            create_uniform_buffer(&self.device, &format_label(label, "Params"), params);

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format_label(label, "Bind Group")),
            layout: &self.gaussian_blur_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: weights_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format_label(label, "Encoder")),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format_label(label, "Compute Pass")),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.gaussian_blur_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let (dispatch_x, dispatch_y) = calculate_dispatch(width as u32, height as u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(())
    }

    #[cfg(feature = "gpu")]
    fn execute_box_blur_pass(
        &self,
        label: &str,
        width: usize,
        height: usize,
        input_view: &TextureView,
        output_view: &TextureView,
        params: &BoxBlurParams,
    ) -> Result<(), BlurError> {
        let params_buffer =
            create_uniform_buffer(&self.device, &format_label(label, "Params"), params);

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format_label(label, "Bind Group")),
            layout: &self.box_blur_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format_label(label, "Encoder")),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format_label(label, "Compute Pass")),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.box_blur_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let (dispatch_x, dispatch_y) = calculate_dispatch(width as u32, height as u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(())
    }

    // ============================================================================
    // Algorithm Components
    // ============================================================================

    #[cfg(feature = "gpu")]
    fn downsample_image(
        &self,
        input_view: &TextureView,
        src_width: usize,
        src_height: usize,
        dst_width: u32,
        dst_height: u32,
    ) -> Result<GpuTexture, BlurError> {
        let downsampled =
            GpuTexture::new_writable(&self.device, dst_width, dst_height, "Downsampled Texture");

        let params = DownsampleParams {
            src_width: src_width as u32,
            src_height: src_height as u32,
            dst_width,
            dst_height,
            _padding: [0; 8],
        };

        self.execute_simple_pass(
            &self.downsample_pipeline,
            &self.downsample_bind_group_layout,
            "Downsample",
            dst_width as usize,
            dst_height as usize,
            input_view,
            &downsampled.view,
            &params,
        )?;

        Ok(downsampled)
    }

    #[cfg(feature = "gpu")]
    fn blur_downsampled_image(
        &self,
        input_view: &TextureView,
        width: u32,
        height: u32,
        adjusted_sigma: f32,
    ) -> Result<GpuTexture, BlurError> {
        // Create intermediate textures
        let texture1 = GpuTexture::new_writable(
            &self.device,
            width,
            height,
            "Intermediate Texture 1 (Downsampled)",
        );
        let texture2 = GpuTexture::new_intermediate(
            &self.device,
            width,
            height,
            "Intermediate Texture 2 (Downsampled)",
        );

        // Create output texture
        let output =
            GpuTexture::new_writable(&self.device, width, height, "Blurred Downsampled Texture");

        // Calculate box sizes for adjusted sigma
        let box_radii = calculate_box_radii(adjusted_sigma);

        // Apply 6 blur passes (3 passes × 2 directions)
        let blur_passes = [
            (input_view, &texture1.view, box_radii[0], 0u32),
            (&texture1.view, &texture2.view, box_radii[0], 1u32),
            (&texture2.view, &texture1.view, box_radii[1], 0u32),
            (&texture1.view, &texture2.view, box_radii[1], 1u32),
            (&texture2.view, &texture1.view, box_radii[2], 0u32),
            (&texture1.view, &output.view, box_radii[2], 1u32),
        ];

        for (i, (input_view, output_view, radius, direction)) in blur_passes.iter().enumerate() {
            let params = BoxBlurParams {
                width,
                height,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                direction: *direction,
                _padding: [0; 7],
            };

            self.execute_box_blur_pass(
                &format_pass_label("Downsampled Box Blur", i + 1),
                width as usize,
                height as usize,
                input_view,
                output_view,
                &params,
            )?;
        }

        Ok(output)
    }

    #[cfg(feature = "gpu")]
    fn upsample_image(
        &self,
        input_view: &TextureView,
        dst_width: usize,
        dst_height: usize,
        src_width: u32,
        src_height: u32,
    ) -> Result<GpuTexture, BlurError> {
        let final_output = GpuTexture::new_writable(
            &self.device,
            dst_width as u32,
            dst_height as u32,
            "Final Output Texture",
        );

        let params = UpsampleParams {
            src_width,
            src_height,
            dst_width: dst_width as u32,
            dst_height: dst_height as u32,
            _padding: [0; 8],
        };

        self.execute_simple_pass(
            &self.upsample_pipeline,
            &self.upsample_bind_group_layout,
            "Upsample",
            dst_width,
            dst_height,
            input_view,
            &final_output.view,
            &params,
        )?;

        Ok(final_output)
    }

    #[cfg(feature = "gpu")]
    fn execute_simple_pass<T: bytemuck::Pod>(
        &self,
        pipeline: &ComputePipeline,
        layout: &BindGroupLayout,
        label: &str,
        width: usize,
        height: usize,
        input_view: &TextureView,
        output_view: &TextureView,
        params: &T,
    ) -> Result<(), BlurError> {
        let params_buffer =
            create_uniform_buffer(&self.device, &format_label(label, "Params"), params);

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some(&format_label(label, "Bind Group")),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format_label(label, "Encoder")),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format_label(label, "Compute Pass")),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let (dispatch_x, dispatch_y) = calculate_dispatch(width as u32, height as u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(())
    }

    // ============================================================================
    // Utility Functions
    // ============================================================================

    /// Precompute Gaussian kernel weights packed into vec4<f32> arrays
    #[cfg(feature = "gpu")]
    fn precompute_gaussian_weights(&self, radius: u32, sigma: f32) -> GaussianWeights {
        let kernel_size = (2 * radius + 1) as usize;
        let mut raw_weights = Vec::with_capacity(kernel_size);
        let mut weight_sum = 0.0;

        for i in 0..kernel_size {
            let x = i as i32 - radius as i32;
            let weight = (-(x * x) as f32 / (2.0 * sigma * sigma)).exp();
            raw_weights.push(weight);
            weight_sum += weight;
        }

        for weight in raw_weights.iter_mut() {
            *weight /= weight_sum;
        }

        let mut weights_data = GaussianWeights {
            weights: [[0.0; 4]; 256],
        };

        let vec4_count = (kernel_size + 3) / 4;

        for i in 0..vec4_count {
            let base_idx = i * 4;

            for j in 0..4 {
                let idx = base_idx + j;
                if idx < kernel_size {
                    weights_data.weights[i][j] = raw_weights[idx];
                }
            }
        }

        weights_data
    }

    /// Read staging buffer contents with simplified pattern
    #[cfg(feature = "gpu")]
    fn read_staging_buffer(
        device: &Device,
        buffer: &Buffer,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, BlurError> {
        let buffer_slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Wait for mapping to complete
        receiver
            .recv()
            .map_err(|e| {
                BlurError::BufferError(format!("Failed to receive buffer mapping result: {}", e))
            })?
            .map_err(|e| BlurError::BufferError(format!("Failed to map buffer: {}", e)))?;

        // Get the mapped data
        let data = buffer_slice.get_mapped_range();

        // Copy data directly - staging buffer already has exact size
        let result_bytes = data.to_vec();

        // Verify size matches expectations
        let expected_bytes = width * height * BYTES_PER_PIXEL as usize;
        if result_bytes.len() != expected_bytes {
            return Err(BlurError::BufferError(format!(
                "Buffer size mismatch: expected {} bytes, got {}",
                expected_bytes,
                result_bytes.len()
            )));
        }

        Ok(result_bytes)
    }
}
