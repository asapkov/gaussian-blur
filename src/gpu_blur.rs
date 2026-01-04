//! GPU-accelerated Gaussian Blur using wgpu with optimized multi-shader approach
//!
//! This module implements a multi-strategy Gaussian blur using WebGPU:
//! - Small sigmas (< 2.0): True Gaussian convolution
//! - Medium sigmas (2.0-32.0): 3-pass box blur approximation  
//! - Large sigmas (> 32.0): Downsample -> blur -> upsample pipeline
//!
//! All shaders assume a workgroup size of 16x16 threads.
//! All textures use Rgba16Float format for higher precision.

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
const GAUSSIAN_THRESHOLD: f32 = 1.0; // CHANGED FROM 2.0 TO 1.0

/// Sigma threshold for using box blur approximation vs downsampling
const DOWNSAMPLE_THRESHOLD: f32 = 32.0; // CHANGED FROM 5.0 BACK TO 32.0

/// Sigma threshold for using 8x vs 4x downsampling
const LARGE_SIGMA_THRESHOLD: f32 = 100.0;

/// Workgroup size in X dimension (assumed by all shaders)
const WORKGROUP_SIZE_X: u32 = 16;

/// Workgroup size in Y dimension (assumed by all shaders)
const WORKGROUP_SIZE_Y: u32 = 16;

/// Texture format used throughout the pipeline - CHANGED TO Rgba16Float for higher precision
const TEXTURE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

/// Row alignment for texture data transfers (bytes)
const ROW_ALIGNMENT: u32 = 256;

/// Bytes per pixel (RGBA) - CHANGED TO 8 bytes for Rgba16Float
const BYTES_PER_PIXEL: u32 = 8;

/// Threshold for using small radius optimized shader
const SMALL_RADIUS_THRESHOLD: u32 = 8;

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
    Timeout(String),
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
            BlurError::Timeout(msg) => write!(f, "Operation timed out: {}", msg),
        }
    }
}

impl std::error::Error for BlurError {}

// ============================================================================
// Shader Parameter Structs
// ============================================================================

/// Parameters for downsample shader
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

/// Parameters for upsample shader
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct UpsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
}

/// Parameters for box blur shader - MUST MATCH SHADER STRUCT SIZE
#[cfg(feature = "gpu")]
#[repr(C, align(16))] // Align to 16 bytes for WGSL compatibility
#[derive(Debug, Clone, Copy)]
struct BoxBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

/// Parameters for Gaussian blur shader
#[cfg(feature = "gpu")]
#[repr(C, align(16))] // Align to 16 bytes for WGSL compatibility
#[derive(Debug, Clone, Copy)]
struct GaussianBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    sigma: f32,
    _padding0: u32,
}

/// Gaussian kernel weights packed into vec4<f32> arrays (256 vec4s = 1024 weights)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GaussianWeights {
    weights: [[f32; 4]; 256],
}

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

/// Convert image to flat RGBA bytes (no padding)
fn image_to_rgba_bytes(image: &[Vec<Pixel>], width: usize, height: usize) -> Vec<u8> {
    let capacity = width * height * 4;
    let mut rgba_data = Vec::with_capacity(capacity);

    for row in image {
        let row_start = rgba_data.len();
        rgba_data.resize(row_start + width * 4, 0);

        let row_slice = &mut rgba_data[row_start..];
        for (x, pixel) in row.iter().enumerate() {
            let offset = x * 4;
            row_slice[offset] = pixel.r;
            row_slice[offset + 1] = pixel.g;
            row_slice[offset + 2] = pixel.b;
            row_slice[offset + 3] = pixel.a;
        }
    }

    rgba_data
}

/// Calculate aligned row size for texture transfers
fn calculate_aligned_row_size(width: usize) -> u32 {
    let bytes_per_row_unaligned = BYTES_PER_PIXEL * width as u32;
    ((bytes_per_row_unaligned + ROW_ALIGNMENT - 1) / ROW_ALIGNMENT) * ROW_ALIGNMENT
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
    box_blur_small_pipeline: ComputePipeline,
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

    #[cfg(feature = "gpu")]
    sampler: wgpu::Sampler,

    sigma: f32,
    radius: i32,
    blur_alpha: bool,
    strategy: BlurStrategy,
    use_small_radius_shader: bool,
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
            let use_small_radius_shader = radius as u32 <= SMALL_RADIUS_THRESHOLD;

            let (device, queue, pipelines, layouts, sampler) = Self::initialize_gpu()
                .await
                .map_err(|e| BlurError::GpuError(format!("Failed to initialize GPU: {}", e)))?;

            Ok(Self {
                device,
                queue,
                downsample_pipeline: pipelines.0,
                upsample_pipeline: pipelines.1,
                box_blur_pipeline: pipelines.2,
                box_blur_small_pipeline: pipelines.3,
                gaussian_blur_pipeline: pipelines.4,
                downsample_bind_group_layout: layouts.0,
                upsample_bind_group_layout: layouts.1,
                box_blur_bind_group_layout: layouts.2,
                gaussian_blur_bind_group_layout: layouts.3,
                sampler,
                sigma,
                radius,
                blur_alpha,
                strategy,
                use_small_radius_shader,
            })
        }
    }

    /// Apply blur to an image and return as 2D pixel array
    pub fn blur(&mut self, image: &[Vec<Pixel>]) -> Result<Vec<Vec<Pixel>>, BlurError> {
        let (bytes, width, height) = self.blur_to_bytes(image)?;

        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(height);

        for y in 0..height {
            let mut row = Vec::with_capacity(width);
            let row_offset = y * width * 4;

            for x in 0..width {
                let pixel_offset = row_offset + x * 4;
                row.push(Pixel::new(
                    bytes[pixel_offset],
                    bytes[pixel_offset + 1],
                    bytes[pixel_offset + 2],
                    bytes[pixel_offset + 3],
                ));
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Apply blur to an image using GPU with optimal strategy
    pub fn blur_to_bytes(
        &mut self,
        image: &[Vec<Pixel>],
    ) -> Result<(Vec<u8>, usize, usize), BlurError> {
        #[cfg(not(feature = "gpu"))]
        return Err(BlurError::GpuFeatureDisabled);

        #[cfg(feature = "gpu")]
        {
            if image.is_empty() || image[0].is_empty() {
                return Ok((Vec::new(), 0, 0));
            }

            let height = image.len();
            let width = image[0].len();

            Self::validate_image_dimensions(&self.device, width, height)?;

            let (_input_texture, input_view) = self.upload_image_to_gpu(image, width, height)?;

            let output_texture = match &self.strategy {
                BlurStrategy::Gaussian => self.apply_gaussian_blur(&input_view, width, height)?,
                BlurStrategy::Box3Pass => self.apply_box_blur_3pass(&input_view, width, height)?,
                BlurStrategy::Downsample(config) => self.apply_downsample_blur_upsample(
                    &input_view,
                    width,
                    height,
                    config.factor,
                    config.adjusted_sigma,
                )?,
            };

            let result = self.download_texture_to_cpu(&output_texture, width, height)?;
            Ok(result)
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
            // Start downsampling earlier for better performance
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
                ComputePipeline, // box_blur (optimized with shared memory)
                ComputePipeline, // box_blur_small (unrolled for small radii)
                ComputePipeline, // gaussian (optimized with shared memory)
            ),
            (
                BindGroupLayout, // downsample
                BindGroupLayout, // upsample
                BindGroupLayout, // box_blur
                BindGroupLayout, // gaussian
            ),
            wgpu::Sampler,
        ),
        String,
    > {
        let instance = Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| format!("Failed to find a suitable GPU adapter: {}", e))?;

        let (device, queue) = adapter
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

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Upsample Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

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
            source: ShaderSource::Wgsl(
                include_str!("shaders/box_blur_separable_optimized.wgsl").into(),
            ),
        });

        let box_blur_small_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Box Blur Small Radius Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/box_blur_small_radius.wgsl").into()),
        });

        let gaussian_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Gaussian Blur Shader"),
            source: ShaderSource::Wgsl(include_str!("shaders/gaussian_blur_optimized.wgsl").into()),
        });

        let downsample_entries = &[
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

        let upsample_entries = &[
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
            BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ];

        let downsample_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Downsample Bind Group Layout"),
            entries: downsample_entries,
        });

        let upsample_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Upsample Bind Group Layout"),
            entries: upsample_entries,
        });

        let box_blur_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Box Blur Bind Group Layout"),
            entries: &[
                downsample_entries[0].clone(),
                downsample_entries[1].clone(),
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
            ],
        });

        let gaussian_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Gaussian Blur Bind Group Layout"),
            entries: &[
                downsample_entries[0].clone(),
                downsample_entries[1].clone(),
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
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
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
            create_pipeline(
                &device,
                &box_blur_layout,
                &box_blur_small_shader,
                "Box Blur Small",
            ),
            create_pipeline(&device, &gaussian_layout, &gaussian_shader, "Gaussian"),
        );

        let layouts = (
            downsample_layout,
            upsample_layout,
            box_blur_layout,
            gaussian_layout,
        );

        Ok((device, queue, pipelines, layouts, sampler))
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
    fn create_texture(
        device: &Device,
        width: u32,
        height: u32,
        label: &str,
        usage: wgpu::TextureUsages,
    ) -> Texture {
        device.create_texture(&TextureDescriptor {
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
        })
    }

    #[cfg(feature = "gpu")]
    fn upload_image_to_gpu(
        &self,
        image: &[Vec<Pixel>],
        width: usize,
        height: usize,
    ) -> Result<(Texture, TextureView), BlurError> {
        let rgba_bytes = image_to_rgba_bytes(image, width, height);
        let bytes_per_row_aligned = calculate_aligned_row_size(width);

        // Convert 8-bit RGBA to 16-bit float for upload
        let aligned_row_size = bytes_per_row_aligned as usize;
        let unaligned_row_size = width * BYTES_PER_PIXEL as usize;

        // Prepare buffer for 16-bit float texture
        let mut float_data = Vec::with_capacity(height * unaligned_row_size);

        for y in 0..height {
            let row_start = y * width * 4;
            for x in 0..width {
                let pixel_offset = row_start + x * 4;

                // Convert each 8-bit channel to f32, then to half precision (f16)
                let r = rgba_bytes[pixel_offset] as f32 / 255.0;
                let g = rgba_bytes[pixel_offset + 1] as f32 / 255.0;
                let b = rgba_bytes[pixel_offset + 2] as f32 / 255.0;
                let a = rgba_bytes[pixel_offset + 3] as f32 / 255.0;

                // Convert f32 to f16 (using simple conversion - in practice use a proper f16 library)
                let r_f16 = half::f16::from_f32(r).to_le_bytes();
                let g_f16 = half::f16::from_f32(g).to_le_bytes();
                let b_f16 = half::f16::from_f32(b).to_le_bytes();
                let a_f16 = half::f16::from_f32(a).to_le_bytes();

                float_data.extend_from_slice(&r_f16);
                float_data.extend_from_slice(&g_f16);
                float_data.extend_from_slice(&b_f16);
                float_data.extend_from_slice(&a_f16);
            }
        }

        if aligned_row_size == unaligned_row_size {
            self.upload_bytes_to_gpu(&float_data, width, height, bytes_per_row_aligned)
        } else {
            let mut padded_data = Vec::with_capacity(height * aligned_row_size);

            for y in 0..height {
                let src_start = y * unaligned_row_size;
                let src_end = src_start + unaligned_row_size;

                padded_data.extend_from_slice(&float_data[src_start..src_end]);
                padded_data.resize(
                    padded_data.len() + (aligned_row_size - unaligned_row_size),
                    0,
                );
            }

            self.upload_bytes_to_gpu(&padded_data, width, height, bytes_per_row_aligned)
        }
    }

    #[cfg(feature = "gpu")]
    fn upload_bytes_to_gpu(
        &self,
        data: &[u8],
        width: usize,
        height: usize,
        bytes_per_row_aligned: u32,
    ) -> Result<(Texture, TextureView), BlurError> {
        let texture = Self::create_texture(
            &self.device,
            width as u32,
            height as u32,
            "Input Texture",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
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

        Ok((texture, view))
    }

    #[cfg(feature = "gpu")]
    fn download_texture_to_cpu(
        &self,
        texture: &Texture,
        width: usize,
        height: usize,
    ) -> Result<(Vec<u8>, usize, usize), BlurError> {
        let bytes_per_row_aligned = calculate_aligned_row_size(width);
        let buffer_size = (bytes_per_row_aligned as u64) * (height as u64);

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });

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

        let buffer_slice = staging_buffer.slice(..);

        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            if let Ok(result) = receiver.try_recv() {
                match result {
                    Ok(()) => {
                        break;
                    }
                    Err(e) => {
                        return Err(BlurError::BufferError(format!(
                            "Failed to map buffer: {}",
                            e
                        )));
                    }
                }
            }

            if start_time.elapsed() > timeout {
                return Err(BlurError::Timeout(
                    "Timeout waiting for buffer mapping".to_string(),
                ));
            }

            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let data = buffer_slice.get_mapped_range();
        let unaligned_row_size = width * BYTES_PER_PIXEL as usize;
        let aligned_row_size = bytes_per_row_aligned as usize;

        let mut float_data = Vec::with_capacity(width * height * 4);
        let mut result = Vec::with_capacity(width * height * 4);

        for y in 0..height {
            let src_start = y * aligned_row_size;
            float_data.extend_from_slice(&data[src_start..src_start + unaligned_row_size]);
        }

        drop(data);
        staging_buffer.unmap();

        // Convert 16-bit float back to 8-bit RGBA
        for i in 0..(width * height) {
            let offset = i * 8; // 8 bytes per pixel (4 channels * 2 bytes each)

            if offset + 7 < float_data.len() {
                let r_f16 = half::f16::from_le_bytes([float_data[offset], float_data[offset + 1]]);
                let g_f16 =
                    half::f16::from_le_bytes([float_data[offset + 2], float_data[offset + 3]]);
                let b_f16 =
                    half::f16::from_le_bytes([float_data[offset + 4], float_data[offset + 5]]);
                let a_f16 =
                    half::f16::from_le_bytes([float_data[offset + 6], float_data[offset + 7]]);

                result.push((r_f16.to_f32() * 255.0).clamp(0.0, 255.0) as u8);
                result.push((g_f16.to_f32() * 255.0).clamp(0.0, 255.0) as u8);
                result.push((b_f16.to_f32() * 255.0).clamp(0.0, 255.0) as u8);
                result.push((a_f16.to_f32() * 255.0).clamp(0.0, 255.0) as u8);
            }
        }

        Ok((result, width, height))
    }

    // ============================================================================
    // Blur Algorithms
    // ============================================================================

    #[cfg(feature = "gpu")]
    fn create_output_texture(
        &self,
        width: u32,
        height: u32,
        label: &str,
    ) -> (Texture, TextureView) {
        let texture = Self::create_texture(
            &self.device,
            width,
            height,
            label,
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );

        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }

    #[cfg(feature = "gpu")]
    fn create_intermediate_texture(
        &self,
        width: u32,
        height: u32,
        label: &str,
    ) -> (Texture, TextureView) {
        let texture = Self::create_texture(
            &self.device,
            width,
            height,
            label,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let view = texture.create_view(&TextureViewDescriptor::default());
        (texture, view)
    }

    #[cfg(feature = "gpu")]
    fn create_uniform_buffer<T: bytemuck::Pod>(&self, data: &T, label: &str) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    #[cfg(feature = "gpu")]
    fn apply_gaussian_blur(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
    ) -> Result<Texture, BlurError> {
        let width_u32 = width as u32;
        let height_u32 = height as u32;

        let weights_data = self.precompute_gaussian_weights(self.radius as u32);
        let weights_buffer = self.create_uniform_buffer(&weights_data, "Gaussian Weights Buffer");

        let (_intermediate_texture, intermediate_view) =
            self.create_intermediate_texture(width_u32, height_u32, "Gaussian Intermediate");

        let (output_texture, output_view) =
            self.create_output_texture(width_u32, height_u32, "Gaussian Output");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gaussian Blur"),
            });

        {
            let params = GaussianBlurParams {
                width: width_u32,
                height: height_u32,
                radius: self.radius as u32,
                blur_alpha: self.blur_alpha as u32,
                direction: 0,
                sigma: self.sigma,
                _padding0: 0,
            };

            let param_buffer = self.create_uniform_buffer(&params, "Gaussian Horizontal Params");

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Gaussian Horizontal Bind Group"),
                layout: &self.gaussian_blur_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&intermediate_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: param_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: weights_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gaussian Horizontal"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.gaussian_blur_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (dispatch_x, dispatch_y) = calculate_dispatch(width_u32, height_u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        {
            let params = GaussianBlurParams {
                width: width_u32,
                height: height_u32,
                radius: self.radius as u32,
                blur_alpha: self.blur_alpha as u32,
                direction: 1,
                sigma: self.sigma,
                _padding0: 0,
            };

            let param_buffer = self.create_uniform_buffer(&params, "Gaussian Vertical Params");

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Gaussian Vertical Bind Group"),
                layout: &self.gaussian_blur_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&intermediate_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: param_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: weights_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gaussian Vertical"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.gaussian_blur_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (dispatch_x, dispatch_y) = calculate_dispatch(width_u32, height_u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(output_texture)
    }

    #[cfg(feature = "gpu")]
    fn apply_box_blur_3pass(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
    ) -> Result<Texture, BlurError> {
        let width_u32 = width as u32;
        let height_u32 = height as u32;

        let box_radii = calculate_box_radii(self.sigma);

        let (_texture1, view1) =
            self.create_intermediate_texture(width_u32, height_u32, "Box Blur Texture 1");
        let (_texture2, view2) =
            self.create_intermediate_texture(width_u32, height_u32, "Box Blur Texture 2");
        let (output_texture, output_view) =
            self.create_output_texture(width_u32, height_u32, "Box Blur Output");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Box Blur 3-Pass"),
            });

        let blur_passes = [
            (input_view, &view1, box_radii[0], 0u32, "Pass 1 Horizontal"),
            (&view1, &view2, box_radii[0], 1u32, "Pass 1 Vertical"),
            (&view2, &view1, box_radii[1], 0u32, "Pass 2 Horizontal"),
            (&view1, &view2, box_radii[1], 1u32, "Pass 2 Vertical"),
            (&view2, &view1, box_radii[2], 0u32, "Pass 3 Horizontal"),
            (&view1, &output_view, box_radii[2], 1u32, "Pass 3 Vertical"),
        ];

        for (input_view, output_view, radius, direction, label) in blur_passes.iter() {
            let params = BoxBlurParams {
                width: width_u32,
                height: height_u32,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                direction: *direction,
                _padding0: 0,
                _padding1: 0,
                _padding2: 0,
            };

            let param_buffer = self.create_uniform_buffer(&params, &format!("Box Blur {}", label));

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("Box Blur Bind Group {}", label)),
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
                        resource: param_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&label),
                timestamp_writes: None,
            });

            // Choose appropriate pipeline based on radius
            if *radius <= SMALL_RADIUS_THRESHOLD && self.use_small_radius_shader {
                compute_pass.set_pipeline(&self.box_blur_small_pipeline);
            } else {
                compute_pass.set_pipeline(&self.box_blur_pipeline);
            }

            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (dispatch_x, dispatch_y) = calculate_dispatch(width_u32, height_u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(output_texture)
    }

    #[cfg(feature = "gpu")]
    fn apply_downsample_blur_upsample(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
        factor: u32,
        adjusted_sigma: f32,
    ) -> Result<Texture, BlurError> {
        let (down_width, down_height) = calculate_downsampled_dimensions(width, height, factor);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Downsample Blur Upsample"),
            });

        let (_downsampled_texture, downsampled_view) = {
            let (texture, view) =
                self.create_intermediate_texture(down_width, down_height, "Downsampled");

            let params = DownsampleParams {
                src_width: width as u32,
                src_height: height as u32,
                dst_width: down_width,
                dst_height: down_height,
            };

            let param_buffer = create_uniform_buffer(&self.device, "Downsample Params", &params);

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Downsample Bind Group"),
                layout: &self.downsample_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: param_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Downsample"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.downsample_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (dispatch_x, dispatch_y) = calculate_dispatch(down_width, down_height);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            (texture, view)
        };

        let (_blurred_down_texture, blurred_down_view) = {
            let box_radii = calculate_box_radii(adjusted_sigma);

            let (_texture1, view1) =
                self.create_intermediate_texture(down_width, down_height, "Downsampled Blur 1");
            let (_texture2, view2) =
                self.create_intermediate_texture(down_width, down_height, "Downsampled Blur 2");
            let (output_texture, output_view) =
                self.create_output_texture(down_width, down_height, "Blurred Downsampled");

            let blur_passes = [
                (
                    &downsampled_view,
                    &view1,
                    box_radii[0],
                    0u32,
                    "Downsample Blur 1H",
                ),
                (&view1, &view2, box_radii[0], 1u32, "Downsample Blur 1V"),
                (&view2, &view1, box_radii[1], 0u32, "Downsample Blur 2H"),
                (&view1, &view2, box_radii[1], 1u32, "Downsample Blur 2V"),
                (&view2, &view1, box_radii[2], 0u32, "Downsample Blur 3H"),
                (
                    &view1,
                    &output_view,
                    box_radii[2],
                    1u32,
                    "Downsample Blur 3V",
                ),
            ];

            for (input_view, output_view, radius, direction, label) in blur_passes.iter() {
                let params = BoxBlurParams {
                    width: down_width,
                    height: down_height,
                    radius: *radius,
                    blur_alpha: self.blur_alpha as u32,
                    direction: *direction,
                    _padding0: 0,
                    _padding1: 0,
                    _padding2: 0,
                };

                let param_buffer = self.create_uniform_buffer(&params, label);

                let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("Downsample Blur {}", label)),
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
                            resource: param_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&label),
                    timestamp_writes: None,
                });

                // Choose appropriate pipeline for downsampled blur
                if *radius <= SMALL_RADIUS_THRESHOLD {
                    compute_pass.set_pipeline(&self.box_blur_small_pipeline);
                } else {
                    compute_pass.set_pipeline(&self.box_blur_pipeline);
                }

                compute_pass.set_bind_group(0, &bind_group, &[]);
                let (dispatch_x, dispatch_y) = calculate_dispatch(down_width, down_height);
                compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            (output_texture, output_view)
        };

        let (final_output, _) = {
            let (texture, view) =
                self.create_output_texture(width as u32, height as u32, "Final Output");

            let params = UpsampleParams {
                src_width: down_width,
                src_height: down_height,
                dst_width: width as u32,
                dst_height: height as u32,
            };

            let param_buffer = create_uniform_buffer(&self.device, "Upsample Params", &params);

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Upsample Bind Group"),
                layout: &self.upsample_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&blurred_down_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: param_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Upsample"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.upsample_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (dispatch_x, dispatch_y) = calculate_dispatch(width as u32, height as u32);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            (texture, view)
        };

        self.queue.submit(Some(encoder.finish()));

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        Ok(final_output)
    }

    // ============================================================================
    // Utility Functions
    // ============================================================================

    /// Precompute Gaussian kernel weights packed into vec4<f32> arrays
    #[cfg(feature = "gpu")]
    fn precompute_gaussian_weights(&self, radius: u32) -> GaussianWeights {
        let kernel_size = (2 * radius + 1) as usize;
        let mut raw_weights = Vec::with_capacity(kernel_size);
        let mut weight_sum = 0.0;

        let sigma_sq_2 = 2.0 * self.sigma * self.sigma;

        for i in 0..kernel_size {
            let x = i as i32 - radius as i32;
            let weight = (-(x * x) as f32 / sigma_sq_2).exp();
            raw_weights.push(weight);
            weight_sum += weight;
        }

        let inv_weight_sum = 1.0 / weight_sum;
        for weight in raw_weights.iter_mut() {
            *weight *= inv_weight_sum;
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
}
