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
const DOWNSAMPLE_THRESHOLD: f32 = 5.0; // CHANGED FROM 32.0 TO 5.0

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
    _padding0: u32, // Padding to make 32 bytes
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
    _padding0: u32, // Padding to make 32 bytes
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
    /// 3-pass box blur approximation for medium sigmas (2.0-5.0)  // CHANGED FROM 32.0 TO 5.0
    Box3Pass,
    /// Downsample -> blur -> upsample for large sigmas (> 5.0)     // CHANGED FROM 32.0 TO 5.0
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

    // Pre-allocate and fill using extend_from_slice for each row
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
            eprintln!(
                "[INIT] Creating GPU blur processor with sigma={}, radius={:?}, blur_alpha={}",
                sigma, radius, blur_alpha
            );

            if sigma <= 0.0 {
                return Err(BlurError::InvalidSigma(sigma));
            }

            let radius = radius.unwrap_or_else(|| (3.0 * sigma).ceil() as i32);
            let strategy = Self::select_strategy(sigma);

            eprintln!("[INIT] Selected strategy: {:?}", strategy);
            eprintln!("[INIT] Using radius: {}", radius);

            let (device, queue, pipelines, layouts, sampler) = Self::initialize_gpu()
                .await
                .map_err(|e| BlurError::GpuError(format!("Failed to initialize GPU: {}", e)))?;

            eprintln!("[INIT] GPU initialized successfully");

            Ok(Self {
                device,
                queue,
                downsample_pipeline: pipelines.0,
                upsample_pipeline: pipelines.1,
                box_blur_pipeline: pipelines.2,
                gaussian_blur_pipeline: pipelines.3,
                downsample_bind_group_layout: layouts.0,
                upsample_bind_group_layout: layouts.1,
                box_blur_bind_group_layout: layouts.2,
                gaussian_blur_bind_group_layout: layouts.3,
                sampler,
                sigma,
                radius,
                blur_alpha,
                strategy,
            })
        }
    }

    /// Apply blur to an image and return as 2D pixel array
    pub fn blur(&mut self, image: &[Vec<Pixel>]) -> Result<Vec<Vec<Pixel>>, BlurError> {
        let (bytes, width, height) = self.blur_to_bytes(image)?;

        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }

        // Convert bytes back to 2D pixel array
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
            eprintln!("[BLUR] Starting blur operation");

            // Validate input
            if image.is_empty() || image[0].is_empty() {
                eprintln!("[BLUR] Empty image");
                return Ok((Vec::new(), 0, 0));
            }

            let height = image.len();
            let width = image[0].len();

            eprintln!("[BLUR] Image size: {}x{}", width, height);

            // Check GPU limits
            Self::validate_image_dimensions(&self.device, width, height)?;

            // Upload image to GPU
            eprintln!("[BLUR] Uploading image to GPU...");
            let (_input_texture, input_view) = self.upload_image_to_gpu(image, width, height)?;
            eprintln!("[BLUR] Image uploaded successfully");

            // Execute selected strategy
            eprintln!("[BLUR] Executing strategy: {:?}", self.strategy);
            let output_texture = match &self.strategy {
                BlurStrategy::Gaussian => {
                    eprintln!("[BLUR] Using Gaussian convolution");
                    self.apply_gaussian_blur(&input_view, width, height)?
                }
                BlurStrategy::Box3Pass => {
                    eprintln!("[BLUR] Using Box 3-pass approximation");
                    self.apply_box_blur_3pass(&input_view, width, height)?
                }
                BlurStrategy::Downsample(config) => {
                    eprintln!(
                        "[BLUR] Using Downsample->Blur->Upsample with factor={}, adjusted_sigma={}",
                        config.factor, config.adjusted_sigma
                    );
                    self.apply_downsample_blur_upsample(
                        &input_view,
                        width,
                        height,
                        config.factor,
                        config.adjusted_sigma,
                    )?
                }
            };

            // Download result from GPU
            eprintln!("[BLUR] Downloading result from GPU...");
            let result = self.download_texture_to_cpu(&output_texture, width, height)?;
            eprintln!(
                "[BLUR] Download complete, result size: {} bytes",
                result.0.len()
            );

            Ok(result)
        }
    }

    // ============================================================================
    // Strategy Selection
    // ============================================================================

    /// Select the optimal blur strategy based on sigma value
    fn select_strategy(sigma: f32) -> BlurStrategy {
        eprintln!("[STRATEGY] Selecting strategy for sigma={}", sigma);

        if sigma <= GAUSSIAN_THRESHOLD {
            eprintln!(
                "[STRATEGY] Using Gaussian convolution (sigma ≤ {})",
                GAUSSIAN_THRESHOLD
            );
            BlurStrategy::Gaussian
        } else if sigma <= DOWNSAMPLE_THRESHOLD {
            eprintln!(
                "[STRATEGY] Using Box 3-pass approximation (sigma {} to {})",
                GAUSSIAN_THRESHOLD, DOWNSAMPLE_THRESHOLD
            );
            BlurStrategy::Box3Pass
        } else {
            let factor = if sigma > LARGE_SIGMA_THRESHOLD {
                eprintln!(
                    "[STRATEGY] Using 8x downsampling (sigma > {})",
                    LARGE_SIGMA_THRESHOLD
                );
                8
            } else if sigma > 64.0 {
                eprintln!("[STRATEGY] Using 4x downsampling (sigma > 64.0)");
                4
            } else {
                eprintln!(
                    "[STRATEGY] Using 2x downsampling (sigma {} to 64.0)",
                    DOWNSAMPLE_THRESHOLD
                );
                2
            };

            let adjusted_sigma = sigma / factor as f32;
            eprintln!(
                "[STRATEGY] Factor: {}, Adjusted sigma: {}",
                factor, adjusted_sigma
            );

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
            wgpu::Sampler,
        ),
        String,
    > {
        eprintln!("[GPU] Initializing GPU...");

        let instance = Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| format!("Failed to find a suitable GPU adapter: {}", e))?;

        eprintln!("[GPU] Adapter found");

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

        eprintln!("[GPU] Device created successfully");

        // Create sampler for upsampling
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

        // Load and compile all shaders
        eprintln!("[GPU] Loading shaders...");

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

        eprintln!("[GPU] Shaders loaded");

        // Define bind group layout entries for different pipelines
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

        eprintln!("[GPU] Creating bind group layouts...");

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

        // Create compute pipelines
        eprintln!("[GPU] Creating compute pipelines...");

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
            create_pipeline(&device, &gaussian_layout, &gaussian_shader, "Gaussian"),
        );

        let layouts = (
            downsample_layout,
            upsample_layout,
            box_blur_layout,
            gaussian_layout,
        );

        eprintln!("[GPU] Initialization complete");

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

        eprintln!("[VALIDATE] Checking dimensions: {}x{}", width, height);
        eprintln!(
            "[VALIDATE] GPU max texture dimension: {}",
            device_limits.max_texture_dimension_2d
        );

        if width as u32 > device_limits.max_texture_dimension_2d {
            return Err(BlurError::InvalidDimensions { width, height });
        }

        if height as u32 > device_limits.max_texture_dimension_2d {
            return Err(BlurError::InvalidDimensions { width, height });
        }

        eprintln!("[VALIDATE] Dimensions OK");
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
        eprintln!(
            "[TEXTURE] Creating: {} ({}x{}), usage={:?}",
            label, width, height, usage
        );

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
        eprintln!("[UPLOAD] Converting image to bytes...");

        // Convert image to bytes
        let rgba_bytes = image_to_rgba_bytes(image, width, height);
        let bytes_per_row_aligned = calculate_aligned_row_size(width);

        eprintln!(
            "[UPLOAD] Image bytes: {} (aligned row: {} bytes)",
            rgba_bytes.len(),
            bytes_per_row_aligned
        );

        // Create padded data if needed
        let aligned_row_size = bytes_per_row_aligned as usize;
        let unaligned_row_size = width * BYTES_PER_PIXEL as usize;

        if aligned_row_size == unaligned_row_size {
            // No padding needed
            eprintln!("[UPLOAD] No padding needed");
            self.upload_bytes_to_gpu(&rgba_bytes, width, height, bytes_per_row_aligned)
        } else {
            // Add padding to each row
            eprintln!(
                "[UPLOAD] Adding padding (unaligned: {}, aligned: {})",
                unaligned_row_size, aligned_row_size
            );

            let mut padded_data = Vec::with_capacity(height * aligned_row_size);

            for y in 0..height {
                let src_start = y * unaligned_row_size;
                let src_end = src_start + unaligned_row_size;

                // Copy row data
                padded_data.extend_from_slice(&rgba_bytes[src_start..src_end]);

                // Add padding
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
        // Create input texture
        let texture = Self::create_texture(
            &self.device,
            width as u32,
            height as u32,
            "Input Texture",
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );

        let view = texture.create_view(&TextureViewDescriptor::default());

        eprintln!("[UPLOAD] Writing texture data ({} bytes)...", data.len());

        // Write image data to texture
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

        eprintln!("[UPLOAD] Upload complete");
        Ok((texture, view))
    }

    #[cfg(feature = "gpu")]
    fn download_texture_to_cpu(
        &self,
        texture: &Texture,
        width: usize,
        height: usize,
    ) -> Result<(Vec<u8>, usize, usize), BlurError> {
        eprintln!("[DOWNLOAD] Starting download...");

        let bytes_per_row_aligned = calculate_aligned_row_size(width);
        let buffer_size = (bytes_per_row_aligned as u64) * (height as u64);

        eprintln!(
            "[DOWNLOAD] Creating staging buffer ({} bytes)...",
            buffer_size
        );

        // Create staging buffer
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

        // Copy texture to buffer
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

        eprintln!("[DOWNLOAD] Submitting copy command...");
        self.queue.submit(Some(encoder.finish()));

        // Map buffer for reading
        let buffer_slice = staging_buffer.slice(..);

        // First, poll to ensure commands are submitted
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Then use map_async
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Poll with timeout
        eprintln!("[DOWNLOAD] Polling device (max 30 seconds)...");
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            // Poll the device
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // Check if we have a result from the receiver
            if let Ok(result) = receiver.try_recv() {
                match result {
                    Ok(()) => {
                        // Success! Buffer is ready
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

            // Small sleep to prevent busy waiting
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        eprintln!("[DOWNLOAD] Getting mapped data...");
        let data = buffer_slice.get_mapped_range();
        let unaligned_row_size = width * BYTES_PER_PIXEL as usize;
        let aligned_row_size = bytes_per_row_aligned as usize;

        let mut result = Vec::with_capacity(width * height * BYTES_PER_PIXEL as usize);

        for y in 0..height {
            let src_start = y * aligned_row_size;
            result.extend_from_slice(&data[src_start..src_start + unaligned_row_size]);
        }

        // Cleanup
        drop(data);
        staging_buffer.unmap();

        eprintln!("[DOWNLOAD] Download complete, got {} bytes", result.len());
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
        eprintln!(
            "[BUFFER] Creating uniform buffer: {} ({} bytes)",
            label,
            std::mem::size_of_val(data)
        );

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
        eprintln!(
            "[GAUSSIAN] Applying Gaussian blur: {}x{}, sigma={}, radius={}",
            width, height, self.sigma, self.radius
        );

        let width_u32 = width as u32;
        let height_u32 = height as u32;

        // Precompute Gaussian kernel weights
        eprintln!("[GAUSSIAN] Precomputing Gaussian weights...");
        let weights_data = self.precompute_gaussian_weights(self.radius as u32);
        let weights_buffer = self.create_uniform_buffer(&weights_data, "Gaussian Weights Buffer");

        // Create textures
        eprintln!("[GAUSSIAN] Creating intermediate texture...");
        let (_intermediate_texture, intermediate_view) =
            self.create_intermediate_texture(width_u32, height_u32, "Gaussian Intermediate");

        eprintln!("[GAUSSIAN] Creating output texture...");
        let (output_texture, output_view) =
            self.create_output_texture(width_u32, height_u32, "Gaussian Output");

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gaussian Blur"),
            });

        // Horizontal pass
        eprintln!("[GAUSSIAN] Running horizontal pass...");
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
            eprintln!(
                "[GAUSSIAN] Horizontal dispatch: {}x{}",
                dispatch_x, dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Vertical pass
        eprintln!("[GAUSSIAN] Running vertical pass...");
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
            eprintln!(
                "[GAUSSIAN] Vertical dispatch: {}x{}",
                dispatch_x, dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        eprintln!("[GAUSSIAN] Submitting commands...");
        self.queue.submit(Some(encoder.finish()));

        // Poll device to ensure completion
        eprintln!("[GAUSSIAN] Waiting for GPU to complete...");
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            if start_time.elapsed() > timeout {
                return Err(BlurError::Timeout(
                    "Timeout waiting for Gaussian blur completion".to_string(),
                ));
            }

            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // For simplicity, we break after polling
            // In a production system, you might want to check actual completion
            break;
        }

        eprintln!("[GAUSSIAN] Gaussian blur completed");
        Ok(output_texture)
    }

    #[cfg(feature = "gpu")]
    fn apply_box_blur_3pass(
        &self,
        input_view: &TextureView,
        width: usize,
        height: usize,
    ) -> Result<Texture, BlurError> {
        eprintln!(
            "[BOX] Applying Box 3-pass blur: {}x{}, sigma={}",
            width, height, self.sigma
        );

        let width_u32 = width as u32;
        let height_u32 = height as u32;

        // Calculate box sizes for 3-pass approximation
        let box_radii = calculate_box_radii(self.sigma);
        eprintln!("[BOX] Box radii: {:?}", box_radii);

        // Check if any radius is 0
        for (i, &radius) in box_radii.iter().enumerate() {
            if radius == 0 {
                eprintln!("[WARNING] Box radius {} is 0!", i);
            }
        }

        // Create textures
        eprintln!("[BOX] Creating intermediate textures...");
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

        // Apply 6 blur passes (3 passes × 2 directions)
        let blur_passes = [
            (input_view, &view1, box_radii[0], 0u32, "Pass 1 Horizontal"),
            (&view1, &view2, box_radii[0], 1u32, "Pass 1 Vertical"),
            (&view2, &view1, box_radii[1], 0u32, "Pass 2 Horizontal"),
            (&view1, &view2, box_radii[1], 1u32, "Pass 2 Vertical"),
            (&view2, &view1, box_radii[2], 0u32, "Pass 3 Horizontal"),
            (&view1, &output_view, box_radii[2], 1u32, "Pass 3 Vertical"),
        ];

        for (i, (input_view, output_view, radius, direction, label)) in
            blur_passes.iter().enumerate()
        {
            eprintln!(
                "[BOX] Running {} (radius={}, direction={})",
                label, radius, direction
            );

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

            compute_pass.set_pipeline(&self.box_blur_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let (dispatch_x, dispatch_y) = calculate_dispatch(width_u32, height_u32);
            eprintln!(
                "[BOX] Pass {} dispatch: {}x{}",
                i + 1,
                dispatch_x,
                dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        eprintln!("[BOX] Submitting commands...");
        self.queue.submit(Some(encoder.finish()));

        // Poll device to ensure completion
        eprintln!("[BOX] Waiting for GPU to complete...");
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            if start_time.elapsed() > timeout {
                return Err(BlurError::Timeout(
                    "Timeout waiting for box blur completion".to_string(),
                ));
            }

            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // For simplicity, we break after polling
            // In a production system, you might want to check actual completion
            break;
        }

        eprintln!("[BOX] Box blur completed");
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
        eprintln!(
            "[DOWNSAMPLE] Applying Downsample->Blur->Upsample: {}x{}, factor={}, adjusted_sigma={}",
            width, height, factor, adjusted_sigma
        );

        // Calculate downsampled dimensions
        let (down_width, down_height) = calculate_downsampled_dimensions(width, height, factor);
        eprintln!(
            "[DOWNSAMPLE] Downsampled dimensions: {}x{}",
            down_width, down_height
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Downsample Blur Upsample"),
            });

        // Step 1: Downsample
        eprintln!("[DOWNSAMPLE] Step 1: Downsampling...");
        let (_downsampled_texture, downsampled_view) = {
            let (texture, view) =
                self.create_intermediate_texture(down_width, down_height, "Downsampled");

            let params = DownsampleParams {
                src_width: width as u32,
                src_height: height as u32,
                dst_width: down_width,
                dst_height: down_height,
            };

            eprintln!(
                "[DOWNSAMPLE] Downsample params: src={}x{}, dst={}x{}",
                params.src_width, params.src_height, params.dst_width, params.dst_height
            );

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
            eprintln!(
                "[DOWNSAMPLE] Downsample dispatch: {}x{}",
                dispatch_x, dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            (texture, view)
        };

        // Step 2: Apply box blur on downsampled image
        eprintln!("[DOWNSAMPLE] Step 2: Applying box blur on downsampled image...");
        let (_blurred_down_texture, blurred_down_view) = {
            // Calculate box sizes for adjusted sigma
            let box_radii = calculate_box_radii(adjusted_sigma);
            eprintln!("[DOWNSAMPLE] Downsampled box radii: {:?}", box_radii);

            // Create textures for blur passes
            let (_texture1, view1) =
                self.create_intermediate_texture(down_width, down_height, "Downsampled Blur 1");
            let (_texture2, view2) =
                self.create_intermediate_texture(down_width, down_height, "Downsampled Blur 2");
            let (output_texture, output_view) =
                self.create_output_texture(down_width, down_height, "Blurred Downsampled");

            // Apply 6 blur passes
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

            for (i, (input_view, output_view, radius, direction, label)) in
                blur_passes.iter().enumerate()
            {
                eprintln!(
                    "[DOWNSAMPLE] Running {} (radius={}, direction={})",
                    label, radius, direction
                );

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

                compute_pass.set_pipeline(&self.box_blur_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let (dispatch_x, dispatch_y) = calculate_dispatch(down_width, down_height);
                eprintln!(
                    "[DOWNSAMPLE] Pass {} dispatch: {}x{}",
                    i + 1,
                    dispatch_x,
                    dispatch_y
                );
                compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            (output_texture, output_view)
        };

        // Step 3: Upsample
        eprintln!("[DOWNSAMPLE] Step 3: Upsampling...");
        let (final_output, _) = {
            let (texture, view) =
                self.create_output_texture(width as u32, height as u32, "Final Output");

            let params = UpsampleParams {
                src_width: down_width,
                src_height: down_height,
                dst_width: width as u32,
                dst_height: height as u32,
            };

            eprintln!(
                "[DOWNSAMPLE] Upsample params: src={}x{}, dst={}x{}",
                params.src_width, params.src_height, params.dst_width, params.dst_height
            );

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
            eprintln!(
                "[DOWNSAMPLE] Upsample dispatch: {}x{}",
                dispatch_x, dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

            (texture, view)
        };

        eprintln!("[DOWNSAMPLE] Submitting all commands...");
        self.queue.submit(Some(encoder.finish()));

        // Poll device to ensure completion
        eprintln!("[DOWNSAMPLE] Waiting for GPU to complete...");
        let start_time = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(30);

        loop {
            if start_time.elapsed() > timeout {
                return Err(BlurError::Timeout(
                    "Timeout waiting for downsample completion".to_string(),
                ));
            }

            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // For simplicity, we break after polling
            // In a production system, you might want to check actual completion
            break;
        }

        eprintln!("[DOWNSAMPLE] Downsample->Blur->Upsample completed");
        Ok(final_output)
    }

    // ============================================================================
    // Utility Functions
    // ============================================================================

    /// Precompute Gaussian kernel weights packed into vec4<f32> arrays
    #[cfg(feature = "gpu")]
    fn precompute_gaussian_weights(&self, radius: u32) -> GaussianWeights {
        eprintln!(
            "[WEIGHTS] Precomputing Gaussian weights for radius={}, sigma={}",
            radius, self.sigma
        );

        let kernel_size = (2 * radius + 1) as usize;
        let mut raw_weights = Vec::with_capacity(kernel_size);
        let mut weight_sum = 0.0;

        // Precompute constants
        let sigma_sq_2 = 2.0 * self.sigma * self.sigma;

        for i in 0..kernel_size {
            let x = i as i32 - radius as i32;
            let weight = (-(x * x) as f32 / sigma_sq_2).exp();
            raw_weights.push(weight);
            weight_sum += weight;
        }

        eprintln!("[WEIGHTS] Weight sum before normalization: {}", weight_sum);

        // Normalize weights
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

        eprintln!("[WEIGHTS] Weights computed, vec4 count: {}", vec4_count);
        weights_data
    }
}
