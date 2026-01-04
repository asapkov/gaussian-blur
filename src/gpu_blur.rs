//! GPU-accelerated Gaussian Blur using wgpu with optimized multi-shader approach

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, ComputePipeline, ComputePipelineDescriptor, Device,
    Instance, Queue, ShaderModuleDescriptor, ShaderSource, StorageTextureAccess, TextureDescriptor,
    TextureFormat, TextureViewDimension,
};

#[cfg(feature = "gpu")]
use bytemuck;

use crate::Pixel;

// Strategy selection thresholds
const GAUSSIAN_THRESHOLD: f32 = 2.0;
const DOWNSAMPLE_THRESHOLD: f32 = 32.0;
const LARGE_SIGMA_THRESHOLD: f32 = 100.0;

// Shader parameters structs
#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _padding: [u32; 8], // Adjust to reach 48 bytes
}

#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct UpsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _padding: [u32; 8], // Adjust to reach 48 bytes
}

#[cfg(feature = "gpu")]
#[repr(C, align(16))] // Align to 16 bytes
#[derive(Debug, Clone, Copy)]
struct BoxBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    _padding: [u32; 7], // Increase from 3 to 7 to reach 48 bytes total
}

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
    _padding: [u32; 6], // Adjust to reach 48 bytes
}

// Gaussian weights struct (256 vec4s = 1024 weights)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GaussianWeights {
    weights: [[f32; 4]; 256], // Packed as vec4<f32> for 16-byte alignment
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

/// Blur strategy based on sigma value
enum BlurStrategy {
    Gaussian, // True Gaussian convolution
    Box3Pass, // 3-pass box blur approximation
    Downsample {
        factor: u32,         // Downscale factor (2, 4, or 8)
        adjusted_sigma: f32, // Sigma after downsampling
    },
}

/// GPU Gaussian Blur processor with optimized multi-shader approach
pub struct GpuGaussianBlur {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    queue: Queue,

    // Separate pipelines for each operation
    #[cfg(feature = "gpu")]
    downsample_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    upsample_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    box_blur_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    gaussian_blur_pipeline: ComputePipeline,

    // Bind group layouts
    #[cfg(feature = "gpu")]
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    #[cfg(feature = "gpu")]
    upsample_bind_group_layout: wgpu::BindGroupLayout,
    #[cfg(feature = "gpu")]
    box_blur_bind_group_layout: wgpu::BindGroupLayout,
    #[cfg(feature = "gpu")]
    gaussian_blur_bind_group_layout: wgpu::BindGroupLayout,

    sigma: f32,
    radius: i32,
    blur_alpha: bool,
    strategy: BlurStrategy,
}

impl GpuGaussianBlur {
    /// Create a new GPU Gaussian Blur processor
    pub async fn new(sigma: f32, radius: Option<i32>, blur_alpha: bool) -> Result<Self, String> {
        let radius = radius.unwrap_or_else(|| (3.0 * sigma).ceil() as i32);

        // Select strategy based on sigma value
        let strategy = Self::select_strategy(sigma);

        #[cfg(not(feature = "gpu"))]
        {
            return Err("GPU feature not enabled. Build with --features gpu".to_string());
        }

        #[cfg(feature = "gpu")]
        {
            // Initialize wgpu
            let instance = Instance::default();

            // Find adapter
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .expect("Failed to find a suitable GPU adapter");

            let info = adapter.get_info();
            println!("Selected adapter: {} ({:?})", info.name, info.device_type);

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
                .expect("Failed to create device");

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

            let gaussian_blur_shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Gaussian Blur Shader"),
                source: ShaderSource::Wgsl(
                    include_str!("shaders/gaussian_blur_separable.wgsl").into(),
                ),
            });

            // Create bind group layouts for each shader
            let downsample_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Downsample Bind Group Layout"),
                    entries: &[
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
                                format: TextureFormat::Rgba8Unorm,
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
                    ],
                });

            let upsample_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Upsample Bind Group Layout"),
                    entries: &[
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
                                format: TextureFormat::Rgba8Unorm,
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
                    ],
                });

            let box_blur_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Box Blur Bind Group Layout"),
                    entries: &[
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
                                format: TextureFormat::Rgba8Unorm,
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
                    ],
                });

            let gaussian_blur_bind_group_layout =
                device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Gaussian Blur Bind Group Layout"),
                    entries: &[
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
                                format: TextureFormat::Rgba8Unorm,
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
                            ty: BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            // Create pipeline layouts
            let downsample_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Downsample Pipeline Layout"),
                    bind_group_layouts: &[&downsample_bind_group_layout],
                    immediate_size: 0,
                });

            let upsample_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Upsample Pipeline Layout"),
                    bind_group_layouts: &[&upsample_bind_group_layout],
                    immediate_size: 0,
                });

            let box_blur_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Box Blur Pipeline Layout"),
                    bind_group_layouts: &[&box_blur_bind_group_layout],
                    immediate_size: 0,
                });

            let gaussian_blur_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Gaussian Blur Pipeline Layout"),
                    bind_group_layouts: &[&gaussian_blur_bind_group_layout],
                    immediate_size: 0,
                });

            // Create compute pipelines
            let downsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Downsample Pipeline"),
                layout: Some(&downsample_pipeline_layout),
                module: &downsample_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let upsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Upsample Pipeline"),
                layout: Some(&upsample_pipeline_layout),
                module: &upsample_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let box_blur_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Box Blur Pipeline"),
                layout: Some(&box_blur_pipeline_layout),
                module: &box_blur_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

            let gaussian_blur_pipeline =
                device.create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Gaussian Blur Pipeline"),
                    layout: Some(&gaussian_blur_pipeline_layout),
                    module: &gaussian_blur_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            Ok(Self {
                device,
                queue,
                downsample_pipeline,
                upsample_pipeline,
                box_blur_pipeline,
                gaussian_blur_pipeline,
                downsample_bind_group_layout,
                upsample_bind_group_layout,
                box_blur_bind_group_layout,
                gaussian_blur_bind_group_layout,
                sigma,
                radius,
                blur_alpha,
                strategy,
            })
        }
    }

    /// Select the optimal blur strategy based on sigma value
    fn select_strategy(sigma: f32) -> BlurStrategy {
        if sigma <= GAUSSIAN_THRESHOLD {
            println!("Strategy: True Gaussian convolution (sigma={:.2})", sigma);
            BlurStrategy::Gaussian
        } else if sigma <= DOWNSAMPLE_THRESHOLD {
            println!(
                "Strategy: 3-pass box blur approximation (sigma={:.2})",
                sigma
            );
            BlurStrategy::Box3Pass
        } else {
            // Determine optimal downscale factor
            let factor = if sigma > LARGE_SIGMA_THRESHOLD {
                8 // 8x downsampling for very large sigmas
            } else if sigma > 64.0 {
                4 // 4x downsampling
            } else {
                2 // 2x downsampling for sigmas > 32
            };

            let adjusted_sigma = sigma / factor as f32;

            println!(
                "Strategy: {factor}x downsampling + blur (sigma={:.2} -> {:.2})",
                sigma, adjusted_sigma
            );

            BlurStrategy::Downsample {
                factor,
                adjusted_sigma,
            }
        }
    }

    /// Calculate optimal box sizes for 3-pass Gaussian approximation
    fn boxes_for_gauss_3pass(sigma: f32) -> [usize; 3] {
        assert!(sigma > 0.0, "Sigma must be positive");

        let n = 3; // Always use 3 passes for good approximation

        // Ideal box width based on Central Limit Theorem
        let w_ideal = ((12.0 * sigma * sigma / n as f32) + 1.0).sqrt();

        // Lower odd integer (floor to nearest odd)
        let mut wl = w_ideal.floor() as usize;
        if wl % 2 == 0 {
            wl -= 1;
        }
        if wl < 1 {
            wl = 1; // Minimum box size
        }

        // Upper odd integer
        let wu = wl + 2;

        // Calculate distribution of sizes
        let numerator =
            12.0 * sigma * sigma - (n * wl * wl) as f32 - (4 * n * wl) as f32 - (3 * n) as f32;

        let denominator = -4.0 * wl as f32 - 4.0;
        let m_ideal = numerator / denominator;

        // Number of passes using wl size
        let m = m_ideal.round().max(0.0).min(n as f32) as usize;

        // Create array of box sizes
        let mut sizes = [wl; 3];
        for i in m..3 {
            sizes[i] = wu;
        }

        // Debug output
        println!("=== 3-Pass Box Blur Approximation ===");
        println!("Target sigma: {:.2}", sigma);
        println!("Ideal box width: {:.2}", w_ideal);
        println!("Box sizes: {:?} (wl={}, wu={})", sizes, wl, wu);
        println!(
            "Approximated sigma: {:.2}",
            (sizes.iter().map(|&w| w as f32 * w as f32).sum::<f32>() / 12.0).sqrt()
        );
        println!();

        sizes
    }

    /// Precompute Gaussian kernel weights packed into vec4<f32> arrays
    #[cfg(feature = "gpu")]
    fn precompute_gaussian_weights(&self, radius: u32, sigma: f32) -> GaussianWeights {
        let kernel_size = (2 * radius + 1) as usize;
        let mut raw_weights = Vec::with_capacity(kernel_size);
        let mut weight_sum = 0.0;

        // Compute raw weights
        for i in 0..kernel_size {
            let x = i as i32 - radius as i32;
            let weight = (-(x * x) as f32 / (2.0 * sigma * sigma)).exp();
            raw_weights.push(weight);
            weight_sum += weight;
        }

        // Normalize weights
        for weight in raw_weights.iter_mut() {
            *weight /= weight_sum;
        }

        // Pack into vec4 arrays (4 weights per vec4)
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

    /// Apply blur to an image and return as 2D pixel array
    pub fn blur(&self, image: &[Vec<Pixel>]) -> Result<Vec<Vec<Pixel>>, String> {
        let (bytes, width, height) = self.blur_to_bytes(image)?;

        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }

        // Convert bytes back to pixels
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

    /// Test basic GPU write functionality
    #[cfg(feature = "gpu")]
    fn test_simple_write(&self, width: usize, height: usize) -> Result<bool, String> {
        println!("\n=== Testing Simple GPU Write ===");

        // Create a simple test texture
        let test_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Test Texture"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let test_view = test_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Simple test shader that writes a gradient pattern
        let test_shader = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Test Shader"),
            source: ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0)
                var output_texture: texture_storage_2d<rgba8unorm, write>;

                @compute @workgroup_size(8, 8, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let x = global_id.x;
                    let y = global_id.y;
                    
                    // Write gradient pattern
                    let r = f32(x % 256u) / 255.0;
                    let g = f32(y % 256u) / 255.0;
                    let b = 0.5;
                    let a = 1.0;
                    
                    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(r, g, b, a));
                }
                "#
                .into(),
            ),
        });

        // Create bind group layout
        let test_bind_group_layout =
            self.device
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("Test Bind Group Layout"),
                    entries: &[BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    }],
                });

        let test_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Test Pipeline Layout"),
                    bind_group_layouts: &[&test_bind_group_layout],
                    immediate_size: 0,
                });

        let test_pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Test Pipeline"),
                layout: Some(&test_pipeline_layout),
                module: &test_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let test_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Test Bind Group"),
            layout: &test_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&test_view),
            }],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Test Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Test Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&test_pipeline);
            compute_pass.set_bind_group(0, &test_bind_group, &[]);

            // Dispatch enough workgroups to cover the entire image
            let dispatch_x = (width as u32 + 7) / 8;
            let dispatch_y = (height as u32 + 7) / 8;
            println!("Test dispatch: {}x{} workgroups", dispatch_x, dispatch_y);
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // Copy to buffer
        let bytes_per_pixel = 4u32;
        let alignment = 256;
        let bytes_per_row_aligned =
            ((bytes_per_pixel * width as u32 + alignment - 1) / alignment) * alignment;
        let output_buffer_size =
            (bytes_per_row_aligned as u64 * height as u64) as wgpu::BufferAddress;

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &test_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
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

        // Read back
        let slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        receiver
            .recv()
            .map_err(|e| format!("Failed to receive buffer: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {}", e))?;

        let data = slice.get_mapped_range();

        // Check first few pixels
        println!("Test results - First 4 pixels:");
        for i in 0..4.min(width) {
            let offset = i * 4;
            if offset + 3 < data.len() {
                println!(
                    "  Pixel {}: R={}, G={}, B={}, A={}",
                    i,
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3]
                );
            }
        }

        // Check if any pixel is non-zero
        let mut has_non_zero = false;
        for &byte in data.iter().take(100.min(data.len())) {
            if byte != 0 {
                has_non_zero = true;
                break;
            }
        }

        println!("Test completed. Has non-zero pixels: {}", has_non_zero);
        Ok(has_non_zero)
    }

    /// Apply blur to an image using GPU with optimal strategy
    pub fn blur_to_bytes(&self, image: &[Vec<Pixel>]) -> Result<(Vec<u8>, usize, usize), String> {
        #[cfg(not(feature = "gpu"))]
        {
            return Err("GPU feature not enabled".to_string());
        }

        #[cfg(feature = "gpu")]
        {
            use std::time::Instant;

            let total_start = Instant::now();

            if image.is_empty() || image[0].is_empty() {
                return Ok((Vec::new(), 0, 0));
            }

            let height = image.len();
            let width = image[0].len();

            println!("Processing image: {}x{} pixels", width, height);

            // Check GPU limits
            let device_limits = self.device.limits();

            if width as u32 > device_limits.max_texture_dimension_2d {
                return Err(format!(
                    "Image width {} exceeds GPU texture dimension limit {}",
                    width, device_limits.max_texture_dimension_2d
                ));
            }

            if height as u32 > device_limits.max_texture_dimension_2d {
                return Err(format!(
                    "Image height {} exceeds GPU texture dimension limit {}",
                    height, device_limits.max_texture_dimension_2d
                ));
            }

            // Test basic GPU functionality first
            println!("Testing basic GPU write functionality...");
            match self.test_simple_write(width.min(16), height.min(16)) {
                Ok(true) => println!("✓ GPU write test passed"),
                Ok(false) => {
                    println!("✗ GPU write test failed - all pixels are zero!");
                    return Err(
                        "GPU write test failed - check shader compilation and texture usage"
                            .to_string(),
                    );
                }
                Err(e) => {
                    println!("✗ GPU write test error: {}", e);
                    return Err(format!("GPU write test error: {}", e));
                }
            }

            // Convert image to flat RGBA bytes
            let mut rgba_data = Vec::with_capacity(width * height * 4);
            for row in image {
                for pixel in row {
                    rgba_data.push(pixel.r);
                    rgba_data.push(pixel.g);
                    rgba_data.push(pixel.b);
                    rgba_data.push(pixel.a);
                }
            }

            // Create input texture
            let input_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Input Texture"),
                size: wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            // Write image data to texture
            let bytes_per_row_unaligned = 4 * width as u32;
            let alignment = 256;
            let bytes_per_row_aligned =
                ((bytes_per_row_unaligned + alignment - 1) / alignment) * alignment;

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &input_texture,
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

            let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Execute selected strategy and get the actual blurred texture
            let output_texture = match &self.strategy {
                BlurStrategy::Gaussian => {
                    let (_, texture) = self.apply_gaussian_blur(&input_view, width, height)?;
                    texture
                }
                BlurStrategy::Box3Pass => {
                    let (_, texture) = self.apply_box_blur_3pass(&input_view, width, height)?;
                    texture
                }
                BlurStrategy::Downsample {
                    factor,
                    adjusted_sigma,
                } => {
                    let (_, texture) = self.apply_downsample_blur_upsample(
                        &input_view,
                        width,
                        height,
                        *factor,
                        *adjusted_sigma,
                    )?;
                    texture
                }
            };

            // Copy result to CPU
            println!("\n=== Copying results to CPU ===");

            let bytes_per_pixel = 4u32;
            let bytes_per_row_aligned =
                ((bytes_per_pixel * width as u32 + alignment - 1) / alignment) * alignment;
            let output_buffer_size =
                (bytes_per_row_aligned as u64 * height as u64) as wgpu::BufferAddress;

            let final_output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Final Output Buffer"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Create encoder for final copy
            let mut final_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Final Copy Encoder"),
                    });

            // Copy texture to buffer - USE THE ACTUAL OUTPUT TEXTURE
            final_encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &output_texture, // CHANGED: Use output_texture from blur operation
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &final_output_buffer,
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

            // Submit final copy
            self.queue.submit(Some(final_encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // Read back image results
            let buffer_slice = final_output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            receiver
                .recv()
                .map_err(|e| format!("Failed to receive buffer: {}", e))?
                .map_err(|e| format!("Failed to map buffer: {}", e))?;

            let data = buffer_slice.get_mapped_range();

            // Extract data, handling row alignment padding
            let mut result_bytes = Vec::with_capacity(width * height * 4);
            let aligned_row_size_bytes = bytes_per_row_aligned as usize;

            for row in 0..height {
                let row_start = row * aligned_row_size_bytes;
                let row_end = row_start + (width * 4);

                if row_end <= data.len() {
                    result_bytes.extend_from_slice(&data[row_start..row_end]);
                } else {
                    let available = data.len().saturating_sub(row_start);
                    if available > 0 {
                        result_bytes.extend_from_slice(&data[row_start..row_start + available]);
                    }
                    let needed = width * 4 - available.min(width * 4);
                    result_bytes.extend(std::iter::repeat(0u8).take(needed));
                }
            }

            // Verify we got the right amount of data
            let expected_bytes = width * height * 4;
            if result_bytes.len() != expected_bytes {
                if result_bytes.len() < expected_bytes {
                    let needed = expected_bytes - result_bytes.len();
                    result_bytes.extend(std::iter::repeat(0u8).take(needed));
                } else {
                    result_bytes.truncate(expected_bytes);
                }
            }

            // Analyze output
            println!("\n=== Output Analysis ===");
            println!("Image size: {}x{}", width, height);
            println!("Total bytes: {}", result_bytes.len());

            // Count non-zero pixels
            let mut non_zero_pixels = 0;
            let mut total_pixels_checked = 0;
            let check_limit = 1000.min(width * height);

            for i in 0..check_limit {
                let offset = i * 4;
                if offset + 3 < result_bytes.len() {
                    total_pixels_checked += 1;
                    if result_bytes[offset] != 0
                        || result_bytes[offset + 1] != 0
                        || result_bytes[offset + 2] != 0
                        || result_bytes[offset + 3] != 0
                    {
                        non_zero_pixels += 1;

                        // Print first few non-zero pixels
                        if non_zero_pixels <= 5 {
                            println!(
                                "  Non-zero pixel {}: R={}, G={}, B={}, A={}",
                                i,
                                result_bytes[offset],
                                result_bytes[offset + 1],
                                result_bytes[offset + 2],
                                result_bytes[offset + 3]
                            );
                        }
                    }
                }
            }

            println!(
                "Checked {} pixels, {} are non-zero",
                total_pixels_checked, non_zero_pixels
            );

            if non_zero_pixels == 0 {
                println!("⚠️ WARNING: All checked pixels are zero!");
                println!("Possible issues:");
                println!("  1. Shader compilation failed");
                println!("  2. Texture usage flags incorrect");
                println!("  3. Dispatch size doesn't cover image");
                println!("  4. Coordinate clamping issues in shader");
            }

            // Cleanup - data is automatically unmapped when dropped
            drop(data);

            println!("Total GPU time: {:?}", total_start.elapsed());

            if let BlurStrategy::Downsample { factor, .. } = &self.strategy {
                let speedup = factor * factor;
                println!(
                    "Performance: ~{}x faster than full-resolution blur",
                    speedup
                );
            }

            Ok((result_bytes, width, height))
        }
    }

    /// Apply true Gaussian blur (for small sigmas)
    #[cfg(feature = "gpu")]
    fn apply_gaussian_blur(
        &self,
        input_view: &wgpu::TextureView,
        width: usize,
        height: usize,
    ) -> Result<(wgpu::TextureView, wgpu::Texture), String> {
        println!("\n=== Applying True Gaussian Blur ===");

        // Precompute Gaussian kernel weights
        let weights_data = self.precompute_gaussian_weights(self.radius as u32, self.sigma);
        println!("Gaussian kernel size: {}", (2 * self.radius as u32 + 1));

        let weights_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gaussian Weights Buffer"),
                contents: bytemuck::cast_slice(&[weights_data]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create intermediate texture for vertical pass
        let intermediate_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Intermediate Texture"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let intermediate_view =
            intermediate_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create output texture
        let output_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Gaussian Blur Output"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Horizontal pass
        println!("Horizontal pass...");
        let horiz_params = GaussianBlurParams {
            width: width as u32,
            height: height as u32,
            radius: self.radius as u32,
            blur_alpha: self.blur_alpha as u32,
            direction: 0, // Horizontal
            sigma: self.sigma,
            _padding: [0; 6],
        };

        let horiz_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Horizontal Gaussian Params"),
                    contents: bytemuck::cast_slice(&[horiz_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let horiz_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Horizontal Gaussian Bind Group"),
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
                    resource: horiz_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: weights_buffer.as_entire_binding(),
                },
            ],
        });

        let mut horiz_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Horizontal Gaussian Encoder"),
                });

        {
            let mut compute_pass = horiz_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Horizontal Gaussian Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.gaussian_blur_pipeline);
            compute_pass.set_bind_group(0, &horiz_bind_group, &[]);

            // Dispatch one workgroup per row for horizontal blur
            let dispatch_x = (width as u32 + 255) / 256;
            let dispatch_y = height as u32;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(horiz_encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Vertical pass
        println!("Vertical pass...");
        let vert_params = GaussianBlurParams {
            width: width as u32,
            height: height as u32,
            radius: self.radius as u32,
            blur_alpha: self.blur_alpha as u32,
            direction: 1, // Vertical
            sigma: self.sigma,
            _padding: [0; 6],
        };

        let vert_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertical Gaussian Params"),
                    contents: bytemuck::cast_slice(&[vert_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let vert_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Vertical Gaussian Bind Group"),
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
                    resource: vert_params_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: weights_buffer.as_entire_binding(),
                },
            ],
        });

        let mut vert_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Vertical Gaussian Encoder"),
                });

        {
            let mut compute_pass = vert_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vertical Gaussian Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.gaussian_blur_pipeline);
            compute_pass.set_bind_group(0, &vert_bind_group, &[]);

            // Dispatch one workgroup per column for vertical blur
            let dispatch_x = width as u32;
            let dispatch_y = (height as u32 + 255) / 256;
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(vert_encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        println!("Gaussian blur completed");
        Ok((output_view, output_texture))
    }

    /// Apply 3-pass box blur approximation
    #[cfg(feature = "gpu")]
    fn apply_box_blur_3pass(
        &self,
        input_view: &wgpu::TextureView,
        width: usize,
        height: usize,
    ) -> Result<(wgpu::TextureView, wgpu::Texture), String> {
        println!("\n=== Applying 3-Pass Box Blur ===");

        // Calculate box sizes for 3-pass approximation
        let box_sizes = Self::boxes_for_gauss_3pass(self.sigma);
        let box_radii = box_sizes.map(|size| ((size as i32 - 1) / 2).max(0) as u32);

        println!("Box radii: {:?}", box_radii);

        // Create intermediate textures
        let texture1 = self.device.create_texture(&TextureDescriptor {
            label: Some("Intermediate Texture 1"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let texture2 = self.device.create_texture(&TextureDescriptor {
            label: Some("Intermediate Texture 2"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let output_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Box Blur Output"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let view1 = texture1.create_view(&wgpu::TextureViewDescriptor::default());
        let view2 = texture2.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Apply 6 blur passes (3 passes × 2 directions)
        let blur_passes = [
            (input_view, &view1, box_radii[0], 0u32),
            (&view1, &view2, box_radii[0], 1u32),
            (&view2, &view1, box_radii[1], 0u32),
            (&view1, &view2, box_radii[1], 1u32),
            (&view2, &view1, box_radii[2], 0u32),
            (&view1, &output_view, box_radii[2], 1u32),
        ];

        for (i, (input_view, output_view, radius, direction)) in blur_passes.iter().enumerate() {
            println!("--- Box Blur Pass {} of 6 ---", i + 1);

            let params = BoxBlurParams {
                width: width as u32,
                height: height as u32,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                direction: *direction,
                _padding: [0; 7],
            };

            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Box Blur Params {}", i + 1)),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("Box Blur Bind Group {}", i + 1)),
                layout: &self.box_blur_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(*input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(*output_view),
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
                    label: Some(&format!("Box Blur Encoder {}", i + 1)),
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Box Blur Compute Pass {}", i + 1)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.box_blur_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch based on direction
                if *direction == 0 {
                    // Horizontal blur: one workgroup per row
                    let dispatch_x = (width as u32 + 255) / 256;
                    let dispatch_y = height as u32;
                    compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                } else {
                    // Vertical blur: one workgroup per column
                    let dispatch_x = width as u32;
                    let dispatch_y = (height as u32 + 255) / 256;
                    compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }
            }

            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            if (i + 1) % 2 == 0 {
                println!("  Completed pass {}/3", (i + 1) / 2);
            }
        }

        println!("Box blur completed");
        Ok((output_view, output_texture))
    }

    /// Apply downsample -> blur -> upsample pipeline
    #[cfg(feature = "gpu")]
    fn apply_downsample_blur_upsample(
        &self,
        input_view: &wgpu::TextureView,
        width: usize,
        height: usize,
        factor: u32,
        adjusted_sigma: f32,
    ) -> Result<(wgpu::TextureView, wgpu::Texture), String> {
        println!("\n=== Downsample -> Blur -> Upsample Pipeline ===");
        println!("Downscale factor: {}x", factor);
        println!("Adjusted sigma: {:.2}", adjusted_sigma);

        // Calculate downsampled dimensions
        let down_width = (width as u32 + factor - 1) / factor;
        let down_height = (height as u32 + factor - 1) / factor;
        println!("Downsampled size: {}x{}", down_width, down_height);

        // === STEP 1: Downsample ===
        println!("\n=== Step 1: Downsampling ===");

        let downsampled_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Downsampled Texture"),
            size: wgpu::Extent3d {
                width: down_width,
                height: down_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let downsampled_view =
            downsampled_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Downsample parameters
        let down_params = DownsampleParams {
            src_width: width as u32,
            src_height: height as u32,
            dst_width: down_width,
            dst_height: down_height,
            _padding: [0; 8],
        };

        let down_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Downsample Params"),
                    contents: bytemuck::cast_slice(&[down_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let down_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&downsampled_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: down_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut down_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Downsample Encoder"),
                });

        {
            let mut compute_pass = down_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Downsample Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.downsample_pipeline);
            compute_pass.set_bind_group(0, &down_bind_group, &[]);

            // Dispatch one workgroup per 8x8 block of downsampled image
            let dispatch_x = (down_width + 7) / 8;
            let dispatch_y = (down_height + 7) / 8;
            println!(
                "Downsample dispatch: {}x{} workgroups",
                dispatch_x, dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(down_encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        println!("Downsample completed");

        // === STEP 2: Apply blur on downsampled image ===
        println!("\n=== Step 2: Applying blur on downsampled image ===");

        // Create intermediate textures for blur
        let intermediate_texture1 = self.device.create_texture(&TextureDescriptor {
            label: Some("Intermediate Texture 1 (Downsampled)"),
            size: wgpu::Extent3d {
                width: down_width,
                height: down_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let intermediate_texture2 = self.device.create_texture(&TextureDescriptor {
            label: Some("Intermediate Texture 2 (Downsampled)"),
            size: wgpu::Extent3d {
                width: down_width,
                height: down_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        // Create blurred texture (at downsampled resolution)
        let blurred_down_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Blurred Downsampled Texture"),
            size: wgpu::Extent3d {
                width: down_width,
                height: down_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let view1 = intermediate_texture1.create_view(&wgpu::TextureViewDescriptor::default());
        let view2 = intermediate_texture2.create_view(&wgpu::TextureViewDescriptor::default());
        let blurred_down_view =
            blurred_down_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Calculate box sizes for adjusted sigma
        let box_sizes = Self::boxes_for_gauss_3pass(adjusted_sigma);
        let box_radii = box_sizes.map(|size| ((size as i32 - 1) / 2).max(0) as u32);

        println!("Adjusted sigma: {:.2}", adjusted_sigma);
        println!("Box radii: {:?}", box_radii);

        // Apply 6 blur passes (3 passes × 2 directions) on downsampled image
        let blur_passes = [
            (&downsampled_view, &view1, box_radii[0], 0u32),
            (&view1, &view2, box_radii[0], 1u32),
            (&view2, &view1, box_radii[1], 0u32),
            (&view1, &view2, box_radii[1], 1u32),
            (&view2, &view1, box_radii[2], 0u32),
            (&view1, &blurred_down_view, box_radii[2], 1u32),
        ];

        for (i, (input_view, output_view, radius, direction)) in blur_passes.iter().enumerate() {
            println!("--- Box Blur Pass {} of 6 (Downsampled) ---", i + 1);

            let params = BoxBlurParams {
                width: down_width,
                height: down_height,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                direction: *direction,
                _padding: [0; 7],
            };

            let params_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Downsampled Box Blur Params {}", i + 1)),
                    contents: bytemuck::cast_slice(&[params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("Downsampled Box Blur Bind Group {}", i + 1)),
                layout: &self.box_blur_bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(*input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(*output_view),
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
                    label: Some(&format!("Downsampled Box Blur Encoder {}", i + 1)),
                });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Downsampled Box Blur Compute Pass {}", i + 1)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.box_blur_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                // Dispatch based on direction
                if *direction == 0 {
                    // Horizontal blur
                    let dispatch_x = (down_width + 255) / 256;
                    let dispatch_y = down_height;
                    compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                } else {
                    // Vertical blur
                    let dispatch_x = down_width;
                    let dispatch_y = (down_height + 255) / 256;
                    compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                }
            }

            self.queue.submit(Some(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            if (i + 1) % 2 == 0 {
                println!("  Completed pass {}/3", (i + 1) / 2);
            }
        }

        println!("Box blur completed on downsampled image");

        // === STEP 3: Upsample ===
        println!("\n=== Step 3: Upsampling ===");

        let final_texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Final Output Texture"),
            size: wgpu::Extent3d {
                width: width as u32,
                height: height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let final_view = final_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let up_params = UpsampleParams {
            src_width: down_width,
            src_height: down_height,
            dst_width: width as u32,
            dst_height: height as u32,
            _padding: [0; 8],
        };

        let up_params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Upsample Params"),
                contents: bytemuck::cast_slice(&[up_params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let up_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Upsample Bind Group"),
            layout: &self.upsample_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&blurred_down_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&final_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: up_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut up_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Upsample Encoder"),
            });

        {
            let mut compute_pass = up_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Upsample Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.upsample_pipeline);
            compute_pass.set_bind_group(0, &up_bind_group, &[]);

            // Dispatch one workgroup per 8x8 block of output image
            let dispatch_x = (width as u32 + 7) / 8;
            let dispatch_y = (height as u32 + 7) / 8;
            println!(
                "Upsample dispatch: {}x{} workgroups",
                dispatch_x, dispatch_y
            );
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        self.queue.submit(Some(up_encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        println!("Upsample completed");

        Ok((final_view, final_texture))
    }
}
