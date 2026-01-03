//! GPU-accelerated Gaussian Blur using wgpu with 3-pass box blur approximation for large sigma

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, ComputePipeline, ComputePipelineDescriptor, Device,
    Instance, Limits, PipelineLayout, Queue, ShaderModuleDescriptor, ShaderSource,
    StorageTextureAccess, TextureDescriptor, TextureFormat, TextureViewDimension,
};

#[cfg(feature = "gpu")]
use bytemuck;

use crate::Pixel;

// Constants for optimized work distribution
const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const DEBUG_BUFFER_SIZE: usize = 1024;
const BOX_BLUR_PASSES: u32 = 3;
const DOWNSAMPLE_FACTOR: u32 = 8; // Downsample 8x for large sigmas

// Shader parameters struct - must match the WGSL struct layout
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    _padding0: u32,
    sigma: f32,
    current_pass: u32,
    blur_direction: u32,
    operation_mode: u32,
    src_width: u32,  // Added: source texture width
    src_height: u32, // Added: source texture height
    dst_width: u32,  // Added: destination texture width
    dst_height: u32, // Added: destination texture height
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
}

// Manually implement Pod and Zeroable for ShaderParameters
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for ShaderParameters {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for ShaderParameters {}

/// GPU Gaussian Blur processor with 3-pass box blur approximation for large sigma
pub struct GpuGaussianBlur {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    queue: Queue,
    #[cfg(feature = "gpu")]
    box_blur_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    pipeline_layout: PipelineLayout,
    #[cfg(feature = "gpu")]
    bind_group_layout: wgpu::BindGroupLayout,
    sigma: f32,
    radius: i32,
    blur_alpha: bool,
}

impl GpuGaussianBlur {
    /// Create a new GPU Gaussian Blur processor
    pub async fn new(sigma: f32, radius: Option<i32>, blur_alpha: bool) -> Result<Self, String> {
        let radius = radius.unwrap_or_else(|| (3.0 * sigma).ceil() as i32);

        #[cfg(not(feature = "gpu"))]
        {
            return Err("GPU feature not enabled. Build with --features gpu".to_string());
        }

        #[cfg(feature = "gpu")]
        {
            // Initialize wgpu
            let instance = Instance::default();

            // Try to find integrated GPU first
            println!("Looking for integrated GPU...");
            let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
            println!("Found {} adapters", adapters.len());

            // First try to find integrated GPU
            let mut found_adapter = None;

            for adapter in adapters.iter() {
                let info = adapter.get_info();
                println!("Found adapter: {} ({:?})", info.name, info.device_type);

                if info.device_type != wgpu::DeviceType::IntegratedGpu {
                    println!("Using Nvidia GPU: {}", info.name);
                    found_adapter = Some(adapter);
                    break;
                }
            }

            let adapter: wgpu::Adapter = if let Some(ref adp) = found_adapter {
                (*adp).clone()
            } else {
                println!("No integrated GPU found, requesting default adapter");
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    })
                    .await
                    .expect("Failed to find a suitable GPU adapter")
            };

            let info = adapter.get_info();
            println!("Selected adapter: {} ({:?})", info.name, info.device_type);

            // Get adapter limits
            let adapter_limits = adapter.limits();
            println!("Adapter limits:");
            println!(
                "  max_texture_dimension_2d: {}",
                adapter_limits.max_texture_dimension_2d
            );
            println!(
                "  max_storage_buffer_binding_size: {}",
                adapter_limits.max_storage_buffer_binding_size
            );
            println!("  max_buffer_size: {}", adapter_limits.max_buffer_size);
            println!(
                "  max_compute_workgroups_per_dimension: {}",
                adapter_limits.max_compute_workgroups_per_dimension
            );

            // Request the maximum limits the adapter supports
            let _required_limits = Limits {
                max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
                max_texture_dimension_3d: adapter_limits.max_texture_dimension_3d,
                max_texture_array_layers: adapter_limits.max_texture_array_layers,
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_uniform_buffer_binding_size: adapter_limits.max_uniform_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                max_storage_buffers_per_shader_stage: adapter_limits
                    .max_storage_buffers_per_shader_stage,
                max_uniform_buffers_per_shader_stage: adapter_limits
                    .max_uniform_buffers_per_shader_stage,
                max_sampled_textures_per_shader_stage: adapter_limits
                    .max_sampled_textures_per_shader_stage,
                max_storage_textures_per_shader_stage: adapter_limits
                    .max_storage_textures_per_shader_stage,
                max_compute_workgroup_storage_size: adapter_limits
                    .max_compute_workgroup_storage_size,
                max_compute_invocations_per_workgroup: adapter_limits
                    .max_compute_invocations_per_workgroup,
                max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
                max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
                max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
                max_compute_workgroups_per_dimension: adapter_limits
                    .max_compute_workgroups_per_dimension,
                ..adapter_limits
            };

            println!("Requesting device with adapter's maximum limits...");

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
                .expect("Failed to create device");

            // Check actual device limits
            let device_limits = device.limits();
            println!("Device granted limits:");
            println!(
                "  max_texture_dimension_2d: {}",
                device_limits.max_texture_dimension_2d
            );
            println!(
                "  max_storage_buffer_binding_size: {}",
                device_limits.max_storage_buffer_binding_size
            );
            println!("  max_buffer_size: {}", device_limits.max_buffer_size);
            println!(
                "  max_compute_workgroups_per_dimension: {}",
                device_limits.max_compute_workgroups_per_dimension
            );

            // Load shader with box blur implementation
            let shader_source = include_str!("shaders/gaussian_blur_box.wgsl");
            println!(
                "Shader source loaded, length: {} bytes",
                shader_source.len()
            );

            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Gaussian Blur Box Approximation Shader"),
                source: ShaderSource::Wgsl(shader_source.into()),
            });

            // Create bind group layout for box blur passes
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Box Blur Bind Group Layout"),
                entries: &[
                    // Input texture (texture_2d<f32> for reading) - binding 0
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
                    // Output texture (storage, write-only, rgba8unorm) - binding 1
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
                    // Parameters buffer - binding 2
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
                    // Debug buffer (storage, read-write) - binding 3
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            // Create pipeline layout
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Blur Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

            // Create compute pipeline for box blur
            let box_blur_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Box Blur Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            Ok(Self {
                device,
                queue,
                box_blur_pipeline,
                pipeline_layout,
                bind_group_layout,
                sigma,
                radius,
                blur_alpha,
            })
        }
    }

    /// Calculate optimal box sizes for 3-pass Gaussian approximation
    /// Uses Central Limit Theorem: 3 box blurs → Gaussian
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

    /// Validate that we can process the given sigma value
    pub fn validate_sigma(&self) -> Result<(), String> {
        println!("=== Sigma Validation ===");
        println!("Sigma: {}, Computed Radius: {}", self.sigma, self.radius);

        // Calculate optimal box sizes for 3-pass approximation
        let box_sizes = Self::boxes_for_gauss_3pass(self.sigma);
        let box_radii = box_sizes.map(|size| ((size as i32 - 1) / 2).max(0) as u32);

        println!("\n=== 3-Pass Box Blur Configuration ===");
        for (i, (&size, &radius)) in box_sizes.iter().zip(box_radii.iter()).enumerate() {
            println!("  Pass {}: box size = {}, radius = {}", i + 1, size, radius);
        }

        // Calculate the actual sigma this approximates
        let approximated_sigma =
            (box_sizes.iter().map(|&w| w as f32 * w as f32).sum::<f32>() / 12.0).sqrt();

        println!("Approximated Gaussian sigma: {:.2}", approximated_sigma);
        println!("Target Gaussian sigma: {:.2}", self.sigma);
        println!(
            "Error: {:.2}%",
            (approximated_sigma - self.sigma).abs() / self.sigma * 100.0
        );

        if self.sigma > 50.0 {
            println!("\nLarge sigma detected ({}).", self.sigma);
            println!("Using 3-pass box blur approximation with 8x downsampling for performance.");
            println!("Quality should match Metal MPS Gaussian blur.");
        }

        Ok(())
    }

    /// Calculate box radius for 3-pass Gaussian approximation
    fn calculate_box_radius(&self) -> [u32; 3] {
        // Get optimal box sizes for 3-pass approximation
        let box_sizes = Self::boxes_for_gauss_3pass(self.sigma);

        // Convert sizes to radii (radius = (size - 1) / 2)
        let mut radii = [0u32; 3];
        for i in 0..3 {
            radii[i] = ((box_sizes[i] as i32 - 1) / 2).max(0) as u32;
        }

        radii
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

    /// Apply blur to an image using GPU with optimized downsampling for large sigma
    /// Returns the blurred image data as bytes (RGBA format) for direct saving
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

            // Validate sigma before proceeding
            if let Err(e) = self.validate_sigma() {
                return Err(format!("Sigma validation failed: {}", e));
            }

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

            // Convert image to flat RGBA bytes
            println!(
                "Input texture first pixel: R:{}, G:{}, B:{}, A:{}",
                image[0][0].r, image[0][0].g, image[0][0].b, image[0][0].a
            );

            let mut rgba_data = Vec::with_capacity(width * height * 4);
            for row in image {
                for pixel in row {
                    rgba_data.push(pixel.r);
                    rgba_data.push(pixel.g);
                    rgba_data.push(pixel.b);
                    rgba_data.push(pixel.a);
                }
            }

            // Create input texture (regular texture, not storage)
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

            // Write image data to texture with proper bytes per row alignment
            let bytes_per_row_unaligned = 4 * width as u32;
            let alignment = 256; // wgpu's COPY_BYTES_PER_ROW_ALIGNMENT
            let bytes_per_row_aligned =
                ((bytes_per_row_unaligned + alignment - 1) / alignment) * alignment;

            println!(
                "Bytes per row alignment: {} -> {} (aligned to {} bytes)",
                bytes_per_row_unaligned, bytes_per_row_aligned, alignment
            );

            // Upload image data
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

            // Determine if we should downsample (for large sigma)
            let should_downsample = self.sigma > 50.0;
            let downscale_factor = if should_downsample {
                DOWNSAMPLE_FACTOR
            } else {
                1
            };
            let down_width = (width as u32 + downscale_factor - 1) / downscale_factor;
            let down_height = (height as u32 + downscale_factor - 1) / downscale_factor;

            if should_downsample {
                println!("\n=== Large sigma detected: Using 8x downsampling for performance ===");
                println!("Downsampled size: {}x{} pixels", down_width, down_height);
                println!(
                    "Original sigma: {:.2}, Adjusted sigma: {:.2}",
                    self.sigma,
                    self.sigma / downscale_factor as f32
                );
            }

            // === STEP 1: Downsample if needed ===
            let (blur_input_view, blur_width, blur_height, adjusted_sigma) = if should_downsample {
                println!("\n=== Step 1: Downsampling 8x ===");

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
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[TextureFormat::Rgba8Unorm],
                });

                let downsampled_view =
                    downsampled_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let down_params = ShaderParameters {
                    width: down_width,   // Destination width (downsampled)
                    height: down_height, // Destination height (downsampled)
                    radius: 0,
                    blur_alpha: self.blur_alpha as u32,
                    _padding0: 0,
                    sigma: 0.0,
                    current_pass: 0,
                    blur_direction: 0,
                    operation_mode: 1,         // Downsample mode
                    src_width: width as u32,   // Original width
                    src_height: height as u32, // Original height
                    dst_width: down_width,     // Same as width for clarity
                    dst_height: down_height,   // Same as height for clarity
                    _padding2: 0.0,
                    _padding3: 0.0,
                    _padding4: 0.0,
                };

                let params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Downsample Params"),
                            contents: bytemuck::cast_slice(&[down_params]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                // Create dummy debug buffer
                let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Debug Buffer"),
                    size: (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Downsample Bind Group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&input_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&downsampled_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: debug_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Downsample Encoder"),
                        });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Downsample Compute Pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.box_blur_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    let dispatch_width = (down_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                    let dispatch_height = (down_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });

                println!("Downsample completed");
                (
                    downsampled_view,
                    down_width,
                    down_height,
                    self.sigma / downscale_factor as f32,
                )
            } else {
                (input_view, width as u32, height as u32, self.sigma)
            };

            // === STEP 2: Calculate box radii for blur ===
            println!("\n=== Step 2: Calculating box blur parameters ===");

            let box_sizes = Self::boxes_for_gauss_3pass(adjusted_sigma);
            let box_radii = box_sizes.map(|size| ((size as i32 - 1) / 2).max(0) as u32);

            println!("Adjusted sigma: {:.2}", adjusted_sigma);
            println!("Box radii: {:?}", box_radii);

            // === STEP 3: Apply 3-pass box blur ===
            println!("\n=== Step 3: Applying 3-pass box blur ===");

            // Create intermediate textures for blur passes
            let intermediate_usage = wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC;

            let intermediate_texture1 = self.device.create_texture(&TextureDescriptor {
                label: Some("Intermediate Texture 1"),
                size: wgpu::Extent3d {
                    width: blur_width,
                    height: blur_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_texture2 = self.device.create_texture(&TextureDescriptor {
                label: Some("Intermediate Texture 2"),
                size: wgpu::Extent3d {
                    width: blur_width,
                    height: blur_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            // Create blurred texture (at downsampled or full resolution)
            let blurred_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Blurred Texture"),
                size: wgpu::Extent3d {
                    width: blur_width,
                    height: blur_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let view1 = intermediate_texture1.create_view(&wgpu::TextureViewDescriptor::default());
            let view2 = intermediate_texture2.create_view(&wgpu::TextureViewDescriptor::default());
            let blurred_view = blurred_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create debug buffer for blur passes
            let debug_buffer_size_bytes =
                (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Blur Debug Buffer"),
                size: debug_buffer_size_bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Apply 6 blur passes (3 passes × 2 directions)
            let blur_passes = [
                (&blur_input_view, &view1, box_radii[0], 0u32, 0u32),
                (&view1, &view2, box_radii[0], 0u32, 1u32),
                (&view2, &view1, box_radii[1], 1u32, 0u32),
                (&view1, &view2, box_radii[1], 1u32, 1u32),
                (&view2, &view1, box_radii[2], 2u32, 0u32),
                (&view1, &blurred_view, box_radii[2], 2u32, 1u32),
            ];

            for (i, (input_view, output_view, radius, pass_num, direction)) in
                blur_passes.iter().enumerate()
            {
                let is_horizontal = *direction == 0;
                println!(
                    "\n--- Box Blur Pass {} of 6 ({} {}) ---",
                    i + 1,
                    if *pass_num == 0 {
                        "Pass 1"
                    } else if *pass_num == 1 {
                        "Pass 2"
                    } else {
                        "Pass 3"
                    },
                    if is_horizontal {
                        "Horizontal"
                    } else {
                        "Vertical"
                    }
                );

                // Update parameters for this pass
                let params = ShaderParameters {
                    width: blur_width,
                    height: blur_height,
                    radius: *radius,
                    blur_alpha: self.blur_alpha as u32,
                    _padding0: 0,
                    sigma: adjusted_sigma,
                    current_pass: *pass_num,
                    blur_direction: *direction,
                    operation_mode: 0,       // Blur mode
                    src_width: blur_width,   // Same as width
                    src_height: blur_height, // Same as height
                    dst_width: blur_width,   // Same as width
                    dst_height: blur_height, // Same as height
                    _padding2: 0.0,
                    _padding3: 0.0,
                    _padding4: 0.0,
                };

                let params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("Blur Params {}", i + 1)),
                            contents: bytemuck::cast_slice(&[params]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                // Create bind group
                let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("Blur Bind Group {}", i + 1)),
                    layout: &self.bind_group_layout,
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
                        BindGroupEntry {
                            binding: 3,
                            resource: debug_buffer.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(&format!("Blur Encoder Pass {}", i + 1)),
                        });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("Blur Compute Pass {}", i + 1)),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.box_blur_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    // Dispatch workgroups
                    let dispatch_width = (blur_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                    let dispatch_height = (blur_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;

                    println!(
                        "  Dispatch: {}x{} workgroups",
                        dispatch_width, dispatch_height
                    );
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                // Submit and wait
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });

                if (i + 1) % 2 == 0 {
                    println!("  Completed pass {}/3", (i + 1) / 2);
                }
            }

            println!("Blur completed");

            // === STEP 4: Upsample if we downsampled ===
            // Store both the view and texture for the final copy
            let (final_view, final_texture) = if should_downsample {
                println!("\n=== Step 4: Upsampling back to original size ===");

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

                let up_params = ShaderParameters {
                    width: blur_width,   // Source width (downsampled)
                    height: blur_height, // Source height (downsampled)
                    radius: 0,
                    blur_alpha: self.blur_alpha as u32,
                    _padding0: 0,
                    sigma: 0.0,
                    current_pass: 0,
                    blur_direction: 0,
                    operation_mode: 2,         // Upsample mode
                    src_width: blur_width,     // Source width (downsampled)
                    src_height: blur_height,   // Source height (downsampled)
                    dst_width: width as u32,   // Destination width (full size)
                    dst_height: height as u32, // Destination height (full size)
                    _padding2: 0.0,
                    _padding3: 0.0,
                    _padding4: 0.0,
                };

                let params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Upsample Params"),
                            contents: bytemuck::cast_slice(&[up_params]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                let debug_buffer2 = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Upsample Debug Buffer"),
                    size: debug_buffer_size_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("Upsample Bind Group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&blurred_view),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&final_view),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: params_buffer.as_entire_binding(),
                        },
                        BindGroupEntry {
                            binding: 3,
                            resource: debug_buffer2.as_entire_binding(),
                        },
                    ],
                });

                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Upsample Encoder"),
                        });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Upsample Compute Pass"),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.box_blur_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    let dispatch_width = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                    let dispatch_height = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                });

                println!("Upsample completed");
                (Some(final_view), Some(final_texture))
            } else {
                // If no downsampling, use the blurred texture directly
                (None, None)
            };

            // === STEP 5: Copy result to buffer ===
            println!("\n=== Step 5: Copying results to CPU ===");

            // Create output buffer
            let bytes_per_pixel = 4u32;
            let bytes_per_row_unaligned = bytes_per_pixel * width as u32;
            let bytes_per_row_aligned =
                ((bytes_per_row_unaligned + alignment - 1) / alignment) * alignment;
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

            // Determine which texture to copy from based on whether we downsampled
            let (copy_texture, copy_width, copy_height) = if should_downsample {
                // When downsampled was used, copy from the upsampled texture
                println!("Copying from upsampled texture (full resolution)");
                (
                    final_texture
                        .as_ref()
                        .expect("Final texture should exist when downsampling"),
                    width as u32,
                    height as u32,
                )
            } else {
                // No downsampling, copy from the blurred texture
                println!("Copying from blurred texture (original resolution)");
                (&blurred_texture, width as u32, height as u32)
            };

            // Copy texture to buffer
            final_encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: copy_texture,
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
                    width: copy_width,
                    height: copy_height,
                    depth_or_array_layers: 1,
                },
            );

            // Submit final copy
            self.queue.submit(Some(final_encoder.finish()));
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // Read back image results
            let buffer_slice = final_output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            self.device.poll(wgpu::PollType::Wait {
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

            // === DEBUG: Verify blur quality ===
            if result_bytes.len() >= 100 {
                println!("\n=== Blur Quality Verification ===");

                println!("First pixel comparison:");
                println!(
                    "  Input:  R={}, G={}, B={}, A={}",
                    rgba_data[0], rgba_data[1], rgba_data[2], rgba_data[3]
                );
                println!(
                    "  Output: R={}, G={}, B={}, A={}",
                    result_bytes[0], result_bytes[1], result_bytes[2], result_bytes[3]
                );

                let mut total_diff = 0i32;
                for i in (0..100.min(rgba_data.len()).min(result_bytes.len())).step_by(4) {
                    for j in 0..3 {
                        total_diff += (rgba_data[i + j] as i32 - result_bytes[i + j] as i32).abs();
                    }
                }

                println!(
                    "Average pixel difference: {:.2}",
                    total_diff as f32 / (100.0 / 4.0)
                );

                if self.sigma > 100.0 && total_diff < 1000 {
                    println!("WARNING: Blur may be too weak for sigma={}", self.sigma);
                }
            }

            // Cleanup
            drop(data);
            final_output_buffer.unmap();

            println!("Total GPU time: {:?}", total_start.elapsed());

            if should_downsample {
                println!("Performance: ~64x faster than full-resolution blur");
                println!("Quality: Should match Metal MPS Gaussian blur");
            }

            Ok((result_bytes, width, height))
        }
    }
}
