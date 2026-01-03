//! GPU-accelerated Gaussian Blur using wgpu with box blur approximation for large sigma

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
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
const TILE_SIZE_X: u32 = 4; // Each thread processes 4 pixels in X
const TILE_SIZE_Y: u32 = 4; // Each thread processes 4 pixels in Y
const DEBUG_BUFFER_SIZE: usize = 1024; // Number of f32 values in debug buffer
const BOX_BLUR_PASSES: u32 = 3; // Number of box blur passes to approximate Gaussian

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
    current_pass: u32, // Which pass we're on (0, 1, or 2 for box blur approximation)
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    _padding10: f32,
    _padding11: f32,
    _padding12: f32,
    _padding13: f32,
    _padding14: f32,
    _padding15: f32,
}

// Manually implement Pod and Zeroable for ShaderParameters
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for ShaderParameters {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for ShaderParameters {}

/// GPU Gaussian Blur processor with box blur approximation for large sigma
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

                if info.device_type == wgpu::DeviceType::IntegratedGpu {
                    println!("Using integrated GPU: {}", info.name);
                    found_adapter = Some(adapter);
                    break;
                }
            }

            /*
            // If no integrated GPU found, use the first available adapter
            if let Some(adapter) = found_adapter {
                adapter.clone()
            } else {
                println!("No integrated GPU found, requesting default adapter");
                instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    })
                    .await
                    .ok_or("Failed to find a suitable GPU adapter")?
            }
            */

            let adapter: wgpu::Adapter = if let Some(ref adp) = found_adapter {
                // Dereference the &Adapter to clone the underlying handle
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
            let required_limits = Limits {
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
                    experimental_features: Default::default(), // Required in v22+
                    trace: wgpu::Trace::Off,                   // Required in v22+
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
                immediate_size: 0, // Replaces push_constant_ranges
            });

            // Create compute pipeline for box blur
            let box_blur_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Box Blur Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("box_blur_pass"), // Now requires Some()
                    compilation_options: wgpu::PipelineCompilationOptions::default(), // Required in v22+
                    cache: None, // Required field in 2026 versions
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

    /// Validate that we can process the given sigma value
    pub fn validate_sigma(&self) -> Result<(), String> {
        println!("=== Sigma Validation ===");
        println!("Sigma: {}, Computed Radius: {}", self.sigma, self.radius);

        // For very large sigma, use box blur approximation
        if self.sigma > 50.0 {
            let box_radius = self.calculate_box_radius();
            println!("Large sigma detected ({}). Using box blur approximation with radius={} for {} passes", 
                self.sigma, box_radius, BOX_BLUR_PASSES);
            println!("This will be much faster and avoid GPU timeouts.");
        }

        Ok(())
    }

    /// Calculate box radius for box blur approximation
    fn calculate_box_radius(&self) -> u32 {
        // For 3 box blur passes, the relationship to approximate Gaussian is:
        // box_radius ≈ sigma * 0.8 / sqrt(3)
        ((self.sigma * 0.8) / (BOX_BLUR_PASSES as f32).sqrt()).ceil() as u32
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

    /// Apply blur to an image using GPU with box blur approximation for large sigma
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

            // If the data needs padding for alignment, create a padded copy
            if bytes_per_row_aligned != bytes_per_row_unaligned {
                println!("Creating aligned data buffer...");
                let aligned_row_size = bytes_per_row_aligned as usize;
                let unaligned_row_size = bytes_per_row_unaligned as usize;

                let mut aligned_data = Vec::with_capacity(aligned_row_size * height);

                for row in 0..height {
                    let row_start = row * unaligned_row_size;
                    let row_end = row_start + unaligned_row_size;
                    aligned_data.extend_from_slice(&rgba_data[row_start..row_end]);

                    // Add padding
                    let padding = aligned_row_size - unaligned_row_size;
                    aligned_data.extend(std::iter::repeat(0u8).take(padding));
                }

                self.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &input_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &aligned_data,
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
            } else {
                // No alignment needed
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
            }

            let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Define usage for intermediate textures (need both read and write capabilities)
            let intermediate_usage = wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC;

            // Create intermediate texture 1 (Rgba8Unorm storage)
            let intermediate_texture1 = self.device.create_texture(&TextureDescriptor {
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
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_view1 =
                intermediate_texture1.create_view(&wgpu::TextureViewDescriptor::default());

            // Create intermediate texture 2 (Rgba8Unorm storage)
            let intermediate_texture2 = self.device.create_texture(&TextureDescriptor {
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
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_view2 =
                intermediate_texture2.create_view(&wgpu::TextureViewDescriptor::default());

            // Create output texture (write-only storage, Rgba8Unorm)
            let output_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Output Texture"),
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

            let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create debug buffer
            let debug_buffer_size_bytes =
                (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Buffer"),
                size: debug_buffer_size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Initialize debug buffer with zeros
            let debug_init_data = vec![0u8; debug_buffer_size_bytes as usize];
            self.queue.write_buffer(&debug_buffer, 0, &debug_init_data);

            // === STEP 1: Test direct write to output texture ===
            println!("\n=== Step 1: Testing Direct Texture Write ===");

            let test_shader_source = r#"
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1, 1, 1)
fn test_write() {
    // Write test colors to first few pixels
    textureStore(output_texture, vec2<u32>(0u, 0u), vec4<f32>(1.0, 0.0, 0.0, 1.0)); // Red
    textureStore(output_texture, vec2<u32>(1u, 0u), vec4<f32>(0.0, 1.0, 0.0, 1.0)); // Green
    textureStore(output_texture, vec2<u32>(2u, 0u), vec4<f32>(0.0, 0.0, 1.0, 1.0)); // Blue
    textureStore(output_texture, vec2<u32>(3u, 0u), vec4<f32>(1.0, 1.0, 1.0, 1.0)); // White
}
"#;

            let test_shader = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Test Write Shader"),
                source: ShaderSource::Wgsl(test_shader_source.into()),
            });

            let test_pipeline = self
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Test Write Pipeline"),
                    layout: Some(&self.pipeline_layout),
                    module: &test_shader,
                    entry_point: Some("test_write"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            // Create a temporary bind group for test
            let test_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Test Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Dummy Params"),
                                size: std::mem::size_of::<ShaderParameters>()
                                    as wgpu::BufferAddress,
                                usage: wgpu::BufferUsages::UNIFORM,
                                mapped_at_creation: false,
                            }),
                            offset: 0,
                            size: None,
                        }),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: debug_buffer.as_entire_binding(),
                    },
                ],
            });

            // Run test
            let mut test_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Test Encoder"),
                    });

            {
                let mut compute_pass =
                    test_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Test Compute Pass"),
                        timestamp_writes: None,
                    });
                compute_pass.set_pipeline(&test_pipeline);
                compute_pass.set_bind_group(0, &test_bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }

            // For Rgba8Unorm, each pixel is 4 bytes
            let bytes_per_pixel = 4u32;
            let bytes_per_row_unaligned = bytes_per_pixel * width as u32;
            let bytes_per_row_aligned =
                ((bytes_per_row_unaligned + alignment - 1) / alignment) * alignment;

            // Calculate output buffer size with aligned rows for Rgba8Unorm
            let output_buffer_size =
                (bytes_per_row_aligned as u64 * height as u64) as wgpu::BufferAddress;

            let test_output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Test Output Buffer"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy texture to buffer
            test_encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &test_output_buffer,
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

            // Submit and wait
            self.queue.submit(Some(test_encoder.finish()));
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }); // CRITICAL: Wait for GPU

            // Read back test results
            let buffer_slice = test_output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }); // CRITICAL: Wait for mapping

            let test_result = match receiver.recv() {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    // Check first few pixels
                    let mut test_pixels = Vec::new();
                    for i in 0..4 {
                        if i * 4 + 3 < data.len() {
                            test_pixels.push((
                                data[i * 4],
                                data[i * 4 + 1],
                                data[i * 4 + 2],
                                data[i * 4 + 3],
                            ));
                        }
                    }

                    println!("Test pixels (should be red, green, blue, white):");
                    for (i, (r, g, b, a)) in test_pixels.iter().enumerate() {
                        println!("  Pixel {}: R={}, G={}, B={}, A={}", i, r, g, b, a);
                    }

                    // Check if we got expected colors (within tolerance)
                    let expected = vec![
                        (255, 0, 0, 255),     // Red
                        (0, 255, 0, 255),     // Green
                        (0, 0, 255, 255),     // Blue
                        (255, 255, 255, 255), // White
                    ];

                    let mut passed = true;
                    for (
                        i,
                        ((actual_r, actual_g, actual_b, actual_a), (exp_r, exp_g, exp_b, exp_a)),
                    ) in test_pixels.iter().zip(expected.iter()).enumerate()
                    {
                        let close = |a: u8, b: u8| (a as i32 - b as i32).abs() < 10;
                        if !close(*actual_r, *exp_r)
                            || !close(*actual_g, *exp_g)
                            || !close(*actual_b, *exp_b)
                            || !close(*actual_a, *exp_a)
                        {
                            println!("✗ Pixel {} doesn't match expected color", i);
                            passed = false;
                        }
                    }

                    if passed {
                        println!("✓ Texture write test passed");
                    } else {
                        println!("✗ Texture write test failed");
                    }

                    passed
                }
                Ok(Err(e)) => {
                    println!("Test mapping failed: {}", e);
                    false
                }
                Err(e) => {
                    println!("Test channel error: {}", e);
                    false
                }
            };

            drop(test_output_buffer);

            if !test_result {
                return Err("Texture write test failed - GPU pipeline not working".to_string());
            }

            // Clear output texture for actual blur
            println!("Clearing output texture for blur passes...");
            let clear_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Clear Encoder"),
                    });
            self.queue.submit(Some(clear_encoder.finish()));
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });

            // === STEP 2: Multiple Box Blur Passes ===
            println!("\n=== Step 2: Multiple Box Blur Passes (approximates Gaussian) ===");

            // Calculate box radius for approximation
            let box_radius = self.calculate_box_radius();
            println!(
                "Using box blur approximation: radius={} for {} passes",
                box_radius, BOX_BLUR_PASSES
            );
            println!("This approximates Gaussian with sigma={}", self.sigma);

            // Define clear pass structure: (input_view, output_view)
            let passes = [
                (&input_view, &intermediate_view1), // Pass 0: Rgba8Unorm -> Rgba8Unorm
                (&intermediate_view1, &intermediate_view2), // Pass 1: Rgba8Unorm -> Rgba8Unorm
                (&intermediate_view2, &output_view), // Pass 2: Rgba8Unorm -> Rgba8Unorm
            ];

            for (pass_index, (input_view, output_view)) in passes.iter().enumerate() {
                println!(
                    "\n--- Box Blur Pass {} of {} ---",
                    pass_index + 1,
                    BOX_BLUR_PASSES
                );

                let params = ShaderParameters {
                    width: width as u32,
                    height: height as u32,
                    radius: box_radius,
                    blur_alpha: self.blur_alpha as u32,
                    _padding0: 0,
                    sigma: self.sigma,
                    current_pass: pass_index as u32,
                    _padding1: 0.0,
                    _padding2: 0.0,
                    _padding3: 0.0,
                    _padding4: 0.0,
                    _padding5: 0.0,
                    _padding6: 0.0,
                    _padding7: 0.0,
                    _padding8: 0.0,
                    _padding9: 0.0,
                    _padding10: 0.0,
                    _padding11: 0.0,
                    _padding12: 0.0,
                    _padding13: 0.0,
                    _padding14: 0.0,
                    _padding15: 0.0,
                };

                let params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(&format!("Parameters Buffer Pass {}", pass_index)),
                            contents: bytemuck::cast_slice(&[params]),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });

                // Create bind group for this pass
                let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                    label: Some(&format!("Box Blur Bind Group Pass {}", pass_index)),
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
                            label: Some(&format!("Box Blur Encoder Pass {}", pass_index)),
                        });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(&format!("Box Blur Compute Pass {}", pass_index)),
                            timestamp_writes: None,
                        });

                    compute_pass.set_pipeline(&self.box_blur_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);

                    // Optimized dispatch for tiled processing
                    let effective_width = (width as u32 + TILE_SIZE_X - 1) / TILE_SIZE_X;
                    let effective_height = (height as u32 + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
                    let dispatch_width =
                        (effective_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                    let dispatch_height =
                        (effective_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;

                    println!(
                        "Dispatch: {}x{} workgroups",
                        dispatch_width, dispatch_height
                    );
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }

                // Submit this pass and wait
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                }); // Wait for GPU to finish

                println!("Pass {} completed", pass_index + 1);
            }

            // === READ DEBUG BUFFER ===
            println!("\n=== Step 3: Debug Analysis ===");
            let debug_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Staging Buffer"),
                size: debug_buffer_size_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy debug buffer to staging buffer
            let mut debug_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Debug Readback Encoder"),
                    });
            debug_encoder.copy_buffer_to_buffer(
                &debug_buffer,
                0,
                &debug_staging_buffer,
                0,
                debug_buffer_size_bytes,
            );
            self.queue.submit(Some(debug_encoder.finish()));

            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }); // Wait for copy

            let debug_slice = debug_staging_buffer.slice(..);
            let (debug_sender, debug_receiver) = std::sync::mpsc::channel();
            debug_slice.map_async(wgpu::MapMode::Read, move |result| {
                debug_sender.send(result).unwrap();
            });

            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }); // Wait for mapping

            debug_receiver
                .recv()
                .map_err(|e| format!("Failed to receive debug buffer: {}", e))?
                .map_err(|e| format!("Failed to map debug buffer: {}", e))?;

            let debug_data = debug_slice.get_mapped_range();
            let debug_bytes = debug_data.to_vec();

            // Parse debug buffer as f32 values
            let mut debug_values = Vec::new();
            for i in 0..(debug_bytes.len() / 4).min(100) {
                let offset = i * 4;
                let value_bytes: [u8; 4] = debug_bytes[offset..offset + 4].try_into().unwrap();
                let value = f32::from_le_bytes(value_bytes);
                debug_values.push(value);
            }

            println!("\n=== Debug Analysis ===");
            println!("Marker (1000 + pass): {}", debug_values[0]);
            println!("Width: {}", debug_values[1]);
            println!("Height: {}", debug_values[2]);
            println!("Radius: {}", debug_values[3]);
            println!("Blur Alpha Flag: {}", debug_values[4]);

            // Show debug pixels for each pass
            for pass in 0..BOX_BLUR_PASSES {
                println!("\n=== Pass {} Debug Pixels ===", pass);
                for i in 0..4 {
                    let offset = (5 + (pass as usize) * 20 + i * 4) as usize;
                    if offset + 3 < debug_values.len() {
                        println!(
                            "  Pixel {}: R={:.1}, G={:.1}, B={:.1}, A={:.1}",
                            i,
                            debug_values[offset],
                            debug_values[offset + 1],
                            debug_values[offset + 2],
                            debug_values[offset + 3]
                        );
                    }
                }
            }

            drop(debug_data);
            debug_staging_buffer.unmap();

            // === COPY OUTPUT TO BUFFER ===
            println!("\n=== Step 4: Copying Results (direct to RGBA8 buffer) ===");

            println!(
                "Rgba8Unorm bytes per row: {} -> {} (aligned)",
                bytes_per_row_unaligned, bytes_per_row_aligned
            );
            println!(
                "Output buffer size: {} bytes ({} aligned rows × {} height)",
                output_buffer_size, bytes_per_row_aligned, height
            );

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

            // Copy texture to buffer with aligned bytes_per_row for Rgba8Unorm
            final_encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &output_texture,
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
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }); // Wait for GPU

            // Read back image results
            let buffer_slice = final_output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            }); // Wait for mapping

            receiver
                .recv()
                .map_err(|e| format!("Failed to receive buffer: {}", e))?
                .map_err(|e| format!("Failed to map buffer: {}", e))?;

            let data = buffer_slice.get_mapped_range();

            // Extract data, handling row alignment padding
            let mut result_bytes = Vec::with_capacity(width * height * 4);
            let aligned_row_size_bytes = bytes_per_row_aligned as usize;

            println!(
                "Extracting data: width={}, height={}, aligned_row_size={}, total_bytes={}",
                width,
                height,
                aligned_row_size_bytes,
                data.len()
            );

            // Extract each row, skipping the padding at the end
            for row in 0..height {
                let row_start = row * aligned_row_size_bytes;
                let row_end = row_start + (width * 4); // 4 bytes per pixel

                if row_end <= data.len() {
                    result_bytes.extend_from_slice(&data[row_start..row_end]);
                } else {
                    // If row is incomplete, pad with zeros
                    let available = data.len().saturating_sub(row_start);
                    if available > 0 {
                        result_bytes.extend_from_slice(&data[row_start..row_start + available]);
                    }
                    // Pad remaining with zeros
                    let needed = width * 4 - available.min(width * 4);
                    result_bytes.extend(std::iter::repeat(0u8).take(needed));
                }

                // Debug: Print first few pixels of first few rows
                if row < 3 && width > 0 {
                    let pixel_idx = row * width * 4;
                    if pixel_idx + 3 < result_bytes.len() {
                        println!(
                            "Row {} first pixel: R={}, G={}, B={}, A={}",
                            row,
                            result_bytes[pixel_idx],
                            result_bytes[pixel_idx + 1],
                            result_bytes[pixel_idx + 2],
                            result_bytes[pixel_idx + 3]
                        );
                    }
                }
            }

            // Verify we got the right amount of data
            let expected_bytes = width * height * 4;
            println!(
                "Extracted {} bytes (expected {})",
                result_bytes.len(),
                expected_bytes
            );

            // Check if we have a full image
            if result_bytes.len() != expected_bytes {
                println!(
                    "ERROR: Extracted {} bytes but expected {}",
                    result_bytes.len(),
                    expected_bytes
                );

                // If we're missing data, pad with zeros
                if result_bytes.len() < expected_bytes {
                    let needed = expected_bytes - result_bytes.len();
                    println!("Padding with {} zeros", needed);
                    result_bytes.extend(std::iter::repeat(0u8).take(needed));
                } else {
                    // If we have too much data, truncate
                    println!("Truncating to expected size");
                    result_bytes.truncate(expected_bytes);
                }
            }

            // Cleanup
            drop(data);
            final_output_buffer.unmap();

            println!("Total GPU time: {:?}", total_start.elapsed());

            // Check if image looks reasonable
            if result_bytes.len() >= 4 {
                println!(
                    "First output pixel: R:{}, G:{}, B:{}, A:{}",
                    result_bytes[0], result_bytes[1], result_bytes[2], result_bytes[3]
                );

                // Check if we have non-zero data
                let mut has_non_zero = false;
                for &value in result_bytes.iter().take(100) {
                    if value != 0 {
                        has_non_zero = true;
                        break;
                    }
                }

                if !has_non_zero {
                    println!("WARNING: First 100 bytes are all zero!");
                }
            }

            Ok((result_bytes, width, height))
        }
    }
}
