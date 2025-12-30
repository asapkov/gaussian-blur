//! GPU-accelerated Gaussian Blur using wgpu with box blur approximation for large sigma

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Queue,
    ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, Limits,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    StorageTextureAccess, TextureDescriptor, TextureViewDimension, TextureFormat,
    PipelineLayout,
};

#[cfg(feature = "gpu")]
use bytemuck;
#[cfg(feature = "gpu")]
use half;

use crate::Pixel;

// Constants for optimized work distribution
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
const TILE_SIZE_X: u32 = 4;  // Each thread processes 4 pixels in X
const TILE_SIZE_Y: u32 = 4;  // Each thread processes 4 pixels in Y
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
    current_pass: u32,  // Which pass we're on (0, 1, or 2 for box blur approximation)
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
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
                .ok_or("Failed to find a suitable GPU adapter")?;

            // Get adapter limits
            let adapter_limits = adapter.limits();
            println!("Adapter limits:");
            println!("  max_texture_dimension_2d: {}", adapter_limits.max_texture_dimension_2d);
            println!("  max_storage_buffer_binding_size: {}", adapter_limits.max_storage_buffer_binding_size);
            println!("  max_buffer_size: {}", adapter_limits.max_buffer_size);
            println!("  max_compute_workgroups_per_dimension: {}", adapter_limits.max_compute_workgroups_per_dimension);

            // Request the maximum limits the adapter supports
            let required_limits = Limits {
                max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
                max_texture_dimension_3d: adapter_limits.max_texture_dimension_3d,
                max_texture_array_layers: adapter_limits.max_texture_array_layers,
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_uniform_buffer_binding_size: adapter_limits.max_uniform_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                max_storage_buffers_per_shader_stage: adapter_limits.max_storage_buffers_per_shader_stage,
                max_uniform_buffers_per_shader_stage: adapter_limits.max_uniform_buffers_per_shader_stage,
                max_sampled_textures_per_shader_stage: adapter_limits.max_sampled_textures_per_shader_stage,
                max_storage_textures_per_shader_stage: adapter_limits.max_storage_textures_per_shader_stage,
                max_compute_workgroup_storage_size: adapter_limits.max_compute_workgroup_storage_size,
                max_compute_invocations_per_workgroup: adapter_limits.max_compute_invocations_per_workgroup,
                max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
                max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
                max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
                max_compute_workgroups_per_dimension: adapter_limits.max_compute_workgroups_per_dimension,
                ..adapter_limits
            };

            println!("Requesting device with adapter's maximum limits...");

            let (device, queue) = adapter
                .request_device(
                    &DeviceDescriptor {
                        label: Some("Gaussian Blur Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits,
                    },
                    None,
                )
                .await
                .map_err(|e| format!("Failed to request device: {}", e))?;

            // Check actual device limits
            let device_limits = device.limits();
            println!("Device granted limits:");
            println!("  max_texture_dimension_2d: {}", device_limits.max_texture_dimension_2d);
            println!("  max_storage_buffer_binding_size: {}", device_limits.max_storage_buffer_binding_size);
            println!("  max_buffer_size: {}", device_limits.max_buffer_size);
            println!("  max_compute_workgroups_per_dimension: {}", device_limits.max_compute_workgroups_per_dimension);

            // Load shader with box blur implementation
            let shader_source = include_str!("shaders/gaussian_blur_box.wgsl");
            println!("Shader source loaded, length: {} bytes", shader_source.len());

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
                    // Output texture (storage, write-only, rgba16float) - binding 1
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba16Float,
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
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Box Blur Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            // Create compute pipeline for box blur
            let box_blur_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Box Blur Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "box_blur_pass",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
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

            // Check if we have enough memory
            let required_buffer_size = (width * height * 4) as u64;
            if required_buffer_size > device_limits.max_buffer_size as u64 {
                return Err(format!(
                    "Image requires {} bytes but GPU buffer limit is {} bytes",
                    required_buffer_size, device_limits.max_buffer_size
                ));
            }

            // Convert image to flat RGBA bytes
            println!("Input texture first pixel: R:{}, G:{}, B:{}, A:{}", 
                image[0][0].r, image[0][0].g, image[0][0].b, image[0][0].a);

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
            let bytes_per_row_aligned = ((bytes_per_row_unaligned + alignment - 1) / alignment) * alignment;
            
            println!("Bytes per row alignment: {} -> {} (aligned to {} bytes)", 
                bytes_per_row_unaligned, bytes_per_row_aligned, alignment);

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
                    wgpu::ImageCopyTexture {
                        texture: &input_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &aligned_data,
                    wgpu::ImageDataLayout {
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
                    wgpu::ImageCopyTexture {
                        texture: &input_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rgba_data,
                    wgpu::ImageDataLayout {
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
            let intermediate_usage = wgpu::TextureUsages::STORAGE_BINDING | 
                                     wgpu::TextureUsages::TEXTURE_BINDING |
                                     wgpu::TextureUsages::COPY_SRC;

            // Create intermediate texture 1 (Rgba16Float storage)
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
                format: TextureFormat::Rgba16Float,
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba16Float],
            });

            let intermediate_view1 = intermediate_texture1.create_view(&wgpu::TextureViewDescriptor::default());

            // Create intermediate texture 2 (Rgba16Float storage)
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
                format: TextureFormat::Rgba16Float,
                usage: intermediate_usage,
                view_formats: &[TextureFormat::Rgba16Float],
            });

            let intermediate_view2 = intermediate_texture2.create_view(&wgpu::TextureViewDescriptor::default());

            // Create output texture (write-only storage, Rgba16Float)
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
                format: TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | 
                       wgpu::TextureUsages::COPY_SRC,
                view_formats: &[TextureFormat::Rgba16Float],
            });

            let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create debug buffer
            let debug_buffer_size_bytes = (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Buffer"),
                size: debug_buffer_size_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Initialize debug buffer with zeros
            let debug_init_data = vec![0u8; debug_buffer_size_bytes as usize];
            self.queue.write_buffer(&debug_buffer, 0, &debug_init_data);

            // === STEP 1: Minimal test shader ===
            println!("\n=== Step 1: Minimal Test ===");
            let minimal_shader_source = r#"
@group(0) @binding(3) var<storage, read_write> debug_buffer: array<f32, 1024>;

@compute @workgroup_size(1, 1, 1)
fn minimal_test() {
    debug_buffer[0] = 123.0;
    debug_buffer[1] = 456.0;
    debug_buffer[2] = 789.0;
}
"#;

            let minimal_shader = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Minimal Test Shader"),
                source: ShaderSource::Wgsl(minimal_shader_source.into()),
            });

            let minimal_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Minimal Test Pipeline"),
                layout: Some(&self.pipeline_layout),
                module: &minimal_shader,
                entry_point: "minimal_test",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            // Create a temporary bind group for minimal test
            let minimal_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Minimal Test Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&intermediate_view1),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.device.create_buffer(&wgpu::BufferDescriptor {
                                label: Some("Dummy Params"),
                                size: std::mem::size_of::<ShaderParameters>() as wgpu::BufferAddress,
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

            // Run minimal test
            let mut minimal_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Minimal Test Encoder"),
            });

            {
                let mut compute_pass = minimal_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Minimal Test Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&minimal_pipeline);
                compute_pass.set_bind_group(0, &minimal_bind_group, &[]);
                compute_pass.dispatch_workgroups(1, 1, 1);
            }

            let minimal_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Minimal Test Staging Buffer"),
                size: debug_buffer_size_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            minimal_encoder.copy_buffer_to_buffer(&debug_buffer, 0, &minimal_staging_buffer, 0, debug_buffer_size_bytes);
            self.queue.submit(Some(minimal_encoder.finish()));

            // Read minimal test results
            self.device.poll(wgpu::Maintain::Wait);
            let minimal_slice = minimal_staging_buffer.slice(..);
            let (minimal_sender, minimal_receiver) = std::sync::mpsc::channel();
            minimal_slice.map_async(wgpu::MapMode::Read, move |result| {
                minimal_sender.send(result).unwrap();
            });

            self.device.poll(wgpu::Maintain::Wait);

            let minimal_result = match minimal_receiver.recv() {
                Ok(Ok(_)) => {
                    let minimal_data = minimal_slice.get_mapped_range();
                    let minimal_bytes = minimal_data.to_vec();
                    
                    let mut minimal_values = Vec::new();
                    for i in 0..3 {
                        let offset = i * 4;
                        if offset + 4 <= minimal_bytes.len() {
                            let value_bytes: [u8; 4] = minimal_bytes[offset..offset+4].try_into().unwrap();
                            let value = f32::from_le_bytes(value_bytes);
                            minimal_values.push(value);
                        }
                    }
                    
                    println!("Minimal test results: {:?}", minimal_values);
                    
                    if minimal_values.get(0) == Some(&123.0) {
                        println!("✓ Minimal test passed");
                        true
                    } else {
                        println!("✗ Minimal test failed");
                        false
                    }
                }
                Ok(Err(e)) => {
                    println!("Minimal test mapping failed: {}", e);
                    false
                }
                Err(e) => {
                    println!("Minimal test channel error: {}", e);
                    false
                }
            };

            if !minimal_result {
                return Err("Minimal GPU test failed".to_string());
            }

            // Reset debug buffer
            println!("Resetting debug buffer to zeros...");
            self.queue.write_buffer(&debug_buffer, 0, &debug_init_data);
            self.device.poll(wgpu::Maintain::Wait);

            // === STEP 2: Multiple Box Blur Passes ===
            println!("\n=== Step 2: Multiple Box Blur Passes (approximates Gaussian) ===");
            
            // Calculate box radius for approximation
            let box_radius = self.calculate_box_radius();
            println!("Using box blur approximation: radius={} for {} passes", box_radius, BOX_BLUR_PASSES);
            println!("This approximates Gaussian with sigma={}", self.sigma);

            // Define clear pass structure: (input_view, output_view)
            let passes = [
                (&input_view, &intermediate_view1),      // Pass 0: Rgba8Unorm -> Rgba16Float
                (&intermediate_view1, &intermediate_view2), // Pass 1: Rgba16Float -> Rgba16Float
                (&intermediate_view2, &output_view),     // Pass 2: Rgba16Float -> Rgba16Float
            ];

            for (pass_index, (input_view, output_view)) in passes.iter().enumerate() {
                println!("\n--- Box Blur Pass {} of {} ---", pass_index + 1, BOX_BLUR_PASSES);
                
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
                
                let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("Box Blur Encoder Pass {}", pass_index)),
                });
                
                {
                    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(&format!("Box Blur Compute Pass {}", pass_index)),
                        timestamp_writes: None,
                    });
                    
                    compute_pass.set_pipeline(&self.box_blur_pipeline);
                    compute_pass.set_bind_group(0, &bind_group, &[]);
                    
                    // Optimized dispatch for tiled processing
                    let effective_width = (width as u32 + TILE_SIZE_X - 1) / TILE_SIZE_X;
                    let effective_height = (height as u32 + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
                    let dispatch_width = (effective_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                    let dispatch_height = (effective_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
                    
                    println!("Dispatch: {}x{} workgroups", dispatch_width, dispatch_height);
                    compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
                }
                
                // Submit this pass
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                
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
            let mut debug_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Debug Readback Encoder"),
            });
            debug_encoder.copy_buffer_to_buffer(&debug_buffer, 0, &debug_staging_buffer, 0, debug_buffer_size_bytes);
            self.queue.submit(Some(debug_encoder.finish()));

            self.device.poll(wgpu::Maintain::Wait);
            let debug_slice = debug_staging_buffer.slice(..);
            let (debug_sender, debug_receiver) = std::sync::mpsc::channel();
            debug_slice.map_async(wgpu::MapMode::Read, move |result| {
                debug_sender.send(result).unwrap();
            });

            self.device.poll(wgpu::Maintain::Wait);

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
                let value_bytes: [u8; 4] = debug_bytes[offset..offset+4].try_into().unwrap();
                let value = f32::from_le_bytes(value_bytes);
                debug_values.push(value);
            }

            println!("\n=== Debug Analysis ===");
            println!("Marker: {}", debug_values[0]);

            if debug_values.len() > 16 {
                println!("\n=== Pixel Tracking (first 4 pixels, last pass) ===");
                for i in 0..4 {
                    let offset = ((BOX_BLUR_PASSES - 1) as usize) * 16 + i * 4;
                    if offset + 3 < debug_values.len() {
                        println!("  Pixel {}: R={:.1}, G={:.1}, B={:.1}, A={:.1}", 
                            i, debug_values[offset], debug_values[offset+1], 
                            debug_values[offset+2], debug_values[offset+3]);
                    }
                }
            }

            drop(debug_data);
            debug_staging_buffer.unmap();

            // === COPY OUTPUT TO BUFFER ===
            println!("\n=== Step 4: Copying Results ===");
            
            // For Rgba16Float, each pixel is 8 bytes (4 × f16 = 8 bytes)
            let bytes_per_pixel = 8u32;
            let bytes_per_row_unaligned_16float = bytes_per_pixel * width as u32;
            let bytes_per_row_aligned_16float = ((bytes_per_row_unaligned_16float + alignment - 1) / alignment) * alignment;

            println!("Rgba16Float bytes per row: {} -> {} (aligned)", 
                bytes_per_row_unaligned_16float, bytes_per_row_aligned_16float);

            // Calculate output buffer size with aligned rows for Rgba16Float
            let output_buffer_size = (bytes_per_row_aligned_16float as u64 * height as u64) as wgpu::BufferAddress;
            println!("Output buffer size: {} bytes ({} aligned rows × {} height)", 
                output_buffer_size, bytes_per_row_aligned_16float, height);

            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Create encoder for final copy
            let mut final_encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Final Copy Encoder"),
            });

            // Copy texture to buffer with aligned bytes_per_row for Rgba16Float
            final_encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &output_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row_aligned_16float),
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
            self.device.poll(wgpu::Maintain::Wait);

            // Read back image results
            let buffer_slice = output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            self.device.poll(wgpu::Maintain::Wait);

            receiver
                .recv()
                .map_err(|e| format!("Failed to receive buffer: {}", e))?
                .map_err(|e| format!("Failed to map buffer: {}", e))?;

            let data = buffer_slice.get_mapped_range();

            // Copy the data, handling row alignment padding and converting f16 to u8
            let mut result_bytes = Vec::with_capacity(width * height * 4);
            let aligned_row_size_bytes = bytes_per_row_aligned_16float as usize;
            let bytes_per_pixel_usize = bytes_per_pixel as usize;
            let row_size_bytes = width * bytes_per_pixel_usize;

            println!("Extracting data: row_size={}, aligned_row_size={}, height={}", 
                row_size_bytes, aligned_row_size_bytes, height);

            // Extract each row, skipping the padding at the end, and convert f16 to u8
            for row in 0..height {
                let row_start = row * aligned_row_size_bytes;
                
                for col in 0..width {
                    let pixel_start = row_start + col * bytes_per_pixel_usize;
                    
                    if pixel_start + 7 < data.len() {
                        // Read f16 values (2 bytes each)
                        let r_f16 = half::f16::from_le_bytes([data[pixel_start], data[pixel_start + 1]]);
                        let g_f16 = half::f16::from_le_bytes([data[pixel_start + 2], data[pixel_start + 3]]);
                        let b_f16 = half::f16::from_le_bytes([data[pixel_start + 4], data[pixel_start + 5]]);
                        let a_f16 = half::f16::from_le_bytes([data[pixel_start + 6], data[pixel_start + 7]]);
                        
                        // Convert to f32 and then to u8, clamping to [0, 255]
                        result_bytes.push((r_f16.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8);
                        result_bytes.push((g_f16.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8);
                        result_bytes.push((b_f16.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8);
                        result_bytes.push((a_f16.to_f32().clamp(0.0, 1.0) * 255.0).round() as u8);
                    } else {
                        // Pad with zeros (transparent black)
                        result_bytes.extend(&[0, 0, 0, 0]);
                    }
                }
            }

            // Verify we got the right amount of data
            println!("Extracted {} bytes (expected {})", 
                result_bytes.len(), width * height * 4);

            // === DEBUG: Analyze the output buffer ===
            println!("\n=== Output Buffer Analysis ===");
            println!("Buffer size: {} bytes (expected {} bytes for {}x{} RGBA)",
                result_bytes.len(), width * height * 4, width, height);

            // Check first few pixels
            println!("First 4 pixels (16 bytes) as u8:");
            for i in 0..16.min(result_bytes.len()) {
                print!("{:3} ", result_bytes[i]);
                if i % 4 == 3 { print!(" | "); }
                if i % 16 == 15 { println!(); }
            }

            // Check if all values are zero or very small
            let mut all_zero = true;
            for &value in result_bytes.iter().take(100) {
                if value != 0 {
                    all_zero = false;
                    break;
                }
            }

            println!("First 100 bytes: all_zero={}", all_zero);

            // Cleanup
            drop(data);
            output_buffer.unmap();

            println!("Total GPU time: {:?}", total_start.elapsed());
            
            // If all values are zero, something is wrong
            if all_zero {
                return Err("GPU produced all zero values! Check shader execution.".to_string());
            }
            
            Ok((result_bytes, width, height))
        }
    }
}
