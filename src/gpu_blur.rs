//! GPU-accelerated Gaussian Blur using wgpu with shared memory optimization

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferDescriptor,
    BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, Limits,
    PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource,
    StorageTextureAccess, TextureDescriptor, TextureViewDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor,
};

#[cfg(feature = "gpu")]
use bytemuck;

use crate::Pixel;

// Constants for optimized work distribution
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
const TILE_SIZE_X: u32 = 4;  // Each thread processes 4 pixels in X
const TILE_SIZE_Y: u32 = 4;  // Each thread processes 4 pixels in Y

// Shader parameters struct - must match the WGSL struct layout
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    sigma: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
}

// Manually implement Pod and Zeroable for ShaderParameters
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for ShaderParameters {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for ShaderParameters {}

/// GPU Gaussian Blur processor with shared memory optimization
pub struct GpuGaussianBlur {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    queue: Queue,
    #[cfg(feature = "gpu")]
    horizontal_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    vertical_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    bind_group_layout: BindGroupLayout,
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

            // Load shader with both passes
            let shader_source = include_str!("shaders/gaussian_blur_shared.wgsl");
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Gaussian Blur Shared Memory Shader"),
                source: ShaderSource::Wgsl(shader_source.into()),
            });

            // Create bind group layout for both passes
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Gaussian Blur Bind Group Layout"),
                entries: &[
                    // Kernel buffer (storage, read-only) - binding 0
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(256), // Minimum 256 bytes for alignment
                        },
                        count: None,
                    },
                    // Input texture (texture_2d<f32> for reading) - binding 1
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Intermediate write texture (storage, write-only) - binding 2
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Intermediate read texture (texture_2d<f32> for reading) - binding 3
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Output texture (storage, write-only) - binding 4
                    BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Parameters buffer - binding 5
                    BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(48), // 48 bytes as per struct
                        },
                        count: None,
                    },
                ],
            });

            // Create pipeline layout
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Gaussian Blur Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            // Create two compute pipelines (horizontal and vertical)
            let horizontal_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Horizontal Blur Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "horizontal_pass",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            let vertical_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Vertical Blur Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "vertical_pass",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            Ok(Self {
                device,
                queue,
                horizontal_pipeline,
                vertical_pipeline,
                bind_group_layout,
                sigma,
                radius,
                blur_alpha,
            })
        }
    }

    /// Apply blur to an image using GPU with shared memory optimization
    pub fn blur(&self, image: &[Vec<Pixel>]) -> Result<Vec<Vec<Pixel>>, String> {
        #[cfg(not(feature = "gpu"))]
        {
            return Err("GPU feature not enabled".to_string());
        }
        
        #[cfg(feature = "gpu")]
        {
            use std::time::Instant;
            
            let total_start = Instant::now();
            let texture_start = Instant::now();
            
            if image.is_empty() || image[0].is_empty() {
                return Ok(Vec::new());
            }

            let height = image.len();
            let width = image[0].len();
            
            // Check GPU limits
            let device_limits = self.device.limits();
            
            // Debug output
            println!("Processing image: {}x{}", width, height);
            println!("GPU texture limit: {}", device_limits.max_texture_dimension_2d);
            println!("GPU buffer limit: {}", device_limits.max_buffer_size);
            
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
            let mut rgba_data = Vec::with_capacity(width * height * 4);
            for row in image {
                for pixel in row {
                    rgba_data.push(pixel.r);
                    rgba_data.push(pixel.g);
                    rgba_data.push(pixel.b);
                    rgba_data.push(pixel.a);
                }
            }

            println!("Texture creation: {:?}", texture_start.elapsed());
            let texture_write_start = Instant::now();

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
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            // Write image data to texture
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
                    bytes_per_row: Some(4 * width as u32),
                    rows_per_image: Some(height as u32),
                },
                wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
            );

            let input_view = input_texture.create_view(&TextureViewDescriptor::default());

            // Create intermediate texture for horizontal pass result (write-only storage)
            let intermediate_write = self.device.create_texture(&TextureDescriptor {
                label: Some("Intermediate Write Texture"),
                size: wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_write_view = intermediate_write.create_view(&TextureViewDescriptor::default());

            // Create intermediate texture for vertical pass reading (regular texture)
            let intermediate_read = self.device.create_texture(&TextureDescriptor {
                label: Some("Intermediate Read Texture"),
                size: wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_read_view = intermediate_read.create_view(&TextureViewDescriptor::default());

            // Create output texture (write-only storage)
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
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let output_view = output_texture.create_view(&TextureViewDescriptor::default());

            println!("Texture write: {:?}", texture_write_start.elapsed());
            let kernel_start = Instant::now();

            // Generate Gaussian kernel
            let kernel = generate_gaussian_kernel(self.radius, self.sigma);
            println!("Kernel size: {} elements, {} bytes", kernel.len(), kernel.len() * 4);
            
            // Pad kernel to at least 64 elements (256 bytes) for alignment
            let padding_size = 64usize.max(kernel.len()); // At least 64 elements
            let mut kernel_padded = vec![0.0f32; padding_size];
            kernel_padded[..kernel.len()].copy_from_slice(&kernel);
            
            println!("Padded kernel size: {} elements, {} bytes", kernel_padded.len(), kernel_padded.len() * 4);

            let kernel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Kernel Buffer"),
                contents: bytemuck::cast_slice(&kernel_padded),
                usage: BufferUsages::STORAGE,
            });

            println!("Kernel generation: {:?}", kernel_start.elapsed());
            let params_start = Instant::now();

            // Create parameters struct
            let params = ShaderParameters {
                width: width as u32,
                height: height as u32,
                radius: self.radius as u32,
                blur_alpha: self.blur_alpha as u32,
                sigma: self.sigma,
                _padding0: 0.0,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
                _padding4: 0.0,
                _padding5: 0.0,
                _padding6: 0.0,
            };

            println!("Shader parameters:");
            println!("  width: {}", params.width);
            println!("  height: {}", params.height);
            println!("  radius: {}", params.radius);
            println!("  blur_alpha: {}", params.blur_alpha);
            println!("  sigma: {}", params.sigma);

            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parameters Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: BufferUsages::UNIFORM,
            });

            // Create bind group
            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Gaussian Blur Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: kernel_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&intermediate_write_view),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&intermediate_read_view),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    BindGroupEntry {
                        binding: 5,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            println!("Params/buffer creation: {:?}", params_start.elapsed());
            let compute_start = Instant::now();

            // Create command encoder
            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("Gaussian Blur Encoder"),
                });

            // Horizontal pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Horizontal Blur Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.horizontal_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                
                // Optimized dispatch for tiled processing
                let effective_width = (width as u32 + TILE_SIZE_X - 1) / TILE_SIZE_X;
                let effective_height = (height as u32 + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
                
                let dispatch_width = (effective_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                let dispatch_height = (effective_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
                
                println!("Optimized dispatch: {}x{} workgroups (processing {}x{} effective pixels)",
                    dispatch_width, dispatch_height, effective_width, effective_height);
                println!("Total threads: {} (was {})", 
                    dispatch_width * dispatch_height * WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y,
                    ((width as u32 + 15) / 16) * ((height as u32 + 15) / 16) * 256);
                
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
            }

            // Copy from intermediate_write to intermediate_read
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: &intermediate_write,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: &intermediate_read,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
            );

            // Vertical pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("Vertical Blur Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.vertical_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                
                // Same optimized dispatch
                let effective_width = (width as u32 + TILE_SIZE_X - 1) / TILE_SIZE_X;
                let effective_height = (height as u32 + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
                
                let dispatch_width = (effective_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                let dispatch_height = (effective_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
                
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
            }

            println!("Compute setup: {:?}", compute_start.elapsed());
            let copy_start = Instant::now();

            // Create staging buffer to read back results
            let output_buffer_size = (width * height * 4) as wgpu::BufferAddress;
            let output_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("Output Buffer"),
                size: output_buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy texture to buffer
            encoder.copy_texture_to_buffer(
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
                        bytes_per_row: Some(4 * width as u32),
                        rows_per_image: Some(height as u32),
                    },
                },
                wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
            );

            println!("Buffer creation: {:?}", copy_start.elapsed());
            let submit_start = Instant::now();

            // Submit commands
            println!("Submitting commands to GPU...");
            self.queue.submit(Some(encoder.finish()));

            println!("Queue submission: {:?}", submit_start.elapsed());
            let readback_start = Instant::now();

            // Read back results
            let buffer_slice = output_buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });

            println!("Waiting for GPU to finish...");
            self.device.poll(wgpu::Maintain::Wait);
            
            receiver
                .recv()
                .map_err(|e| format!("Failed to receive buffer: {}", e))?
                .map_err(|e| format!("Failed to map buffer: {}", e))?;

            let data = buffer_slice.get_mapped_range();
            let result_bytes: &[u8] = &data;

            println!("Readback: {:?}", readback_start.elapsed());
            let convert_start = Instant::now();

            // Convert back to 2D pixel array
            let mut result = vec![vec![Pixel::new(0, 0, 0, 0); width]; height];
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * width + x) * 4;
                    result[y][x] = Pixel::new(
                        result_bytes[idx],
                        result_bytes[idx + 1],
                        result_bytes[idx + 2],
                        result_bytes[idx + 3],
                    );
                }
            }

            // Cleanup
            drop(data);
            output_buffer.unmap();

            println!("Conversion: {:?}", convert_start.elapsed());
            println!("Total GPU time: {:?}", total_start.elapsed());
            println!("GPU blur completed successfully!");
            Ok(result)
        }
    }
    
    /// Benchmark GPU performance
    pub async fn benchmark(&self) -> Result<f64, String> {
        #[cfg(feature = "gpu")]
        {
            use std::time::Instant;
            
            println!("Running GPU benchmark...");
            
            // Test different sizes
            let sizes = [(1024, 1024), (2048, 2048), (4096, 4096)];
            
            for &(width, height) in &sizes {
                println!("\nBenchmarking {}x{} image:", width, height);
                
                let data = vec![128u8; width * height * 4];
                
                // Time texture upload
                let upload_start = Instant::now();
                let texture = self.device.create_texture(&TextureDescriptor {
                    size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: TextureFormat::Rgba8Unorm,
                    usage: TextureUsages::COPY_DST,
                    label: Some("Benchmark Texture"),
                    view_formats: &[],
                });
                
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * width as u32),
                        rows_per_image: Some(height as u32),
                    },
                    wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
                );
                
                let upload_time = upload_start.elapsed();
                let upload_bandwidth = (width * height * 4) as f64 / upload_time.as_secs_f64() / 1e9;
                println!("  Upload: {:.3} ms ({:.2} GB/s)", upload_time.as_secs_f64() * 1000.0, upload_bandwidth);
            }
            
            Ok(0.0)
        }
        
        #[cfg(not(feature = "gpu"))]
        Err("GPU feature not enabled".to_string())
    }
}

/// Generate Gaussian kernel (same as CPU version)
fn generate_gaussian_kernel(radius: i32, sigma: f32) -> Vec<f32> {
    use std::f32::consts::PI;
    
    let size = (radius * 2 + 1) as usize;
    let mut kernel = vec![0.0; size];
    let sigma2 = 2.0 * sigma * sigma;
    let sqrt_two_pi_sigma = (2.0 * PI).sqrt() * sigma;

    let mut sum = 0.0;
    for i in 0..size {
        let x = (i as i32 - radius) as f32;
        let value = (-x * x / sigma2).exp() / sqrt_two_pi_sigma;
        kernel[i] = value;
        sum += value;
    }

    // Normalize kernel so weights sum to 1
    let inv_sum = 1.0 / sum;
    for value in kernel.iter_mut() {
        *value *= inv_sum;
    }

    kernel
}
