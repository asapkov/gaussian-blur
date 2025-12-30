//! Metal-style GPU-accelerated Variable Gaussian Blur using wgpu
//! Adapted from VariableGaussianBlur.metal

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Queue,
    ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor,
    Instance, Limits, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, StorageTextureAccess, TextureDescriptor, TextureViewDimension,
    TextureFormat, PipelineLayout,
};

#[cfg(feature = "gpu")]
use bytemuck;

use crate::Pixel;

// Constants for work distribution
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
const MAX_SAMPLES_DEFAULT: f32 = 16.0; // Default maximum samples for Gaussian blur

// Uniforms struct matching WGSL layout - 48 bytes total
#[cfg(feature = "gpu")]
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
struct ShaderUniforms {
    bounding_rect: [f32; 4],    // x, y, width, height (16 bytes)
    radius: f32,                // Maximum blur radius (4 bytes)
    max_samples: f32,           // Maximum samples per pixel (4 bytes)
    vertical: f32,              // 0.0 = X axis, 1.0 = Y axis (4 bytes)
    normalize_edges: f32,       // 0.0 = false, 1.0 = true (4 bytes)
    _padding0: f32,             // Padding (4 bytes)
    _padding1: f32,             // Padding (4 bytes) 
    _padding2: f32,             // Padding (4 bytes)
    _padding3: f32,             // Padding (4 bytes) - Total: 48 bytes
}

#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for ShaderUniforms {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for ShaderUniforms {}

/// Metal-style GPU Variable Gaussian Blur processor
pub struct MetalGpuGaussianBlur {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    queue: Queue,
    #[cfg(feature = "gpu")]
    pipeline: ComputePipeline,           // Single pipeline for both axes
    #[cfg(feature = "gpu")]
    pipeline_layout: PipelineLayout,
    #[cfg(feature = "gpu")]
    bind_group_layout: wgpu::BindGroupLayout,
    max_radius: f32,                     // Maximum blur radius
    max_samples: f32,                    // Maximum samples per pixel
    normalize_edges: bool,               // Whether to normalize edges
}

impl MetalGpuGaussianBlur {
    /// Create a new Metal-style GPU Gaussian Blur processor
    pub async fn new(max_radius: f32, max_samples: Option<f32>, normalize_edges: bool) -> Result<Self, String> {
        let max_samples = max_samples.unwrap_or(MAX_SAMPLES_DEFAULT);
        
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
            println!("Metal-style Blur - Adapter limits:");
            println!("  max_texture_dimension_2d: {}", adapter_limits.max_texture_dimension_2d);

            // Request the maximum limits
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
                        label: Some("Metal Gaussian Blur Device"),
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
            println!("  max_compute_workgroups_per_dimension: {}", device_limits.max_compute_workgroups_per_dimension);

            // Load the Metal-style shader
            let shader_source = include_str!("shaders/variable_gaussian_blur.wgsl");
            println!("Shader source loaded, length: {} bytes", shader_source.len());

            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Variable Gaussian Blur Shader"),
                source: ShaderSource::Wgsl(shader_source.into()),
            });

            // Create bind group layout (4 bindings total)
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Variable Blur Bind Group Layout"),
                entries: &[
                    // Input texture - binding 0
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Mask texture - binding 1
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Output texture - binding 2
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
                    // Uniforms buffer - binding 3
                    BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<ShaderUniforms>() as u64),
                        },
                        count: None,
                    },
                ],
            });

            // Create pipeline layout
            let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Variable Blur Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            // Create compute pipeline
            let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Variable Blur Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "compute_blur",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            Ok(Self {
                device,
                queue,
                pipeline,
                pipeline_layout,
                bind_group_layout,
                max_radius,
                max_samples,
                normalize_edges,
            })
        }
    }

    /// Apply variable blur to an image using a mask texture
    pub fn blur_with_mask(
        &self,
        image: &[Vec<Pixel>],
        mask: &[Vec<f32>], // 2D array of alpha values (0.0 to 1.0)
    ) -> Result<Vec<Vec<Pixel>>, String> {
        let (bytes, width, height) = self.blur_with_mask_to_bytes(image, mask)?;
        
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

    /// Apply variable blur and return as bytes (RGBA format)
    pub fn blur_with_mask_to_bytes(
        &self,
        image: &[Vec<Pixel>],
        mask: &[Vec<f32>],
    ) -> Result<(Vec<u8>, usize, usize), String> {
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

            // Validate mask dimensions
            if mask.len() != height || mask[0].len() != width {
                return Err(format!(
                    "Mask dimensions {}x{} don't match image dimensions {}x{}",
                    mask.len(),
                    if mask.is_empty() { 0 } else { mask[0].len() },
                    height,
                    width
                ));
            }

            println!("Processing variable blur on image: {}x{} pixels", width, height);

            // Check GPU limits
            let device_limits = self.device.limits();
            if width as u32 > device_limits.max_texture_dimension_2d {
                return Err(format!(
                    "Image width {} exceeds GPU texture dimension limit {}",
                    width, device_limits.max_texture_dimension_2d
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

            // Convert mask to flat f32 bytes (single channel alpha)
            let mut mask_data = Vec::with_capacity(width * height * 4); // 4 bytes per f32
            for row in mask {
                for &alpha in row {
                    mask_data.extend_from_slice(&alpha.to_le_bytes());
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

            let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create mask texture (R32Float format for single channel f32)
            let mask_texture = self.device.create_texture(&TextureDescriptor {
                label: Some("Mask Texture"),
                size: wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[TextureFormat::R32Float],
            });

            // Write mask data to texture
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &mask_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &mask_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width as u32), // 4 bytes per f32
                    rows_per_image: Some(height as u32),
                },
                wgpu::Extent3d {
                    width: width as u32,
                    height: height as u32,
                    depth_or_array_layers: 1,
                },
            );

            let mask_view = mask_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create intermediate write texture for horizontal pass result (storage, write-only)
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
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_write_view = intermediate_write.create_view(&wgpu::TextureViewDescriptor::default());

            // Create intermediate read texture for vertical pass (regular texture)
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
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_read_view = intermediate_read.create_view(&wgpu::TextureViewDescriptor::default());

            // Create output texture
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

            // Create uniforms for horizontal pass
            let horizontal_uniforms = ShaderUniforms {
                bounding_rect: [0.0, 0.0, width as f32, height as f32],
                radius: self.max_radius,
                max_samples: self.max_samples,
                vertical: 0.0, // Horizontal pass
                normalize_edges: self.normalize_edges as u32 as f32,
                _padding0: 0.0,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
            };

            let horizontal_uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Horizontal Uniforms Buffer"),
                contents: bytemuck::cast_slice(&[horizontal_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Create bind group for horizontal pass (4 bindings)
            let horizontal_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Horizontal Pass Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,  // input_texture
                        resource: wgpu::BindingResource::TextureView(&input_view),
                    },
                    BindGroupEntry {
                        binding: 1,  // mask_texture
                        resource: wgpu::BindingResource::TextureView(&mask_view),
                    },
                    BindGroupEntry {
                        binding: 2,  // output_texture (intermediate_write for horizontal pass)
                        resource: wgpu::BindingResource::TextureView(&intermediate_write_view),
                    },
                    BindGroupEntry {
                        binding: 3,  // uniforms
                        resource: horizontal_uniforms_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create uniforms for vertical pass
            let vertical_uniforms = ShaderUniforms {
                bounding_rect: [0.0, 0.0, width as f32, height as f32],
                radius: self.max_radius,
                max_samples: self.max_samples,
                vertical: 1.0, // Vertical pass
                normalize_edges: self.normalize_edges as u32 as f32,
                _padding0: 0.0,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
            };

            let vertical_uniforms_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertical Uniforms Buffer"),
                contents: bytemuck::cast_slice(&[vertical_uniforms]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Create bind group for vertical pass (4 bindings)
            let vertical_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
                label: Some("Vertical Pass Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,  // input_texture (now intermediate_read texture)
                        resource: wgpu::BindingResource::TextureView(&intermediate_read_view),
                    },
                    BindGroupEntry {
                        binding: 1,  // mask_texture
                        resource: wgpu::BindingResource::TextureView(&mask_view),
                    },
                    BindGroupEntry {
                        binding: 2,  // output_texture (final output)
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    BindGroupEntry {
                        binding: 3,  // uniforms
                        resource: vertical_uniforms_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create command encoder
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Variable Blur Encoder"),
            });

            // Horizontal pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Horizontal Blur Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &horizontal_bind_group, &[]);

                // Dispatch workgroups
                let dispatch_width = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                let dispatch_height = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;

                println!("Horizontal dispatch: {}x{} workgroups", dispatch_width, dispatch_height);
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
            }

            // Copy from intermediate_write (storage) to intermediate_read (texture) for synchronization
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
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Vertical Blur Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &vertical_bind_group, &[]);

                // Dispatch workgroups
                let dispatch_width = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
                let dispatch_height = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;

                println!("Vertical dispatch: {}x{} workgroups", dispatch_width, dispatch_height);
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
            }

            // Create staging buffer to read back results
            let output_buffer_size = (width * height * 4) as wgpu::BufferAddress;
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Output Buffer"),
                size: output_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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

            // Submit commands
            self.queue.submit(Some(encoder.finish()));

            // Wait for completion
            self.device.poll(wgpu::Maintain::Wait);

            // Read back results
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
            let result_bytes = data.to_vec();

            println!("Total GPU variable blur time: {:?}", total_start.elapsed());
            
            // Debug: check first pixel
            if result_bytes.len() >= 4 {
                println!("First output pixel: R:{}, G:{}, B:{}, A:{}",
                    result_bytes[0], result_bytes[1], result_bytes[2], result_bytes[3]);
            }
            
            Ok((result_bytes, width, height))
        }
    }

    /// Apply uniform blur (all pixels use same radius)
    pub fn blur_uniform(&self, image: &[Vec<Pixel>]) -> Result<Vec<Vec<Pixel>>, String> {
        // Create a uniform mask (all 1.0)
        let height = image.len();
        let width = image[0].len();
        
        let uniform_mask: Vec<Vec<f32>> = vec![vec![1.0; width]; height];
        
        self.blur_with_mask(image, &uniform_mask)
    }

    /// Get the maximum blur radius
    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }

    /// Get the maximum samples per pixel
    pub fn max_samples(&self) -> f32 {
        self.max_samples
    }

    /// Check if edge normalization is enabled
    pub fn normalize_edges(&self) -> bool {
        self.normalize_edges
    }
}
