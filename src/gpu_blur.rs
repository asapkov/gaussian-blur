//! GPU-accelerated Gaussian Blur using wgpu with shared memory optimization

#[cfg(feature = "gpu")]
use wgpu::{
    util::DeviceExt, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Queue,
    ComputePipeline,
    ComputePipelineDescriptor, Device, DeviceDescriptor, Instance, Limits,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
    StorageTextureAccess, TextureDescriptor, TextureViewDimension, TextureFormat,
};

#[cfg(feature = "gpu")]
use bytemuck;

use crate::Pixel;

// Constants for optimized work distribution
const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
const TILE_SIZE_X: u32 = 4;  // Each thread processes 4 pixels in X
const TILE_SIZE_Y: u32 = 4;  // Each thread processes 4 pixels in Y
const DEBUG_BUFFER_SIZE: usize = 1024; // Number of f32 values in debug buffer

// Shader parameters struct - must match the WGSL struct layout
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    kernel_size: u32,
    blur_alpha: u32,
    _padding0: u32,  // Padding to make total struct size multiple of 16
    sigma: f32,
    kernel_scale: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
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

            // Create bind group layout for both passes with debug buffer
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
                            min_binding_size: None,
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
                    // Intermediate write texture (storage, write-only, 8-bit) - binding 2
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
                    // Output texture (storage, write-only, 8-bit) - binding 4
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
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Debug buffer (storage, read-write) - binding 6
                    BindGroupLayoutEntry {
                        binding: 6,
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

    /// Validate kernel generation for debugging
    pub fn validate_kernel(&self) -> Result<(), String> {
        let kernel = generate_gaussian_kernel(self.radius, self.sigma);
        
        println!("=== Kernel Validation ===");
        println!("Sigma: {}, Radius: {}", self.sigma, self.radius);
        println!("Kernel size: {} (expected: {})", kernel.len(), 2 * self.radius + 1);
        
        // Check for non-finite values
        for (i, &value) in kernel.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("Kernel[{}] = {} is not finite!", i, value));
            }
        }
        
        // Check sum
        let sum: f32 = kernel.iter().sum();
        println!("Kernel sum: {}", sum);
        
        if (sum - 1.0).abs() > 0.01 {
            return Err(format!("Kernel not normalized! Sum = {}", sum));
        }
        
        // Check extremes
        let max_val = kernel.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = kernel.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        println!("Min value: {}, Max value: {}", min_val, max_val);
        
        if max_val == 0.0 {
            return Err("All kernel values are zero!".to_string());
        }
        
        Ok(())
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

    /// Apply blur to an image using GPU with shared memory optimization
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

            // Validate kernel before proceeding
            if let Err(e) = self.validate_kernel() {
                return Err(format!("Kernel validation failed: {}", e));
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

            // Create intermediate texture for horizontal pass result (8-bit storage)
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
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[TextureFormat::Rgba8Unorm],
            });

            let intermediate_read_view = intermediate_read.create_view(&wgpu::TextureViewDescriptor::default());

            // Create output texture (write-only storage, 8-bit)
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

            // Generate Gaussian kernel
            let mut kernel = generate_gaussian_kernel(self.radius, self.sigma);

            // Use more reasonable scaling to avoid overflow
            let mut scale_factor = if self.sigma > 10.0 {
                1_000.0  // 1 thousand for large sigma
            } else if self.sigma > 5.0 {
                10_000.0  // 10 thousand for medium sigma
            } else {
                100_000.0  // 100 thousand for small sigma
            };

            // But also check the actual kernel values
            let min_val = kernel.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = kernel.iter().fold(0.0f32, |a, &b| a.max(b));
            
            // Ensure the smallest value is at least 1.0 after scaling
            let scaled_min = min_val * scale_factor;
            if scaled_min < 1.0 {
                // Increase scale so smallest value is at least 1.0
                scale_factor = (1.0 / min_val).ceil();
                println!("Adjusted scale factor to {} for min value {} (scaled min = {})", 
                    scale_factor, min_val, min_val * scale_factor);
            }
            
            // Ensure we don't overflow 32-bit float precision
            let max_possible_sum = 255.0 * scale_factor * (2 * self.radius + 1) as f32;
            let max_safe_sum = 16_777_216.0; // 2^24, max precise integer in 32-bit float
            
            if max_possible_sum > max_safe_sum {
                // Reduce scale to avoid overflow
                scale_factor = max_safe_sum / (255.0 * (2 * self.radius + 1) as f32);
                println!("Reduced scale factor to {} to avoid overflow (max sum would be {})", 
                    scale_factor, max_possible_sum);
                
                // Check if scaled min is still reasonable
                if min_val * scale_factor < 0.5 {
                    println!("WARNING: Minimum kernel value after scaling is {} (may cause precision issues)", 
                        min_val * scale_factor);
                }
            }
            
            println!("Using scale factor: {} for sigma={} (min={}, max={}, scaled_min={})", 
                scale_factor, self.sigma, min_val, max_val, min_val * scale_factor);

            // Scale the kernel
            for weight in &mut kernel {
                *weight *= scale_factor;
            }

            // kernel now sums to scale_factor, not 1.0
            let scaled_sum = kernel.iter().sum::<f32>();
            println!("Scaled kernel sum (should be {}): {}", scale_factor, scaled_sum);
            
            // DEBUG: Check kernel
            println!("=== Kernel Debug ===");
            println!("Sigma: {}, Radius: {}, Kernel size: {}", 
                self.sigma, self.radius, kernel.len());
            
            // Show first few values
            let show_count = 5.min(kernel.len());
            println!("First {} kernel values: {:?}", show_count, &kernel[..show_count]);
            
            // Show middle values
            if kernel.len() > 10 {
                let mid = kernel.len() / 2;
                println!("Middle kernel values (around index {}): {:?}", 
                    mid, &kernel[mid-2..mid+3.min(kernel.len()-mid)]);
            }
            
            // Check for NaN/Inf
            for (i, &value) in kernel.iter().enumerate() {
                if !value.is_finite() {
                    eprintln!("ERROR: Kernel[{}] = {} is not finite!", i, value);
                    return Err(format!("Invalid kernel value at index {}", i));
                }
            }

            // Pad kernel to meet WGSL storage buffer alignment requirements
            // Storage buffers require 16-byte alignment for vec4<f32> access
            // We need to pad to a multiple of 4 floats (16 bytes)
            let padding_floats = ((kernel.len() + 3) / 4) * 4; // Round up to multiple of 4
            let mut kernel_padded = vec![0.0f32; padding_floats];
            kernel_padded[..kernel.len()].copy_from_slice(&kernel);

            println!("Kernel padded from {} to {} elements ({} bytes)", 
                kernel.len(), kernel_padded.len(), kernel_padded.len() * 4);

            // Check if kernel fits in buffer limits
            let kernel_buffer_size = kernel_padded.len() * 4; // 4 bytes per f32
            if kernel_buffer_size > device_limits.max_storage_buffer_binding_size as usize {
                return Err(format!(
                    "Kernel buffer size {} exceeds GPU storage buffer limit {}",
                    kernel_buffer_size, device_limits.max_storage_buffer_binding_size
                ));
            }

            let kernel_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Kernel Buffer"),
                contents: bytemuck::cast_slice(&kernel_padded),
                usage: wgpu::BufferUsages::STORAGE,
            });

            let params = ShaderParameters {
                width: width as u32,
                height: height as u32,
                radius: self.radius as u32,
                kernel_size: kernel.len() as u32,
                blur_alpha: self.blur_alpha as u32,
                _padding0: 0,
                sigma: self.sigma,
                kernel_scale: scale_factor,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
                _padding4: 0.0,
                _padding5: 0.0,
                _padding6: 0.0,
                _padding7: 0.0,
                _padding8: 0.0,
                _padding9: 0.0,
            };

            let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parameters Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            // Create debug buffer: counter (4 bytes) + DEBUG_BUFFER_SIZE f32 values
            let debug_buffer_size = (std::mem::size_of::<u32>() + DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
            let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Buffer"),
                size: debug_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Initialize debug buffer with zeros and counter at 0
            let mut debug_init_data = vec![0u8; debug_buffer_size as usize];
            // Set counter to 0 (first 4 bytes)
            debug_init_data[0..4].copy_from_slice(&0u32.to_le_bytes());
            self.queue.write_buffer(&debug_buffer, 0, &debug_init_data);

            // Create bind group with debug buffer
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
                    BindGroupEntry {
                        binding: 6,
                        resource: debug_buffer.as_entire_binding(),
                    },
                ],
            });

            // Create command encoder
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Gaussian Blur Encoder"),
                });

            // Horizontal pass
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
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

                println!("Horizontal dispatch: {}x{} workgroups", dispatch_width, dispatch_height);
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
            }

            // Add memory barrier to ensure horizontal pass completes
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

                println!("Vertical dispatch: {}x{} workgroups", dispatch_width, dispatch_height);
                compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
            }

            // Create staging buffer to read back debug results
            let debug_staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Debug Staging Buffer"),
                size: debug_buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy debug buffer to staging buffer
            encoder.copy_buffer_to_buffer(&debug_buffer, 0, &debug_staging_buffer, 0, debug_buffer_size);

            // Create staging buffer to read back image results
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

            // === READ DEBUG BUFFER FIRST ===
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
            
            // Parse debug buffer
            // First 4 bytes: atomic counter (u32)
            let counter_bytes: [u8; 4] = debug_bytes[0..4].try_into().unwrap();
            let counter = u32::from_le_bytes(counter_bytes);
            
            // Next bytes: f32 values
            let mut debug_values = Vec::new();
            for i in 0..DEBUG_BUFFER_SIZE.min(counter as usize) {
                let offset = 4 + i * 4;
                if offset + 4 <= debug_bytes.len() {
                    let value_bytes: [u8; 4] = debug_bytes[offset..offset+4].try_into().unwrap();
                    let value = f32::from_le_bytes(value_bytes);
                    debug_values.push(value);
                }
            }
            
            println!("=== Debug Buffer Analysis ===");
            println!("Debug counter: {} (max {})", counter, DEBUG_BUFFER_SIZE);
            
            if debug_values.is_empty() {
                println!("WARNING: No debug values written!");
            } else {
                println!("First {} debug values:", debug_values.len().min(50));
                for (i, &value) in debug_values.iter().enumerate().take(50) {
                    match i {
                        0 => println!("  [0] sigma = {:.2}", value),
                        1 => println!("  [1] radius = {:.0}", value),
                        2 => println!("  [2] kernel_scale = {:.2}", value),
                        3 => println!("  [3] width = {:.0}", value),
                        4 => println!("  [4] height = {:.0}", value),
                        5 => println!("  [5] kernel[0] = {:.6}", value),
                        6 => println!("  [6] kernel[1] = {:.6}", value),
                        7 => println!("  [7] kernel[mid] = {:.6}", value),
                        8 => println!("  [8] input pixel R = {:.1}", value),
                        9 => println!("  [9] input pixel G = {:.1}", value),
                        10 => println!("  [10] input pixel B = {:.1}", value),
                        11 => println!("  [11] input pixel A = {:.1}", value),
                        _ => {
                            if i < 30 {
                                let label = match i {
                                    12 => "pixel x",
                                    13 => "weight_sum",
                                    14 => "sum.r before norm",
                                    15 => "sum.g before norm",
                                    16 => "sum.b before norm",
                                    17 => "sum.a before norm",
                                    18 => "sum.r after norm",
                                    19 => "sum.g after norm",
                                    20 => "sum.b after norm",
                                    21 => "sum.a after norm",
                                    22 => "intermediate R",
                                    23 => "intermediate G",
                                    24 => "intermediate B",
                                    25 => "intermediate A",
                                    26 => "final output R",
                                    27 => "final output G",
                                    28 => "final output B",
                                    29 => "final output A",
                                    _ => "value",
                                };
                                println!("  [{}] {} = {:.3}", i, label, value);
                            } else if i == 30 {
                                println!("  ... ({} more values)", debug_values.len() - 30);
                            }
                        }
                    }
                }
            }
            
            drop(debug_data);
            debug_staging_buffer.unmap();

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

            // Copy the data to a Vec<u8> that we can return
            let result_bytes = data.to_vec();

            // === DEBUG: Analyze the output buffer ===
            println!("=== Output Buffer Analysis ===");
            println!("Buffer size: {} bytes (expected {} bytes for {}x{} RGBA)",
                result_bytes.len(), width * height * 4, width, height);

            if result_bytes.len() != width * height * 4 {
                return Err(format!("Wrong buffer size! Got {}, expected {}", 
                    result_bytes.len(), width * height * 4));
            }

            // Check first few pixels
            println!("First 4 pixels (16 bytes) as u8:");
            for i in 0..16.min(result_bytes.len()) {
                print!("{:3} ", result_bytes[i]);
                if i % 4 == 3 { print!(" | "); }
                if i % 16 == 15 { println!(); }
            }

            // Check if all values are zero or very small
            let mut all_zero = true;
            let mut all_same = true;
            let first_value = result_bytes[0];

            for &value in result_bytes.iter().take(100) {
                if value != 0 {
                    all_zero = false;
                }
                if value != first_value {
                    all_same = false;
                }
            }

            println!("First 100 bytes: all_zero={}, all_same={}", all_zero, all_same);

            // Calculate histogram of values
            let mut histogram = [0u32; 256];
            for &value in result_bytes.iter() {
                histogram[value as usize] += 1;
            }

            println!("Value distribution (most common):");
            let mut sorted: Vec<(usize, u32)> = histogram.iter().enumerate()
                .map(|(i, &count)| (i, count))
                .filter(|&(_, count)| count > 0)
                .collect();
            sorted.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

            for (value, count) in sorted.iter().take(10) {
                println!("  Value {}: {} pixels", value, count);
            }

            // Check middle of image
            let mid_offset = (width * height * 2) * 4; // Middle of image
            if mid_offset + 3 < result_bytes.len() {
                println!("Middle pixel (offset {}): R:{}, G:{}, B:{}, A:{}",
                    mid_offset,
                    result_bytes[mid_offset],
                    result_bytes[mid_offset + 1],
                    result_bytes[mid_offset + 2],
                    result_bytes[mid_offset + 3]
                );
            }

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

/// Generate Gaussian kernel with improved numerical stability for large sigma
fn generate_gaussian_kernel(radius: i32, sigma: f32) -> Vec<f32> {
    use std::f32::consts::PI;
    
    let size = (radius * 2 + 1) as usize;
    let mut kernel = vec![0.0; size];
    
    // For very large sigma, we need special handling
    if sigma > 50.0 {
        // For extremely large sigma, the Gaussian is essentially uniform
        let uniform_value = 1.0 / size as f32;
        kernel.iter_mut().for_each(|v| *v = uniform_value);
        return kernel;
    }
    
    let sigma2 = 2.0 * sigma * sigma;
    
    // Use log-space calculation for better numerical stability with large radius
    let log_sqrt_two_pi_sigma = (2.0 * PI).sqrt().ln() + sigma.ln();
    
    for i in 0..size {
        let x = (i as i32 - radius) as f32;
        let exponent = -x * x / sigma2;
        
        // For very small values, we can get underflow to 0
        // exp(-20) ≈ 2.06e-9, exp(-30) ≈ 9.36e-14
        if exponent < -30.0 {
            kernel[i] = 0.0;
        } else {
            let log_value = exponent - log_sqrt_two_pi_sigma;
            kernel[i] = log_value.exp();
        }
    }
    
    // Normalize kernel
    let sum: f32 = kernel.iter().sum();
    
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for value in kernel.iter_mut() {
            *value *= inv_sum;
        }
    } else {
        // Fallback to uniform kernel if all values are 0
        eprintln!("WARNING: Kernel sum is 0! Using uniform kernel.");
        let uniform_value = 1.0 / size as f32;
        kernel.iter_mut().for_each(|v| *v = uniform_value);
    }
    
    // Final verification
    let final_sum: f32 = kernel.iter().sum();
    if (final_sum - 1.0).abs() > 0.001 {
        eprintln!("WARNING: Kernel normalization failed! Final sum = {}", final_sum);
    }
    
    kernel
}
