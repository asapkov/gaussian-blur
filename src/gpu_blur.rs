//! GPU-accelerated Gaussian Blur using wgpu with multi-strategy approach

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
const DEBUG_BUFFER_SIZE: usize = 1024;
const BOX_BLUR_PASSES: u32 = 3;

// Strategy selection thresholds
const GAUSSIAN_THRESHOLD: f32 = 2.0;
const DOWNSAMPLE_THRESHOLD: f32 = 32.0;
const LARGE_SIGMA_THRESHOLD: f32 = 100.0;

// Shader parameters struct - must match the WGSL struct layout
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    operation_mode: u32, // 0 = box blur, 1 = downsample, 2 = upsample, 3 = gaussian blur
    sigma: f32,
    current_pass: u32, // Which pass we're on (for multi-pass operations)
    src_width: u32,    // Source texture width (for downsample/upsample)
    src_height: u32,   // Source texture height
    dst_width: u32,    // Destination texture width
    dst_height: u32,   // Destination texture height
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
}

// Manually implement Pod and Zeroable for ShaderParameters
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Pod for ShaderParameters {}
#[cfg(feature = "gpu")]
unsafe impl bytemuck::Zeroable for ShaderParameters {}

/// Blur strategy based on sigma value
enum BlurStrategy {
    Gaussian, // True Gaussian convolution
    Box3Pass, // 3-pass box blur approximation
    Downsample {
        factor: u32,         // Downscale factor (2, 4, or 8)
        adjusted_sigma: f32, // Sigma after downsampling
    },
}

/// GPU Gaussian Blur processor with multi-strategy approach
pub struct GpuGaussianBlur {
    #[cfg(feature = "gpu")]
    device: Device,
    #[cfg(feature = "gpu")]
    queue: Queue,
    #[cfg(feature = "gpu")]
    compute_pipeline: ComputePipeline,
    #[cfg(feature = "gpu")]
    pipeline_layout: PipelineLayout,
    #[cfg(feature = "gpu")]
    bind_group_layout: wgpu::BindGroupLayout,
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

            // Request the maximum limits the adapter supports
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

            // Load shader
            let shader_source = include_str!("shaders/gaussian_blur_shared.wgsl");
            println!(
                "Shader source loaded, length: {} bytes",
                shader_source.len()
            );

            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Gaussian Blur Multi-Strategy Shader"),
                source: ShaderSource::Wgsl(shader_source.into()),
            });

            // Create bind group layout
            let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Blur Bind Group Layout"),
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

            // Create compute pipeline
            let compute_pipeline =
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Blur Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

            Ok(Self {
                device,
                queue,
                compute_pipeline,
                pipeline_layout,
                bind_group_layout,
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

            // Execute selected strategy
            let (final_view, final_texture) = match &self.strategy {
                BlurStrategy::Gaussian => self.apply_gaussian_blur(&input_view, width, height)?,
                BlurStrategy::Box3Pass => self.apply_box_blur_3pass(&input_view, width, height)?,
                BlurStrategy::Downsample {
                    factor,
                    adjusted_sigma,
                } => self.apply_downsample_blur_upsample(
                    &input_view,
                    width,
                    height,
                    *factor,
                    *adjusted_sigma,
                )?,
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

            // Copy texture to buffer
            final_encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &final_texture,
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

            // Cleanup
            drop(data);
            // final_output_buffer.unmap();

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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create debug buffer
        let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Buffer"),
            size: (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Horizontal pass
        println!("Horizontal pass...");
        let horiz_params = ShaderParameters {
            width: width as u32,
            height: height as u32,
            radius: self.radius as u32,
            blur_alpha: self.blur_alpha as u32,
            operation_mode: 3, // Gaussian blur mode
            sigma: self.sigma,
            current_pass: 0, // Horizontal
            src_width: width as u32,
            src_height: height as u32,
            dst_width: width as u32,
            dst_height: height as u32,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
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
            layout: &self.bind_group_layout,
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
                    resource: debug_buffer.as_entire_binding(),
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

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &horiz_bind_group, &[]);

            let dispatch_width = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let dispatch_height = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        self.queue.submit(Some(horiz_encoder.finish()));
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        // Vertical pass
        println!("Vertical pass...");
        let vert_params = ShaderParameters {
            width: width as u32,
            height: height as u32,
            radius: self.radius as u32,
            blur_alpha: self.blur_alpha as u32,
            operation_mode: 3, // Gaussian blur mode
            sigma: self.sigma,
            current_pass: 1, // Vertical
            src_width: width as u32,
            src_height: height as u32,
            dst_width: width as u32,
            dst_height: height as u32,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
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
            layout: &self.bind_group_layout,
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
                    resource: debug_buffer.as_entire_binding(),
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

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &vert_bind_group, &[]);

            let dispatch_width = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let dispatch_height = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        self.queue.submit(Some(vert_encoder.finish()));
        self.device.poll(wgpu::PollType::Wait {
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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[TextureFormat::Rgba8Unorm],
        });

        let view1 = texture1.create_view(&wgpu::TextureViewDescriptor::default());
        let view2 = texture2.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create debug buffer
        let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Box Blur Debug Buffer"),
            size: (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Apply 6 blur passes (3 passes × 2 directions)
        let blur_passes = [
            (input_view, &view1, box_radii[0], 0u32, 0u32),
            (&view1, &view2, box_radii[0], 0u32, 1u32),
            (&view2, &view1, box_radii[1], 1u32, 0u32),
            (&view1, &view2, box_radii[1], 1u32, 1u32),
            (&view2, &view1, box_radii[2], 2u32, 0u32),
            (&view1, &output_view, box_radii[2], 2u32, 1u32),
        ];

        for (i, (input_view, output_view, radius, pass_num, direction)) in
            blur_passes.iter().enumerate()
        {
            println!("--- Box Blur Pass {} of 6 ---", i + 1);

            let params = ShaderParameters {
                width: width as u32,
                height: height as u32,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                operation_mode: 0, // Box blur mode
                sigma: self.sigma,
                current_pass: *direction, // 0 = horizontal, 1 = vertical
                src_width: width as u32,
                src_height: height as u32,
                dst_width: width as u32,
                dst_height: height as u32,
                _padding2: 0.0,
                _padding3: 0.0,
                _padding4: 0.0,
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

                compute_pass.set_pipeline(&self.compute_pipeline);
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

        // Create debug buffer
        let debug_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Downsample Debug Buffer"),
            size: (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Downsample parameters
        let down_params = ShaderParameters {
            width: down_width,
            height: down_height,
            radius: 0,
            blur_alpha: self.blur_alpha as u32,
            operation_mode: 1, // Downsample mode
            sigma: 0.0,
            current_pass: 0,
            src_width: width as u32,
            src_height: height as u32,
            dst_width: down_width,
            dst_height: down_height,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
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
            layout: &self.bind_group_layout,
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
                BindGroupEntry {
                    binding: 3,
                    resource: debug_buffer.as_entire_binding(),
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

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &down_bind_group, &[]);

            let dispatch_width = (down_width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let dispatch_height = (down_height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        self.queue.submit(Some(down_encoder.finish()));
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        println!("Downsample completed");

        // === STEP 2: Apply blur on downsampled image ===
        println!("\n=== Step 2: Applying blur on downsampled image ===");

        // Calculate box sizes for 3-pass approximation on downsampled image
        let box_sizes = Self::boxes_for_gauss_3pass(adjusted_sigma);
        let box_radii = box_sizes.map(|size| ((size as i32 - 1) / 2).max(0) as u32);

        println!("Adjusted sigma: {:.2}", adjusted_sigma);
        println!("Box radii: {:?}", box_radii);

        // Create intermediate textures for blur passes
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

        // Create debug buffer for blur passes
        let debug_buffer2 = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Downsampled Blur Debug Buffer"),
            size: (DEBUG_BUFFER_SIZE * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Apply 6 blur passes (3 passes × 2 directions) on downsampled image
        let blur_passes = [
            (&downsampled_view, &view1, box_radii[0], 0u32, 0u32),
            (&view1, &view2, box_radii[0], 0u32, 1u32),
            (&view2, &view1, box_radii[1], 1u32, 0u32),
            (&view1, &view2, box_radii[1], 1u32, 1u32),
            (&view2, &view1, box_radii[2], 2u32, 0u32),
            (&view1, &blurred_down_view, box_radii[2], 2u32, 1u32),
        ];

        for (i, (input_view, output_view, radius, pass_num, direction)) in
            blur_passes.iter().enumerate()
        {
            println!("--- Box Blur Pass {} of 6 (Downsampled) ---", i + 1);

            let params = ShaderParameters {
                width: down_width,
                height: down_height,
                radius: *radius,
                blur_alpha: self.blur_alpha as u32,
                operation_mode: 0, // Box blur mode
                sigma: adjusted_sigma,
                current_pass: *direction, // 0 = horizontal, 1 = vertical
                src_width: down_width,
                src_height: down_height,
                dst_width: down_width,
                dst_height: down_height,
                _padding2: 0.0,
                _padding3: 0.0,
                _padding4: 0.0,
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
                        resource: debug_buffer2.as_entire_binding(),
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

                compute_pass.set_pipeline(&self.compute_pipeline);
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

        let up_params = ShaderParameters {
            width: width as u32,
            height: height as u32,
            radius: 0,
            blur_alpha: self.blur_alpha as u32,
            operation_mode: 2, // Upsample mode
            sigma: 0.0,
            current_pass: 0,
            src_width: down_width,
            src_height: down_height,
            dst_width: width as u32,
            dst_height: height as u32,
            _padding2: 0.0,
            _padding3: 0.0,
            _padding4: 0.0,
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
            layout: &self.bind_group_layout,
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
                BindGroupEntry {
                    binding: 3,
                    resource: debug_buffer2.as_entire_binding(),
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

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &up_bind_group, &[]);

            let dispatch_width = (width as u32 + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let dispatch_height = (height as u32 + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        self.queue.submit(Some(up_encoder.finish()));
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        println!("Upsample completed");

        // Verify first pixel
        let debug_bytes = self.read_texture_to_cpu(&final_texture, 1, 1)?;
        if debug_bytes.len() >= 4 {
            println!(
                "First pixel after upsample: R={}, G={}, B={}, A={}",
                debug_bytes[0], debug_bytes[1], debug_bytes[2], debug_bytes[3]
            );
        }

        Ok((final_view, final_texture))
    }

    /// Helper function to read texture data for debugging
    #[cfg(feature = "gpu")]
    fn read_texture_to_cpu(
        &self,
        texture: &wgpu::Texture,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, String> {
        let alignment = 256;
        let bytes_per_row_unaligned = 4 * width as u32;
        let bytes_per_row_aligned =
            ((bytes_per_row_unaligned + alignment - 1) / alignment) * alignment;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Read Buffer"),
            size: (bytes_per_row_aligned as u64 * height as u64) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Debug Read Encoder"),
            });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
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
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        let buffer_slice = buffer.slice(..);
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
        let mut result_bytes = Vec::with_capacity(width * height * 4);

        for row in 0..height {
            let row_start = row * bytes_per_row_aligned as usize;
            let row_end = row_start + (width * 4);

            if row_end <= data.len() {
                result_bytes.extend_from_slice(&data[row_start..row_end]);
            }
        }

        drop(data);
        // buffer.unmap();
        Ok(result_bytes)
    }
}
