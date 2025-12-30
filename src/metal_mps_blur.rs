// metal_mps_blur.rs - Fixed with proper synchronization for Intel/NVIDIA
#![allow(unused_variables)]

#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2::{msg_send, rc::Retained, runtime::ProtocolObject, ClassType};
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal::{
    MTLCreateSystemDefaultDevice, MTLCommandQueue, MTLDevice, MTLPixelFormat,
    MTLRegion, MTLSize, MTLOrigin, MTLTexture, MTLTextureDescriptor,
    MTLTextureUsage, MTLStorageMode, MTLCommandBuffer, MTLBlitCommandEncoder,
    MTLResource,  // Import MTLResource trait
};
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc2_metal_performance_shaders::MPSImageGaussianBlur;
#[cfg(all(feature = "metal", target_os = "macos"))]
use std::ptr::NonNull;

pub struct MetalMPSBlur {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    sigma: f32,
}

impl MetalMPSBlur {
    pub fn new(sigma: f32, _kernel_size: Option<u32>) -> Result<Self, String> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            // Get the default Metal device
            let device = MTLCreateSystemDefaultDevice()
                .ok_or_else(|| "Failed to get Metal device".to_string())?;
            
            // Create command queue
            let command_queue = device
                .newCommandQueue()
                .ok_or_else(|| "Failed to create command queue".to_string())?;
            
            Ok(Self {
                device,
                command_queue,
                sigma,
            })
        }
        
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            Err("Metal not available on this platform".to_string())
        }
    }
    
    pub fn blur_to_bytes(&self, image_data: &[u8], width: u32, height: u32, sigma: Option<f32>) -> Result<Vec<u8>, String> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let sigma = sigma.unwrap_or(self.sigma);
            
            if image_data.len() != (width * height * 4) as usize {
                return Err(format!(
                    "Image data size mismatch: expected {} bytes, got {} bytes",
                    width * height * 4,
                    image_data.len()
                ));
            }
            
            println!("DEBUG: Metal blur started - width: {}, height: {}, sigma: {}", width, height, sigma);
            println!("DEBUG: Input data length: {}", image_data.len());
            
            // For large images on 2GB VRAM, let's check if we should downscale
            if width > 4096 || height > 4096 {
                println!("WARNING: Large image may exceed VRAM limits on GT 750M (2GB).");
                println!("Consider using a smaller image or CPU backend.");
            }
            
            // 1. Create Metal Textures with Managed storage mode for Intel/NVIDIA
            let desc = MTLTextureDescriptor::new();
            unsafe {
                desc.setPixelFormat(MTLPixelFormat::RGBA8Unorm);
                desc.setWidth(width as usize);
                desc.setHeight(height as usize);
                desc.setMipmapLevelCount(1);
                // CRITICAL: Use Managed mode for Intel/NVIDIA Macs
                desc.setStorageMode(MTLStorageMode::Managed);
                // Must include ShaderWrite for destination texture
                desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
            }
            
            let src_tex = self.device
                .newTextureWithDescriptor(&desc)
                .ok_or_else(|| "Failed to create source texture".to_string())?;
            
            let dest_tex = self.device
                .newTextureWithDescriptor(&desc)
                .ok_or_else(|| "Failed to create destination texture".to_string())?;

            println!("DEBUG: Textures created successfully (Managed mode)");
            
            // 2. Upload CPU pixels to GPU with proper synchronization
            let region = MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize { 
                    width: width as usize, 
                    height: height as usize, 
                    depth: 1 
                },
            };
            
            // Calculate bytes per row
            let bytes_per_row = (width * 4) as usize;
            
            let bytes_ptr = NonNull::new(image_data.as_ptr() as *mut u8)
                .ok_or_else(|| "Failed to create NonNull pointer".to_string())?;
            
            unsafe {
                src_tex.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region, 
                    0, 
                    bytes_ptr.cast::<std::ffi::c_void>(), 
                    bytes_per_row
                );
            }
            
            // CRITICAL FOR MANAGED STORAGE: Sync CPU→GPU using blit encoder
            let upload_cmd_buffer = self.command_queue
                .commandBuffer()
                .ok_or_else(|| "Failed to create upload command buffer".to_string())?;
            
            let upload_blit_encoder = upload_cmd_buffer
                .blitCommandEncoder()
                .ok_or_else(|| "Failed to create upload blit encoder".to_string())?;
            
            unsafe {
                use std::mem::transmute;
                let src_resource: &ProtocolObject<dyn MTLResource> = transmute(&*src_tex);
                upload_blit_encoder.synchronizeResource(src_resource);
                let _: () = msg_send![&upload_blit_encoder, endEncoding];
            }
            
            unsafe {
                let _: () = msg_send![&upload_cmd_buffer, commit];
                let _: () = msg_send![&upload_cmd_buffer, waitUntilCompleted];
            }

            println!("DEBUG: Data uploaded and synchronized to GPU");
            println!("DEBUG: First few pixels of input: {:?}", &image_data[0..16.min(image_data.len())]);
            
            // TEST: Verify upload by reading back
            let mut test_bytes = vec![0u8; 16];
            let test_ptr = NonNull::new(test_bytes.as_mut_ptr() as *mut u8)
                .ok_or_else(|| "Failed to create test pointer".to_string())?;
            
            unsafe {
                src_tex.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    test_ptr.cast::<std::ffi::c_void>(),
                    bytes_per_row,
                    MTLRegion {
                        origin: MTLOrigin { x: 0, y: 0, z: 0 },
                        size: MTLSize { width: 2, height: 2, depth: 1 },
                    },
                    0
                );
            }
            println!("DEBUG: Upload verification - first 2x2 pixels from GPU: {:?}", &test_bytes[0..16]);
            
            // 3. Apply Gaussian Blur (MPS)
            let blur_kernel: Retained<MPSImageGaussianBlur> = unsafe {
                let obj = msg_send![MPSImageGaussianBlur::class(), alloc];
                // Try f32 (standard for MPS)
                msg_send![obj, initWithDevice: &*self.device, sigma: sigma as f32]
            };

            println!("DEBUG: Blur kernel created with sigma={}", sigma);
            
            // Try to set edge mode if available
            unsafe {
                // MPSImageEdgeModeClamp = 1, Zero = 0
                // Try both values
                let _: () = msg_send![&blur_kernel, setEdgeMode: 1]; // Clamp
            }
            
            let cmd_buffer = self.command_queue
                .commandBuffer()
                .ok_or_else(|| "Failed to create command buffer".to_string())?;
            
            // Encode the blur operation
            unsafe {
                let _: () = msg_send![
                    &blur_kernel, 
                    encodeToCommandBuffer: &*cmd_buffer, 
                    sourceTexture: &*src_tex, 
                    destinationTexture: &*dest_tex
                ];
            }
            
            // CRITICAL: Add synchronization for Managed mode (GPU→CPU)
            let download_blit_encoder = cmd_buffer
                .blitCommandEncoder()
                .ok_or_else(|| "Failed to create download blit encoder".to_string())?;
            
            unsafe {
                use std::mem::transmute;
                let dest_resource: &ProtocolObject<dyn MTLResource> = transmute(&*dest_tex);
                download_blit_encoder.synchronizeResource(dest_resource);
                let _: () = msg_send![&download_blit_encoder, endEncoding];
            }
            
            // Commit and wait for completion
            unsafe {
                let _: () = msg_send![&cmd_buffer, commit];
                let _: () = msg_send![&cmd_buffer, waitUntilCompleted];
            }

            println!("DEBUG: GPU computation completed and synchronized");
            
            // 4. Download GPU result to CPU
            let mut out_bytes = vec![0u8; (width * height * 4) as usize];
            let out_bytes_ptr = NonNull::new(out_bytes.as_mut_ptr() as *mut u8)
                .ok_or_else(|| "Failed to create NonNull pointer".to_string())?;
            
            unsafe {
                dest_tex.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                    out_bytes_ptr.cast::<std::ffi::c_void>(), 
                    bytes_per_row, 
                    region, 
                    0
                );
            }

            println!("DEBUG: Result downloaded from GPU");
            println!("DEBUG: First few pixels of output: {:?}", &out_bytes[0..16.min(out_bytes.len())]);
            
            // Check if output is actually blurred
            // With sigma=20, the first pixel should be different from input
            let mut blur_detected = false;
            for i in (0..out_bytes.len()).step_by(4).take(10) {
                if i + 3 < out_bytes.len() && i + 3 < image_data.len() {
                    let input_pixel = &image_data[i..i+4];
                    let output_pixel = &out_bytes[i..i+4];
                    
                    // Check if RGB channels differ (ignore alpha for now)
                    let diff = (0..3).map(|j| {
                        (input_pixel[j] as i32 - output_pixel[j] as i32).abs()
                    }).sum::<i32>();
                    
                    if diff > 10 { // Arbitrary threshold
                        blur_detected = true;
                        println!("DEBUG: Blur detected at pixel {} - Input: {:?}, Output: {:?}, Diff: {}", 
                                i/4, input_pixel, output_pixel, diff);
                        break;
                    }
                }
            }
            
            if !blur_detected {
                println!("WARNING: No blur detected - output appears identical to input!");
                println!("DEBUG: First pixel comparison - Input: {:?}, Output: {:?}", 
                        &image_data[0..4], &out_bytes[0..4]);
            } else {
                println!("DEBUG: Blur detected successfully!");
            }
            
            Ok(out_bytes)
        }
        
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            Err("Metal not available on this platform".to_string())
        }
    }
}

pub fn blur_with_metal(image_data: &[u8], width: u32, height: u32, sigma: f32) -> Result<Vec<u8>, String> {
    let blur = MetalMPSBlur::new(sigma, None)?;
    blur.blur_to_bytes(image_data, width, height, Some(sigma))
}
