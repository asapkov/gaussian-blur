use gaussian_blur::{Pixel, image_to_pixels, pixels_to_image, UnifiedGaussianBlur};
use image::ImageReader;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load test image
    println!("Loading test image...");
    let img = ImageReader::open("test.png")?
        .decode()?
        .into_rgba8();
    
    let pixels = image_to_pixels(&img);
    println!("Image size: {}x{}", img.width(), img.height());
    
    println!("\n=== Shared Memory GPU Benchmark ===");
    println!("Note: First run includes shader compilation time\n");
    
    // Test 1: CPU with SIMD (baseline)
    println!("1. Testing CPU with SIMD...");
    let cpu_start = Instant::now();
    let cpu_blur = UnifiedGaussianBlur::new(2.0, None, true)
        .with_cpu()
        .with_simd(true)
        .with_threads(4)
        .blur(&pixels);
    let cpu_time = cpu_start.elapsed();
    println!("   CPU time: {:?}", cpu_time);
    
    // Save CPU result for comparison
    let cpu_img = pixels_to_image(&cpu_blur);
    cpu_img.save("output_cpu_simd.png")?;
    
    // Test 2: GPU with shared memory
    println!("\n2. Testing GPU with Shared Memory...");
    let gpu_start = Instant::now();
    let gpu_blur = UnifiedGaussianBlur::new(2.0, None, true)
        .with_gpu()
        .blur(&pixels);
    let gpu_time = gpu_start.elapsed();
    println!("   GPU time: {:?}", gpu_time);
    
    // Save GPU result
    let gpu_img = pixels_to_image(&gpu_blur);
    gpu_img.save("output_gpu_shared.png")?;
    
    println!("\n=== Performance Summary ===");
    println!("CPU (SIMD, 4 threads): {:?}", cpu_time);
    println!("GPU (Shared Memory):    {:?}", gpu_time);
    println!("Speedup: {:.2}x faster", 
             cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    
    // Verify results are similar
    println!("\n=== Result Verification ===");
    let mut max_diff = 0;
    for y in 0..img.height().min(100) as usize {
        for x in 0..img.width().min(100) as usize {
            let cpu_pixel = cpu_blur[y][x];
            let gpu_pixel = gpu_blur[y][x];
            let diff = (cpu_pixel.r as i32 - gpu_pixel.r as i32).abs()
                     + (cpu_pixel.g as i32 - gpu_pixel.g as i32).abs()
                     + (cpu_pixel.b as i32 - gpu_pixel.b as i32).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    println!("Maximum pixel difference: {}", max_diff);
    if max_diff <= 10 {
        println!("✓ Results are consistent");
    } else {
        println!("⚠ Results differ significantly");
    }
    
    Ok(())
}
