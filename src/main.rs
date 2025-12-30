//! Gaussian Blur CLI tool
use clap::{Parser, ValueEnum};
use gaussian_blur::{Pixel, UnifiedGaussianBlur, image_to_pixels};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image path
    #[arg(short, long)]
    input: String,

    /// Output image path
    #[arg(short, long)]
    output: String,

    /// Sigma value for Gaussian blur
    #[arg(short, long, default_value_t = 2.0)]
    sigma: f32,

    /// Kernel radius (optional, auto-calculated from sigma if not specified)
    #[arg(short, long)]
    radius: Option<i32>,

    /// Blur algorithm to use
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,

    /// Number of threads to use (CPU only, 0 = auto)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Enable SIMD optimizations (CPU only)
    #[arg(long, default_value_t = true)]
    simd: bool,

    /// Blur alpha channel
    #[arg(long, default_value_t = true)]
    blur_alpha: bool,
}

#[derive(Clone, Debug, ValueEnum)]
enum Backend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Loading image: {}", args.input);
    
    // Load image
    let img = image::open(&args.input)?;
    let rgba_img = img.to_rgba8();
    let (width, height) = rgba_img.dimensions();
    
    println!("Image size: {}x{}", width, height);
    
    // Convert to pixels
    let pixels = image_to_pixels(&rgba_img);
    
    // Create blur processor
    let mut blur = UnifiedGaussianBlur::new(args.sigma, args.radius, args.blur_alpha);
    
    match args.backend {
        Backend::Cpu => {
            println!("Using CPU backend with sigma={}, radius={:?}", args.sigma, args.radius);
            blur = blur.with_cpu();
            
            if args.threads > 0 {
                blur = blur.with_threads(args.threads);
                println!("Using {} threads", args.threads);
            }
            
            if args.simd {
                blur = blur.with_simd(true);
                println!("SIMD enabled");
            }
        }
        #[cfg(feature = "gpu")]
        Backend::Gpu => {
            println!("Using GPU backend with sigma={}, radius={:?}", args.sigma, args.radius);
            blur = blur.with_gpu();
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => {
            println!("Using Metal backend with sigma={}, radius={:?}", args.sigma, args.radius);
            blur = blur.with_metal();
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        Backend::Metal => {
            println!("Metal backend not available on this platform, falling back to CPU");
            blur = blur.with_cpu();
        }
        #[cfg(not(feature = "gpu"))]
        Backend::Gpu => {
            println!("GPU backend not available (feature not enabled), falling back to CPU");
            blur = blur.with_cpu();
        }
    }
    
    println!("Applying Gaussian blur...");
    
    // Apply blur and get raw bytes directly
    let (blurred_bytes, out_width, out_height) = blur.blur_to_bytes(&pixels)
        .map_err(|e| format!("Blur failed: {}", e))?;
    
    println!("Blur complete. Output size: {}x{}", out_width, out_height);
    
    // Save output image
    println!("Saving to: {}", args.output);
    image::save_buffer(
        &args.output,
        &blurred_bytes,
        out_width,
        out_height,
        image::ColorType::Rgba8,
    )?;
    
    println!("Done!");
    Ok(())
}
