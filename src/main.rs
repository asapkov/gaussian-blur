#[cfg(feature = "image-io")]
use clap::{Parser, ValueEnum};
#[cfg(feature = "image-io")]
use image::ImageReader;

use gaussian_blur::*;
use std::path::PathBuf;
use std::time::Instant;

/// Gaussian Blur image processing tool
#[cfg(feature = "image-io")]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input image file
    #[arg(short, long)]
    input: PathBuf,

    /// Output image file
    #[arg(short, long)]
    output: PathBuf,

    /// Blur intensity (sigma)
    #[arg(short, long, default_value_t = 2.0)]
    sigma: f32,

    /// Kernel radius (optional, defaults to 3 * sigma)
    #[arg(short, long)]
    radius: Option<i32>,

    /// Blur algorithm to use
    #[arg(long, value_enum, default_value_t = Algorithm::Optimized)]
    algorithm: Algorithm,

    /// Backend to use (CPU or GPU)
    #[arg(long, value_enum, default_value_t = Backend::Cpu)]
    backend: Backend,

    /// Number of threads to use (0 = auto-detect, CPU only)
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Blur alpha channel
    #[arg(long, default_value_t = true)]
    blur_alpha: bool,

    /// Enable SIMD optimizations (CPU only)
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    simd: bool,

    /// Output format (auto-detected from extension if not specified)
    #[arg(long)]
    format: Option<String>,
}

#[cfg(feature = "image-io")]
#[derive(Clone, ValueEnum, Debug)]
enum Algorithm {
    /// Simple single-threaded implementation
    Simple,
    /// Optimized with SIMD and multithreading
    Optimized,
    /// Fast 3x3 blur
    Fast3x3,
    /// Fast 5x5 blur
    Fast5x5,
    /// Unified CPU/GPU implementation
    Unified,
}

#[cfg(feature = "image-io")]
#[derive(Clone, ValueEnum, Debug)]
enum Backend {
    /// CPU backend (supports SIMD and multithreading)
    Cpu,
    /// GPU backend (requires 'gpu' feature)
    Gpu,
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load image
    println!("Loading image: {:?}", args.input);
    let img = ImageReader::open(&args.input)?
        .decode()?
        .into_rgba8();

    // Convert to pixels
    println!("Image dimensions: {}x{}", img.width(), img.height());
    let pixels = image_to_pixels(&img);

    // Apply blur
    println!("Applying Gaussian blur with sigma={}, algorithm={:?}, backend={:?}, simd={}",
             args.sigma, args.algorithm, args.backend, args.simd);

    let start = Instant::now();
    let blurred_pixels = match args.algorithm {
        Algorithm::Simple => {
            simple_gaussian_blur(&pixels, args.sigma, args.radius, args.blur_alpha)
        }
        Algorithm::Optimized => {
            let threads = if args.threads == 0 {
                num_cpus::get()
            } else {
                args.threads
            };

            GaussianBlur::new(args.sigma, args.radius, args.blur_alpha)
                .with_simd(args.simd)
                .with_threads(threads)
                .blur(&pixels)
        }
        Algorithm::Fast3x3 => {
            gaussian_blur_3x3(&pixels, args.blur_alpha)
        }
        Algorithm::Fast5x5 => {
            gaussian_blur_5x5(&pixels, args.blur_alpha)
        }
        Algorithm::Unified => {
            let threads = if args.threads == 0 {
                num_cpus::get()
            } else {
                args.threads
            };

            let blur = UnifiedGaussianBlur::new(args.sigma, args.radius, args.blur_alpha)
                .with_simd(args.simd)
                .with_threads(threads);
                
            match args.backend {
                Backend::Cpu => blur.with_cpu(),
                Backend::Gpu => {
                    #[cfg(feature = "gpu")]
                    {
                        blur.with_gpu()
                    }
                    #[cfg(not(feature = "gpu"))]
                    {
                        eprintln!("Warning: GPU backend requested but 'gpu' feature not enabled. Falling back to CPU.");
                        blur.with_cpu()
                    }
                }
            }.blur(&pixels)
        }
    };
    let duration = start.elapsed();

    println!("Blur completed in {:?}", duration);

    // Convert back to image
    let blurred_img = pixels_to_image(&blurred_pixels);

    // Save image
    println!("Saving image: {:?}", args.output);
    blurred_img.save(&args.output)?;

    println!("Done!");
    Ok(())
}

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!("This binary requires the 'image-io' feature to be enabled.");
    eprintln!("Build with: cargo build --release --features image-io");
    eprintln!("Or run: cargo run --release --features image-io -- <arguments>");
    std::process::exit(1);
}
