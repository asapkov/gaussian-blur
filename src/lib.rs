//! Gaussian Blur implementation with multithreading, SIMD, and GPU support

#![feature(portable_simd)]

use std::f32::consts::PI;
use rayon::prelude::*;
use std::simd::f32x4;

// At the top with other imports
#[cfg(feature = "gpu")]
use pollster;

/// RGBA Pixel structure
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Pixel {
    /// Create a new pixel
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create from RGB values (alpha = 255)
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Convert to float array
    pub fn to_f32_array(&self) -> [f32; 4] {
        [
            self.r as f32,
            self.g as f32,
            self.b as f32,
            self.a as f32,
        ]
    }

    /// Convert to SIMD vector (f32x4)
    pub fn to_simd(&self) -> f32x4 {
        f32x4::from_array([
            self.r as f32,
            self.g as f32,
            self.b as f32,
            self.a as f32,
        ])
    }

    /// Create from float array
    pub fn from_f32_array(arr: [f32; 4]) -> Self {
        Self {
            r: arr[0].clamp(0.0, 255.0) as u8,
            g: arr[1].clamp(0.0, 255.0) as u8,
            b: arr[2].clamp(0.0, 255.0) as u8,
            a: arr[3].clamp(0.0, 255.0) as u8,
        }
    }

    /// Create from SIMD vector
    pub fn from_simd(simd: f32x4) -> Self {
        let arr = simd.to_array();
        Self {
            r: arr[0].clamp(0.0, 255.0) as u8,
            g: arr[1].clamp(0.0, 255.0) as u8,
            b: arr[2].clamp(0.0, 255.0) as u8,
            a: arr[3].clamp(0.0, 255.0) as u8,
        }
    }

    /// Blend with another pixel
    pub fn blend(&self, other: &Pixel, alpha: f32) -> Pixel {
        let a = alpha.clamp(0.0, 1.0);
        let inv_a = 1.0 - a;

        Pixel::new(
            (self.r as f32 * inv_a + other.r as f32 * a) as u8,
            (self.g as f32 * inv_a + other.g as f32 * a) as u8,
            (self.b as f32 * inv_a + other.b as f32 * a) as u8,
            (self.a as f32 * inv_a + other.a as f32 * a) as u8,
        )
    }
}

/// Gaussian kernel generator
pub struct GaussianKernel {
    radius: i32,
    sigma: f32,
    kernel: Vec<f32>,
}

impl GaussianKernel {
    /// Create a new Gaussian kernel
    pub fn new(sigma: f32, radius: Option<i32>) -> Self {
        let radius = radius.unwrap_or_else(|| (3.0 * sigma).ceil() as i32);
        let kernel = generate_gaussian_kernel(radius, sigma);

        Self {
            radius,
            sigma,
            kernel,
        }
    }

    /// Get kernel radius
    pub fn radius(&self) -> i32 {
        self.radius
    }

    /// Get sigma value
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Get kernel weights
    pub fn kernel(&self) -> &[f32] {
        &self.kernel
    }
}

/// Gaussian Blur processor (CPU)
pub struct GaussianBlur {
    kernel: GaussianKernel,
    blur_alpha: bool,
    num_threads: Option<usize>,
    use_simd: bool,
}

impl GaussianBlur {
    /// Create a new Gaussian Blur processor
    pub fn new(sigma: f32, radius: Option<i32>, blur_alpha: bool) -> Self {
        Self {
            kernel: GaussianKernel::new(sigma, radius),
            blur_alpha,
            num_threads: None,
            use_simd: false,
        }
    }

    /// Set number of threads to use (None = use all available)
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }

    /// Enable SIMD optimizations
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.use_simd = enable;
        self
    }

    /// Apply blur to an image
    pub fn blur(&self, image: &[Vec<Pixel>]) -> Vec<Vec<Pixel>> {
        if image.is_empty() || image[0].is_empty() {
            return Vec::new();
        }

        let height = image.len();
        let width = image[0].len();

        // Configure rayon thread pool if specified
        if let Some(threads) = self.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .unwrap_or_default();
        }

        if self.use_simd {
            self.blur_simd(image, height, width)
        } else {
            self.blur_scalar(image, height, width)
        }
    }

    /// Scalar implementation (no SIMD)
    fn blur_scalar(&self, image: &[Vec<Pixel>], height: usize, width: usize) -> Vec<Vec<Pixel>> {
        let kernel = self.kernel.kernel();
        let radius = self.kernel.radius();
        let blur_alpha = self.blur_alpha;

        // First pass: horizontal blur
        let temp: Vec<Vec<[f32; 4]>> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![[0.0f32; 4]; width];

                for x in 0..width {
                    let mut sum = [0.0f32; 4];
                    let mut weight_sum = 0.0;

                    for kx in -radius..=radius {
                        let px = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;
                        let pixel = image[y][px].to_f32_array();
                        let weight = kernel[(kx + radius) as usize];

                        sum[0] += pixel[0] * weight;
                        sum[1] += pixel[1] * weight;
                        sum[2] += pixel[2] * weight;
                        if blur_alpha {
                            sum[3] += pixel[3] * weight;
                        }
                        weight_sum += weight;
                    }

                    sum[0] /= weight_sum;
                    sum[1] /= weight_sum;
                    sum[2] /= weight_sum;
                    sum[3] /= weight_sum;

                    if !blur_alpha {
                        sum[3] = image[y][x].a as f32;
                    }
                    row[x] = sum;
                }
                row
            })
            .collect();

        // Second pass: vertical blur
        (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![Pixel::new(0, 0, 0, 0); width];

                for x in 0..width {
                    let mut sum = [0.0f32; 4];
                    let mut weight_sum = 0.0;

                    for ky in -radius..=radius {
                        let py = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let pixel = temp[py][x];
                        let weight = kernel[(ky + radius) as usize];

                        sum[0] += pixel[0] * weight;
                        sum[1] += pixel[1] * weight;
                        sum[2] += pixel[2] * weight;
                        if blur_alpha {
                            sum[3] += pixel[3] * weight;
                        }
                        weight_sum += weight;
                    }

                    sum[0] /= weight_sum;
                    sum[1] /= weight_sum;
                    sum[2] /= weight_sum;
                    sum[3] /= weight_sum;

                    if !blur_alpha {
                        sum[3] = image[y][x].a as f32;
                    }
                    row[x] = Pixel::from_f32_array(sum);
                }
                row
            })
            .collect()
    }

    /// SIMD-optimized implementation using portable SIMD
    fn blur_simd(&self, image: &[Vec<Pixel>], height: usize, width: usize) -> Vec<Vec<Pixel>> {
        let kernel = self.kernel.kernel();
        let radius = self.kernel.radius();
        let blur_alpha = self.blur_alpha;

        // First pass: horizontal blur with SIMD
        let temp: Vec<Vec<f32x4>> = (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![f32x4::splat(0.0); width];

                for x in 0..width {
                    let mut sum = f32x4::splat(0.0);
                    let mut weight_sum = 0.0f32;

                    for kx in -radius..=radius {
                        let px = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;
                        let pixel = image[y][px].to_simd();
                        let weight = kernel[(kx + radius) as usize];
                        let weight_vec = f32x4::splat(weight);

                        sum += pixel * weight_vec;
                        weight_sum += weight;
                    }

                    let inv_weight_sum = 1.0 / weight_sum;
                    let normalized = sum * f32x4::splat(inv_weight_sum);

                    if !blur_alpha {
                        // Preserve original alpha
                        let mut arr = normalized.to_array();
                        arr[3] = image[y][x].a as f32;
                        row[x] = f32x4::from_array(arr);
                    } else {
                        row[x] = normalized;
                    }
                }
                row
            })
            .collect();

        // Second pass: vertical blur with SIMD
        (0..height)
            .into_par_iter()
            .map(|y| {
                let mut row = vec![Pixel::new(0, 0, 0, 0); width];

                for x in 0..width {
                    let mut sum = f32x4::splat(0.0);
                    let mut weight_sum = 0.0f32;

                    for ky in -radius..=radius {
                        let py = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let pixel = temp[py][x];
                        let weight = kernel[(ky + radius) as usize];
                        let weight_vec = f32x4::splat(weight);

                        sum += pixel * weight_vec;
                        weight_sum += weight;
                    }

                    let inv_weight_sum = 1.0 / weight_sum;
                    let normalized = sum * f32x4::splat(inv_weight_sum);

                    let final_pixel = if !blur_alpha {
                        // Preserve original alpha
                        let mut arr = normalized.to_array();
                        arr[3] = image[y][x].a as f32;
                        Pixel::from_f32_array(arr)
                    } else {
                        Pixel::from_simd(normalized)
                    };

                    row[x] = final_pixel;
                }
                row
            })
            .collect()
    }

    /// Blur in-place (overwrites the input image)
    pub fn blur_in_place(&self, image: &mut [Vec<Pixel>]) {
        let blurred = self.blur(image);
        for (y, row) in blurred.into_iter().enumerate() {
            image[y] = row;
        }
    }

    /// Get the kernel radius
    pub fn radius(&self) -> i32 {
        self.kernel.radius()
    }

    /// Get the sigma value
    pub fn sigma(&self) -> f32 {
        self.kernel.sigma()
    }
}

/// Generate Gaussian kernel
fn generate_gaussian_kernel(radius: i32, sigma: f32) -> Vec<f32> {
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

/// Simple Gaussian blur (single-threaded, for comparison)
pub fn simple_gaussian_blur(
    image: &[Vec<Pixel>],
    sigma: f32,
    radius: Option<i32>,
    blur_alpha: bool,
) -> Vec<Vec<Pixel>> {
    let radius = radius.unwrap_or_else(|| (3.0 * sigma).ceil() as i32);
    let kernel = generate_gaussian_kernel(radius, sigma);

    let height = image.len();
    let width = image[0].len();
    let mut result = vec![vec![Pixel::new(0, 0, 0, 0); width]; height];

    // Horizontal pass
    let mut temp = vec![vec![[0.0f32; 4]; width]; height];
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0f32; 4];
            let mut weight_sum = 0.0;

            for kx in -radius..=radius {
                let px = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;
                let pixel = image[y][px].to_f32_array();
                let weight = kernel[(kx + radius) as usize];

                sum[0] += pixel[0] * weight;
                sum[1] += pixel[1] * weight;
                sum[2] += pixel[2] * weight;
                if blur_alpha {
                    sum[3] += pixel[3] * weight;
                }
                weight_sum += weight;
            }

            sum[0] /= weight_sum;
            sum[1] /= weight_sum;
            sum[2] /= weight_sum;
            sum[3] /= weight_sum;

            if !blur_alpha {
                sum[3] = image[y][x].a as f32;
            }
            temp[y][x] = sum;
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0f32; 4];
            let mut weight_sum = 0.0;

            for ky in -radius..=radius {
                let py = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                let pixel = temp[py][x];
                let weight = kernel[(ky + radius) as usize];

                sum[0] += pixel[0] * weight;
                sum[1] += pixel[1] * weight;
                sum[2] += pixel[2] * weight;
                if blur_alpha {
                    sum[3] += pixel[3] * weight;
                }
                weight_sum += weight;
            }

            sum[0] /= weight_sum;
            sum[1] /= weight_sum;
            sum[2] /= weight_sum;
            sum[3] /= weight_sum;

            if !blur_alpha {
                sum[3] = image[y][x].a as f32;
            }
            result[y][x] = Pixel::from_f32_array(sum);
        }
    }

    result
}

/// Optimized 3x3 Gaussian blur
pub fn gaussian_blur_3x3(image: &[Vec<Pixel>], blur_alpha: bool) -> Vec<Vec<Pixel>> {
    let height = image.len();
    let width = image[0].len();

    // Predefined normalized 3x3 kernel
    let kernel: &[f32] = &[
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0,
        1.0/16.0, 2.0/16.0, 1.0/16.0
    ];

    (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![Pixel::new(0, 0, 0, 0); width];

            for x in 0..width {
                let mut sum = [0.0f32; 4];
                let mut k = 0;

                for ky in -1..=1 {
                    for kx in -1..=1 {
                        let py = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let px = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;

                        let pixel = image[py][px].to_f32_array();
                        let weight = kernel[k];

                        sum[0] += pixel[0] * weight;
                        sum[1] += pixel[1] * weight;
                        sum[2] += pixel[2] * weight;
                        if blur_alpha {
                            sum[3] += pixel[3] * weight;
                        }
                        k += 1;
                    }
                }

                if !blur_alpha {
                    sum[3] = image[y][x].a as f32;
                }

                row[x] = Pixel::from_f32_array(sum);
            }
            row
        })
        .collect()
}

/// Optimized 5x5 Gaussian blur
pub fn gaussian_blur_5x5(image: &[Vec<Pixel>], blur_alpha: bool) -> Vec<Vec<Pixel>> {
    let height = image.len();
    let width = image[0].len();

    // Predefined normalized 5x5 kernel (approximation)
    let kernel: &[f32] = &[
        1.0/273.0,  4.0/273.0,  7.0/273.0,  4.0/273.0, 1.0/273.0,
        4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0,
        7.0/273.0, 26.0/273.0, 41.0/273.0, 26.0/273.0, 7.0/273.0,
        4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0,
        1.0/273.0,  4.0/273.0,  7.0/273.0,  4.0/273.0, 1.0/273.0
    ];

    (0..height)
        .into_par_iter()
        .map(|y| {
            let mut row = vec![Pixel::new(0, 0, 0, 0); width];

            for x in 0..width {
                let mut sum = [0.0f32; 4];
                let mut k = 0;

                for ky in -2..=2 {
                    for kx in -2..=2 {
                        let py = (y as i32 + ky).clamp(0, height as i32 - 1) as usize;
                        let px = (x as i32 + kx).clamp(0, width as i32 - 1) as usize;

                        let pixel = image[py][px].to_f32_array();
                        let weight = kernel[k];

                        sum[0] += pixel[0] * weight;
                        sum[1] += pixel[1] * weight;
                        sum[2] += pixel[2] * weight;
                        if blur_alpha {
                            sum[3] += pixel[3] * weight;
                        }
                        k += 1;
                    }
                }

                if !blur_alpha {
                    sum[3] = image[y][x].a as f32;
                }

                row[x] = Pixel::from_f32_array(sum);
            }
            row
        })
        .collect()
}

// Re-export GPU module if feature is enabled
#[cfg(feature = "gpu")]
pub use gpu_blur::GpuGaussianBlur;

#[cfg(feature = "gpu")]
mod gpu_blur;

/// Enum to choose processing backend
pub enum BlurBackend {
    Cpu,
    Gpu,
}

/// Unified Gaussian Blur processor that can use CPU or GPU
pub struct UnifiedGaussianBlur {
    sigma: f32,
    radius: Option<i32>,
    blur_alpha: bool,
    backend: BlurBackend,
    num_threads: Option<usize>,
    use_simd: bool,
}

impl UnifiedGaussianBlur {
    /// Create a new unified Gaussian Blur processor
    pub fn new(sigma: f32, radius: Option<i32>, blur_alpha: bool) -> Self {
        Self {
            sigma,
            radius,
            blur_alpha,
            backend: BlurBackend::Cpu,
            num_threads: None,
            use_simd: false,
        }
    }
    
    /// Use GPU backend (requires 'gpu' feature)
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self) -> Self {
        self.backend = BlurBackend::Gpu;
        self
    }
    
    /// Use CPU backend
    pub fn with_cpu(mut self) -> Self {
        self.backend = BlurBackend::Cpu;
        self
    }
    
    /// Set number of threads (CPU only)
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }
    
    /// Enable SIMD optimizations (CPU only)
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.use_simd = enable;
        self
    }

    /// Apply blur and return raw RGBA bytes (width, height, bytes)
    /// This is more efficient for GPU processing and direct saving
    pub fn blur_to_bytes(&self, image: &[Vec<Pixel>]) -> Result<(Vec<u8>, usize, usize), String> {
        if image.is_empty() || image[0].is_empty() {
            return Ok((Vec::new(), 0, 0));
        }

        let height = image.len();
        let width = image[0].len();

        match self.backend {
            BlurBackend::Cpu => {
                // For CPU, blur normally then convert to bytes
                let cpu_blur = GaussianBlur::new(self.sigma, self.radius, self.blur_alpha)
                    .with_simd(self.use_simd);

                let pixels = if let Some(threads) = self.num_threads {
                    cpu_blur.with_threads(threads).blur(image)
                } else {
                    cpu_blur.blur(image)
                };

                let mut bytes = Vec::with_capacity(width * height * 4);

                for row in &pixels {
                    for pixel in row {
                        bytes.push(pixel.r);
                        bytes.push(pixel.g);
                        bytes.push(pixel.b);
                        bytes.push(pixel.a);
                    }
                }

                Ok((bytes, width, height))
            }
            #[cfg(feature = "gpu")]
            BlurBackend::Gpu => {
                let future = async {
                    match GpuGaussianBlur::new(self.sigma, self.radius, self.blur_alpha).await {
                        Ok(gpu_blur) => gpu_blur.blur_to_bytes(image),
                        Err(e) => Err(format!("Failed to create GPU blur: {}", e)),
                    }
                };

                pollster::block_on(future)
            }
            #[cfg(not(feature = "gpu"))]
            BlurBackend::Gpu => {
                panic!("GPU backend requires 'gpu' feature to be enabled");
            }
        }
    }
    
    // Also update the existing blur() method to use blur_to_bytes for GPU
    /// Apply blur using selected backend
    pub fn blur(&self, image: &[Vec<Pixel>]) -> Vec<Vec<Pixel>> {
        match self.blur_to_bytes(image) {
            Ok((bytes, width, height)) => {
                if width == 0 || height == 0 {
                    return Vec::new();
                }

                // Convert bytes back to pixels for compatibility
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

                result
            }
            Err(e) => {
                eprintln!("Blur failed: {}", e);
                Vec::new()
            }
        }
    }
}

/// Convert image::RgbaImage to Vec<Vec<Pixel>>
#[cfg(feature = "image-io")]
pub fn image_to_pixels(img: &image::RgbaImage) -> Vec<Vec<Pixel>> {
    let width = img.width() as usize;
    let height = img.height() as usize;

    let mut pixels = vec![vec![Pixel::default(); width]; height];

    for y in 0..height {
        for x in 0..width {
            let rgba = img.get_pixel(x as u32, y as u32).0;
            pixels[y][x] = Pixel::new(rgba[0], rgba[1], rgba[2], rgba[3]);
        }
    }

    pixels
}

/// Convert Vec<Vec<Pixel>> to image::RgbaImage
#[cfg(feature = "image-io")]
pub fn pixels_to_image(pixels: &[Vec<Pixel>]) -> image::RgbaImage {
    let height = pixels.len();
    let width = pixels[0].len();

    let mut img = image::RgbaImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let pixel = pixels[y][x];
            img.put_pixel(x as u32, y as u32, image::Rgba([pixel.r, pixel.g, pixel.b, pixel.a]));
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_pixel_conversions() {
        let pixel = Pixel::new(255, 128, 64, 32);
        let array = pixel.to_f32_array();
        let pixel2 = Pixel::from_f32_array(array);

        assert_eq!(pixel, pixel2);
        assert_eq!(array, [255.0, 128.0, 64.0, 32.0]);

        // Test SIMD conversions
        let simd = pixel.to_simd();
        let pixel3 = Pixel::from_simd(simd);
        assert_eq!(pixel, pixel3);
    }

    #[test]
    fn test_gaussian_blur() {
        // Create a simple 3x3 test image
        let image = vec![
            vec![Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0)],
            vec![Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0)],
            vec![Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0)],
        ];

        // Test scalar version
        let blur = GaussianBlur::new(1.0, Some(1), true);
        let blurred = blur.blur(&image);

        assert_eq!(blurred.len(), 3);
        assert_eq!(blurred[0].len(), 3);

        // Center pixel should still be mostly red
        let center = blurred[1][1];
        assert!(center.r > 200);
        assert!(center.g < 50);
        assert!(center.b < 50);

        // Test SIMD version
        let blur_simd = GaussianBlur::new(1.0, Some(1), true).with_simd(true);
        let blurred_simd = blur_simd.blur(&image);

        // Results should be similar (allow for small floating point differences)
        for y in 0..3 {
            for x in 0..3 {
                let p1 = blurred[y][x];
                let p2 = blurred_simd[y][x];
                assert!((p1.r as i32 - p2.r as i32).abs() <= 2);
                assert!((p1.g as i32 - p2.g as i32).abs() <= 2);
                assert!((p1.b as i32 - p2.b as i32).abs() <= 2);
            }
        }
    }
}
