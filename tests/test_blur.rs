use gaussian_blur::{Pixel, GaussianBlur, simple_gaussian_blur, gaussian_blur_3x3};
use tempfile::NamedTempFile;
use image::{RgbaImage, Rgba};

#[test]
fn test_pixel_conversions() {
    let pixel = Pixel::new(255, 128, 64, 32);
    let array = pixel.to_f32_array();
    let pixel2 = Pixel::from_f32_array(array);
    
    assert_eq!(pixel, pixel2);
    assert_eq!(array, [255.0, 128.0, 64.0, 32.0]);
}

#[test]
fn test_simple_blur() {
    // Create a 3x3 test image
    let image = vec![
        vec![Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0)],
        vec![Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0)],
        vec![Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0), Pixel::rgb(255, 0, 0)],
    ];
    
    let blurred = simple_gaussian_blur(&image, 1.0, Some(1), true);
    
    // After blur, all pixels should still be red (but slightly darker at edges)
    assert_eq!(blurred.len(), 3);
    assert_eq!(blurred[0].len(), 3);
    
    // Center pixel should be mostly red
    let center = blurred[1][1];
    assert!(center.r > 200);
    assert!(center.g < 50);
    assert!(center.b < 50);
}

#[test]
fn test_optimized_blur() {
    let image = vec![
        vec![Pixel::rgb(255, 0, 0), Pixel::rgb(0, 255, 0)],
        vec![Pixel::rgb(0, 0, 255), Pixel::rgb(255, 255, 255)],
    ];
    
    let blur = GaussianBlur::new(0.5, Some(1), true);
    let blurred = blur.blur(&image);
    
    assert_eq!(blurred.len(), 2);
    assert_eq!(blurred[0].len(), 2);
}

#[test]
fn test_fast_3x3_blur() {
    let image = vec![
        vec![Pixel::rgb(255, 0, 0); 5]; 5
    ];
    
    let blurred = gaussian_blur_3x3(&image, true);
    
    assert_eq!(blurred.len(), 5);
    assert_eq!(blurred[0].len(), 5);
    
    // All pixels should still be red-ish after blur
    for row in &blurred {
        for pixel in row {
            assert!(pixel.r > 200);
            assert!(pixel.g < 100);
            assert!(pixel.b < 100);
        }
    }
}

#[test]
fn test_image_conversions() {
    // Create a simple 2x2 image
    let mut img = RgbaImage::new(2, 2);
    img.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
    img.put_pixel(1, 0, Rgba([0, 255, 0, 255]));
    img.put_pixel(0, 1, Rgba([0, 0, 255, 255]));
    img.put_pixel(1, 1, Rgba([255, 255, 255, 255]));
    
    // Convert to pixels
    let pixels = gaussian_blur::image_to_pixels(&img);
    assert_eq!(pixels.len(), 2);
    assert_eq!(pixels[0].len(), 2);
    
    // Convert back to image
    let img2 = gaussian_blur::pixels_to_image(&pixels);
    
    // Compare pixels
    for y in 0..2 {
        for x in 0..2 {
            let p1 = img.get_pixel(x, y);
            let p2 = img2.get_pixel(x, y);
            assert_eq!(p1, p2);
        }
    }
}

#[test]
fn test_in_place_blur() {
    let mut image = vec![
        vec![Pixel::rgb(255, 0, 0); 10]; 10
    ];
    
    let original_pixel = image[5][5];
    let blur = GaussianBlur::new(1.0, None, true);
    
    blur.blur_in_place(&mut image);
    
    // Pixel should be changed after blur
    assert_ne!(image[5][5], original_pixel);
    // But still red-ish
    assert!(image[5][5].r > 200);
}