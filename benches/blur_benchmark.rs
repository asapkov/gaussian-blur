use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gaussian_blur::{
    GaussianBlur, Pixel, 
    simple_gaussian_blur, 
    gaussian_blur_3x3, 
    gaussian_blur_5x5
};

fn create_test_image(width: usize, height: usize) -> Vec<Vec<Pixel>> {
    let mut image = vec![vec![Pixel::rgb(0, 0, 0); width]; height];
    
    // Create a gradient pattern
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = ((x as f32 + y as f32) / (width as f32 + height as f32) * 255.0) as u8;
            image[y][x] = Pixel::rgb(r, g, b);
        }
    }
    
    image
}

fn bench_simple_blur(c: &mut Criterion) {
    let image = create_test_image(256, 256);
    
    c.bench_function("simple_gaussian_blur_256x256", |b| {
        b.iter(|| {
            black_box(simple_gaussian_blur(black_box(&image), 2.0, None, true))
        })
    });
}

fn bench_optimized_blur_single_thread(c: &mut Criterion) {
    let image = create_test_image(256, 256);
    let blur = GaussianBlur::new(2.0, None, true)
        .with_simd(true)
        .with_threads(1);
    
    c.bench_function("optimized_blur_1thread_256x256", |b| {
        b.iter(|| {
            black_box(blur.blur(black_box(&image)))
        })
    });
}

fn bench_optimized_blur_multi_thread(c: &mut Criterion) {
    let image = create_test_image(256, 256);
    let blur = GaussianBlur::new(2.0, None, true)
        .with_simd(true);
    
    c.bench_function("optimized_blur_multithread_256x256", |b| {
        b.iter(|| {
            black_box(blur.blur(black_box(&image)))
        })
    });
}

fn bench_fast_3x3_blur(c: &mut Criterion) {
    let image = create_test_image(256, 256);
    
    c.bench_function("fast_3x3_blur_256x256", |b| {
        b.iter(|| {
            black_box(gaussian_blur_3x3(black_box(&image), true))
        })
    });
}

fn bench_fast_5x5_blur(c: &mut Criterion) {
    let image = create_test_image(256, 256);
    
    c.bench_function("fast_5x5_blur_256x256", |b| {
        b.iter(|| {
            black_box(gaussian_blur_5x5(black_box(&image), true))
        })
    });
}

fn bench_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_sizes");
    
    for (name, (width, height)) in [
        ("128x128", (128, 128)),
        ("512x512", (512, 512)),
        ("1024x1024", (1024, 1024)),
    ] {
        let image = create_test_image(width, height);
        let blur = GaussianBlur::new(2.0, None, true)
            .with_simd(true);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(blur.blur(black_box(&image)))
            })
        });
    }
    
    group.finish();
}

fn bench_sigma_values(c: &mut Criterion) {
    let image = create_test_image(256, 256);
    let mut group = c.benchmark_group("sigma_values");
    
    for sigma in [0.5, 1.0, 2.0, 3.0, 5.0] {
        let blur = GaussianBlur::new(sigma, None, true)
            .with_simd(true);
        
        group.bench_function(format!("sigma_{}", sigma), |b| {
            b.iter(|| {
                black_box(blur.blur(black_box(&image)))
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(20)  // Reduced for faster benchmarks
        .warm_up_time(std::time::Duration::from_secs(2))
        .measurement_time(std::time::Duration::from_secs(5));
    targets = 
        bench_simple_blur,
        bench_optimized_blur_single_thread,
        bench_optimized_blur_multi_thread,
        bench_fast_3x3_blur,
        bench_fast_5x5_blur,
        bench_different_sizes,
        bench_sigma_values
);

criterion_main!(benches);