#!/bin/bash
echo "Testing all sigma values after shader fixes..."

echo "1. Testing sigma=1.0 (Gaussian)..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3" RUST_BACKTRACE=full RUST_LOG=wgpu=warn cargo +nightly run --release --features "image-io,gpu,metal" -- --input input.png --output test_sigma1.png --sigma 1.0 --backend gpu

echo "2. Testing sigma=10.0 (Box Blur)..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3" RUST_BACKTRACE=full RUST_LOG=wgpu=warn cargo +nightly run --release --features "image-io,gpu,metal" -- --input input.png --output test_sigma10.png --sigma 10.0 --backend gpu

echo "3. Testing sigma=50.0 (Downsample+Blur+Upsample)..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3" RUST_BACKTRACE=full RUST_LOG=wgpu=warn cargo +nightly run --release --features "image-io,gpu,metal" -- --input input.png --output test_sigma50.png --sigma 50.0 --backend gpu

echo "Done! Check if all output images have non-zero pixels."