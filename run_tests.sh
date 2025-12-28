i#!/bin/bash

# Test script for Gaussian blur algorithms

INPUT="test.png"
OUTPUT_PREFIX="output"

echo "Testing Gaussian blur algorithms on $INPUT"

# Test 1: Simple algorithm
echo -e "\n=== Testing Simple Algorithm ==="
cargo run --release --features image-io -- \
    --input "$INPUT" \
    --output "${OUTPUT_PREFIX}_simple.png" \
    --sigma 2.0 \
    --algorithm simple \
    --threads 4

# Test 2: Optimized algorithm
echo -e "\n=== Testing Optimized Algorithm ==="
cargo run --release --features image-io -- \
    --input "$INPUT" \
    --output "${OUTPUT_PREFIX}_optimized.png" \
    --sigma 2.0 \
    --algorithm optimized \
    --threads 4

# Test 3: Fast 3x3 algorithm
echo -e "\n=== Testing Fast 3x3 Algorithm ==="
cargo run --release --features image-io -- \
    --input "$INPUT" \
    --output "${OUTPUT_PREFIX}_fast3x3.png" \
    --sigma 2.0 \
    --algorithm fast3x3 \
    --threads 4

# Test 4: Fast 5x5 algorithm
echo -e "\n=== Testing Fast 5x5 Algorithm ==="
cargo run --release --features image-io -- \
    --input "$INPUT" \
    --output "${OUTPUT_PREFIX}_fast5x5.png" \
    --sigma 2.0 \
    --algorithm fast5x5 \
    --threads 4

# Test 5: Different sigma values
echo -e "\n=== Testing Different Sigma Values ==="
for sigma in 1.0 2.0 3.0 5.0; do
    echo "Sigma: $sigma"
    cargo run --release --features image-io -- \
        --input "$INPUT" \
        --output "${OUTPUT_PREFIX}_sigma${sigma}.png" \
        --sigma "$sigma" \
        --algorithm optimized \
        --threads 4
done

echo -e "\nAll tests completed!"
ls -la ${OUTPUT_PREFIX}_*.png
