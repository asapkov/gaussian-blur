#!/usr/bin/env python3
import subprocess
import time
import os

algorithms = ["Simple", "Optimized", "Fast3x3", "Fast5x5"]

print("Comparing Gaussian Blur Algorithms")
print("=" * 50)

for algo in algorithms:
    output_file = f"output_{algo.lower()}.png"
    
    # Build the command
    cmd = [
        "cargo", "run", "--release", "--features", "image-io", "--",
        "--input", "test.png",
        "--output", output_file,
        "--sigma", "2.0",
        "--algorithm", algo,
        "--threads", "4"
    ]
    
    print(f"\nRunning {algo} algorithm...")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        file_size = os.path.getsize(output_file)
        print(f"✓ Success: {elapsed:.2f} seconds, Output: {output_file} ({file_size:,} bytes)")
        
        # Print any output from the program
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line and not line.startswith("Loading") and not line.startswith("Saving"):
                    print(f"  {line}")
    else:
        print(f"✗ Failed: {result.stderr}")

print("\n" + "=" * 50)
print("Comparison complete!")
