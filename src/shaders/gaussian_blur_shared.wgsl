// Gaussian Blur Compute Shader with Debug Output

struct Parameters {
    width: u32,
    height: u32,
    radius: u32,
    kernel_size: u32,
    blur_alpha: u32,
    _padding0: u32,
    sigma: f32,
    kernel_scale: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
};

struct DebugOutput {
    counter: atomic<u32>,
    values: array<f32, 1024>,
}

// Storage buffer for kernel (pre-scaled values)
@group(0) @binding(0)
var<storage, read> kernel: array<f32>;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var intermediate_write: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var intermediate_read: texture_2d<f32>;

@group(0) @binding(4)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(5)
var<uniform> params: Parameters;

@group(0) @binding(6)
var<storage, read_write> debug_buffer: DebugOutput;

// Constants
const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TILE_SIZE_X = 4u;
const TILE_SIZE_Y = 4u;

fn get_kernel_weight(index: u32) -> f32 {
    if (index < params.kernel_size) {
        return kernel[index];
    }
    return 0.0;
}

// Helper to convert from [0, 255] range in textures to proper f32
fn texture_sample_normalized(tex: texture_2d<f32>, coords: vec2<i32>) -> vec4<f32> {
    let pixel = textureLoad(tex, coords, 0);
    // Input texture has values in [0, 1] but represents [0, 255]
    return pixel * 255.0;
}

// Helper to write debug values
fn write_debug(value: f32) {
    let index = atomicAdd(&debug_buffer.counter, 1u);
    if (index < 1024u) {
        debug_buffer.values[index] = value;
    }
}

// Simple test to verify debug writing works
fn test_debug() {
    // Write some test values
    write_debug(1.0);
    write_debug(2.0);
    write_debug(3.0);
    write_debug(4.0);
    write_debug(5.0);
    
    // Write kernel info
    write_debug(f32(params.kernel_size));
    if (params.kernel_size > 0u) {
        write_debug(kernel[0]);
        if (params.kernel_size > 1u) {
            write_debug(kernel[1]);
        }
        let mid_idx = params.kernel_size / 2u;
        if (mid_idx < params.kernel_size) {
            write_debug(kernel[mid_idx]);
        }
    }
}

// Horizontal pass with debug
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn horizontal_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let radius = params.radius;
    let tile_start_x = global_id.x * TILE_SIZE_X;
    let tile_start_y = global_id.y * TILE_SIZE_Y;

    // DEBUG: First thread writes debug info
    if (tile_start_x == 0u && tile_start_y == 0u) {
        // Run debug test first
        test_debug();
        
        // Write parameter values
        write_debug(params.sigma);                    // sigma
        write_debug(f32(params.radius));              // radius
        write_debug(params.kernel_scale);             // kernel_scale
        write_debug(f32(params.width));               // width
        write_debug(f32(params.height));              // height

        // Check if we can read from input texture
        let test_coords = vec2<i32>(0, 0);
        
        // Test reading texture at coordinate 0,0
        let pixel00 = texture_sample_normalized(input_texture, test_coords);
        write_debug(pixel00.r);                         // input pixel R at (0,0)
        write_debug(pixel00.g);                         // input pixel G at (0,0)
        write_debug(pixel00.b);                         // input pixel B at (0,0)
        write_debug(pixel00.a);                         // input pixel A at (0,0)
        
        // Test reading at coordinate 1,0
        if (params.width > 1u) {
            let pixel10 = texture_sample_normalized(input_texture, vec2<i32>(1, 0));
            write_debug(pixel10.r);                     // input pixel R at (1,0)
            write_debug(pixel10.g);                     // input pixel G at (1,0)
            write_debug(pixel10.b);                     // input pixel B at (1,0)
            write_debug(pixel10.a);                     // input pixel A at (1,0)
        }
        
        // Test reading at coordinate 2,0
        if (params.width > 2u) {
            let pixel20 = texture_sample_normalized(input_texture, vec2<i32>(2, 0));
            write_debug(pixel20.r);                     // input pixel R at (2,0)
            write_debug(pixel20.g);                     // input pixel G at (2,0)
            write_debug(pixel20.b);                     // input pixel B at (2,0)
            write_debug(pixel20.a);                     // input pixel A at (2,0)
        }
    }

    // Process tile
    for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
        let y = tile_start_y + dy;
        if (y >= params.height) { break; }

        for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
            let x = tile_start_x + dx;
            if (x >= params.width) { break; }

            // Apply horizontal blur
            var sum = vec4<f32>(0.0);
            var weight_sum = 0.0;

            for (var k = 0u; k <= 2u * radius; k++) {
                let sample_x = clamp(i32(x) - i32(radius) + i32(k), 0, i32(params.width) - 1);
                let weight = get_kernel_weight(k);
                let pixel = texture_sample_normalized(input_texture, vec2<i32>(sample_x, i32(y)));
                sum += pixel * weight;
                weight_sum += weight;
            }

            // DEBUG: For first 4 pixels, write intermediate values
            if (x < 4u && y == 0u) {
                write_debug(f32(x));                  // pixel x coordinate
                write_debug(weight_sum);              // sum of weights used
                write_debug(sum.r);                   // sum.r before normalization
                write_debug(sum.g);                   // sum.g before normalization
                write_debug(sum.b);                   // sum.b before normalization
                write_debug(sum.a);                   // sum.a before normalization
            }

            // Normalize by weight_sum (which should be approx kernel_scale)
            if (weight_sum > 0.0) {
                sum /= weight_sum;
            }

            // DEBUG: For first 4 pixels, write final values
            if (x < 4u && y == 0u) {
                write_debug(sum.r);                   // sum.r after normalization
                write_debug(sum.g);                   // sum.g after normalization
                write_debug(sum.b);                   // sum.b after normalization
                write_debug(sum.a);                   // sum.a after normalization
            }

            // Store result (converted back to [0, 1] range for storage texture)
            textureStore(intermediate_write, vec2<u32>(x, y), sum / 255.0);
        }
    }
}

// Vertical pass with debug
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn vertical_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let radius = params.radius;
    let tile_start_x = global_id.x * TILE_SIZE_X;
    let tile_start_y = global_id.y * TILE_SIZE_Y;

    // DEBUG: First thread writes more debug info
    if (tile_start_x == 0u && tile_start_y == 0u) {
        // Check intermediate texture sample
        let pixel00 = texture_sample_normalized(intermediate_read, vec2<i32>(0, 0));
        write_debug(pixel00.r);                         // intermediate R at (0,0)
        write_debug(pixel00.g);                         // intermediate G at (0,0)
        write_debug(pixel00.b);                         // intermediate B at (0,0)
        write_debug(pixel00.a);                         // intermediate A at (0,0)
        
        // Check intermediate at (1,0)
        if (params.width > 1u) {
            let pixel10 = texture_sample_normalized(intermediate_read, vec2<i32>(1, 0));
            write_debug(pixel10.r);                     // intermediate R at (1,0)
            write_debug(pixel10.g);                     // intermediate G at (1,0)
            write_debug(pixel10.b);                     // intermediate B at (1,0)
            write_debug(pixel10.a);                     // intermediate A at (1,0)
        }
    }

    // Process tile
    for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
        let y = tile_start_y + dy;
        if (y >= params.height) { break; }

        for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
            let x = tile_start_x + dx;
            if (x >= params.width) { break; }

            // Apply vertical blur
            var sum = vec4<f32>(0.0);
            var weight_sum = 0.0;

            for (var k = 0u; k <= 2u * radius; k++) {
                let sample_y = clamp(i32(y) - i32(radius) + i32(k), 0, i32(params.height) - 1);
                let weight = get_kernel_weight(k);
                let pixel = texture_sample_normalized(intermediate_read, vec2<i32>(i32(x), sample_y));
                sum += pixel * weight;
                weight_sum += weight;
            }

            // Normalize
            if (weight_sum > 0.0) {
                sum /= weight_sum;
            }

            // Preserve alpha if blur_alpha is false (0)
            if (params.blur_alpha == 0u) {
                let original = texture_sample_normalized(input_texture, vec2<i32>(i32(x), i32(y)));
                sum.a = original.a;
            }

            // DEBUG: For first 4 pixels, write final output values
            if (x < 4u && y == 0u) {
                write_debug(sum.r);                   // final output R
                write_debug(sum.g);                   // final output G
                write_debug(sum.b);                   // final output B
                write_debug(sum.a);                   // final output A
            }

            // Store result
            textureStore(output_texture, vec2<u32>(x, y), sum / 255.0);
        }
    }
}
