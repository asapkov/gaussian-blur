// Gaussian Blur using Box Blur Approximation (3 passes approximates Gaussian)
// Direct to Rgba8Unorm storage texture - simplest and fastest for PNG output

struct Parameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    _padding0: u32,
    sigma: f32,
    current_pass: u32,  // Which pass we're on (0, 1, or 2 for box blur approximation)
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    _padding10: f32,
    _padding11: f32,
    _padding12: f32,
    _padding13: f32,
    _padding14: f32,
    _padding15: f32,
};

@group(0) @binding(3)
var<storage, read_write> debug_buffer: array<f32, 1024>;

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: Parameters;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TILE_SIZE_X = 4u;
const TILE_SIZE_Y = 4u;

fn texture_sample_normalized(tex: texture_2d<f32>, coords: vec2<i32>) -> vec4<f32> {
    let pixel = textureLoad(tex, coords, 0);
    return pixel;
}

// Simple box blur - much faster than Gaussian convolution
fn apply_box_blur(x: u32, y: u32, radius: u32, tex: texture_2d<f32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    var count = 0.0;

    let iradius = i32(radius);
    let ix = i32(x);
    let iy = i32(y);

    // For even passes, do horizontal blur; for odd passes, do vertical blur
    if (params.current_pass % 2u == 0u) {
        // Horizontal blur
        for (var k = -iradius; k <= iradius; k++) {
            let sample_x = clamp(ix + k, 0, i32(params.width) - 1);
            let pixel = texture_sample_normalized(tex, vec2<i32>(sample_x, iy));
            sum += pixel;
            count += 1.0;
        }
    } else {
        // Vertical blur
        for (var k = -iradius; k <= iradius; k++) {
            let sample_y = clamp(iy + k, 0, i32(params.height) - 1);
            let pixel = texture_sample_normalized(tex, vec2<i32>(ix, sample_y));
            sum += pixel;
            count += 1.0;
        }
    }

    if (count > 0.0) {
        return sum / count;
    }
    return vec4<f32>(0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn box_blur_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let tile_start_x = global_id.x * TILE_SIZE_X;
    let tile_start_y = global_id.y * TILE_SIZE_Y;

    // Enhanced debug info
    if (global_id.x == 0u && global_id.y == 0u) {
        debug_buffer[0] = 1000.0 + f32(params.current_pass); // Marker with pass number
        debug_buffer[1] = f32(params.width);  // Store width
        debug_buffer[2] = f32(params.height); // Store height
        debug_buffer[3] = f32(params.radius); // Store radius
        debug_buffer[4] = f32(params.blur_alpha); // Store blur_alpha flag
    }

    for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
        let y = tile_start_y + dy;
        if (y >= params.height) { break; }

        for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
            let x = tile_start_x + dx;
            if (x >= params.width) { break; }

            // Apply box blur - use VAR instead of LET since we modify it below
            var blurred = apply_box_blur(x, y, params.radius, input_texture);

            // Preserve alpha if blur_alpha is false
            if (params.blur_alpha == 0u) {
                let original = texture_sample_normalized(input_texture, vec2<i32>(i32(x), i32(y)));
                blurred.a = original.a;
            }

            // Store debug info for first pixel (multiply by 255 for debug)
            if (x < 4u && y == 0u) {
                let base_offset = params.current_pass * 20u + x * 4u;
                debug_buffer[base_offset + 5u] = blurred.r * 255.0;
                debug_buffer[base_offset + 6u] = blurred.g * 255.0;
                debug_buffer[base_offset + 7u] = blurred.b * 255.0;
                debug_buffer[base_offset + 8u] = blurred.a * 255.0;
            }

            // Store result directly as Rgba8Unorm (textureStore handles conversion)
            textureStore(output_texture, vec2<u32>(x, y), blurred);
        }
    }
}
