// Simple O(1) sliding window box blur
// Uses separate passes for horizontal and vertical

struct BoxBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var<uniform> params: BoxBlurParams;

// Simple sliding window implementation
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let radius = i32(params.radius);
    let diameter = 2 * radius + 1;
    let inv_diameter = 1.0 / f32(diameter);

    if params.direction == 0u {
        // HORIZONTAL PASS - compute sliding window for each row
        
        // Use a simple approach: for each workgroup row, compute prefix sums
        var sum = vec4<f32>(0.0);

        if x == 0u {
            // First pixel in row: compute full sum
            for (var i: i32 = -radius; i <= radius; i++) {
                let sample_x = clamp(i, 0, i32(params.width) - 1);
                sum += textureLoad(input_texture, vec2<i32>(sample_x, i32(y)), 0);
            }
        } else {
            // Subsequent pixels: use previous sum + new pixel - old pixel
            let prev_sum = textureLoad(input_texture, vec2<i32>(i32(x) - 1, i32(y)), 0);
            let add_x = clamp(i32(x) + radius, 0, i32(params.width) - 1);
            let remove_x = clamp(i32(x) - radius - 1, 0, i32(params.width) - 1);

            let new_pixel = textureLoad(input_texture, vec2<i32>(add_x, i32(y)), 0);
            let old_pixel = textureLoad(input_texture, vec2<i32>(remove_x, i32(y)), 0);

            sum = prev_sum + new_pixel - old_pixel;
        }

        var result = sum * inv_diameter;
        
        // Preserve alpha if needed
        if params.blur_alpha == 0u {
            let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            result.a = original.a;
        }

        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
    } else {
        // VERTICAL PASS - similar sliding window approach

        var sum = vec4<f32>(0.0);

        if y == 0u {
            // First pixel in column: compute full sum
            for (var i: i32 = -radius; i <= radius; i++) {
                let sample_y = clamp(i, 0, i32(params.height) - 1);
                sum += textureLoad(input_texture, vec2<i32>(i32(x), sample_y), 0);
            }
        } else {
            // Subsequent pixels: use previous sum + new pixel - old pixel
            let prev_sum = textureLoad(input_texture, vec2<i32>(i32(x), i32(y) - 1), 0);
            let add_y = clamp(i32(y) + radius, 0, i32(params.height) - 1);
            let remove_y = clamp(i32(y) - radius - 1, 0, i32(params.height) - 1);

            let new_pixel = textureLoad(input_texture, vec2<i32>(i32(x), add_y), 0);
            let old_pixel = textureLoad(input_texture, vec2<i32>(i32(x), remove_y), 0);

            sum = prev_sum + new_pixel - old_pixel;
        }

        var result = sum * inv_diameter;
        
        // Preserve alpha if needed
        if params.blur_alpha == 0u {
            let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            result.a = original.a;
        }

        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
    }
}