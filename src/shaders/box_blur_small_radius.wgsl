// Unrolled box blur for small fixed radii (1-8)

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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x_u = global_id.x;
    let y_u = global_id.y;

    if x_u >= params.width || y_u >= params.height {
        return;
    }

    let x_i = i32(x_u);
    let y_i = i32(y_u);

    var sum = vec4<f32>(0.0);
    let count_inv = 1.0 / f32(2u * params.radius + 1u);

    let radius_i = i32(params.radius);

    if params.direction == 0u {
        // Horizontal blur - unrolled for small radii
        switch params.radius {
            case 1u: {
                sum += textureLoad(input_texture, vec2<i32>(clamp(x_i - 1, 0, i32(params.width) - 1), y_i), 0);
                sum += textureLoad(input_texture, vec2<i32>(x_i, y_i), 0);
                sum += textureLoad(input_texture, vec2<i32>(clamp(x_i + 1, 0, i32(params.width) - 1), y_i), 0);
            }
            case 2u: {
                for (var i: i32 = -2; i <= 2; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            case 3u: {
                for (var i: i32 = -3; i <= 3; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            case 4u: {
                for (var i: i32 = -4; i <= 4; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            case 5u: {
                for (var i: i32 = -5; i <= 5; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            case 6u: {
                for (var i: i32 = -6; i <= 6; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            case 7u: {
                for (var i: i32 = -7; i <= 7; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            case 8u: {
                for (var i: i32 = -8; i <= 8; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
            default: {
                // Fallback for larger radii
                for (var i = -radius_i; i <= radius_i; i = i + 1) {
                    let sample_x = clamp(x_i + i, 0, i32(params.width) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0);
                }
            }
        }
    } else {
        // Vertical blur - unrolled for small radii
        switch params.radius {
            case 1u: {
                sum += textureLoad(input_texture, vec2<i32>(x_i, clamp(y_i - 1, 0, i32(params.height) - 1)), 0);
                sum += textureLoad(input_texture, vec2<i32>(x_i, y_i), 0);
                sum += textureLoad(input_texture, vec2<i32>(x_i, clamp(y_i + 1, 0, i32(params.height) - 1)), 0);
            }
            case 2u: {
                for (var i: i32 = -2; i <= 2; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            case 3u: {
                for (var i: i32 = -3; i <= 3; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            case 4u: {
                for (var i: i32 = -4; i <= 4; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            case 5u: {
                for (var i: i32 = -5; i <= 5; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            case 6u: {
                for (var i: i32 = -6; i <= 6; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            case 7u: {
                for (var i: i32 = -7; i <= 7; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            case 8u: {
                for (var i: i32 = -8; i <= 8; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
            default: {
                // Fallback for larger radii
                for (var i = -radius_i; i <= radius_i; i = i + 1) {
                    let sample_y = clamp(y_i + i, 0, i32(params.height) - 1);
                    sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0);
                }
            }
        }
    }

    var result = sum * count_inv;

    // Preserve alpha if needed
    if params.blur_alpha == 0u {
        let original = textureLoad(input_texture, vec2<i32>(x_i, y_i), 0);
        result.a = original.a;
    }

    textureStore(output_texture, vec2<i32>(x_i, y_i), result);
}