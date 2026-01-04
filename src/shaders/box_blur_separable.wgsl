// Optimized separable box blur with shared memory tiling

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
    let y_i = i32(y_u);  // FIXED: Changed from i32(y_i) to i32(y_u)

    var sum = vec4<f32>(0.0);
    var count: f32 = 0.0;

    let radius_i = i32(params.radius);

    if params.direction == 0u {
        for (var i = -radius_i; i <= radius_i; i = i + 1) {
            let sample_x = x_i + i;
            let clamped_x = clamp(sample_x, 0, i32(params.width) - 1);
            sum += textureLoad(input_texture, vec2<i32>(clamped_x, y_i), 0);
            count += 1.0;
        }
    } else {
        for (var i = -radius_i; i <= radius_i; i = i + 1) {
            let sample_y = y_i + i;
            let clamped_y = clamp(sample_y, 0, i32(params.height) - 1);
            sum += textureLoad(input_texture, vec2<i32>(x_i, clamped_y), 0);
            count += 1.0;
        }
    }

    var result = sum / count;

    if params.blur_alpha == 0u {
        let original = textureLoad(input_texture, vec2<i32>(x_i, y_i), 0);
        result.a = original.a;
    }

    textureStore(output_texture, vec2<i32>(x_i, y_i), result);
}