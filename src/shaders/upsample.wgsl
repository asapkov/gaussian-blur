// Bilinear upsample shader with variable scale factor

struct UpsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: UpsampleParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;

    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    let src_x_norm = f32(dst_x) / f32(params.dst_width - 1u);
    let src_y_norm = f32(dst_y) / f32(params.dst_height - 1u);

    let src_x_f = src_x_norm * f32(params.src_width - 1u);
    let src_y_f = src_y_norm * f32(params.src_height - 1u);

    let x0 = u32(floor(src_x_f));
    let y0 = u32(floor(src_y_f));
    let x1 = min(x0 + 1u, params.src_width - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);

    let fx = src_x_f - f32(x0);
    let fy = src_y_f - f32(y0);

    let p00 = textureLoad(input_texture, vec2<i32>(i32(x0), i32(y0)), 0);
    let p10 = textureLoad(input_texture, vec2<i32>(i32(x1), i32(y0)), 0);
    let p01 = textureLoad(input_texture, vec2<i32>(i32(x0), i32(y1)), 0);
    let p11 = textureLoad(input_texture, vec2<i32>(i32(x1), i32(y1)), 0);

    let top = mix(p00, p10, fx);
    let bottom = mix(p01, p11, fx);
    let result = mix(top, bottom, fy);

    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), result);
}