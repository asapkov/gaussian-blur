// Box downsampling shader with variable factor

struct DownsampleParams {
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
var<uniform> params: DownsampleParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;

    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    let scale_x = f32(params.src_width) / f32(params.dst_width);
    let scale_y = f32(params.src_height) / f32(params.dst_height);

    let src_start_x = u32(f32(dst_x) * scale_x);
    let src_start_y = u32(f32(dst_y) * scale_y);
    let src_end_x = u32(f32(dst_x + 1u) * scale_x);
    let src_end_y = u32(f32(dst_y + 1u) * scale_y);

    var sum = vec4<f32>(0.0);
    var count = 0.0;

    for (var y = src_start_y; y < src_end_y && y < params.src_height; y++) {
        for (var x = src_start_x; x < src_end_x && x < params.src_width; x++) {
            sum += textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            count += 1.0;
        }
    }

    let result = sum / count;
    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), result);
}