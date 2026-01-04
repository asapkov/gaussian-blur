// Improved box downsampling with proper coordinate handling

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

    // Calculate scale factors
    let scale_x = f32(params.src_width) / f32(params.dst_width);
    let scale_y = f32(params.src_height) / f32(params.dst_height);
    
    // Calculate source pixel range with proper rounding
    let src_start_x_f = f32(dst_x) * scale_x;
    let src_start_y_f = f32(dst_y) * scale_y;
    let src_end_x_f = f32(dst_x + 1u) * scale_x;
    let src_end_y_f = f32(dst_y + 1u) * scale_y;
    
    // Convert to integer ranges with proper rounding
    let src_start_x = u32(floor(src_start_x_f));
    let src_start_y = u32(floor(src_start_y_f));
    let src_end_x = u32(ceil(src_end_x_f));
    let src_end_y = u32(ceil(src_end_y_f));
    
    // Clamp to source image bounds
    let start_x = src_start_x;
    let start_y = src_start_y;
    let end_x = min(src_end_x, params.src_width);
    let end_y = min(src_end_y, params.src_height);

    var sum = vec4<f32>(0.0);
    var count = 0.0;
    
    // Average all pixels in the source region
    for (var y = start_y; y < end_y; y++) {
        for (var x = start_x; x < end_x; x++) {
            sum += textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            count += 1.0;
        }
    }

    var result = vec4<f32>(0.0);
    if count > 0.0 {
        result = sum / count;
    }

    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), result);
}