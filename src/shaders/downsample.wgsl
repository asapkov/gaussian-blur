// Box downsampling shader with variable factor

struct DownsampleParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    // No padding needed - struct is already 16 bytes (4*4)
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
    
    // CRITICAL: Check bounds
    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    // DEBUG: Set top-left pixel to cyan to verify downsample shader execution
    if dst_x == 0u && dst_y == 0u {
        textureStore(output_texture, vec2<i32>(0, 0), vec4<f32>(0.0, 1.0, 1.0, 1.0));
        return;
    }
    
    // DEBUG: Set top-right pixel to magenta
    if dst_x == params.dst_width - 1u && dst_y == 0u {
        textureStore(output_texture, vec2<i32>(i32(params.dst_width - 1u), 0), vec4<f32>(1.0, 0.0, 1.0, 1.0));
        return;
    }
    
    // DEBUG: Set bottom-left pixel to lime green
    if dst_x == 0u && dst_y == params.dst_height - 1u {
        textureStore(output_texture, vec2<i32>(0, i32(params.dst_height - 1u)), vec4<f32>(0.5, 1.0, 0.0, 1.0));
        return;
    }

    // Calculate scale factor (could be 2x, 4x, 8x, etc.)
    let scale_x = f32(params.src_width) / f32(params.dst_width);
    let scale_y = f32(params.src_height) / f32(params.dst_height);
    
    // Calculate source pixel range
    let src_start_x = u32(f32(dst_x) * scale_x);
    let src_start_y = u32(f32(dst_y) * scale_y);
    let src_end_x = u32(f32(dst_x + 1u) * scale_x);
    let src_end_y = u32(f32(dst_y + 1u) * scale_y);

    var sum = vec4<f32>(0.0);
    var count = 0.0;
    
    // Average all pixels in the source region
    for (var y = src_start_y; y < src_end_y && y < params.src_height; y++) {
        for (var x = src_start_x; x < src_end_x && x < params.src_width; x++) {
            sum += textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            count += 1.0;
        }
    }

    let result = sum / count;
    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), result);
}