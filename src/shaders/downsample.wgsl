// Simple 2x downsample shader

struct ShaderParameters {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _padding: vec4<u32>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: ShaderParameters;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;
    
    // CRITICAL: Check bounds
    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    // Map to source coordinates (2x2 average)
    let src_x = dst_x * 2u;
    let src_y = dst_y * 2u;

    // Sample 2x2 block
    var sum = vec4<f32>(0.0);
    var count = 0.0;

    for (var dx = 0u; dx < 2u; dx = dx + 1u) {
        for (var dy = 0u; dy < 2u; dy = dy + 1u) {
            let sample_x = min(src_x + dx, params.src_width - 1u);
            let sample_y = min(src_y + dy, params.src_height - 1u);
            sum += textureLoad(input_texture, vec2<i32>(i32(sample_x), i32(sample_y)), 0);
            count += 1.0;
        }
    }

    let result = sum / count;
    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), result);
}