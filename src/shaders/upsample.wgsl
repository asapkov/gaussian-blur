// Simple bilinear upsample shader

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

    // Map to source coordinates
    let src_x = f32(dst_x) * f32(params.src_width) / f32(params.dst_width);
    let src_y = f32(dst_y) * f32(params.src_height) / f32(params.dst_height);

    // Simple nearest-neighbor for debugging
    let sample_x = u32(floor(src_x));
    let sample_y = u32(floor(src_y));

    let clamped_x = min(sample_x, params.src_width - 1u);
    let clamped_y = min(sample_y, params.src_height - 1u);

    let color = textureLoad(input_texture, vec2<i32>(i32(clamped_x), i32(clamped_y)), 0);
    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), color);
}