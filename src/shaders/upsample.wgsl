// Bilinear upsample shader with proper coordinate mapping

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

    // Map destination pixel to source texture space
    // Use +0.5 to sample from pixel centers
    let src_x = (f32(dst_x) + 0.5) * f32(params.src_width) / f32(params.dst_width);
    let src_y = (f32(dst_y) + 0.5) * f32(params.src_height) / f32(params.dst_height);
    
    // Get integer coordinates
    let x0 = u32(floor(src_x - 0.5));
    let y0 = u32(floor(src_y - 0.5));
    let x1 = min(x0 + 1u, params.src_width - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);
    
    // Calculate fractional parts
    let fx = src_x - f32(x0) - 0.5;
    let fy = src_y - f32(y0) - 0.5;
    
    // Clamp fx and fy to [0, 1] to handle edge cases
    let fx_clamped = clamp(fx, 0.0, 1.0);
    let fy_clamped = clamp(fy, 0.0, 1.0);
    
    // Sample four points
    let p00 = textureLoad(input_texture, vec2<i32>(i32(x0), i32(y0)), 0);
    let p10 = textureLoad(input_texture, vec2<i32>(i32(x1), i32(y0)), 0);
    let p01 = textureLoad(input_texture, vec2<i32>(i32(x0), i32(y1)), 0);
    let p11 = textureLoad(input_texture, vec2<i32>(i32(x1), i32(y1)), 0);
    
    // Bilinear interpolation
    let top = mix(p00, p10, fx_clamped);
    let bottom = mix(p01, p11, fx_clamped);
    let result = mix(top, bottom, fy_clamped);

    textureStore(output_texture, vec2<i32>(i32(dst_x), i32(dst_y)), result);
}