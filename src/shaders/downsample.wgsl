// Optimized 2x2 average downsample shader

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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;
    
    // Check bounds
    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }
    
    // Calculate source block coordinates (2x2 average)
    let src_base_x = dst_x * 2u;
    let src_base_y = dst_y * 2u;

    var sum = vec4<f32>(0.0);
    var samples = 0u;
    
    // Average 2x2 block with bounds checking
    for (var dy = 0u; dy < 2u; dy++) {
        for (var dx = 0u; dx < 2u; dx++) {
            let src_x = src_base_x + dx;
            let src_y = src_base_y + dy;

            if src_x < params.src_width && src_y < params.src_height {
                sum += textureLoad(input_texture, vec2<i32>(i32(src_x), i32(src_y)), 0);
                samples += 1u;
            }
        }
    }
    
    // Use if-else instead of ternary operator
    var avg: vec4<f32>;
    if samples > 0u {
        avg = sum / f32(samples);
    } else {
        avg = vec4<f32>(0.0);
    }

    textureStore(output_texture, vec2<u32>(dst_x, dst_y), avg);
}