// Optimized separable box blur with shared memory tiling

struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    _padding: vec3<u32>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: ShaderParameters;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // CRITICAL: Check bounds
    if x >= params.width || y >= params.height {
        return;
    }

    var sum = vec4<f32>(0.0);
    var count: f32 = 0.0;

    let radius = i32(params.radius);

    if params.direction == 0u {
        // Horizontal blur
        for (var i = -radius; i <= radius; i = i + 1) {
            let sample_x = i32(x) + i;
            let clamped_x = clamp(sample_x, 0, i32(params.width) - 1);
            sum += textureLoad(input_texture, vec2<i32>(clamped_x, i32(y)), 0);
            count += 1.0;
        }
    } else {
        // Vertical blur
        for (var i = -radius; i <= radius; i = i + 1) {
            let sample_y = i32(y) + i;
            let clamped_y = clamp(sample_y, 0, i32(params.height) - 1);
            sum += textureLoad(input_texture, vec2<i32>(i32(x), clamped_y), 0);
            count += 1.0;
        }
    }

    var result = sum / count;
    
    // Preserve alpha if needed
    if params.blur_alpha == 0u {
        let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
        result.a = original.a;
    }

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
}