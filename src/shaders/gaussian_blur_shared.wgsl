// Gaussian Blur Compute Shader - Simple global memory version
// Two-pass separable Gaussian blur

struct Parameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    sigma: f32,
    _padding: vec3<f32>,
};

// Storage buffer for kernel
@group(0) @binding(0)
var<storage, read> kernel: array<f32>;

@group(0) @binding(1)
var input_texture: texture_2d<f32>;

@group(0) @binding(2)
var intermediate_write: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var intermediate_read: texture_2d<f32>;

@group(0) @binding(4)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(5)
var<uniform> params: Parameters;

// Horizontal pass (blur in X direction)
@compute @workgroup_size(16, 16, 1)
fn horizontal_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let radius = params.radius;
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    var sum = vec4<f32>(0.0);
    let start_x = i32(x) - i32(radius);
    
    for (var k = 0u; k <= 2u * radius; k++) {
        let sample_x = clamp(start_x + i32(k), 0, i32(params.width) - 1);
        sum += textureLoad(input_texture, vec2<i32>(sample_x, i32(y)), 0) * kernel[k];
    }
    
    textureStore(intermediate_write, vec2<u32>(x, y), sum);
}

// Vertical pass (blur in Y direction)
@compute @workgroup_size(16, 16, 1)
fn vertical_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let radius = params.radius;
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    var sum = vec4<f32>(0.0);
    let start_y = i32(y) - i32(radius);
    
    for (var k = 0u; k <= 2u * radius; k++) {
        let sample_y = clamp(start_y + i32(k), 0, i32(params.height) - 1);
        sum += textureLoad(intermediate_read, vec2<i32>(i32(x), sample_y), 0) * kernel[k];
    }
    
    // Apply alpha preservation if needed
    var result = sum;
    if (params.blur_alpha == 0u) {
        let original = textureLoad(
            input_texture,
            vec2<i32>(i32(x), i32(y)),
            0
        );
        result.a = original.a;
    }
    
    textureStore(output_texture, vec2<u32>(x, y), clamp(result, vec4<f32>(0.0), vec4<f32>(1.0)));
}
