// Gaussian Blur Compute Shader - Optimized for performance

struct Parameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    sigma: f32,
    _padding: vec3<f32>, // Padding for 16-byte alignment
};

// Storage buffer for kernel - fastest access pattern
@group(0) @binding(2)
var<storage, read> kernel: array<f32>;

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var<uniform> params: Parameters;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Early exit for out-of-bounds
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let radius = i32(params.radius);
    let kernel_size = 2 * radius + 1;
    
    // Pre-fetch kernel weights to registers for faster access
    // (GPU will optimize this, but being explicit helps)
    var kernel_weights: array<f32, 64>;
    for (var i = 0u; i < u32(kernel_size); i++) {
        kernel_weights[i] = kernel[i];
    }
    
    // Horizontal pass with loop unrolling hint
    var sum_h = vec4<f32>(0.0);
    var kx = -radius;
    
    // Process in batches of 4 for better ILP
    while (kx <= radius) {
        let px0 = clamp(i32(x) + kx, 0, i32(params.width) - 1);
        let sample0 = textureLoad(input_texture, vec2<i32>(px0, i32(y)), 0);
        let weight0 = kernel_weights[u32(kx + radius)];
        sum_h += sample0 * weight0;
        
        // Unroll a few iterations (compiler will decide)
        if (kx + 1 <= radius) {
            let px1 = clamp(i32(x) + kx + 1, 0, i32(params.width) - 1);
            let sample1 = textureLoad(input_texture, vec2<i32>(px1, i32(y)), 0);
            let weight1 = kernel_weights[u32(kx + radius + 1)];
            sum_h += sample1 * weight1;
        }
        
        kx += 2;
    }
    
    // Vertical pass
    var sum_v = vec4<f32>(0.0);
    var ky = -radius;
    
    while (ky <= radius) {
        let py0 = clamp(i32(y) + ky, 0, i32(params.height) - 1);
        let sample0 = textureLoad(input_texture, vec2<i32>(i32(x), py0), 0);
        let weight0 = kernel_weights[u32(ky + radius)];
        sum_v += sample0 * weight0;
        
        if (ky + 1 <= radius) {
            let py1 = clamp(i32(y) + ky + 1, 0, i32(params.height) - 1);
            let sample1 = textureLoad(input_texture, vec2<i32>(i32(x), py1), 0);
            let weight1 = kernel_weights[u32(ky + radius + 1)];
            sum_v += sample1 * weight1;
        }
        
        ky += 2;
    }
    
    // Combine passes (separable Gaussian approximation)
    var result = (sum_h + sum_v) * 0.5;
    
    // Preserve alpha if needed
    if (params.blur_alpha == 0u) {
        let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
        result.a = original.a;
    }
    
    // Fast clamp using min/max
    result = max(result, vec4<f32>(0.0));
    result = min(result, vec4<f32>(1.0));
    
    textureStore(output_texture, vec2<u32>(x, y), result);
}
