// True separable Gaussian blur with precomputed weights and dithering
// Uses vec4<f32> array for proper 16-byte alignment

struct GaussianBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    sigma: f32,
};

struct GaussianWeights {
    weights: array<vec4<f32>, 256>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: GaussianBlurParams;

@group(0) @binding(3)
var<uniform> gaussian_weights: GaussianWeights;

// Simple pseudo-random hash for dithering
fn hash12(p: vec2<f32>) -> f32 {
    var h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    if params.direction == 0u {
        // HORIZONTAL BLUR
        let radius = i32(params.radius);
        var sum = vec4<f32>(0.0);
        var weight_sum = 0.0;

        for (var k = -radius; k <= radius; k++) {
            let sample_x = clamp(i32(x) + k, 0, i32(params.width) - 1);
            let weight_idx = u32(k + radius);

            let vec4_idx = weight_idx / 4u;
            let component_idx = weight_idx % 4u;
            let weight_vec = gaussian_weights.weights[vec4_idx];
            var weight: f32;

            if component_idx == 0u {
                weight = weight_vec.x;
            } else if component_idx == 1u {
                weight = weight_vec.y;
            } else if component_idx == 2u {
                weight = weight_vec.z;
            } else {
                weight = weight_vec.w;
            }

            sum += textureLoad(input_texture, vec2<i32>(sample_x, i32(y)), 0) * weight;
            weight_sum += weight;
        }

        var result: vec4<f32>;
        if weight_sum > 0.0 {
            result = sum / weight_sum;
        } else {
            result = vec4<f32>(0.0);
        }

        // Add subtle dithering to reduce banding
        let dither = hash12(vec2<f32>(f32(x), f32(y))) * 0.0039; // 1/256
        
        // Preserve alpha if needed
        if params.blur_alpha == 0u {
            let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            result = vec4<f32>(result.rgb + vec3<f32>(dither), original.a);
        } else {
            result = vec4<f32>(result.rgb + vec3<f32>(dither), result.a);
        }

        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
    } else {
        // VERTICAL BLUR
        let radius = i32(params.radius);
        var sum = vec4<f32>(0.0);
        var weight_sum = 0.0;

        for (var k = -radius; k <= radius; k++) {
            let sample_y = clamp(i32(y) + k, 0, i32(params.height) - 1);
            let weight_idx = u32(k + radius);

            let vec4_idx = weight_idx / 4u;
            let component_idx = weight_idx % 4u;
            let weight_vec = gaussian_weights.weights[vec4_idx];
            var weight: f32;

            if component_idx == 0u {
                weight = weight_vec.x;
            } else if component_idx == 1u {
                weight = weight_vec.y;
            } else if component_idx == 2u {
                weight = weight_vec.z;
            } else {
                weight = weight_vec.w;
            }

            sum += textureLoad(input_texture, vec2<i32>(i32(x), sample_y), 0) * weight;
            weight_sum += weight;
        }

        var result: vec4<f32>;
        if weight_sum > 0.0 {
            result = sum / weight_sum;
        } else {
            result = vec4<f32>(0.0);
        }

        // Add subtle dithering to reduce banding
        let dither = hash12(vec2<f32>(f32(x), f32(y))) * 0.0039; // 1/256
        
        // Preserve alpha if needed
        if params.blur_alpha == 0u {
            let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            result = vec4<f32>(result.rgb + vec3<f32>(dither), original.a);
        } else {
            result = vec4<f32>(result.rgb + vec3<f32>(dither), result.a);
        }

        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
    }
}