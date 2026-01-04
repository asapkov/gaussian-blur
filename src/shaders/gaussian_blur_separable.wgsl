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
var output_texture: texture_storage_2d<rgba16float, write>;

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
    let x_u = global_id.x;
    let y_u = global_id.y;

    if x_u >= params.width || y_u >= params.height {
        return;
    }

    let x_i = i32(x_u);
    let y_i = i32(y_u);
    var result: vec4<f32>;

    if params.direction == 0u {
        // HORIZONTAL BLUR
        let radius_i = i32(params.radius);
        var sum = vec4<f32>(0.0);
        var weight_sum = 0.0;

        for (var k = -radius_i; k <= radius_i; k = k + 1) {
            let sample_x = clamp(x_i + k, 0, i32(params.width) - 1);
            let weight_idx = u32(k + radius_i);

            let vec4_idx = weight_idx / 4u;
            let component_idx = weight_idx % 4u;
            let weight_vec = gaussian_weights.weights[vec4_idx];
            var weight: f32;

            // WGSL doesn't have switch for non-integer types, use if-else
            if component_idx == 0u {
                weight = weight_vec.x;
            } else if component_idx == 1u {
                weight = weight_vec.y;
            } else if component_idx == 2u {
                weight = weight_vec.z;
            } else {
                weight = weight_vec.w;
            }

            sum += textureLoad(input_texture, vec2<i32>(sample_x, y_i), 0) * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            result = sum / weight_sum;
        } else {
            result = vec4<f32>(0.0);
        }
    } else {
        // VERTICAL BLUR
        let radius_i = i32(params.radius);
        var sum = vec4<f32>(0.0);
        var weight_sum = 0.0;

        for (var k = -radius_i; k <= radius_i; k = k + 1) {
            let sample_y = clamp(y_i + k, 0, i32(params.height) - 1);
            let weight_idx = u32(k + radius_i);

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

            sum += textureLoad(input_texture, vec2<i32>(x_i, sample_y), 0) * weight;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            result = sum / weight_sum;
        } else {
            result = vec4<f32>(0.0);
        }
    }

    // Add subtle dithering to reduce banding
    let dither = hash12(vec2<f32>(f32(x_u), f32(y_u))) * 0.0039; // 1/256
    
    // Preserve alpha if needed
    if params.blur_alpha == 0u {
        let original = textureLoad(input_texture, vec2<i32>(x_i, y_i), 0);
        result = vec4<f32>(result.rgb + vec3<f32>(dither), original.a);
    } else {
        result = vec4<f32>(result.rgb + vec3<f32>(dither), result.a);
    }

    textureStore(output_texture, vec2<i32>(x_i, y_i), result);
}