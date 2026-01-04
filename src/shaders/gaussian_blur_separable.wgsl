// True separable Gaussian blur with precomputed weights
// Uses vec4<f32> array for proper 16-byte alignment

struct GaussianBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,  // 0 = preserve alpha, 1 = blur alpha
    direction: u32,   // 0 = horizontal, 1 = vertical
    sigma: f32,
    // No padding needed - struct is already 28 bytes (4*5 + 4)
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

// Use array with stride 16 (vec4<f32>) to meet alignment requirements
// Max kernel size = 1024 weights (256 * 4) supporting radius up to 511
@group(0) @binding(3)
var<uniform> gaussian_weights: GaussianWeights;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    // CRITICAL: Check bounds
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

            // Extract weight from vec4 array (packed 4 weights per vec4)
            let vec4_idx = weight_idx / 4u;
            let component_idx = weight_idx % 4u;
            let weight_vec = gaussian_weights.weights[vec4_idx];
            var weight: f32;

            // Get the correct component
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

        // Preserve alpha if needed
        if params.blur_alpha == 0u {
            let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            result.a = original.a;
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

            // Extract weight from vec4 array (packed 4 weights per vec4)
            let vec4_idx = weight_idx / 4u;
            let component_idx = weight_idx % 4u;
            let weight_vec = gaussian_weights.weights[vec4_idx];
            var weight: f32;

            // Get the correct component
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

        // Preserve alpha if needed
        if params.blur_alpha == 0u {
            let original = textureLoad(input_texture, vec2<i32>(i32(x), i32(y)), 0);
            result.a = original.a;
        }

        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
    }
}