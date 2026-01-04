// Optimized Gaussian blur with shared memory tiling

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

var<workgroup> tile: array<vec4<f32>, 576>; // 24x24 tile for 16x16 workgroup + 8 pixel halo

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tile_width = 16u;
    let tile_height = 16u;
    let halo = params.radius;

    let base_x = workgroup_id.x * tile_width;
    let base_y = workgroup_id.y * tile_height;

    let local_idx = local_id.y * tile_width + local_id.x;
    
    // Load tile with halo into shared memory
    let tile_w = tile_width + 2u * halo;

    for (var i = 0u; i < tile_w * tile_w; i += 256u) {
        let load_idx = local_idx + i;
        if load_idx >= tile_w * tile_w {
            break;
        }

        let tile_x = load_idx % tile_w;
        let tile_y = load_idx / tile_w;

        let global_x = i32(base_x) + i32(tile_x) - i32(halo);
        let global_y = i32(base_y) + i32(tile_y) - i32(halo);

        let clamped_x = clamp(global_x, 0, i32(params.width) - 1);
        let clamped_y = clamp(global_y, 0, i32(params.height) - 1);

        tile[load_idx] = textureLoad(input_texture, vec2<i32>(clamped_x, clamped_y), 0);
    }

    workgroupBarrier();
    
    // Process from shared memory
    let x_u = global_id.x;
    let y_u = global_id.y;

    if x_u >= params.width || y_u >= params.height {
        return;
    }

    let x_i = i32(x_u);
    let y_i = i32(y_u);

    let local_x_u = local_id.x + halo;
    let local_y_u = local_id.y + halo;
    let local_x = i32(local_x_u);
    let local_y = i32(local_y_u);

    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    let radius_i = i32(params.radius);

    if params.direction == 0u {
        // Horizontal blur from shared memory
        for (var k = -radius_i; k <= radius_i; k = k + 1) {
            let sample_local_x = local_x + k;
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

            let idx = u32(sample_local_x) + local_y_u * tile_w;
            sum += tile[idx] * weight;
            weight_sum += weight;
        }
    } else {
        // Vertical blur from shared memory
        for (var k = -radius_i; k <= radius_i; k = k + 1) {
            let sample_local_y = local_y + k;
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

            let idx = local_x_u + u32(sample_local_y) * tile_w;
            sum += tile[idx] * weight;
            weight_sum += weight;
        }
    }

    var result: vec4<f32>;
    if weight_sum > 0.0 {
        result = sum / weight_sum;
    } else {
        result = vec4<f32>(0.0);
    }

    if params.blur_alpha == 0u {
        let idx_center = local_x_u + local_y_u * tile_w;
        let original = tile[idx_center];
        result.a = original.a;
    }

    textureStore(output_texture, vec2<i32>(x_i, y_i), result);
}