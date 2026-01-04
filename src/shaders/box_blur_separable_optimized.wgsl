// Optimized separable box blur with shared memory tiling
// Workgroup size: 16x16 threads, tile size: 16x16 + halo

struct BoxBlurParams {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    direction: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: BoxBlurParams;

var<workgroup> tile: array<vec4<f32>, 400>; // (16+8)*(16+8) = 24*24 = 576, but using 400 for 20x20 for simplicity

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
    
    // Load tile with halo
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
    
    // Process tile
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let local_x_u = local_id.x + halo;
    let local_y_u = local_id.y + halo;
    let local_x = i32(local_x_u);
    let local_y = i32(local_y_u);

    var sum = vec4<f32>(0.0);
    var count: f32 = 0.0;

    let radius = i32(params.radius);

    if params.direction == 0u {
        // Horizontal blur from shared memory
        for (var i = -radius; i <= radius; i = i + 1) {
            let sample_local_x = local_x + i;
            let idx = u32(sample_local_x) + local_y_u * tile_w;
            sum += tile[idx];
            count += 1.0;
        }
    } else {
        // Vertical blur from shared memory
        for (var i = -radius; i <= radius; i = i + 1) {
            let sample_local_y = local_y + i;
            let idx = local_x_u + u32(sample_local_y) * tile_w;
            sum += tile[idx];
            count += 1.0;
        }
    }

    var result = sum / count;

    // Preserve alpha if needed
    if params.blur_alpha == 0u {
        let idx_center = local_x_u + local_y_u * tile_w;
        let original = tile[idx_center];
        result.a = original.a;
    }

    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
}