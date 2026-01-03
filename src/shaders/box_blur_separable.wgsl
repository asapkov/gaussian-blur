// Optimized separable box blur with shared memory tiling

struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,  // 0 = preserve alpha, 1 = blur alpha
    direction: u32,   // 0 = horizontal, 1 = vertical
    _padding: vec3<u32>,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: ShaderParameters;

// Tile size optimized for shared memory
const TILE_WIDTH = 256u;
const TILE_HEIGHT = 1u;  // For horizontal blur
const WORKGROUP_SIZE = 256u;

var<workgroup> tile: array<vec4<f32>, TILE_WIDTH>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>) {

    if params.direction == 0u {
        // HORIZONTAL BLUR
        let y = global_id.y;

        if y >= params.height {
            return;
        }
        
        // Each workgroup processes one row
        let tile_start_x = i32(global_id.x) * i32(TILE_WIDTH) - i32(params.radius);
        
        // Load tile into shared memory
        for (var i = 0u; i < TILE_WIDTH; i += WORKGROUP_SIZE) {
            let load_idx = local_id.x + i;
            if load_idx < TILE_WIDTH {
                let sample_x = clamp(tile_start_x + i32(load_idx), 0, i32(params.width) - 1);
                tile[load_idx] = textureLoad(input_texture, vec2<i32>(sample_x, i32(y)), 0);
            }
        }

        workgroupBarrier();
        
        // Process output pixels
        let output_x = global_id.x * TILE_WIDTH + local_id.x;

        if output_x < params.width {
            var sum = vec4<f32>(0.0);
            var count = 0.0;

            let radius = i32(params.radius);
            let tile_offset = local_id.x + u32(radius);

            for (var k = -radius; k <= radius; k++) {
                let tile_idx = i32(tile_offset) + k;
                if tile_idx >= 0 && tile_idx < i32(TILE_WIDTH) {
                    sum += tile[u32(tile_idx)];
                    count += 1.0;
                }
            }

            var result = sum / count;
            
            // Preserve alpha if needed
            if params.blur_alpha == 0u {
                let original_x = clamp(tile_start_x + i32(tile_offset), 0, i32(params.width) - 1);
                let original = textureLoad(input_texture, vec2<i32>(original_x, i32(y)), 0);
                result.a = original.a;
            }

            textureStore(output_texture, vec2<u32>(output_x, y), result);
        }
    } else {
        // VERTICAL BLUR (similar structure but transposed)
        let x = global_id.x;

        if x >= params.width {
            return;
        }
        
        // Each workgroup processes one column
        let tile_start_y = i32(global_id.y) * i32(TILE_WIDTH) - i32(params.radius);
        
        // Load tile into shared memory
        for (var i = 0u; i < TILE_WIDTH; i += WORKGROUP_SIZE) {
            let load_idx = local_id.x + i;
            if load_idx < TILE_WIDTH {
                let sample_y = clamp(tile_start_y + i32(load_idx), 0, i32(params.height) - 1);
                tile[load_idx] = textureLoad(input_texture, vec2<i32>(i32(x), sample_y), 0);
            }
        }

        workgroupBarrier();
        
        // Process output pixels
        let output_y = global_id.y * TILE_WIDTH + local_id.x;

        if output_y < params.height {
            var sum = vec4<f32>(0.0);
            var count = 0.0;

            let radius = i32(params.radius);
            let tile_offset = local_id.x + u32(radius);

            for (var k = -radius; k <= radius; k++) {
                let tile_idx = i32(tile_offset) + k;
                if tile_idx >= 0 && tile_idx < i32(TILE_WIDTH) {
                    sum += tile[u32(tile_idx)];
                    count += 1.0;
                }
            }

            var result = sum / count;
            
            // Preserve alpha if needed
            if params.blur_alpha == 0u {
                let original_y = clamp(tile_start_y + i32(tile_offset), 0, i32(params.height) - 1);
                let original = textureLoad(input_texture, vec2<i32>(i32(x), original_y), 0);
                result.a = original.a;
            }

            textureStore(output_texture, vec2<u32>(x, output_y), result);
        }
    }
}