// Gaussian Blur Compute Shader - Optimized work distribution
// Two-pass separable Gaussian blur with tiling

struct Parameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    sigma: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
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

// Constants - tuned for performance
const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TILE_SIZE_X = 4u;   // Each thread processes 4 pixels horizontally
const TILE_SIZE_Y = 4u;   // Each thread processes 4 pixels vertically

// Horizontal pass - each thread processes TILE_SIZE_X pixels
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn horizontal_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let radius = params.radius;
    
    // Calculate which tile of pixels this thread processes
    let tile_start_x = global_id.x * TILE_SIZE_X;
    let tile_start_y = global_id.y * TILE_SIZE_Y;
    
    // Process TILE_SIZE_X × TILE_SIZE_Y pixels
    for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
        let y = tile_start_y + dy;
        if (y >= params.height) {
            break;
        }
        
        for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
            let x = tile_start_x + dx;
            if (x >= params.width) {
                break;
            }
            
            // Apply horizontal blur
            var sum = vec4<f32>(0.0);
            let start_x = i32(x) - i32(radius);
            
            for (var k = 0u; k <= 2u * radius; k++) {
                let sample_x = clamp(start_x + i32(k), 0, i32(params.width) - 1);
                sum += textureLoad(input_texture, vec2<i32>(sample_x, i32(y)), 0) * kernel[k];
            }
            
            textureStore(intermediate_write, vec2<u32>(x, y), sum);
        }
    }
}

// Vertical pass - each thread processes TILE_SIZE_X pixels
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn vertical_pass(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let radius = params.radius;
    
    // Calculate which tile of pixels this thread processes
    let tile_start_x = global_id.x * TILE_SIZE_X;
    let tile_start_y = global_id.y * TILE_SIZE_Y;
    
    // Process TILE_SIZE_X × TILE_SIZE_Y pixels
    for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
        let y = tile_start_y + dy;
        if (y >= params.height) {
            break;
        }
        
        for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
            let x = tile_start_x + dx;
            if (x >= params.width) {
                break;
            }
            
            // Apply vertical blur
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
    }
}
