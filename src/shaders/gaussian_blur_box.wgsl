// src/shaders/gaussian_blur_box.wgsl
// THREE-PASS BOX BLUR APPROXIMATION FOR GAUSSIAN BLUR
// Based on Central Limit Theorem (3 box blurs → Gaussian)

// MUST MATCH Rust's ShaderParameters struct layout exactly
struct ShaderParameters {
    // Must be in same order and type as Rust struct
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    _padding0: u32,
    sigma: f32,
    current_pass: u32,
    // Padding fields (must match Rust padding)
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    _padding10: f32,
    _padding11: f32,
    _padding12: f32,
    _padding13: f32,
    _padding14: f32,
    _padding15: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ShaderParameters;
@group(0) @binding(3) var<storage, read_write> debug_buffer: array<f32>;

fn box_blur_horizontal(coord: vec2<f32>, radius: u32) -> vec4<f32> {
    let image_size = vec2<f32>(f32(params.width), f32(params.height));
    let pixel_size = vec2(1.0) / image_size;
    
    // Convert to signed integer for loop bounds
    let radius_i = i32(radius);
    let total_samples = 2 * radius_i + 1;
    let weight = 1.0 / f32(total_samples);
    
    var sum = vec4<f32>(0.0);
    
    // Sample horizontally
    for (var i = -radius_i; i <= radius_i; i += 1) {
        let offset = f32(i);
        let sample_coord = coord + vec2(offset * pixel_size.x, 0.0);
        
        // Clamp coordinates to valid range
        let clamped_coord = clamp(sample_coord, vec2(0.0), vec2(1.0));
        let texel_coord = vec2<i32>(floor(clamped_coord * image_size));
        
        sum += textureLoad(input_tex, texel_coord, 0) * weight;
    }
    
    return sum;
}

fn box_blur_vertical(coord: vec2<f32>, radius: u32) -> vec4<f32> {
    let image_size = vec2<f32>(f32(params.width), f32(params.height));
    let pixel_size = vec2(1.0) / image_size;
    
    // Convert to signed integer for loop bounds
    let radius_i = i32(radius);
    let total_samples = 2 * radius_i + 1;
    let weight = 1.0 / f32(total_samples);
    
    var sum = vec4<f32>(0.0);
    
    // Sample vertically
    for (var i = -radius_i; i <= radius_i; i += 1) {
        let offset = f32(i);
        let sample_coord = coord + vec2(0.0, offset * pixel_size.y);
        
        // Clamp coordinates to valid range
        let clamped_coord = clamp(sample_coord, vec2(0.0), vec2(1.0));
        let texel_coord = vec2<i32>(floor(clamped_coord * image_size));
        
        sum += textureLoad(input_tex, texel_coord, 0) * weight;
    }
    
    return sum;
}

@compute @workgroup_size(8, 8, 1)
fn box_blur_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check bounds
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    
    let image_size = vec2<f32>(f32(params.width), f32(params.height));
    let coord = vec2<f32>(global_id.xy) / image_size;
    let radius = params.radius;
    
    var result: vec4<f32>;
    
    // For box blur approximation, each pass applies both horizontal and vertical
    // In the 3-pass approach, we alternate between horizontal and vertical
    // For simplicity, we'll do both in each shader invocation
    let temp_horizontal = box_blur_horizontal(coord, radius);
    result = box_blur_vertical(coord, radius);
    
    // For debugging: store some values
    if (global_id.x < 4u && global_id.y < 4u && params.current_pass < 3u) {
        let debug_idx = params.current_pass * 20u + (global_id.y * 4u + global_id.x) * 4u;
        debug_buffer[debug_idx + 0u] = result.r;
        debug_buffer[debug_idx + 1u] = result.g;
        debug_buffer[debug_idx + 2u] = result.b;
        debug_buffer[debug_idx + 3u] = result.a;
    }
    
    // Store marker
    if (global_id.x == 0u && global_id.y == 0u) {
        debug_buffer[0u] = 1000.0 + f32(params.current_pass);
        debug_buffer[1u] = f32(params.width);
        debug_buffer[2u] = f32(params.height);
        debug_buffer[3u] = f32(params.radius);
        debug_buffer[4u] = f32(params.blur_alpha);
    }
    
    // Clamp to valid range for storage texture
    result = clamp(result, vec4(0.0), vec4(1.0));
    
    // Write to output texture (RGBA8Unorm expects values 0-1)
    textureStore(output_tex, vec2<i32>(global_id.xy), result);
}