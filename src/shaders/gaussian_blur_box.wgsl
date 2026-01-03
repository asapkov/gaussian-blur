// src/shaders/gaussian_blur_box.wgsl
// SEPARABLE BOX BLUR (Horizontal + Vertical passes)
// Much faster for large radii: O(2*radius) vs O(radius²)

struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    _padding0: u32,
    sigma: f32,
    current_pass: u32,      // 0,1,2 for 3-pass approximation
    blur_direction: u32,    // 0 = horizontal, 1 = vertical
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

@compute @workgroup_size(8, 8, 1)
fn box_blur_pass(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check bounds
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;
    
    let coord = vec2<f32>(f32(pixel_x), f32(pixel_y)) / vec2<f32>(f32(params.width), f32(params.height));
    
    let radius = i32(params.radius);
    let total_samples = f32(2 * radius + 1);
    let weight = 1.0 / total_samples;
    
    var sum = vec4<f32>(0.0);
    
    if (params.blur_direction == 0u) {
        // HORIZONTAL BLUR
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let offset_x = f32(dx) / f32(params.width);
            let sample_coord = vec2<f32>(coord.x + offset_x, coord.y);
            let clamped_coord = clamp(sample_coord, vec2(0.0), vec2(1.0));
            let texel_coord = vec2<i32>(floor(clamped_coord * vec2<f32>(f32(params.width), f32(params.height))));
            
            sum += textureLoad(input_tex, texel_coord, 0) * weight;
        }
    } else {
        // VERTICAL BLUR
        for (var dy = -radius; dy <= radius; dy = dy + 1) {
            let offset_y = f32(dy) / f32(params.height);
            let sample_coord = vec2<f32>(coord.x, coord.y + offset_y);
            let clamped_coord = clamp(sample_coord, vec2(0.0), vec2(1.0));
            let texel_coord = vec2<i32>(floor(clamped_coord * vec2<f32>(f32(params.width), f32(params.height))));
            
            sum += textureLoad(input_tex, texel_coord, 0) * weight;
        }
    }
    
    // Preserve alpha if needed
    if (params.blur_alpha == 0u) {
        let original_pixel = textureLoad(input_tex, vec2<i32>(i32(pixel_x), i32(pixel_y)), 0);
        sum.a = original_pixel.a;
    }
    
    // Clamp and store
    sum = clamp(sum, vec4(0.0), vec4(1.0));
    textureStore(output_tex, vec2<i32>(i32(pixel_x), i32(pixel_y)), sum);
    
    // Debug output
    if (pixel_x < 4u && pixel_y < 4u && params.current_pass < 3u) {
        let debug_idx = params.current_pass * 40u + params.blur_direction * 20u + (pixel_y * 4u + pixel_x) * 4u;
        if (debug_idx + 3u < 1024u) {
            debug_buffer[debug_idx + 0u] = sum.r;
            debug_buffer[debug_idx + 1u] = sum.g;
            debug_buffer[debug_idx + 2u] = sum.b;
            debug_buffer[debug_idx + 3u] = sum.a;
        }
    }
    
    // Store marker
    if (global_id.x == 0u && global_id.y == 0u) {
        debug_buffer[0u] = 1000.0 + f32(params.current_pass);
        debug_buffer[1u] = f32(params.width);
        debug_buffer[2u] = f32(params.height);
        debug_buffer[3u] = f32(params.radius);
        debug_buffer[4u] = f32(params.blur_alpha);
        debug_buffer[5u] = f32(params.blur_direction);
    }
}