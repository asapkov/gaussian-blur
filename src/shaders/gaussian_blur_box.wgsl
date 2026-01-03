// SEPARABLE BOX BLUR WITH DOWNSAMPLING SUPPORT - FIXED VERSION

struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    _padding0: u32,
    sigma: f32,
    current_pass: u32,
    blur_direction: u32,    // 0 = horizontal, 1 = vertical
    operation_mode: u32,    // 0 = blur, 1 = downsample, 2 = upsample
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
};

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: ShaderParameters;
@group(0) @binding(3) var<storage, read_write> debug_buffer: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // COMMON LOGIC FOR ALL OPERATIONS
    // Determine which pixel we're writing to based on operation mode
    var write_x: u32;
    var write_y: u32;
    var max_x: u32;
    var max_y: u32;

    if params.operation_mode == 1u {
        // DOWNSAMPLE: writing to downsampled texture
        write_x = global_id.x;
        write_y = global_id.y;
        max_x = params.width;
        max_y = params.height;
    } else if params.operation_mode == 2u {
        // UPSAMPLE: writing to full-size texture
        write_x = global_id.x;
        write_y = global_id.y;
        max_x = params.dst_width;
        max_y = params.dst_height;
    } else {
        // BLUR: writing to current texture
        write_x = global_id.x;
        write_y = global_id.y;
        max_x = params.width;
        max_y = params.height;
    }
    
    // Check bounds
    if write_x >= max_x || write_y >= max_y {
        return;
    }
    
    // DOWNSAMPLE
    if params.operation_mode == 1u {
        // Calculate source coordinate in original texture
        let src_x = write_x * 8u + 4u;  // Center of 8x8 block
        let src_y = write_y * 8u + 4u;
        
        // Convert to normalized texture coordinates
        let uv_x = f32(src_x) / f32(params.src_width);
        let uv_y = f32(src_y) / f32(params.src_height);
        
        // Convert to texel coordinates
        let texel_x = u32(uv_x * f32(params.src_width));
        let texel_y = u32(uv_y * f32(params.src_height));
        
        // Clamp to texture bounds
        let clamped_x = min(texel_x, params.src_width - 1u);
        let clamped_y = min(texel_y, params.src_height - 1u);

        let sample_color = textureLoad(input_tex, vec2<i32>(i32(clamped_x), i32(clamped_y)), 0);
        textureStore(output_tex, vec2<i32>(i32(write_x), i32(write_y)), sample_color);
        return;
    }
    
    // UPSAMPLE
    if params.operation_mode == 2u {
        // Calculate source coordinate in downsampled texture
        let src_x = write_x / 8u;
        let src_y = write_y / 8u;
        
        // Convert to normalized texture coordinates in source
        let uv_x = f32(src_x) / f32(params.src_width);
        let uv_y = f32(src_y) / f32(params.src_height);
        
        // Convert to texel coordinates
        let texel_x = u32(uv_x * f32(params.src_width));
        let texel_y = u32(uv_y * f32(params.src_height));
        
        // Clamp to source texture bounds
        let clamped_x = min(texel_x, params.src_width - 1u);
        let clamped_y = min(texel_y, params.src_height - 1u);

        let sample_color = textureLoad(input_tex, vec2<i32>(i32(clamped_x), i32(clamped_y)), 0);
        textureStore(output_tex, vec2<i32>(i32(write_x), i32(write_y)), sample_color);
        return;
    }
    
    // BOX BLUR
    let radius = i32(params.radius);
    let total_samples = f32(2 * radius + 1);
    let weight = 1.0 / total_samples;

    var sum = vec4<f32>(0.0);
    var valid_samples: f32 = 0.0;

    if params.blur_direction == 0u {
        // HORIZONTAL BLUR
        for (var dx = -radius; dx <= radius; dx = dx + 1) {
            let sample_x = i32(write_x) + dx;
            if sample_x >= 0 && sample_x < i32(params.width) {
                let texel_coord = vec2<i32>(sample_x, i32(write_y));
                sum += textureLoad(input_tex, texel_coord, 0);
                valid_samples += 1.0;
            }
        }
    } else {
        // VERTICAL BLUR
        for (var dy = -radius; dy <= radius; dy = dy + 1) {
            let sample_y = i32(write_y) + dy;
            if sample_y >= 0 && sample_y < i32(params.height) {
                let texel_coord = vec2<i32>(i32(write_x), sample_y);
                sum += textureLoad(input_tex, texel_coord, 0);
                valid_samples += 1.0;
            }
        }
    }
    
    // Average with actual valid samples count
    var avg: vec4<f32>;
    if valid_samples > 0.0 {
        avg = sum / valid_samples;
    } else {
        avg = vec4<f32>(0.0);
    }
    
    // Preserve alpha if needed
    var final_color: vec4<f32>;
    if params.blur_alpha == 0u {
        let original_texel = vec2<i32>(i32(write_x), i32(write_y));
        let original = textureLoad(input_tex, original_texel, 0);
        final_color = vec4<f32>(avg.rgb, original.a);
    } else {
        final_color = avg;
    }
    
    // Clamp and store
    textureStore(output_tex, vec2<i32>(i32(write_x), i32(write_y)), clamp(final_color, vec4(0.0), vec4(1.0)));
}