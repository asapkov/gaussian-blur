// Gaussian Blur with Multiple Strategies
// Supports: True Gaussian, 3-pass Box Blur, Downsample/Upsample

struct ShaderParameters {
    width: u32,
    height: u32,
    radius: u32,
    blur_alpha: u32,
    operation_mode: u32,    // 0 = box blur, 1 = downsample, 2 = upsample, 3 = gaussian blur
    sigma: f32,
    current_pass: u32,      // Which pass we're on (for multi-pass operations)
    src_width: u32,         // Source texture width (for downsample/upsample)
    src_height: u32,        // Source texture height
    dst_width: u32,         // Destination texture width
    dst_height: u32,        // Destination texture height
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
};

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> params: ShaderParameters;

@group(0) @binding(3)
var<storage, read_write> debug_buffer: array<f32, 1024>;

const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const TILE_SIZE_X = 4u;
const TILE_SIZE_Y = 4u;

fn texture_sample_normalized(tex: texture_2d<f32>, coords: vec2<i32>) -> vec4<f32> {
    return textureLoad(tex, coords, 0);
}

// True Gaussian blur for small sigmas (S ≤ 2.0)
fn apply_gaussian_blur(x: u32, y: u32, radius: u32, sigma: f32, tex: texture_2d<f32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;

    let iradius = i32(radius);
    let ix = i32(x);
    let iy = i32(y);
    
    // For even passes, do horizontal blur; for odd passes, do vertical blur
    if params.current_pass % 2u == 0u {
        // Horizontal blur with Gaussian weights
        for (var k = -iradius; k <= iradius; k++) {
            let sample_x = clamp(ix + k, 0, i32(params.width) - 1);
            let distance_sq = f32(k * k);
            let weight = exp(-distance_sq / (2.0 * sigma * sigma));

            let pixel = texture_sample_normalized(tex, vec2<i32>(sample_x, iy));
            sum += pixel * weight;
            weight_sum += weight;
        }
    } else {
        // Vertical blur with Gaussian weights
        for (var k = -iradius; k <= iradius; k++) {
            let sample_y = clamp(iy + k, 0, i32(params.height) - 1);
            let distance_sq = f32(k * k);
            let weight = exp(-distance_sq / (2.0 * sigma * sigma));

            let pixel = texture_sample_normalized(tex, vec2<i32>(ix, sample_y));
            sum += pixel * weight;
            weight_sum += weight;
        }
    }

    if weight_sum > 0.0 {
        return sum / weight_sum;
    }
    return vec4<f32>(0.0);
}

// Simple box blur - used for 3-pass approximation
fn apply_box_blur(x: u32, y: u32, radius: u32, tex: texture_2d<f32>) -> vec4<f32> {
    var sum = vec4<f32>(0.0);
    var count = 0.0;

    let iradius = i32(radius);
    let ix = i32(x);
    let iy = i32(y);

    // For even passes, do horizontal blur; for odd passes, do vertical blur
    if params.current_pass % 2u == 0u {
        // Horizontal blur
        for (var k = -iradius; k <= iradius; k++) {
            let sample_x = clamp(ix + k, 0, i32(params.width) - 1);
            let pixel = texture_sample_normalized(tex, vec2<i32>(sample_x, iy));
            sum += pixel;
            count += 1.0;
        }
    } else {
        // Vertical blur
        for (var k = -iradius; k <= iradius; k++) {
            let sample_y = clamp(iy + k, 0, i32(params.height) - 1);
            let pixel = texture_sample_normalized(tex, vec2<i32>(ix, sample_y));
            sum += pixel;
            count += 1.0;
        }
    }

    if count > 0.0 {
        return sum / count;
    }
    return vec4<f32>(0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Determine which operation we're performing
    if params.operation_mode == 1u {
        // DOWNSAMPLE operation
        let dst_x = global_id.x;
        let dst_y = global_id.y;

        if dst_x >= params.width || dst_y >= params.height {
            return;
        }
        
        // Calculate downscale factor (should be integer)
        let factor_x = params.src_width / params.width;
        let factor_y = params.src_height / params.height;
        
        // Sample from center of the block
        let src_x = dst_x * factor_x + factor_x / 2u;
        let src_y = dst_y * factor_y + factor_y / 2u;
        
        // Clamp to source bounds
        let clamped_src_x = min(src_x, params.src_width - 1u);
        let clamped_src_y = min(src_y, params.src_height - 1u);

        let sample = textureLoad(input_texture,
            vec2<i32>(i32(clamped_src_x), i32(clamped_src_y)), 0);

        textureStore(output_texture, vec2<u32>(dst_x, dst_y), sample);
        return;
    } else if params.operation_mode == 2u {
        // UPSAMPLE operation
        let dst_x = global_id.x;
        let dst_y = global_id.y;

        if dst_x >= params.dst_width || dst_y >= params.dst_height {
            return;
        }
        
        // Calculate upscale factor (should be integer)
        let factor_x = params.dst_width / params.src_width;
        let factor_y = params.dst_height / params.src_height;
        
        // Map back to source coordinates
        let src_x = dst_x / factor_x;
        let src_y = dst_y / factor_y;

        let clamped_src_x = min(src_x, params.src_width - 1u);
        let clamped_src_y = min(src_y, params.src_height - 1u);

        let sample = textureLoad(input_texture,
            vec2<i32>(i32(clamped_src_x), i32(clamped_src_y)), 0);

        textureStore(output_texture, vec2<u32>(dst_x, dst_y), sample);
        return;
    } else if params.operation_mode == 3u {
        // TRUE GAUSSIAN BLUR (for small sigmas)
        let tile_start_x = global_id.x * TILE_SIZE_X;
        let tile_start_y = global_id.y * TILE_SIZE_Y;

        for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
            let y = tile_start_y + dy;
            if y >= params.height { break; }

            for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
                let x = tile_start_x + dx;
                if x >= params.width { break; }

                var blurred = apply_gaussian_blur(x, y, params.radius, params.sigma, input_texture);
                
                // Preserve alpha if needed
                if params.blur_alpha == 0u {
                    let original = textureLoad(input_texture,
                        vec2<i32>(i32(x), i32(y)), 0);
                    blurred.a = original.a;
                }

                textureStore(output_texture, vec2<u32>(x, y), blurred);
            }
        }
    } else {
        // BOX BLUR (default operation, operation_mode == 0)
        let tile_start_x = global_id.x * TILE_SIZE_X;
        let tile_start_y = global_id.y * TILE_SIZE_Y;

        for (var dy = 0u; dy < TILE_SIZE_Y; dy++) {
            let y = tile_start_y + dy;
            if y >= params.height { break; }

            for (var dx = 0u; dx < TILE_SIZE_X; dx++) {
                let x = tile_start_x + dx;
                if x >= params.width { break; }

                var blurred = apply_box_blur(x, y, params.radius, input_texture);
                
                // Preserve alpha if needed
                if params.blur_alpha == 0u {
                    let original = textureLoad(input_texture,
                        vec2<i32>(i32(x), i32(y)), 0);
                    blurred.a = original.a;
                }

                textureStore(output_texture, vec2<u32>(x, y), blurred);
            }
        }
    }
}