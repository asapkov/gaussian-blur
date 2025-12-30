// VariableGaussianBlur.wgsl
// Metal-style variable Gaussian blur for WebGPU

const PI: f32 = 3.141592653589793;
const HUGE_VALF: f32 = 1e20;

struct Uniforms {
    bounding_rect: vec4<f32>,    // x, y, width, height (16 bytes)
    radius: f32,                 // Maximum blur radius (4 bytes)
    max_samples: f32,           // Maximum samples per pixel (4 bytes)
    vertical: f32,              // 0.0 = X axis, 1.0 = Y axis (4 bytes)
    normalize_edges: f32,       // 0.0 = false, 1.0 = true (4 bytes)
    // Add explicit padding to make it 48 bytes total
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var mask_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

/**
 Formula of a gaussian function for single axis as described by
 https://en.wikipedia.org/wiki/Gaussian_blur .
 Creates a "bell curve" shape which we'll use for the weight of each sample when averaging.
 */
fn gaussian(distance: f32, sigma: f32) -> f32 {
    // Calculate the exponent of the Gaussian equation.
    let gaussian_exponent = -(distance * distance) / (2.0 * sigma * sigma);
    
    // Calculate and return the entire Gaussian equation.
    return (1.0 / (2.0 * PI * sigma * sigma)) * exp(gaussian_exponent);
}

/**
 Calculate pixel color using the weighted average of multiple samples along the X or Y axis.
 */
fn gaussian_blur_1d(
    position: vec2<u32>,
    texture: texture_2d<f32>,
    tex_size: vec2<u32>,
    bounding_rect: vec4<f32>,
    normalize_edges: bool,
    radius: f32,
    axis_multiplier: vec2<f32>,
    max_samples: f32
) -> vec4<f32> {
    // Calculate how far apart the samples should be
    let interval = max(1.0, radius / max_samples);
    
    // Take the first sample at the current position
    let weight = gaussian(0.0, radius / 3.0);
    
    var weighted_color_sum = textureLoad(texture, position, 0) * weight;
    var total_weight = weight;
    
    // If the radius is high enough to take more samples, take them
    if (interval <= radius) {
        // Set up bounding box for samples
        let min_bound = vec2<f32>(bounding_rect.x, bounding_rect.y);
        let max_bound = vec2<f32>(bounding_rect.x + bounding_rect.z, bounding_rect.y + bounding_rect.w);
        
        // WGSL uses select() for conditional assignment
        let min_sample_pos = select(vec2<f32>(-HUGE_VALF), min_bound, normalize_edges);
        let max_sample_pos = select(vec2<f32>(HUGE_VALF), max_bound, normalize_edges);
        
        let pos_f32 = vec2<f32>(position);
        
        // Take samples at intervals up to the blur radius
        var distance: f32 = interval;
        while (distance <= radius) {
            // Calculate the sample offset
            let offset_distance = axis_multiplier * distance;
            
            // Calculate the weight for this distance
            let weight = gaussian(distance, radius / 3.0);
            
            // Positive direction sample
            let positive_offset_pos = pos_f32 + offset_distance;
            let positive_pos_int = vec2<u32>(clamp(positive_offset_pos, vec2<f32>(0.0), vec2<f32>(tex_size) - vec2<f32>(1.0)));
            
            // Check if sample is within bounds
            if (!normalize_edges || all(positive_offset_pos <= max_sample_pos)) {
                weighted_color_sum += textureLoad(texture, positive_pos_int, 0) * weight;
                total_weight += weight;
            }
            
            // Negative direction sample
            let negative_offset_pos = pos_f32 - offset_distance;
            let negative_pos_int = vec2<u32>(clamp(negative_offset_pos, vec2<f32>(0.0), vec2<f32>(tex_size) - vec2<f32>(1.0)));
            
            // Check if sample is within bounds
            if (!normalize_edges || all(negative_offset_pos >= min_sample_pos)) {
                weighted_color_sum += textureLoad(texture, negative_pos_int, 0) * weight;
                total_weight += weight;
            }
            
            distance += interval;
        }
    }
    
    // Return the weighted average
    return weighted_color_sum / total_weight;
}

@compute @workgroup_size(16, 16, 1)
fn compute_blur(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check bounds
    let tex_size = textureDimensions(input_texture);
    if (global_id.x >= tex_size.x || global_id.y >= tex_size.y) {
        return;
    }
    
    let pos = global_id.xy;
    
    // Calculate UV within bounding rect
    let rect_uv = vec2<f32>(
        f32(global_id.x) / uniforms.bounding_rect.z,
        f32(global_id.y) / uniforms.bounding_rect.w
    );
    
    // Check if pixel is outside bounding rect when edge normalization is enabled
    if (uniforms.normalize_edges == 1.0 && 
        (rect_uv.x < 0.0 || rect_uv.x > 1.0 || rect_uv.y < 0.0 || rect_uv.y > 1.0)) {
        textureStore(output_texture, pos, vec4<f32>(0.0));
        return;
    }
    
    // Sample mask alpha at current position
    let mask_sample = textureLoad(mask_texture, pos, 0);
    let mask_alpha = mask_sample.r;  // Using R channel since mask is R32Float
    
    // Calculate pixel blur radius based on mask
    let pixel_radius = mask_alpha * uniforms.radius;
    
    // Apply blur if radius >= 1 pixel
    if (pixel_radius >= 1.0) {
        // Determine axis (X for horizontal pass, Y for vertical pass)
        let axis_multiplier = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), uniforms.vertical == 0.0);
        
        // Apply Gaussian blur
        let blurred_color = gaussian_blur_1d(
            pos,
            input_texture,
            tex_size,
            uniforms.bounding_rect,
            uniforms.normalize_edges == 1.0,
            pixel_radius,
            axis_multiplier,
            uniforms.max_samples
        );
        
        textureStore(output_texture, pos, blurred_color);
    } else {
        // Return original pixel if blur radius is less than 1
        let original_color = textureLoad(input_texture, pos, 0);
        textureStore(output_texture, pos, original_color);
    }
}
