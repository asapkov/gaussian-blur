// VariableGaussianBlur.wgsl
// Metal-style variable Gaussian blur for WebGPU

const PI: f32 = 3.141592653589793;
const HUGE_VALF: f32 = 1e20;

struct Uniforms {
    bounding_rect: vec4<f32>,    // x, y, width, height
    radius: f32,                 // Maximum blur radius
    max_samples: f32,           // Maximum samples per pixel
    vertical: f32,              // 0.0 = X axis, 1.0 = Y axis
    normalize_edges: f32,       // 0.0 = false, 1.0 = true
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var mask_texture: texture_2d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

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
    tex_coord: vec2<f32>,
    texture: texture_2d<f32>,
    sampler: sampler,
    bounding_rect: vec4<f32>,
    normalize_edges: bool,
    radius: f32,
    axis_multiplier: vec2<f32>,
    max_samples: f32
) -> vec4<f32> {
    // Get texture dimensions
    let tex_size = textureDimensions(texture);
    
    // Calculate pixel position in texture space
    let position = tex_coord * vec2<f32>(tex_size);
    
    // Calculate how far apart the samples should be
    let interval = max(1.0, radius / max_samples);
    
    // Take the first sample at the current position
    let weight = gaussian(0.0, radius / 3.0);
    
    var weighted_color_sum = textureSample(texture, sampler, tex_coord) * weight;
    var total_weight = weight;
    
    // If the radius is high enough to take more samples, take them
    if (interval <= radius) {
        // Set up bounding box for samples
        let min_sample_pos = normalize_edges ? 
            vec2<f32>(bounding_rect.x, bounding_rect.y) : 
            vec2<f32>(-HUGE_VALF);
        let max_sample_pos = normalize_edges ? 
            vec2<f32>(bounding_rect.x + bounding_rect.z, bounding_rect.y + bounding_rect.w) : 
            vec2<f32>(HUGE_VALF);
        
        // Take samples at intervals up to the blur radius
        var distance: f32 = interval;
        while (distance <= radius) {
            // Calculate the sample offset
            let offset_distance = axis_multiplier * distance;
            
            // Calculate the weight for this distance
            let weight = gaussian(distance, radius / 3.0);
            
            // Positive direction sample
            let positive_offset_pos = position + offset_distance;
            let positive_uv = positive_offset_pos / vec2<f32>(tex_size);
            
            // Check if sample is within bounds
            if (!normalize_edges || all(positive_offset_pos <= max_sample_pos)) {
                weighted_color_sum += textureSample(texture, sampler, positive_uv) * weight;
                total_weight += weight;
            }
            
            // Negative direction sample
            let negative_offset_pos = position - offset_distance;
            let negative_uv = negative_offset_pos / vec2<f32>(tex_size);
            
            // Check if sample is within bounds
            if (!normalize_edges || all(negative_offset_pos >= min_sample_pos)) {
                weighted_color_sum += textureSample(texture, sampler, negative_uv) * weight;
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
    
    // Calculate UV coordinate
    let uv = vec2<f32>(global_id.xy) / vec2<f32>(tex_size);
    
    // Calculate UV within bounding rect
    let rect_uv = vec2<f32>(
        f32(global_id.x) / uniforms.bounding_rect.z,
        f32(global_id.y) / uniforms.bounding_rect.w
    );
    
    // Check if pixel is outside bounding rect when edge normalization is enabled
    if (uniforms.normalize_edges == 1.0 && 
        (rect_uv.x < 0.0 || rect_uv.x > 1.0 || rect_uv.y < 0.0 || rect_uv.y > 1.0)) {
        textureStore(output_texture, vec2<i32>(global_id.xy), vec4<f32>(0.0));
        return;
    }
    
    // Sample mask alpha at current position
    let mask_sample = textureSample(mask_texture, tex_sampler, uv);
    let mask_alpha = mask_sample.r;  // Using R channel since mask is R32Float
    
    // Calculate pixel blur radius based on mask
    let pixel_radius = mask_alpha * uniforms.radius;
    
    // Apply blur if radius >= 1 pixel
    if (pixel_radius >= 1.0) {
        // Determine axis (X for horizontal pass, Y for vertical pass)
        let axis_multiplier = uniforms.vertical == 0.0 ? 
            vec2<f32>(1.0, 0.0) : 
            vec2<f32>(0.0, 1.0);
        
        // Apply Gaussian blur
        let blurred_color = gaussian_blur_1d(
            uv,
            input_texture,
            tex_sampler,
            uniforms.bounding_rect,
            uniforms.normalize_edges == 1.0,
            pixel_radius,
            axis_multiplier,
            uniforms.max_samples
        );
        
        textureStore(output_texture, vec2<i32>(global_id.xy), blurred_color);
    } else {
        // Return original pixel if blur radius is less than 1
        let original_color = textureSample(input_texture, tex_sampler, uv);
        textureStore(output_texture, vec2<i32>(global_id.xy), original_color);
    }
}
