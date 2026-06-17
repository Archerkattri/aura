// AURA native carrier CUDA entry points.
//
// These kernels define the production ABI target for replacing the current
// torch-autograd reference path. They are source scaffolds until wired through
// a compiled extension and validated against renderer/benchmark gates.

extern "C" __global__ void aura_surface_forward_kernel(
    const float* color,
    const float* opacity,
    const float* confidence,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    out_color[i * 3 + 0] = color[i * 3 + 0];
    out_color[i * 3 + 1] = color[i * 3 + 1];
    out_color[i * 3 + 2] = color[i * 3 + 2];
    out_transmittance[i] = fminf(fmaxf(1.0f - opacity[i], 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i], 0.0f), 1.0f);
    out_residual[i] = 0;
}

extern "C" __global__ void aura_volume_forward_kernel(
    const float* color,
    const float* density,
    const float* path_length,
    const float* confidence,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    out_color[i * 3 + 0] = color[i * 3 + 0];
    out_color[i * 3 + 1] = color[i * 3 + 1];
    out_color[i * 3 + 2] = color[i * 3 + 2];
    out_transmittance[i] = fminf(fmaxf(expf(-density[i] * fmaxf(path_length[i], 0.0f)), 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i], 0.0f), 1.0f);
    out_residual[i] = 0;
}

extern "C" __global__ void aura_beta_forward_kernel(
    const float* color,
    const float* opacity,
    const float* confidence,
    const float* alpha,
    const float* beta,
    const float* u,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float a = fmaxf(alpha[i], 1.0e-6f);
    const float b = fmaxf(beta[i], 1.0e-6f);
    const float x = fminf(fmaxf(u[i], 0.0f), 1.0f);
    const float support = powf(x, a - 1.0f) * powf(1.0f - x, b - 1.0f);
    out_color[i * 3 + 0] = color[i * 3 + 0];
    out_color[i * 3 + 1] = color[i * 3 + 1];
    out_color[i * 3 + 2] = color[i * 3 + 2];
    out_transmittance[i] = fminf(fmaxf(1.0f - opacity[i] * fminf(fmaxf(support, 0.0f), 1.0f), 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i], 0.0f), 1.0f);
    out_residual[i] = 0;
}

extern "C" __global__ void aura_gabor_forward_kernel(
    const float* color,
    const float* opacity,
    const float* confidence,
    const float* frequency,
    const float* phase,
    const float* bandwidth,
    const float* hit_point,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const float dot =
        hit_point[i * 3 + 0] * frequency[i * 3 + 0] +
        hit_point[i * 3 + 1] * frequency[i * 3 + 1] +
        hit_point[i * 3 + 2] * frequency[i * 3 + 2];
    const float bw = fminf(fmaxf(bandwidth[i], 0.0f), 1.0f);
    const float wave = 0.5f + 0.5f * sinf(6.28318530718f * dot + phase[i]);
    const float modulation = 1.0f - bw + bw * wave;
    out_color[i * 3 + 0] = fminf(fmaxf(color[i * 3 + 0] * modulation, 0.0f), 1.0f);
    out_color[i * 3 + 1] = fminf(fmaxf(color[i * 3 + 1] * modulation, 0.0f), 1.0f);
    out_color[i * 3 + 2] = fminf(fmaxf(color[i * 3 + 2] * modulation, 0.0f), 1.0f);
    out_transmittance[i] = fminf(fmaxf(1.0f - opacity[i], 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i] * bw, 0.0f), 1.0f);
    out_residual[i] = 0;
}

extern "C" __global__ void aura_neural_forward_kernel(
    const float* color,
    const float* opacity,
    const float* confidence,
    const float* residual_scale,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    out_color[i * 3 + 0] = color[i * 3 + 0];
    out_color[i * 3 + 1] = color[i * 3 + 1];
    out_color[i * 3 + 2] = color[i * 3 + 2];
    const float confidence_scale = fminf(fmaxf(1.0f - residual_scale[i] * 0.25f, 0.0f), 1.0f);
    out_transmittance[i] = fminf(fmaxf(1.0f - opacity[i], 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i] * confidence_scale, 0.0f), 1.0f);
    out_residual[i] = 1;
}

extern "C" __global__ void aura_semantic_forward_kernel(
    const float* color,
    const float* opacity,
    const float* confidence,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    out_color[i * 3 + 0] = color[i * 3 + 0];
    out_color[i * 3 + 1] = color[i * 3 + 1];
    out_color[i * 3 + 2] = color[i * 3 + 2];
    out_transmittance[i] = fminf(fmaxf(1.0f - opacity[i], 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i], 0.0f), 1.0f);
    out_residual[i] = 0;
}

extern "C" __global__ void aura_gaussian_forward_kernel(
    const float* color,
    const float* opacity,
    const float* confidence,
    float* out_color,
    float* out_transmittance,
    float* out_confidence,
    unsigned char* out_residual,
    int count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    out_color[i * 3 + 0] = color[i * 3 + 0];
    out_color[i * 3 + 1] = color[i * 3 + 1];
    out_color[i * 3 + 2] = color[i * 3 + 2];
    out_transmittance[i] = fminf(fmaxf(1.0f - opacity[i], 0.0f), 1.0f);
    out_confidence[i] = fminf(fmaxf(confidence[i], 0.0f), 1.0f);
    out_residual[i] = 0;
}

__device__ bool aura_ray_aabb_intersect(
    const float* origin,
    const float* direction,
    const float* box_min,
    const float* box_max,
    float* out_enter,
    float* out_exit,
    float* out_normal
) {
    float t_min = 0.0f;
    float t_max = 3.402823466e+38f;
    float normal_x = 0.0f;
    float normal_y = 0.0f;
    float normal_z = 0.0f;

    for (int axis = 0; axis < 3; ++axis) {
        const float ray_o = origin[axis];
        const float ray_d = direction[axis];
        const float b_min = box_min[axis];
        const float b_max = box_max[axis];
        if (fabsf(ray_d) < 1.0e-8f) {
            if (ray_o < b_min || ray_o > b_max) {
                return false;
            }
            continue;
        }
        float inv_d = 1.0f / ray_d;
        float t0 = (b_min - ray_o) * inv_d;
        float t1 = (b_max - ray_o) * inv_d;
        float sign = -1.0f;
        if (t0 > t1) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
            sign = 1.0f;
        }
        if (t0 > t_min) {
            t_min = t0;
            normal_x = 0.0f;
            normal_y = 0.0f;
            normal_z = 0.0f;
            if (axis == 0) {
                normal_x = sign;
            } else if (axis == 1) {
                normal_y = sign;
            } else {
                normal_z = sign;
            }
        }
        t_max = fminf(t_max, t1);
        if (t_min > t_max) {
            return false;
        }
    }

    if (t_max < 0.0f) {
        return false;
    }
    *out_enter = t_min;
    *out_exit = t_max;
    out_normal[0] = normal_x;
    out_normal[1] = normal_y;
    out_normal[2] = normal_z;
    return true;
}

extern "C" __global__ void aura_render_rays_kernel(
    const float* ray_origins,
    const float* ray_directions,
    const float* element_mins,
    const float* element_maxs,
    const int* carrier_ids,
    const float* colors,
    const float* opacities,
    const float* confidences,
    const int* material_ids,
    const int* semantic_ids,
    float* out_color,
    float* out_alpha,
    float* out_transmittance,
    float* out_depth,
    float* out_normal,
    float* out_confidence,
    unsigned char* out_residual,
    int* out_material_id,
    int* out_semantic_id,
    int* ordered_hits,
    int ray_count,
    int element_count,
    int max_hits
) {
    const int ray_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_i >= ray_count) {
        return;
    }

    const float* origin = ray_origins + ray_i * 3;
    const float* direction = ray_directions + ray_i * 3;
    float best_depth = 3.402823466e+38f;
    float best_exit = 3.402823466e+38f;
    int best_element = -1;
    float best_normal[3] = {0.0f, 0.0f, 0.0f};

    for (int hit_i = 0; hit_i < max_hits; ++hit_i) {
        ordered_hits[ray_i * max_hits + hit_i] = -1;
    }

    for (int element_i = 0; element_i < element_count; ++element_i) {
        float enter_depth = 0.0f;
        float exit_depth = 0.0f;
        float normal[3] = {0.0f, 0.0f, 0.0f};
        const bool hit = aura_ray_aabb_intersect(
            origin,
            direction,
            element_mins + element_i * 3,
            element_maxs + element_i * 3,
            &enter_depth,
            &exit_depth,
            normal
        );
        if (!hit) {
            continue;
        }
        if (enter_depth < best_depth) {
            best_depth = enter_depth;
            best_exit = exit_depth;
            best_element = element_i;
            best_normal[0] = normal[0];
            best_normal[1] = normal[1];
            best_normal[2] = normal[2];
        }
    }

    if (best_element < 0) {
        out_color[ray_i * 3 + 0] = 0.0f;
        out_color[ray_i * 3 + 1] = 0.0f;
        out_color[ray_i * 3 + 2] = 0.0f;
        out_alpha[ray_i] = 0.0f;
        out_transmittance[ray_i] = 1.0f;
        out_depth[ray_i] = 3.402823466e+38f;
        out_normal[ray_i * 3 + 0] = 0.0f;
        out_normal[ray_i * 3 + 1] = 0.0f;
        out_normal[ray_i * 3 + 2] = 0.0f;
        out_confidence[ray_i] = 0.0f;
        out_residual[ray_i] = 0;
        out_material_id[ray_i] = -1;
        out_semantic_id[ray_i] = -1;
        return;
    }

    const float opacity = fminf(fmaxf(opacities[best_element], 0.0f), 1.0f);
    out_color[ray_i * 3 + 0] = colors[best_element * 3 + 0];
    out_color[ray_i * 3 + 1] = colors[best_element * 3 + 1];
    out_color[ray_i * 3 + 2] = colors[best_element * 3 + 2];
    out_alpha[ray_i] = opacity;
    out_transmittance[ray_i] = fminf(fmaxf(1.0f - opacity, 0.0f), 1.0f);
    out_depth[ray_i] = best_depth;
    out_normal[ray_i * 3 + 0] = best_normal[0];
    out_normal[ray_i * 3 + 1] = best_normal[1];
    out_normal[ray_i * 3 + 2] = best_normal[2];
    out_confidence[ray_i] = fminf(fmaxf(confidences[best_element], 0.0f), 1.0f);
    out_residual[ray_i] = carrier_ids[best_element] == 4 ? 1 : 0;
    out_material_id[ray_i] = material_ids[best_element];
    out_semantic_id[ray_i] = semantic_ids[best_element];
    if (max_hits > 0) {
        ordered_hits[ray_i * max_hits] = best_element;
    }

    (void)best_exit;
}

extern "C" void aura_render_rays_launcher(
    const float* ray_origins,
    const float* ray_directions,
    const float* element_mins,
    const float* element_maxs,
    const int* carrier_ids,
    const float* colors,
    const float* opacities,
    const float* confidences,
    const int* material_ids,
    const int* semantic_ids,
    float* out_color,
    float* out_alpha,
    float* out_transmittance,
    float* out_depth,
    float* out_normal,
    float* out_confidence,
    unsigned char* out_residual,
    int* out_material_id,
    int* out_semantic_id,
    int* ordered_hits,
    int ray_count,
    int element_count,
    int max_hits,
    int threads_per_block
) {
    if (ray_count <= 0) {
        return;
    }
    int threads = threads_per_block;
    if (threads <= 0) {
        threads = 128;
    }
    if (threads > 1024) {
        threads = 1024;
    }
    const int block_count = (ray_count + threads - 1) / threads;
    aura_render_rays_kernel<<<block_count, threads>>>(
        ray_origins,
        ray_directions,
        element_mins,
        element_maxs,
        carrier_ids,
        colors,
        opacities,
        confidences,
        material_ids,
        semantic_ids,
        out_color,
        out_alpha,
        out_transmittance,
        out_depth,
        out_normal,
        out_confidence,
        out_residual,
        out_material_id,
        out_semantic_id,
        ordered_hits,
        ray_count,
        element_count,
        max_hits
    );
}
