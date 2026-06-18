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

__device__ bool aura_ray_plane_intersect(
    const float* origin,
    const float* direction,
    const float* box_min,
    const float* box_max,
    const float* plane_point,
    const float* normal,
    float* out_depth,
    float* out_normal
) {
    if (
        !isfinite(plane_point[0]) || !isfinite(plane_point[1]) || !isfinite(plane_point[2]) ||
        !isfinite(normal[0]) || !isfinite(normal[1]) || !isfinite(normal[2])
    ) {
        return false;
    }
    const float denom = direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
    if (fabsf(denom) < 1.0e-8f) {
        return false;
    }
    const float depth = (
        (plane_point[0] - origin[0]) * normal[0] +
        (plane_point[1] - origin[1]) * normal[1] +
        (plane_point[2] - origin[2]) * normal[2]
    ) / denom;
    if (depth < 0.0f) {
        return false;
    }
    const float point[3] = {
        origin[0] + direction[0] * depth,
        origin[1] + direction[1] * depth,
        origin[2] + direction[2] * depth,
    };
    for (int axis = 0; axis < 3; ++axis) {
        if (point[axis] < box_min[axis] - 1.0e-5f || point[axis] > box_max[axis] + 1.0e-5f) {
            return false;
        }
    }
    *out_depth = depth;
    out_normal[0] = normal[0];
    out_normal[1] = normal[1];
    out_normal[2] = normal[2];
    return true;
}

__device__ float aura_clamp_unit(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ bool aura_beta_radii_valid(const float* radii) {
    return isfinite(radii[0]) && isfinite(radii[1]) && isfinite(radii[2]) && radii[0] > 0.0f && radii[1] > 0.0f && radii[2] > 0.0f;
}

__device__ bool aura_gaussian_geometry_valid(
    const float* mean,
    const float* inverse_covariance,
    float support_radius_sq
) {
    if (!isfinite(support_radius_sq) || support_radius_sq <= 0.0f) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(mean[axis])) {
            return false;
        }
    }
    for (int index = 0; index < 9; ++index) {
        if (!isfinite(inverse_covariance[index])) {
            return false;
        }
    }
    return true;
}

__device__ void aura_matvec3(const float* matrix, const float* vector, float* out) {
    out[0] = matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
    out[1] = matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
    out[2] = matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
}

__device__ void aura_gaussian_ellipsoid_normal(
    const float* origin,
    const float* direction,
    float depth,
    const float* mean,
    const float* inverse_covariance,
    float* out_normal
) {
    const float delta[3] = {
        origin[0] + direction[0] * depth - mean[0],
        origin[1] + direction[1] * depth - mean[1],
        origin[2] + direction[2] * depth - mean[2],
    };
    float gradient[3] = {0.0f, 0.0f, 0.0f};
    aura_matvec3(inverse_covariance, delta, gradient);
    const float norm = sqrtf(gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2]);
    if (norm <= 1.0e-12f) {
        out_normal[0] = 0.0f;
        out_normal[1] = 0.0f;
        out_normal[2] = 0.0f;
        return;
    }
    out_normal[0] = gradient[0] / norm;
    out_normal[1] = gradient[1] / norm;
    out_normal[2] = gradient[2] / norm;
}

__device__ bool aura_ray_gaussian_ellipsoid_intersect(
    const float* origin,
    const float* direction,
    const float* mean,
    const float* inverse_covariance,
    float support_radius_sq,
    float* out_enter,
    float* out_exit,
    float* out_normal
) {
    if (!aura_gaussian_geometry_valid(mean, inverse_covariance, support_radius_sq)) {
        return false;
    }
    const float delta[3] = {
        origin[0] - mean[0],
        origin[1] - mean[1],
        origin[2] - mean[2],
    };
    float inv_direction[3] = {0.0f, 0.0f, 0.0f};
    float inv_delta[3] = {0.0f, 0.0f, 0.0f};
    aura_matvec3(inverse_covariance, direction, inv_direction);
    aura_matvec3(inverse_covariance, delta, inv_delta);
    const float a = inv_direction[0] * direction[0] + inv_direction[1] * direction[1] + inv_direction[2] * direction[2];
    const float b = 2.0f * (inv_delta[0] * direction[0] + inv_delta[1] * direction[1] + inv_delta[2] * direction[2]);
    const float c = inv_delta[0] * delta[0] + inv_delta[1] * delta[1] + inv_delta[2] * delta[2] - support_radius_sq;
    const float discriminant = b * b - 4.0f * a * c;
    if (a <= 1.0e-8f || discriminant < 0.0f) {
        return false;
    }
    const float root = sqrtf(fmaxf(discriminant, 0.0f));
    const float denom = 2.0f * a;
    const float near_depth = (-b - root) / denom;
    const float far_depth = (-b + root) / denom;
    const float entry = near_depth >= 0.0f ? near_depth : 0.0f;
    if (far_depth < 0.0f || entry < 0.0f) {
        return false;
    }
    *out_enter = entry;
    *out_exit = fmaxf(far_depth, 0.0f);
    aura_gaussian_ellipsoid_normal(origin, direction, entry, mean, inverse_covariance, out_normal);
    return true;
}

__device__ void aura_beta_ellipsoid_normal(
    const float* origin,
    const float* direction,
    float depth,
    const float* center,
    const float* radii,
    float* out_normal
) {
    const float point[3] = {
        origin[0] + direction[0] * depth,
        origin[1] + direction[1] * depth,
        origin[2] + direction[2] * depth,
    };
    const float gx = (point[0] - center[0]) / fmaxf(radii[0] * radii[0], 1.0e-12f);
    const float gy = (point[1] - center[1]) / fmaxf(radii[1] * radii[1], 1.0e-12f);
    const float gz = (point[2] - center[2]) / fmaxf(radii[2] * radii[2], 1.0e-12f);
    const float norm = sqrtf(gx * gx + gy * gy + gz * gz);
    if (norm <= 1.0e-12f) {
        out_normal[0] = 0.0f;
        out_normal[1] = 0.0f;
        out_normal[2] = 0.0f;
        return;
    }
    out_normal[0] = gx / norm;
    out_normal[1] = gy / norm;
    out_normal[2] = gz / norm;
}

__device__ bool aura_ray_beta_ellipsoid_intersect(
    const float* origin,
    const float* direction,
    const float* box_min,
    const float* box_max,
    const float* radii,
    float* out_enter,
    float* out_exit,
    float* out_normal
) {
    if (!aura_beta_radii_valid(radii)) {
        return false;
    }
    const float center[3] = {
        (box_min[0] + box_max[0]) * 0.5f,
        (box_min[1] + box_max[1]) * 0.5f,
        (box_min[2] + box_max[2]) * 0.5f,
    };
    const float scaled_origin[3] = {
        (origin[0] - center[0]) / radii[0],
        (origin[1] - center[1]) / radii[1],
        (origin[2] - center[2]) / radii[2],
    };
    const float scaled_direction[3] = {
        direction[0] / radii[0],
        direction[1] / radii[1],
        direction[2] / radii[2],
    };
    const float a = (
        scaled_direction[0] * scaled_direction[0] +
        scaled_direction[1] * scaled_direction[1] +
        scaled_direction[2] * scaled_direction[2]
    );
    const float b = 2.0f * (
        scaled_origin[0] * scaled_direction[0] +
        scaled_origin[1] * scaled_direction[1] +
        scaled_origin[2] * scaled_direction[2]
    );
    const float c = (
        scaled_origin[0] * scaled_origin[0] +
        scaled_origin[1] * scaled_origin[1] +
        scaled_origin[2] * scaled_origin[2] - 1.0f
    );
    const float discriminant = b * b - 4.0f * a * c;
    if (a <= 1.0e-8f || discriminant < 0.0f) {
        return false;
    }
    const float root = sqrtf(fmaxf(discriminant, 0.0f));
    const float denom = 2.0f * a;
    const float near_depth = (-b - root) / denom;
    const float far_depth = (-b + root) / denom;
    const float entry = fmaxf(fminf(near_depth, far_depth), 0.0f);
    const float exit_depth = fmaxf(near_depth, far_depth);
    if (exit_depth < entry) {
        return false;
    }
    *out_enter = entry;
    *out_exit = exit_depth;
    aura_beta_ellipsoid_normal(origin, direction, entry, center, radii, out_normal);
    return true;
}

__device__ float aura_beta_support(
    const float* point,
    const float* box_min,
    const float* box_max,
    const float* radii,
    float alpha,
    float beta
) {
    const float safe_alpha = fmaxf(alpha, 1.0e-6f);
    const float safe_beta = fmaxf(beta, 1.0e-6f);
    const float center[3] = {
        (box_min[0] + box_max[0]) * 0.5f,
        (box_min[1] + box_max[1]) * 0.5f,
        (box_min[2] + box_max[2]) * 0.5f,
    };
    float u = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        const float radius = fmaxf(radii[axis], 1.0e-6f);
        u += aura_clamp_unit(1.0f - fabsf(point[axis] - center[axis]) / radius);
    }
    u /= 3.0f;
    float raw = powf(u, safe_alpha - 1.0f) * powf(1.0f - u, safe_beta - 1.0f);
    if (safe_alpha > 1.0f && safe_beta > 1.0f) {
        const float mode = aura_clamp_unit((safe_alpha - 1.0f) / fmaxf(safe_alpha + safe_beta - 2.0f, 1.0e-6f));
        const float peak = powf(mode, safe_alpha - 1.0f) * powf(1.0f - mode, safe_beta - 1.0f);
        if (peak > 0.0f) {
            raw /= peak;
        }
    }
    return aura_clamp_unit(raw);
}

extern "C" __global__ void aura_render_rays_kernel(
    const float* ray_origins,
    const float* ray_directions,
    const float* element_mins,
    const float* element_maxs,
    const float* plane_points,
    const float* plane_normals,
    const float* beta_support_radii,
    const float* gaussian_means,
    const float* gaussian_inverse_covariances,
    const float* gaussian_support_radius_sq,
    const int* carrier_ids,
    const float* colors,
    const float* opacities,
    const float* confidences,
    const float* payload_params,
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
    const int AURA_RENDER_MAX_ORDERED_HITS = 64;
    const int ray_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_i >= ray_count) {
        return;
    }

    const float* origin = ray_origins + ray_i * 3;
    const float* direction = ray_directions + ray_i * 3;
    const int stored_hit_limit = max_hits < AURA_RENDER_MAX_ORDERED_HITS ? max_hits : AURA_RENDER_MAX_ORDERED_HITS;
    float hit_depths[AURA_RENDER_MAX_ORDERED_HITS];
    float hit_exits[AURA_RENDER_MAX_ORDERED_HITS];
    float hit_normals[AURA_RENDER_MAX_ORDERED_HITS * 3];
    int hit_elements[AURA_RENDER_MAX_ORDERED_HITS];
    int stored_hit_count = 0;

    for (int hit_i = 0; hit_i < max_hits; ++hit_i) {
        ordered_hits[ray_i * max_hits + hit_i] = -1;
    }
    for (int hit_i = 0; hit_i < AURA_RENDER_MAX_ORDERED_HITS; ++hit_i) {
        hit_depths[hit_i] = 3.402823466e+38f;
        hit_exits[hit_i] = 3.402823466e+38f;
        hit_elements[hit_i] = -1;
        hit_normals[hit_i * 3 + 0] = 0.0f;
        hit_normals[hit_i * 3 + 1] = 0.0f;
        hit_normals[hit_i * 3 + 2] = 0.0f;
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
        if (carrier_ids[element_i] == 0 || carrier_ids[element_i] == 3) {
            const bool plane_hit = aura_ray_plane_intersect(
                origin,
                direction,
                element_mins + element_i * 3,
                element_maxs + element_i * 3,
                plane_points + element_i * 3,
                plane_normals + element_i * 3,
                &enter_depth,
                normal
            );
            if (plane_hit) {
                exit_depth = enter_depth;
            } else if (!hit) {
                continue;
            }
        } else if (carrier_ids[element_i] == 2) {
            const float* beta_radii = beta_support_radii + element_i * 3;
            float beta_enter = 0.0f;
            float beta_exit = 0.0f;
            float beta_normal[3] = {0.0f, 0.0f, 0.0f};
            const bool beta_hit = aura_ray_beta_ellipsoid_intersect(
                origin,
                direction,
                element_mins + element_i * 3,
                element_maxs + element_i * 3,
                beta_radii,
                &beta_enter,
                &beta_exit,
                beta_normal
            );
            if (beta_hit && hit) {
                if (beta_enter > enter_depth) {
                    enter_depth = beta_enter;
                    normal[0] = beta_normal[0];
                    normal[1] = beta_normal[1];
                    normal[2] = beta_normal[2];
                }
                exit_depth = fminf(exit_depth, beta_exit);
                if (exit_depth < enter_depth) {
                    continue;
                }
            } else if (aura_beta_radii_valid(beta_radii)) {
                continue;
            } else if (!hit) {
                continue;
            }
        } else if (carrier_ids[element_i] == 6) {
            const float* gaussian_mean = gaussian_means + element_i * 3;
            const float* gaussian_inverse_covariance = gaussian_inverse_covariances + element_i * 9;
            const float gaussian_support = gaussian_support_radius_sq[element_i];
            float gaussian_enter = 0.0f;
            float gaussian_exit = 0.0f;
            float gaussian_normal[3] = {0.0f, 0.0f, 0.0f};
            const bool gaussian_hit = aura_ray_gaussian_ellipsoid_intersect(
                origin,
                direction,
                gaussian_mean,
                gaussian_inverse_covariance,
                gaussian_support,
                &gaussian_enter,
                &gaussian_exit,
                gaussian_normal
            );
            if (gaussian_hit && hit) {
                if (gaussian_enter > enter_depth) {
                    enter_depth = gaussian_enter;
                    normal[0] = gaussian_normal[0];
                    normal[1] = gaussian_normal[1];
                    normal[2] = gaussian_normal[2];
                }
                exit_depth = fminf(exit_depth, gaussian_exit);
                if (exit_depth < enter_depth) {
                    continue;
                }
            } else if (aura_gaussian_geometry_valid(gaussian_mean, gaussian_inverse_covariance, gaussian_support)) {
                continue;
            } else if (!hit) {
                continue;
            }
        } else if (!hit) {
            continue;
        }
        if (stored_hit_limit <= 0) {
            continue;
        }
        int insert_at = stored_hit_count;
        while (insert_at > 0 && enter_depth < hit_depths[insert_at - 1]) {
            --insert_at;
        }
        if (insert_at >= stored_hit_limit) {
            continue;
        }
        const int last = stored_hit_count < stored_hit_limit ? stored_hit_count : stored_hit_limit - 1;
        for (int move_i = last; move_i > insert_at; --move_i) {
            hit_depths[move_i] = hit_depths[move_i - 1];
            hit_exits[move_i] = hit_exits[move_i - 1];
            hit_elements[move_i] = hit_elements[move_i - 1];
            hit_normals[move_i * 3 + 0] = hit_normals[(move_i - 1) * 3 + 0];
            hit_normals[move_i * 3 + 1] = hit_normals[(move_i - 1) * 3 + 1];
            hit_normals[move_i * 3 + 2] = hit_normals[(move_i - 1) * 3 + 2];
        }
        hit_depths[insert_at] = enter_depth;
        hit_exits[insert_at] = exit_depth;
        hit_elements[insert_at] = element_i;
        hit_normals[insert_at * 3 + 0] = normal[0];
        hit_normals[insert_at * 3 + 1] = normal[1];
        hit_normals[insert_at * 3 + 2] = normal[2];
        if (stored_hit_count < stored_hit_limit) {
            ++stored_hit_count;
        }
    }

    if (stored_hit_count <= 0) {
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

    float color_r = 0.0f;
    float color_g = 0.0f;
    float color_b = 0.0f;
    float remaining = 1.0f;
    float confidence_num = 0.0f;
    float confidence_den = 0.0f;
    unsigned char residual = 0;

    for (int hit_i = 0; hit_i < stored_hit_count; ++hit_i) {
        const int element_i = hit_elements[hit_i];
        const int carrier_id = carrier_ids[element_i];
        const float opacity = aura_clamp_unit(opacities[element_i]);
        const float* payload = payload_params + element_i * 5;
        float transmittance = aura_clamp_unit(1.0f - opacity);
        float confidence_value = aura_clamp_unit(confidences[element_i]);
        float color_r_hit = colors[element_i * 3 + 0];
        float color_g_hit = colors[element_i * 3 + 1];
        float color_b_hit = colors[element_i * 3 + 2];
        if (carrier_id == 1) {
            const float density = fmaxf(payload[0], 0.0f);
            const float volume_opacity = aura_clamp_unit(payload[1]);
            const float path_length = fmaxf(hit_exits[hit_i] - hit_depths[hit_i], 0.0f);
            const float alpha = volume_opacity * (1.0f - expf(-density * path_length));
            transmittance = aura_clamp_unit(1.0f - alpha);
        } else if (carrier_id == 2) {
            float point[3] = {
                origin[0] + direction[0] * hit_depths[hit_i],
                origin[1] + direction[1] * hit_depths[hit_i],
                origin[2] + direction[2] * hit_depths[hit_i],
            };
            const float support = aura_beta_support(
                point,
                element_mins + element_i * 3,
                element_maxs + element_i * 3,
                beta_support_radii + element_i * 3,
                payload[0],
                payload[1]
            );
            transmittance = aura_clamp_unit(1.0f - opacity * support);
        } else if (carrier_id == 3) {
            float point[3] = {
                origin[0] + direction[0] * hit_depths[hit_i],
                origin[1] + direction[1] * hit_depths[hit_i],
                origin[2] + direction[2] * hit_depths[hit_i],
            };
            const float phase = payload[3];
            const float bandwidth = aura_clamp_unit(payload[4]);
            const float dot = point[0] * payload[0] + point[1] * payload[1] + point[2] * payload[2];
            const float modulation = 1.0f - bandwidth + bandwidth * (0.5f + 0.5f * sinf(6.28318530718f * dot + phase));
            color_r_hit = aura_clamp_unit(color_r_hit * modulation);
            color_g_hit = aura_clamp_unit(color_g_hit * modulation);
            color_b_hit = aura_clamp_unit(color_b_hit * modulation);
            confidence_value = aura_clamp_unit(confidence_value * bandwidth);
        } else if (carrier_id == 4) {
            confidence_value = aura_clamp_unit(confidence_value * (1.0f - payload[0] * 0.25f));
        }
        const float alpha = 1.0f - transmittance;
        const float weight = remaining * alpha;
        color_r += weight * color_r_hit;
        color_g += weight * color_g_hit;
        color_b += weight * color_b_hit;
        confidence_num += weight * confidence_value;
        confidence_den += weight;
        remaining *= transmittance;
        residual = residual || (carrier_id == 4);
        ordered_hits[ray_i * max_hits + hit_i] = element_i;
    }

    const int first_element = hit_elements[0];
    out_color[ray_i * 3 + 0] = color_r;
    out_color[ray_i * 3 + 1] = color_g;
    out_color[ray_i * 3 + 2] = color_b;
    out_alpha[ray_i] = fminf(fmaxf(1.0f - remaining, 0.0f), 1.0f);
    out_transmittance[ray_i] = fminf(fmaxf(remaining, 0.0f), 1.0f);
    out_depth[ray_i] = hit_depths[0];
    out_normal[ray_i * 3 + 0] = hit_normals[0];
    out_normal[ray_i * 3 + 1] = hit_normals[1];
    out_normal[ray_i * 3 + 2] = hit_normals[2];
    out_confidence[ray_i] = confidence_den > 1.0e-8f ? confidence_num / confidence_den : 0.0f;
    out_residual[ray_i] = residual;
    out_material_id[ray_i] = material_ids[first_element];
    out_semantic_id[ray_i] = semantic_ids[first_element];

    (void)hit_exits;
}

extern "C" void aura_render_rays_launcher(
    const float* ray_origins,
    const float* ray_directions,
    const float* element_mins,
    const float* element_maxs,
    const float* plane_points,
    const float* plane_normals,
    const float* beta_support_radii,
    const float* gaussian_means,
    const float* gaussian_inverse_covariances,
    const float* gaussian_support_radius_sq,
    const int* carrier_ids,
    const float* colors,
    const float* opacities,
    const float* confidences,
    const float* payload_params,
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
        plane_points,
        plane_normals,
        beta_support_radii,
        gaussian_means,
        gaussian_inverse_covariances,
        gaussian_support_radius_sq,
        carrier_ids,
        colors,
        opacities,
        confidences,
        payload_params,
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
