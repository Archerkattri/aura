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
