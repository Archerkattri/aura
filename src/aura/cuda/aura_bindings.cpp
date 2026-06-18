#include <torch/extension.h>

#include <limits>
#include <vector>

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
);

namespace {

void require_cuda_float_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_cuda_int_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt32, name, " must be int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void require_shape2(const torch::Tensor& tensor, const char* name, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank 2");
    TORCH_CHECK(tensor.size(0) == dim0 && tensor.size(1) == dim1, name, " has invalid shape");
}

void require_shape1(const torch::Tensor& tensor, const char* name, int64_t dim0) {
    TORCH_CHECK(tensor.dim() == 1, name, " must be rank 1");
    TORCH_CHECK(tensor.size(0) == dim0, name, " has invalid shape");
}

}  // namespace

pybind11::dict render_rays(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor element_mins,
    torch::Tensor element_maxs,
    torch::Tensor carrier_ids,
    torch::Tensor colors,
    torch::Tensor opacities,
    torch::Tensor confidences,
    torch::Tensor material_ids,
    torch::Tensor semantic_ids,
    int64_t max_hits,
    int64_t threads_per_block
) {
    require_cuda_float_tensor(ray_origins, "ray_origins");
    require_cuda_float_tensor(ray_directions, "ray_directions");
    require_cuda_float_tensor(element_mins, "element_mins");
    require_cuda_float_tensor(element_maxs, "element_maxs");
    require_cuda_int_tensor(carrier_ids, "carrier_ids");
    require_cuda_float_tensor(colors, "colors");
    require_cuda_float_tensor(opacities, "opacities");
    require_cuda_float_tensor(confidences, "confidences");
    require_cuda_int_tensor(material_ids, "material_ids");
    require_cuda_int_tensor(semantic_ids, "semantic_ids");

    TORCH_CHECK(max_hits > 0, "max_hits must be positive");
    TORCH_CHECK(threads_per_block > 0 && threads_per_block <= 1024, "threads_per_block must be in [1, 1024]");
    TORCH_CHECK(ray_origins.dim() == 2 && ray_origins.size(1) == 3, "ray_origins must be rayCount x 3");
    const int64_t ray_count_64 = ray_origins.size(0);
    require_shape2(ray_directions, "ray_directions", ray_count_64, 3);
    TORCH_CHECK(element_mins.dim() == 2 && element_mins.size(1) == 3, "element_mins must be elementCount x 3");
    const int64_t element_count_64 = element_mins.size(0);
    require_shape2(element_maxs, "element_maxs", element_count_64, 3);
    require_shape1(carrier_ids, "carrier_ids", element_count_64);
    require_shape2(colors, "colors", element_count_64, 3);
    require_shape1(opacities, "opacities", element_count_64);
    require_shape1(confidences, "confidences", element_count_64);
    require_shape1(material_ids, "material_ids", element_count_64);
    require_shape1(semantic_ids, "semantic_ids", element_count_64);
    TORCH_CHECK(ray_count_64 <= static_cast<int64_t>(std::numeric_limits<int>::max()), "ray_count exceeds int32 launcher ABI");
    TORCH_CHECK(element_count_64 <= static_cast<int64_t>(std::numeric_limits<int>::max()), "element_count exceeds int32 launcher ABI");
    TORCH_CHECK(max_hits <= static_cast<int64_t>(std::numeric_limits<int>::max()), "max_hits exceeds int32 launcher ABI");

    const int ray_count = static_cast<int>(ray_count_64);
    const int element_count = static_cast<int>(element_count_64);
    const int hit_count = static_cast<int>(max_hits);
    const int threads = static_cast<int>(threads_per_block);

    auto float_options = ray_origins.options();
    auto int_options = carrier_ids.options();
    auto byte_options = torch::TensorOptions().dtype(torch::kUInt8).device(ray_origins.device());

    torch::Tensor out_color = torch::empty({ray_count, 3}, float_options);
    torch::Tensor out_alpha = torch::empty({ray_count}, float_options);
    torch::Tensor out_transmittance = torch::empty({ray_count}, float_options);
    torch::Tensor out_depth = torch::empty({ray_count}, float_options);
    torch::Tensor out_normal = torch::empty({ray_count, 3}, float_options);
    torch::Tensor out_confidence = torch::empty({ray_count}, float_options);
    torch::Tensor out_residual = torch::empty({ray_count}, byte_options);
    torch::Tensor out_material_id = torch::empty({ray_count}, int_options);
    torch::Tensor out_semantic_id = torch::empty({ray_count}, int_options);
    torch::Tensor ordered_hits = torch::empty({ray_count, hit_count}, int_options);

    aura_render_rays_launcher(
        ray_origins.data_ptr<float>(),
        ray_directions.data_ptr<float>(),
        element_mins.data_ptr<float>(),
        element_maxs.data_ptr<float>(),
        carrier_ids.data_ptr<int>(),
        colors.data_ptr<float>(),
        opacities.data_ptr<float>(),
        confidences.data_ptr<float>(),
        material_ids.data_ptr<int>(),
        semantic_ids.data_ptr<int>(),
        out_color.data_ptr<float>(),
        out_alpha.data_ptr<float>(),
        out_transmittance.data_ptr<float>(),
        out_depth.data_ptr<float>(),
        out_normal.data_ptr<float>(),
        out_confidence.data_ptr<float>(),
        out_residual.data_ptr<unsigned char>(),
        out_material_id.data_ptr<int>(),
        out_semantic_id.data_ptr<int>(),
        ordered_hits.data_ptr<int>(),
        ray_count,
        element_count,
        hit_count,
        threads
    );

    pybind11::dict outputs;
    outputs["out_color"] = out_color;
    outputs["out_alpha"] = out_alpha;
    outputs["out_transmittance"] = out_transmittance;
    outputs["out_depth"] = out_depth;
    outputs["out_normal"] = out_normal;
    outputs["out_confidence"] = out_confidence;
    outputs["out_residual"] = out_residual;
    outputs["out_material_id"] = out_material_id;
    outputs["out_semantic_id"] = out_semantic_id;
    outputs["ordered_hits"] = ordered_hits;
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_rays", &render_rays, "Launch the AURA CUDA renderer over packed ray and scene tensors");
}
