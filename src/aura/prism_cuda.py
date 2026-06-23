"""PRISM CUDA fast path — a hand-written CUDA tile-compositing kernel for the
Gaussian carrier footprint, built with torch's runtime ``load_inline`` (compiled
against the local nvcc, no prebuilt wheel).

This is the start of the "alternative to gsplat made for AURA" at the kernel
level. The torch-side projection + tile binning + depth sort (in ``prism.py``)
are reused; this module only replaces the expensive per-pixel front-to-back
alpha-composite scan with a fused CUDA kernel. One CUDA thread renders one pixel,
walking its tile's depth-sorted carrier list with early termination — the same
structure as gsplat's ``rasterize_to_pixels``.

v1 implements the **forward** Gaussian composite (fast rendering / eval). The
differentiable CUDA backward is the documented next step; differentiable
training continues to use the pure-torch tiled compositor in ``prism.py`` (which
also runs on the GPU via torch ops). Falls back cleanly if CUDA/nvcc is absent.
"""

from __future__ import annotations

import functools

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// One thread per pixel; one block per tile (ts*ts threads). Each thread walks
// its tile's depth-sorted carrier slot list [T,K] (idxTK, -1 = empty) and does
// front-to-back alpha compositing of the Gaussian footprint.
__global__ void prism_forward_kernel(
    const long*  __restrict__ idxTK,    // [T*K]
    const float* __restrict__ means2d,  // [N*2]
    const float* __restrict__ conics,   // [N*3] (a,b,c) inverse 2D cov
    const float* __restrict__ colors,   // [N*3]
    const float* __restrict__ opac,     // [N]
    int ntx, int ts, int W, int H, int K,
    float* __restrict__ out,            // [H*W*3]
    float* __restrict__ final_T)        // [H*W]
{
    int tile = blockIdx.x;
    int local = threadIdx.x;
    if (local >= ts * ts) return;
    int tx = tile % ntx;
    int ty = tile / ntx;
    int px = tx * ts + (local % ts);
    int py = ty * ts + (local / ts);
    if (px >= W || py >= H) return;

    float Tt = 1.0f, c0 = 0.f, c1 = 0.f, c2 = 0.f;
    float fx = (float)px, fy = (float)py;
    const long* tile_list = idxTK + (long)tile * K;
    for (int k = 0; k < K; ++k) {
        long idx = tile_list[k];
        if (idx < 0) break;
        float dx = fx - means2d[idx * 2 + 0];
        float dy = fy - means2d[idx * 2 + 1];
        float a = conics[idx * 3 + 0];
        float b = conics[idx * 3 + 1];
        float c = conics[idx * 3 + 2];
        float power = a * dx * dx + 2.f * b * dx * dy + c * dy * dy;
        if (power < 0.f) power = 0.f;
        float w = __expf(-0.5f * power);
        float al = opac[idx] * w;
        if (al > 0.999f) al = 0.999f;
        float contrib = Tt * al;
        c0 += contrib * colors[idx * 3 + 0];
        c1 += contrib * colors[idx * 3 + 1];
        c2 += contrib * colors[idx * 3 + 2];
        Tt *= (1.f - al);
        if (Tt < 1e-4f) break;
    }
    int pix = (py * W + px);
    out[pix * 3 + 0] = c0;
    out[pix * 3 + 1] = c1;
    out[pix * 3 + 2] = c2;
    final_T[pix] = Tt;
}

std::vector<torch::Tensor> prism_forward(
    torch::Tensor idxTK, torch::Tensor means2d, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opac,
    int ntx, int ts, int W, int H)
{
    int T = idxTK.size(0);
    int K = idxTK.size(1);
    auto out = torch::zeros({H, W, 3}, means2d.options());
    auto final_T = torch::ones({H, W}, means2d.options());
    int threads = ts * ts;
    prism_forward_kernel<<<T, threads>>>(
        idxTK.data_ptr<long>(), means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opac.data_ptr<float>(),
        ntx, ts, W, H, K, out.data_ptr<float>(), final_T.data_ptr<float>());
    return {out, final_T};
}
"""

_CPP_SRC = "std::vector<torch::Tensor> prism_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);"


@functools.lru_cache(maxsize=1)
def _load():
    """Compile + cache the CUDA extension; returns the module or None."""
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
        if not torch.cuda.is_available():
            return None
        return load_inline(
            name="prism_cuda_ext",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["prism_forward"],
            verbose=False,
        )
    except Exception:
        return None


def cuda_available() -> bool:
    return _load() is not None


def render_gaussians_cuda(means, quats, scales, opacities, colors, viewmat, K,
                          width, height, *, tile: int = 16, max_per_tile: int = 256):
    """Forward-render a Gaussian scene with the PRISM CUDA kernel. Returns an
    [H,W,3] image. Raises if the extension is unavailable (caller may fall back
    to the torch path). Reuses prism.py's projection + tile binning."""

    import torch
    from .prism import quats_scales_to_cov3d, project_gaussians

    ext = _load()
    if ext is None:
        raise RuntimeError("PRISM CUDA extension unavailable (no nvcc/CUDA?)")

    cov = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov, viewmat, K, width, height, torch)
    idxTK, ntx = _bin_tiles(proj, colors, opacities, width, height, torch, tile, max_per_tile)
    # Gather contiguous per-carrier arrays (visible subset) for the kernel.
    means2d = proj.means2d.contiguous()
    conics = proj.conics.contiguous()
    col = colors[proj.index].contiguous()
    op = opacities[proj.index].contiguous()
    out, _T = ext.prism_forward(
        idxTK.contiguous(), means2d, conics, col, op, int(ntx), int(tile), int(width), int(height)
    )
    return out


def _bin_tiles(proj, colors, opacities, width, height, torch, tile, max_per_tile):
    """Tile-bin + depth-sort visible carriers into a padded [T,K] slot index
    (shared logic with prism.composite_tiled)."""

    device = colors.device
    M = int(proj.index.shape[0])
    ntx = (width + tile - 1) // tile
    nty = (height + tile - 1) // tile
    T = ntx * nty
    cx = proj.means2d[:, 0]; cy = proj.means2d[:, 1]
    r = proj.radii.clamp(min=0.0)
    tx0 = torch.clamp(torch.floor((cx - r) / tile).long(), 0, ntx - 1)
    tx1 = torch.clamp(torch.floor((cx + r) / tile).long(), 0, ntx - 1)
    ty0 = torch.clamp(torch.floor((cy - r) / tile).long(), 0, nty - 1)
    ty1 = torch.clamp(torch.floor((cy + r) / tile).long(), 0, nty - 1)
    nx = (tx1 - tx0 + 1).clamp(min=1); ny = (ty1 - ty0 + 1).clamp(min=1)
    cnt = nx * ny
    total = int(cnt.sum().item())
    if total == 0:
        return torch.full((T, 1), -1, dtype=torch.long, device=device), ntx
    starts = cnt.cumsum(0) - cnt
    carrier_of = torch.repeat_interleave(torch.arange(M, device=device), cnt)
    local = torch.arange(total, device=device) - starts[carrier_of]
    lx = local % nx[carrier_of]; ly = local // nx[carrier_of]
    tile_id = (ty0[carrier_of] + ly) * ntx + (tx0[carrier_of] + lx)
    dmax = float(proj.depths.max().item()) + 1.0
    key = tile_id.to(torch.float64) * dmax + proj.depths[carrier_of].to(torch.float64)
    order = torch.argsort(key)
    carrier_sorted = carrier_of[order]; tile_sorted = tile_id[order]
    counts = torch.bincount(tile_sorted, minlength=T)
    tstart = counts.cumsum(0) - counts
    pos = torch.arange(total, device=device) - tstart[tile_sorted]
    keep = pos < max_per_tile
    Kc = int(min(max_per_tile, int(counts.max().item())))
    idxTK = torch.full((T, Kc), -1, dtype=torch.long, device=device)
    idxTK[tile_sorted[keep], pos[keep]] = carrier_sorted[keep]
    return idxTK, ntx
