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

// Forward WITHOUT early-termination: processes all K valid slots so the
// transmittance T_k is well-defined for every splat in the backward reverse
// pass. Saves final_T per pixel (product over all valid splats).
__global__ void prism_forward_full_kernel(
    const long* __restrict__ idxTK, const float* __restrict__ means2d,
    const float* __restrict__ conics, const float* __restrict__ colors,
    const float* __restrict__ opac, int ntx, int ts, int W, int H, int K,
    float* __restrict__ out, float* __restrict__ final_T)
{
    int tile = blockIdx.x, local = threadIdx.x;
    if (local >= ts * ts) return;
    int px = (tile % ntx) * ts + (local % ts);
    int py = (tile / ntx) * ts + (local / ts);
    if (px >= W || py >= H) return;
    float Tt = 1.f, c0 = 0.f, c1 = 0.f, c2 = 0.f, fx = (float)px, fy = (float)py;
    const long* tl = idxTK + (long)tile * K;
    for (int k = 0; k < K; ++k) {
        long idx = tl[k]; if (idx < 0) break;
        float dx = fx - means2d[idx*2], dy = fy - means2d[idx*2+1];
        float a = conics[idx*3], b = conics[idx*3+1], c = conics[idx*3+2];
        float power = a*dx*dx + 2.f*b*dx*dy + c*dy*dy; if (power < 0.f) power = 0.f;
        float w = __expf(-0.5f*power);
        float al = opac[idx]*w; if (al > 0.999f) al = 0.999f;
        float contrib = Tt*al;
        c0 += contrib*colors[idx*3]; c1 += contrib*colors[idx*3+1]; c2 += contrib*colors[idx*3+2];
        Tt *= (1.f-al);
    }
    int pix = py*W+px;
    out[pix*3]=c0; out[pix*3+1]=c1; out[pix*3+2]=c2; final_T[pix]=Tt;
}

// Per-pixel reverse-traversal backward (3DGS-style): reconstructs T_k from the
// saved final_T and accumulates gradients into per-carrier buffers via atomics.
__global__ void prism_backward_kernel(
    const long* __restrict__ idxTK, const float* __restrict__ means2d,
    const float* __restrict__ conics, const float* __restrict__ colors,
    const float* __restrict__ opac, const float* __restrict__ final_T,
    const float* __restrict__ grad_out, int ntx, int ts, int W, int H, int K,
    float* __restrict__ g_means2d, float* __restrict__ g_conics,
    float* __restrict__ g_colors, float* __restrict__ g_opac)
{
    int tile = blockIdx.x, local = threadIdx.x;
    if (local >= ts * ts) return;
    int px = (tile % ntx) * ts + (local % ts);
    int py = (tile / ntx) * ts + (local / ts);
    if (px >= W || py >= H) return;
    int pix = py*W+px;
    float dC0 = grad_out[pix*3], dC1 = grad_out[pix*3+1], dC2 = grad_out[pix*3+2];
    float fx = (float)px, fy = (float)py;
    const long* tl = idxTK + (long)tile * K;
    int nvalid = 0; for (int k = 0; k < K; ++k) { if (tl[k] < 0) break; nvalid++; }
    float Tt = final_T[pix];      // T after all valid splats
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;   // suffix color accumulated from behind
    for (int k = nvalid - 1; k >= 0; --k) {
        long idx = tl[k];
        float dx = fx - means2d[idx*2], dy = fy - means2d[idx*2+1];
        float a = conics[idx*3], b = conics[idx*3+1], c = conics[idx*3+2];
        float power = a*dx*dx + 2.f*b*dx*dy + c*dy*dy; if (power < 0.f) power = 0.f;
        float w = __expf(-0.5f*power);
        float al = opac[idx]*w; bool clamped = false;
        if (al > 0.999f) { al = 0.999f; clamped = true; }
        float one_m = 1.f - al;
        float T_k = Tt / one_m;          // T before this splat
        float contrib = T_k * al;
        float col0 = colors[idx*3], col1 = colors[idx*3+1], col2 = colors[idx*3+2];
        // grad wrt color
        atomicAdd(&g_colors[idx*3],   contrib * dC0);
        atomicAdd(&g_colors[idx*3+1], contrib * dC1);
        atomicAdd(&g_colors[idx*3+2], contrib * dC2);
        // dC/d alpha = T_k*color - suffix/(1-al)
        float dC_da = dC0*(T_k*col0 - s0/one_m) + dC1*(T_k*col1 - s1/one_m) + dC2*(T_k*col2 - s2/one_m);
        float dL_dal = clamped ? 0.f : dC_da;       // clamp kills gradient
        // alpha = opac*w
        atomicAdd(&g_opac[idx], dL_dal * w);
        float dL_dw = dL_dal * opac[idx];
        float dL_dpower = dL_dw * (-0.5f * w);
        atomicAdd(&g_conics[idx*3],   dL_dpower * dx * dx);
        atomicAdd(&g_conics[idx*3+1], dL_dpower * 2.f * dx * dy);
        atomicAdd(&g_conics[idx*3+2], dL_dpower * dy * dy);
        float dpow_dmx = -(2.f*a*dx + 2.f*b*dy);
        float dpow_dmy = -(2.f*b*dx + 2.f*c*dy);
        atomicAdd(&g_means2d[idx*2],   dL_dpower * dpow_dmx);
        atomicAdd(&g_means2d[idx*2+1], dL_dpower * dpow_dmy);
        // advance suffix + transmittance for next (earlier) splat
        s0 += contrib*col0; s1 += contrib*col1; s2 += contrib*col2;
        Tt = T_k;
    }
}

std::vector<torch::Tensor> prism_forward_full(
    torch::Tensor idxTK, torch::Tensor means2d, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opac, int ntx, int ts, int W, int H)
{
    int T = idxTK.size(0), K = idxTK.size(1);
    auto out = torch::zeros({H, W, 3}, means2d.options());
    auto final_T = torch::ones({H, W}, means2d.options());
    prism_forward_full_kernel<<<T, ts*ts>>>(
        idxTK.data_ptr<long>(), means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opac.data_ptr<float>(), ntx, ts, W, H, K,
        out.data_ptr<float>(), final_T.data_ptr<float>());
    return {out, final_T};
}

std::vector<torch::Tensor> prism_backward(
    torch::Tensor idxTK, torch::Tensor means2d, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opac, torch::Tensor final_T,
    torch::Tensor grad_out, int ntx, int ts, int W, int H)
{
    int T = idxTK.size(0), K = idxTK.size(1);
    auto g_means2d = torch::zeros_like(means2d);
    auto g_conics = torch::zeros_like(conics);
    auto g_colors = torch::zeros_like(colors);
    auto g_opac = torch::zeros_like(opac);
    prism_backward_kernel<<<T, ts*ts>>>(
        idxTK.data_ptr<long>(), means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opac.data_ptr<float>(), final_T.data_ptr<float>(),
        grad_out.contiguous().data_ptr<float>(), ntx, ts, W, H, K,
        g_means2d.data_ptr<float>(), g_conics.data_ptr<float>(),
        g_colors.data_ptr<float>(), g_opac.data_ptr<float>());
    return {g_means2d, g_conics, g_colors, g_opac};
}
"""

_CPP_SRC = (
    "std::vector<torch::Tensor> prism_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);\n"
    "std::vector<torch::Tensor> prism_forward_full(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);\n"
    "std::vector<torch::Tensor> prism_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, int);"
)


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
            functions=["prism_forward", "prism_forward_full", "prism_backward"],
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


def _make_autograd_fn():
    import torch

    ext = _load()
    if ext is None:
        return None

    class _PrismRasterize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, idxTK, means2d, conics, colors, opac, ntx, ts, W, H):
            out, final_T = ext.prism_forward_full(idxTK, means2d, conics, colors, opac, ntx, ts, W, H)
            ctx.save_for_backward(idxTK, means2d, conics, colors, opac, final_T)
            ctx.dims = (ntx, ts, W, H)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            idxTK, means2d, conics, colors, opac, final_T = ctx.saved_tensors
            ntx, ts, W, H = ctx.dims
            gm, gc, gcol, gop = ext.prism_backward(
                idxTK, means2d, conics, colors, opac, final_T, grad_out, ntx, ts, W, H
            )
            return (None, gm, gc, gcol, gop, None, None, None, None)

    return _PrismRasterize


@functools.lru_cache(maxsize=1)
def _autograd_fn():
    return _make_autograd_fn()


def render_gaussians_cuda_diff(means, quats, scales, opacities, colors, viewmat, K,
                               width, height, *, tile: int = 16, max_per_tile: int = 256):
    """Differentiable PRISM CUDA render: forward + custom CUDA backward via an
    autograd.Function. Projection/binning run in torch (autograd flows through
    them); the composite forward/backward run in the CUDA kernels. Gradients
    reach means/quats/scales/opacity/colors. Raises if CUDA is unavailable."""

    import torch
    from .prism import quats_scales_to_cov3d, project_gaussians

    fn = _autograd_fn()
    if fn is None:
        raise RuntimeError("PRISM CUDA extension unavailable")
    cov = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov, viewmat, K, width, height, torch)
    idxTK, ntx = _bin_tiles(proj, colors, opacities, width, height, torch, tile, max_per_tile)
    means2d = proj.means2d
    conics = proj.conics
    col = colors[proj.index]
    op = opacities[proj.index]
    return fn.apply(idxTK.contiguous(), means2d, conics, col, op, int(ntx), int(tile), int(width), int(height))


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
