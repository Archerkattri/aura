"""PRISM CUDA fast path — hand-written CUDA tile-compositing kernels for AURA's
own differentiable rasterizer, compiled at runtime via torch ``load_inline``
against the local nvcc (no prebuilt wheel).

Supports the full PRISM footprint family in CUDA — **gaussian (0), beta (1),
gabor (2)** — for both forward rendering and the differentiable backward, with
analytic gradients. One CUDA thread renders one pixel, walking its tile's
depth-sorted carrier list (gsplat-style). The torch-side projection + tile
binning + depth sort (``prism.py``) are reused; these kernels replace the
expensive per-pixel composite scan and its backward.

Footprint weight ``w`` (before opacity) and its analytic derivatives:
  gaussian: w = exp(-0.5 p),                       p = aᐧdx²+2bᐧdxᐧdy+cᐧdy²
  beta:     w = max(1 - sqrt(p)/3, 0)^be           (compact support, DBS-style)
  gabor:    w = exp(-0.5 p) ᐧ (1+cos(θ))/2,          θ = fxᐧdx + fyᐧdy + phase
Gabor's modulation depends on (dx,dy) directly, so its means2d gradient has an
extra term beyond the shared ``dw/dp`` path (handled in the kernel). Falls back
to the torch compositor if CUDA/nvcc is unavailable.
"""

from __future__ import annotations

import functools

# Footprint type codes (kept in sync with prism.py footprint registry).
FOOTPRINT_GAUSSIAN = 0
FOOTPRINT_BETA = 1
FOOTPRINT_GABOR = 2

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Footprint weight + derivatives. Returns w; sets dw_dp (d w / d power) and, for
// gabor, the extra direct (dx,dy) modulation derivatives dw_dmod_dx/dy and the
// gabor-param grads. For gaussian/beta the extra terms are zero.
__device__ __forceinline__ float prism_footprint(
    int ftype, float p, float dx, float dy,
    float fxk, float fyk, float phasek, float be,
    float* dw_dp, float* dw_ddx_extra, float* dw_ddy_extra,
    float* dw_dfx, float* dw_dfy, float* dw_dphase)
{
    *dw_dp = 0.f; *dw_ddx_extra = 0.f; *dw_ddy_extra = 0.f;
    *dw_dfx = 0.f; *dw_dfy = 0.f; *dw_dphase = 0.f;
    if (ftype == 1) {  // beta: (1 - r/3)^be, r=sqrt(p)
        float r = sqrtf(fmaxf(p, 0.f));
        float t = 1.f - r / 3.f;
        if (t <= 0.f) return 0.f;
        float w = powf(t, be);
        // dw/dp = be*t^(be-1) * (-1/3) * dr/dp,  dr/dp = 1/(2r)
        if (r > 1e-6f) *dw_dp = be * powf(t, be - 1.f) * (-1.f / 3.f) * (0.5f / r);
        return w;
    } else if (ftype == 2) {  // gabor: exp(-0.5p) * (1+cos(theta))/2
        float env = __expf(-0.5f * p);
        float theta = fxk * dx + fyk * dy + phasek;
        float mod = 0.5f * (1.f + cosf(theta));
        float w = env * mod;
        *dw_dp = -0.5f * env * mod;                 // via envelope
        float dmod = -0.5f * sinf(theta);           // d mod / d theta
        *dw_ddx_extra = env * dmod * fxk;           // d w / d dx (direct, not via p)
        *dw_ddy_extra = env * dmod * fyk;
        *dw_dfx = env * dmod * dx;
        *dw_dfy = env * dmod * dy;
        *dw_dphase = env * dmod;
        return w;
    } else {  // gaussian
        float w = __expf(-0.5f * p);
        *dw_dp = -0.5f * w;
        return w;
    }
}

template <bool EARLY>
__global__ void prism_fwd(
    const long* __restrict__ idxTK, const float* __restrict__ means2d,
    const float* __restrict__ conics, const float* __restrict__ colors,
    const float* __restrict__ opac, const float* __restrict__ freq,
    const float* __restrict__ phase, const long* __restrict__ ftypes, float be,
    int volumetric, int ntx, int ts, int W, int H, int K,
    float* __restrict__ out, float* __restrict__ final_T)
{
    int tile = blockIdx.x, local = threadIdx.x;
    if (local >= ts * ts) return;
    int px = (tile % ntx) * ts + (local % ts);
    int py = (tile / ntx) * ts + (local / ts);
    if (px >= W || py >= H) return;
    float Tt = 1.f, c0 = 0.f, c1 = 0.f, c2 = 0.f, fx = (float)px, fy = (float)py;
    const long* tl = idxTK + (long)tile * K;
    float dp, ex, ey, gfx, gfy, gph;
    for (int k = 0; k < K; ++k) {
        long idx = tl[k]; if (idx < 0) break;
        int ft = (int)ftypes[idx];
        float dx = fx - means2d[idx*2], dy = fy - means2d[idx*2+1];
        float a = conics[idx*3], b = conics[idx*3+1], c = conics[idx*3+2];
        float p = a*dx*dx + 2.f*b*dx*dy + c*dy*dy; if (p < 0.f) p = 0.f;
        float w = prism_footprint(ft, p, dx, dy, freq[idx*2], freq[idx*2+1],
                                  phase[idx], be, &dp, &ex, &ey, &gfx, &gfy, &gph);
        float al = volumetric ? (1.f - __expf(-fmaxf(opac[idx]*w, 0.f))) : (opac[idx]*w);
        if (al > 0.999f) al = 0.999f;
        float contrib = Tt*al;
        c0 += contrib*colors[idx*3]; c1 += contrib*colors[idx*3+1]; c2 += contrib*colors[idx*3+2];
        Tt *= (1.f-al);
        if (EARLY && Tt < 1e-4f) break;
    }
    int pix = py*W+px;
    out[pix*3]=c0; out[pix*3+1]=c1; out[pix*3+2]=c2; final_T[pix]=Tt;
}

__global__ void prism_bwd(
    const long* __restrict__ idxTK, const float* __restrict__ means2d,
    const float* __restrict__ conics, const float* __restrict__ colors,
    const float* __restrict__ opac, const float* __restrict__ freq,
    const float* __restrict__ phase, const long* __restrict__ ftypes,
    const float* __restrict__ final_T,
    const float* __restrict__ grad_out, float be, int volumetric,
    int ntx, int ts, int W, int H, int K,
    float* __restrict__ g_means2d, float* __restrict__ g_conics,
    float* __restrict__ g_colors, float* __restrict__ g_opac,
    float* __restrict__ g_freq, float* __restrict__ g_phase)
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
    float Tt = final_T[pix];
    float s0 = 0.f, s1 = 0.f, s2 = 0.f;
    float dp, ex, ey, gfx, gfy, gph;
    for (int k = nvalid - 1; k >= 0; --k) {
        long idx = tl[k];
        int ft = (int)ftypes[idx];
        float dx = fx - means2d[idx*2], dy = fy - means2d[idx*2+1];
        float a = conics[idx*3], b = conics[idx*3+1], c = conics[idx*3+2];
        float p = a*dx*dx + 2.f*b*dx*dy + c*dy*dy; if (p < 0.f) p = 0.f;
        float w = prism_footprint(ft, p, dx, dy, freq[idx*2], freq[idx*2+1],
                                  phase[idx], be, &dp, &ex, &ey, &gfx, &gfy, &gph);
        float tau = opac[idx]*w;
        float al = volumetric ? (1.f - __expf(-fmaxf(tau, 0.f))) : tau;
        bool clamped = false;
        if (al > 0.999f) { al = 0.999f; clamped = true; }
        float one_m = 1.f - al;
        float T_k = Tt / one_m;
        float contrib = T_k * al;
        float col0 = colors[idx*3], col1 = colors[idx*3+1], col2 = colors[idx*3+2];
        atomicAdd(&g_colors[idx*3],   contrib*dC0);
        atomicAdd(&g_colors[idx*3+1], contrib*dC1);
        atomicAdd(&g_colors[idx*3+2], contrib*dC2);
        float dC_da = dC0*(T_k*col0 - s0/one_m) + dC1*(T_k*col1 - s1/one_m) + dC2*(T_k*col2 - s2/one_m);
        float dL_dal = clamped ? 0.f : dC_da;
        // alpha = opac*w (billboard) or 1-exp(-opac*w) (volumetric); d al/d(opac*w)
        // = 1 or exp(-tau). Fold that factor into the opac/w grads.
        float fac = volumetric ? __expf(-fmaxf(tau, 0.f)) : 1.f;
        atomicAdd(&g_opac[idx], dL_dal * fac * w);
        float dL_dw = dL_dal * fac * opac[idx];
        // w depends on p (-> conic + means2d) and, for gabor, directly on dx,dy + gabor params.
        float dL_dp = dL_dw * dp;
        atomicAdd(&g_conics[idx*3],   dL_dp * dx * dx);
        atomicAdd(&g_conics[idx*3+1], dL_dp * 2.f * dx * dy);
        atomicAdd(&g_conics[idx*3+2], dL_dp * dy * dy);
        float dp_dmx = -(2.f*a*dx + 2.f*b*dy);
        float dp_dmy = -(2.f*b*dx + 2.f*c*dy);
        // d dx / d mx = -1 ; gabor extra terms use that (ex = dw/d dx).
        float dL_dmx = dL_dp * dp_dmx + dL_dw * ex * (-1.f);
        float dL_dmy = dL_dp * dp_dmy + dL_dw * ey * (-1.f);
        atomicAdd(&g_means2d[idx*2],   dL_dmx);
        atomicAdd(&g_means2d[idx*2+1], dL_dmy);
        if (ft == 2) {
            atomicAdd(&g_freq[idx*2],   dL_dw * gfx);
            atomicAdd(&g_freq[idx*2+1], dL_dw * gfy);
            atomicAdd(&g_phase[idx],    dL_dw * gph);
        }
        s0 += contrib*col0; s1 += contrib*col1; s2 += contrib*col2;
        Tt = T_k;
    }
}

static torch::Tensor _zeros_like(torch::Tensor x){ return torch::zeros_like(x); }

std::vector<torch::Tensor> prism_forward(
    torch::Tensor idxTK, torch::Tensor means2d, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opac, torch::Tensor freq, torch::Tensor phase,
    torch::Tensor ftypes, double be, int volumetric, int ntx, int ts, int W, int H)
{
    int K = idxTK.size(1);
    auto out = torch::zeros({H, W, 3}, means2d.options());
    auto fT = torch::ones({H, W}, means2d.options());
    prism_fwd<true><<<idxTK.size(0), ts*ts>>>(
        idxTK.data_ptr<long>(), means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opac.data_ptr<float>(), freq.data_ptr<float>(),
        phase.data_ptr<float>(), ftypes.data_ptr<long>(), (float)be, volumetric, ntx, ts, W, H, K,
        out.data_ptr<float>(), fT.data_ptr<float>());
    return {out, fT};
}

std::vector<torch::Tensor> prism_forward_full(
    torch::Tensor idxTK, torch::Tensor means2d, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opac, torch::Tensor freq, torch::Tensor phase,
    torch::Tensor ftypes, double be, int volumetric, int ntx, int ts, int W, int H)
{
    int K = idxTK.size(1);
    auto out = torch::zeros({H, W, 3}, means2d.options());
    auto fT = torch::ones({H, W}, means2d.options());
    prism_fwd<false><<<idxTK.size(0), ts*ts>>>(
        idxTK.data_ptr<long>(), means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opac.data_ptr<float>(), freq.data_ptr<float>(),
        phase.data_ptr<float>(), ftypes.data_ptr<long>(), (float)be, volumetric, ntx, ts, W, H, K,
        out.data_ptr<float>(), fT.data_ptr<float>());
    return {out, fT};
}

std::vector<torch::Tensor> prism_backward(
    torch::Tensor idxTK, torch::Tensor means2d, torch::Tensor conics,
    torch::Tensor colors, torch::Tensor opac, torch::Tensor freq, torch::Tensor phase,
    torch::Tensor ftypes, torch::Tensor final_T, torch::Tensor grad_out, double be, int volumetric,
    int ntx, int ts, int W, int H)
{
    int K = idxTK.size(1);
    auto gm = torch::zeros_like(means2d), gc = torch::zeros_like(conics);
    auto gcol = torch::zeros_like(colors), gop = torch::zeros_like(opac);
    auto gfreq = torch::zeros_like(freq), gph = torch::zeros_like(phase);
    prism_bwd<<<idxTK.size(0), ts*ts>>>(
        idxTK.data_ptr<long>(), means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opac.data_ptr<float>(), freq.data_ptr<float>(),
        phase.data_ptr<float>(), ftypes.data_ptr<long>(), final_T.data_ptr<float>(),
        grad_out.contiguous().data_ptr<float>(), (float)be, volumetric, ntx, ts, W, H, K,
        gm.data_ptr<float>(), gc.data_ptr<float>(), gcol.data_ptr<float>(),
        gop.data_ptr<float>(), gfreq.data_ptr<float>(), gph.data_ptr<float>());
    return {gm, gc, gcol, gop, gfreq, gph};
}
"""

_CPP_SRC = (
    "std::vector<torch::Tensor> prism_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int, int, int, int, int);\n"
    "std::vector<torch::Tensor> prism_forward_full(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int, int, int, int, int);\n"
    "std::vector<torch::Tensor> prism_backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int, int, int, int, int);"
)


@functools.lru_cache(maxsize=1)
def _load():
    try:
        import torch
        from torch.utils.cpp_extension import load_inline
        if not torch.cuda.is_available():
            return None
        return load_inline(
            name="prism_cuda_ext_v4",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["prism_forward", "prism_forward_full", "prism_backward"],
            verbose=False,
        )
    except Exception:
        return None


def cuda_available() -> bool:
    return _load() is not None


def _footprint_code(footprint) -> int:
    if footprint in ("beta", FOOTPRINT_BETA):
        return FOOTPRINT_BETA
    if footprint in ("gabor", FOOTPRINT_GABOR):
        return FOOTPRINT_GABOR
    return FOOTPRINT_GAUSSIAN


def _gabor_arrays(torch, idx, device, freq=None, phase=None):
    n = idx.shape[0]
    f = (freq[idx] if freq is not None else torch.zeros((n, 2), dtype=torch.float32, device=device))
    p = (phase[idx] if phase is not None else torch.zeros((n,), dtype=torch.float32, device=device))
    return f.contiguous(), p.contiguous()


def _ftypes_visible(torch, idx, device, footprint="gaussian", ftypes=None):
    """Per-visible-carrier footprint-type long tensor: either gathered from a
    per-carrier ``ftypes`` (over ALL carriers) or broadcast from a footprint name."""
    if ftypes is not None:
        return ftypes[idx].to(torch.long).contiguous()
    return torch.full((idx.shape[0],), _footprint_code(footprint), dtype=torch.long, device=device).contiguous()


def render_gaussians_cuda(means, quats, scales, opacities, colors, viewmat, K,
                          width, height, *, tile=16, max_per_tile=256,
                          footprint="gaussian", ftypes=None, beta_exp=2.0, freq=None, phase=None,
                          volumetric=False):
    """Forward-render a scene with the PRISM CUDA kernel. ``footprint`` broadcasts
    one kernel to all carriers; ``ftypes`` (per-carrier int codes over all
    carriers) renders a HETEROGENEOUS mix. ``volumetric`` uses EVER-style
    1-exp(-opacity*w) alpha. Reuses prism.py projection + binning."""
    import torch
    from .prism import quats_scales_to_cov3d, project_gaussians
    ext = _load()
    if ext is None:
        raise RuntimeError("PRISM CUDA extension unavailable (no nvcc/CUDA?)")
    cov = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov, viewmat, K, width, height, torch)
    idxTK, ntx = _bin_tiles(proj, colors, opacities, width, height, torch, tile, max_per_tile)
    f, p = _gabor_arrays(torch, proj.index, colors.device, freq, phase)
    ft = _ftypes_visible(torch, proj.index, colors.device, footprint, ftypes)
    op = opacities[proj.index]
    if getattr(proj, "opacity_comp", None) is not None:
        op = op * proj.opacity_comp
    out, _T = ext.prism_forward(
        idxTK.contiguous(), proj.means2d.contiguous(), proj.conics.contiguous(),
        colors[proj.index].contiguous(), op.contiguous(), f, p, ft,
        float(beta_exp), int(bool(volumetric)), int(ntx), int(tile), int(width), int(height))
    return out


def _make_autograd_fn():
    import torch
    ext = _load()
    if ext is None:
        return None

    class _PrismRasterize(torch.autograd.Function):
        @staticmethod
        def forward(ctx, idxTK, means2d, conics, colors, opac, freq, phase, ftypes, be, vol, ntx, ts, W, H):
            out, fT = ext.prism_forward_full(idxTK, means2d, conics, colors, opac, freq, phase, ftypes, be, vol, ntx, ts, W, H)
            ctx.save_for_backward(idxTK, means2d, conics, colors, opac, freq, phase, ftypes, fT)
            ctx.meta = (be, vol, ntx, ts, W, H)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            idxTK, means2d, conics, colors, opac, freq, phase, ftypes, fT = ctx.saved_tensors
            be, vol, ntx, ts, W, H = ctx.meta
            gm, gc, gcol, gop, gfreq, gph = ext.prism_backward(
                idxTK, means2d, conics, colors, opac, freq, phase, ftypes, fT, grad_out, be, vol, ntx, ts, W, H)
            return (None, gm, gc, gcol, gop, gfreq, gph, None, None, None, None, None, None, None)

    return _PrismRasterize


@functools.lru_cache(maxsize=1)
def _autograd_fn():
    return _make_autograd_fn()


def render_gaussians_cuda_diff(means, quats, scales, opacities, colors, viewmat, K,
                               width, height, *, tile=16, max_per_tile=256,
                               footprint="gaussian", ftypes=None, beta_exp=2.0, freq=None, phase=None,
                               volumetric=False):
    """Differentiable PRISM CUDA render (forward + custom CUDA backward).
    ``footprint`` broadcasts one kernel; ``ftypes`` renders a heterogeneous mix;
    ``volumetric`` uses EVER-style alpha. Gradients reach
    means/quats/scales/opacity/colors (and gabor freq/phase)."""
    import torch
    from .prism import quats_scales_to_cov3d, project_gaussians
    fn = _autograd_fn()
    if fn is None:
        raise RuntimeError("PRISM CUDA extension unavailable")
    cov = quats_scales_to_cov3d(quats, scales, torch)
    proj = project_gaussians(means, cov, viewmat, K, width, height, torch)
    idxTK, ntx = _bin_tiles(proj, colors, opacities, width, height, torch, tile, max_per_tile)
    f, p = _gabor_arrays(torch, proj.index, colors.device, freq, phase)
    if freq is not None:
        f = freq[proj.index].contiguous()
    if phase is not None:
        p = phase[proj.index].contiguous()
    ft = _ftypes_visible(torch, proj.index, colors.device, footprint, ftypes)
    op = opacities[proj.index]
    if getattr(proj, "opacity_comp", None) is not None:
        op = op * proj.opacity_comp
    return fn.apply(idxTK.contiguous(), proj.means2d, proj.conics, colors[proj.index],
                    op, f, p, ft, float(beta_exp), int(bool(volumetric)),
                    int(ntx), int(tile), int(width), int(height))


def _bin_tiles(proj, colors, opacities, width, height, torch, tile, max_per_tile):
    """Tile-bin + depth-sort visible carriers into a padded [T,K] slot index."""
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
