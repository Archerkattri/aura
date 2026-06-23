#!/usr/bin/env python3
"""PRISM speed benchmark: forward-render throughput of AURA's own rasterizer
(CUDA kernel vs the pure-torch tiled compositor) and gsplat, at several carrier
counts and resolutions. Writes JSON.

Usage: python experiments/prism_benchmark.py --out experiments/results/benchmark.json
"""
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _scene(torch, n, w, h, dev):
    means = torch.randn(n, 3, device=dev) * 0.6 + torch.tensor([0, 0, 3.0], device=dev)
    q = torch.randn(n, 4, device=dev); q = q / q.norm(dim=1, keepdim=True)
    s = torch.rand(n, 3, device=dev) * 0.05 + 0.02
    o = torch.rand(n, device=dev) * 0.5 + 0.4
    c = torch.rand(n, 3, device=dev)
    K = torch.tensor([[float(w), 0, w / 2], [0, float(w), h / 2], [0, 0, 1.0]], device=dev)
    vm = torch.eye(4, device=dev)
    return means, q, s, o, c, K, vm


def _bench(fn, iters=10):
    import torch
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms/frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/benchmark.json")
    args = ap.parse_args()
    import torch
    from aura.prism import quats_scales_to_cov3d, project_gaussians, composite_tiled
    from aura.prism_cuda import cuda_available, render_gaussians_cuda
    dev = "cuda"
    rows = []
    try:
        import gsplat
        from gsplat import rasterization
        has_gsplat = True
    except Exception:
        has_gsplat = False
    for n in (50_000, 100_000, 200_000):
        for (w, h) in ((512, 512), (979, 546)):
            m, q, s, o, c, K, vm = _scene(torch, n, w, h, dev)
            row = {"carriers": n, "width": w, "height": h}
            if cuda_available():
                _ = render_gaussians_cuda(m, q, s, o, c, vm, K, w, h)  # warmup/compile
                row["prism_cuda_ms"] = round(_bench(lambda: render_gaussians_cuda(m, q, s, o, c, vm, K, w, h)), 2)
                row["prism_cuda_fps"] = round(1000.0 / row["prism_cuda_ms"], 1)
            def torch_tiled():
                cov = quats_scales_to_cov3d(q, s, torch)
                proj = project_gaussians(m, cov, vm, K, w, h, torch)
                composite_tiled(proj, c, o, w, h, torch)
            row["prism_torch_ms"] = round(_bench(torch_tiled, iters=5), 2)
            if has_gsplat:
                def gs():
                    rasterization(means=m, quats=q, scales=s, opacities=o, colors=c,
                                  viewmats=vm.unsqueeze(0), Ks=K.unsqueeze(0), width=w, height=h, packed=False)
                gs()
                row["gsplat_ms"] = round(_bench(gs), 2)
                row["gsplat_fps"] = round(1000.0 / row["gsplat_ms"], 1)
            rows.append(row)
            print(row, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"benchmark": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
