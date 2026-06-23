#!/usr/bin/env python3
"""Smoking-gun test: train gsplat 3DGS using the FULL COLMAP poses directly
(quaternion+translation, world-to-camera), bypassing the manifest's lossy
camera_origin/look_at representation. If this hits ~24 dB while the manifest
path caps at ~14-16, the manifest camera conversion (which drops roll) is the bug.
"""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import torch
from aura.ingest.colmap import load_colmap_model
from eval_psnr import load_jpg_as_rgb, resize_pixels


def quat_to_R(qw, qx, qy, qz, torch):
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz); qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return torch.tensor([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)],
    ], dtype=torch.float32)


def main():
    dev = "cuda"; scale = 0.25; iters = 7000
    from gsplat import rasterization
    cams, images, points, _ = load_colmap_model("data/tanks/truck/sparse/0")
    cam = cams[images[0].camera_id]
    # intrinsics
    p = cam.params
    if len(p) >= 4:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        fx = fy = p[0]; cx = cam.width/2; cy = cam.height/2
    fullw, fullh = cam.width, cam.height
    w = max(1, int(fullw*scale)); h = max(1, int(fullh*scale))
    K = torch.tensor([[fx*scale,0,cx*scale],[0,fy*scale,cy*scale],[0,0,1.0]], device=dev)

    viewmats = {}
    for im in images:
        R = quat_to_R(im.qw, im.qx, im.qy, im.qz, torch).to(dev)
        t = torch.tensor([im.tx, im.ty, im.tz], device=dev)
        V = torch.eye(4, device=dev); V[:3,:3] = R; V[:3,3] = t
        viewmats[im.name] = V

    xyz = torch.tensor([list(pt.xyz) for pt in points], dtype=torch.float32, device=dev)
    rgb = torch.tensor([[c/255 for c in pt.rgb] for pt in points], dtype=torch.float32, device=dev).clamp(0,1)
    n = xyz.shape[0]
    means = xyz.clone().requires_grad_(True)
    colors = rgb.clone().requires_grad_(True)
    with torch.no_grad():
        d = torch.full((n,), 0.05, device=dev)
        for s in range(0, n, 4096):
            sl = xyz[s:s+4096]; dd = torch.cdist(sl, xyz)
            dd.scatter_(1, torch.arange(s, min(s+4096,n), device=dev).unsqueeze(1), float("inf"))
            d[s:s+4096] = dd.min(1).values.clamp(min=1e-6)
    log_s = torch.log(d).unsqueeze(1).repeat(1,3).clone().requires_grad_(True)
    quats = torch.zeros(n,4,device=dev); quats[:,0]=1; quats=quats.requires_grad_(True)
    op = torch.logit(torch.full((n,),0.1,device=dev)).requires_grad_(True)
    opt = torch.optim.Adam([
        {"params":[means],"lr":1.6e-4},{"params":[log_s],"lr":5e-3},{"params":[quats],"lr":1e-3},
        {"params":[op],"lr":5e-2},{"params":[colors],"lr":2.5e-3}], eps=1e-15)

    names = [im.name for im in images if (Path("data/tanks/truck/images")/im.name).exists()]
    gt_cache = {}
    def gt_of(name):
        if name not in gt_cache:
            gw,gh,g = load_jpg_as_rgb(str(Path("data/tanks/truck/images")/name))
            if (gw,gh)!=(w,h): g = resize_pixels(g,gw,gh,w,h)
            gt_cache[name] = torch.tensor(g,dtype=torch.float32,device=dev).reshape(h,w,3)
        return gt_cache[name]
    def render(name):
        out,_,_ = rasterization(means=means, quats=quats/quats.norm(dim=-1,keepdim=True),
            scales=torch.exp(log_s), opacities=torch.sigmoid(op), colors=colors,
            viewmats=viewmats[name].unsqueeze(0), Ks=K.unsqueeze(0), width=w, height=h)
        return out[0]
    for it in range(iters):
        name = names[it % len(names)]
        img = render(name); loss = torch.abs(img - gt_of(name)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) % 1000 == 0: print(f"  iter {it+1}/{iters} L1={loss.item():.4f}", flush=True)
    # eval on a held-out deterministic subset
    stride = max(1, len(names)//5); sel = names[::stride][:5]; ps=[]
    with torch.no_grad():
        for name in sel:
            img = render(name); mse = ((img-gt_of(name))**2).mean().item()
            ps.append(10*math.log10(1/mse) if mse>0 else 99); print(f"  {name}: {ps[-1]:.2f} dB")
    print(f"\nDIRECT-COLMAP-POSE gsplat: {sum(ps)/len(ps):.2f} dB @ {scale}x  (vs manifest path ~14 dB)")


if __name__ == "__main__":
    main()
