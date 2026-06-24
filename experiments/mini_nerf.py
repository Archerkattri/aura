"""Compact from-scratch NeRF (frequency-encoded MLP + volume rendering) for the
lineage comparison figure. Trains on the COLMAP cameras (full poses) and exposes
render(frame, scale) -> (W,H,flat_rgb). Not SOTA — a faithful, recognizable NeRF
to represent the NeRF stage in Photogrammetry -> NeRF -> 3DGS -> AURA.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image


def _quat_to_R(qw, qx, qy, qz, device):
    n = math.sqrt(qw*qw+qx*qx+qy*qy+qz*qz); qw,qx,qy,qz = qw/n,qx/n,qy/n,qz/n
    return torch.tensor([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]], dtype=torch.float32, device=device)


def _posenc(x, L):
    out = [x]
    for i in range(L):
        f = 2.0 ** i
        out += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self, Lx=10, Ld=4, W=256):
        super().__init__()
        self.Lx, self.Ld = Lx, Ld
        dx = 3 + 3 * 2 * Lx; dd = 3 + 3 * 2 * Ld
        self.b1 = nn.Sequential(nn.Linear(dx, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU(),
                                nn.Linear(W, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU())
        self.b2 = nn.Sequential(nn.Linear(W + dx, W), nn.ReLU(), nn.Linear(W, W), nn.ReLU())
        self.sigma = nn.Linear(W, 1)
        self.feat = nn.Linear(W, W)
        self.rgb = nn.Sequential(nn.Linear(W + dd, W // 2), nn.ReLU(), nn.Linear(W // 2, 3))

    def forward(self, x, d):
        ex = _posenc(x, self.Lx); ed = _posenc(d, self.Ld)
        h = self.b1(ex); h = self.b2(torch.cat([h, ex], -1))
        sigma = torch.relu(self.sigma(h)).squeeze(-1)
        rgb = torch.sigmoid(self.rgb(torch.cat([self.feat(h), ed], -1)))
        return sigma, rgb


def train_nerf(colmap_dir, image_root, scale=0.125, iters=6000, device="cuda", log=print):
    from aura.ingest.colmap import load_colmap_model
    cams, images, points, _ = load_colmap_model(colmap_dir)
    cam = cams[images[0].camera_id]; p = cam.params
    fx, fy = (p[0], p[1]) if len(p) >= 4 else (p[0], p[0])
    cx, cy = (p[2], p[3]) if len(p) >= 4 else (cam.width/2, cam.height/2)
    W = max(1, int(cam.width*scale)); H = max(1, int(cam.height*scale))
    fxs, fys, cxs, cys = fx*scale, fy*scale, cx*scale, cy*scale
    # scene bounds from point depths
    xyz = torch.tensor([list(pt.xyz) for pt in points], dtype=torch.float32, device=device)
    cam_data = []
    for im in images:
        ip = Path(image_root) / im.name
        if not ip.exists():
            continue
        R = _quat_to_R(im.qw, im.qx, im.qy, im.qz, device)  # world->cam
        t = torch.tensor([im.tx, im.ty, im.tz], device=device)
        origin = -R.T @ t
        img = torch.from_numpy(np.asarray(Image.open(ip).convert("RGB").resize((W, H)), dtype=np.float32)/255).to(device)
        cam_data.append((R, origin, img, im.name))
    # near/far from points in front of a sample camera
    with torch.no_grad():
        R0, o0, _, _ = cam_data[0]
        zc = (xyz - o0) @ R0[2]
        near = max(float(zc[zc > 0].quantile(0.02)), 0.05); far = float(zc[zc > 0].quantile(0.98)) * 1.2

    net = NeRF().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    dirs_cam = torch.stack([(xs + 0.5 - cxs)/fxs, (ys + 0.5 - cys)/fys, torch.ones_like(xs)], -1).reshape(-1, 3)

    def render_rays(R, origin, pix_idx, n_samples=64, perturb=True):
        d_cam = dirs_cam[pix_idx]
        d_world = d_cam @ R  # cam->world = R^T applied as row-vec @ R
        d_world = d_world / d_world.norm(dim=-1, keepdim=True)
        tvals = torch.linspace(0, 1, n_samples, device=device)
        z = near * (1 - tvals) + far * tvals
        z = z.expand(pix_idx.shape[0], n_samples).clone()
        if perturb:
            mid = 0.5 * (z[:, 1:] + z[:, :-1])
            z = z + torch.rand_like(z) * 0  # keep simple/stable
        pts = origin[None, None, :] + d_world[:, None, :] * z[:, :, None]
        dd = d_world[:, None, :].expand(-1, n_samples, -1)
        sigma, rgb = net(pts.reshape(-1, 3), dd.reshape(-1, 3))
        sigma = sigma.reshape(pix_idx.shape[0], n_samples); rgb = rgb.reshape(pix_idx.shape[0], n_samples, 3)
        delta = torch.cat([z[:, 1:] - z[:, :-1], torch.full((z.shape[0], 1), 1e10, device=device)], -1)
        alpha = 1 - torch.exp(-sigma * delta)
        T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1 - alpha + 1e-10], -1), -1)[:, :-1]
        w = alpha * T
        return (w[..., None] * rgb).sum(1)

    npix = H * W
    for it in range(iters):
        R, origin, img, _ = cam_data[it % len(cam_data)]
        idx = torch.randint(0, npix, (2048,), device=device)
        pred = render_rays(R, origin, idx)
        gt = img.reshape(-1, 3)[idx]
        loss = ((pred - gt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (it + 1) % 1000 == 0:
            log(f"  [nerf] iter {it+1}/{iters} loss={loss.item():.4f}")

    by_name = {n: (R, o) for (R, o, _, n) in cam_data}

    @torch.no_grad()
    def render(frame, frame_scale):
        # match by image basename; reuse the trained scale's R/origin (pose is scale-independent)
        name = Path(frame["image_path"]).name
        R, origin = by_name.get(name, cam_data[0][:2])
        out = torch.zeros(npix, 3, device=device)
        for s in range(0, npix, 4096):
            out[s:s+4096] = render_rays(R, origin, torch.arange(s, min(s+4096, npix), device=device), perturb=False)
        return W, H, out.clamp(0, 1).reshape(-1).cpu().tolist()

    return render
