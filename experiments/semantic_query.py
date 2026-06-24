#!/usr/bin/env python3
"""Open-vocabulary text query over the distilled semantic groups (runs in .dbs_venv).

Uses the DINO-distilled groups from `semantic_distill.py`. For a text query, render
each group in isolation, CLIP-image-embed it, compare to the CLIP text embedding,
and highlight the best-matching group. A group-level open-vocab query — honest CLIP
usage (image+text embeddings), no faked dense features.
"""
import argparse
import math
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "/tmp/dbs")
import numpy as np
import torch
import imageio.v2 as imageio
import open_clip
from gsplat.rendering import rasterization
from scene import Scene, BetaModel

_C0 = 0.28209479177387814


def load(source, model_path, sb_number):
    args = Namespace(sh_degree=0, sb_number=sb_number, source_path=source, model_path=model_path,
                     images="images", resolution=-1, white_background=False, data_device="cuda",
                     eval=True, cap_max=1000000, init_type="sfm")
    bm = BetaModel(0, sb_number); sc = Scene(args, bm, load_iteration=-1, shuffle=False)
    return sc, bm


def K_of(cam):
    K = torch.zeros((3, 3), device="cuda")
    K[0, 0] = 0.5 * cam.image_width / math.tan(cam.FoVx / 2)
    K[1, 1] = 0.5 * cam.image_height / math.tan(cam.FoVy / 2)
    K[0, 2] = cam.image_width / 2; K[1, 2] = cam.image_height / 2; K[2, 2] = 1
    return K


def render(means, quats, scales, opac, betas, colors, cam):
    K = K_of(cam); vm = cam.world_view_transform.transpose(0, 1).unsqueeze(0)
    W, H = int(cam.image_width), int(cam.image_height)
    out, _, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opac, betas=betas,
                              colors=colors, viewmats=vm, Ks=K.unsqueeze(0), width=W, height=H,
                              sb_number=None, sb_params=None, sh_degree=None,
                              backgrounds=torch.ones(1, 3, device="cuda"))
    return out[0, ..., :3].clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(Path(__file__).resolve().parent.parent / "data/tanks/truck"))
    ap.add_argument("--model", default="/tmp/dbs_out/truck_beta")
    ap.add_argument("--feat", default="/tmp/dbs_out/truck_beta/carrier_features.npz")
    ap.add_argument("--sb-number", type=int, default=2)
    ap.add_argument("--queries", nargs="+", default=["a truck", "a wheel", "the ground", "a building"])
    ap.add_argument("--highlight", default="a wheel")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "docs/semantic_query_truck.png"))
    ap.add_argument("--view", type=int, default=20)
    a = ap.parse_args()

    sc, bm = load(a.source, a.model, a.sb_number)
    means = bm.get_xyz.detach(); quats = bm.get_rotation.detach(); scales = bm.get_scaling.detach()
    opac = bm.get_opacity.squeeze(-1).detach(); betas = bm.get_beta.squeeze().detach()
    albedo = (0.5 + _C0 * bm._sh0.detach().squeeze(1)).clamp(0, 1)
    labels = torch.tensor(np.load(a.feat)["labels"], device="cuda")
    cam = sorted(sc.getTrainCameras(), key=lambda c: getattr(c, "image_name", "0"))[a.view]

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip = clip.cuda().eval(); tok = open_clip.get_tokenizer("ViT-B-32")
    from torchvision.transforms.functional import resize, normalize

    def clip_img_embed(img_hwc):
        x = img_hwc.permute(2, 0, 1).unsqueeze(0)
        x = resize(x, [224, 224], antialias=True)
        x = normalize(x, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        with torch.no_grad():
            return torch.nn.functional.normalize(clip.encode_image(x.cuda()), dim=-1)

    groups = sorted(int(g) for g in labels.unique().tolist() if g >= 0)
    gemb = {}
    for g in groups:
        m = labels == g
        masked_opac = torch.where(m, opac, torch.zeros_like(opac))
        img = render(means, quats, scales, masked_opac, betas, albedo, cam)
        gemb[g] = clip_img_embed(img)
    with torch.no_grad():
        temb = torch.nn.functional.normalize(clip.encode_text(tok(a.queries).cuda()), dim=-1)

    print("=== group-level open-vocab query (cosine ×100) ===", flush=True)
    G = torch.cat([gemb[g] for g in groups], 0)            # [n_groups, D]
    sims = (temb @ G.T)                                     # [n_queries, n_groups]
    for qi, q in enumerate(a.queries):
        best = groups[int(sims[qi].argmax())]
        row = " ".join(f"g{g}:{sims[qi, j]*100:4.1f}" for j, g in enumerate(groups))
        print(f"  {q:14s} -> group {best}   [{row}]", flush=True)

    # highlight the best group for the chosen query
    qi = a.queries.index(a.highlight) if a.highlight in a.queries else 0
    best = groups[int(sims[qi].argmax())]
    hl = torch.where((labels == best).unsqueeze(-1), albedo,
                     albedo.mean(-1, keepdim=True).repeat(1, 3) * 0.35)   # grey out the rest
    img = render(means, quats, scales, opac, betas, hl, cam)
    imageio.imwrite(a.out, (img.cpu().numpy() * 255).astype("uint8"))
    print(f"highlighted '{a.highlight}' = group {best} -> {a.out}")


if __name__ == "__main__":
    main()
