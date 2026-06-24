#!/usr/bin/env python3
"""Train DBS with a UNIFORM, frozen Beta shape (the best *single* type), keeping
everything else (incl. spherical-Beta colour) identical to full DBS.

Used for Track 1b: does *adaptive per-region* β (learned, full DBS) beat the
*best single* uniform β? This is the routing-vs-single-type test, with colour held
constant so only the kernel-shape DOF differs.

  UNIFORM_BETA=<activated β> python dbs_uniform_beta.py -s <truck> --eval \
      --sb_number 2 --beta_lr 0 --iterations 30000 --disable_viewer \
      --model_path <out>

The monkeypatch sets every carrier's β to the constant at init; `--beta_lr 0`
freezes it. sb colour stays learnable (sb_number 2), matching full DBS.
"""
import math
import os
import sys

sys.path.insert(0, "/tmp/dbs")

import torch
import torch.nn as nn
from scene import beta_model as BM

_TARGET = float(os.environ["UNIFORM_BETA"])      # activated β = 4*exp(_beta)
_orig_create = BM.BetaModel.create_from_pcd


def _patched_create(self, *a, **k):
    _orig_create(self, *a, **k)
    raw = math.log(_TARGET / 4.0)                # invert beta_activation
    self._beta = nn.Parameter(torch.full_like(self._beta.data, raw).requires_grad_(True))
    print(f"[uniform-beta] frozen β = {_TARGET} (raw {raw:.4f}) on {self._beta.shape[0]} carriers", flush=True)


BM.BetaModel.create_from_pcd = _patched_create

# Hand control to DBS's own training entrypoint with the patched model class.
import runpy

runpy.run_path("/tmp/dbs/train.py", run_name="__main__")
