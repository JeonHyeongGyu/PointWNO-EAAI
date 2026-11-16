"""Thin wrappers around the unchanged benchmark code (code/revision_2026/pointwno_v2_shock_benchmark_r1)."""
from __future__ import annotations

import json
import os
import sys

from . import _deps

_deps.activate()   # external upstream WNO file + release pin policy, before any research module is imported

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BM_PKG = os.path.join(REPO, "code", "revision_2026", "pointwno_v2_shock_benchmark_r1")
os.environ.setdefault("MPLBACKEND", "Agg")
for _p in (BM_PKG, os.path.join(BM_PKG, "naca")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bm_common as B  # noqa: E402
import train_naca_bm as TN  # noqa: E402  (sets the import paths of the vendored model code)
import torch  # noqa: E402

ARMS = {"V2J2": "PointWNO (canonical WNO kernel, db2, J = 2)", "FNOM": "PointFNO (parameter-matched, 34 x 26 modes)"}


def run_config(arm: str, seed: int) -> dict:
    """Registered run configuration (configs/naca/BM_NACA_<arm>_s<seed>.json), checked against the benchmark registry."""
    if arm not in ARMS or seed not in B.SEEDS:
        raise ValueError(f"released runs: arms {sorted(ARMS)}, seeds {B.SEEDS}")
    with open(os.path.join(BM_PKG, "configs", "naca", f"BM_NACA_{arm}_s{seed}.json")) as f:
        cfg = json.load(f)
    TN.check_config(cfg)
    return cfg


def build_model(arm: str, seed: int = 0):
    """Model built by the training script's own build_model (same classes, arguments and layer replacement), after
    seeding the global RNG with the run seed as the trainer does. The torch RNG state after construction equals the
    value recorded by the training run (rng_state_sha256_after_model; checked in tests/test_release.py): construction
    starts from the same seed and consumes the same random stream as in training. Load a released checkpoint for the
    trained weights."""
    cfg = run_config(arm, seed)
    TN.OS.set_seed(cfg["C"]["seed"])
    model = TN.build_model(cfg)
    n = count_params(model)
    if n != cfg["expected_params"]:
        raise RuntimeError(f"parameter count {n} != registered {cfg['expected_params']}")
    return model


def count_params(model) -> int:
    """Trainable real parameters (a complex parameter counts twice), as reported in the manuscript."""
    return TN.count_real(model)


def make_vrg(device="cpu", grid_h=128, grid_w=96):
    """The 2-D virtual reference grid used by all airfoil models (identical for every airfoil)."""
    return TN.TF.make_internal_vrg(grid_h, grid_w, stats=None, device=torch.device(device))


def load_checkpoint(model, path, device="cpu"):
    """Load a released pcwno_best.pt (state_dict) with strict key matching."""
    sd = torch.load(path, map_location=device, weights_only=False)
    TN.prime_buffers(model, sd)
    model.load_state_dict(sd, strict=True)
    return model
