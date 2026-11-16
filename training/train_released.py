"""Training launcher for the released configurations (PointWNO V2J2, PointFNO FNOM; seeds 0-2).

Runs the UNCHANGED trainer code/revision_2026/pointwno_v2_shock_benchmark_r1/naca/train_naca_bm.py (same arguments)
after pointwno._deps.activate(), which provides the external upstream WNO file and the release pin policy
(see pointwno/_deps.py). Writes RELEASE_DEPENDENCIES.json into --out_root (not with --dry_run).

  python -B training/train_released.py --config code/revision_2026/pointwno_v2_shock_benchmark_r1/configs/naca/BM_NACA_V2J2_s0.json \
      --npz data/naca/Data_allpoints_xymomcp_from_dat.npz --out_root results/revision_2026/pointwno_v2_shock_benchmark_r1 [--dry_run]

Not verified for this release: a full 500-epoch run (the released checkpoints were not retrained).
"""
import json
import os
import runpy
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from pointwno import _deps  # noqa: E402

_deps.activate()
TRAINER = os.path.join(REPO, "code", "revision_2026", "pointwno_v2_shock_benchmark_r1", "naca", "train_naca_bm.py")
args = sys.argv[1:]
if "--dry_run" not in args and "--out_root" in args:
    out = os.path.abspath(args[args.index("--out_root") + 1])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "RELEASE_DEPENDENCIES.json"), "w") as f:
        json.dump(_deps.record(), f, indent=1)
sys.path.insert(0, os.path.dirname(TRAINER))
sys.argv = [TRAINER] + args
runpy.run_path(TRAINER, run_name="__main__")
