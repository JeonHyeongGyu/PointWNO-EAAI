"""Write data/naca/normalization_stats.json: the train-split min-max statistics exactly as computed by the unchanged
loader (vendored manuscript script, load_allpoints_npz). CPU, reads the dataset only.

  python -B tools/export_normalization_stats.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from pointwno.api import TN, B  # noqa: E402

npz = os.path.join(REPO, "data", "naca", "Data_allpoints_xymomcp_from_dat.npz")
assert B.sha256_file(npz) == B.NACA_NPZ_SHA256
r = TN.OS.load_allpoints_npz(npz, knn_k=6)
stats, tr, va, te = r[9], r[6], r[7], r[8]
out = {"source": "load_allpoints_npz of code/revision_2026/2d_original_r1/vendored/"
                  "pcwno_vrg_lpe_torch_airfoil_momentum_allpoints_fromdat_masked_steplr.py (unchanged)",
       "npz_sha256": B.NACA_NPZ_SHA256, "computed_on": "training split only (96 cases)",
       "formula": "a_norm = (a - min) / (max - min + 1e-12)",
       "stats": {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
       "split": {"train": [int(i) for i in tr], "val": [int(i) for i in va], "test": [int(i) for i in te]}}
with open(os.path.join(REPO, "data", "naca", "normalization_stats.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["stats"], indent=1))
