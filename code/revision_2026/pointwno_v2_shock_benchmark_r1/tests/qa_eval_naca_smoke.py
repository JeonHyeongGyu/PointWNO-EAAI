"""Local QA only: run the NACA evaluator end to end on SMOKE checkpoints (2 epochs). The only relaxation is verify_run (a
smoke run is not a 500-epoch registered run); everything else (checkpoint load, reload check, point order, GT regions,
profile metrics, failure counts, model_metrics.json) is the production code path.
  python -B tests/qa_eval_naca_smoke.py --npz P --smoke_root <root with naca/smoke/*> --out <dir below the benchmark root>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "naca"))
import eval_naca_bm as EN  # noqa: E402

B, TN = EN.B, EN.TN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--smoke_root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B.assert_out_root_safe(os.path.abspath(a.out))
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtc = B.read_json(B.GT_CASES)["NACA"]
    D = EN.load_data(a.npz, dev)

    def verify_smoke(m, reg_fno):
        rd = m["run_dir"]
        assert B.read_status(rd)["status"] == "DONE"
        meta, vs = B.read_json(os.path.join(rd, "run_meta.json")), B.read_json(os.path.join(rd, "val_summary.json"))
        return meta, vs

    EN.verify_run = verify_smoke
    res = {}
    for arm in ("V2J2", "FNOM"):
        rid = B.run_id("naca", arm, 0, smoke=True)
        m = {"model_id": rid, "arm": arm, "seed": 0, "kind": "new", "run_dir": os.path.join(a.smoke_root, "naca", "smoke", rid),
             "arm_rec": B.arm("naca", arm)}
        od = os.path.join(a.out, rid)
        EN.evaluate(m, D, gtc, dev, od)
        mm = B.read_json(os.path.join(od, "model_metrics.json"))
        rl = B.read_json(os.path.join(od, "reload.json"))
        ok = rl["pass"] and mm["metrics"]["P1_shock_p_nrmse"]["n_used"] == 7 and math.isfinite(mm["metrics"]["global_nrmse_mean"]["value"])
        res[rid] = {"reload_rel": rl["rel_diff"], "P1": mm["metrics"]["P1_shock_p_nrmse"], "P2": mm["metrics"]["P2_thickness_abs_log"],
                    "failures": {k: (v["count"], v["den"]) for k, v in mm["failures"].items()}, "ok": ok,
                    "infer_ms": mm["cost"]["infer_ms_per_case_median"]}
    print(json.dumps(res, indent=1, default=str))
    print("QA_EVAL_NACA=" + ("PASS" if all(r["ok"] for r in res.values()) else "FAIL"))


if __name__ == "__main__":
    main()
