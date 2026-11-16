"""Re-evaluate the released NACA checkpoints with the UNCHANGED frozen evaluator (eval_naca_bm.evaluate, bm_naca_eval_v1)
and compare the result with the official evaluation records shipped in this repository.

Validation set only (12 airfoils). The 12 test airfoils are overwritten with NaN / mask 0 right after loading by the
unchanged loader and are never evaluated.

  python -B evaluation/evaluate_released.py --npz data/naca/Data_allpoints_xymomcp_from_dat.npz [--device cuda] [--only V2J2_s0,FNOM_s1]
  python -B evaluation/evaluate_released.py --compare_only

Requires the external upstream WNO file (python -B tools/fetch_wno_upstream.py); see pointwno/_deps.py for the two
research files that are not redistributed and how their registry pins are checked.
One deviation from eval_naca_bm.verify_run, and only this one: the released run directories do not contain
last_model.pt (the final-epoch weights; not used by any result). SHA256SUMS_run.txt still lists it, so a missing
last_model.pt is recorded as "absent (not released)" instead of failing. Every other listed file must match its SHA-256.
Reload check: the evaluator recomputes the validation loss and requires |recomputed - stored| / stored <= 1e-4 (frozen
RELOAD_TOL). This tolerance was set for re-evaluation on the training GPU (NVIDIA H100 NVL, where the official records
have rel_diff 0). On other GPUs the loss can differ by about 1e-4 (different kernels / summation order), and a run that
fails the check writes reload.json but no metrics. --reload_tol T re-runs with a larger, explicitly recorded tolerance
into a SEPARATE output directory (reproduced_reload_tol_<T>/); the default and the official records are not affected.
Outputs: results/revision_2026/pointwno_v2_shock_benchmark_r1/reproduced[_reload_tol_<T>]/naca/eval/<run_id>/ and comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BM = os.path.join(REPO, "code", "revision_2026", "pointwno_v2_shock_benchmark_r1")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, REPO)
from pointwno import _deps  # noqa: E402

_deps.activate()   # external upstream WNO file (tools/fetch_wno_upstream.py) + release pin policy
sys.path.insert(0, os.path.join(BM, "naca"))
sys.path.insert(0, BM)
import eval_naca_bm as EV  # noqa: E402  (unchanged)

B, TN = EV.B, EV.TN
CKPT = os.path.join(REPO, "checkpoints", "naca")
OFFICIAL = os.path.join(REPO, "results", "revision_2026", "pointwno_v2_shock_benchmark_r1", "collect", "naca", "eval")
OUT_BASE = os.path.join(REPO, "results", "revision_2026", "pointwno_v2_shock_benchmark_r1")
OUT = os.path.join(OUT_BASE, "reproduced")
RELEASED = [(a, s) for a in ("V2J2", "FNOM") for s in B.SEEDS]
KEYS = ["val_loss", "global_nrmse_mean", "P1_shock_p_nrmse", "P2_thickness_abs_log", "thickness_ratio", "location_abs_err",
        "jump_abs_rel_err", "grad_err", "post_overshoot", "pre_undershoot", "tv_abs_rel_err", "spurious_osc"]


def verify_released(m, reg_fno):
    """= eval_naca_bm.verify_run for kind 'new', except that an unreleased last_model.pt is recorded, not fatal."""
    rd = m["run_dir"]
    for n in ("complete.json", "SHA256SUMS_run.txt", "run_meta.json", "val_summary.json", "pcwno_best.pt"):
        if not os.path.isfile(os.path.join(rd, n)):
            raise RuntimeError(f"{m['model_id']}: missing {n} in {rd}")
    check = {}
    for line in open(os.path.join(rd, "SHA256SUMS_run.txt")):
        s, n = line.rstrip("\n").split("  ", 1)
        p = os.path.join(rd, n)
        if not os.path.isfile(p) and n == "last_model.pt":
            check[n] = "absent (not released)"
        elif B.sha256_file(p) != s:
            raise RuntimeError(f"{m['model_id']}: sha256 mismatch {n}")
        else:
            check[n] = "ok"
    meta, vs = B.read_json(os.path.join(rd, "run_meta.json")), B.read_json(os.path.join(rd, "val_summary.json"))
    cfg = meta["provenance"]["config"]
    assert cfg["run_id"] == m["model_id"] and not cfg["smoke"], "config"
    assert B.read_status(rd)["status"] == "DONE"
    C = cfg["C"]
    assert int(C["seed"]) == m["seed"], "seed"
    for k in EV.PROTO_KEYS:
        assert C[k] == reg_fno["config"][k], ("protocol differs", k, C[k], reg_fno["config"][k])
    assert int(vs.get("epochs_completed", -1)) == 500, "epochs_completed != 500"
    print(f"  {m['model_id']}: run files {check}", flush=True)
    return meta, vs


def released_models():
    arms = {a["arm"]: a for a in B.registry()["datasets"]["naca"]["arms"]}
    return [{"model_id": B.run_id("naca", a, s), "arm": a, "seed": s, "kind": "new",
             "run_dir": os.path.join(CKPT, B.run_id("naca", a, s)), "arm_rec": arms[a]} for a, s in RELEASED]


def compare(eroot, out):
    rows, worst = [], 0.0
    for a, s in RELEASED:
        rid = B.run_id("naca", a, s)
        po, pr = os.path.join(OFFICIAL, rid, "model_metrics.json"), os.path.join(eroot, rid, "model_metrics.json")
        if not os.path.isfile(pr):
            pl = os.path.join(eroot, rid, "reload.json")
            if os.path.isfile(pl):
                rl = B.read_json(pl)
                rows.append({"run_id": rid, "metric": "reload_check", "official": rl["stored_best_val_loss"],
                             "reproduced": rl["recomputed_val_loss"], "rel_diff": rl["rel_diff"], "status": f"FAIL (tol {rl['tol']:g}); no metrics"})
            else:
                rows.append({"run_id": rid, "metric": "-", "status": "not re-evaluated"})
            continue
        o, r = B.read_json(po)["metrics"], B.read_json(pr)["metrics"]
        for k in KEYS:
            vo, vr = o[k]["value"], r[k]["value"]
            rel = abs(vr - vo) / abs(vo) if (math.isfinite(vo) and vo != 0) else float("nan")
            worst = max(worst, rel) if math.isfinite(rel) else worst
            rows.append({"run_id": rid, "metric": k, "official": vo, "reproduced": vr, "rel_diff": rel,
                         "n_official": f"{o[k]['n_finite']}/{o[k]['n_used']}", "n_reproduced": f"{r[k]['n_finite']}/{r[k]['n_used']}"})
        rl = B.read_json(os.path.join(eroot, rid, "reload.json"))
        rows.append({"run_id": rid, "metric": "reload_check", "official": rl["stored_best_val_loss"], "reproduced": rl["recomputed_val_loss"],
                     "rel_diff": rl["rel_diff"], "status": f"PASS (tol {rl['tol']:g})" if rl["pass"] else "FAIL"})
    os.makedirs(out, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"comparison: {os.path.relpath(os.path.join(out, 'comparison.csv'), REPO)}; max relative difference of the metrics {worst:.3e}")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(REPO, "data", "naca", "Data_allpoints_xymomcp_from_dat.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma-separated <arm>_s<seed>, e.g. V2J2_s0")
    ap.add_argument("--compare_only", action="store_true")
    ap.add_argument("--reload_tol", type=float, default=None,
                    help="cross-hardware re-evaluation only: reload tolerance instead of the frozen 1e-4 (separate output dir)")
    a = ap.parse_args()
    out = OUT
    if a.reload_tol is not None:
        out = os.path.join(OUT_BASE, f"reproduced_reload_tol_{a.reload_tol:g}")
        EV.RELOAD_TOL = a.reload_tol
        print(f"NOTE reload tolerance {a.reload_tol:g} instead of the frozen {1e-4:g}; output {os.path.relpath(out, REPO)}", flush=True)
    eroot = os.path.join(out, "naca", "eval")
    if not a.compare_only:
        import torch
        B.assert_out_root_safe(out, eaai=REPO)
        TN.verify_dependencies()
        if B.sha256_file(a.npz) != B.NACA_NPZ_SHA256:
            sys.exit("NACA npz sha256 differs from the registered dataset")
        dev = torch.device(a.device)
        gtc = B.read_json(B.GT_CASES)["NACA"]
        D = EV.load_data(a.npz, dev)
        assert D["val_ids"] == gtc["val_ids"]
        EV.verify_run = verify_released
        todo = released_models()
        if a.only:
            todo = [m for m in todo if f"{m['arm']}_s{m['seed']}" in a.only.split(",")]
        os.makedirs(eroot, exist_ok=True)
        failed = []
        for m in todo:
            od = os.path.join(eroot, m["model_id"])
            if os.path.isfile(os.path.join(od, "DONE_val")):
                print(f"skip {m['model_id']} (already evaluated: {os.path.relpath(od, REPO)})")
                continue
            if os.path.exists(od):
                if os.path.isfile(os.path.join(od, "reload.json")) and len(os.listdir(od)) <= 2:
                    print(f"skip {m['model_id']} (reload check already failed at this tolerance: {os.path.relpath(od, REPO)})")
                    failed.append(m["model_id"])
                    continue
                raise SystemExit(f"{od} exists without DONE_val; remove it by hand")
            try:
                EV.evaluate(m, D, gtc, dev, od)
                print(f"EVAL=PASS {m['model_id']}", flush=True)
            except RuntimeError as e:
                failed.append(m["model_id"])
                print(f"EVAL=FAIL {m['model_id']}: {e}", flush=True)
        print(f"evaluated {len(todo) - len(failed)}/{len(todo)}; failed {failed}", flush=True)
        B.atomic_json({"release_dependencies": _deps.record(), "reload_tol": EV.RELOAD_TOL, "device": str(dev),
                       "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "torch": torch.__version__,
                       "evaluated": [m["model_id"] for m in todo if m["model_id"] not in failed], "failed": failed,
                       "note": "re-evaluation with this repository; not the official records"},
                      os.path.join(out, "evaluation_record.json"))
    compare(eroot, out)


if __name__ == "__main__":
    main()
