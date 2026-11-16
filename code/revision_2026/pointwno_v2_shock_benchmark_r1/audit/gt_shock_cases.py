"""Freeze the GT-based shock case sets, anchors and regions of the VALIDATION cases of all three datasets, BEFORE any
model of this benchmark exists. Ground truth only; no prediction, no model, no test case is read (NACA / RAE: only the
validation rows are indexed; nozzle: the GT block of earlier validation metric files, identical across models).

  NACA  (M 0.65, alpha 3 deg, 120 variable-geometry cases, all supercritical; docs/A1_PREPARATION.md 6, A1_SHOCK_EVIDENCE.md)
        valid shock case = Lovely-Haimes v2 class 'shock' on the upper surface (results/a1_prep/2d_shock_evidence_all_per_case.csv,
        computed 2026-10-14 from the native SU2 .vtu, before any of these models). Anchor s_a = x of max dCp/dx of the GT
        upper-surface Cp of the TRAINING npz within +-0.05 c of the LH candidate x. Shock region (Cp NRMSE) = non-wall
        points with |x - x_c| <= 0.05 and upper-surface y(x) < y <= y_c + 0.10 (x_c, y_c = LH candidate).
        Shock-absent cases (LH no_candidate / smooth_recompression) get NO shock metric; they are reported as such.
  RAE2822 valid shock side = CFD surface class LE_shock / midchord_shock (rae_shock.detect on the native CFD surface,
        arrays up_cls / lo_cls of rae2822_grid256_r1). Anchor = CFD shock position up_xs / lo_xs. Shock region =
        rae_metrics 'shock_band' (Lovely-Haimes cells of the GT grid dilated by 0.02 c; unchanged code).
  NOZZLE valid shock case = GT axis shock present (nozzle_metrics_r1_v1 axis_shock.present_gt, unchanged code); region =
        frozen 'shock' region of nozzle_metrics_r1_v1.
The GT profile of every valid case is run through shock/profile_metrics.rise once here; a GT profile failure is recorded
(the case is then not a thickness case) - thresholds are NOT tuned to remove failures.
  python -B gt_shock_cases.py --naca_npz <npz> --out configs/gt_shock_cases.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "shock"))
import bm_common as B  # noqa: E402
import profile_metrics as PM  # noqa: E402

NACA_LH_CSV = os.path.join(B.EAAI, "results", "a1_prep", "2d_shock_evidence_all_per_case.csv")
NACA_REGION_DX, NACA_REGION_DY = 0.05, 0.10
NACA_ANCHOR_DX = 0.05
NOZZLE_GT_SOURCE = os.path.join(B.EAAI, "results", "revision_2026", "shock_evidence_pkg_r1", "qa_w32_staging", "eval")


def naca_upper_surface(a):
    """Rows of the npz case array a (x, y, mx, my, Cp): wall points = rows with zero momentum, stored first and contiguous,
    upper surface LE -> TE (row 0 .. argmax x), then lower TE -> LE."""
    w = np.where((a[:, 2] == 0) & (a[:, 3] == 0))[0]
    assert len(w) > 100 and np.all(np.diff(w) == 1) and w[0] == 0, "wall rows not contiguous at the start"
    k = int(np.argmax(a[w, 0]))
    up = w[:k + 1]
    assert np.all(np.diff(a[up, 0]) >= -1e-3), "upper surface not LE -> TE (beyond the leading-edge nose)"
    assert a[up, 1].mean() > a[w[k:], 1].mean(), "upper / lower surface swapped"
    return w, up


def naca_region(a, wall_rows, up_rows, xc, yc):
    xs_u, ys_u = a[up_rows, 0], a[up_rows, 1]
    o = np.argsort(xs_u, kind="stable")
    yu = np.interp(a[:, 0], xs_u[o], ys_u[o])
    isw = np.zeros(len(a), bool)
    isw[wall_rows] = True
    return (~isw) & (np.abs(a[:, 0] - xc) <= NACA_REGION_DX) & (a[:, 1] > yu) & (a[:, 1] <= yc + NACA_REGION_DY)


def naca(npz_path):
    if B.sha256_file(npz_path) != B.NACA_NPZ_SHA256:
        raise SystemExit("NACA npz sha256 differs from the registered one")
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    val, test = reg["split"]["val_idx"], set(reg["split"]["test_idx"])
    assert not set(val) & test
    src = np.load(npz_path, allow_pickle=True)
    lh = {int(r["case_id"]): r for r in csv.DictReader(open(NACA_LH_CSV))}
    out = {"source": {"npz_sha256": B.NACA_NPZ_SHA256, "lh_csv": os.path.relpath(NACA_LH_CSV, B.EAAI),
                      "lh_csv_sha256": B.sha256_file(NACA_LH_CSV), "split_from": os.path.relpath(B.NACA_FNO_REGISTRY, B.EAAI)},
           "minf": B.NACA_MINF, "region": {"dx": NACA_REGION_DX, "dy_above_candidate": NACA_REGION_DY},
           "anchor_rule": f"argmax dCp/dx of GT upper-surface Cp within +-{NACA_ANCHOR_DX} c of the LH candidate x",
           "val_ids": val, "cases": {}}
    for g in val:
        a = np.asarray(src["data"][g], np.float64)
        r = lh[g]
        grp, cls = r["case_v2_group"], r["upper_v2"]
        rec = {"filename": str(src["filenames"][g]), "n_points": int(a.shape[0]), "lh_group": grp, "lh_upper_class": cls,
               "lh_upper_M1": float(r["upper_M1"]) if r["upper_M1"] not in ("", "nan") else None}
        wall, up = naca_upper_surface(a)
        rec["n_wall"], rec["n_upper"] = int(len(wall)), int(len(up))
        if grp == "shock_present":
            xc, yc = float(r["upper_x"]), float(r["upper_y"])
            s, q = a[up, 0], a[up, 4]
            win = np.abs(s - xc) <= NACA_ANCHOR_DX
            g_ = np.gradient(q, s)
            ia = np.where(win)[0][int(np.argmax(g_[win]))]
            s_a = float(s[ia])
            rg = PM.rise(s, q, s_a)
            reg_m = naca_region(a, wall, up, xc, yc)
            rec.update({"valid_shock": True, "candidate_xy": [xc, yc], "anchor_x": s_a,
                        "lh_surf_jump_x": float(r["upper_surf_jump_x"]), "region_n_points": int(reg_m.sum()),
                        "gt_profile": {k: rg.get(k) for k in ("fail", "q1", "q2", "jump", "s10", "s90", "width", "loc")}})
        else:
            rec.update({"valid_shock": False, "reason": f"LH group {grp} (upper class {cls})"})
        out["cases"][str(g)] = rec
    out["n_valid_shock"] = sum(c["valid_shock"] for c in out["cases"].values())
    out["n_gt_profile_failed"] = sum(1 for c in out["cases"].values() if c["valid_shock"] and c["gt_profile"]["fail"])
    return out


def rae(data_dir):
    if B.sha256_file(os.path.join(data_dir, "SHA256SUMS.txt")) != B.RAE_DATA_SUMS_SHA256:
        raise SystemExit("RAE data SHA256SUMS differs")
    B.add_path(B.RAE_EVAL_PKG, os.path.join(B.RAE_CODE, "rae2822_pure_r1"))
    import eval_point as EP  # noqa: E402  (unchanged)
    import rae_metrics as M  # noqa: E402
    from pathlib import Path
    env = EP.metric_setup(Path(data_dir))
    val_rows = [int(i) for i in np.where(env["split"] == "val")[0]]
    up_xs, lo_xs = np.load(os.path.join(data_dir, "up_xs.npy")), np.load(os.path.join(data_dir, "lo_xs.npy"))
    af, iu, il, gm = env["af"], env["iu"], env["il"], env["gm"]
    out = {"source": {"data_sums_sha256": B.RAE_DATA_SUMS_SHA256, "class_rule": "CFD surface class in rae_metrics.SHOCK_CLASSES",
                      "anchor": "CFD shock x (up_xs / lo_xs)", "profile": "GT grid Cp sampled at airfoil.npy points (eval_point "
                      "GridMap.sample, unchanged)"},
           "shock_classes": list(M.SHOCK_CLASSES), "val_rows": val_rows, "cases": {}}
    for row in val_rows:
        Y = np.asarray(env["targets"][row], np.float64)
        cp_ref, _ = gm.sample(Y[:1], af)
        rec = {"up_cls": str(env["up_cls"][row]), "lo_cls": str(env["lo_cls"][row]), "sides": {}}
        for side, ix, xs in (("up", iu, up_xs), ("lo", il, lo_xs)):
            if rec[f"{side}_cls"] in M.SHOCK_CLASSES:
                s_a = float(xs[row])
                rg = PM.rise(af[ix, 0], cp_ref[ix, 0], s_a)
                rec["sides"][side] = {"anchor_x": s_a, "gt_profile": {k: rg.get(k) for k in ("fail", "q1", "q2", "jump", "width", "loc")}}
        rec["valid_shock"] = bool(rec["sides"])
        out["cases"][str(row)] = rec
    out["n_val"] = len(val_rows)
    out["n_valid_shock_cases"] = sum(c["valid_shock"] for c in out["cases"].values())
    out["n_valid_shock_sides"] = sum(len(c["sides"]) for c in out["cases"].values())
    out["n_gt_profile_failed_sides"] = sum(1 for c in out["cases"].values() for s in c["sides"].values() if s["gt_profile"]["fail"])
    return out


def nozzle():
    files = sorted(os.listdir(NOZZLE_GT_SOURCE))
    gts, shas = [], {}
    for d in files:
        p = os.path.join(NOZZLE_GT_SOURCE, d, "metrics_val.json")
        m = B.read_json(p)
        shas[d] = B.sha256_file(p)
        gts.append({c: {"present_gt": v["axis_shock"]["present_gt"], "gt": v["axis_shock"]["gt"]} for c, v in m["cases"].items()})
    assert all(json.dumps(g, sort_keys=True) == json.dumps(gts[0], sort_keys=True) for g in gts), "GT blocks differ across files"
    return {"source": {"files": {d: os.path.relpath(os.path.join(NOZZLE_GT_SOURCE, d, "metrics_val.json"), B.EAAI) for d in files},
                       "sha256": shas, "note": "GT block of nozzle_metrics_r1_v1 (model independent; identical in all 6 files)"},
            "val_ids": sorted(gts[0]), "cases": {c: {"valid_shock": bool(v["present_gt"]), "x_s_gt": v["gt"]["x_s"],
                                                      "dP_gt": v["gt"]["dP"], "width_10_90_gt": v["gt"]["width_10_90"]}
                                                  for c, v in gts[0].items()},
            "n_valid_shock": sum(bool(v["present_gt"]) for v in gts[0].values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naca_npz", required=True)
    ap.add_argument("--rae_data", default=B.RAE_DATA_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"refusing to overwrite {a.out}")
    rec = {"version": "bm_gt_shock_cases_v1", "frozen_before_any_benchmark_model": True, "profile_version": PM.PROFILE_VERSION,
           "profile_half_window": PM.HALF, "NACA": naca(a.naca_npz), "RAE2822": rae(a.rae_data), "NOZZLE": nozzle()}
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True, default=str)
    print(json.dumps({"NACA_valid": rec["NACA"]["n_valid_shock"], "NACA_gt_fail": rec["NACA"]["n_gt_profile_failed"],
                      "RAE_valid_cases": rec["RAE2822"]["n_valid_shock_cases"], "RAE_sides": rec["RAE2822"]["n_valid_shock_sides"],
                      "RAE_gt_fail": rec["RAE2822"]["n_gt_profile_failed_sides"], "NOZZLE_valid": rec["NOZZLE"]["n_valid_shock"]}))
    print("GT_SHOCK_CASES=WRITTEN")


if __name__ == "__main__":
    main()
