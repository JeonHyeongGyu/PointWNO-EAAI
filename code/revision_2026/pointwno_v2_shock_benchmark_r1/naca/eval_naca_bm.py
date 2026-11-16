"""NACA validation evaluation of every benchmark model (KITSM02 GPU; VALIDATION ONLY, test cases blocked as in training).

Models (configs/benchmark_registry.json, dataset 'naca'): new runs V2J2 / FNOM s0-2 (this package, must be DONE) and the
reused F1 (results/fno2d_cmp_r1) and C3 (results/capacity_r1[_seeds]) runs. Every model is verified (SHA256SUMS_run.txt,
complete.json, run_meta config == registered protocol, parameter count) and its best checkpoint is re-evaluated in THIS job:
  reload check  : recomputed validation total loss == stored best_val_loss (rel <= 1e-4; GPU reduction order)
  global        : per-case RMSE of the min-max normalised targets (= NRMSE by the train range; evaluate_and_save formula),
                  mean over x-momentum, y-momentum, Cp
  shock (valid GT cases of configs/gt_shock_cases.json only):
                  P1 Cp NRMSE over the frozen GT region points; P2 + secondary from shock/profile_metrics on the upper-surface
                  Cp (physical) at the frozen anchor; surface shock detection rae_shock.detect(x, Cp, M 0.65) on GT and prediction
  validity      : points with p <= 0 implied by Cp (Cp <= -2 / (gamma M^2)); non-finite predictions
  cost          : inference ms / case (CUDA-synchronised, batch 4 / 4 cases), peak memory; training cost from the run files
Output per model: <out_root>/naca/eval/<model_id>/ {source.json, reload.json, case_metrics.csv, model_metrics.json,
pred_val.npz, DONE_val}; existing DONE_val -> skipped; an incomplete directory is renamed *.incomplete_<t>.
Exit 0 all PASS | 12 a model failed (others still evaluated) | 3 deadline
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PKG, "shock"))
import train_naca_bm as TN  # noqa: E402  (also sets the import paths)
import profile_metrics as PMX  # noqa: E402

B, TF, T, OS = TN.B, TN.TF, TN.T, TN.OS
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

B.add_path(os.path.join(B.RAE_CODE, "rae2822_pure_r1"))
import rae_shock as RS  # noqa: E402  (unchanged surface detector; M_inf passed explicitly)

EVAL_VERSION = "bm_naca_eval_v1"
RELOAD_TOL = 1e-4
GAMMA = 1.4
CP_VACUUM = -2.0 / (GAMMA * B.NACA_MINF ** 2)
EST_MIN = 3.0
SHOCK_CLASSES = ("LE_shock", "midchord_shock")      # = rae_metrics.SHOCK_CLASSES
PROTO_KEYS = ("epochs", "batch", "lr", "step_size", "gamma", "min_lr", "grid_h", "grid_w", "grid_range", "wavelet", "wno_level",
              "wno_width", "wno_layers", "grad_weight", "knn_k")


def models(out_root):
    reg = B.registry()["datasets"]["naca"]
    out = []
    for a in reg["arms"]:
        for s in a["seeds"]:
            if a["status"] == "new":
                out.append({"model_id": B.run_id("naca", a["arm"], s), "arm": a["arm"], "seed": s, "kind": "new",
                            "run_dir": os.path.join(out_root, "naca", "runs", B.run_id("naca", a["arm"], s)), "arm_rec": a})
            elif a["status"] == "reused":
                sd = a["location"]["server_dir"]
                rel = sd[str(s)] if isinstance(sd, dict) else sd.format(seed=s)
                out.append({"model_id": f"BM_NACA_REUSE_{a['arm']}_s{s}", "arm": a["arm"], "seed": s, "kind": "reused",
                            "run_dir": os.path.join(B.EAAI, rel), "arm_rec": a})
    return out


def verify_run(m, reg_fno):
    rd = m["run_dir"]
    for n in ("complete.json", "SHA256SUMS_run.txt", "run_meta.json", "val_summary.json", "pcwno_best.pt"):
        if not os.path.isfile(os.path.join(rd, n)):
            raise RuntimeError(f"{m['model_id']}: missing {n} in {rd}")
    for line in open(os.path.join(rd, "SHA256SUMS_run.txt")):
        s, n = line.rstrip("\n").split("  ", 1)
        if B.sha256_file(os.path.join(rd, n)) != s:
            raise RuntimeError(f"{m['model_id']}: sha256 mismatch {n}")
    meta, vs = B.read_json(os.path.join(rd, "run_meta.json")), B.read_json(os.path.join(rd, "val_summary.json"))
    if m["kind"] == "new":
        cfg = meta["provenance"]["config"]
        assert cfg["run_id"] == m["model_id"] and not cfg["smoke"], "config"
        assert B.read_status(rd)["status"] == "DONE"
        C = cfg["C"]
    else:
        C = meta["config"]
        if m["arm"] == "F1":
            assert meta["registry_sha256"] == B.NACA_FNO_REGISTRY_SHA256 and meta["trial"] == "FNO_F1_L6W32M34"
            assert meta["fno"] == {"width": 32, "layers": 6, "modes": [34, 34]}
        else:
            assert meta.get("backward") == "r1_exact_autograd_v1" and meta["trial"] == "CAP_C3_L6W32"
        assert meta["npz_sha256"] == B.NACA_NPZ_SHA256 and meta["smoke_tag"] is None
        assert vs["test_evaluated"] is False
    assert int(C["seed"]) == m["seed"], "seed"
    for k in PROTO_KEYS:
        assert C[k] == reg_fno["config"][k], ("protocol differs", k, C[k], reg_fno["config"][k])
    assert int(vs.get("epochs_completed", -1)) == 500, "epochs_completed != 500"
    return meta, vs


def build_for(m):
    a = m["arm_rec"]
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    if a["model"]["kernel"] == "v3":
        C = dict(reg["config"], seed=m["seed"])
        mdl = TF.PCWNO(grid_hw=(C["grid_h"], C["grid_w"]), wno_width=C["wno_width"], wno_level=C["wno_level"],
                       wno_layers=C["wno_layers"], wno_wavelet=C["wavelet"], wno_grid_range=tuple(C["grid_range"]))
        import wno2d_r1 as W
        assert W.convert_to_r1(mdl) == C["wno_layers"]
        return mdl
    if m["kind"] == "new":
        return TN.build_model(B.read_json(os.path.join(m["run_dir"], "run_meta.json"))["provenance"]["config"])
    C = dict(reg["config"], seed=m["seed"], fno_core_init_seed_offset=reg["fno_core_init_seed_offset"], fno_width=32, fno_layers=6,
             fno_modes=a["model"]["modes"])
    return TF.build_model(C)


def upper_rows(a):
    w = np.where((a[:, 2] == 0) & (a[:, 3] == 0))[0]
    k = int(np.argmax(a[w, 0]))
    return w, w[:k + 1]


def region_mask(a, wall, up, xc, yc, reg):
    xs_u, ys_u = a[up, 0], a[up, 1]
    o = np.argsort(xs_u, kind="stable")
    yu = np.interp(a[:, 0], xs_u[o], ys_u[o])
    isw = np.zeros(len(a), bool)
    isw[wall] = True
    return (~isw) & (np.abs(a[:, 0] - xc) <= reg["dx"]) & (a[:, 1] > yu) & (a[:, 1] <= yc + reg["dy_above_candidate"])


def evaluate(m, D, gtc, dev, od):
    reg_fno = B.read_json(B.NACA_FNO_REGISTRY)
    meta, vs = verify_run(m, reg_fno)
    model = build_for(m).to(dev)
    n_real = TN.count_real(model)
    if n_real != m["arm_rec"]["params_trainable_real"]:
        raise RuntimeError(f"{m['model_id']}: params {n_real} != {m['arm_rec']['params_trainable_real']}")
    sd = torch.load(os.path.join(m["run_dir"], "pcwno_best.pt"), map_location=dev, weights_only=False)
    TN.prime_buffers(model, sd)
    model.load_state_dict(sd, strict=True)
    model.eval()
    trainer = OS.MaskedGradTrainer(model, lr=1e-3, grad_weight=0.0)
    os.makedirs(od)
    B.atomic_json({"model_id": m["model_id"], "arm": m["arm"], "seed": m["seed"], "kind": m["kind"],
                   "run_dir": os.path.relpath(m["run_dir"], B.EAAI), "best_sha256": B.sha256_file(os.path.join(m["run_dir"], "pcwno_best.pt")),
                   "best_epoch": vs.get("best_epoch"), "params_trainable_real": n_real, "eval_version": EVAL_VERSION},
                  os.path.join(od, "source.json"))
    tot, n, preds, tms = 0.0, 0, {}, []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch in D["val_loader"]:
            xyz, y, mask, kidx = (batch[k].to(dev) for k in ("xyz", "y", "mask", "knn_idx"))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            total, _, _, pred = trainer.eval(xyz, D["grid_ref"], y, mask, kidx)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            tms.append(1000.0 * (time.perf_counter() - t0) / xyz.size(0))
            bs = xyz.size(0)
            tot += total * bs
            n += bs
            for b in range(bs):
                gid = int(batch["case_id"][b])
                preds[gid] = pred[b, :int(batch["n_valid"][b]), :].double().cpu().numpy()
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == "cuda" else float("nan")
    vl = tot / n
    rel = abs(vl - vs["best_val_loss"]) / max(abs(vs["best_val_loss"]), 1e-30)
    rl = {"recomputed_val_loss": vl, "stored_best_val_loss": vs["best_val_loss"], "rel_diff": rel, "tol": RELOAD_TOL,
          "pass": rel <= RELOAD_TOL}
    B.atomic_json(rl, os.path.join(od, "reload.json"))
    if not rl["pass"]:
        raise RuntimeError(f"{m['model_id']}: reload check failed rel {rel:.3e}")
    st = D["stats"]
    lo = np.array([st["u_min"], st["v_min"], st["p_min"]])
    rg = np.array([st["u_max"] - st["u_min"], st["v_max"] - st["v_min"], st["p_max"] - st["p_min"]])
    rows, flat, offs = [], [], [0]
    for gid in D["val_ids"]:
        a = D["raw"][gid]
        pn = preds[gid]
        assert pn.shape[0] == a.shape[0]
        yn = (a[:, 2:5] - lo) / (rg + 1e-12)
        e = pn - yn
        P = pn * (rg + 1e-12) + lo                                        # physical (mx, my, Cp)
        r = {"case": gid, "filename": D["filenames"][gid], "n_points": int(a.shape[0]),
             **{f"nrmse_{t}": float(np.sqrt(np.mean(e[:, k] ** 2))) for k, t in enumerate(T.TARGETS)}}
        r["global_nrmse_mean"] = float(np.mean([r[f"nrmse_{t}"] for t in T.TARGETS]))
        r["n_nonphysical_p"] = int(np.sum(P[:, 2] <= CP_VACUUM))
        r["nonfinite_pred"] = bool(not np.all(np.isfinite(P)))
        c = gtc["cases"][str(gid)]
        r["gt_valid_shock"] = bool(c["valid_shock"])
        wall, up = upper_rows(a)
        x_up, cp_g, cp_p = a[up, 0], a[up, 4], P[up, 2]
        dg, dp = RS.detect(x_up, cp_g, B.NACA_MINF), RS.detect(x_up, cp_p, B.NACA_MINF)
        r["gt_surface_cls"], r["pred_surface_cls"] = dg["cls"], dp["cls"]
        if c["valid_shock"]:
            xc, yc = c["candidate_xy"]
            msk = region_mask(a, wall, up, xc, yc, gtc["region"])
            assert int(msk.sum()) == c["region_n_points"], "frozen region differs"
            r["P1_shock_p_nrmse"] = float(np.sqrt(np.mean((P[msk, 2] - a[msk, 4]) ** 2)) / rg[2])
            met, rgt, rpr = PMX.compare(x_up, cp_p, cp_g, c["anchor_x"])
            r.update({f"surf_{k}": v for k, v in met.items()})
            r["P2_thickness_abs_log"] = met["thickness_abs_log"]
            r["fail_thickness_not_computable"] = rpr["fail"] is not None
            r["gt_surface_detect_is_shock"] = dg["cls"] in SHOCK_CLASSES
            r["fail_pred_shock_not_detected"] = r["gt_surface_detect_is_shock"] and dp["cls"] not in SHOCK_CLASSES
            r["surf_xs_abs_err_detector"] = abs(dp["xs"] - dg["xs"]) if (np.isfinite(dp["xs"]) and np.isfinite(dg["xs"])) else float("nan")
        else:
            r["pred_false_surface_shock"] = dp["cls"] in SHOCK_CLASSES
        rows.append(r)
        flat.append(P.astype(np.float32))
        offs.append(offs[-1] + P.shape[0])
    with open(os.path.join(od, "case_metrics.csv"), "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(od, "pred_val.npz"), case_ids=np.array(D["val_ids"]), offsets=np.array(offs),
                        pred_phys=np.concatenate(flat), columns=np.array(["x_momentum", "y_momentum", "Cp"]))
    mm = summarize(rows, rl, meta, vs, m, tms, peak)
    B.atomic_json(mm, os.path.join(od, "model_metrics.json"))
    with open(os.path.join(od, "DONE_val"), "w") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "eval_version": EVAL_VERSION, "pbs_job": os.environ.get("PBS_JOBID")}))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()


METRIC_MAP = [  # (standard key, case key, valid-shock only)
    ("P1_shock_p_nrmse", "P1_shock_p_nrmse", True), ("P2_thickness_abs_log", "P2_thickness_abs_log", True),
    ("thickness_ratio", "surf_thickness_ratio", True), ("location_abs_err", "surf_location_abs_err", True),
    ("jump_abs_rel_err", "surf_jump_abs_rel_err", True), ("grad_err", "surf_grad_nrmse", True),
    ("post_overshoot", "surf_post_overshoot", True), ("pre_undershoot", "surf_pre_undershoot", True),
    ("tv_abs_rel_err", "surf_tv_abs_rel_err", True), ("spurious_osc", "surf_spurious_osc", True),
    ("global_nrmse_mean", "global_nrmse_mean", False)]


def summarize(rows, rl, meta, vs, m, tms, peak):
    out = {"dataset": "naca", "model_id": m["model_id"], "arm": m["arm"], "seed": m["seed"], "source": m["kind"], "metrics": {},
           "failures": {}, "cost": {}}
    for key, ck, shock in METRIC_MAP:
        use = [r for r in rows if r["gt_valid_shock"]] if shock else rows
        vals = [float(r[ck]) for r in use if ck in r and r[ck] is not None and math.isfinite(float(r[ck]))]
        out["metrics"][key] = {"value": float(np.mean(vals)) if vals else float("nan"), "n_finite": len(vals), "n_used": len(use)}
    out["metrics"]["val_loss"] = {"value": rl["recomputed_val_loss"], "n_finite": 1, "n_used": 1}
    val = [r for r in rows if r["gt_valid_shock"]]
    out["failures"] = {
        "gt_shock_present": {"count": len(val), "den": len(rows)},
        "pred_shock_not_detected": {"count": sum(bool(r.get("fail_pred_shock_not_detected")) for r in val),
                                    "den": sum(bool(r.get("gt_surface_detect_is_shock")) for r in val)},
        "thickness_not_computable": {"count": sum(bool(r.get("fail_thickness_not_computable")) for r in val), "den": len(val)},
        "physically_invalid_pred": {"count": sum(r["n_nonphysical_p"] > 0 or r["nonfinite_pred"] for r in rows), "den": len(rows)},
        "nonfinite_metric": {"count": sum(1 for r in val for k in ("P1_shock_p_nrmse", "P2_thickness_abs_log")
                                          if not math.isfinite(float(r.get(k, float("nan"))))), "den": 2 * len(val)},
        "false_surface_shock": {"count": sum(bool(r.get("pred_false_surface_shock")) for r in rows if not r["gt_valid_shock"]),
                                "den": sum(not r["gt_valid_shock"] for r in rows)}}
    for k, v in out["failures"].items():
        v["rate"] = v["count"] / v["den"] if v["den"] else float("nan")
        v["cases"] = []
    out["failures"]["thickness_not_computable"]["cases"] = [r["case"] for r in val if r.get("fail_thickness_not_computable")]
    out["failures"]["pred_shock_not_detected"]["cases"] = [r["case"] for r in val if r.get("fail_pred_shock_not_detected")]
    ms = list(csv.DictReader(open(os.path.join(m["run_dir"], "metrics.csv"))))
    es = [float(r["epoch_sec"]) for r in ms]
    out["cost"] = {"params_trainable_real": m["arm_rec"]["params_trainable_real"], "train_s_per_epoch_median": float(np.median(es)),
                   "train_gpu_h": float(np.sum(es)) / 3600.0, "infer_ms_per_case_median": float(np.median(tms)),
                   "eval_peak_mem_mib": peak, "train_peak_mem_mib": vs.get("peak_alloc_mib_max", vs.get("peak_cuda_mem_mb")),
                   "gpu": meta.get("gpu"), "timing_job": os.environ.get("PBS_JOBID"), "best_epoch": vs.get("best_epoch")}
    return out


def load_data(npz, dev):
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    C = reg["config"]
    (inputs, targets, masks, knn_idx, filenames, n_points, train_idx, val_idx, test_idx, stats, _) = OS.load_allpoints_npz(npz, knn_k=C["knn_k"])
    for k, v in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        assert [int(i) for i in v] == reg["split"][f"{k}_idx"], k
    T.block_test(inputs, targets, masks, test_idx)
    val_ds = OS.AllPointsMaskedDataset(val_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    src = np.load(npz, allow_pickle=True)
    raw = {int(g): np.asarray(src["data"][int(g)], np.float64) for g in val_idx}
    grid_ref = TF.make_internal_vrg(C["grid_h"], C["grid_w"], stats=None, device=dev)
    return {"val_loader": DataLoader(val_ds, batch_size=C["batch"], shuffle=False, drop_last=False), "grid_ref": grid_ref,
            "stats": stats, "val_ids": [int(g) for g in val_idx], "raw": raw, "filenames": {int(g): str(filenames[int(g)]) for g in val_idx}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_root", default=B.OUT_ROOT_DEFAULT)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="comma-separated model ids (QA)")
    a = ap.parse_args()
    out_root = os.path.abspath(a.out_root)
    B.assert_out_root_safe(out_root)
    TN.verify_dependencies()
    if B.sha256_file(a.npz) != B.NACA_NPZ_SHA256:
        sys.exit("NACA npz sha256 differs")
    dev = torch.device(a.device)
    gtc = B.read_json(B.GT_CASES)["NACA"]
    D = load_data(a.npz, dev)
    assert D["val_ids"] == gtc["val_ids"]
    eroot = os.path.join(out_root, "naca", "eval")
    os.makedirs(eroot, exist_ok=True)
    failed, done, todo = [], 0, models(out_root)
    if a.only:
        todo = [m for m in todo if m["model_id"] in a.only.split(",")]
    for m in todo:
        od = os.path.join(eroot, m["model_id"])
        if os.path.isfile(os.path.join(od, "DONE_val")):
            done += 1
            continue
        if a.deadline and time.time() + EST_MIN * 60 > a.deadline:
            print("EVAL budget stop", flush=True)
            sys.exit(3)
        if os.path.exists(od):
            os.replace(od, od + f".incomplete_{int(time.time())}")
        try:
            evaluate(m, D, gtc, dev, od)
            done += 1
            print(f"EVAL=PASS {m['model_id']}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(m["model_id"])
            print(f"EVAL=FAIL {m['model_id']}: {e!r}"[:1500], flush=True)
    print(f"NACA_EVAL={'PASS' if not failed else 'FAIL'} {done}/{len(todo)} failed={failed}", flush=True)
    sys.exit(0 if not failed else 12)


if __name__ == "__main__":
    main()
