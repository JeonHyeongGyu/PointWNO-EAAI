"""2D FNO comparison stage: FNO trials x seeds 0, 1, 2 next to the reused WNO C3 seeds 0, 1, 2, in the format of
collect_seeds.py (validation only).

  python -B code/revision_2026/2d_fno_cmp_r1/collect_fno.py [--wno_root <EAAI root holding the C3 run dirs>] \
      [--fno_root results/fno2d_cmp_r1] [--fno_gpumon_dir results/fno2d_cmp_r1_gpumon] --out <new directory>

Works on the server or on a retrieved analysis copy (checkpoints may be absent).  Nothing is read from the test split.
Checks per run (a run that fails a check is reported, never used):
  both arms: SHA256SUMS_run.txt (present files; only pcwno_best.pt / last_model.pt may be absent), complete.json, 500 epochs, test never
             evaluated, parameters == registry, config.seed == the seed of the directory, 12 validation cases == registry val_idx
  WNO C3:    runner sha256 == pinned train_orig2d.py, exact backward in all 6 layers, registry sha256 == the registered C3 registry of
             that seed, seed 0 == the stage-1 record; the 3-seed means are compared with the values reported by the author
  FNO:       runner sha256 == train_fno2d.py of this package, registry sha256 == fno_registry.json of this package, trial id, FNO
             width / layers / modes, every other config value == the registry config (= C3 config)
Outputs (new directory, nothing overwritten):
  fno_runs.csv          one row per arm x seed (missing / failed runs carry no values)
  fno_summary.csv       per arm: mean, sample std (ddof 1), min, max over the 3 seeds; only when all 3 seeds passed
  fno_val_cases.csv     arm x seed x validation case: NRMSE of pcwno_best.pt (x-mom, y-mom, Cp, mean3)
  fno_case_seedmean.csv arm x validation case: mean over the 3 seeds (complete arms only)
  fno_paired.csv        per validation case (12 pairs), A - B for F1-C3, F2-C3, F2-F1, per target and mean3: basis seed_mean and each
                        seed; mean difference, sd, 95 % t interval, exact two-sided sign-flip permutation p (2^12), case counts
  fno_seed_paired.csv   per seed, A - B of the run-level values (best_val_loss, NRMSE per target, mean3); 3 differences, their mean,
                        and counts; no p-value (n = 3)
  fno_curves.csv        arm x seed x epoch
  collect_checks.json, SHA256SUMS.txt
NRMSE = RMSE in the train-only min-max normalized target space (evaluate_and_save formula of the runners), per validation case.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REGISTRY = os.path.join(HERE, "configs", "fno_registry.json")
RUNNER = os.path.join(HERE, "train_fno2d.py")
WNO_RUNNER = "code/revision_2026/2d_original_r1/train_orig2d.py"
TARGETS = ("x_momentum", "y_momentum", "pressure_coefficient")
METRICS = TARGETS + ("mean3",)
OPTIONAL = {"pcwno_best.pt", "last_model.pt"}
SEEDS = (0, 1, 2)
TOL_LOSS, TOL_NRMSE = 1e-9, 1e-7          # stage-1 record: best val loss to 8 decimals, mean NRMSE to 8 decimals
TOL_AUTHOR = 1e-10                        # author's 3-seed means: 10 decimals
SCALARS = ("best_val_loss",) + tuple(f"val_NRMSE_{k}" for k in METRICS)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_sums(d):
    listed = {}
    for ln in open(os.path.join(d, "SHA256SUMS_run.txt")).read().splitlines():
        if ln.strip():
            h, fn = ln.split(None, 1)
            listed[fn] = h
    missing = sorted(fn for fn in listed if not os.path.isfile(os.path.join(d, fn)))
    bad = sorted(fn for fn in listed if fn not in missing and sha256(os.path.join(d, fn)) != listed[fn])
    return listed, missing, bad


def gpumon_peak(gdir, prefix):
    """Largest memory.used [MiB] over the nvidia-smi sample files starting with prefix (supplementary)."""
    if not gdir or not os.path.isdir(gdir):
        return None, []
    files = sorted(f for f in os.listdir(gdir) if f.startswith(prefix) and f.endswith(".csv"))
    vals = []
    for f in files:
        for ln in open(os.path.join(gdir, f)).read().splitlines():
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 2 and parts[-1].endswith("MiB"):
                try:
                    vals.append(float(parts[-1].split()[0]))
                except ValueError:
                    pass
    return (max(vals) if vals else None), files


def t975(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:                                  # df = 11 (12 validation cases) without scipy
        if df == 11:
            return 2.200985160082949
        raise


def paired(a, b):
    """A - B over the same validation cases (arrays aligned by case)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    md, sd = float(d.mean()), float(d.std(ddof=1))
    half = t975(n - 1) * sd / math.sqrt(n)
    perm = None
    if n <= 16:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        perm = float((np.abs(signs @ d) / n >= abs(md) * (1 - 1e-12)).mean())
    mb = float(np.mean(b))
    return {"n_cases": n, "mean_A": float(np.mean(a)), "mean_B": mb, "mean_diff_A_minus_B": md, "rel_diff_pct_of_B": 100 * md / mb,
            "sd_diff": sd, "ci95_lo": md - half, "ci95_hi": md + half, "ci95_includes_0": bool(md - half <= 0 <= md + half),
            "perm_p_exact_two_sided": perm, "n_A_lower": int((d < 0).sum()), "n_B_lower": int((d > 0).sum()), "n_tie": int((d == 0).sum())}


def read_run(d):
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    vs = json.load(open(os.path.join(d, "val_summary.json")))
    vc = list(csv.DictReader(open(os.path.join(d, "val_case_RMSE.csv"))))
    met = list(csv.DictReader(open(os.path.join(d, "metrics.csv"))))
    return meta, vs, vc, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wno_root", default=ROOT, help="directory holding results/capacity_r1* (the EAAI root, or an analysis copy)")
    ap.add_argument("--fno_root", default=os.path.join(ROOT, "results", "fno2d_cmp_r1"))
    ap.add_argument("--fno_gpumon_dir", default=os.path.join(ROOT, "results", "fno2d_cmp_r1_gpumon"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    reg = json.load(open(REGISTRY))
    reg_sha, runner_sha = sha256(REGISTRY), sha256(RUNNER)
    runner_rel = os.path.relpath(RUNNER, ROOT).replace(os.sep, "/")
    W = reg["wno_reference"]
    val_ids = sorted(reg["split"]["val_idx"])
    arms = [{"arm": "C3", "id": W["id"], "operator": "wno", "params_trainable_real": W["params_trainable_real"],
             "layers": reg["config"]["wno_layers"], "width": reg["config"]["wno_width"], "modes": "", "role": "WNO reference (reused)"}]
    arms += [{"arm": t["arm"], "id": t["id"], "operator": "fno", "params_trainable_real": t["params_trainable_real"],
              "layers": t["fno"]["layers"], "width": t["fno"]["width"], "modes": "x".join(map(str, t["fno"]["modes"])), "role": t["role"],
              "trial": t} for t in reg["trials"]]
    rows, cases, curves, checks, percase = [], [], [], {}, {}
    for A in arms:
        for s in SEEDS:
            key = f"{A['arm']}_seed{s}"
            if A["operator"] == "wno":
                ref = W["runs"][str(s)]
                d = os.path.join(a.wno_root, ref["dir"])
                gdir, gpref = os.path.join(a.wno_root, ref["gpumon_dir"]), ref["gpumon_prefix"]
            else:
                d = os.path.join(a.fno_root, A["id"], f"seed{s}")
                gdir, gpref = a.fno_gpumon_dir, f"train_{A['id']}_seed{s}_"
            base = {k: A[k] for k in ("arm", "id", "operator", "layers", "width", "modes", "params_trainable_real", "role")}
            base.update(seed=s, dir=d)
            if not os.path.isfile(os.path.join(d, "complete.json")):
                rows.append(dict(base, status="missing"))
                checks[key] = {"status": "missing", "dir": d}
                continue
            listed, missing, bad = run_sums(d)
            meta, vs, vc, met = read_run(d)
            nr = {t: vs["val_case_mean_NRMSE"][t] for t in TARGETS}
            ck = {"run_sums_ok": not bad and set(missing) <= OPTIONAL,
                  "epochs_500": vs["epochs_completed"] == 500 and len(met) == 500,
                  "test_not_evaluated": vs["test_evaluated"] is False,
                  "test_blocked_12": meta["test_cases_blocked"] == 12,
                  "params": meta["params_trainable_real"] == A["params_trainable_real"],
                  "config_seed": meta["config"]["seed"] == s,
                  "val_cases_12_registry": sorted(int(r["global_data_id"]) for r in vc) == val_ids and vs["n_val_cases"] == 12}
            if A["operator"] == "wno":
                ck["runner_sha"] = meta["code_sha256"].get(WNO_RUNNER) == W["runner_sha256"]
                ck["exact_backward_6_layers"] = meta["backward"] == W["backward"] and meta["n_waveconv_r1"] == reg["config"]["wno_layers"]
                ck["registry_sha"] = meta["registry_sha256"] == ref["registry_sha256"]
                ck["config_equals_registry"] = {k: v for k, v in meta["config"].items() if k != "seed"} == reg["config"]
                if s == 0:
                    st = W["stage1_seed0"]
                    ck["stage1_record"] = (abs(vs["best_val_loss"] - st["best_val_loss"]) <= TOL_LOSS
                                           and abs(sum(nr.values()) / 3 - st["val_NRMSE_mean3"]) <= TOL_NRMSE)
            else:
                t = A["trial"]
                ck["runner_sha"] = meta["code_sha256"].get(runner_rel) == runner_sha
                ck["registry_sha"] = meta["registry_sha256"] == reg_sha
                ck["trial_id"] = meta["trial"] == A["id"] and meta["operator"] == "fno" and meta["fno"] == t["fno"]
                expect = dict(reg["config"], seed=s, fno_core_init_seed_offset=reg["fno_core_init_seed_offset"], fno_width=t["fno"]["width"],
                              fno_layers=t["fno"]["layers"], fno_modes=list(t["fno"]["modes"]))
                ck["config_equals_registry"] = meta["config"] == expect
            ok = all(ck.values())
            mon, mon_files = gpumon_peak(gdir, gpref)
            checks[key] = {"status": "ok" if ok else "FAILED_CHECKS", "checks": ck, "files_absent": missing, "files_bad": bad, "dir": d,
                           "gpumon_files": mon_files}
            best = os.path.join(d, "pcwno_best.pt")
            es = [float(r["epoch_sec"]) for r in met]
            rows.append(dict(base, status="ok" if ok else "FAILED_CHECKS", best_epoch=vs["best_epoch"], best_val_loss=vs["best_val_loss"],
                             **{f"val_NRMSE_{t}": nr[t] for t in TARGETS}, val_NRMSE_mean3=sum(nr.values()) / 3,
                             final_val_loss=float(met[-1]["val_total"]), final_train_loss=float(met[-1]["train_total"]),
                             train_sec=vs["train_sec"], mean_epoch_sec=sum(es) / len(es), median_epoch_sec=float(np.median(es)),
                             peak_cuda_mem_allocated_mb=vs["peak_cuda_mem_mb"], nvidia_smi_peak_used_mib=mon,
                             ckpt_bytes_actual=os.path.getsize(best) if os.path.isfile(best) else None,
                             gpu=meta["gpu"], torch=meta["torch"], pbs_jobid=meta["pbs_jobid"]))
            if ok:
                percase[(A["arm"], s)] = {}
            for r in vc:
                v = {t: float(r[f"NRMSE_{t}"]) for t in TARGETS}
                v["mean3"] = sum(v.values()) / 3
                cases.append({"arm": A["arm"], "id": A["id"], "seed": s, "status": "ok" if ok else "FAILED_CHECKS",
                              "global_data_id": int(r["global_data_id"]), "filename": r["filename"], **{f"NRMSE_{k}": v[k] for k in METRICS}})
                if ok:
                    percase[(A["arm"], s)][int(r["global_data_id"])] = v
            for r in met:
                curves.append({"arm": A["arm"], "id": A["id"], "seed": s, "epoch": r["epoch"], "train_total": r["train_total"],
                               "val_total": r["val_total"], "lr": r["lr"], "epoch_sec": r["epoch_sec"], "is_best": r["is_best"]})

    # ---- per arm over the 3 seeds (only when every seed passed)
    SUMM = SCALARS + ("best_epoch", "final_val_loss", "train_sec", "mean_epoch_sec", "peak_cuda_mem_allocated_mb", "nvidia_smi_peak_used_mib")
    summary, complete = [], {}
    for A in arms:
        rr = sorted((r for r in rows if r["arm"] == A["arm"] and r["status"] == "ok"), key=lambda r: r["seed"])
        complete[A["arm"]] = [r["seed"] for r in rr] == list(SEEDS)
        row = {k: A[k] for k in ("arm", "id", "operator", "layers", "width", "modes", "params_trainable_real", "role")}
        row.update(params_diff_vs_C3=A["params_trainable_real"] - W["params_trainable_real"], n_seeds_ok=len(rr),
                   seeds_ok=" ".join(str(r["seed"]) for r in rr), status="complete" if complete[A["arm"]] else "incomplete")
        if complete[A["arm"]]:
            for k in SUMM:
                v = [r[k] for r in rr]
                if any(x is None for x in v):
                    row.update({f"{k}_mean": "", f"{k}_std": ""})
                    continue
                v = np.asarray(v, float)
                row.update({f"{k}_mean": float(v.mean()), f"{k}_std": float(v.std(ddof=1)), f"{k}_min": float(v.min()), f"{k}_max": float(v.max()),
                            **{f"{k}_seed{s}": float(x) for s, x in zip(SEEDS, v)}})
        summary.append(row)

    c3row = [r for r in summary if r["arm"] == "C3"][0]
    rep = W["author_reported_3seed_mean"]
    author_check = ({"best_val_loss": abs(c3row["best_val_loss_mean"] - rep["best_val_loss"]) <= TOL_AUTHOR,
                     "val_NRMSE_mean3": abs(c3row["val_NRMSE_mean3_mean"] - rep["val_NRMSE_mean3"]) <= TOL_AUTHOR,
                     "recomputed": {"best_val_loss": c3row["best_val_loss_mean"], "val_NRMSE_mean3": c3row["val_NRMSE_mean3_mean"]},
                     "reported": {"best_val_loss": rep["best_val_loss"], "val_NRMSE_mean3": rep["val_NRMSE_mean3"]}}
                    if complete["C3"] else {"status": "C3 incomplete: not compared"})

    seedmean = []
    for A in arms:
        if not complete[A["arm"]]:
            continue
        for cid in sorted(percase[(A["arm"], 0)]):
            seedmean.append({"arm": A["arm"], "id": A["id"], "global_data_id": cid, "n_seeds": 3,
                             **{f"NRMSE_{k}": float(np.mean([percase[(A["arm"], s)][cid][k] for s in SEEDS])) for k in METRICS}})

    pairs = [(t["arm"], "C3") for t in reg["trials"]] + [(x["arm"], y["arm"]) for i, x in enumerate(reg["trials"]) for y in reg["trials"][:i]]
    paired_rows, seed_rows = [], []
    for A, B in pairs:
        for basis, s in [("seed_mean", None)] + [(f"seed{s}", s) for s in SEEDS]:
            head = {"pair": f"{A}-{B}", "A": A, "B": B, "basis": basis}
            if s is None:
                if not (complete[A] and complete[B]):
                    paired_rows.append(dict(head, metric="", status="not computed (incomplete seeds)"))
                    continue
                va = {r["global_data_id"]: {k: r[f"NRMSE_{k}"] for k in METRICS} for r in seedmean if r["arm"] == A}
                vb = {r["global_data_id"]: {k: r[f"NRMSE_{k}"] for k in METRICS} for r in seedmean if r["arm"] == B}
            else:
                if (A, s) not in percase or (B, s) not in percase:
                    paired_rows.append(dict(head, metric="", status="not computed (run missing / failed)"))
                    continue
                va, vb = percase[(A, s)], percase[(B, s)]
            ids = sorted(va)
            if ids != sorted(vb):
                paired_rows.append(dict(head, metric="", status="case sets differ"))
                continue
            for k in METRICS:
                paired_rows.append(dict(head, metric=f"NRMSE_{k}", status="ok", **paired([va[i][k] for i in ids], [vb[i][k] for i in ids])))
        ra = {r["seed"]: r for r in rows if r["arm"] == A and r["status"] == "ok"}
        rb = {r["seed"]: r for r in rows if r["arm"] == B and r["status"] == "ok"}
        both = [s for s in SEEDS if s in ra and s in rb]
        for k in SCALARS:
            d = {s: ra[s][k] - rb[s][k] for s in both}
            seed_rows.append({"pair": f"{A}-{B}", "A": A, "B": B, "metric": k, "seeds": " ".join(map(str, both)),
                              **{f"diff_seed{s}": d[s] for s in both},
                              "mean_diff": float(np.mean(list(d.values()))) if d else "",
                              "n_A_lower": sum(v < 0 for v in d.values()), "n_B_lower": sum(v > 0 for v in d.values()),
                              "status": "ok" if len(both) == 3 else f"only {len(both)} seed pair(s)"})

    os.makedirs(a.out)

    def write(name, data):
        with open(os.path.join(a.out, name), "x", newline="") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in data for k in r)), restval="")
                w.writeheader()
                w.writerows(data)
    write("fno_runs.csv", rows)
    write("fno_summary.csv", summary)
    write("fno_val_cases.csv", cases)
    write("fno_case_seedmean.csv", seedmean)
    write("fno_paired.csv", paired_rows)
    write("fno_seed_paired.csv", seed_rows)
    write("fno_curves.csv", curves)
    res = {"all_present_ok": all(v["status"] == "ok" for v in checks.values()), "complete_arms": complete, "runs": checks,
           "c3_3seed_vs_author_report": author_check, "test_used": False, "registry_sha256": reg_sha, "runner_sha256": runner_sha,
           "wno_root": os.path.abspath(a.wno_root), "fno_root": os.path.abspath(a.fno_root), "analysis": reg["analysis"]}
    json.dump(res, open(os.path.join(a.out, "collect_checks.json"), "x"), indent=1)
    with open(os.path.join(a.out, "SHA256SUMS.txt"), "x") as f:
        for fn in sorted(os.listdir(a.out)):
            if fn != "SHA256SUMS.txt":
                f.write(f"{sha256(os.path.join(a.out, fn))}  {fn}\n")
    for r in rows:
        print(r["arm"], r["id"], f"seed{r['seed']}", r["status"], r.get("best_val_loss", ""), r.get("val_NRMSE_mean3", ""), flush=True)
    for r in summary:
        print(r["arm"], r["status"], r.get("best_val_loss_mean", ""), r.get("best_val_loss_std", ""), r.get("val_NRMSE_mean3_mean", ""), flush=True)
    print("C3 3-seed means vs author report:", json.dumps(author_check), flush=True)
    sys.exit(0 if all(v["status"] in ("ok", "missing") for v in checks.values()) else 1)


if __name__ == "__main__":
    main()
