"""3-seed comparison collector (CPU, read-only on evaluation outputs). VALIDATION ONLY.

Reads the standard model_metrics.json of every model:
  naca   <out_root>/naca/eval/<model_id>/model_metrics.json
  rae    <out_root>/rae/eval/<model_id>/model_metrics.json
  nozzle <out_root>/nozzle/shock/<LATEST or --nozzle_tag>/<model_id>/model_metrics.json
Writes <out_root>/collect/<tag>/:
  model_metrics_long.csv  every model x metric (value, n_finite, n_used); missing models listed, never imputed
  failures.csv            every model x failure code (count / denominator / rate / cases)
  cost.csv                parameters, train s/epoch, GPU-h, peak memory, inference ms / case (+ timing source)
  paired.csv              every registered pair x metric: same-seed differences a - b, mean, SD, 95 % t interval, sign
                          count, verdict (frozen rule), failure-confounded flag
  claims.json             per dataset: the frozen claim rule applied to the MAIN and decision-context pairs
  BENCHMARK_REPORT.md     tables of the above
Datasets are never pooled. Metric definitions differ by dataset (eval_protocol.json); identical keys are NOT comparable
across datasets.
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
sys.path.insert(0, PKG)
import bm_common as B  # noqa: E402

PRIMARY = ("P1_shock_p_nrmse", "P2_thickness_abs_log")
GUARD = ("val_loss", "global_nrmse_mean")
SECONDARY = ("location_abs_err", "jump_abs_rel_err", "grad_err", "post_overshoot", "pre_undershoot", "tv_abs_rel_err",
             "spurious_osc", "shock_mach_nrmse")
DESCRIPTIVE_ONLY = ("thickness_ratio",)          # target 1; its distance is P2
SHOCK_METRICS = set(PRIMARY) | set(SECONDARY)


def load_models(out_root, nozzle_tag):
    found = {}
    reg = B.registry()["datasets"]
    for ds in ("naca", "rae", "nozzle"):
        if ds == "nozzle":
            sroot = os.path.join(out_root, "nozzle", "shock")
            tag = nozzle_tag or (open(os.path.join(sroot, "LATEST")).read().strip() if os.path.isfile(os.path.join(sroot, "LATEST")) else None)
            base = os.path.join(sroot, tag) if tag else None
        else:
            base = os.path.join(out_root, ds, "eval")
        for a in reg[ds]["arms"]:
            for s in a["seeds"]:
                if ds == "nozzle":
                    mid = a["model_id"].format(seed=s)
                elif a["status"] in ("new", "optional"):
                    mid = B.run_id(ds, a["arm"], s)
                else:
                    mid = f"BM_{ds.upper()}_REUSE_{a['arm']}_s{s}"
                p = os.path.join(base, mid, "model_metrics.json") if base else None
                found[(ds, a["arm"], s)] = B.read_json(p) if p and os.path.isfile(p) else None
    return found


def paired(found, ds, pair, metric):
    d, per, conf = [], [], False
    for s in B.SEEDS:
        ma, mb = found.get((ds, pair["a"], s)), found.get((ds, pair["b"], s))
        if not ma or not mb or metric not in ma["metrics"] or metric not in mb["metrics"]:
            continue
        va, vb = ma["metrics"][metric]["value"], mb["metrics"][metric]["value"]
        if va is None or vb is None or not (math.isfinite(va) and math.isfinite(vb)):
            continue
        d.append(va - vb)
        per.append(f"s{s}:{va - vb:.6g}")
        if metric in SHOCK_METRICS and ma["metrics"][metric]["n_finite"] != mb["metrics"][metric]["n_finite"]:
            conf = True
    lo, hi = B.t_ci95(d)
    mu = sum(d) / len(d) if d else float("nan")
    sd = (sum((x - mu) ** 2 for x in d) / (len(d) - 1)) ** 0.5 if len(d) > 1 else float("nan")
    v = B.verdict(d, pair["a"], pair["b"])
    return {"dataset": ds, "pair": pair["id"], "kind": pair["kind"], "a": pair["a"], "b": pair["b"], "metric": metric,
            "group": "primary" if metric in PRIMARY else "guard" if metric in GUARD else "secondary",
            "n_seeds": len(d), "mean_diff_a_minus_b": mu, "sd_diff": sd, "ci95_lo": lo, "ci95_hi": hi,
            "a_lower_seeds": sum(x < 0 for x in d), "per_seed_diff": ";".join(per), "verdict": v,
            "failure_confounded": conf, "note": pair.get("note", "")}


def claim(ds, rows, pairs):
    """Frozen claim rule (eval_protocol.json claim_rule)."""
    get = {(r["pair"], r["metric"]): r for r in rows}
    mains = [p for p in pairs if p["kind"] == "main"]
    ctx = [p for p in pairs if p["kind"] == "decision_context"]
    res = {"main_pairs": {}, "decision_context": {}}
    v2_ok, fno_ok, complete = [], [], True
    for p in mains:
        pv = {m: get[(p["id"], m)]["verdict"] for m in PRIMARY + GUARD if (p["id"], m) in get}
        complete &= all(not pv.get(m, "incomplete").startswith("incomplete") for m in PRIMARY)
        a_low = f"{p['a']} consistently lower"
        b_low = f"{p['b']} consistently lower"
        v2 = all(pv.get(m) == a_low for m in PRIMARY) and all(pv.get(m) != b_low for m in GUARD)
        fn = all(pv.get(m) == b_low for m in PRIMARY) and all(pv.get(m) != a_low for m in GUARD)
        res["main_pairs"][p["id"]] = {"verdicts": pv, "v2_condition": v2, "fno_condition": fn,
                                      "failure_confounded": any(get[(p["id"], m)]["failure_confounded"] for m in PRIMARY if (p["id"], m) in get)}
        v2_ok.append(v2)
        fno_ok.append(fn)
    ctx_block_v2, ctx_block_fno = False, False
    for p in ctx:
        pv = {m: get[(p["id"], m)]["verdict"] for m in PRIMARY if (p["id"], m) in get}
        res["decision_context"][p["id"]] = pv
        if any(pv.get(m) == f"{p['b']} consistently lower" for m in PRIMARY):
            ctx_block_v2 = True
        if any(pv.get(m) == f"{p['a']} consistently lower" for m in PRIMARY):
            ctx_block_fno = True
    if not complete:
        res["claim"] = "INCOMPLETE: a main-pair primary metric has fewer than 3 paired seeds; no claim"
    elif any(v2_ok) and not ctx_block_v2:
        res["claim"] = "V2 shock-capturing advantage established on this dataset (frozen rule; n = 3, descriptive)"
    elif any(fno_ok) and not ctx_block_fno:
        res["claim"] = "FNO shock-capturing advantage established on this dataset (frozen rule; n = 3, descriptive)"
    else:
        res["claim"] = "no shock-capturing difference established on this dataset"
    res["context_blocked_v2"], res["context_blocked_fno"] = ctx_block_v2, ctx_block_fno
    return res


def write_csv(path, rows):
    if not rows:
        open(path, "w").close()
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else (f"{x:.4g}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=B.OUT_ROOT_DEFAULT)
    ap.add_argument("--nozzle_tag", default=None)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    out_root = os.path.abspath(a.out_root)
    B.assert_out_root_safe(out_root)
    od = os.path.join(out_root, "collect", a.tag or time.strftime("%Y%m%d_%H%M%S"))
    if os.path.exists(od):
        raise SystemExit(f"refusing to overwrite {od}")
    found = load_models(out_root, a.nozzle_tag)
    reg = B.registry()["datasets"]
    long, fails, cost, missing = [], [], [], []
    for (ds, arm, s), mm in sorted(found.items()):
        if mm is None:
            missing.append(f"{ds}/{arm}/s{s}")
            continue
        for k, v in mm["metrics"].items():
            long.append({"dataset": ds, "arm": arm, "seed": s, "model_id": mm["model_id"], "metric": k, **v})
        for k, v in mm["failures"].items():
            fails.append({"dataset": ds, "arm": arm, "seed": s, "model_id": mm["model_id"], "failure": k, "count": v["count"],
                          "den": v["den"], "rate": v.get("rate"), "cases": ";".join(map(str, v.get("cases", [])))})
        cost.append({"dataset": ds, "arm": arm, "seed": s, "model_id": mm["model_id"],
                     **{k: v for k, v in mm["cost"].items() if not isinstance(v, dict)}})
    prows, claims = [], {}
    for ds in ("naca", "rae", "nozzle"):
        rows = []
        for p in reg[ds]["pairs"]:
            for m in PRIMARY + GUARD + SECONDARY:
                rows.append(paired(found, ds, p, m))
        prows += rows
        claims[ds] = claim(ds, rows, reg[ds]["pairs"])
    os.makedirs(od)
    write_csv(os.path.join(od, "model_metrics_long.csv"), long)
    write_csv(os.path.join(od, "failures.csv"), fails)
    write_csv(os.path.join(od, "cost.csv"), cost)
    write_csv(os.path.join(od, "paired.csv"), prows)
    summ = {"version": "bm_collect_v1", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "registry_sha256": B.registry_sha(),
            "protocol_sha256": B.protocol_sha(), "missing_models": missing, "claims": claims, "validation_only": True,
            "n_models_found": sum(v is not None for v in found.values()), "n_models_registered": len(found)}
    B.atomic_json(summ, os.path.join(od, "claims.json"))
    L = [f"# Canonical PointWNO V2 vs PointFNO — validation benchmark report ({summ['time']})", "",
         "Validation only. n = 3 seeds per arm; the verdict rule is descriptive (no p-values). Datasets are never pooled.", "",
         f"Missing models ({len(missing)}): {', '.join(missing) if missing else 'none'}", ""]
    for ds in ("naca", "rae", "nozzle"):
        L += [f"## {ds}: {reg[ds]['label']}", "", f"**Claim (frozen rule):** {claims[ds]['claim']}", "",
              "| pair | metric | n | mean a-b | 95 % CI | a lower (seeds) | verdict | failure-confounded |", "|---|---|---|---|---|---|---|---|"]
        for r in prows:
            if r["dataset"] == ds and r["group"] in ("primary", "guard"):
                L.append(f"| {r['pair']} ({r['a']}-{r['b']}) | {r['metric']} | {r['n_seeds']} | {fmt(r['mean_diff_a_minus_b'])} | "
                         f"[{fmt(r['ci95_lo'])}, {fmt(r['ci95_hi'])}] | {r['a_lower_seeds']} | {r['verdict']} | {r['failure_confounded']} |")
        L += ["", "Failure counts (count/denominator per seed model):", ""]
        for f in fails:
            if f["dataset"] == ds and f["count"]:
                L.append(f"- {f['model_id']} {f['failure']}: {f['count']}/{f['den']} ({f['cases']})")
        L.append("")
    with open(os.path.join(od, "BENCHMARK_REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(od, "SHA256SUMS.txt"), "w") as f:
        for n in sorted(os.listdir(od)):
            if n != "SHA256SUMS.txt":
                f.write(f"{B.sha256_file(os.path.join(od, n))}  {n}\n")
    print(json.dumps({ds: c["claim"] for ds, c in claims.items()}, indent=1))
    print(f"COLLECT=DONE found {summ['n_models_found']}/{summ['n_models_registered']} -> {od}")


if __name__ == "__main__":
    main()
