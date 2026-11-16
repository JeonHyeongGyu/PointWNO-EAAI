"""Static + synthetic tests of the benchmark package (CPU; no training, no optimizer step; local data read-only).
  python -B tests/test_static_bm.py [--naca_npz P] [--w32_eval DIR]   -> prints T01..T14 PASS/FAIL, exit 0 if all PASS
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for p in (PKG, os.path.join(PKG, "shock"), os.path.join(PKG, "naca"), os.path.join(PKG, "rae"), os.path.join(PKG, "collect"),
          os.path.join(PKG, "nozzle"), os.path.join(PKG, "audit")):
    sys.path.insert(0, p)
import bm_common as B  # noqa: E402

RES = {}


def test(name):
    def deco(f):
        def run(*a):
            try:
                f(*a)
                RES[name] = "PASS"
            except Exception as e:  # noqa: BLE001
                RES[name] = f"FAIL {e!r}"[:400]
                traceback.print_exc()
        run.__name__ = name
        return run
    return deco


@test("T01_registry_regenerates")
def t01():
    r = subprocess.run([sys.executable, "-B", os.path.join(PKG, "make_registry.py"), "--check"], capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert "REGISTRY_CHECK=PASS" in r.stdout, r.stdout + r.stderr


@test("T02_configs_pass_trainer_checks")
def t02():
    import train_naca_bm as TN
    import train_rae_bm as TRB
    n = 0
    for ds, mod in (("naca", TN), ("rae", TRB)):
        for f in sorted(os.listdir(os.path.join(PKG, "configs", ds))):
            cfg = B.read_json(os.path.join(PKG, "configs", ds, f))
            mod.check_config(cfg)
            assert cfg["run_id"] + ".json" == f
            n += 1
            bad = copy.deepcopy(cfg)
            if ds == "naca":
                bad["C"]["lr"] = 2e-3
            else:
                bad["train"]["lr"] = 2e-3
            try:
                mod.check_config(bad)
                raise RuntimeError(f"tampered config accepted: {f}")
            except AssertionError:
                pass
    assert n == 8 + 12, n


@test("T03_parameter_matching")
def t03():
    reg = B.registry()["datasets"]
    p = {(ds, a["arm"]): a["params_trainable_real"] for ds in reg for a in reg[ds]["arms"]}
    assert p[("naca", "V2J2")] == p[("naca", "FNOM")] == 21801731
    assert p[("rae", "V2J2")] == p[("rae", "FNO34")] == p[("rae", "V3")] == 28487140
    assert p[("nozzle", "V2C")] == p[("nozzle", "F00")] == p[("nozzle", "W00")] == 16297350
    assert p[("nozzle", "V2S")] == p[("nozzle", "F10")] == p[("nozzle", "W11")] == 57585030
    for ds in reg:
        for pr in reg[ds]["pairs"]:
            if pr["kind"] == "main":
                assert p[(ds, pr["a"])] == p[(ds, pr["b"])], pr


@test("T04_naca_rng_alignment")
def t04():
    import torch
    import train_naca_bm as TN
    import wno2d_r1 as W
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    for s in (0, 1, 2):
        out = {}
        for arm in ("V2J2", "FNOM"):
            cfg = B.read_json(B.run_config_path("naca", arm, s))
            TN.OS.set_seed(s)
            m = TN.build_model(cfg)
            out[arm] = (B.sha256_bytes(torch.get_rng_state().numpy().tobytes()), m.state_dict())
        C = dict(reg["config"], seed=s)
        TN.OS.set_seed(s)
        c3 = TN.PCWNO(grid_hw=(C["grid_h"], C["grid_w"]), wno_width=32, wno_level=1, wno_layers=6, wno_wavelet="db2",
                      wno_grid_range=tuple(C["grid_range"]))
        W.convert_to_r1(c3)
        rc3 = B.sha256_bytes(torch.get_rng_state().numpy().tobytes())
        assert out["V2J2"][0] == out["FNOM"][0] == rc3, f"global RNG after model differs (seed {s})"
        shared = [k for k in c3.state_dict() if k.startswith(("sa.", "head.", "fp."))]
        assert shared
        for k in shared:
            assert torch.equal(c3.state_dict()[k], out["V2J2"][1][k]) and torch.equal(c3.state_dict()[k], out["FNOM"][1][k]), k
        v2core = [k for k in out["V2J2"][1] if k.startswith("wno.") and ".conv." not in k]
        for k in v2core:
            assert torch.equal(c3.state_dict()[k], out["V2J2"][1][k]), k


@test("T05_rae_init_alignment")
def t05():
    import torch
    import train_rae_bm as TRB
    for s in (0, 1):
        sd = {}
        for arm in ("V2J2", "FNO34", "V3"):
            cfg = B.read_json(B.run_config_path("rae", arm, s))
            torch.manual_seed(s)
            sd[arm] = TRB.build_model(cfg).state_dict()
        for k in sd["V2J2"]:
            if k.startswith(("sa.", "condition_proj.")):
                assert torch.equal(sd["V2J2"][k], sd["FNO34"][k]) and torch.equal(sd["V2J2"][k], sd["V3"][k]), k
            if not k.startswith("wno.conv."):
                assert torch.equal(sd["V2J2"][k], sd["V3"][k]), k


@test("T06_out_root_guard")
def t06():
    for bad in (os.path.join(B.EAAI, "results", "fno2d_cmp_r1", "x"), os.path.join(B.EAAI, "results", "revision_2026", "nozzle_point_hpo_r1"),
                os.path.join(B.EAAI, "results", "revision_2026"), "/tmp/x", os.path.join(B.EAAI, "RAE2822", "results")):
        try:
            B.assert_out_root_safe(bad)
            raise RuntimeError(f"accepted {bad}")
        except SystemExit:
            pass
    B.assert_out_root_safe(B.OUT_ROOT_DEFAULT)
    B.assert_out_root_safe(os.path.join(B.OUT_ROOT_DEFAULT, "local_qa", "x"))


@test("T07_verdict_rule")
def t07():
    assert B.verdict([-1.0, -1.1, -0.9], "A", "B") == "A consistently lower"
    assert B.verdict([1.0, 1.1, 0.9], "A", "B") == "B consistently lower"
    assert B.verdict([-1.0, -0.1, -3.0], "A", "B") == "no detectable difference at n = 3"     # same sign, CI includes 0
    assert B.verdict([-1.0, 0.5, -1.0], "A", "B") == "no detectable difference at n = 3"
    assert B.verdict([-1.0, -1.0], "A", "B").startswith("incomplete")
    lo, hi = B.t_ci95([1.0, 2.0, 3.0])
    assert abs(lo - (2 - 4.303 / math.sqrt(3))) < 1e-9 and abs(hi - (2 + 4.303 / math.sqrt(3))) < 1e-9


@test("T08_profile_metrics_synthetic")
def t08():
    import numpy as np
    import profile_metrics as PM
    s = np.linspace(0, 1, 2001)
    d = 0.004
    g = np.tanh((s - 0.4) / d)
    r = PM.rise(s, g, 0.4)
    want = 2 * d * math.atanh(0.8)
    assert r["fail"] is None and abs(r["width"] - want) / want < 0.02, (r["width"], want)
    p = np.tanh((s - 0.4) / (2 * d))
    m, _, _ = PM.compare(s, p, g, 0.4)
    assert abs(m["thickness_ratio"] - 2.0) < 0.05 and abs(m["thickness_abs_log"] - math.log(2)) < 0.03
    assert m["location_abs_err"] < 1e-3 and m["jump_abs_rel_err"] < 1e-3 and m["spurious_osc"] == 0.0
    osc = g + 0.2 * np.sin(2 * np.pi * s / 0.01) * (np.abs(s - 0.4) < 0.05)
    m2, _, _ = PM.compare(s, osc, g, 0.4)
    assert m2["spurious_osc"] > 0.1 and m2["tv_abs_rel_err"] > 0.1
    m3, _, rp = PM.compare(s, np.zeros_like(s), g, 0.4)
    assert rp["fail"] == "no_rise" and math.isnan(m3["thickness_abs_log"]) and not m3["gt_profile_failed"]
    _, rg, _ = PM.compare(s, g, np.zeros_like(s), 0.4)
    assert rg["fail"] == "no_rise"


@test("T09_shell_syntax_and_dry_launcher")
def t09():
    for f in ("pbs/bm_KITSM02.pbs", "launcher/submit_bm.sh", "server/pack_bm_evidence_readonly.sh"):
        r = subprocess.run(["bash", "-n", os.path.join(PKG, f)], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)
    if os.path.isfile(os.path.join(PKG, "MANIFEST_SHA256.txt")):
        env = dict(os.environ, EAAI_ROOT=B.EAAI, PY=sys.executable, NACA_NPZ="/nonexistent.npz", BM_SUBMIT="0")
        r = subprocess.run(["bash", os.path.join(PKG, "launcher", "submit_bm.sh"), "smoke", "all"], capture_output=True, text=True, env=env)
        assert r.returncode == 0 and r.stdout.count("[dry] qsub") == 3 and "submitted" not in r.stdout, r.stdout + r.stderr
        r = subprocess.run(["bash", os.path.join(PKG, "launcher", "submit_bm.sh"), "full", "all"], capture_output=True, text=True, env=env)
        assert r.returncode == 2 and "BM_APPROVED_FULL" in r.stdout, r.stdout


@test("T10_gt_cases_frozen")
def t10():
    g = B.read_json(B.GT_CASES)
    assert g["NACA"]["n_valid_shock"] == 7 and g["NACA"]["n_gt_profile_failed"] == 0
    assert g["RAE2822"]["n_valid_shock_cases"] == 102 and g["RAE2822"]["n_valid_shock_sides"] == 138
    assert g["NOZZLE"]["n_valid_shock"] == 6
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    assert g["NACA"]["val_ids"] == reg["split"]["val_idx"] and not set(g["NACA"]["val_ids"]) & set(reg["split"]["test_idx"])


@test("T11_naca_region_same_definition")
def t11(npz):
    import numpy as np
    import eval_naca_bm as EN
    import gt_shock_cases as GS
    if not npz:
        raise RuntimeError("no --naca_npz")
    g = B.read_json(B.GT_CASES)["NACA"]
    src = np.load(npz, allow_pickle=True)
    for gid, c in g["cases"].items():
        if not c["valid_shock"]:
            continue
        a = np.asarray(src["data"][int(gid)], np.float64)
        w1, u1 = EN.upper_rows(a)
        w2, u2 = GS.naca_upper_surface(a)
        assert np.array_equal(w1, w2) and np.array_equal(u1, u2)
        m1 = EN.region_mask(a, w1, u1, *c["candidate_xy"], g["region"])
        m2 = GS.naca_region(a, w2, u2, *c["candidate_xy"])
        assert np.array_equal(m1, m2) and int(m1.sum()) == c["region_n_points"]


@test("T12_nozzle_rows_equal_f10m")
def t12(w32):
    import numpy as np
    sys.path.insert(0, os.path.join(B.REV, "nozzle_f10_matched_completion_r1"))
    import physics_eval_f10m as PF
    import nozzle_shock_bm as NS
    if not w32:
        raise RuntimeError("no --w32_eval")
    gt = NS.AC.ValGT(None)
    for op in ("WNO", "FNO"):
        for s in (0, 2):
            ed = os.path.join(w32, f"HPO_BASE_{op}_s{s}")
            m = {"model_id": f"X_{op}_{s}", "arm": op, "seed": s, "eval_dir": ed, "cand_id": op, "op": op}
            g1, r1 = NS.load_model(m, gt)
            g2, r2 = PF.load_model(m, gt)
            a = NS.case_rows(m, *g1, gt, r1)
            b = PF.case_rows(m, *g2, gt, r2)
            for x, y in zip(a, b):
                for k, v in y.items():
                    if k in x:
                        assert (x[k] == v) or (isinstance(v, float) and math.isnan(v) and math.isnan(x[k])), (k, x[k], v)


@test("T13_collector_synthetic")
def t13(w32_root):
    import collect_bm as CB
    tmp = tempfile.mkdtemp(dir=os.path.join(B.OUT_ROOT_DEFAULT, "local_qa"))
    try:
        reg = B.registry()["datasets"]
        found = {}
        for ds in reg:
            for a in reg[ds]["arms"]:
                for s in B.SEEDS:
                    base = {"model_id": f"{ds}_{a['arm']}_{s}", "metrics": {}, "failures": {}, "cost": {}}
                    for k in CB.PRIMARY + CB.GUARD + CB.SECONDARY:
                        v = 1.0 + 0.01 * s
                        if a["arm"].startswith("V2") and ds == "naca":
                            v -= 0.5                               # V2 consistently lower everywhere on NACA
                        if ds == "rae" and a["arm"] == "FNO34" and k in CB.PRIMARY:
                            v -= 0.5                               # FNO consistently lower on RAE primaries
                        base["metrics"][k] = {"value": v, "n_finite": 7, "n_used": 7}
                    found[(ds, a["arm"], s)] = base
        found[("nozzle", "V2C", 1)]["metrics"]["P1_shock_p_nrmse"]["n_finite"] = 5
        rows = {ds: [CB.paired(found, ds, p, m) for p in reg[ds]["pairs"] for m in CB.PRIMARY + CB.GUARD] for ds in reg}
        c = {ds: CB.claim(ds, rows[ds], reg[ds]["pairs"]) for ds in reg}
        assert c["naca"]["claim"].startswith("V2 shock-capturing advantage"), c["naca"]
        assert c["rae"]["claim"].startswith("FNO shock-capturing advantage"), c["rae"]
        assert c["nozzle"]["claim"].startswith("no shock-capturing difference"), c["nozzle"]
        assert any(r["failure_confounded"] for r in rows["nozzle"] if r["pair"] == "NOZ-MAIN-1")
        # decision-context block: NACA V2 better than FNOM but F1 consistently better on a primary -> no V2 claim
        for s in B.SEEDS:
            found[("naca", "F1", s)]["metrics"]["P1_shock_p_nrmse"]["value"] = 0.1 + 0.01 * s
        rows2 = [CB.paired(found, "naca", p, m) for p in reg["naca"]["pairs"] for m in CB.PRIMARY + CB.GUARD]
        assert not CB.claim("naca", rows2, reg["naca"]["pairs"])["claim"].startswith("V2"), "context block ignored"
        # incomplete main pair
        found[("rae", "V2J2", 2)] = None
        rows3 = [CB.paired(found, "rae", p, m) for p in reg["rae"]["pairs"] for m in CB.PRIMARY + CB.GUARD]
        assert CB.claim("rae", rows3, reg["rae"]["pairs"])["claim"].startswith("INCOMPLETE")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test("T14_v2_source_audit")
def t14():
    out = tempfile.mkdtemp(dir=os.path.join(B.OUT_ROOT_DEFAULT, "local_qa"))
    try:
        r = subprocess.run([sys.executable, "-B", os.path.join(PKG, "audit", "audit_v2_source.py"), "--out", out], capture_output=True,
                           text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        assert "V2_SOURCE_AUDIT=PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naca_npz", default=None)
    ap.add_argument("--w32_eval", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(B.OUT_ROOT_DEFAULT, "local_qa"), exist_ok=True)
    t01(); t02(); t03(); t04(); t05(); t06(); t07(); t08(); t09(); t10(); t11(a.naca_npz); t12(a.w32_eval); t13(a.w32_eval); t14()
    for k, v in RES.items():
        print(f"{k}: {v}")
    n_ok = sum(v == "PASS" for v in RES.values())
    print(f"STATIC_TESTS={'PASS' if n_ok == len(RES) else 'FAIL'} {n_ok}/{len(RES)}")
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == "__main__":
    main()
