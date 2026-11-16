"""Static and non-training checks of the 2D FNO comparison package (local CPU; no optimizer step, no training, no GPU).

  python -B code/revision_2026/2d_fno_cmp_r1/tests/check_fno_static.py --npz <Data_allpoints_xymomcp_from_dat.npz> \
      --scratch <empty scratch dir> --out <result json (new file)>

S1 registry: pins, config == WNO C3 config (minus seed), C3 seed registries, split == splits/2d_orig_rng0_forced60_118.json, FNO core file
   == the file that trained B2, parameter counts / differences from built models
S2 engine equivalence: main() of train_fno2d.py vs main() of train_orig2d.py (the C3 runner), statement by statement (AST); the
   statements that differ are exactly the registered ones; the epoch loop and everything after it are identical
S3 RNG alignment with WNO C3 (seeds 0, 1, 2; F1, F2): CPU / numpy / python RNG state after model construction == after the exact C3
   construction (+ r1 conversion); LPE / FP / head initial values == C3; shuffle order of 2 epochs identical; per-forward RNG use identical
   (train 1 draw, eval 0); FNO core deterministic per seed and different between seeds
S4 one forward + one backward per FNO trial on CPU (no optimizer step): shapes, finite values, every trainable tensor gets a gradient
S5 runner command line: refusals (trial, seed, smoke settings, npz sha) and a dry run (test blocked, 96 / 12, parameters, no files)
S6 collector on synthetic run directories: statistics vs an independent computation, C3 author-report check, incomplete arm, tampered run
S7 PBS: syntax, registered runs, manifest / gate / overwrite checks, no delete commands, smoke gate logic on synthetic smoke directories
S8 test-split isolation and package manifest
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(PKG, "..", "..", ".."))
sys.path.insert(0, PKG)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_fno2d as F  # noqa: E402
T = F.T
REG_PATH = os.path.join(PKG, "configs", "fno_registry.json")
REG = json.load(open(REG_PATH))
RESULTS = []
B2_MODELS_SHA = "4fa4947ab4d568553583241a92b01cdff50508ce2222bcbeeaf3cc93992fe81b"
B2_SPLIT_SHA = "aa76db5210af7c1a46de0497705d747dfcdd1f974d446986e1c8d66f5253fee5"


def check(group, name, ok, detail=""):
    RESULTS.append({"group": group, "name": name, "pass": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {group} {name} {detail}", flush=True)


def sha(p):
    return F.sha256(p)


def cfg(trial, seed):
    return dict(REG["config"], seed=seed, fno_core_init_seed_offset=REG["fno_core_init_seed_offset"], fno_width=trial["fno"]["width"],
                fno_layers=trial["fno"]["layers"], fno_modes=list(trial["fno"]["modes"]))


def rng_all():
    return (hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(), hashlib.sha256(repr(random.getstate()).encode()).hexdigest())


def build_c3(seed):
    """The C3 sequence of train_orig2d.main between set_seed and the trainer (data loading draws no torch / numpy / python numbers)."""
    C = dict(REG["config"], seed=seed)
    T.OS.set_seed(seed)
    F.make_internal_vrg(C["grid_h"], C["grid_w"], stats=None, device=torch.device("cpu"))
    m = F.PCWNO(grid_hw=(C["grid_h"], C["grid_w"]), wno_width=C["wno_width"], wno_level=C["wno_level"], wno_layers=C["wno_layers"],
                wno_wavelet=C["wavelet"], wno_grid_range=tuple(C["grid_range"]))
    n = T.W.convert_to_r1(m)
    return m, n


def build_fno(trial, seed):
    C = cfg(trial, seed)
    T.OS.set_seed(seed)
    F.make_internal_vrg(C["grid_h"], C["grid_w"], stats=None, device=torch.device("cpu"))
    return F.build_model(C)


def shared_state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items() if not k.startswith("wno.")}


def shuffle_order(epochs=2):
    from torch.utils.data import DataLoader
    dl = DataLoader(list(range(96)), batch_size=4, shuffle=True, drop_last=False)
    out = []
    for _ in range(epochs):
        for b in dl:
            out.append(b.tolist())
            torch.rand(1)                                         # the LPE's per-step draw in training mode
    return out


def pts(n=256):
    g = torch.Generator().manual_seed(123)
    p = torch.rand(1, n, 3, generator=g)
    p[..., 2] = 0
    return p


# ------------------------------------------------------------------ S1
def s1():
    bad = [p for p, s in REG["pinned_sha256"].items() if sha(os.path.join(ROOT, p)) != s]
    check("S1", "registry_pins", not bad, str(bad))
    c3 = json.load(open(os.path.join(ROOT, REG["config_source"]["registry"])))
    check("S1", "config_equals_C3_config_minus_seed", REG["config"] == {k: v for k, v in c3["config"].items() if k != "seed"}
          and sha(os.path.join(ROOT, REG["config_source"]["registry"])) == REG["config_source"]["sha256"])
    same = all(REG[k] == c3[k] for k in ("npz_sha256", "split", "checkpoint_policy", "test_policy", "precision_policy"))
    check("S1", "split_npz_policies_equal_C3", same)
    for s in ("1", "2"):
        r = json.load(open(os.path.join(ROOT, REG["wno_reference"]["runs"][s]["registry"])))
        check("S1", f"C3_seed{s}_registry_equals_C3_except_seed",
              {k: v for k, v in r["config"].items() if k != "seed"} == REG["config"] and r["config"]["seed"] == int(s)
              and r["split"] == REG["split"] and r["params_trainable_real"] == REG["params_trainable_real_C3"])
    sp_path = os.path.join(ROOT, "splits", "2d_orig_rng0_forced60_118.json")
    sp = json.load(open(sp_path))
    check("S1", "split_equals_split_file_of_B2", sha(sp_path) == B2_SPLIT_SHA and all(
        [int(i) for i in sp[k]] == REG["split"][f"{k}_idx"] for k in ("train", "val", "test")))
    check("S1", "fno_core_file_is_B2_file", sha(os.path.join(ROOT, "code/pwno2d/models.py")) == B2_MODELS_SHA == REG["fno_core"]["sha256"])
    check("S1", "fno_core_class_is_pwno2d", F.FNO2dGridCore.__module__ == "pwno2d.models")
    torch.manual_seed(0)
    m3, _ = build_c3(0)
    p3 = sum(p.numel() * (2 if p.is_complex() else 1) for p in m3.parameters() if p.requires_grad)
    check("S1", "C3_params_28486403", p3 == REG["params_trainable_real_C3"] == 28486403, str(p3))
    for t in REG["trials"]:
        m = build_fno(t, 0)
        g = F.count_parameters(m)
        check("S1", f"{t['id']}_params", g["total"] == t["params_trainable_real"] and g == t["param_groups"]
              and t["params_diff_vs_C3"] == g["total"] - p3, f"{g['total']} diff {g['total'] - p3}")
        check("S1", f"{t['id']}_modes_fit_grid", 2 * t["fno"]["modes"][0] <= REG["config"]["grid_h"]
              and t["fno"]["modes"][1] <= REG["config"]["grid_w"] // 2 + 1)
        check("S1", f"{t['id']}_seeds_0_1_2", t["seeds"] == [0, 1, 2])
    f1 = [t for t in REG["trials"] if t["arm"] == "F1"][0]
    check("S1", "F1_depth_width_equal_C3_modes_equal_B2", (f1["fno"]["layers"], f1["fno"]["width"]) == (REG["config"]["wno_layers"], REG["config"]["wno_width"])
          and f1["fno"]["modes"] == [34, 34])


# ------------------------------------------------------------------ S2
def main_statements(path):
    tree = ast.parse(open(path).read())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    src = open(path).read().splitlines()
    return [(ast.dump(s), src[s.lineno - 1].strip()) for s in fn.body]


EXPECTED_ORIG_ONLY = [
    'ap.add_argument("--out_root", default=os.path.join(ROOT, "results", "orig2d_r1"))',
    "reg, trial = check_registry(reg_path, a.trial, a.npz)",
    'C = reg["config"]',
    "rel = os.path.relpath(os.path.abspath(a.out_root), ROOT).replace(os.sep, \"/\")",
    'if any(rel == f or rel.startswith(f + "/") for f in FORBIDDEN_ROOTS):',
    "out = os.path.join(a.out_root, trial[\"id\"], f\"seed{C['seed']}\")",
    "if a.smoke_tag:",
    "if not a.dry_run:",
    "model = PCWNO(grid_hw=(C[\"grid_h\"], C[\"grid_w\"]), wno_width=C[\"wno_width\"], wno_level=C[\"wno_level\"],",
    "n_r1 = W.convert_to_r1(model) if trial[\"backward\"] == W.R1_BACKWARD_VERSION else 0",
    "if trial[\"backward\"] not in (W.R1_BACKWARD_VERSION, \"library_pytorch_wavelets_1.3.0\"):",
    "if n_r1 != (C[\"wno_layers\"] if trial[\"backward\"] == W.R1_BACKWARD_VERSION else 0):",
    "if n_params_real != reg[\"params_trainable_real\"]:",
    "info = {\"trial\": trial[\"id\"], \"backward\": trial[\"backward\"], \"n_waveconv_r1\": n_r1, \"params_trainable_real\": n_params_real,",
    "code_files = [os.path.abspath(__file__), OS.__file__, sys.modules[\"pcwno_vrg_lpe_torch_airfoil_momentum_fno_same\"].__file__,",
]


def s2():
    orig = main_statements(os.path.join(ROOT, "code/revision_2026/2d_original_r1/train_orig2d.py"))
    new = main_statements(os.path.join(PKG, "train_fno2d.py"))
    j, orig_only, matched = 0, [], []
    for d, line in orig:
        k = next((i for i in range(j, len(new)) if new[i][0] == d), None)
        if k is None:
            orig_only.append(line)
        else:
            matched.append(line)
            j = k + 1
    check("S2", "differing_statements_exactly_registered", orig_only == EXPECTED_ORIG_ONLY,
          f"{len(orig_only)} differ / {len(matched)} identical: " + json.dumps([x for x in orig_only if x not in EXPECTED_ORIG_ONLY]))
    i_loop = [i for i, (_, l) in enumerate(orig) if l.startswith("for ep in range(1, epochs + 1):")][0]
    tail_o = [d for d, _ in orig[i_loop - 4:]]                       # metrics header, best_val, t_run, ... loop, and everything after it
    tail_n = [d for d, _ in new[len(new) - len(tail_o):]]
    check("S2", "epoch_loop_and_all_later_statements_identical", tail_o == tail_n, f"{len(tail_o)} statements")
    for key in ("OS.set_seed(C[\"seed\"])", "train_loader = DataLoader(", "trainer = OS.MaskedGradTrainer(", "scheduler = torch.optim.lr_scheduler.StepLR("):
        check("S2", f"identical: {key[:40]}", any(l.startswith(key) for l in matched))
    for name in ("block_test", "val_case_rmse", "numerics", "sha256", "resolve"):
        check("S2", f"helper_is_C3_runner_function: {name}", getattr(F, name) is getattr(T, name))
    check("S2", "build_model_draws_core_in_fork_rng", "fork_rng(devices=[])" in open(os.path.join(PKG, "train_fno2d.py")).read())


# ------------------------------------------------------------------ S3
def s3():
    for s in (0, 1, 2):
        m3, n = build_c3(s)
        st3, sh3 = rng_all(), shared_state(m3)
        o3 = shuffle_order()
        check("S3", f"C3_seed{s}_r1_layers", n == 6)
        for t in REG["trials"]:
            mf = build_fno(t, s)
            stf, shf = rng_all(), shared_state(mf)
            of = shuffle_order()
            check("S3", f"{t['arm']}_seed{s}_rng_state_after_model_equals_C3", stf == st3)
            same = sorted(sh3) == sorted(shf) and all(torch.equal(sh3[k], shf[k]) for k in sh3)
            check("S3", f"{t['arm']}_seed{s}_LPE_FP_head_initial_values_equal_C3", same, f"{len(sh3)} tensors")
            check("S3", f"{t['arm']}_seed{s}_shuffle_order_2_epochs_equals_C3", of == o3)
    # per-forward RNG use (train: the LPE's 1 draw; eval: none), C3 vs F1 / F2
    x, g = pts(), F.make_internal_vrg(128, 96, stats=None, device=torch.device("cpu"))
    m3, _ = build_c3(0)
    models = {"C3": m3, **{t["arm"]: build_fno(t, 0) for t in REG["trials"]}}
    use = {}
    for name, m in models.items():
        for mode in ("train", "eval"):
            m.train(mode == "train")
            torch.manual_seed(7)
            with torch.no_grad():
                m(x, g)
            use[(name, mode)] = rng_all()[0]
    torch.manual_seed(7); torch.rand(1); one = rng_all()[0]
    torch.manual_seed(7); zero = rng_all()[0]
    for name in models:
        check("S3", f"{name}_forward_rng_use_train1_eval0", use[(name, "train")] == one and use[(name, "eval")] == zero)
    for t in REG["trials"]:
        a, b, c = build_fno(t, 1), build_fno(t, 1), build_fno(t, 2)
        ka = {k: v for k, v in a.state_dict().items() if k.startswith("wno.")}
        kb = {k: v for k, v in b.state_dict().items() if k.startswith("wno.")}
        kc = {k: v for k, v in c.state_dict().items() if k.startswith("wno.")}
        check("S3", f"{t['arm']}_core_deterministic_per_seed", all(torch.equal(ka[k], kb[k]) for k in ka))
        check("S3", f"{t['arm']}_core_differs_between_seeds", all(not torch.equal(ka[k], kc[k]) for k in ka if ka[k].numel() > 1 and "bias" not in k))


# ------------------------------------------------------------------ S4
def s4():
    """One fresh process per trial (local RAM limit under ulimit -v)."""
    for t in REG["trials"]:
        r = subprocess.run([sys.executable, "-B", os.path.abspath(__file__), "--s4_trial", t["id"]], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True)
        got = [json.loads(ln[5:]) for ln in r.stdout.splitlines() if ln.startswith("S4JS ")]
        for g in got:
            check("S4", g["name"], g["pass"], g["detail"])
        if r.returncode or not got:
            check("S4", f"{t['arm']}_subprocess", False, r.stdout.strip()[-300:])


def s4_one(trial_id):
    """LPE parameters are frozen here (local RAM limit: the LPE graph over 12,288 VRG nodes does not fit under ulimit -v); the LPE is
    the same code object as in WNO C3. Core (FNO), lift / projection, FP and head get one backward pass (no optimizer step)."""
    x, g = pts(), F.make_internal_vrg(128, 96, stats=None, device=torch.device("cpu"))
    for t in [x for x in REG["trials"] if x["id"] == trial_id]:
        m = build_fno(t, 0)
        m.train()
        for p in m.sa.parameters():
            p.requires_grad_(False)
        y = m(x, g)
        check("S4", f"{t['arm']}_output_shape_finite", tuple(y.shape) == (1, 256, 3) and bool(torch.isfinite(y).all()))
        y.square().mean().backward()
        miss = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all()))]
        zero = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None and float(p.grad.abs().sum()) == 0.0]
        check("S4", f"{t['arm']}_core_fp_head_params_finite_grad", not miss and sum(p.requires_grad for p in m.parameters()) == sum(1 for n, _ in m.named_parameters() if not n.startswith("sa.")), str(miss))
        check("S4", f"{t['arm']}_spectral_and_lift_grads_nonzero", not [n for n in zero if n.startswith("wno.")], str(zero))
        sd = m.state_dict()
        check("S4", f"{t['arm']}_buffers_after_forward", "wno.fixed_grid_xyhw" in sd and sd["grid_perm_hw"].numel() == 128 * 96)
    for r in RESULTS:
        print("S4JS " + json.dumps(r), flush=True)


# ------------------------------------------------------------------ S5
def run_cli(args):
    r = subprocess.run([sys.executable, "-B", os.path.join(PKG, "train_fno2d.py"), "--registry", REG_PATH] + args, cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return r.returncode, r.stdout


def s5(npz):
    before = sorted(d for d in os.listdir(os.path.join(ROOT, "results")) if d.startswith("fno2d"))
    cases = [("unregistered_trial", ["--trial", "FNO_X", "--seed", "0", "--npz", npz], "is not registered"),
             ("seed_3", ["--trial", "FNO_F1_L6W32M34", "--seed", "3", "--npz", npz], "seed 3 is not registered"),
             ("smoke_3_epochs", ["--trial", "FNO_F1_L6W32M34", "--seed", "0", "--npz", npz, "--max_epochs", "3", "--smoke_tag", "smoke"], "smoke must be"),
             ("smoke_without_tag", ["--trial", "FNO_F1_L6W32M34", "--seed", "0", "--npz", npz, "--max_epochs", "2"], "must be used together"),
             ("wrong_npz", ["--trial", "FNO_F1_L6W32M34", "--seed", "0", "--npz", REG_PATH], "npz_sha256")]
    for name, args, msg in cases:
        code, out = run_cli(args)
        check("S5", f"refuse_{name}", code != 0 and msg in out, out.strip().splitlines()[-1][:120] if out.strip() else "")
    for t in REG["trials"]:
        code, out = run_cli(["--trial", t["id"], "--seed", "2", "--npz", npz, "--device", "cpu", "--dry_run"])
        js = json.loads(out[out.index("{\n"):]) if "{\n" in out else {}
        ok = (code == 0 and js.get("dry_run") is True and js["test_cases_blocked"] == 12 and js["n_train"] == 96 and js["n_val"] == 12
              and js["params_trainable_real"] == t["params_trainable_real"] and js["seed"] == 2
              and js["out"] == os.path.join(ROOT, "results", "fno2d_cmp_r1", t["id"], "seed2"))
        check("S5", f"dry_run_{t['arm']}_seed2", ok, f"rng {js.get('rng_state_sha256_after_model', '')[:12]}")
    after = sorted(d for d in os.listdir(os.path.join(ROOT, "results")) if d.startswith("fno2d"))
    check("S5", "no_result_dirs_created", before == after == [])


# ------------------------------------------------------------------ S6
def write_run(d, operator, seed, case_vals, best_loss, extra_meta):
    os.makedirs(d)
    TG = T.TARGETS
    ids = REG["split"]["val_idx"]
    with open(os.path.join(d, "val_case_RMSE.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["global_data_id", "filename", "n_points"] + [f"NRMSE_{t}" for t in TG])
        for cid, v in zip(ids, case_vals):
            w.writerow([cid, f"case{cid}.dat", 16000] + [repr(float(x)) for x in v])
    nr = {t: float(np.mean([v[k] for v in case_vals])) for k, t in enumerate(TG)}
    json.dump({"best_epoch": 123, "best_val_loss": best_loss, "epochs_completed": 500, "train_sec": 1000.0 + seed,
               "val_case_mean_NRMSE": nr, "n_val_cases": 12, "test_evaluated": False, "peak_cuda_mem_mb": 5000.0 + seed},
              open(os.path.join(d, "val_summary.json"), "w"))
    with open(os.path.join(d, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_total", "train_data", "train_grad", "val_total", "val_data", "val_grad", "lr", "epoch_sec", "is_best"])
        for ep in range(1, 501):
            w.writerow([ep, 0.01, 0.01, 0, 0.003, 0.003, 0, 1e-3, 2.0, 0])
    meta = dict({"params_trainable_real": 0, "test_cases_blocked": 12, "gpu": "NVIDIA H100 NVL", "torch": "2.5.1+cu121", "pbs_jobid": "1.X"},
                **extra_meta)
    json.dump(meta, open(os.path.join(d, "run_meta.json"), "w"))
    open(os.path.join(d, "pcwno_best.pt"), "wb").write(b"x" * 10)
    with open(os.path.join(d, "SHA256SUMS_run.txt"), "w") as f:
        for fn in sorted(os.listdir(d)):
            if fn != "SHA256SUMS_run.txt":
                f.write(f"{sha(os.path.join(d, fn))}  {fn}\n")
    json.dump({"finished": "x"}, open(os.path.join(d, "complete.json"), "w"))
    return nr


def cases_with_mean3(rng, mean3):
    v = rng.uniform(0.005, 0.09, size=(12, 3))
    return v + (mean3 - v.mean())                                  # mean over cases and targets == mean3


def s6(scratch):
    rng = np.random.default_rng(5)
    W = REG["wno_reference"]
    wroot, froot = os.path.join(scratch, "wno"), os.path.join(scratch, "fno")
    c3_loss = {0: W["stage1_seed0"]["best_val_loss"], 1: 0.0025, 2: 3 * W["author_reported_3seed_mean"]["best_val_loss"] - W["stage1_seed0"]["best_val_loss"] - 0.0025}
    c3_m3 = {0: W["stage1_seed0"]["val_NRMSE_mean3"], 1: 0.04, 2: 3 * W["author_reported_3seed_mean"]["val_NRMSE_mean3"] - W["stage1_seed0"]["val_NRMSE_mean3"] - 0.04}
    vals = {}
    for s in (0, 1, 2):
        ref = W["runs"][str(s)]
        vals[("C3", s)] = cases_with_mean3(rng, c3_m3[s])
        write_run(os.path.join(wroot, ref["dir"]), "wno", s, vals[("C3", s)], c3_loss[s],
                  {"params_trainable_real": W["params_trainable_real"], "config": dict(REG["config"], seed=s),
                   "code_sha256": {"code/revision_2026/2d_original_r1/train_orig2d.py": W["runner_sha256"]}, "backward": W["backward"],
                   "n_waveconv_r1": 6, "registry_sha256": ref["registry_sha256"], "trial": W["id"]})
    runner_rel = os.path.relpath(os.path.join(PKG, "train_fno2d.py"), ROOT)
    f1, f2 = REG["trials"][0], REG["trials"][1]
    for t, seeds in ((f1, (0, 1, 2)), (f2, (0,))):
        for s in seeds:
            vals[(t["arm"], s)] = cases_with_mean3(rng, 0.041 + 0.001 * s)
            write_run(os.path.join(froot, t["id"], f"seed{s}"), "fno", s, vals[(t["arm"], s)], 0.0027 + 1e-4 * s,
                      {"params_trainable_real": t["params_trainable_real"], "config": cfg(t, s), "code_sha256": {runner_rel: sha(os.path.join(PKG, "train_fno2d.py"))},
                       "registry_sha256": sha(REG_PATH), "trial": t["id"], "operator": "fno", "fno": t["fno"], "seed": s})
    out = os.path.join(scratch, "collect1")
    r = subprocess.run([sys.executable, "-B", os.path.join(PKG, "collect_fno.py"), "--wno_root", wroot, "--fno_root", froot,
                        "--fno_gpumon_dir", os.path.join(scratch, "nogpumon"), "--out", out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    check("S6", "collector_exit0", r.returncode == 0, r.stdout.strip().splitlines()[-1][:200] if r.stdout.strip() else "")
    ck = json.load(open(os.path.join(out, "collect_checks.json")))
    check("S6", "runs_ok_and_missing", all(ck["runs"][k]["status"] == "ok" for k in ("C3_seed0", "C3_seed1", "C3_seed2", "F1_seed0", "F1_seed1", "F1_seed2", "F2_seed0"))
          and ck["runs"]["F2_seed1"]["status"] == "missing")
    check("S6", "complete_arms", ck["complete_arms"] == {"C3": True, "F1": True, "F2": False})
    check("S6", "C3_author_report_match", ck["c3_3seed_vs_author_report"].get("best_val_loss") is True
          and ck["c3_3seed_vs_author_report"].get("val_NRMSE_mean3") is True, json.dumps(ck["c3_3seed_vs_author_report"].get("recomputed")))
    summ = {r["arm"]: r for r in csv.DictReader(open(os.path.join(out, "fno_summary.csv")))}
    l3 = np.array([c3_loss[s] for s in (0, 1, 2)])
    check("S6", "C3_mean_std_ddof1", abs(float(summ["C3"]["best_val_loss_mean"]) - l3.mean()) < 1e-15
          and abs(float(summ["C3"]["best_val_loss_std"]) - l3.std(ddof=1)) < 1e-15)
    check("S6", "F2_incomplete_no_stats", summ["F2"]["status"] == "incomplete" and summ["F2"].get("best_val_loss_mean", "") == "")
    pr = [r for r in csv.DictReader(open(os.path.join(out, "fno_paired.csv"))) if r["pair"] == "F1-C3" and r["basis"] == "seed_mean"]
    ok = len(pr) == 4
    for k, row in zip(range(4), sorted(pr, key=lambda r: ["NRMSE_x_momentum", "NRMSE_y_momentum", "NRMSE_pressure_coefficient", "NRMSE_mean3"].index(r["metric"]))):
        a = np.mean([vals[("F1", s)] for s in (0, 1, 2)], axis=0)
        b = np.mean([vals[("C3", s)] for s in (0, 1, 2)], axis=0)
        a, b = (a.mean(axis=1), b.mean(axis=1)) if k == 3 else (a[:, k], b[:, k])
        d = a - b
        cnt = sum(abs(sum(sg * x for sg, x in zip(signs, d))) / 12 >= abs(d.mean()) * (1 - 1e-12) for signs in itertools.product((1, -1), repeat=12))
        half = 2.200985160082949 * d.std(ddof=1) / math.sqrt(12)
        ok &= (abs(float(row["mean_diff_A_minus_B"]) - d.mean()) < 1e-12 and abs(float(row["ci95_lo"]) - (d.mean() - half)) < 1e-9
               and abs(float(row["perm_p_exact_two_sided"]) - cnt / 4096) < 1e-12 and int(row["n_A_lower"]) == int((d < 0).sum()))
    check("S6", "paired_F1_C3_seedmean_vs_independent", ok)
    pf2 = [r for r in csv.DictReader(open(os.path.join(out, "fno_paired.csv"))) if r["pair"] == "F2-C3"]
    check("S6", "F2_pairs_seedmean_not_computed_seed0_computed", any(r["basis"] == "seed_mean" and r["status"].startswith("not computed") for r in pf2)
          and any(r["basis"] == "seed0" and r["status"] == "ok" for r in pf2) and any(r["basis"] == "seed1" and r["status"].startswith("not computed") for r in pf2))
    sp = [r for r in csv.DictReader(open(os.path.join(out, "fno_seed_paired.csv"))) if r["pair"] == "F1-C3" and r["metric"] == "best_val_loss"][0]
    exp = [0.0027 + 1e-4 * s - c3_loss[s] for s in (0, 1, 2)]
    check("S6", "seed_paired_best_val_loss", all(abs(float(sp[f"diff_seed{s}"]) - exp[s]) < 1e-15 for s in (0, 1, 2)) and sp["status"] == "ok")
    sums_ok = all(sha(os.path.join(out, fn)) == h for h, fn in (ln.split(None, 1) for ln in open(os.path.join(out, "SHA256SUMS.txt")).read().splitlines()))
    check("S6", "outputs_and_sums", sums_ok and sorted(os.listdir(out)) == sorted(["fno_runs.csv", "fno_summary.csv", "fno_val_cases.csv", "fno_case_seedmean.csv",
                                                                               "fno_paired.csv", "fno_seed_paired.csv", "fno_curves.csv", "collect_checks.json", "SHA256SUMS.txt"]))
    r2 = subprocess.run([sys.executable, "-B", os.path.join(PKG, "collect_fno.py"), "--wno_root", wroot, "--fno_root", froot, "--out", out],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    check("S6", "refuse_existing_out", r2.returncode != 0 and "exists" in r2.stdout)
    shutil.copytree(froot, os.path.join(scratch, "fno_tampered"))
    p = os.path.join(scratch, "fno_tampered", f1["id"], "seed1", "val_case_RMSE.csv")
    open(p, "a").write("\n")
    out3 = os.path.join(scratch, "collect_tampered")
    r3 = subprocess.run([sys.executable, "-B", os.path.join(PKG, "collect_fno.py"), "--wno_root", wroot, "--fno_root", os.path.join(scratch, "fno_tampered"),
                         "--out", out3], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    ck3 = json.load(open(os.path.join(out3, "collect_checks.json")))
    check("S6", "tampered_run_flagged_not_used", r3.returncode == 1 and ck3["runs"]["F1_seed1"]["status"] == "FAILED_CHECKS"
          and ck3["complete_arms"]["F1"] is False)


# ------------------------------------------------------------------ S7
def s7(scratch):
    sm, tr = os.path.join(PKG, "pbs", "smoke_fno_KITSM02.pbs"), os.path.join(PKG, "pbs", "train_fno_KITSM02.pbs")
    for p in (sm, tr):
        check("S7", f"bash_n_{os.path.basename(p)}", subprocess.run(["bash", "-n", p]).returncode == 0)
        txt = open(p).read()
        check("S7", f"no_delete_{os.path.basename(p)}", not re.search(r"(^|[\s;|&])(rm|rmdir|unlink|shred)\s", txt) and "shutil.rmtree" not in txt)
        check("S7", f"manifest_checks_{os.path.basename(p)}", "SHA256SUMS_fno.txt" in txt and "SHA256SUMS_orig2d.txt" in txt
              and "2e33cb1cca9a4fd226f731a896731fea89cc3d785fd46637c094c68f2d165e66" in txt)
    runs = {f"{t['id']}:{s}" for t in REG["trials"] for s in t["seeds"]}
    smt = open(sm).read()
    check("S7", "smoke_ALL_RUNS_equals_registry", set(re.search(r'ALL_RUNS="([^"]+)"', smt).group(1).split()) == runs)
    trt = open(tr).read()
    check("S7", "train_case_lists_equal_registry", "FNO_F1_L6W32M34|FNO_F2_L4W83M16)" in trt and "0|1|2)" in trt)
    check("S7", "train_gate_exit8_overwrite_exit5", "exit 8" in trt and 'test -e $OUT && { echo "FATAL $OUT exists' in trt and "exit 5" in trt)
    check("S7", "gate_keys_written_by_smoke", all(k in smt for k in ("fno_sums_sha256=fno_sha", "orig2d_sums_sha256=orig_sha", "run=rid", "seed=s", "ok=all")))
    # smoke gate logic on synthetic smoke directories (cwd = fake EAAI root with the C3 run_meta of the same seed)
    code = smt.split("<<'PYEOF'\n", 1)[1].split("\nPYEOF", 1)[0]
    gate_py = os.path.join(scratch, "gate_check.py")
    open(gate_py, "w").write(code)
    fake = os.path.join(scratch, "gate_root")
    t = REG["trials"][0]

    def smoke_dir(name, seed, mutate=None, ckpt_bytes=None, c3_torch="2.5.1+cu121"):
        c3d = os.path.join(fake, REG["wno_reference"]["runs"][str(seed)]["dir"])
        if not os.path.isdir(c3d):
            os.makedirs(c3d)
            json.dump({"torch": c3_torch, "gpu": "NVIDIA H100 NVL"}, open(os.path.join(c3d, "run_meta.json"), "w"))
        d = os.path.join(fake, "results", "fno2d_cmp_r1_smoke", name, t["id"], f"seed{seed}", "smoke")
        os.makedirs(d)
        meta = {"operator": "fno", "fno": t["fno"], "params_trainable_real": t["params_trainable_real"], "registry_sha256": sha(REG_PATH),
                "trial": t["id"], "config": {"seed": seed}, "seed": seed, "out": os.path.join("/x/results/fno2d_cmp_r1_smoke", t["id"], f"seed{seed}", "smoke"),
                "rng_state_sha256_after_model": "a" * 64, "test_cases_blocked": 12, "gpu": "NVIDIA H100 NVL", "torch": "2.5.1+cu121", "param_groups": {}}
        if mutate:
            mutate(meta)
        json.dump(meta, open(os.path.join(d, "run_meta.json"), "w"))
        json.dump({"test_evaluated": False, "peak_cuda_mem_mb": 1234.0}, open(os.path.join(d, "val_summary.json"), "w"))
        with open(os.path.join(d, "metrics.csv"), "w") as f:
            f.write("epoch,train_total,val_total,epoch_sec\n1,0.1,0.05,3.0\n2,0.05,0.04,2.5\n")
        with open(os.path.join(d, "val_case_RMSE.csv"), "w") as f:
            f.write("global_data_id\n" + "\n".join(str(i) for i in REG["split"]["val_idx"]) + "\n")
        for fn in ("pcwno_best.pt", "last_model.pt"):
            with open(os.path.join(d, fn), "wb") as f:
                f.truncate(ckpt_bytes or t["ckpt_bytes_expected"])
        json.dump({}, open(os.path.join(d, "complete.json"), "w"))
        with open(os.path.join(d, "SHA256SUMS_run.txt"), "w") as f:
            for fn in sorted(os.listdir(d)):
                if fn not in ("SHA256SUMS_run.txt", "complete.json"):
                    f.write(f"{sha(os.path.join(d, fn))}  {fn}\n")
        return d

    def gate(d, seed, name):
        g = os.path.join(scratch, f"gate_{name}.json")
        r = subprocess.run([sys.executable, "-B", gate_py, REG_PATH, t["id"], str(seed), d, g, "f" * 64, "o" * 64, "job"], cwd=fake,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        return r.returncode, os.path.isfile(g), r.stdout
    os.makedirs(fake)
    c, made, out = gate(smoke_dir("good", 1), 1, "good")
    check("S7", "gate_pass_writes_gate", c == 0 and made and json.load(open(os.path.join(scratch, "gate_good.json")))["fno_sums_sha256"] == "f" * 64, out[-200:] if c else "")
    for name, kw in (("wrong_params", {"mutate": lambda m: m.update(params_trainable_real=1)}),
                     ("wrong_registry", {"mutate": lambda m: m.update(registry_sha256="0" * 64)}),
                     ("wrong_out", {"mutate": lambda m: m.update(out="/x/results/fno2d_cmp_r1/a/seed1")}),
                     ("ckpt_size", {"ckpt_bytes": 1000})):
        c, made, out = gate(smoke_dir(name, 1, **kw), 1, name)
        check("S7", f"gate_refuses_{name}", c != 0 and not made)
    c, made, out = gate(smoke_dir("other_torch", 2, c3_torch="2.8.0"), 2, "other_torch")
    check("S7", "gate_refuses_torch_differs_from_C3", c != 0 and not made)


# ------------------------------------------------------------------ S8
def s8():
    src = open(os.path.join(PKG, "train_fno2d.py")).read()
    check("S8", "runner_no_test_loader", "test_loader" not in src and "test_ds" not in src and "evaluate_and_save(" not in src)
    uses = [ln.strip() for ln in src.splitlines() if "test_idx" in ln]
    check("S8", "runner_test_idx_only_load_check_block", len(uses) == 3 and any("block_test(" in u for u in uses), json.dumps(uses))
    col = open(os.path.join(PKG, "collect_fno.py")).read()
    check("S8", "collector_reads_no_test", "test_idx" not in col and "test_split" not in col)
    m = os.path.join(PKG, "SHA256SUMS_fno.txt")
    if os.path.isfile(m):
        lines = [ln.split(None, 1) for ln in open(m).read().splitlines() if ln.strip()]
        listed = {fn for _, fn in lines}
        files = {os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/") for dp, dn, fs in os.walk(PKG) for f in fs
                 if "__pycache__" not in dp and f != "SHA256SUMS_fno.txt"}
        check("S8", "manifest_matches_package", listed == files and all(sha(os.path.join(ROOT, fn)) == h for h, fn in lines), f"{len(lines)} files")
    else:
        check("S8", "manifest_matches_package", False, "SHA256SUMS_fno.txt not written yet")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--s4_trial":
        torch.set_num_threads(4)
        s4_one(sys.argv[2])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f"{a.out} exists")
    if os.path.exists(a.scratch) and os.listdir(a.scratch):
        sys.exit(f"{a.scratch} is not empty")
    os.makedirs(a.scratch, exist_ok=True)
    torch.set_num_threads(4)
    for f, args in ((s1, ()), (s2, ()), (s3, ()), (s4, ()), (s5, (a.npz,)), (s6, (a.scratch,)), (s7, (a.scratch,)), (s8, ())):
        try:
            f(*args)
        except Exception as ex:                                     # a crashing group is a failure, not a silent skip
            check(f.__name__.upper(), "group_crashed", False, f"{type(ex).__name__}: {ex}")
    n_pass = sum(r["pass"] for r in RESULTS)
    res = {"n_checks": len(RESULTS), "n_pass": n_pass, "all_pass": n_pass == len(RESULTS), "training": False, "optimizer_steps": 0,
           "device": "cpu", "torch": torch.__version__, "registry_sha256": sha(REG_PATH), "checks": RESULTS}
    json.dump(res, open(a.out, "x"), indent=1)
    print(f"{n_pass}/{len(RESULTS)} PASS")
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
