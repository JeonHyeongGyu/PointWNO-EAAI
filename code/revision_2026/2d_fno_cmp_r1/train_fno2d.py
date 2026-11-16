"""EAAI-26-17015, 2D FNO comparison stage (2026-11-01): PointFNO runs on the training engine of the Original 2D PointWNO
capacity study, so that the only registered difference to WNO C3 (6 layers x width 32) is the latent operator.

  python -B code/revision_2026/2d_fno_cmp_r1/train_fno2d.py --registry code/revision_2026/2d_fno_cmp_r1/configs/fno_registry.json \
      --trial FNO_F1_L6W32M34 --seed 0 --npz <Data_allpoints_xymomcp_from_dat.npz> [--device cuda] [--dry_run] [--max_epochs 2 --smoke_tag smoke]

main() follows main() of code/revision_2026/2d_original_r1/train_orig2d.py (the runner of C3 seed 0 / 1 / 2, unchanged, sha256
pinned; its helper functions are imported, not copied) in the same order: set_seed -> load_allpoints_npz (manuscript loader: split
default_rng(0), train-only min-max) -> test blocking -> datasets -> DataLoaders on the GLOBAL torch RNG -> make_internal_vrg -> model
-> MaskedGradTrainer (Adam) -> StepLR -> epoch loop train / val / best-by-val-total-loss (strict <) / StepLR / min_lr clamp ->
last_model.pt -> validation per-case RMSE of the best checkpoint.  tests/check_fno_static.py compares the two main() functions
statement by statement.  Registered differences:
  1. model: PCWNO is built EXACTLY as for WNO C3 (same class, arguments wno_width 32 / wno_layers 6 / db2 / level 1, same order of
     random draws), so LPE, FP and head get the C3 initial values of this seed and the global RNG (DataLoader shuffle, the LPE's
     per-step torch.rand) continues exactly as in the C3 run of this seed.  Then the WNO core is replaced by FNO2dGridCore of
     code/pwno2d/models.py (the core of FNO B2, unchanged, sha256 pinned), whose weights are drawn inside torch.random.fork_rng
     from torch.manual_seed(seed + fno_core_init_seed_offset): the global CPU RNG state is restored afterwards.
  2. no wavelet backward conversion (an FNO has no DWT); the registry has no backward field for FNO trials.
  3. seed from --seed (registered seeds only) instead of config.seed; trial-specific FNO width / layers / modes from the registry.
  4. output roots fixed (results/fno2d_cmp_r1, smoke results/fno2d_cmp_r1_smoke); an existing run directory is refused even if empty.
  5. run_meta.json additionally records the FNO configuration, the parameter groups and the sha256 of the CPU RNG state after model
     construction (equal to the state after the C3 construction of the same seed; checked in tests/check_fno_static.py).
Checkpoint file names are those of the WNO runs (pcwno_best.pt, last_model.pt) so that one collector reads both arms.
No test data is used: the test cases are overwritten with NaN / mask 0 right after loading; no test loader exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CODE = os.path.join(ROOT, "code")
ORIG = os.path.join(CODE, "revision_2026", "2d_original_r1")
for p in (ORIG, os.path.join(ORIG, "vendored"), os.path.join(CODE, "third_party", "pointwno2d_original"), CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_orig2d as T  # noqa: E402  WNO C3 runner (unchanged): sha256, resolve, block_test, numerics, val_case_rmse, TARGETS
import pcwno_vrg_lpe_torch_airfoil_momentum_allpoints_fromdat_masked_steplr as OS  # noqa: E402  manuscript script, unchanged
from pcwno_vrg_lpe_torch_airfoil_momentum_fno_same import PCWNO, make_internal_vrg, assert_grid_shape  # noqa: E402
from pwno2d.models import FNO2dGridCore, count_parameters  # noqa: E402  FNO core of B2, unchanged

sha256, resolve, block_test, numerics, val_case_rmse, TARGETS = T.sha256, T.resolve, T.block_test, T.numerics, T.val_case_rmse, T.TARGETS
OUT_ROOTS = {"train": "results/fno2d_cmp_r1", "smoke": "results/fno2d_cmp_r1_smoke"}


def check_registry(reg_path, trial_id, seed, npz):
    reg = json.load(open(reg_path))
    t = [x for x in reg["trials"] if x["id"] == trial_id]
    if len(t) != 1:
        sys.exit(f"trial {trial_id!r} is not registered in {reg_path}")
    if seed not in t[0]["seeds"]:
        sys.exit(f"seed {seed} is not registered for {trial_id} (registered: {t[0]['seeds']})")
    bad = [p for p, s in reg["pinned_sha256"].items() if sha256(resolve(p)) != s]
    if bad:
        sys.exit(f"files differ from the sha256 pinned in {reg_path}: {bad}")
    if npz is not None and sha256(npz) != reg["npz_sha256"]:
        sys.exit(f"{npz}: sha256 != registry npz_sha256")
    return reg, t[0]


def build_model(C):
    """WNO C3 construction (LPE / FP / head initial values and the global RNG stream of C3), then the FNO core from its own stream."""
    model = PCWNO(grid_hw=(C["grid_h"], C["grid_w"]), wno_width=C["wno_width"], wno_level=C["wno_level"],
                  wno_layers=C["wno_layers"], wno_wavelet=C["wavelet"], wno_grid_range=tuple(C["grid_range"]))
    sa_out = model.wno.fc0.in_features - 2
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(C["seed"] + C["fno_core_init_seed_offset"])
        core = FNO2dGridCore(width=C["fno_width"], layers=C["fno_layers"], modes1=C["fno_modes"][0], modes2=C["fno_modes"][1],
                             in_channel=sa_out, grid_range=tuple(C["grid_range"]))
    model.wno = core
    return model


def rng_state_sha256():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--smoke_tag", default=None, help="required with --max_epochs")
    ap.add_argument("--dry_run", action="store_true", help="registry, data, test blocking, model; no training, no files")
    a = ap.parse_args()
    if (a.max_epochs is None) != (a.smoke_tag is None):
        sys.exit("--max_epochs and --smoke_tag must be used together")
    reg_path = resolve(a.registry)
    reg, trial = check_registry(reg_path, a.trial, a.seed, a.npz)
    C = dict(reg["config"], seed=a.seed, fno_core_init_seed_offset=reg["fno_core_init_seed_offset"],
             fno_width=trial["fno"]["width"], fno_layers=trial["fno"]["layers"], fno_modes=list(trial["fno"]["modes"]))
    out = os.path.join(ROOT, OUT_ROOTS["smoke" if a.smoke_tag else "train"], trial["id"], f"seed{C['seed']}")
    if a.smoke_tag:
        if a.smoke_tag != "smoke" or a.max_epochs != reg["smoke_epochs"]:
            sys.exit(f"smoke must be --max_epochs {reg['smoke_epochs']} --smoke_tag smoke")
        out = os.path.join(out, a.smoke_tag)
    epochs = C["epochs"] if a.max_epochs is None else int(a.max_epochs)

    # ---- main() of the manuscript script / train_orig2d.py, same order
    OS.set_seed(C["seed"])
    if not a.dry_run:
        if os.path.lexists(out):
            sys.exit(f"{out} exists: no resume, no overwrite (move it aside by hand)")
        os.makedirs(out)
    (inputs, targets, masks, knn_idx, filenames, n_points,
     train_idx, val_idx, test_idx, stats, max_points) = OS.load_allpoints_npz(a.npz, knn_k=C["knn_k"])
    for k, v in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if [int(i) for i in v] != reg["split"][f"{k}_idx"]:
            sys.exit(f"{k}_idx differs from the registry (manuscript log)")
    n_blocked = block_test(inputs, targets, masks, test_idx)
    train_ds = OS.AllPointsMaskedDataset(train_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    val_ds = OS.AllPointsMaskedDataset(val_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    train_loader = DataLoader(train_ds, batch_size=C["batch"], shuffle=True, drop_last=False)      # global RNG, as the original
    val_loader = DataLoader(val_ds, batch_size=C["batch"], shuffle=False, drop_last=False)
    device = torch.device(a.device)
    grid_ref = make_internal_vrg(C["grid_h"], C["grid_w"], stats=None, device=device)
    assert_grid_shape(grid_ref, C["grid_h"], C["grid_w"])
    model = build_model(C).to(device)
    rng_after_model = rng_state_sha256()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
    if n_params_real != trial["params_trainable_real"]:
        sys.exit(f"trainable real parameters {n_params_real} != registry {trial['params_trainable_real']}")
    trainer = OS.MaskedGradTrainer(model, lr=C["lr"], grad_weight=C["grad_weight"])
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.opt, step_size=C["step_size"], gamma=C["gamma"])
    info = {"trial": trial["id"], "seed": C["seed"], "operator": "fno", "fno": trial["fno"], "params_trainable_real": n_params_real,
            "param_groups": count_parameters(model), "params_trainable_tensors_numel": n_params, "test_cases_blocked": n_blocked,
            "n_train": len(train_ds), "n_val": len(val_ds), "out": out, "epochs": epochs, "rng_state_sha256_after_model": rng_after_model}
    if a.dry_run:
        print(json.dumps(dict(info, dry_run=True), indent=1))
        return

    code_files = [os.path.abspath(__file__), T.__file__, OS.__file__, sys.modules["pcwno_vrg_lpe_torch_airfoil_momentum_fno_same"].__file__,
                  sys.modules["wavelet_convolution_v3"].__file__, sys.modules["pwno2d.models"].__file__, reg_path]
    import pytorch_wavelets
    meta = dict(info, registry=os.path.relpath(reg_path, ROOT), registry_sha256=sha256(reg_path), config=C,
                code_sha256={os.path.relpath(p, ROOT): sha256(p) for p in code_files}, npz=os.path.abspath(a.npz), npz_sha256=reg["npz_sha256"],
                torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), pytorch_wavelets=pytorch_wavelets.__version__,
                python=platform.python_version(), host=platform.node(), device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, numerics=numerics(),
                pbs_jobid=os.environ.get("PBS_JOBID"), started=time.strftime("%Y-%m-%d %H:%M:%S"), smoke_tag=a.smoke_tag,
                checkpoint_policy=reg["checkpoint_policy"], test_policy=reg["test_policy"])
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "x"), indent=1)

    metrics_csv = os.path.join(out, "metrics.csv")
    with open(metrics_csv, "x", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_total", "train_data", "train_grad", "val_total", "val_data", "val_grad", "lr",
                                "epoch_sec", "is_best"] + [f"val_rmse_{t}" for t in TARGETS])
    best_val, best_epoch = float("inf"), 0
    t_run = time.perf_counter()
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        lr_used = trainer.opt.param_groups[0]["lr"]
        train_total = train_data = train_grad = 0.0
        n_train = 0
        for batch in train_loader:
            xyz, y, mask, kidx = (batch[k].to(device) for k in ("xyz", "y", "mask", "knn_idx"))
            total, dloss, gloss = trainer.step(xyz, grid_ref, y, mask, kidx)
            bs = xyz.size(0)
            train_total += total * bs; train_data += dloss * bs; train_grad += gloss * bs; n_train += bs
        train_total /= n_train; train_data /= n_train; train_grad /= n_train
        val_total = val_data = val_grad = 0.0
        n_val, sse, nn_ = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                xyz, y, mask, kidx = (batch[k].to(device) for k in ("xyz", "y", "mask", "knn_idx"))
                total, dloss, gloss, pred = trainer.eval(xyz, grid_ref, y, mask, kidx)
                bs = xyz.size(0)
                val_total += total * bs; val_data += dloss * bs; val_grad += gloss * bs; n_val += bs
                for b in range(bs):                                          # extra metric from the same predictions
                    n = int(batch["n_valid"][b])
                    sse.append(((pred[b, :n, :] - y[b, :n, :]) ** 2).sum(dim=0).double().cpu().numpy()); nn_.append(n)
        val_total /= n_val; val_data /= n_val; val_grad /= n_val
        if not np.isfinite(val_total):
            sys.exit(f"non-finite validation loss at epoch {ep}")
        vr = np.sqrt(np.array(sse) / np.array(nn_)[:, None]).mean(axis=0)
        is_best = val_total < best_val
        if is_best:                                                          # the manuscript rule: strict <, val total loss
            best_val, best_epoch = val_total, ep
            torch.save(model.state_dict(), os.path.join(out, "pcwno_best.pt"))
        scheduler.step()
        for group in trainer.opt.param_groups:                               # min_lr clamp, as the original
            if group["lr"] < C["min_lr"]:
                group["lr"] = C["min_lr"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{train_total:.6e}", f"{train_data:.6e}", f"{train_grad:.6e}", f"{val_total:.6e}", f"{val_data:.6e}",
                                    f"{val_grad:.6e}", f"{lr_used:.6e}", f"{time.perf_counter() - t0:.3f}", int(is_best)]
                                   + [f"{v:.8e}" for v in vr])
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[Epoch {ep}] train {train_total:.6e} val {val_total:.6e} best {best_val:.6e}@{best_epoch} "
                  f"val_rmse {' '.join(f'{v:.5f}' for v in vr)} lr {lr_used:.2e}", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "last_model.pt"))
    train_sec = time.perf_counter() - t_run

    model.load_state_dict(torch.load(os.path.join(out, "pcwno_best.pt"), map_location=device))
    rows = val_case_rmse(model, val_loader, grid_ref, device, filenames)
    with open(os.path.join(out, "val_case_RMSE.csv"), "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_completed": epochs, "train_sec": train_sec,
            "val_case_mean_NRMSE": {t: float(np.mean([r[f"NRMSE_{t}"] for r in rows])) for t in TARGETS}, "n_val_cases": len(rows),
            "test_evaluated": False, "peak_cuda_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
    json.dump(summ, open(os.path.join(out, "val_summary.json"), "x"), indent=1)
    files = sorted(f for f in os.listdir(out) if os.path.isfile(os.path.join(out, f)) and f not in ("SHA256SUMS_run.txt", "complete.json"))
    with open(os.path.join(out, "SHA256SUMS_run.txt"), "x") as f:
        for fn in files:
            f.write(f"{sha256(os.path.join(out, fn))}  {fn}\n")
    json.dump(dict(summ, finished=time.strftime("%Y-%m-%d %H:%M:%S"), trial=trial["id"]), open(os.path.join(out, "complete.json"), "x"), indent=1)
    print("val", summ["val_case_mean_NRMSE"], "best epoch", best_epoch, "| test: never evaluated", flush=True)


if __name__ == "__main__":
    main()
