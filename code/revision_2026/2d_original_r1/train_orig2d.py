"""EAAI-26-17015, Original 2D PointWNO retraining (2026-10-29): the manuscript run, statement for statement, with one
registered change of the wavelet backward (or none, for the library-backward control) and the test split blocked.

  python -B code/revision_2026/2d_original_r1/train_orig2d.py --registry code/revision_2026/2d_original_r1/configs/orig2d_registry.json \
      --trial ORIG2D_r1 --npz <Data_allpoints_xymomcp_from_dat.npz> [--device cuda] [--dry_run] [--max_epochs N --smoke_tag smoke]

Source of every statement: the manuscript script Code_Final/pcwno_vrg_lpe_torch_airfoil_momentum_allpoints_fromdat_masked_steplr.py
(vendored unchanged in vendored/, sha256 pinned) and its model module (code/third_party/pointwno2d_original, sha256 pinned).
main() of that script is followed in the same order (set_seed -> load_allpoints_npz -> datasets -> DataLoaders on the GLOBAL
torch RNG -> make_internal_vrg -> PCWNO -> MaskedGradTrainer -> StepLR -> epoch loop train / val / best-by-val-loss / StepLR /
min_lr clamp), with the manuscript run's command-line values (log header: db2, level 1, batch 4, step 100, gamma 0.5, min_lr 1e-5,
grad_weight 0, knn_k 6; script defaults: width 128, 6 layers, grid 128x96, grid_range (1, 1), lr 1e-3, epochs 500, seed 0).
Differences, all registered in configs/orig2d_registry.json:
  1. backward: trial ORIG2D_r1 swaps the class of the 6 WaveConv2d layers to WaveConv2dR1 (exact autograd adjoint, forward
     bit-identical, no RNG drawn); trial ORIG2D_lib keeps the library backward (control: same harness, same hardware).
  2. test: the test cases are overwritten with NaN / mask 0 right after loading, no test dataset or loader is built and nothing is
     evaluated on test.  (The arrays are built by the original loader from the single npz, as in the manuscript run.)
  3. outputs: progress bars off (tqdm draws no random numbers); metrics.csv gets extra columns (lr, epoch time, val RMSE per
     target computed from the same validation predictions); the final weights are also saved (last_model.pt, weights only);
     after training the best checkpoint is evaluated on VALIDATION with the formula of evaluate_and_save (per-case RMSE).
No resume: a stopped run is restarted from scratch in a new directory (the RNG stream of the manuscript run cannot be resumed).
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
for p in (os.path.join(HERE, "vendored"), os.path.join(CODE, "third_party", "pointwno2d_original"),
          os.path.join(CODE, "revision_2026", "2d_recovery", "r1_exact_backward")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import pcwno_vrg_lpe_torch_airfoil_momentum_allpoints_fromdat_masked_steplr as OS  # noqa: E402  manuscript script, unchanged
from pcwno_vrg_lpe_torch_airfoil_momentum_fno_same import PCWNO, make_internal_vrg, assert_grid_shape  # noqa: E402
import wno2d_r1 as W  # noqa: E402  WaveConv2dR1 / convert_to_r1 (2D recovery r1 package, unchanged)

TARGETS = ("x_momentum", "y_momentum", "pressure_coefficient")
FORBIDDEN_ROOTS = ("results/ext_2d", "results/a1_2d", "results/hpo_2d", "results/sd3d", "results/revision_2026", "results/2d_recovery_r1")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def check_registry(reg_path, trial_id, npz):
    reg = json.load(open(reg_path))
    t = [x for x in reg["trials"] if x["id"] == trial_id]
    if len(t) != 1:
        sys.exit(f"trial {trial_id!r} is not registered in {reg_path}")
    bad = [p for p, s in reg["pinned_sha256"].items() if sha256(resolve(p)) != s]
    if bad:
        sys.exit(f"files differ from the sha256 pinned in {reg_path}: {bad}")
    if npz is not None and sha256(npz) != reg["npz_sha256"]:
        sys.exit(f"{npz}: sha256 != registry npz_sha256")
    return reg, t[0]


def block_test(inputs, targets, masks, test_idx):
    """NaN inputs / targets and zero mask for every test case (after the original loader, before anything else)."""
    for i in test_idx:
        inputs[int(i)][...] = np.nan
        targets[int(i)][...] = np.nan
        masks[int(i)][...] = 0.0
    return len(test_idx)


def numerics():
    return {"cudnn.allow_tf32": torch.backends.cudnn.allow_tf32, "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "cudnn.benchmark": torch.backends.cudnn.benchmark,
            "cudnn.deterministic": torch.backends.cudnn.deterministic}


@torch.no_grad()
def val_case_rmse(model, loader, grid_ref, device, filenames):
    """evaluate_and_save's per-case formula (sqrt(mean over real points) per channel), on the validation loader, no file."""
    model.eval()
    rows = []
    for batch in loader:
        xyz, y = batch["xyz"].to(device), batch["y"].to(device)
        pred = model(xyz, grid_ref)
        err = pred - y
        for b in range(xyz.shape[0]):
            n = int(batch["n_valid"][b])
            r = torch.sqrt((err[b, :n, :] ** 2).mean(dim=0)).cpu().numpy()
            gid = int(batch["case_id"][b])
            rows.append({"global_data_id": gid, "filename": str(filenames[gid]), "n_points": n,
                         **{f"NRMSE_{t}": float(r[k]) for k, t in enumerate(TARGETS)}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_root", default=os.path.join(ROOT, "results", "orig2d_r1"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke only")
    ap.add_argument("--smoke_tag", default=None, help="required with --max_epochs")
    ap.add_argument("--dry_run", action="store_true", help="registry, data, test blocking, model; no training, no files")
    a = ap.parse_args()
    if (a.max_epochs is None) != (a.smoke_tag is None):
        sys.exit("--max_epochs and --smoke_tag must be used together")
    reg_path = resolve(a.registry)
    reg, trial = check_registry(reg_path, a.trial, a.npz)
    C = reg["config"]
    rel = os.path.relpath(os.path.abspath(a.out_root), ROOT).replace(os.sep, "/")
    if any(rel == f or rel.startswith(f + "/") for f in FORBIDDEN_ROOTS):
        sys.exit(f"--out_root {a.out_root} is inside a directory of earlier results; refused")
    out = os.path.join(a.out_root, trial["id"], f"seed{C['seed']}")
    if a.smoke_tag:
        out = os.path.join(out, a.smoke_tag)
    epochs = C["epochs"] if a.max_epochs is None else int(a.max_epochs)

    # ---- main() of the manuscript script, same order
    OS.set_seed(C["seed"])
    if not a.dry_run:
        if os.path.exists(out) and os.listdir(out):
            sys.exit(f"{out} exists and is not empty: no resume, no overwrite (start a new run directory)")
        os.makedirs(out, exist_ok=True)
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
    model = PCWNO(grid_hw=(C["grid_h"], C["grid_w"]), wno_width=C["wno_width"], wno_level=C["wno_level"],
                  wno_layers=C["wno_layers"], wno_wavelet=C["wavelet"], wno_grid_range=tuple(C["grid_range"])).to(device)
    n_r1 = W.convert_to_r1(model) if trial["backward"] == W.R1_BACKWARD_VERSION else 0
    if trial["backward"] not in (W.R1_BACKWARD_VERSION, "library_pytorch_wavelets_1.3.0"):
        sys.exit(f"unknown backward {trial['backward']!r}")
    if n_r1 != (C["wno_layers"] if trial["backward"] == W.R1_BACKWARD_VERSION else 0):
        sys.exit(f"converted {n_r1} WaveConv2d layers")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
    if n_params_real != reg["params_trainable_real"]:
        sys.exit(f"trainable real parameters {n_params_real} != registry {reg['params_trainable_real']}")
    trainer = OS.MaskedGradTrainer(model, lr=C["lr"], grad_weight=C["grad_weight"])
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.opt, step_size=C["step_size"], gamma=C["gamma"])
    info = {"trial": trial["id"], "backward": trial["backward"], "n_waveconv_r1": n_r1, "params_trainable_real": n_params_real,
            "params_trainable_tensors_numel": n_params, "test_cases_blocked": n_blocked, "n_train": len(train_ds), "n_val": len(val_ds),
            "out": out, "epochs": epochs}
    if a.dry_run:
        print(json.dumps(dict(info, dry_run=True), indent=1))
        return

    code_files = [os.path.abspath(__file__), OS.__file__, sys.modules["pcwno_vrg_lpe_torch_airfoil_momentum_fno_same"].__file__,
                  sys.modules["wavelet_convolution_v3"].__file__, W.__file__, W.WAVELET_EXACT_PATH, reg_path]
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
