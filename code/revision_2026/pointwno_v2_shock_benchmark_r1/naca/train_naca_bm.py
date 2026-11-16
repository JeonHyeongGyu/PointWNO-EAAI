"""NACA (2-D variable geometry, M 0.65) PointWNO-V2 / PointFNO training run of the benchmark, with atomic checkpoint + resume.

  python -B naca/train_naca_bm.py --config configs/naca/<run_id>.json --npz <Data_allpoints_xymomcp_from_dat.npz>
         --out_root <results/revision_2026/pointwno_v2_shock_benchmark_r1> [--budget_s S] [--stop_after_epoch E (smoke)]
         [--device cuda] [--dry_run]

Statement order = main() of code/revision_2026/2d_fno_cmp_r1/train_fno2d.py (itself = train_orig2d.py / the manuscript
script; all imported unchanged, sha256 pinned): set_seed -> load_allpoints_npz (manuscript split default_rng(0), train-only
min-max) -> test blocking (NaN / mask 0) -> datasets -> DataLoaders on the GLOBAL torch RNG -> make_internal_vrg -> model ->
MaskedGradTrainer (Adam 1e-3) -> StepLR(100, 0.5) -> epoch loop train / val / best-by-val-total-loss (strict <) / StepLR /
min_lr clamp. Precision: PyTorch defaults (as the manuscript run and F1).
Models (registered in configs/benchmark_registry.json):
  kernel 'v2'  : PCWNO built EXACTLY as WNO C3 (same class / arguments / global-RNG draws -> LPE, FP, head initial values and
                 the shuffle / LPE torch.rand stream of C3 and F1 of this seed), then its 6 wavelet layers are replaced by
                 wno_variants.V2Conv (official WNO V2.0.0 kernel, exact-adjoint DWT) drawn from a DEDICATED generator seeded
                 20261110 + seed (global RNG untouched).
  kernel 'fno' : train_fno2d.build_model (unchanged): C3 construction, then FNO2dGridCore drawn inside fork_rng(seed + 1000).
Resume (addition; train_fno2d has none): after every epoch last.pt is written atomically with model, optimizer, scheduler,
best weights, best value / epoch and the CPU / CUDA / numpy / python RNG states, so a resumed run continues the SAME global RNG
stream (DataLoader shuffle draws its seed from the global CPU generator at the start of each epoch). metrics.csv rows after the
checkpointed epoch are dropped on resume; pcwno_best.pt is rewritten from last.pt. Provenance (config, code, data, pins) is
hashed into rec_sha; a resume with a different rec_sha is refused.
Status (status.json): RUNNING / INTERRUPTED / DONE / DIVERGED / ERROR. DONE is never retrained; DIVERGED / ERROR are terminal.
No test data is used: test cases are overwritten with NaN / mask 0 right after loading; no test loader exists.
Exit: 0 DONE | 3 INTERRUPTED (budget / smoke stop; resubmit resumes) | 4 terminal | 2 refused
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import bm_common as B  # noqa: E402

ORIG = os.path.join(B.REV, "2d_original_r1")
B.add_path(B.CODE, os.path.join(B.CODE, "third_party", "pointwno2d_original"), os.path.join(ORIG, "vendored"), ORIG,
           os.path.join(B.REV, "2d_recovery", "r1_exact_backward"), B.NACA_FNO_PKG)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_fno2d as TF  # noqa: E402  (unchanged; imports train_orig2d + manuscript script + FNO core)

T, OS = TF.T, TF.OS
PCWNO, make_internal_vrg, assert_grid_shape = TF.PCWNO, TF.make_internal_vrg, TF.assert_grid_shape
TRAINER_VERSION = "bm_naca_train_v1"
EXIT_OK, EXIT_INTERRUPTED, EXIT_TERMINAL, EXIT_REFUSED = 0, 3, 4, 2


class Stop(Exception):
    pass


def verify_dependencies():
    """fno_registry.json and every file it pins + the 2d_fno_cmp_r1 manifest (EAAI-rooted) + wno_variants.py."""
    if B.sha256_file(B.NACA_FNO_REGISTRY) != B.NACA_FNO_REGISTRY_SHA256:
        raise SystemExit("STOP fno_registry.json differs")
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    B.verify_pins(reg["pinned_sha256"], B.EAAI)
    B.verify_manifest(os.path.join(B.NACA_FNO_PKG, "SHA256SUMS_fno.txt"), B.NACA_FNO_MANIFEST_SHA256, "NACA_FNO_PKG", base=B.EAAI)
    B.load_wno_variants()
    return reg


def check_config(cfg):
    """A run config may differ from the registered F1 protocol only in model (kernel) and the seed."""
    reg = B.read_json(B.NACA_FNO_REGISTRY)
    assert cfg["dataset"] == "naca" and cfg["package"] == B.PACKAGE
    base = dict(reg["config"])
    C = cfg["C"]
    for k, v in base.items():
        if k == "source_of_values":
            continue
        if k == "epochs":
            assert C["epochs"] == (cfg["smoke_epochs"] if cfg["smoke"] else v), "epochs"
            continue
        assert C[k] == v, ("config differs from the registered protocol", k)
    assert C["seed"] == cfg["seed"] in B.SEEDS and cfg["split"] == reg["split"]
    assert cfg["npz_sha256"] == reg["npz_sha256"] == B.NACA_NPZ_SHA256
    k = cfg["model"]["kernel"]
    assert k in ("v2", "fno")
    if k == "v2":
        assert cfg["model"]["level"] in (1, 2) and cfg["model"]["init_generator_seed"] == B.V2_INIT_SEED_BASE + cfg["seed"]
    else:
        assert C["fno_core_init_seed_offset"] == reg["fno_core_init_seed_offset"] and C["fno_width"] == 32 and C["fno_layers"] == 6
    expect = B.read_json(B.REGISTRY)["datasets"]["naca"]["arms"]
    a = [x for x in expect if x["arm"] == cfg["arm"]][0]
    assert B.model_for_seed(a["model"], cfg["seed"]) == cfg["model"] and a["params_trainable_real"] == cfg["expected_params"]


def build_model(cfg):
    C = cfg["C"]
    if cfg["model"]["kernel"] == "fno":
        return TF.build_model(C)                                   # unchanged F1 construction, own modes
    model = PCWNO(grid_hw=(C["grid_h"], C["grid_w"]), wno_width=C["wno_width"], wno_level=C["wno_level"],
                  wno_layers=C["wno_layers"], wno_wavelet=C["wavelet"], wno_grid_range=tuple(C["grid_range"]))
    WV = B.load_wno_variants()
    g = torch.Generator().manual_seed(cfg["model"]["init_generator_seed"])
    n = WV.to_v2(model.wno, cfg["model"]["level"], g)
    assert n == C["wno_layers"]
    c0 = model.wno.conv[0]
    assert [c0.modes1, c0.modes2] == cfg["model"]["coef_grid"], ([c0.modes1, c0.modes2], cfg["model"]["coef_grid"])
    return model


def prime_buffers(model, sd):
    """PCWNO registers grid_perm_hw (empty) and the core's fixed_grid_xyhw (None) at construction and fills them on the first
    forward; a state_dict saved after training holds their filled shapes. Re-shape those buffers (values are then loaded by
    load_state_dict; they are recomputed identically from grid_ref in every forward anyway)."""
    n = 0
    for k, v in sd.items():
        path, _, name = k.rpartition(".")
        mod = model.get_submodule(path) if path else model
        if name in mod._buffers and (mod._buffers[name] is None or tuple(mod._buffers[name].shape) != tuple(v.shape)):
            ref = next(model.parameters())
            mod._buffers[name] = torch.empty(v.shape, dtype=v.dtype, device=ref.device)
            n += 1
    return n


def count_real(model):
    return int(sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad))


def rng_states():
    return {"torch_cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_states(s):
    torch.set_rng_state(s["torch_cpu"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def atomic_torch_save(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def provenance(cfg, cfg_path, reg):
    files = [os.path.abspath(__file__), os.path.join(PKG, "bm_common.py"), TF.__file__, T.__file__, OS.__file__,
             sys.modules["pcwno_vrg_lpe_torch_airfoil_momentum_fno_same"].__file__, sys.modules["pwno2d.models"].__file__,
             B.WNO_VARIANTS, B.REGISTRY, B.PROTOCOL]
    return {"config": cfg, "config_sha256": B.sha256_file(cfg_path), "trainer_version": TRAINER_VERSION,
            "code_sha256": {os.path.relpath(p, B.EAAI): B.sha256_file(p) for p in files},
            "fno_registry_sha256": B.NACA_FNO_REGISTRY_SHA256, "pinned_sha256": reg["pinned_sha256"],
            "npz_sha256": B.NACA_NPZ_SHA256}


METRIC_COLS = ["epoch", "train_total", "train_data", "train_grad", "val_total", "val_data", "val_grad", "lr", "epoch_sec",
               "is_best", "peak_alloc_mib"] + [f"val_rmse_{t}" for t in T.TARGETS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_root", default=B.OUT_ROOT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget_s", type=float, default=None)
    ap.add_argument("--stop_after_epoch", type=int, default=None, help="smoke resume test only")
    ap.add_argument("--break_stale_lock", default=None)
    ap.add_argument("--dry_run", action="store_true", help="checks + model build + data; no file, no training")
    a = ap.parse_args()
    cfg = B.read_json(a.config)
    check_config(cfg)
    if a.stop_after_epoch is not None and not cfg["smoke"]:
        sys.exit("--stop_after_epoch is a smoke-only resume test")
    reg = verify_dependencies()
    if B.sha256_file(a.npz) != B.NACA_NPZ_SHA256:
        sys.exit(f"{a.npz}: sha256 != registered NACA npz")
    out_root = os.path.abspath(a.out_root)
    if not a.dry_run:
        B.assert_out_root_safe(out_root)
    run_dir = os.path.join(out_root, "naca", "smoke" if cfg["smoke"] else "runs", cfg["run_id"])
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but not available; training is refused on CPU")
    st = B.read_status(run_dir) if os.path.isdir(run_dir) else None
    if st and st["status"] == "DONE":
        print(f"[skip] {cfg['run_id']} already DONE", flush=True)
        sys.exit(EXIT_OK)
    if st and st["status"] in ("DIVERGED", "ERROR"):
        print(f"[stop] {cfg['run_id']} terminal status {st['status']}; never retried automatically", flush=True)
        sys.exit(EXIT_TERMINAL)
    rec = provenance(cfg, a.config, reg)
    rec_sha = B.canonical_hash(rec)
    C = cfg["C"]

    # ---- main() of train_fno2d / train_orig2d, same order
    OS.set_seed(C["seed"])
    (inputs, targets, masks, knn_idx, filenames, n_points,
     train_idx, val_idx, test_idx, stats, max_points) = OS.load_allpoints_npz(a.npz, knn_k=C["knn_k"])
    for k, v in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if [int(i) for i in v] != cfg["split"][f"{k}_idx"]:
            sys.exit(f"{k}_idx differs from the registry (manuscript log)")
    n_blocked = T.block_test(inputs, targets, masks, test_idx)
    train_ds = OS.AllPointsMaskedDataset(train_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    val_ds = OS.AllPointsMaskedDataset(val_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    train_loader = DataLoader(train_ds, batch_size=C["batch"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=C["batch"], shuffle=False, drop_last=False)
    grid_ref = make_internal_vrg(C["grid_h"], C["grid_w"], stats=None, device=device)
    assert_grid_shape(grid_ref, C["grid_h"], C["grid_w"])
    model = build_model(cfg).to(device)
    n_real = count_real(model)
    if n_real != cfg["expected_params"]:
        sys.exit(f"trainable real parameters {n_real} != registered {cfg['expected_params']}")
    trainer = OS.MaskedGradTrainer(model, lr=C["lr"], grad_weight=C["grad_weight"])
    scheduler = torch.optim.lr_scheduler.StepLR(trainer.opt, step_size=C["step_size"], gamma=C["gamma"])
    info = {"run_id": cfg["run_id"], "arm": cfg["arm"], "seed": C["seed"], "model": cfg["model"], "params_trainable_real": n_real,
            "test_cases_blocked": n_blocked, "n_train": len(train_ds), "n_val": len(val_ds), "epochs": C["epochs"],
            "rng_state_sha256_after_model": B.sha256_bytes(torch.get_rng_state().numpy().tobytes())}
    if a.dry_run:
        print(json.dumps(dict(info, dry_run=True), indent=1))
        return
    os.makedirs(run_dir, exist_ok=True)
    with B.Lock(os.path.join(run_dir, "RUN.lock"), a.break_stale_lock):
        code = _locked(a, cfg, rec, rec_sha, run_dir, model, trainer, scheduler, train_loader, val_loader, grid_ref, device,
                       filenames, info)
    sys.exit(code)


def _locked(a, cfg, rec, rec_sha, run_dir, model, trainer, scheduler, train_loader, val_loader, grid_ref, device, filenames, info):
    C = cfg["C"]
    meta_p, last_p, best_p = (os.path.join(run_dir, n) for n in ("run_meta.json", "last.pt", "pcwno_best.pt"))
    metrics_p, ev_p = os.path.join(run_dir, "metrics.csv"), os.path.join(run_dir, "events.jsonl")
    if os.path.exists(meta_p):
        old = B.read_json(meta_p)
        if old["rec_sha"] != rec_sha:
            raise SystemExit(f"STOP {cfg['run_id']}: stored provenance {old['rec_sha'][:16]} != current {rec_sha[:16]}; refusing to mix")
    else:
        stray = [f for f in os.listdir(run_dir) if f != "RUN.lock"]
        if stray:
            raise SystemExit(f"STOP {cfg['run_id']}: non-empty run dir without run_meta.json: {stray}")
        import pytorch_wavelets
        B.atomic_json(dict(info, rec_sha=rec_sha, provenance=rec, torch=torch.__version__, cuda=torch.version.cuda,
                           cudnn=torch.backends.cudnn.version(), pytorch_wavelets=pytorch_wavelets.__version__,
                           python=platform.python_version(), host=platform.node(), device=str(device),
                           gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None, numerics=T.numerics(),
                           test_cases_loaded_as_data=False, test_policy="test cases NaN / mask 0 after loading; never evaluated"),
                      meta_p)
    start, best_val, best_epoch, best_state = 1, float("inf"), 0, None
    resumed = os.path.exists(last_p)
    if resumed:
        ck = torch.load(last_p, map_location="cpu", weights_only=False)
        if ck["rec_sha"] != rec_sha:
            raise SystemExit("STOP last.pt provenance mismatch")
        prime_buffers(model, ck["model"])
        model.load_state_dict(ck["model"], strict=True)
        trainer.opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        best_val, best_epoch, best_state = ck["best_val"], ck["best_epoch"], ck["best_state"]
        set_rng_states(ck["rng"])
        start = ck["epoch"] + 1
        rows = list(csv.DictReader(open(metrics_p))) if os.path.exists(metrics_p) else []
        rows = [r for r in rows if int(r["epoch"]) <= ck["epoch"]]
        tmp = metrics_p + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_COLS)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, metrics_p)
        if best_state is not None:
            atomic_torch_save(best_state, best_p)
    elif not os.path.exists(metrics_p):
        with open(metrics_p, "w", newline="") as f:
            csv.writer(f).writerow(METRIC_COLS)
    B.append_jsonl(ev_p, kind="resume" if resumed else "start", epoch=start)
    B.write_status(run_dir, "RUNNING", epoch=start)
    print(f"[run] {cfg['run_id']} start_epoch={start} resumed={resumed} params={info['params_trainable_real']}", flush=True)
    epochs = C["epochs"]
    t_job = time.time()
    try:
        for ep in range(start, epochs + 1):
            t0 = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
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
                    for b in range(bs):
                        n = int(batch["n_valid"][b])
                        sse.append(((pred[b, :n, :] - y[b, :n, :]) ** 2).sum(dim=0).double().cpu().numpy()); nn_.append(n)
            val_total /= n_val; val_data /= n_val; val_grad /= n_val
            if not np.isfinite(val_total) or not np.isfinite(train_total):
                B.append_jsonl(ev_p, kind="diverged", epoch=ep)
                B.write_status(run_dir, "DIVERGED", epoch=ep)
                print(f"non-finite loss at epoch {ep}: DIVERGED (terminal)", flush=True)
                return EXIT_TERMINAL
            vr = np.sqrt(np.array(sse) / np.array(nn_)[:, None]).mean(axis=0)
            is_best = val_total < best_val
            if is_best:
                best_val, best_epoch = val_total, ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                atomic_torch_save(best_state, os.path.join(run_dir, "pcwno_best.pt"))
            scheduler.step()
            for group in trainer.opt.param_groups:
                if group["lr"] < C["min_lr"]:
                    group["lr"] = C["min_lr"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else -1.0
            esec = time.perf_counter() - t0
            with open(os.path.join(run_dir, "metrics.csv"), "a", newline="") as f:
                csv.writer(f).writerow([ep, f"{train_total:.6e}", f"{train_data:.6e}", f"{train_grad:.6e}", f"{val_total:.6e}",
                                        f"{val_data:.6e}", f"{val_grad:.6e}", f"{lr_used:.6e}", f"{esec:.3f}", int(is_best),
                                        f"{peak:.1f}"] + [f"{v:.8e}" for v in vr])
            atomic_torch_save({"model": model.state_dict(), "opt": trainer.opt.state_dict(), "sched": scheduler.state_dict(),
                               "epoch": ep, "best_val": best_val, "best_epoch": best_epoch, "best_state": best_state,
                               "rng": rng_states(), "rec_sha": rec_sha}, os.path.join(run_dir, "last.pt"))
            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(f"[Epoch {ep}] train {train_total:.6e} val {val_total:.6e} best {best_val:.6e}@{best_epoch} "
                      f"lr {lr_used:.2e} {esec:.1f}s peak {peak:.0f}MiB", flush=True)
            if a.stop_after_epoch is not None and ep >= a.stop_after_epoch and ep < epochs:
                raise Stop(f"smoke resume test: stop after epoch {ep}")
            if a.budget_s is not None and ep < epochs and (time.time() - t_job) + 1.5 * esec > a.budget_s:
                raise Stop("budget")
    except Stop as e:
        B.append_jsonl(ev_p, kind="interrupted", reason=str(e))
        B.write_status(run_dir, "INTERRUPTED", reason=str(e))
        print(f"INTERRUPTED: {e} (resubmit resumes from last.pt)", flush=True)
        return EXIT_INTERRUPTED
    except torch.cuda.OutOfMemoryError:
        B.append_jsonl(ev_p, kind="oom", trace=traceback.format_exc()[-2000:])
        B.write_status(run_dir, "ERROR", reason="OOM")
        return EXIT_TERMINAL
    except Exception:  # noqa: BLE001
        B.append_jsonl(ev_p, kind="error", trace=traceback.format_exc()[-4000:])
        B.write_status(run_dir, "ERROR", reason="exception")
        traceback.print_exc()
        return EXIT_TERMINAL
    return _finish(cfg, run_dir, rec_sha, model, val_loader, grid_ref, device, filenames, best_val, best_epoch)


def _finish(cfg, run_dir, rec_sha, model, val_loader, grid_ref, device, filenames, best_val, best_epoch):
    C = cfg["C"]
    torch.save(model.state_dict(), os.path.join(run_dir, "last_model.pt"))
    rows_m = list(csv.DictReader(open(os.path.join(run_dir, "metrics.csv"))))
    assert [int(r["epoch"]) for r in rows_m] == list(range(1, C["epochs"] + 1)), "metrics.csv epochs incomplete"
    model.load_state_dict(torch.load(os.path.join(run_dir, "pcwno_best.pt"), map_location=device))
    rows = T.val_case_rmse(model, val_loader, grid_ref, device, filenames)
    with open(os.path.join(run_dir, "val_case_RMSE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    es = np.array([float(r["epoch_sec"]) for r in rows_m])
    summ = {"run_id": cfg["run_id"], "best_epoch": best_epoch, "best_val_loss": best_val, "epochs_completed": C["epochs"],
            "train_s_total": float(es.sum()), "train_s_per_epoch_median": float(np.median(es)),
            "peak_alloc_mib_max": max(float(r["peak_alloc_mib"]) for r in rows_m),
            "val_case_mean_NRMSE": {t: float(np.mean([r[f"NRMSE_{t}"] for r in rows])) for t in T.TARGETS},
            "n_val_cases": len(rows), "test_evaluated": False, "rec_sha": rec_sha,
            "best_sha256": B.sha256_file(os.path.join(run_dir, "pcwno_best.pt"))}
    B.atomic_json(summ, os.path.join(run_dir, "val_summary.json"))
    names = ["pcwno_best.pt", "last_model.pt", "run_meta.json", "metrics.csv", "val_case_RMSE.csv", "val_summary.json"]
    with open(os.path.join(run_dir, "SHA256SUMS_run.txt"), "w") as f:
        for n in names:
            f.write(f"{B.sha256_file(os.path.join(run_dir, n))}  {n}\n")
    B.atomic_json(dict(summ, finished=time.strftime("%Y-%m-%dT%H:%M:%S")), os.path.join(run_dir, "complete.json"))
    B.append_jsonl(os.path.join(run_dir, "events.jsonl"), kind="done", best_epoch=best_epoch)
    B.write_status(run_dir, "DONE", best_epoch=best_epoch)
    print("TRAINING=DONE", json.dumps({k: summ[k] for k in ("run_id", "best_epoch", "best_val_loss", "train_s_per_epoch_median")}),
          flush=True)
    return EXIT_OK


if __name__ == "__main__":
    main()
