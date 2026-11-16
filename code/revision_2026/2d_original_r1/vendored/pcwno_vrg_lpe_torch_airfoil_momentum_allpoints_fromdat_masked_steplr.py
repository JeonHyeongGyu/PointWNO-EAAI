import os
import csv
import argparse
import random
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pcwno_vrg_lpe_torch_airfoil_momentum_fno_same import (
    PCWNO,
    make_internal_vrg,
    assert_grid_shape,
)


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_split(n_cases: int):
    rng = np.random.default_rng(0)

    forced_test = np.array([60, 118], dtype=int)
    forced_test = forced_test[(forced_test >= 0) & (forced_test < n_cases)]

    all_idx = np.arange(n_cases)
    remain = np.setdiff1d(all_idx, forced_test, assume_unique=False)

    n_tr = int(0.8 * n_cases)
    n_va = int(0.1 * n_cases)
    n_te = n_cases - n_tr - n_va

    if forced_test.size > n_te:
        n_te = forced_test.size
        n_tr = max(0, n_cases - n_va - n_te)

    perm = rng.permutation(remain)
    i_tr = perm[:n_tr]
    i_va = perm[n_tr:n_tr + n_va]
    i_te_extra = perm[n_tr + n_va:n_tr + n_va + (n_te - forced_test.size)]
    i_te = np.concatenate([forced_test, i_te_extra], axis=0)

    return i_tr, i_va, i_te


class AllPointsMaskedDataset(Dataset):
    def __init__(self, case_ids, inputs, targets, masks, knn_idx, filenames, n_points):
        self.case_ids = np.asarray(case_ids, dtype=np.int64)
        self.inputs = inputs
        self.targets = targets
        self.masks = masks
        self.knn_idx = knn_idx
        self.filenames = filenames
        self.n_points = n_points

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        gid = int(self.case_ids[idx])
        return {
            "xyz": torch.from_numpy(self.inputs[gid]).float(),
            "y": torch.from_numpy(self.targets[gid]).float(),
            "mask": torch.from_numpy(self.masks[gid]).float(),
            "knn_idx": torch.from_numpy(self.knn_idx[gid]).long(),
            "case_id": torch.tensor(gid, dtype=torch.long),
            "n_valid": torch.tensor(int(self.n_points[gid]), dtype=torch.long),
        }


def load_allpoints_npz(path, knn_k=6):
    src = np.load(path, allow_pickle=True)

    data_obj = src["data"]
    filenames = src["filenames"]
    n_points = src["n_points"]
    columns = list(src["columns"])

    expected = ["x", "y", "x_momentum", "y_momentum", "pressure_coefficient"]
    if columns != expected:
        raise RuntimeError(f"Unexpected columns: {columns}")

    n_cases = len(data_obj)
    max_points = int(np.max(n_points))

    i_tr, i_va, i_te = make_split(n_cases)

    all_train = np.concatenate([np.asarray(data_obj[i], dtype=np.float32) for i in i_tr], axis=0)

    x_min, x_max = float(all_train[:, 0].min()), float(all_train[:, 0].max())
    y_min, y_max = float(all_train[:, 1].min()), float(all_train[:, 1].max())
    u_min, u_max = float(all_train[:, 2].min()), float(all_train[:, 2].max())
    v_min, v_max = float(all_train[:, 3].min()), float(all_train[:, 3].max())
    p_min, p_max = float(all_train[:, 4].min()), float(all_train[:, 4].max())

    def mm(a, amin, amax):
        return (a - amin) / (amax - amin + 1e-12)

    inputs = []
    targets = []
    masks = []
    knn_indices = []

    print("Using allpoints npz:", path)
    print("columns =", columns)
    print("n_cases =", n_cases)
    print("n_points min/max/mean =", int(np.min(n_points)), int(np.max(n_points)), float(np.mean(n_points)))
    print("max_points for padding =", max_points)
    print("train_idx =", i_tr)
    print("val_idx   =", i_va)
    print("test_idx  =", i_te)
    print("60 in test_idx =", 60 in i_te)
    print("118 in test_idx =", 118 in i_te)
    print("Normalization ranges from train split:")
    print("x:", x_min, x_max)
    print("y:", y_min, y_max)
    print("x_momentum:", u_min, u_max)
    print("y_momentum:", v_min, v_max)
    print("pressure_coefficient:", p_min, p_max)

    for gid in range(n_cases):
        d = np.asarray(data_obj[gid], dtype=np.float32)
        n = d.shape[0]

        xyz = np.zeros((max_points, 3), dtype=np.float32)
        yout = np.zeros((max_points, 3), dtype=np.float32)
        mask = np.zeros((max_points,), dtype=np.float32)
        kidx = np.zeros((max_points, knn_k), dtype=np.int64)

        x_n = mm(d[:, 0], x_min, x_max)
        y_n = mm(d[:, 1], y_min, y_max)
        u_n = mm(d[:, 2], u_min, u_max)
        v_n = mm(d[:, 3], v_min, v_max)
        p_n = mm(d[:, 4], p_min, p_max)

        xyz[:n, 0] = x_n
        xyz[:n, 1] = y_n
        xyz[:n, 2] = 0.0

        yout[:n, 0] = u_n
        yout[:n, 1] = v_n
        yout[:n, 2] = p_n

        mask[:n] = 1.0

        # padded points are far outside [0,1] so they are not selected by local queries
        xyz[n:, 0] = 10.0
        xyz[n:, 1] = 10.0
        xyz[n:, 2] = 0.0

        tree = cKDTree(xyz[:n, :2])
        _, qidx = tree.query(xyz[:n, :2], k=knn_k + 1)
        if qidx.ndim == 1:
            qidx = qidx[:, None]
        qidx = qidx[:, 1:knn_k + 1]
        kidx[:n, :] = qidx.astype(np.int64)

        inputs.append(xyz)
        targets.append(yout)
        masks.append(mask)
        knn_indices.append(kidx)

        if gid in [0, 60, 118, 119]:
            print(f"case {gid}: original={d.shape}, padded xyz={xyz.shape}, target={yout.shape}")

    stats = dict(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        u_min=u_min, u_max=u_max,
        v_min=v_min, v_max=v_max,
        p_min=p_min, p_max=p_max,
    )

    return inputs, targets, masks, knn_indices, filenames, n_points, i_tr, i_va, i_te, stats, max_points


def masked_mse_loss(pred, target, mask):
    """
    Eq. (7)-style masked MSE.

    pred/target: (B, Nmax, C)
    mask:        (B, Nmax)

    For each sample j:
        loss_j = sum_i mask_ji * ||pred_ji - target_ji||^2 / (N_j * C)

    Then:
        loss = mean_j loss_j

    This prevents cases with more original points from dominating the batch loss.
    """
    err2 = (pred - target) ** 2
    w = mask.unsqueeze(-1)
    C = pred.shape[-1]

    per_case_sum = (err2 * w).sum(dim=(1, 2))
    n_valid = mask.sum(dim=1).clamp_min(1.0)

    per_case_mse = per_case_sum / (n_valid * C)
    return per_case_mse.mean()


def masked_knn_gradient_loss(xyz, pred, target, mask, knn_idx):
    """
    Coordinate-based local finite-difference gradient matching.

    xyz:     (B,N,3)
    pred:    (B,N,3)
    target:  (B,N,3)
    mask:    (B,N)
    knn_idx: (B,N,K)
    """
    B, N, C = pred.shape
    K = knn_idx.shape[-1]

    gather_idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, C)

    pred_nb = torch.gather(
        pred.unsqueeze(1).expand(B, N, N, C),
        dim=2,
        index=gather_idx,
    )

    targ_nb = torch.gather(
        target.unsqueeze(1).expand(B, N, N, C),
        dim=2,
        index=gather_idx,
    )

    xyz_nb = torch.gather(
        xyz[:, :, :2].unsqueeze(1).expand(B, N, N, 2),
        dim=2,
        index=knn_idx.unsqueeze(-1).expand(-1, -1, -1, 2),
    )

    center_pred = pred.unsqueeze(2)
    center_targ = target.unsqueeze(2)
    center_xyz = xyz[:, :, :2].unsqueeze(2)

    dist = torch.linalg.norm(center_xyz - xyz_nb, dim=-1, keepdim=True).clamp_min(1e-6)

    grad_pred = (center_pred - pred_nb) / dist
    grad_targ = (center_targ - targ_nb) / dist

    err2 = (grad_pred - grad_targ) ** 2

    valid = mask.unsqueeze(-1).unsqueeze(-1)
    denom = valid.sum() * K * C + 1e-12

    return (err2 * valid).sum() / denom


class MaskedGradTrainer:
    def __init__(self, model, lr=1e-3, grad_weight=1e-3):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.grad_weight = grad_weight

    def step(self, xyz, grid_ref, y, mask, knn_idx):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        pred = self.model(xyz, grid_ref)

        data_loss = masked_mse_loss(pred, y, mask)

        if self.grad_weight > 0.0:
            grad_loss = masked_knn_gradient_loss(xyz, pred, y, mask, knn_idx)
            loss = data_loss + self.grad_weight * grad_loss
        else:
            grad_loss = torch.zeros((), device=xyz.device, dtype=data_loss.dtype)
            loss = data_loss

        loss.backward()
        self.opt.step()

        return float(loss.detach().cpu()), float(data_loss.detach().cpu()), float(grad_loss.detach().cpu())

    @torch.no_grad()
    def eval(self, xyz, grid_ref, y, mask, knn_idx):
        self.model.eval()
        pred = self.model(xyz, grid_ref)

        data_loss = masked_mse_loss(pred, y, mask)

        if self.grad_weight > 0.0:
            grad_loss = masked_knn_gradient_loss(xyz, pred, y, mask, knn_idx)
            loss = data_loss + self.grad_weight * grad_loss
        else:
            grad_loss = torch.zeros((), device=xyz.device, dtype=data_loss.dtype)
            loss = data_loss

        return float(loss.detach().cpu()), float(data_loss.detach().cpu()), float(grad_loss.detach().cpu()), pred


def denorm_y(y, stats):
    out = y.copy()
    out[:, 0] = out[:, 0] * (stats["u_max"] - stats["u_min"]) + stats["u_min"]
    out[:, 1] = out[:, 1] * (stats["v_max"] - stats["v_min"]) + stats["v_min"]
    out[:, 2] = out[:, 2] * (stats["p_max"] - stats["p_min"]) + stats["p_min"]
    return out


def denorm_xyz(xyz, stats):
    out = xyz.copy()
    out[:, 0] = out[:, 0] * (stats["x_max"] - stats["x_min"]) + stats["x_min"]
    out[:, 1] = out[:, 1] * (stats["y_max"] - stats["y_min"]) + stats["y_min"]
    return out


def save_case_dat(save_dir, case_id, filename, xyz, pred, target, stats):
    out_dir = Path(save_dir) / "dat"
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_phys = denorm_xyz(xyz, stats)
    pred_phys = denorm_y(pred, stats)
    targ_phys = denorm_y(target, stats)
    err = np.abs(pred_phys - targ_phys)

    path = out_dir / f"global{case_id:03d}_pcwno_allpoints_masked_prediction.dat"

    with open(path, "w") as f:
        f.write('TITLE = "PCWNO allpoints masked prediction"\n')
        f.write(
            'VARIABLES = "X", "Y", '
            '"x_momentum_true", "y_momentum_true", "pressure_coefficient_true", '
            '"x_momentum_pred", "y_momentum_pred", "pressure_coefficient_pred", '
            '"x_momentum_abs_error", "y_momentum_abs_error", "pressure_coefficient_abs_error"\n'
        )
        f.write(f'ZONE T="{filename}", N={xyz.shape[0]}, F=POINT\n')

        for i in range(xyz.shape[0]):
            f.write(
                f"{xyz_phys[i,0]:.9e} {xyz_phys[i,1]:.9e} "
                f"{targ_phys[i,0]:.9e} {targ_phys[i,1]:.9e} {targ_phys[i,2]:.9e} "
                f"{pred_phys[i,0]:.9e} {pred_phys[i,1]:.9e} {pred_phys[i,2]:.9e} "
                f"{err[i,0]:.9e} {err[i,1]:.9e} {err[i,2]:.9e}\n"
            )

    return str(path)


@torch.no_grad()
def evaluate_and_save(model, loader, grid_ref, stats, save_dir, device, filenames):
    model.eval()

    rows = []
    sum_rmse = np.zeros(3, dtype=np.float64)
    count = 0

    for batch in loader:
        xyz = batch["xyz"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        case_ids = batch["case_id"].cpu().numpy()
        n_valids = batch["n_valid"].cpu().numpy()

        pred = model(xyz, grid_ref)

        err = pred - y
        for b in range(xyz.shape[0]):
            n = int(n_valids[b])
            rmse = torch.sqrt((err[b, :n, :] ** 2).mean(dim=0)).cpu().numpy()

            sum_rmse += rmse
            count += 1

            gid = int(case_ids[b])
            row = {
                "global_data_id": gid,
                "filename": str(filenames[gid]),
                "n_points": n,
                "NRMSE_x_momentum": float(rmse[0]),
                "NRMSE_y_momentum": float(rmse[1]),
                "NRMSE_pressure_coefficient": float(rmse[2]),
            }
            rows.append(row)

            if gid in [60, 118]:
                path = save_case_dat(
                    save_dir,
                    gid,
                    str(filenames[gid]),
                    xyz[b, :n, :].cpu().numpy(),
                    pred[b, :n, :].cpu().numpy(),
                    y[b, :n, :].cpu().numpy(),
                    stats,
                )
                print("[OK] saved target dat:", path)

    avg = sum_rmse / max(count, 1)

    csv_path = Path(save_dir) / "test_case_RMSE.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("[Test] Average normalized RMSE (x_momentum, y_momentum, Cp):", avg[0], avg[1], avg[2])
    print("[OK] saved:", csv_path)

    for row in rows:
        if row["global_data_id"] in [60, 118]:
            print(row)

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allpoints_npz", default="./Data_allpoints_xymomcp_from_dat.npz")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--save_dir", default="./logs/pcwno_momentum_allpoints_fromdat_masked_128x96_db2_level1_ep500")
    parser.add_argument("--grid_h", type=int, default=128)
    parser.add_argument("--grid_w", type=int, default=96)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wno_level", type=int, default=2)
    parser.add_argument("--wno_width", type=int, default=128)
    parser.add_argument("--wno_layers", type=int, default=6)
    parser.add_argument("--grid_range", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--grad_weight", type=float, default=0.0)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--knn_k", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "dat"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "figs"), exist_ok=True)

    (
        inputs, targets, masks, knn_idx, filenames, n_points,
        train_idx, val_idx, test_idx, stats, max_points
    ) = load_allpoints_npz(args.allpoints_npz, knn_k=args.knn_k)

    train_ds = AllPointsMaskedDataset(train_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    val_ds = AllPointsMaskedDataset(val_idx, inputs, targets, masks, knn_idx, filenames, n_points)
    test_ds = AllPointsMaskedDataset(test_idx, inputs, targets, masks, knn_idx, filenames, n_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    device = torch.device(args.device)
    grid_ref = make_internal_vrg(args.grid_h, args.grid_w, stats=None, device=device)
    assert_grid_shape(grid_ref, args.grid_h, args.grid_w)

    model = PCWNO(
        grid_hw=(args.grid_h, args.grid_w),
        wno_width=args.wno_width,
        wno_level=args.wno_level,
        wno_layers=args.wno_layers,
        wno_wavelet=args.wavelet,
        wno_grid_range=tuple(args.grid_range),
    ).to(device)

    trainer = MaskedGradTrainer(model, lr=args.lr, grad_weight=args.grad_weight)
    scheduler = torch.optim.lr_scheduler.StepLR(
        trainer.opt,
        step_size=args.step_size,
        gamma=args.gamma
    )

    print("Configuration:")
    print("  allpoints_npz =", args.allpoints_npz)
    print("  save_dir      =", args.save_dir)
    print("  max_points    =", max_points)
    print("  batch         =", args.batch)
    print("  wavelet       =", args.wavelet)
    print("  wno_level     =", args.wno_level)
    print("  grad_weight   =", args.grad_weight)
    print("  step_size     =", args.step_size)
    print("  gamma         =", args.gamma)
    print("  min_lr        =", args.min_lr)
    print("  knn_k         =", args.knn_k)
    print("  device        =", device)

    metrics_csv = os.path.join(args.save_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_total", "train_data", "train_grad", "val_total", "val_data", "val_grad"])

    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        train_total = train_data = train_grad = 0.0
        n_train = 0

        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {ep}/{args.epochs} [train]", ncols=120, leave=False)
        for batch in pbar:
            xyz = batch["xyz"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            kidx = batch["knn_idx"].to(device)

            total, dloss, gloss = trainer.step(xyz, grid_ref, y, mask, kidx)

            bs = xyz.size(0)
            train_total += total * bs
            train_data += dloss * bs
            train_grad += gloss * bs
            n_train += bs

            pbar.set_postfix(total=f"{total:.3e}", data=f"{dloss:.3e}", grad=f"{gloss:.3e}")

        train_total /= n_train
        train_data /= n_train
        train_grad /= n_train

        val_total = val_data = val_grad = 0.0
        n_val = 0

        with torch.no_grad():
            pbar_v = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {ep}/{args.epochs} [val]", ncols=120, leave=False)
            for batch in pbar_v:
                xyz = batch["xyz"].to(device)
                y = batch["y"].to(device)
                mask = batch["mask"].to(device)
                kidx = batch["knn_idx"].to(device)

                total, dloss, gloss, _ = trainer.eval(xyz, grid_ref, y, mask, kidx)

                bs = xyz.size(0)
                val_total += total * bs
                val_data += dloss * bs
                val_grad += gloss * bs
                n_val += bs

                pbar_v.set_postfix(total=f"{total:.3e}", data=f"{dloss:.3e}", grad=f"{gloss:.3e}")

        val_total /= n_val
        val_data /= n_val
        val_grad /= n_val

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep,
                f"{train_total:.6e}",
                f"{train_data:.6e}",
                f"{train_grad:.6e}",
                f"{val_total:.6e}",
                f"{val_data:.6e}",
                f"{val_grad:.6e}",
            ])

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), os.path.join(args.save_dir, "pcwno.pt"))

        scheduler.step()

        # Clamp learning rate to min_lr
        for group in trainer.opt.param_groups:
            if group["lr"] < args.min_lr:
                group["lr"] = args.min_lr

        if ep == 1 or ep % args.step_size == 0 or ep == args.epochs:
            print("[LR]", trainer.opt.param_groups[0]["lr"])

        if ep == 1 or ep % 10 == 0:
            print(
                f"[Epoch {ep}] "
                f"train total/data/grad = {train_total:.6e}/{train_data:.6e}/{train_grad:.6e}, "
                f"val total/data/grad = {val_total:.6e}/{val_data:.6e}/{val_grad:.6e}, "
                f"best_val = {best_val:.6e}"
            )

    print("Best val loss:", best_val)

    model.load_state_dict(torch.load(os.path.join(args.save_dir, "pcwno.pt"), map_location=device))
    evaluate_and_save(model, test_loader, grid_ref, stats, args.save_dir, device, filenames)

    print("Saved to", args.save_dir)


if __name__ == "__main__":
    main()
