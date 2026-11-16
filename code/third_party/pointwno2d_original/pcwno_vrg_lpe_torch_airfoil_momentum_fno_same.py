#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file PC-WNO trainer (PyTorch)
- Maintains the original pipeline: VRG → LPE( SA-MSG ) → Grid-Net( WNO2d ) → FP( IDW ) → Point head
- Uses relative paths by default: ./Data.npy, ./circle_grid.dat
- Mirrors the role of train_MSG_CNN_CFDnet_orig2.py but runs in PyTorch and swaps U-Net→WNO.

Dependencies (kept external to avoid duplicating large code):
  * wno2d_airfoil_gpu.py   (must expose class WNO2d)
  * wavelet_convolution_v3.py (used by WNO2d)
Place these next to this file.

Data.npy layout (as in the TF script): columns [x, y, p, u, v].
Grid file (circle_grid.dat): whitespace-delimited columns x y (z optional).

Outputs:
  ./logs/pcwno_single/pcwno.pt               - model weights
  ./logs/pcwno_single/metrics.csv            - train/val loss history
  ./logs/pcwno_single/preds_caseXXXX.npy     - optional predictions on test set (denormalized)
"""
from __future__ import annotations
import os
import csv
import math
import time
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Import user's WNO2d (kept as dependency to avoid duplicating large code)
# -----------------------------------------------------------------------------
# Wavelet conv primitive used by WNO grid network
from wavelet_convolution_v3 import WaveConv2d


# -----------------------------
# WNO2d: 2D Wavelet Neural Operator (embedded)
# -----------------------------
class WNO2d(nn.Module):
    """2D WNO head that:
    - concatenates (x,y) grid coordinates internally,
    - applies stacked WaveConv2d blocks + 1x1 conv,
    - outputs 128-channel grid features; (u,v,p) are produced by PointHead after FP.


    Args:
        width: hidden width
        level: wavelet level for WaveConv2d
        layers: number of WaveConv2d+1x1 blocks
        size: [H, W] spatial size
        wavelet: wavelet family string (e.g., 'db2')
        in_channel: input feature channels (excluding the 2 grid channels)
        grid_range: (range_x, range_y) linspace extents
        omega: spectral/wavelet bandwidth control passed to WaveConv2d
        padding: optional right/bottom padding
    """
    def __init__(self, width, level, layers, size, wavelet,
                 in_channel, grid_range, omega=4, padding=0):
        super().__init__()
        self.register_buffer("fixed_grid_xyhw", None, persistent=True)
        self.level      = level
        self.width      = width
        self.layers     = layers
        self.size       = size
        self.wavelet    = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range
        self.omega      = omega
        self.padding    = padding

        # (in_channel + 2) because we append (x,y) grid in forward()
        self.fc0 = nn.Linear(in_channel + 2, width)
        self.conv = nn.ModuleList()
        self.w    = nn.ModuleList()
        for _ in range(layers):
            self.conv.append(WaveConv2d(width, width, level, size=size,
                                        wavelet=wavelet, omega=omega))
            self.w.append(nn.Conv2d(width, width, 1))
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 128)  # (UC, VC, PC)
        self.register_buffer("grid_perm_hw", torch.empty(0, dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,H,W,C_in) → (B,H,W,3)
        grid = self.get_grid(x.shape, x.device)   # (B,H,W,2)
        x = torch.cat((x, grid), dim=-1)          # (B,H,W,C_in+2)
        x = self.fc0(x)                           # (B,H,W,width)
        x = x.permute(0, 3, 1, 2)                 # (B,width,H,W)

        if self.padding:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for i, (conv, w) in enumerate(zip(self.conv, self.w)):
            x = conv(x) + w(x)
            if i != self.layers - 1:
                x = F.mish(x)
        if self.padding:
            x = x[..., :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 1)                 # (B,H,W,width)
        x = F.mish(self.fc1(x))                   # (B,H,W,128)
        return self.fc2(x)                        # (B,H,W,128)


    def get_grid(self, shape, device):
        B, H, W = shape[0], shape[1], shape[2]
        if self.fixed_grid_xyhw is not None:
            g = self.fixed_grid_xyhw.to(device)           # (H,W,2)
            return g.unsqueeze(0).expand(B, H, W, 2)      # (B,H,W,2)
        # fallback: 기존 linspace
        gx = torch.linspace(0, self.grid_range[0], W, device=device)
        gy = torch.linspace(0, self.grid_range[1], H, device=device)
        gx = gx.view(1,1,W,1).repeat(B,H,1,1)
        gy = gy.view(1,H,1,1).repeat(B,1,W,1)
        return torch.cat((gx, gy), dim=-1)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 distances between (B, M, C) and (B, N, C) → (B, M, N)."""
    a2 = (a ** 2).sum(-1, keepdim=True)        # (B, M, 1)
    b2 = (b ** 2).sum(-1).unsqueeze(-2)        # (B, 1, N)
    ab = torch.matmul(a, b.transpose(1, 2))    # (B, M, N)
    d2 = a2 + b2 - 2 * ab
    return torch.clamp(d2, min=0.0).sqrt()

def build_model(H: int, W: int, wavelet='db2', wno_width=128, wno_level=3, wno_layers=6):
    model = PCWNO(grid_hw=(H, W),
                  wno_width=wno_width, wno_level=wno_level, wno_layers=wno_layers,
                  wno_wavelet=wavelet, wno_grid_range=(1.0, 1.0)).to(args.device)
    return model

def train_one_epoch(trainer, loader, grid_ref, device):
    trainer.model.train()
    total = 0.0
    for xyz, y in loader:
        loss = trainer.step(xyz.to(device), grid_ref, y.to(device))
        total += loss * xyz.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def validate_one_epoch(trainer, loader, grid_ref, device):
    trainer.model.eval()
    total = 0.0
    for xyz, y in loader:
        l, _ = trainer.eval(xyz.to(device), grid_ref, y.to(device))
        total += l * xyz.size(0)
    return total / len(loader.dataset)

def _estimate_freestream(xy, uvp, flag_openboundary=None):
    """
    xy: (N,2) or (N,3), uvp: (N,3)=[u,v,p], flag_openboundary: (N,) in {0,1} or None
    returns U_inf, p_inf (floats)
    """
    import numpy as np
    if flag_openboundary is not None:
        mask = (flag_openboundary > 0.5)
        if mask.sum() >= max(8, int(0.01*mask.size)):
            u_ff = uvp[mask, 0]; v_ff = uvp[mask, 1]; p_ff = uvp[mask, 2]
        else:
            u_ff = uvp[:, 0]; v_ff = uvp[:, 1]; p_ff = uvp[:, 2]
    else:
        u_ff = uvp[:, 0]; v_ff = uvp[:, 1]; p_ff = uvp[:, 2]
    U_inf = float(np.mean((u_ff**2 + v_ff**2)**0.5))
    p_inf = float(np.median(p_ff))
    # 안전장치
    U_inf = 1.0 if U_inf <= 1e-8 else U_inf
    p_inf = 1.0 if abs(p_inf) <= 1e-8 else p_inf
    return U_inf, p_inf

# -----------------------------------------------------------------------------
# VRG: grid → nearest point mapping (vectorized)
# -----------------------------------------------------------------------------

def vrg_snap_grid_to_points(points_xyz: torch.Tensor,
                            grid_xyz_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Snap each grid node to its nearest point in the point cloud (per batch).
    Args:
        points_xyz  : (B, N, 3)
        grid_xyz_ref: (G, 3) reference grid (shared across batch) OR (B, G, 3)
    Returns:
        grid_xyz_snapped: (B, G, 3)  centroids (subset of points)
        nn_indices      : (B, G)    indices of nearest points per grid node
    """
    B, N, _ = points_xyz.shape
    if grid_xyz_ref.dim() == 2:  # (G, 3) → broadcast
        grid_xyz = grid_xyz_ref.unsqueeze(0).expand(B, -1, -1).to(points_xyz.device)
    else:
        grid_xyz = grid_xyz_ref.to(points_xyz.device)
    # distances: (B, G, N)
    d = pairwise_dist(grid_xyz, points_xyz)
    nn_idx = d.argmin(dim=-1)  # (B, G)
    batch = torch.arange(B, device=points_xyz.device).unsqueeze(-1)
    grid_xyz_snapped = points_xyz[batch, nn_idx, :]  # (B, G, 3)
    return grid_xyz_snapped, nn_idx

def grid_xy_hw_from_ref(grid_xyz_ref, grid_hw, center_xy=None):
    """
    반환: xy_hw (H,W,2), idx_hw (H,W)  ← idx_hw는 '원본 G 인덱스'의 (H,W) 배치 결과
    """
    H, W = grid_hw
    g = grid_xyz_ref[..., :2].detach().cpu().numpy()  # (G,2)
    assert g.shape[0] == H * W

    if center_xy is None:
        cx, cy = float(g[:,0].mean()), float(g[:,1].mean())
    else:
        cx, cy = center_xy

    r  = np.hypot(g[:,0]-cx, g[:,1]-cy)
    th = np.arctan2(g[:,1]-cy, g[:,0]-cx)

    order      = np.argsort(r)
    idx_sorted = order
    chunks     = np.array_split(idx_sorted, H)

    xy_hw  = np.zeros((H, W, 2), dtype=np.float32)
    idx_hw = np.zeros((H, W),    dtype=np.int64)

    for i, chunk in enumerate(chunks):
        th_chunk = th[chunk]
        sort_idx = np.argsort(th_chunk)
        ch = chunk[sort_idx]

        if len(ch) > W:   ch = ch[:W]
        elif len(ch) < W: ch = np.pad(ch, (0, W - len(ch)), mode='wrap')

        xy_hw[i, :, 0] = g[ch, 0]
        xy_hw[i, :, 1] = g[ch, 1]
        idx_hw[i, :]   = ch                    # ← 원본 인덱스의 (H,W) 배치 결과

    return torch.from_numpy(xy_hw), torch.from_numpy(idx_hw)

def downsample_grid_to_HW(grid_ref: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    grid_ref: (G,3), 원래 H0xW0 형태에서 균등 서브샘플링하여 (H,W)로 축소
    """
    G = int(grid_ref.shape[0])
    H0 = int(round((G)**0.5))
    W0 = int(G // H0)
    assert H0*W0 == G, f"Cannot infer H0,W0 from G={G}"
    g = grid_ref.view(H0, W0, -1).cpu().numpy()
    sh, sw = H0 // H, W0 // W
    assert sh >= 1 and sw >= 1 and H0 % H == 0 and W0 % W == 0, \
        f"({H0},{W0}) → ({H},{W}) downsample not integer."
    ds = g[::sh, ::sw, :]
    return torch.from_numpy(ds.reshape(H*W, -1)).to(grid_ref.dtype).to(grid_ref.device)

def make_internal_vrg(H: int, W: int, stats: dict | None, device: torch.device) -> torch.Tensor:
    """
    외부 파일 없이 내부에서 (H×W) 고정 격자(VRG)를 생성한다.
    - 좌표 범위는 [0,1] × [0,1] 로 정의한다.
    - 모델/데이터 정규화가 [0,1] 기준이므로 stats가 있어도 범위를 [0,1]로 둔다.
    반환: (H*W, 3) [x,y,z] with z=0
    """
    xv = torch.linspace(0.0, 1.0, W, device=device)
    yv = torch.linspace(0.0, 1.0, H, device=device)
    xx, yy = torch.meshgrid(xv, yv, indexing='xy')  # (W,H)
    grid = torch.stack([xx.T.reshape(-1), yy.T.reshape(-1),
                        torch.zeros(H*W, device=device)], dim=1)  # (H*W,3)
    return grid


# -----------------------------------------------------------------------------
# LPE: SA-MSG (multi-radius local patch embedding)
# -----------------------------------------------------------------------------

class SharedMLP(nn.Module):
    def __init__(self, channels: List[int], activation=nn.ReLU, bn: bool = False):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, G, K)
        return self.net(x)




class SA_MSG(nn.Module):
    """Simplified PointNet++ SA-MSG in PyTorch.
    - For each grid centroid, query neighbors from the original point cloud
      with multiple radii; apply per-scale Shared MLP to relative coords;
      aggregate via max-pooling across K.
    """
    def __init__(self,
                 npoint: int,
                 radius_list: List[float],
                 nsample_list: List[int],
                 mlp_list: List[List[int]],
                 in_ch: int = 6,  # using relative coords only by default
                 activation=nn.ReLU,
                 bn: bool = False):
        super().__init__()
        assert len(radius_list) == len(nsample_list) == len(mlp_list)
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.mlp = nn.ModuleList()
        for mlp in mlp_list:
            self.mlp.append(SharedMLP([in_ch] + mlp, activation=activation, bn=bn))

    @torch.no_grad()
    def _knn_ball(self, centroids: torch.Tensor, points: torch.Tensor,
                  radius: float, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d = pairwise_dist(centroids, points)  # (B,G,N)
        if radius > 0:
            mask = d <= radius
            d_masked = d.clone()
            d_masked[~mask] = float('inf')
        else:
            d_masked = d
        K = min(k, points.shape[1])
        dists, idx = torch.topk(d_masked, k=K, dim=-1, largest=False, sorted=False)
        no_neighbor = torch.isinf(dists[..., 0])
        if no_neighbor.any():
            dists_knn, idx_knn = torch.topk(d, k=K, dim=-1, largest=False, sorted=False)
            dists[no_neighbor] = dists_knn[no_neighbor]
            idx[no_neighbor] = idx_knn[no_neighbor]
        return idx, dists

    def forward(self, l0_xyz: torch.Tensor, grid_xyz: torch.Tensor,
                l0_points: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, G, _ = grid_xyz.shape
        feats = []
        for radius, nsample, mlp in zip(self.radius_list, self.nsample_list, self.mlp):
            idx, dists = self._knn_ball(grid_xyz, l0_xyz, radius=radius, k=nsample)  # (B,G,K)
            batch = torch.arange(B, device=l0_xyz.device).view(B, 1, 1)
            neigh = l0_xyz[batch, idx, :]  # (B,G,K,3)
            rel = (neigh - grid_xyz.unsqueeze(2))                          # (B,G,K,3)
            cent = grid_xyz.unsqueeze(2).expand(-1, -1, rel.shape[2], -1)  # (B,G,K,3)
            x = torch.cat([rel, cent], dim=-1)                             # (B,G,K,6)
            x = x.permute(0, 3, 1, 2).contiguous()                         # (B,6,G,K)
            x = mlp(x)  # (B, Cout, G, K)
            x = torch.amax(x, dim=-1)  # (B, Cout, G)
            feats.append(x)
        l1_points = torch.cat(feats, dim=1).transpose(1, 2).contiguous()  # (B,G,C)
        l1_xyz = grid_xyz
        
        if self.training and torch.rand(1).item() < 0.02:  # 가끔만
            print("STD l1_points:", float(l1_points.std()))
        
        return l1_xyz, l1_points

# -----------------------------------------------------------------------------
# FP: Feature Propagation (IDW, k=3)
# -----------------------------------------------------------------------------

class FP_IDW(nn.Module):
    def __init__(self, k: int = 3, p: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.k = k
        self.p = p
        self.eps = eps


    def forward(self, src_feat, src_xyz, dst_xyz):
        # src: (G,·) → dst: (N,·)
        d = pairwise_dist(dst_xyz, src_xyz)            # (B,N,G)
        K = min(self.k, src_xyz.shape[1])
        dists, idx = torch.topk(d, k=K, dim=-1, largest=False, sorted=False)  # (B,N,K)
        weights = 1.0 / (dists ** self.p + self.eps)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        B, N, _ = idx.shape
        batch = torch.arange(B, device=src_xyz.device).view(B,1,1)
        neigh = src_feat[batch, idx, :]                # (B,N,K,C)
        out = torch.einsum('bnk,bnkc->bnc', weights, neigh)  # (B,N,C)
        return out

# -----------------------------------------------------------------------------
# Point-wise head: shared 1x1 conv → (u,v,p)
# -----------------------------------------------------------------------------

class PointHead(nn.Module):
    def __init__(self, in_ch: int, hidden=(256, 128), out_ch: int = 3, p_drop: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, hidden[0], 1), nn.ReLU(),
            nn.Conv1d(hidden[0], hidden[1], 1), nn.ReLU(),
            nn.Conv1d(hidden[1], out_ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)

# -----------------------------------------------------------------------------
# Full model: SA-MSG → WNO → FP → Head
# -----------------------------------------------------------------------------

class PCWNO(nn.Module):
    def __init__(self,
                 grid_hw: Tuple[int, int] = (64, 64),
                 
                 sa_radius=(0.1, 0.2, 0.4),
                 sa_nsample=(16, 32, 64),
                 sa_mlps=((16, 16, 32), (32, 32, 64), (64, 64, 128)),
                 wno_width=128,
                 wno_level=3,
                 wno_layers=6,
                 wno_wavelet='db2',
                 wno_grid_range=(1.0, 1.0),
                 fp_k=3,
                 fp_p=2.0,
                 head_hidden=(128, 64),
                 bn=False):
        super().__init__()
        H, W = grid_hw
        self.grid_hw = grid_hw
        self.sa = SA_MSG(npoint=H * W,
                         radius_list=list(sa_radius),
                         nsample_list=list(sa_nsample),
                         mlp_list=[list(m) for m in sa_mlps],
                         in_ch=6, bn=bn)
        sa_out_ch = sum(m[-1] for m in sa_mlps)
        self.wno = WNO2d(width=wno_width, level=wno_level, layers=wno_layers,
                         size=[H, W], wavelet=wno_wavelet,
                         in_channel=sa_out_ch,
                         grid_range=list(wno_grid_range),
                         omega=4, padding=0)
        self.fp = FP_IDW(k=3, p=2.0)
        self.head = PointHead(in_ch=128, hidden=head_hidden, out_ch=3, p_drop=0.0)
        self.register_buffer("grid_perm_hw", torch.empty(0, dtype=torch.long), persistent=True)

    def forward(self, l0_xyz, grid_xyz_ref):
        B, N, _ = l0_xyz.shape
        # 0) 배치 브로드캐스트
        grid_xyz = (grid_xyz_ref.unsqueeze(0).expand(B, -1, -1) if grid_xyz_ref.dim()==2 else grid_xyz_ref).to(l0_xyz.device)

        # 1) SA-MSG: (B,G,3), (B,G,C)
        l1_xyz, l1_points = self.sa(l0_xyz=l0_xyz, grid_xyz=grid_xyz, l0_points=None)

        # 2) 고정 그리드/퍼뮤테이션 계산
        H, W = self.grid_hw
        device = l0_xyz.device
        grid_ref = grid_xyz_ref[0] if grid_xyz_ref.dim()==3 else grid_xyz_ref
        xy_hw, idx_hw = grid_xy_hw_from_ref(grid_ref, self.grid_hw, center_xy=None)

        # 1) 고정 그리드: 처음 한 번만 '대입' (재-register 금지)
        if self.wno.fixed_grid_xyhw is None:
            self.wno.fixed_grid_xyhw = xy_hw.to(device)

        # 2) 퍼뮤테이션: 이미 버퍼가 있으므로 모양만 맞추어 '값 갱신'
        perm_new = idx_hw.view(-1).to(torch.long).to(device)
        if self.grid_perm_hw.numel() != perm_new.numel():
            # 크기가 다르면 그냥 텐서를 새로 '대입' (버퍼 속성 유지됨)
            self.grid_perm_hw = perm_new
        else:
            # 크기가 같으면 in-place 복사
            self.grid_perm_hw.copy_(perm_new)

        perm = self.grid_perm_hw
        l1_xyz    = l1_xyz[:, perm, :]     # (B,G,3)
        l1_points = l1_points[:, perm, :]  # (B,G,C)

        # 3) (H,W,C)로 바꿔서 WNO 통과
        l1_grid   = l1_points.view(B, H, W, -1)
        grid_out  = self.wno(l1_grid)                # (B,H,W,128)
        grid_flat = grid_out.view(B, H*W, 128)       # (B,G,128)

        # 4) FP: 128짜리 grid 특징만 업샘플 (concat 제거)
        points_feat = self.fp(grid_flat, l1_xyz, l0_xyz)  # (B,N,128)  ← 아래 (D)에서 FP 시그니처 맞춰요
        pred = self.head(points_feat)                     # (B,N,3)

        return pred





# -----------------------------------------------------------------------------
# Dataset & normalization (mirrors TF script behaviour)
# -----------------------------------------------------------------------------

class PointDataset(Dataset):
    def __init__(self, xyz: np.ndarray, y: np.ndarray):
        self.xyz = torch.from_numpy(xyz).float()  # (M,N,3)
        self.y   = torch.from_numpy(y).float()    # (M,N,3)
    def __len__(self):
        return self.xyz.shape[0]
    def __getitem__(self, i):
        return self.xyz[i], self.y[i]


def load_dataset2(in_file, args):
    """Replicates the TF loader/normalizer in train_MSG_CNN_CFDnet_orig2.py.
    허용 레이아웃:
      - (M,N,5): [x,y,p,u,v]       ← 기존과 동일
      - (M,N,6): [x,y,z,p,u,v] 또는 [x,y,p,u,v,z]  ← z는 무시
    반환: (in_tr, in_va, in_te, out_tr, out_va, out_te, stats)
    """
    # 1) 데이터 로드
    if isinstance(in_file, np.ndarray):
        Data = in_file
    else:
        Data = np.load(in_file)  # (M, N, C)

    assert Data.ndim == 3 and Data.shape[2] in (5, 6), \
        f"Expected (M,N,5|6), got {Data.shape}"

    M, N, C = Data.shape

    # 2) 채널 매핑 결정
    # 기본 가정: 5채널은 [x,y,p,u,v]
    # 6채널은 보통 [x,y,z,p,u,v] 또는 [x,y,p,u,v,z]
    # → z 채널은 버리고 [x,y,p,u,v]만 사용
    if C == 5:
        # 5채널: [x,y,p,u,v]
        xi, yi, pi, ui, vi = 0, 1, 2, 3, 4
    else:  # C == 6
        # 우선 기존 z-채널 레이아웃들부터 검사
        z2_std = float(np.nanstd(Data[:, :, 2]))
        z5_std = float(np.nanstd(Data[:, :, 5]))
        z2_small = z2_std < 1e-12
        z5_small = z5_std < 1e-12

        if z2_small and not z5_small:
            # [x,y,z,p,u,v]
            xi, yi, pi, ui, vi = 0, 1, 3, 4, 5
            layout = "[x,y,z,p,u,v]"
        elif z5_small and not z2_small:
            # [x,y,p,u,v,z]
            xi, yi, pi, ui, vi = 0, 1, 2, 3, 4
            layout = "[x,y,p,u,v,z]"
        elif z2_small and z5_small:
            # 둘 다 거의 0 → [x,y,z,p,u,v]로 가정
            xi, yi, pi, ui, vi = 0, 1, 3, 4, 5
            layout = "[x,y,z,p,u,v] (both z≈0)"
        else:
            # ✅ 새 패턴: [x, y, rho, u, v, p/Cp] 로 해석 (rho는 버림)
            xi, yi, pi, ui, vi = 0, 1, 5, 3, 4
            layout = "[x,y,rho,u,v,p/Cp] (rho dropped)"

        print("Detected 6-channel layout:", layout)

        # --- 채널 매핑 확정 후 올바른 디버그 출력 ---
    try:
        print("DEBUG Data layout check:")
        print("x min/max =", float(Data[:, :, 0].min()), float(Data[:, :, 0].max()))
        print("y min/max =", float(Data[:, :, 1].min()), float(Data[:, :, 1].max()))
        # 매핑된 인덱스를 사용하여 정확한 라벨로 출력
        print("p(Cp) min/max =", float(Data[:, :, pi].min()), float(Data[:, :, pi].max()))
        print("u min/max =", float(Data[:, :, ui].min()), float(Data[:, :, ui].max()))
        print("v min/max =", float(Data[:, :, vi].min()), float(Data[:, :, vi].max()))
        print("DEBUG Data shape:", Data.shape, "min/max =",
              float(Data.min()), float(Data.max()))
    except Exception as e:
        print("DEBUG layout check failed:", repr(e))


    # 3) 포인트 수 강제 검증
    assert N == 15000, f"각 케이스 포인트 수 N={N}. 본 스크립트는 15000에 맞춰져 있습니다."

    # 4) 입력/출력 텐서 구성 (입력: xyz(단 z=0), 출력: (u,v,p))
    input_data  = np.zeros((M, N, 3), dtype=np.float32)
    output_data = np.zeros((M, N, 3), dtype=np.float32)

    input_data[:, :, 0]  = Data[:, :, xi]  # x
    input_data[:, :, 1]  = Data[:, :, yi]  # y
    input_data[:, :, 2]  = 0.0             # z는 강제로 0

    output_data[:, :, 0] = Data[:, :, ui]  # u
    output_data[:, :, 1] = Data[:, :, vi]  # v
    output_data[:, :, 2] = Data[:, :, pi]  # p

    if args.use_nondim:
        for i in range(M):
            uvp = output_data[i]       # (N,3) = [u,v,p]
            # 자유류 추정: flag가 없다면 전체에서 통계로 추정
            U_inf = float(args.uinf) if args.uinf is not None else float(np.mean(np.hypot(uvp[:,0], uvp[:,1])))
            p_inf = float(args.pinf) if args.pinf is not None else float(np.median(uvp[:,2]))
            U_inf = 1.0 if U_inf <= 1e-8 else U_inf
            p_inf = 1.0 if abs(p_inf) <= 1e-8 else p_inf

            # u, v는 자유류로 무차원화
            output_data[i,:,0] = uvp[:,0] / U_inf   # u/U_inf
            output_data[i,:,1] = uvp[:,1] / U_inf   # v/U_inf

            # p 채널 처리: Cp라면 그대로, 정적압이라면 p/p_inf
            if getattr(args, 'p_is_cp', False):
                output_data[i,:,2] = uvp[:,2]       # keep Cp as-is
            else:
                output_data[i,:,2] = uvp[:,2] / p_inf


    # 6) 8:1:1 분할 — FNO와 동일하게 NACA_304912(index=60), NACA_594009(index=118)를 test 고정
    rng = np.random.default_rng(0)

    # 강제 test 대상 (원본 케이스 인덱스)
    forced_test = np.array([60, 118], dtype=int)

    # 유효성 확인: 범위 밖 인덱스는 제거
    forced_test = forced_test[(forced_test >= 0) & (forced_test < M)]

    # 나머지 인덱스 집합
    all_idx = np.arange(M)
    remain = np.setdiff1d(all_idx, forced_test, assume_unique=False)

    # 목표 크기(8:1:1)
    n_tr = int(0.8 * M)
    n_va = int(0.1 * M)
    n_te = M - n_tr - n_va

    # 강제 대상이 test 용량보다 많으면 test 용량 확장, train을 줄여 총합 M 유지
    if forced_test.size > n_te:
        n_te = forced_test.size
        n_tr = max(0, M - n_va - n_te)

    # 남은 인덱스를 섞어서 채움
    perm = rng.permutation(remain)
    i_tr = perm[:n_tr]
    i_va = perm[n_tr:n_tr + n_va]
    i_te_extra = perm[n_tr + n_va : n_tr + n_va + (n_te - forced_test.size)]
    i_te = np.concatenate([forced_test, i_te_extra], axis=0)


    # 입력(X): x,y도 train-set 통계만 사용해 정규화
    x_min, x_max = float(input_data[i_tr,:,0].min()), float(input_data[i_tr,:,0].max())
    y_min, y_max = float(input_data[i_tr,:,1].min()), float(input_data[i_tr,:,1].max())
    u_min, u_max = float(output_data[i_tr,:,0].min()), float(output_data[i_tr,:,0].max())
    v_min, v_max = float(output_data[i_tr,:,1].min()), float(output_data[i_tr,:,1].max())
    p_min, p_max = float(output_data[i_tr,:,2].min()), float(output_data[i_tr,:,2].max())

    def _mm(a, b):
        a, b = float(a), float(b)
        return (a, b if b != a else a + 1e-6)

    x_min, x_max = _mm(x_min, x_max); y_min, y_max = _mm(y_min, y_max)
    u_min, u_max = _mm(u_min, u_max); v_min, v_max = _mm(v_min, v_max); p_min, p_max = _mm(p_min, p_max)

    # 전셋에 동일 통계 적용
    input_data[:,:,0] = (input_data[:,:,0] - x_min) / (x_max - x_min)
    input_data[:,:,1] = (input_data[:,:,1] - y_min) / (y_max - y_min)
    output_data[:,:,0] = (output_data[:,:,0] - u_min) / (u_max - u_min)
    output_data[:,:,1] = (output_data[:,:,1] - v_min) / (v_max - v_min)
    output_data[:,:,2] = (output_data[:,:,2] - p_min) / (p_max - p_min)

    stats = dict(u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max,
                 p_min=p_min, p_max=p_max, x_min=x_min, x_max=x_max,
                 y_min=y_min, y_max=y_max)
    
    in_tr, in_va, in_te    = input_data[i_tr],  input_data[i_va],  input_data[i_te]
    out_tr, out_va, out_te = output_data[i_tr], output_data[i_va], output_data[i_te]
    
    return (in_tr, in_va, in_te, out_tr, out_va, out_te, stats)


def load_dataset_auto(in_file, args):
    """
    in_file:
      - .npy: (M,N,5) = [x,y,p,u,v]
      - .npz:
          (A) key 'data' 에 (M,N,5) 배열 1개가 들어있는 packed 형식
              ├─ (A-1) data 가 곧 ndarray(M,N,5)
              ├─ (A-2) data 가 object 이고 내부가 ndarray(M,N,5)
              └─ (A-3) data 가 object 이고 내부가 dict{x,y,u,v,p}
          (B) key 들이 x,y,u,v,p 로 흩어져 있는 분해 형식
          (C) arr_0 한 덩어리 관용 형식
    반환은 load_dataset2()와 동일.
    """
    import os
    ext = os.path.splitext(in_file)[1].lower()
    if ext == '.npy':
        return load_dataset2(in_file, args)

    if ext == '.npz':
        z = np.load(in_file, allow_pickle=True)
        keys = set(z.files)

        # --- (A) packed: 'data' 1개 배열에 모두 들어있는 경우 ---
        if 'data' in keys and 'x' not in keys and 'y' not in keys:
            arr = z['data']

            # (A-1) data 가 곧 (M,N,5)
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 5:
                return load_dataset2(arr)

            # (A-2) data 가 object → 내부가 ndarray(M,N,5)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                try:
                    obj = arr.item()  # 0-d or 1-elem object array 가정
                except Exception:
                    # 혹시 (K,) object 배열이면 첫 원소만 시도
                    obj = arr[0]
                if isinstance(obj, np.ndarray) and obj.ndim == 3 and obj.shape[-1] == 5:
                    return load_dataset2(obj)
                # (A-3) data 가 object → 내부가 dict{x,y,u,v,p}
                if isinstance(obj, dict):
                    def grab(d, ks):
                        for k in ks:
                            if k in d: return d[k]
                        return None
                    Xs = grab(obj, ['x','xs','x_coord','x_coor'])
                    Ys = grab(obj, ['y','ys','y_coord','y_coor'])
                    Us = grab(obj, ['u','u_vel','u_velocity'])
                    Vs = grab(obj, ['v','v_vel','v_velocity'])
                    Ps = grab(obj, ['p','p_static','pressure'])
                    if all(v is not None for v in [Xs,Ys,Us,Vs,Ps]):
                        M = len(Xs)
                        Ns = [min(x.size, y.size, u.size, v.size, p.size)
                              for x,y,u,v,p in zip(Xs, Ys, Us, Vs, Ps)]
                        N = int(min(Ns))
                        Data = np.zeros((M, N, 5), dtype=np.float32)
                        for i in range(M):
                            xi = Xs[i].reshape(-1)[:N]
                            yi = Ys[i].reshape(-1)[:N]
                            ui = Us[i].reshape(-1)[:N]
                            vi = Vs[i].reshape(-1)[:N]
                            pi = Ps[i].reshape(-1)[:N]
                            Data[i,:,0] = xi
                            Data[i,:,1] = yi
                            Data[i,:,2] = pi
                            Data[i,:,3] = ui
                            Data[i,:,4] = vi
                        return load_dataset2(Data)
                raise ValueError("NPZ packed 'data'를 (M,N,5) 또는 {x,y,u,v,p}로 해석할 수 없습니다.")

        # --- (B) 분해: x,y,u,v,p가 각각 키로 있는 경우 ---
        def pick(dset_keys, cands):
            for k in cands:
                if k in dset_keys: return k
            for k in dset_keys:
                if k.lower() in cands: return k
            return None

        kx = pick(keys, ['x','xs','x_coord','x_coor'])
        ky = pick(keys, ['y','ys','y_coord','y_coor'])
        ku = pick(keys, ['u','u_vel','u_velocity'])
        kv = pick(keys, ['v','v_vel','v_velocity'])
        kp = pick(keys, ['p','p_static','pressure'])

        if all(k is not None for k in [kx,ky,ku,kv,kp]):
            Xs, Ys, Us, Vs, Ps = z[kx], z[ky], z[ku], z[kv], z[kp]
            M = len(Xs)
            Ns = [min(x.size, y.size, u.size, v.size, p.size)
                  for x,y,u,v,p in zip(Xs, Ys, Us, Vs, Ps)]
            N = int(min(Ns))
            Data = np.zeros((M, N, 5), dtype=np.float32)
            for i in range(M):
                xi = Xs[i].reshape(-1)[:N]
                yi = Ys[i].reshape(-1)[:N]
                ui = Us[i].reshape(-1)[:N]
                vi = Vs[i].reshape(-1)[:N]
                pi = Ps[i].reshape(-1)[:N]
                Data[i,:,0] = xi
                Data[i,:,1] = yi
                Data[i,:,2] = pi
                Data[i,:,3] = ui
                Data[i,:,4] = vi
            return load_dataset2(Data)

        # --- (C) 관용 키: arr_0 한 덩어리 ---
        if 'arr_0' in keys and len(keys) == 1:
            arr0 = z['arr_0']
            if isinstance(arr0, np.ndarray) and arr0.ndim == 3 and arr0.shape[-1] == 5:
                return load_dataset2(arr0)

        raise KeyError(f"Unsupported .npz keys {keys}. "
                       f"'data'=(M,N,5) 또는 x,y,u,v,p가 필요합니다.")

    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------------------------------------------------------
# Denormalization & saving helpers
# -----------------------------------------------------------------------------

def compute_errors_norm_and_phys(y_norm, yhat_norm, stats):
    """
    y_norm, yhat_norm: (N,3) normalized (u,v,p)
    stats: min/max를 담은 dict
    반환:
      - err_phys, abs_err_phys, ygt_phys, yhat_phys
      - err_norm, abs_err_norm
    """
    import numpy as np
    # 정규화 공간 에러
    err_norm = yhat_norm - y_norm
    abs_err_norm = np.abs(err_norm)
    # 물리(역정규화) 공간
    ygt_phys  = denorm_uvp_np(y_norm,  stats)
    yhat_phys = denorm_uvp_np(yhat_norm, stats)
    err_phys  = yhat_phys - ygt_phys
    abs_err_phys = np.abs(err_phys)
    return {
        "err_norm": err_norm, "abs_err_norm": abs_err_norm,
        "err_phys": err_phys, "abs_err_phys": abs_err_phys,
        "ygt_phys": ygt_phys, "yhat_phys": yhat_phys,
    }

def denorm_uvp_np(arr: np.ndarray, stats: dict) -> np.ndarray:
    """Denormalize UVP channels from [0,1] back to original scale."""
    out = arr.copy()
    out[..., 0] = out[..., 0] * (stats['u_max'] - stats['u_min']) + stats['u_min']
    out[..., 1] = out[..., 1] * (stats['v_max'] - stats['v_min']) + stats['v_min']
    out[..., 2] = out[..., 2] * (stats['p_max'] - stats['p_min']) + stats['p_min']
    return out

def denorm_xy_np(xyz_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Denormalize XY (Z stays zero). xyz_norm shape: (..., 3)."""
    out = xyz_norm.copy()
    out[..., 0] = out[..., 0] * (stats['x_max'] - stats['x_min']) + stats['x_min']
    out[..., 1] = out[..., 1] * (stats['y_max'] - stats['y_min']) + stats['y_min']
    return out

_CH = ['u','v','p']

def _scatter_save(x, y, c, title: str, path: str, vmin=None, vmax=None):
    plt.figure(figsize=(5,4), dpi=150)
    sc = plt.scatter(x, y, c=c, s=6, vmin=vmin, vmax=vmax)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(sc)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_case_outputs(save_dir: str, case_id: int,
                      xyz_norm: np.ndarray, yhat_norm: np.ndarray, y_norm: np.ndarray,
                      stats: dict):
    figs_dir = os.path.join(save_dir, 'figs')
    dat_dir  = os.path.join(save_dir, 'dat')
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(dat_dir,  exist_ok=True)

    # 좌표 역정규화
    xyz  = denorm_xy_np(xyz_norm,  stats)
    x, ycoord = xyz[:,0], xyz[:,1]

    # 에러(정규화/물리) 동시 산출
    errs = compute_errors_norm_and_phys(y_norm, yhat_norm, stats)
    ygt_phys   = errs["ygt_phys"]
    yhat_phys  = errs["yhat_phys"]
    err_phys   = errs["err_phys"]
    aerr_phys  = errs["abs_err_phys"]
    err_norm   = errs["err_norm"]
    aerr_norm  = errs["abs_err_norm"]

    # .dat 저장 (물리 + 정규화 에러 모두 포함)
    header = (
        "x y "
        "u_gt v_gt p_gt "
        "u_pred v_pred p_pred "
        "u_err v_err p_err "
        "u_abs_err v_abs_err p_abs_err "
        "u_err_norm v_err_norm p_err_norm "
        "u_abs_err_norm v_abs_err_norm p_abs_err_norm"
    )
    dat = np.column_stack([
        x, ycoord,
        ygt_phys[:,0],  ygt_phys[:,1],  ygt_phys[:,2],
        yhat_phys[:,0], yhat_phys[:,1], yhat_phys[:,2],
        err_phys[:,0],  err_phys[:,1],  err_phys[:,2],
        aerr_phys[:,0], aerr_phys[:,1], aerr_phys[:,2],
        err_norm[:,0],  err_norm[:,1],  err_norm[:,2],
        aerr_norm[:,0], aerr_norm[:,1], aerr_norm[:,2],
    ])
    np.savetxt(os.path.join(dat_dir, f'case{case_id:04d}.dat'), dat, fmt='%.8e', header=header)

    # 그림 저장: GT 범위로 컬러바 고정 + |error| 자동 스케일
    CH = ['u','v','p']
    for j, ch in enumerate(CH):
        gmin, gmax = float(ygt_phys[:, j].min()), float(ygt_phys[:, j].max())
        if gmax == gmin:
            pad = 1e-6; gmin -= pad; gmax += pad

        _scatter_save(x, ycoord, ygt_phys[:, j],
            f'{ch.upper()} ground truth (case {case_id:04d})',
            os.path.join(figs_dir, f'{ch}_gt_{case_id:04d}.png'),
            vmin=gmin, vmax=gmax)

        _scatter_save(x, ycoord, yhat_phys[:, j],
            f'{ch.upper()} prediction (case {case_id:04d})',
            os.path.join(figs_dir, f'{ch}_pred_{case_id:04d}.png'),
            vmin=gmin, vmax=gmax)

        _scatter_save(x, ycoord, err_phys[:, j],
            f'{ch.upper()} error (GT scale) (case {case_id:04d})',
            os.path.join(figs_dir, f'{ch}_err_gtrange_{case_id:04d}.png'),
            vmin=gmin, vmax=gmax)

        ae = aerr_phys[:, j]
        emax = float(ae.max()) if float(ae.max()) != 0.0 else 1e-12
        _scatter_save(x, ycoord, ae,
            f'|{ch.upper()} error| (auto scale) (case {case_id:04d})',
            os.path.join(figs_dir, f'{ch}_abs_err_{case_id:04d}.png'),
            vmin=0.0, vmax=emax)

def assert_grid_shape(grid_ref: torch.Tensor, H: int, W: int):
    """
    grid_ref: (H*W, 3) 또는 (B, H*W, 3)를 기대.
    """
    if grid_ref.dim() == 2:
        G = grid_ref.shape[0]
    elif grid_ref.dim() == 3:
        G = grid_ref.shape[1]
    else:
        raise AssertionError(f"grid_ref shape invalid: {tuple(grid_ref.shape)}")

    if G != H * W:
        raise AssertionError(f"Grid size mismatch: got G={G}, expected {H}*{W}={H*W}")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def local_variation_weight(xyz, y_true, K=8, alpha=1.5):
    d = pairwise_dist(xyz, xyz)
    K_eff = min(K, xyz.shape[1])
    _, idx = torch.topk(d, k=K_eff, dim=-1, largest=False, sorted=False)
    b = torch.arange(xyz.size(0), device=xyz.device).view(-1,1,1)
    p_nb = y_true[b, idx, 2:3]                # p 채널
    p_mu = p_nb.mean(dim=-2, keepdim=True)
    local_var = ((p_nb - p_mu)**2).mean(dim=-2)
    mean_var  = local_var.mean(dim=1, keepdim=True)
    w = 1.0 + alpha * (local_var / (mean_var + 1e-12))
    return torch.clamp(w, 1.0, 1.0 + 2.0*alpha)

def weighted_mixed_loss_stable(y_pred, y_true, xyz, eps=1e-2, K=8, alpha=1.5):
    with torch.no_grad():
        w = local_variation_weight(xyz, y_true, K=K, alpha=alpha)
    err = y_pred - y_true
    mse = (w * (err**2)).mean()
    denom = torch.clamp(y_true.abs(), min=eps)
    rel   = (w * (err / denom)**2).mean()
    return 0.5*mse + 0.5*rel

def loss_boundary_with_p_demean(y_pred, y_true, xyz,
                                w_ch=(1.0, 1.0, 4.0),
                                eps=1e-2, K=8, alpha=1.5):
    with torch.no_grad():
        w = local_variation_weight(xyz, y_true, K=K, alpha=alpha)
    if w.size(-1) == 1:
        w = w.repeat(1, 1, 3)
    w_ch_t = torch.tensor(w_ch, device=y_pred.device, dtype=y_pred.dtype).view(1,1,3)
    err = y_pred - y_true
    mse = (w * (err**2) * w_ch_t).mean()
    denom = torch.clamp(y_true.abs(), min=eps)
    rel  = (w * (err/denom)**2 * w_ch_t).mean()
    base = 0.5*mse + 0.5*rel
    p_pred = y_pred[..., 2:3]
    p_true = y_true[..., 2:3]
    p_bias = (p_pred - p_true).mean(dim=1, keepdim=True)
    p_rmse = (((p_pred - p_bias) - p_true)**2).mean()
    return base + 0.5*p_rmse

# ============================================================================ #

@torch.no_grad()
def _safe_count_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B, N) or (B, N, 1), {0,1}
    returns: (B, 1, 1), clamped >= 1
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    cnt = mask.sum(dim=(1, 2), keepdim=True)
    return torch.clamp(cnt, min=1.0)

def loss_masked_mse(y_pred: torch.Tensor,
                    y_true: torch.Tensor,
                    mask: torch.Tensor,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Masked MSE = (1/B) * Σ_j (1/(N_j*C)) * Σ_{i,c} [ (ŷ - y)^2 * mask ]
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    diff2  = (y_pred - y_true) ** 2
    masked = diff2 * mask
    num = masked.sum(dim=(1, 2))
    den = _safe_count_mask(mask).squeeze(-1).squeeze(-1) * y_pred.size(-1)
    per_sample_mse = num / (den + eps)
    return per_sample_mse.mean()

def loss_masked_mse_adapter(y_pred: torch.Tensor,
                            y_true: torch.Tensor,
                            xyz: torch.Tensor = None,
                            flag: torch.Tensor = None,
                            mask: torch.Tensor = None,
                            eps: float = 1e-8,
                            **kwargs) -> torch.Tensor:
    """
    유지: (y_pred, y_true, xyz, flag, ...) 시그니처
    flag가 없으면 전체 유효영역으로 가정.
    """
    B, N, C = y_pred.shape
    if mask is None:
        if flag is not None:
            if flag.dim() == 2:
                mask = (flag > 0).float()
            elif flag.dim() == 3:
                fluid_ch = 0
                mask = (flag[..., fluid_ch] > 0).float()
            else:
                raise ValueError("Unsupported flag shape for mask.")
        else:
            mask = torch.ones((B, N), dtype=y_pred.dtype, device=y_pred.device)
    return loss_masked_mse(y_pred, y_true, mask, eps=eps)
# ============================================================================ #

class PCWNOTrainer:
    def __init__(self, model: PCWNO, lr=3e-3, weight_decay=1e-6, w_bc: float = 2.0):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        def _loss_bw_mse(y_pred, y_true, points_flag=None, w_bc=2.0):
            diff2 = (y_pred - y_true) ** 2
            if points_flag is None:
                return diff2.mean()
            w = torch.ones_like(y_pred[..., :1])
            w = torch.where(points_flag.unsqueeze(-1) > 0.5, w_bc * w, w)
            return (diff2 * w).mean()

        self.w_bc = float(w_bc)
        self.loss_fn = lambda y_pred, y_true, points_flag=None: _loss_bw_mse(
            y_pred, y_true, points_flag=points_flag, w_bc=self.w_bc
        )


    def step(self, points_xyz, grid_xyz_ref, y_true):
        self.model.train()
        self.optim.zero_grad()
        y_pred = self.model(points_xyz, grid_xyz_ref)
        loss   = self.loss_fn(y_pred, y_true, points_flag=None)
        loss.backward()
        self.optim.step()
        return loss.item()

    @torch.no_grad()
    def eval(self, points_xyz, grid_xyz_ref, y_true):
        self.model.eval()
        y_pred = self.model(points_xyz, grid_xyz_ref)
        loss   = self.loss_fn(y_pred, y_true, points_flag=None)
        return loss.item(), y_pred

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='PC-WNO single-file trainer')
    parser.add_argument('--data', default='./Data_15000_mach0.8.npy', help='Path to Data_15000_mach0.8.npy')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', default='./logs/pcwno_airfoil')
    parser.add_argument('--grid_h', type=int, default=64)
    parser.add_argument('--grid_w', type=int, default=64)
    parser.add_argument('--use_nondim', action='store_true',
                        help='Use freestream-based nondimensionalization before min-max normalization (paper-aligned).')
    parser.add_argument('--p_is_cp', action='store_true', default=True,
                    help='Treat the p channel as pressure coefficient (Cp). If True, skip p nondimensionalization.')
    parser.add_argument('--uinf', type=float, default=None,
                        help='Freestream speed U_inf. If None, estimate from far-field/open boundary.')
    parser.add_argument('--pinf', type=float, default=None,
                        help='Freestream static pressure p_inf. If None, estimate from far-field/open boundary median.')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Fluid density for meta-consistency (not used in nondim, but kept for logging).')
    parser.add_argument('--use_train_stats', action='store_true', default=True,
                        help='Compute min/max only on train split and reuse for val/test (paper-aligned).')
    parser.add_argument('--w_bc', type=float, default=2.0,
                        help='Boundary weighting factor for loss at boundary/near-wall points.')
    parser.add_argument('--wavelet', default='db2')
    parser.add_argument('--wno_width', type=int, default=128)
    parser.add_argument('--wno_level', type=int, default=3)
    parser.add_argument('--wno_layers', type=int, default=6)
    parser.add_argument('--grid_range', type=float, nargs=2, default=(1.0,1.0))
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'figs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'dat'), exist_ok=True)

    in_tr, in_va, in_te, out_tr, out_va, out_te, stats = load_dataset_auto(args.data, args)
    
    # grid_np = load_grid_xyz(args.grid, stats=stats)  # not stats=None
    # grid_xyz_ref = torch.from_numpy(grid_np).float().to(device)
    
    # 내부 균일 VRG: [0,1]x[0,1], (H,W)
    grid_ref = make_internal_vrg(args.grid_h, args.grid_w, stats=None, device=torch.device(args.device))


    assert_grid_shape(grid_ref, args.grid_h, args.grid_w)

    train_ds = PointDataset(in_tr, out_tr)
    val_ds   = PointDataset(in_va, out_va)
    test_ds  = PointDataset(in_te, out_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)

    model = PCWNO(grid_hw=(args.grid_h, args.grid_w),
                  wno_width=args.wno_width, wno_level=args.wno_level, wno_layers=args.wno_layers,
                  wno_wavelet=args.wavelet, wno_grid_range=tuple(args.grid_range)).to(args.device)
    trainer = PCWNOTrainer(model, lr=args.lr, w_bc=args.w_bc)

    metrics_csv = os.path.join(args.save_dir, 'metrics.csv')
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','val_loss'])

    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        # Train
        epoch_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {ep}/{args.epochs} [train]", ncols=100, leave=False)
        for i, (xyz, y) in enumerate(pbar, 1):
            xyz = xyz.to(args.device)
            y   = y.to(args.device)
            loss = trainer.step(xyz, grid_ref, y)
            epoch_loss += loss * xyz.size(0)
            pbar.set_postfix(loss=f"{loss:.3e}")
        epoch_loss /= len(train_ds)

        # Val
        val_loss = 0.0
        with torch.no_grad():
            pbar_v = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {ep}/{args.epochs} [val]", ncols=100, leave=False)
            for i, (xyz, y) in enumerate(pbar_v, 1):
                xyz = xyz.to(args.device)
                y   = y.to(args.device)
                l, _ = trainer.eval(xyz, grid_ref, y)
                val_loss += l * xyz.size(0)
                pbar_v.set_postfix(loss=f"{l:.3e}")
        val_loss /= len(val_ds)

        # Log
        with open(metrics_csv, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f"{epoch_loss:.6e}", f"{val_loss:.6e}"])

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'pcwno.pt'))

    # Test + save GT/PRED/ERR images & .dat per case
    model.eval()
    print('Best val loss:', best_val)
    last = model.head.mlp[-1]  # PointHead의 마지막 1x1 Conv
    print('||head last W||:', float(last.weight.detach().norm()), 
        'mean(head last b):', float(last.bias.detach().mean()))

    # 평균 RMSE(u,v,p) (정규화 공간) 출력
    u_rmse, v_rmse, p_rmse = compute_average_rmse_pointwise(test_loader, model, grid_ref, args.device)
    print('[Test] Average normalized RMSE (u, v, p):', u_rmse, v_rmse, p_rmse)

    # 케이스별 .dat/그림 저장
    evaluate_and_save_cases(model, test_loader, grid_ref, stats, args.save_dir, args.device)

    print('Saved to', args.save_dir)

    
@torch.no_grad()
def test_and_save(model, loader, grid_ref, stats, save_dir, device):
    model.eval()
    case_off = 0
    for xyz, y in loader:
        yhat = model(xyz.to(device), grid_ref).cpu().numpy()
        ygt  = y.numpy()
        xyzn = xyz.numpy()
        B = yhat.shape[0]
        for b in range(B):
            save_case_outputs(save_dir, case_off + b, xyzn[b], yhat[b], ygt[b], stats)
        case_off += B

def compute_average_rmse_pointwise(loader, model, grid_ref, device):
    """
    PointUNet의 test()와 동일 개념:
      각 배치/포인트에 대해 (정규화 공간) RMSE를 산출하고
      데이터셋 평균을 반환.
    """
    import numpy as np
    model.eval()
    sum_u = sum_v = sum_p = 0.0
    cnt = 0
    with torch.no_grad():
        for xyz, y in loader:
            xyz = xyz.to(device)
            y   = y.to(device)
            yhat = model(xyz, grid_ref)

            err = yhat - y  # 정규화 공간
            rmse = torch.sqrt((err ** 2).mean(dim=1))  # (B,3) 채널별 평균
            sum_u += rmse[:, 0].sum().item()
            sum_v += rmse[:, 1].sum().item()
            sum_p += rmse[:, 2].sum().item()
            cnt   += rmse.size(0)
    return (sum_u/cnt, sum_v/cnt, sum_p/cnt)

@torch.no_grad()
def evaluate_and_save_cases(model, loader, grid_ref, stats, save_dir, device):
    model.eval()
    case_off = 0
    for xyz, y in loader:
        xyz = xyz.to(device)
        y   = y.to(device)
        yhat = model(xyz, grid_ref).cpu().numpy()
        ygt  = y.cpu().numpy()
        xyzn = xyz.cpu().numpy()
        B = yhat.shape[0]
        for b in range(B):
            save_case_outputs(save_dir, case_off + b, xyzn[b], yhat[b], ygt[b], stats)
        case_off += B




if __name__ == '__main__':
    main()
