"""Model construction for the 2D PointFNO / PointWNO comparison.

Both models are the ORIGINAL ``PCWNO`` pipeline (vendored, unmodified, in
``code/third_party/pointwno2d_original``):

    VRG -> SA_MSG (LPE) -> grid core -> FP_IDW -> PointHead

``operator = "wno"`` keeps the original ``WNO2d`` core, so its ``state_dict`` is
key-for-key compatible with the original checkpoints.

``operator = "fno"`` replaces only ``model.wno`` with ``FNO2dGridCore``, which has
the same interface, the same lift / projection layers and the same activation
as ``WNO2d``; the only difference is the operator layer (Fourier instead of
wavelet). LPE, grid ordering, FP and head are the same code objects.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "third_party", "pointwno2d_original")
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # the vendored file imports pyplot at module level

from pcwno_vrg_lpe_torch_airfoil_momentum_fno_same import PCWNO, make_internal_vrg  # noqa: E402

LATENT_OUT = 128  # width of the grid feature handed to FP and the head (fixed in the original PCWNO)


class SpectralConv2d(nn.Module):
    """Standard FNO spectral convolution (same formulation as the 3D ``fno2d.py`` of the original project)."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if 2 * self.modes1 > h or self.modes2 > w // 2 + 1:
            raise ValueError(f"modes ({self.modes1}, {self.modes2}) do not fit a {h}x{w} grid")
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(h, w))


class FNO2dGridCore(nn.Module):
    """Drop-in replacement for the original ``WNO2d``: same I/O, same lift/projection, Fourier layers."""

    def __init__(self, width: int, layers: int, modes1: int, modes2: int, in_channel: int, grid_range=(1.0, 1.0)):
        super().__init__()
        self.register_buffer("fixed_grid_xyhw", None, persistent=True)
        self.width, self.layers = width, layers
        self.modes1, self.modes2 = modes1, modes2
        self.grid_range = list(grid_range)
        self.fc0 = nn.Linear(in_channel + 2, width)
        self.conv = nn.ModuleList([SpectralConv2d(width, width, modes1, modes2) for _ in range(layers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, LATENT_OUT)
        self.fc2 = nn.Linear(LATENT_OUT, LATENT_OUT)

    def get_grid(self, shape, device):
        b, h, w = shape[0], shape[1], shape[2]
        if self.fixed_grid_xyhw is not None:
            return self.fixed_grid_xyhw.to(device).unsqueeze(0).expand(b, h, w, 2)
        gx = torch.linspace(0, self.grid_range[0], w, device=device).view(1, 1, w, 1).repeat(b, h, 1, 1)
        gy = torch.linspace(0, self.grid_range[1], h, device=device).view(1, h, 1, 1).repeat(b, 1, w, 1)
        return torch.cat((gx, gy), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self.get_grid(x.shape, x.device)), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        for i, (conv, w) in enumerate(zip(self.conv, self.w)):
            x = conv(x) + w(x)
            if i != self.layers - 1:
                x = F.mish(x)
        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.mish(self.fc1(x)))


def build_model(cfg: dict) -> nn.Module:
    """cfg is the ``model`` block of a config file."""
    grid_hw = tuple(cfg["grid_hw"])
    op = cfg["operator"]
    if op == "wno":
        return PCWNO(grid_hw=grid_hw, wno_width=cfg["width"], wno_level=cfg["wno_level"],
                     wno_layers=cfg["layers"], wno_wavelet=cfg["wavelet"], wno_grid_range=(1.0, 1.0))
    if op == "fno":
        # build the shared pipeline with a throw-away minimal WNO core, then swap the core
        model = PCWNO(grid_hw=grid_hw, wno_width=2, wno_level=1, wno_layers=1, wno_wavelet="db2",
                      wno_grid_range=(1.0, 1.0))
        sa_out = model.wno.fc0.in_features - 2
        model.wno = FNO2dGridCore(width=cfg["width"], layers=cfg["layers"], modes1=cfg["modes"][0],
                                  modes2=cfg["modes"][1], in_channel=sa_out)
        return model
    raise ValueError(f"unknown operator {op!r}")


def build_vrg(cfg: dict, device) -> torch.Tensor:
    """The original internal VRG: a uniform (H x W) grid on [0, 1]^2 in normalized coordinates."""
    h, w = cfg["grid_hw"]
    return make_internal_vrg(h, w, stats=None, device=device)


@torch.no_grad()
def prime(model: nn.Module, grid_ref: torch.Tensor) -> None:
    """Run one tiny forward pass so the lazily created buffers (``fixed_grid_xyhw``, ``grid_perm_hw``) exist.

    The original model creates these buffers on its first forward call. They must exist before a
    checkpoint is loaded, and creating them here keeps ``state_dict`` keys identical before and after training.
    """
    was_training = model.training
    model.eval()
    dev = grid_ref.device
    pts = torch.rand(1, 128, 3, device=dev)
    pts[..., 2] = 0.0
    model(pts, grid_ref)
    model.train(was_training)


def count_parameters(model: nn.Module) -> dict:
    """Real-valued trainable parameter counts (a complex parameter counts as two)."""
    groups = {"lpe": 0, "operator_layers": 0, "lift_projection": 0, "head": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel() * (2 if p.is_complex() else 1)
        if name.startswith("sa."):
            groups["lpe"] += n
        elif name.startswith("wno.conv.") or name.startswith("wno.w."):
            groups["operator_layers"] += n
        elif name.startswith("wno."):
            groups["lift_projection"] += n
        elif name.startswith("head."):
            groups["head"] += n
        else:
            groups["other"] += n
    groups["total"] = sum(groups.values())
    return groups
