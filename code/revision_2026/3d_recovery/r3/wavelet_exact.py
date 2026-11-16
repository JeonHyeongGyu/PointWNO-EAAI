"""SD3D r3: 2D DWT / IDWT with the SAME forward as pytorch_wavelets 1.3.0 (DWTForward / DWTInverse) and an EXACT
backward obtained by autograd through supported PyTorch operations.

Root cause (docs/revision_2026/3d_recovery_20261020/sd3d_r3_wno_backward_correction.md, section 1):
pytorch_wavelets computes the forward with plain tensor operations but wraps them in custom autograd Functions
(lowlevel.AFB2D, lowlevel.SFB2D) whose hand-written backward passes are not the adjoints of the forward for
mode='symmetric' with filters longer than 2:
  analysis   A = D C_h E          exact A^T = E^T (D C_h)^T ;  AFB2D.backward uses R (crop) in place of E^T
  synthesis  S = R_{L-2} U_g      exact S^T = (D C_g) Z      ;  SFB2D.backward uses E_sym in place of Z (zero pad)
(E: symmetric extension, D C_h: correlation + stride 2, U_g: transposed convolution, R: crop, Z: zero padding).

This module keeps the forward arithmetic of the library (same filters from lowlevel.prep_filt_*, same F.conv2d /
F.conv_transpose2d calls, same strides, groups and paddings, same reshapes) and only removes the custom Functions,
so that autograd differentiates the forward itself. The symmetric extension, which the library builds by integer
indexing (lowlevel.mypad), is built here by torch.flip + torch.cat: it produces bit-identical values and its
backward is a deterministic fold (sum of the mirrored gradients) instead of an indexed scatter-add.
Only J levels with mode 'symmetric' are implemented here (the r3 configuration); other modes raise.
Nothing in the pytorch_wavelets installation or in the vendored sources is modified.
"""
from __future__ import annotations

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets.dwt import lowlevel as LL

R3_BACKWARD_VERSION = "r3_exact_autograd_v1"


def _sym_extend(x: torch.Tensor, p1: int, p2: int, d: int) -> torch.Tensor:
    """numpy/pywt 'symmetric' extension (half-sample, x[-1] = x[0]) along dim d, as lowlevel.mypad(mode='symmetric')."""
    n = x.shape[d]
    if p1 > n or p2 > n:
        raise ValueError(f"symmetric extension of {p1}/{p2} samples needs at least that many samples, got {n}")
    parts = []
    if p1 > 0:
        parts.append(torch.flip(x.narrow(d, 0, p1), dims=(d,)))
    parts.append(x)
    if p2 > 0:
        parts.append(torch.flip(x.narrow(d, n - p2, p2), dims=(d,)))
    return torch.cat(parts, dim=d) if len(parts) > 1 else x


def afb1d_sym(x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, dim: int) -> torch.Tensor:
    """lowlevel.afb1d(x, h0, h1, mode='symmetric', dim) without the custom Function (same arithmetic)."""
    C = x.shape[1]
    d = dim % 4
    s = (2, 1) if d == 2 else (1, 2)
    N = x.shape[d]
    L = h0.numel()
    shape = [1, 1, 1, 1]
    shape[d] = L
    h0 = h0.reshape(*shape)
    h1 = h1.reshape(*shape)
    h = torch.cat([h0, h1] * C, dim=0)
    outsize = pywt.dwt_coeff_len(N, L, mode="symmetric")
    p = 2 * (outsize - 1) - N + L
    x = _sym_extend(x, p // 2, (p + 1) // 2, d)
    return F.conv2d(x, h, stride=s, groups=C)


def sfb1d_sym(lo: torch.Tensor, hi: torch.Tensor, g0: torch.Tensor, g1: torch.Tensor, dim: int) -> torch.Tensor:
    """lowlevel.sfb1d(lo, hi, g0, g1, mode='symmetric', dim): already plain operations; differentiated by autograd."""
    return LL.sfb1d(lo, hi, g0, g1, mode="symmetric", dim=dim)


class DWTExact(nn.Module):
    """Drop-in for pytorch_wavelets.DWTForward(J, wave, mode='symmetric'): same buffers, same outputs, exact backward."""

    def __init__(self, J: int = 1, wave: str = "db1", mode: str = "symmetric"):
        super().__init__()
        if mode != "symmetric":
            raise ValueError("DWTExact implements mode='symmetric' only (the r3 configuration)")
        w = pywt.Wavelet(wave) if isinstance(wave, str) else wave
        filts = LL.prep_filt_afb2d(w.dec_lo, w.dec_hi, w.dec_lo, w.dec_hi)
        self.register_buffer("h0_col", filts[0])
        self.register_buffer("h1_col", filts[1])
        self.register_buffer("h0_row", filts[2])
        self.register_buffer("h1_row", filts[3])
        self.J = J
        self.mode = mode

    def forward(self, x):
        yh = []
        ll = x
        for _ in range(self.J):
            # = lowlevel.AFB2D.forward
            lohi = afb1d_sym(ll, self.h0_row, self.h1_row, dim=3)
            y = afb1d_sym(lohi, self.h0_col, self.h1_col, dim=2)
            s = y.shape
            y = y.reshape(s[0], -1, 4, s[-2], s[-1])
            ll = y[:, :, 0].contiguous()
            yh.append(y[:, :, 1:].contiguous())
        return ll, yh


class IDWTExact(nn.Module):
    """Drop-in for pytorch_wavelets.DWTInverse(wave, mode='symmetric'): same buffers, same output, exact backward."""

    def __init__(self, wave: str = "db1", mode: str = "symmetric"):
        super().__init__()
        if mode != "symmetric":
            raise ValueError("IDWTExact implements mode='symmetric' only (the r3 configuration)")
        w = pywt.Wavelet(wave) if isinstance(wave, str) else wave
        filts = LL.prep_filt_sfb2d(w.rec_lo, w.rec_hi, w.rec_lo, w.rec_hi)
        self.register_buffer("g0_col", filts[0])
        self.register_buffer("g1_col", filts[1])
        self.register_buffer("g0_row", filts[2])
        self.register_buffer("g1_row", filts[3])
        self.mode = mode

    def forward(self, coeffs):
        yl, yh = coeffs
        ll = yl
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2], ll.shape[-1], device=ll.device, dtype=ll.dtype)
            # 'Unpad' added dimensions (as DWTInverse.forward)
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[..., :-1, :]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[..., :-1]
            # = lowlevel.SFB2D.forward
            lh, hl, hh = torch.unbind(h, dim=2)
            lo = sfb1d_sym(ll, lh, self.g0_col, self.g1_col, dim=2)
            hi = sfb1d_sym(hl, hh, self.g0_col, self.g1_col, dim=2)
            ll = sfb1d_sym(lo, hi, self.g0_row, self.g1_row, dim=3)
        return ll
