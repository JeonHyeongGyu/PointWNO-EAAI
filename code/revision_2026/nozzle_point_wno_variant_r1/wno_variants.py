"""WNO kernel variants for the canonical-WNO follow-up experiment (r1). Operator code only; no data, no training.

Three wavelet-integral layers, all with the SAME transforms as the registered models (DWTExact / IDWTExact of
code/revision_2026/3d_recovery/r3/wavelet_exact.py: pytorch_wavelets forward arithmetic, exact autograd backward,
mode 'symmetric'):

  V3Conv(flip=False)  "V3-BASE"  independent re-implementation of the vendored wavelet_convolution_v3.WaveConv2d
                      forward (lines 305-345) as used through wno2d_r1.WaveConv2dR1 (lines 46-75): per coarsest
                      sub-band  rfft2 -> two (e_x x e_y) complex blocks (rows [:e_x] and [-e_x:], cols [:e_y]) ->
                      irfft2; finer-level details = 0; IDWT; crop. Built FROM an existing WaveConv2dR1 by re-using its
                      Parameter objects (same names, shapes, values) -> bit-identical forward / backward (tested).
  V3Conv(flip=True)   "V3-FLIP"  identical parameters and FFT block sizes; ONLY the position of the retained Fourier
                      block of the three DETAIL sub-bands changes. Each detail band b is modulated by
                      m_b[p, q] = (-1)^p (b = 0, high-pass along rows / dim -2), (-1)^q (b = 1, along cols / dim -1),
                      (-1)^(p+q) (b = 2, both) before the rfft2 and again after the irfft2:
                          y_b = m_b * F^-1( M * P_b * F(m_b * c_b) ).
                      Multiplying by (-1)^p shifts the coefficient-grid spectrum by X/2 bins, so the retained block
                      sits next to the coefficient Nyquist frequency instead of DC; because the high-pass + decimation
                      of the DWT maps the physical band [B, 2B] onto the coefficient grid in REVERSED order
                      (frequency inversion; canonical audit SUBBAND_INFORMATION_FLOW 2.2), the retained block then
                      covers the LOWER edge of each detail band (next to the LL band) instead of its upper edge.
                      The approximation (LL) band is not modulated. m_b is +-1, its own inverse and parameter-free.
                      Exact half-spectrum shift for even X / Y (nozzle 130 x 18, RAE pure 66 x 66); half-bin offset
                      for odd sizes (RAE point 65 x 65), reported by the probes.
  V2Conv             "V2-CANONICAL"  the kernel of the official WNO Version 2.0.0 (csccm-iitd/WNO
                      Version 2.0.0/wavelet_convolution.py, class WaveConv2d, lines 146-259): real weights
                      weights1..4 of shape (C_in, C_out, X, Y) (X x Y = coarsest coefficient grid), coefficient-wise
                      channel mixing einsum('bixy,ioxy->boxy') on LL_J (weights1) and on the three coarsest details
                      (weights2/3/4, orientation index 0/1/2), finer-level details = 0 (zeros_like), IDWT.
                      Differences from the official class (documented, tested): (a) DWT / IDWT backward = exact
                      adjoint (forward values bit-identical; the official library backward is not the adjoint for
                      'symmetric', which is why the registered models use the exact backward too); (b) the
                      resolution-change branch ('factor', official lines 226-243) is not used (input size must equal
                      the construction size; asserted); (c) init draws scale * U[0,1) (official distribution) from a
                      dedicated torch.Generator instead of the global RNG, and the shape is found with a zero dummy
                      instead of torch.randn (so the global RNG of the surrounding model is not consumed); (d) the
                      output is cropped to the input size (identity on every registered grid; asserted in tests).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)
for p in (os.path.join(REV, "3d_recovery", "r3"),):
    if p not in sys.path:
        sys.path.insert(0, p)
from wavelet_exact import DWTExact, IDWTExact  # noqa: E402  (SD3D r3, unchanged)

VARIANTS_VERSION = "wno_variants_r1_v1"
V3_WEIGHT_NAMES = ("weights_a1", "weights_a2", "weights_h1", "weights_h2",
                   "weights_v1", "weights_v2", "weights_d1", "weights_d2")
V2_WEIGHT_NAMES = ("weights1", "weights2", "weights3", "weights4")
DETAIL_W = (("weights_h1", "weights_h2"), ("weights_v1", "weights_v2"), ("weights_d1", "weights_d2"))


def flip_mask(band: int, X: int, Y: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(-1)^p / (-1)^q / (-1)^(p+q) on an X x Y coefficient grid for detail orientation 0 / 1 / 2."""
    p = torch.arange(X, device=device)
    q = torch.arange(Y, device=device)
    sp = (1 - 2 * (p % 2)).to(dtype).view(X, 1)
    sq = (1 - 2 * (q % 2)).to(dtype).view(1, Y)
    if band == 0:
        return sp.expand(X, Y)
    if band == 1:
        return sq.expand(X, Y)
    if band == 2:
        return sp * sq
    raise ValueError(band)


def coef_grid(hw, level, wavelet="db2"):
    from pytorch_wavelets import DWT
    a, _ = DWT(J=level, mode="symmetric", wave=wavelet)(torch.zeros(1, 1, *hw))
    return int(a.shape[-2]), int(a.shape[-1])


class V3Conv(nn.Module):
    """V3-BASE (flip=False) / V3-FLIP (flip=True). Parameters are taken over from a WaveConv2dR1 (no new draws)."""

    def __init__(self, level, wavelet, mode, size, weights: dict, flip: bool):
        super().__init__()
        if mode != "symmetric":
            raise ValueError("exact-backward transforms support mode='symmetric' only")
        self.level, self.wavelet, self.mode, self.size, self.flip = int(level), wavelet, mode, list(size), bool(flip)
        for n in V3_WEIGHT_NAMES:
            w = weights[n]
            if not isinstance(w, nn.Parameter):
                raise TypeError(n)
            setattr(self, n, w)                                   # same Parameter object -> same values / identity

    @classmethod
    def from_r1(cls, conv, flip: bool):
        return cls(conv.level, conv.wavelet, conv.mode, conv.size, {n: getattr(conv, n) for n in V3_WEIGHT_NAMES}, flip)

    @staticmethod
    def spectral(waves, w1, w2):
        """= wavelet_convolution_v3.WaveConv2d.spectralconv (vendored lines 257-303), same operations and order."""
        xw = torch.fft.rfft2(waves)
        B, C, X, Y_r = xw.shape
        _, _, Wx, Wy = w1.shape
        mx, my = min(X, Wx), min(Y_r, Wy)
        out = torch.zeros(B, w1.shape[1], X, Y_r, dtype=torch.cfloat, device=waves.device)
        out[:, :, :mx, :my] = torch.einsum("bixy,ioxy->boxy", xw[:, :, :mx, :my], w1[:, :, :mx, :my])
        out[:, :, -mx:, :my] = torch.einsum("bixy,ioxy->boxy", xw[:, :, -mx:, :my], w2[:, :, :mx, :my])
        return torch.fft.irfft2(out, s=(waves.shape[-2], waves.shape[-1]))

    def forward(self, x):
        in_shape = x.shape
        if x.shape[-1] != self.size[-1] or x.shape[-2] != self.size[-2]:
            raise ValueError(f"input {tuple(x.shape[-2:])} != construction size {self.size}")
        dwt = DWTExact(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
        x_ft, x_coeff = dwt(x.contiguous())
        out_coeff = [torch.zeros_like(c, device=x.device) for c in x_coeff]
        out_ft = self.spectral(x_ft, self.weights_a1, self.weights_a2)
        for b, (n1, n2) in enumerate(DETAIL_W):
            c = x_coeff[-1][:, :, b, :, :].clone()
            if self.flip:
                m = flip_mask(b, c.shape[-2], c.shape[-1], c.device, c.dtype)
                c = c * m
                out_coeff[-1][:, :, b, :, :] = self.spectral(c, getattr(self, n1), getattr(self, n2)) * m
            else:
                out_coeff[-1][:, :, b, :, :] = self.spectral(c, getattr(self, n1), getattr(self, n2))
        x_rec = IDWTExact(mode=self.mode, wave=self.wavelet).to(x.device)((out_ft, out_coeff))
        return x_rec[..., :in_shape[-2], :in_shape[-1]]


class V2Conv(nn.Module):
    """Official WNO V2.0.0 kernel (coefficient-wise real multiplication) with exact-backward transforms."""

    def __init__(self, in_channels, out_channels, level, size, wavelet, mode="symmetric", generator=None):
        super().__init__()
        if mode != "symmetric":
            raise ValueError("exact-backward transforms support mode='symmetric' only")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.level, self.size, self.wavelet, self.mode = int(level), list(size), wavelet, mode
        self.modes1, self.modes2 = coef_grid(self.size, self.level, wavelet)
        self.scale = 1 / (in_channels * out_channels)                       # official line 193
        for n in V2_WEIGHT_NAMES:                                           # official lines 194-197, same order
            setattr(self, n, nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,
                                                                  generator=generator)))

    @staticmethod
    def mul2d(inp, weights):
        return torch.einsum("bixy,ioxy->boxy", inp, weights)               # official line 215

    def forward(self, x):
        in_shape = x.shape
        if x.shape[-1] != self.size[-1] or x.shape[-2] != self.size[-2]:
            raise ValueError(f"input {tuple(x.shape[-2:])} != construction size {self.size}")
        dwt = DWTExact(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
        x_ft, x_coeff = dwt(x.contiguous())
        out_coeff = [torch.zeros_like(c, device=x.device) for c in x_coeff]            # official line 247
        out_ft = self.mul2d(x_ft, self.weights1)                                         # official line 250
        out_coeff[-1][:, :, 0, :, :] = self.mul2d(x_coeff[-1][:, :, 0, :, :].clone(), self.weights2)  # 252
        out_coeff[-1][:, :, 1, :, :] = self.mul2d(x_coeff[-1][:, :, 1, :, :].clone(), self.weights3)  # 253
        out_coeff[-1][:, :, 2, :, :] = self.mul2d(x_coeff[-1][:, :, 2, :, :].clone(), self.weights4)  # 254
        x_rec = IDWTExact(mode=self.mode, wave=self.wavelet).to(x.device)((out_ft, out_coeff))       # 257-258
        return x_rec[..., :in_shape[-2], :in_shape[-1]]


# ----------------------------------------------------------------------------------------------- core conversion
def to_v3(core, flip: bool):
    """Replace every WaveConv2dR1 of a WNO2d core by V3Conv re-using its parameters. Returns the number replaced."""
    from wno2d_r1 import WaveConv2dR1
    n = 0
    for i, conv in enumerate(core.conv):
        if type(conv) is not WaveConv2dR1:
            raise TypeError(type(conv))
        core.conv[i] = V3Conv.from_r1(conv, flip)
        n += 1
    return n


def to_v2(core, level: int, generator: torch.Generator):
    """Replace every spectral layer of a WNO2d core by a freshly initialised V2Conv (layer order 0..L-1, one
    generator). Everything else in the core (fc0, 1x1 convs, fc1, fc2, grid buffer) is untouched."""
    n = 0
    for i, conv in enumerate(core.conv):
        size = list(conv.size)
        cin = cout = core.width
        core.conv[i] = V2Conv(cin, cout, level, size, conv.wavelet, conv.mode, generator=generator)
        n += 1
    return n


def kernel_params(core) -> int:
    return int(sum(p.numel() * (2 if p.is_complex() else 1) for n, p in core.named_parameters() if n.startswith("conv.")))
