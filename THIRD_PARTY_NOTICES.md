# Third-party code, external dependency and notices

The files under `code/` are byte-identical copies of the research code that trained and evaluated the released models
(the SHA-256 of each training-relevant file is recorded in `checkpoints/naca/*/run_meta.json`), with the exceptions
listed in section 1. None of them is edited here.

## 1. Official WNO repository — external dependency, NOT redistributed

| | |
|---|---|
| Repository | https://github.com/csccm-iitd/WNO (T. Tripura, S. Chakraborty; *Computer Methods in Applied Mechanics and Engineering* 404 (2023) 115783) |
| License | **none declared.** No LICENSE / COPYING / NOTICE file at commit `fa6b5a7` or at the current head `e2b46cc`, none in the history, GitHub metadata `license: null`, no license statement in the README or in the source files (checked 2026-11-16). The README asks users to cite the WNO paper. |
| Used here | `wavelet_convolution_v3.py` at commit `fa6b5a79ad97ad5c2f8ee3773fe3c732d344dee3`, **unmodified** (SHA-256 `5b80065a…`). Not included in this repository or its release assets; you download it yourself with `tools/fetch_wno_upstream.py` into `external/` (excluded from version control). |

Because no license grants redistribution, this repository does not contain upstream WNO code, modified copies of it, or
patches containing it. Whether and how you may use the downloaded file is governed by its authors; this repository
makes no statement on their behalf.

Two files of the original research code are therefore **not part of this release**:

* `code/third_party/pointwno2d_original/wavelet_convolution_v3.py` — the modified copy of the upstream file used during
  training (SHA-256 `50b9fcdd…`; changes only in `WaveConv2d.forward` and `WaveConv2d.spectralconv`). The released
  models only run `WaveConv2d.__init__` while being constructed and then replace every such layer, so the unmodified
  upstream file gives bit-identical model construction, checkpoint loading, forward outputs and gradients for all six
  released models (verified; `results/release_verification_20261116/`).
* `code/revision_2026/2d_recovery/r1_exact_backward/wno2d_r1.py` — exact-backward variant of the sub-band WNO layer,
  derived from the modified `WaveConv2d.forward`. Used only by sub-band WNO runs, which are not released; it is replaced
  by a placeholder module that raises on any use (`pointwno/_deps.py`). `code/pwno2d_ext/`, imported only by it, is
  also omitted.

The registry pins of these two files are checked by `pointwno/_deps.py` against this substitution; every other pinned
file is checked unchanged.

## 2. Code of this work that builds on other projects

| File | Relation | License of the other project |
|---|---|---|
| `code/revision_2026/3d_recovery/r3/wavelet_exact.py` (`afb1d_sym`, `sfb1d_sym`) | adapted from `afb1d` / `sfb1d` of `pytorch_wavelets/dwt/lowlevel.py` (same arithmetic, custom autograd removed; stated in the docstrings) | MIT, notice below |
| `code/pwno2d/models.py` (`SpectralConv2d`) | standard Fourier-neural-operator spectral convolution in the formulation of Li et al. (ICLR 2021) | the FNO reference code is MIT (notice below); a line-by-line comparison with its original 2-D script was not possible (file no longer at its original location) |
| `code/revision_2026/nozzle_point_wno_variant_r1/wno_variants.py` (`V2Conv`) | independent implementation of the canonical WNO Version 2.0.0 kernel (coefficient-wise channel mixing); no upstream code | — |
| `code/third_party/pointwno2d_original/pcwno_vrg_lpe_torch_airfoil_momentum_fno_same.py` | PointWNO 2-D model of the authors (VRG, LPE `SA_MSG`, `WNO2d` grid network, FP, head). The point-cloud interface follows PointUNet (Hwang, Kim, Park, Choi, *Physics of Fluids* 37, 057126, 2025) conceptually; no PointUNet code is contained. The `WNO2d` container uses the constructor arguments and the layer structure of the upstream `WNO2d` class (lift, [operator + 1×1 convolution] × L, projection); its code is written separately (only attribute assignments coincide) | — |

### pytorch_wavelets (applies to the adapted parts of `wavelet_exact.py`)

```
MIT License

Copyright (c) 2020 Fergal Cotter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

(pytorch_wavelets states that its DWT sections are under this license; its DTCWT parts carry further conditions, but
no DTCWT code is used or adapted here.)

### Fourier neural operator reference code (precautionary notice for `SpectralConv2d`)

The reference implementation is now distributed in https://github.com/neuraloperator/neuraloperator, whose LICENSE reads:

```
MIT License

Copyright (c) 2023 NeuralOperator developers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 3. Python dependencies (installed by the user, not redistributed)

PyTorch (BSD-3-Clause), pytorch_wavelets 1.3.0 (MIT for the DWT parts), PyWavelets (MIT), ptwt 0.1.9 (EUPL-1.2), NumPy,
SciPy, pandas, Matplotlib, tqdm. The upstream WNO file imports `ptwt` and `pytorch_wavelets` at import time.

## 4. Known quirks of the unchanged model file

`SA_MSG.forward` prints `STD l1_points` with probability 0.02 per training step and draws one number from the global
torch RNG each step; the module imports `pandas` and `matplotlib.pyplot` at import time. The upstream WNO file prints an
installation hint if `ptwt` or `pytorch_wavelets` is missing.
