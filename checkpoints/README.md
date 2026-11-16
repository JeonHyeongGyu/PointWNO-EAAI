# Trained models (airfoil family, shock benchmark)

Six checkpoints: canonical PointWNO and parameter-matched PointFNO, three training seeds each. These are the models of
the manuscript's shock benchmark (Table "Shock benchmark", airfoil-family rows) and of the reference rows of the
original-PointUNet comparison.

| Directory | Model (manuscript name) | Seed | Parameters | Best epoch | `pcwno_best.pt` SHA-256 |
|---|---|---|---|---|---|
| `naca/BM_NACA_V2J2_s0` | PointWNO (canonical kernel, db2, J = 2) | 0 | 21,801,731 | 490 | `d0c6a327e0c8118b7154962fea1e455df598f85962c1484c9e06a218bdc5d8f3` |
| `naca/BM_NACA_V2J2_s1` | PointWNO | 1 | 21,801,731 | 497 | `ac208b6218eb68db09b7128db792284e750e4bd8833437894a19031f84546d99` |
| `naca/BM_NACA_V2J2_s2` | PointWNO | 2 | 21,801,731 | 438 | `b9b8bd239be756e58c1586d1439302f49324e9edd1818827f43671a171b5d8a2` |
| `naca/BM_NACA_FNOM_s0` | PointFNO (34 × 26 modes) | 0 | 21,801,731 | 152 | `c309a6b863d0d8d39e8684f80f214a317572638253ae12fa2025c07e5ad8d073` |
| `naca/BM_NACA_FNOM_s1` | PointFNO | 1 | 21,801,731 | 102 | `481a1ce97090a465e94aea00214d29c279528c44c161b5e797f2b60c1bd19dcb` |
| `naca/BM_NACA_FNOM_s2` | PointFNO | 2 | 21,801,731 | 101 | `b9bc86e4ecdd7726e298b4f37ccdb65b904c50dc05dd4d4efa65cb9029194581` |

* Each `pcwno_best.pt` (≈ 87 MB, a PyTorch `state_dict`) is distributed as a GitHub Release asset —
  **TODO(release-url): link to be added when the release is created**. Place it in the directory of its run.
* The small run records in each directory are part of the repository: `run_meta.json` (configuration, environment,
  SHA-256 of every code file used for training), `metrics.csv` (per-epoch losses), `val_summary.json`, `complete.json`,
  `status.json`, `val_case_RMSE.csv`, `SHA256SUMS_run.txt`.
* `pcwno_best.pt` is the weight set with the lowest validation loss (strict first minimum), the selection rule of the
  manuscript. `last_model.pt` (final-epoch weights, used by no result) is listed in `SHA256SUMS_run.txt` but not
  released.
* Training: NVIDIA H100 NVL, Python 3.10.21, torch 2.5.1+cu121, pytorch_wavelets 1.3.0, matmul TF32 off, cuDNN TF32 on,
  500 epochs, batch 4, Adam 1e-3, StepLR (100, 0.5), min lr 1e-5.
* Run identifiers are those of the training and evaluation records and are not renamed (manuscript Supplementary
  Table S1: `V2J2` = PointWNO, `FNOM` = PointFNO).

Load one:

```python
from pointwno import build_model, load_checkpoint
model = load_checkpoint(build_model("V2J2", seed=0), "checkpoints/naca/BM_NACA_V2J2_s0/pcwno_best.pt")
```

## Not released

The other trained models of the manuscript are not in this release: sub-band WNO capacity / wavelet-family models,
component ablation, original PointUNet, RAE2822, nozzle and SuperWing models.
