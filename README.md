# PointWNO-EAAI

Code, data and trained models for

> H.-g. Jeon and S. Choi, **A Point-Cloud-Based Wavelet Neural Operator Framework for Aerodynamic Field Prediction on
> Variable Geometries**, *Engineering Applications of Artificial Intelligence*, manuscript EAAI-26-17015 —
> **under revision** (not yet accepted; no DOI).

Data-Driven Simulation and Design Optimization Laboratory, Gwangju Institute of Science and Technology (GIST).

## Overview

CFD solutions on variable geometries live on irregular point sets whose size and topology change from case to case.
Wavelet neural operators (WNO; Tripura and Chakraborty, 2023) are formulated on regular grids. PointWNO connects the two:

1. **Local point encoding (LPE)** — each node of a fixed **virtual reference grid (VRG)** encodes the CFD points in its
   neighbourhood with multiscale ball queries and shared MLPs;
2. **WNO** — the solution operator is learned on the VRG latent tensor;
3. **Feature propagation (FP)** — the latent features are interpolated back to every original CFD point
   (inverse-distance weighting), and a pointwise **head** predicts the outputs.

No mesh connectivity and no interpolation of the CFD solution onto a grid are used.

What is taken from prior work and what is new:

* The point-cloud interface (VRG, multiscale LPE, FP, point head) follows **PointUNet** (Hwang, Kim, Park and Choi,
  *Physics of Fluids* 37, 057126, 2025), where the latent block is a convolutional U-Net.
* The wavelet integral layer is that of **WNO**. The benchmark model uses the kernel of the official WNO Version 2.0.0
  (coefficient-wise channel mixing of the level-J approximation and detail sub-bands).
* This work combines them, replaces the U-Net by a WNO, and compares the wavelet kernel with a **parameter-matched
  Fourier kernel** (PointFNO) that shares every other component. It does not introduce a new wavelet transform, a new
  neural-operator layer, or a new point-cloud encoder.

## Scope of this release

| Released | Content |
|---|---|
| Code | the training and evaluation code of the airfoil-family shock benchmark (canonical PointWNO vs PointFNO), byte-identical to the files recorded in the training runs, **except two WNO-derived files that are not redistributed** (see *Third-party WNO dependency*) |
| Data | the generated 2-D NACA 4-digit dataset (120 cases), split, normalization statistics (`data/README.md`) |
| Models | 6 checkpoints: PointWNO and PointFNO × 3 seeds (`checkpoints/README.md`) |
| Records | official validation evaluation records of these 6 models and the frozen benchmark collector output |

**Not in this repository:** any code of the official WNO repository (one of its files must be downloaded separately,
see below). **Not released:** the original-PointUNet comparison, the component ablation, the sub-band WNO capacity and
wavelet-family studies, and the RAE2822, nozzle-plume and SuperWing experiments (code, models and in-house nozzle data).
SuperWing and RAE2822 are public datasets of other authors; see `data/README.md` for their sources.

## Architecture (airfoil family)

| Component | Setting |
|---|---|
| Input / output | point coordinates `[x, y, 0]` (min–max normalized) → `[ρu, ρv, C_p]` at every input point |
| VRG | 128 × 96 nodes of a uniform grid in [0, 1]², identical for all airfoils, radial-shell × angle order in the latent tensor |
| LPE | 3 scales, radii (0.1, 0.2, 0.4), neighbours (16, 32, 64), MLPs [16,16,32], [32,32,64], [64,64,128] → 224 channels |
| Lifting | Linear(224 + 2 → 32) |
| Operator | 6 layers × width 32, Mish |
| PointWNO kernel (`V2J2`) | canonical WNO V2 kernel, db2, symmetric extension, level J = 2 → 34 × 26 coefficient grid; per-coefficient real channel mixing of LL_J and the three level-J detail bands; finer details set to zero; exact-adjoint DWT backward |
| PointFNO kernel (`FNOM`) | FNO spectral convolution with 34 × 26 modes (two corner blocks), same width / depth |
| FP | inverse-distance weighting, k = 3, p = 2 |
| Head | [128, 64, 3] |
| Trainable parameters | **21,801,731 for both models** (checked in code) |
| Training | masked MSE, Adam 1e-3, StepLR(step 100, γ 0.5, lower bound 1e-5), batch 4, 500 epochs, best validation loss kept |

Code map: `docs/CODE_MAP.md`.

## Third-party WNO dependency (download separately)

The model code imports `WaveConv2d` from `wavelet_convolution_v3.py` of the official WNO repository
(<https://github.com/csccm-iitd/WNO>). That repository declares **no license** (no license file at the used commit or
at its current head; GitHub reports no license). This repository therefore does **not** contain that file, a modified
copy of it, or a patch of it, and the GitHub release assets do not contain it either. You download the unmodified file
yourself from the fixed commit:

| | |
|---|---|
| Commit | `fa6b5a79ad97ad5c2f8ee3773fe3c732d344dee3` (the file was later removed from the default branch but remains in the history) |
| File | `wavelet_convolution_v3.py`, SHA-256 `5b80065afdcf2a24ce1b9e60d946132fa4d7a3da54a285bfa2055d32bac0e3b6` (git blob `9bf0a942…`) |
| Destination | `external/csccm-iitd_WNO_fa6b5a7/wavelet_convolution_v3.py` (ignored by git) |

```bash
python -B tools/fetch_wno_upstream.py                  # downloads and verifies; writes nothing if the hash differs
python -B tools/fetch_wno_upstream.py --from_file F    # alternatively, a copy you obtained yourself (e.g. git clone + checkout)
```

Any use of the downloaded file is subject to whatever terms its authors set; this repository grants nothing for it.

Why this works for the released models: during training, a slightly modified copy of this file was used (changes
only in `WaveConv2d.forward` / `spectralconv`). Both released models call only `WaveConv2d.__init__` while they are
built and then replace every such layer (`V2Conv` in PointWNO, the Fourier core in PointFNO), so the modified methods
never run. With the unmodified upstream file, all six checkpoints give bit-identical construction (state dict and RNG
state), forward outputs, losses and parameter gradients compared with the original research code (single-threaded CPU
check; `results/release_verification_20261116/`). The exact-backward sub-band layer `wno2d_r1.py` (also WNO-derived, used
only by sub-band WNO runs that are not released) is replaced by a placeholder module. `pointwno/_deps.py` sets this up
and checks the registry pins of the two omitted files against this substitution; every other pinned file is checked
unchanged. Details: `THIRD_PARTY_NOTICES.md`.

**Entry points.** Use the release entry points (`evaluation/evaluate_released.py`, `training/train_released.py`, the
`pointwno` package, `tests/`). Calling `code/.../train_naca_bm.py` or `eval_naca_bm.py` directly does not work in this
release, because the omitted files are not on the import path.

## Installation

Verified locally with Python 3.10.18, torch 2.8.0+cu128 (CUDA 12.8) on Linux (WSL2), in an existing environment with
the versions of `requirements.txt`; installation into a new environment from `requirements.txt` was not tested. The checkpoints were trained with
torch 2.5.1+cu121 on an NVIDIA H100 NVL.

```bash
conda create -n pointwno python=3.10 -y && conda activate pointwno
pip install torch --index-url https://download.pytorch.org/whl/cu128   # choose the build for your CUDA
pip install -r requirements.txt
python -B tools/fetch_wno_upstream.py
```

## Reproduction

All commands are run from the repository root. Set `PYTHONDONTWRITEBYTECODE=1` if you want no `__pycache__`.

1. **Get the data and checkpoints.** Download `Data_allpoints_xymomcp_from_dat.npz` into `data/naca/` and each
   `pcwno_best.pt` into its run directory `checkpoints/naca/BM_NACA_<arm>_s<seed>/` (release assets; see the two README
   files). Check them:

   ```bash
   python -B tools/verify_release.py --require_assets   # SHA-256 of every file in SHA256SUMS.txt
   ```

2. **Run the checks** (CPU, about 30 s; needs the external WNO file; tests that need the data or checkpoints are skipped
   when they are absent):

   ```bash
   python -m pytest -q tests/test_release.py
   ```

   They verify the SHA-256 pins of the code; that every code file recorded by the training runs is present
   byte-identically (except the two omitted files, whose recorded values are checked instead); the external WNO file and
   that it is the module the model imports; that no repository file reproduces lines of it; the parameter count
   21,801,731 of both models; the RNG state after model construction against the training record (six runs); a forward
   pass, loss and backward step on a synthetic point cloud; the VRG; the shock-width metric on an analytic profile;
   strict loading of the six checkpoints; the dataset hash and the 96 / 12 / 12 split.

3. **Re-evaluate the released checkpoints** with the unchanged evaluator (validation set; GPU recommended):

   ```bash
   python -B evaluation/evaluate_released.py --device cuda
   ```

   Outputs go to `results/revision_2026/pointwno_v2_shock_benchmark_r1/reproduced/` together with `comparison.csv`
   (reproduced vs official value of every metric). The evaluator first recomputes the validation loss and requires a
   relative difference ≤ 1e-4 from the stored value. On a GPU other than the training GPU this check can fail by a small
   margin (see *Reproduced results*); `--reload_tol 1e-3` repeats the evaluation with that tolerance into a separate,
   labelled directory.

4. **Training** (optional; 36–39 min per model on an H100 NVL, peak GPU memory about 17 GB at batch 4). The trainer refuses any code,
   configuration or data file whose SHA-256 differs from the registered one:

   ```bash
   python -B training/train_released.py \
       --config code/revision_2026/pointwno_v2_shock_benchmark_r1/configs/naca/BM_NACA_V2J2_s0.json \
       --npz data/naca/Data_allpoints_xymomcp_from_dat.npz \
       --out_root results/revision_2026/pointwno_v2_shock_benchmark_r1 --device cuda
   ```

   `training/train_released.py` runs the unchanged trainer `code/revision_2026/pointwno_v2_shock_benchmark_r1/naca/train_naca_bm.py`
   with the same arguments. Add `--dry_run` to run all checks, load the data and build the model without training or
   writing files; it prints `rng_state_sha256_after_model`, which equals the value in the corresponding
   `checkpoints/naca/*/run_meta.json` (verified for all six runs).

   What was verified for training (2026-11-16): the dry run for all six configurations; and 2-epoch smoke runs
   (`configs/naca/BM_NACA_SMOKE_{V2J2,FNOM}_s0.json`) on an RTX 4060 Ti, which reproduced the losses, the per-case
   validation errors and the best weights of smoke runs of the original research code on the same GPU
   (`results/release_verification_20261116/`; both models identical). Not verified: a full 500-epoch run, and training on the
   training GPU. Retraining on other hardware is not expected to be bit-identical to the released checkpoints (cuDNN is
   used non-deterministically, `cudnn.deterministic = False`, as in the original runs). The sub-band WNO training runs
   of the manuscript (capacity, wavelet-family studies) cannot be run with this release (`wno2d_r1.py` is omitted).

5. **Use a model in Python:**

   ```python
   from pointwno import build_model, load_checkpoint, count_params, make_vrg
   model = load_checkpoint(build_model("V2J2", seed=0), "checkpoints/naca/BM_NACA_V2J2_s0/pcwno_best.pt")
   print(count_params(model))   # 21801731
   ```

## Results (validation set, 12 airfoils)

Manuscript shock benchmark, airfoil family; seed mean (s.d.) over three seeds; Δ = PointWNO − PointFNO, same seed,
with the 95 % t interval (df = 2). P1: shock-region pressure NRMSE; P2: shock-width error |ln(w_pred / w_CFD)|; Global:
mean NRMSE over ρu, ρv, C_p. **Validation results**: the same 12 airfoils were used for checkpoint selection, so the
absolute errors are optimistically biased. The 12 test airfoils were not evaluated.

| Metric | PointWNO (`V2J2`) | PointFNO (`FNOM`) | Δ [95 % interval] | PointWNO lower |
|---|---|---|---|---|
| Parameters | 21,801,731 | 21,801,731 | | |
| Global NRMSE (×10⁻²) | 3.339 (0.051) | 4.503 (0.150) | −1.163 [−1.482, −0.844] | 3/3 seeds |
| P1 (×10⁻²) | 3.222 (0.238) | 3.880 (0.345) | −0.658 [−1.065, −0.251] | 3/3 seeds |
| P2 | 0.242 (0.012) | 0.346 (0.029) | −0.104 [−0.154, −0.054] | 3/3 seeds |

These are the official records in `results/revision_2026/pointwno_v2_shock_benchmark_r1/collect/` (evaluation of the
released checkpoints on the training GPU). Their re-evaluation with this repository is described below.

**Re-evaluation with this repository** (2026-11-16, NVIDIA RTX 4060 Ti, torch 2.8.0, in a fresh copy containing only
the repository files, the release assets and the downloaded upstream WNO file; record in
`results/release_verification_20261116/`):

* With the evaluator's frozen reload tolerance (1e-4), **2 of the 6 checkpoints pass** on this GPU; the other 4 differ
  from the stored validation loss by 1.15e-4 to 2.34e-4 (relative; 0 on the training GPU) and are not evaluated further.
* With `--reload_tol 1e-3`, all 6 pass. Relative differences from the official values: validation loss ≤ 2.3e-4,
  global NRMSE ≤ 1.1e-4, P1 ≤ 5.0e-4, P2 ≤ 2.5e-3; case and failure counts are identical, and the three-seed means in
  the table above are reproduced to the digits shown (the s.d. differ by at most 0.001). Secondary surface metrics
  differ by up to 3.2 %; the shock-location error of PointFNO seed 2 differs by 18.9 % (0.0300 vs 0.0243) because the
  steepest-rise sample of one airfoil (case 74) moves.
* The values are identical, row by row, to a re-evaluation with the original research files, and on the CPU the released
  code gives bit-identical predictions, losses and gradients to the original research code for all six checkpoints
  (single-threaded).

The reported results are therefore reproduced from the released checkpoints up to GPU-dependent floating-point
differences; they are not bit-identical to the official records, and the frozen 1e-4 reload check fails for 4 of 6
checkpoints on a GPU other than the training GPU.

## Limitations

* The comparisons use 12 validation airfoils (also used for checkpoint selection) and three seeds per model, without
  correction for multiple comparisons; an undetected difference does not establish equivalence.
* The wavelet kernel is **not better on every shock benchmark**. On RAE2822 and on the nozzle plume the primary shock
  errors of PointWNO and PointFNO did not differ detectably, and on unseen 3-D wings the Fourier model remained more
  accurate (manuscript Sections 5 and 6).
* The airfoil family contains weak shocks only, one flow condition (M∞ 0.65, α 3°) and one mesh generator; accuracy
  depends on the input point distribution, and invariance under independent remeshing was not established.
* The 2-D VRG is fixed and includes nodes inside the airfoil; LPE and FP use brute-force neighbour searches.

## Citation

The article is under revision and has no DOI yet. Please cite it as

```
H.-g. Jeon, S. Choi, A Point-Cloud-Based Wavelet Neural Operator Framework for Aerodynamic Field Prediction on
Variable Geometries, Engineering Applications of Artificial Intelligence, under revision (manuscript EAAI-26-17015).
```

(`CITATION.cff` contains the same information.) This entry will be updated with the publication details if the
article is accepted.

## License

**Not yet specified.** The license of the code, data and models will be set by the copyright holders. Until a
license file is added, no reuse rights are granted beyond viewing. Part of the code derives from the official WNO
repository, which declares no license (see `THIRD_PARTY_NOTICES.md`).

Third-party code and origins: `THIRD_PARTY_NOTICES.md`.

## Contact

Corresponding author: Seongim Choi (GIST).
