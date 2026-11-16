# Code map

The research code keeps its original directory layout (`code/...`, `RAE2822/code/...`) because every file is pinned by
SHA-256 relative to the repository root and the training records reference these paths. `pointwno/` is a thin public
interface; it contains no model code.

| Path | Role |
|---|---|
| `pointwno/api.py` | `build_model(arm, seed)`, `load_checkpoint`, `count_params`, `make_vrg`, `run_config` |
| `pointwno/_deps.py` | release bootstrap: external upstream WNO file (path, commit, SHA-256), `wno2d_r1` placeholder, pin policy for the two omitted files |
| `external/` | destination of the downloaded upstream WNO file (not in version control; only its README is) |
| `code/revision_2026/pointwno_v2_shock_benchmark_r1/bm_common.py` | paths, SHA-256 pins, run ids, statistics / verdict rule |
| `.../configs/benchmark_registry.json`, `eval_protocol.json` | registered models, parameter counts, pre-specified pairs and metrics |
| `.../configs/gt_shock_cases.json` | frozen ground-truth shock cases, anchors and shock regions (from CFD only) |
| `.../configs/naca/BM_NACA_{V2J2,FNOM}_s{0,1,2}.json` | the six run configurations (and two 2-epoch smoke configurations) |
| `.../naca/train_naca_bm.py` | trainer: seeding, data, model build (`build_model`), Adam / StepLR loop, best-by-validation, atomic resume |
| `.../naca/eval_naca_bm.py` | frozen validation evaluator (`bm_naca_eval_v1`): reload check, NRMSE, P1, P2, surface metrics, failure counts |
| `.../shock/profile_metrics.py` | 1-D surface shock-profile metrics (10–90 % width → P2, location, jump, gradient, overshoot, TV, oscillation) |
| `.../collect/collect_bm.py` | 3-seed paired comparison and claim rule (consistently lower = all 3 seeds and 95 % t interval excluding 0) |
| `.../audit/gt_shock_cases.py` | script that froze `gt_shock_cases.json` (uses CFD data only) |
| `code/revision_2026/nozzle_point_wno_variant_r1/wno_variants.py` | `V2Conv`: canonical WNO V2 kernel (exact-adjoint DWT) replacing the layers of the WNO grid network |
| `code/revision_2026/3d_recovery/r3/wavelet_exact.py` | `DWTExact` / `IDWTExact`: pytorch_wavelets forward, exact adjoint backward, mode symmetric |
| `code/revision_2026/2d_fno_cmp_r1/train_fno2d.py` | `build_model` of PointFNO (C3 construction, then `FNO2dGridCore` drawn inside `fork_rng(seed + 1000)`) |
| `code/revision_2026/2d_original_r1/train_orig2d.py` | shared runner helpers (`block_test`, numerics, targets) |
| `code/revision_2026/2d_original_r1/vendored/pcwno_..._masked_steplr.py` | manuscript data pipeline: `load_allpoints_npz`, `make_split`, `AllPointsMaskedDataset`, `masked_mse_loss`, `MaskedGradTrainer` |
| `code/third_party/pointwno2d_original/pcwno_vrg_lpe_torch_airfoil_momentum_fno_same.py` | `PCWNO`, `SA_MSG` (LPE), `WNO2d`, `FP_IDW`, `PointHead`, `make_internal_vrg` (VRG) |
| `external/csccm-iitd_WNO_fa6b5a7/wavelet_convolution_v3.py` (downloaded) | upstream `WaveConv2d`; only its constructor runs (layers are replaced) — see `THIRD_PARTY_NOTICES.md` |
| `code/pwno2d/models.py` | `FNO2dGridCore`, `SpectralConv2d` (PointFNO core) |
| `code/revision_2026/2d_fno_cmp_r1/configs/fno_registry.json` | protocol configuration, split, pinned SHA-256 of the model files |
| `code/revision_2026/2d_capacity*_r1/configs/.../CAP_C3_L6W32*.json` | pinned configuration of the 6 × 32 construction shared by both models |
| `RAE2822/code/revision_2026/rae2822_pure_r1/rae_shock.py` | surface shock detector (`detect`) used by the evaluator for shock / no-shock classification |
| `evaluation/evaluate_released.py` | re-evaluation of the released checkpoints with the unchanged evaluator + comparison with the official records |
| `training/train_released.py` | launcher of the unchanged trainer `train_naca_bm.py` (after `pointwno._deps.activate()`) |
| `tools/fetch_wno_upstream.py` | download + SHA-256 / git-blob verification of the upstream WNO file |
| `tools/fetch_release_assets.py` | download + SHA-256 verification of the release assets (dataset, checkpoints) |
| `tools/export_normalization_stats.py`, `tools/verify_release.py`, `tools/make_release_manifest.py` | normalization statistics; release manifest check / generation |
| `tests/test_release.py` | CPU release checks |

Files kept only because the pinned manifest `code/revision_2026/2d_fno_cmp_r1/SHA256SUMS_fno.txt` lists them (not needed
to run anything here): `collect_fno.py`, `pbs/*.pbs` (batch scripts of the training cluster), `tests/check_fno_static.py`.

Model construction order (both models, from `train_naca_bm.build_model`): the sub-band WNO model `PCWNO` of the
capacity study (6 × 32) is constructed with the global RNG seeded by the run seed, so that LPE, FP and head receive
identical initial values in both models; then its six wavelet layers are replaced by `V2Conv` drawn from a dedicated
generator (seed 20261110 + seed) for PointWNO, or the grid network is replaced by `FNO2dGridCore` drawn inside
`fork_rng(seed + 1000)` for PointFNO.

Not part of this release (present in the research code, omitted here): `code/third_party/pointwno2d_original/wavelet_convolution_v3.py`
(modified copy of the upstream WNO file), `code/revision_2026/2d_recovery/r1_exact_backward/wno2d_r1.py` (exact-backward
sub-band WNO layer, WNO-derived; replaced by a placeholder module) and `code/pwno2d_ext/` (imported only by `wno2d_r1.py`).
