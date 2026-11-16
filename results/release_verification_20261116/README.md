# Release verification, 2026-11-16

All re-evaluations here were run on an NVIDIA GeForce RTX 4060 Ti 8 GB, torch 2.8.0+cu128, Python 3.10.18, WSL2. The
official records (`results/.../collect/`) were produced on the training GPU (NVIDIA H100 NVL, torch 2.5.1+cu121). The
files below are new verification records, not results of the training runs.

## 1. Equivalence of the released code with the original research code

The release does not contain the modified WNO file used during training; it uses the unmodified upstream file
(`THIRD_PARTY_NOTICES.md`). `equivalence_check_script.py` loads each of the six checkpoints in (a) the original research
code (modified file, SHA-256 `50b9fcdd…`) and (b) this release (upstream file `5b80065a…`), runs one validation airfoil
(case 119) on the CPU, and hashes the prediction, the masked-MSE loss and every parameter gradient.

| File | Threads | Result |
|---|---|---|
| `equivalence_research_code_cpu1thread.txt` vs `equivalence_release_code_cpu1thread.txt` | 1 | **6/6 bit-identical**: parameter count, prediction, loss and all parameter gradients |
| `equivalence_research_code_cpu8threads.txt` vs `equivalence_release_code_cpu8threads.txt` | 8 | prediction and loss bit-identical 6/6; gradient hashes differ — but they also differ between two runs of the SAME code with 8 threads (non-deterministic multi-threaded CPU backward; largest relative difference per tensor about 1e-6 for V2J2 and 2e-4 for FNOM in small bias gradients, equal in size within and across code bases) |

## 2. Re-evaluation of the six checkpoints (validation set, frozen evaluator `bm_naca_eval_v1`, batch 4)

Two runs, each with two tolerance settings:

* `comparison_reload_tol_1e-4.csv`, `comparison_reload_tol_1e-3.csv` — first local re-evaluation, with the original
  vendored files (before the WNO dependency was separated).
* `independent_copy_comparison_reload_tol_1e-4.csv`, `independent_copy_comparison_reload_tol_1e-3.csv`,
  `independent_copy_evaluation_record_tol_*.json` — after the separation, in a fresh copy containing only the git-tracked
  files, the release assets and the downloaded upstream WNO file, run with `python -I` (no research directory on any
  import path; every loaded project module came from the copy).

Both runs give **identical values in every row** (30/30 rows at 1e-4, 78/78 at 1e-3).

| Check | Result |
|---|---|
| Frozen reload tolerance 1e-4 | **2/6 PASS** (V2J2_s0, FNOM_s1). FAIL: V2J2_s1 1.15e-4, V2J2_s2 1.24e-4, FNOM_s0 1.86e-4, FNOM_s2 2.34e-4 relative difference of the recomputed validation loss (0 on the training GPU). A failed run writes no metrics. |
| `--reload_tol 1e-3` (separate output directory, labelled) | 6/6 PASS |
| Largest relative difference from the official values (6 runs, 1e-3 run) | validation loss 2.3e-4, global NRMSE 1.1e-4, P1 5.0e-4, P2 2.5e-3 |
| Case counts, shock-case counts, failure counts | identical in all six runs |
| Three-seed means (×10⁻² for global NRMSE and P1) | V2J2 3.339 / 3.222 / 0.242, FNOM 4.503 / 3.880 / 0.346 — identical to three decimals; the s.d. differ by at most 0.001 |
| Secondary surface metrics | up to 3.2 % (pre-undershoot) |
| Shock-location error | FNOM_s2: 0.02996 (official) vs 0.02429 (this GPU), 18.9 %: in airfoil 74 the steepest-rise sample moves from 0.0595 to 0.0198 chord. All other runs: 0. |

## 3. Training path

* `training/train_released.py --dry_run` in the fresh copy, all six configurations: parameter count 21,801,731 and
  `rng_state_sha256_after_model` equal to the training records (6/6).
* 2-epoch smoke runs (`BM_NACA_SMOKE_{V2J2,FNOM}_s0.json`, the registered smoke configurations) with
  `training/train_released.py` in the fresh copy, on the RTX 4060 Ti, compared with smoke runs of the original research
  code on the same GPU (made earlier, 2026-11-12): every loss column of `metrics.csv` (timing and memory excluded),
  `val_case_RMSE.csv` and the best weights `pcwno_best.pt` are **identical** (V2J2 sha256 `e700698a…`, FNOM `64799753…`;
  `smoke_*` files, `smoke_best_weights_sha256.txt`). About 14 min per epoch on this 8 GB GPU (memory spill).
* Not run: a full 500-epoch training, and any training on the training GPU (H100). Agreement on a different GPU, or
  over 500 epochs, is not established by these checks.
