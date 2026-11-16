# pointwno_v2_shock_benchmark_r1 — Canonical PointWNO V2 vs PointFNO on NACA / RAE2822 / nozzle (validation only)

Hypothesis (no direction assumed): does the canonical WNO V2 kernel (official WNO Version 2.0.0) give PointWNO a
shock-capturing behaviour that is distinguishable from an exactly parameter-matched Fourier PointFNO?

| file | role |
|---|---|
| `bm_common.py` | paths, sha256 pins, out-root guard, run ids, statistics / verdict rule |
| `make_registry.py` | generates `configs/benchmark_registry.json`, `configs/eval_protocol.json` and the NACA / RAE run configs (parameter counts counted from the models; `--check` regenerates and compares) |
| `audit/audit_v2_source.py` | canonical V2 source audit (official V2.0.0 equivalence, adjoint, single source) |
| `audit/gt_shock_cases.py` | freezes the GT shock cases / anchors / regions (`configs/gt_shock_cases.json`), GT only |
| `naca/train_naca_bm.py`, `naca/eval_naca_bm.py` | NACA trainer (train_fno2d order + atomic resume) and validation evaluator |
| `rae/train_rae_bm.py`, `rae/eval_rae_bm.py` | RAE trainer (train_point epoch body, 1,000 epochs, resume) and evaluator |
| nozzle | trained by the **unchanged** `nozzle_point_wno_variant_r1/train_var.py` (V2C / V2S configs) into this package's root; evaluated by the unchanged `evaluate_var.py`; `nozzle/nozzle_shock_bm.py` = shock metrics (same as physics_eval_f10m) |
| `shock/profile_metrics.py` | frozen 1-D surface shock metrics (10–90 % width, location, jump, gradient, overshoot, TV, oscillation) |
| `preflight_bm.py`, `smoke_check_bm.py` | read-only preflight incl. FNO reuse audit; smoke gate + GPU-h projection |
| `collect/collect_bm.py`, `collect/check_integrity_bm.py` | 3-seed paired comparison + claim rule; read-only snapshot / completion checker |
| `pbs/bm_KITSM02.pbs`, `launcher/submit_bm.sh` | one-GPU job (smoke / train / eval / collect); dry-run-by-default launcher |
| `server/pack_bm_evidence_readonly.sh` | results packer (no checkpoints) |
| `tests/test_static_bm.py` | T01–T14 |

Outputs: `results/revision_2026/pointwno_v2_shock_benchmark_r1/{naca,rae,nozzle}/{runs,eval}`, `nozzle/shock`, `collect`.
Docs and commands: `docs/revision_2026/pointwno_v2_shock_benchmark_r1_20261111/`.
