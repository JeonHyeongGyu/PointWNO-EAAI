# Data

## Airfoil family (NACA 4-digit, 2-D) — generated in this work

| Item | Value |
|---|---|
| File | `data/naca/Data_allpoints_xymomcp_from_dat.npz` (69.5 MB) |
| SHA-256 | `af48dba038e033d62ecb3bd0dd4cbea32f3ccbcf37d3f36f0a3848e19fce8963` |
| Download | release asset of [v1.0.0](https://github.com/JeonHyeongGyu/PointWNO-EAAI/releases/tag/v1.0.0): `python -B tools/fetch_release_assets.py --only data` (downloads, checks the SHA-256 and places the file) |
| Cases | 120 NACA 4-digit airfoils, one flow condition each |
| Solver | SU2, RANS, Spalart–Allmaras; M∞ = 0.65, Re = 1.5 × 10⁷, α = 3° (manuscript Section 3) |
| Mesh | unstructured O-type mesh generated for each geometry (19,256–20,377 vertices); only points inside the window −0.5 ≤ x ≤ 1.5, −0.7 ≤ y ≤ 0.7 (chord-normalized) are stored |
| Points per case | 15,652–16,749 (mean 16,417.8; 1,970,132 in total) |

The file is the exact dataset used for every airfoil-family result of the manuscript; the training and evaluation code
refuses any file with a different SHA-256. After downloading, place it at the path above and check
`sha256sum data/naca/Data_allpoints_xymomcp_from_dat.npz`.

### Format

`numpy.load(path, allow_pickle=True)` — the per-case arrays have different lengths and are stored as a NumPy object
array, so `allow_pickle=True` is required (only load a file whose SHA-256 matches the value above).

| Key | Type / shape | Content |
|---|---|---|
| `data` | object array (120,), element *i*: float64 (N_i, 5) | point-wise values of case *i* |
| `columns` | object array (5,) | `x`, `y`, `x_momentum`, `y_momentum`, `pressure_coefficient` |
| `filenames` | object array (120,) | case identifiers (e.g. `crop_naca_005010.dat`) |
| `n_points` | int64 (120,) | N_i |

* The case index *i* (0–119) is the identifier used by the splits, the configurations and all evaluation records.
* `x`, `y`: coordinates normalized by the chord (range −0.5…1.5, −0.7…0.7). The model input is `[x, y, 0]`; the third
  coordinate is zero so that the interface is the same as for 3-D data.
* `x_momentum`, `y_momentum`: the momentum components ρu, ρv as stored in the SU2 output (stored values range
  −36.8…311.3 and −139.1…257.8). Their dimensional unit is not documented in the local records and is not restated here.
* `pressure_coefficient`: C_p = (p − p∞) / (½ ρ∞ U∞²).
* Filenames: the six digits are consistent with camber (%), camber location (%) and thickness (%) of the NACA
  4-digit section (e.g. `304912`: 3 %, 49 %, 12 %), which matches the sampled ranges (camber 0–6 %, location 40–60 %,
  thickness 9–14 %). This reading has not been checked against the generation scripts, which are not part of this release.

### Split and normalization

* `splits/2d_orig_rng0_forced60_118.json`: 96 training / 12 validation / 12 test cases. Rule (manuscript script
  `make_split`): `numpy.random.default_rng(0)` over the cases except 60 and 118, which are forced into the test set.
  The same indices are stored in `code/revision_2026/2d_fno_cmp_r1/configs/fno_registry.json` and in every run
  configuration; the loader checks them.
* `data/naca/normalization_stats.json`: min–max statistics of the training split, written by
  `tools/export_normalization_stats.py` with the unchanged loader: `a_norm = (a − min) / (max − min + 1e-12)`.
* **Test set.** The 12 test airfoils are overwritten with NaN / mask 0 right after loading by both the training and the
  evaluation code. No result in the manuscript or in this repository uses them, and no test result is reported here.

### Virtual reference grid (VRG)

The 2-D VRG has 128 × 96 nodes of a uniform grid in the normalized domain [0, 1]², identical for every airfoil (nodes
inside the airfoil are kept). It is not read from a file: `make_internal_vrg(128, 96)` in
`code/third_party/pointwno2d_original/pcwno_vrg_lpe_torch_airfoil_momentum_fno_same.py` builds it, and
`pointwno.make_vrg()` returns it. In the latent tensor the nodes are ordered by radial shell and angle (`PCWNO.forward`).

## Other datasets of the manuscript (not redistributed here)

| Dataset | Source | Status in this repository |
|---|---|---|
| SuperWing (3-D) | Hugging Face `yunplus/SuperWing`, CC BY-SA 4.0, DOI 10.57967/hf/8651 (Yang et al., *Scientific Data* 13, 1113, 2026) | not included; obtain from the original source |
| RAE2822 | Catalani et al., *Computers & Fluids* 251 (2023) 105759, DOI 10.1016/j.compfluid.2022.105759; distributed on Hugging Face as "RAE2822 Airfoil Dataset" | not included; obtain from the original source |
| Nozzle plume | generated in-house with SU2 | not included in this release (release not yet approved) |
