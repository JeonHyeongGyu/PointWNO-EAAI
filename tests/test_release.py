"""Release checks of PointWNO-EAAI (CPU; no training, no data download).

  python -m pytest -q tests/test_release.py

Requires the external upstream WNO file (python -B tools/fetch_wno_upstream.py). Tests that need the dataset
(data/naca/) or the checkpoints (checkpoints/naca/*/pcwno_best.pt) are skipped when the files are not present.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MPLBACKEND", "Agg")

import torch  # noqa: E402

import pointwno as PW  # noqa: E402
from pointwno import _deps  # noqa: E402
from pointwno.api import TN, B  # noqa: E402

CKPT = os.path.join(REPO, "checkpoints", "naca")
NPZ = os.path.join(REPO, "data", "naca", "Data_allpoints_xymomcp_from_dat.npz")
RUNS = [(a, s) for a in ("V2J2", "FNOM") for s in (0, 1, 2)]
PARAMS = 21_801_731


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_pinned_dependencies():
    """fno_registry pins, the 2d_fno_cmp_r1 manifest and the wno_variants pin all verify (as in training / evaluation)."""
    TN.verify_dependencies()


@pytest.mark.parametrize("arm,seed", RUNS)
def test_code_is_the_training_code(arm, seed):
    """Every code / config file whose SHA-256 the training run recorded is present byte-identically, except the two
    WNO-derived files that are not redistributed: their recorded value is the documented one and they are absent."""
    meta = json.load(open(os.path.join(CKPT, f"BM_NACA_{arm}_s{seed}", "run_meta.json")))
    prov = meta["provenance"]
    for rel, s in {**prov["code_sha256"], **prov["pinned_sha256"]}.items():
        if rel in _deps.SUBSTITUTED:
            assert s == _deps.SUBSTITUTED[rel] and not os.path.exists(os.path.join(REPO, rel)), rel
        else:
            assert sha(os.path.join(REPO, rel)) == s, rel
    assert prov["config"] == PW.run_config(arm, seed)
    assert prov["npz_sha256"] == B.NACA_NPZ_SHA256


def test_external_wno_dependency():
    """The upstream file is the unmodified commit-fa6b5a7 file, and it is the module the model code imports."""
    import wavelet_convolution_v3 as WC
    assert sha(_deps.WNO_PATH) == _deps.WNO_SHA256
    assert os.path.realpath(WC.__file__) == os.path.realpath(_deps.WNO_PATH)
    assert sys.modules["pcwno_vrg_lpe_torch_airfoil_momentum_fno_same"].WaveConv2d is WC.WaveConv2d


def test_wno2d_r1_placeholder():
    W = sys.modules["wno2d_r1"]
    assert W.__release_placeholder__ is True
    with pytest.raises(_deps.DependencyError):
        W.convert_to_r1


@pytest.mark.parametrize("arm,seed", RUNS)
def test_rng_state_after_construction_matches_training(arm, seed):
    PW.build_model(arm, seed)
    got = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    meta = json.load(open(os.path.join(CKPT, f"BM_NACA_{arm}_s{seed}", "run_meta.json")))
    assert got == meta["rng_state_sha256_after_model"]


def _nontrivial(text):
    import re
    out = []
    for x in text.replace("\r\n", "\n").split("\n"):
        y = re.sub(r"\s", "", x)
        if len(y) > 20 and not y.startswith(("#", "import", "from", "\"\"\"")):
            out.append(y)
    return out


def test_no_upstream_wno_code_redistributed():
    """No file of the repository reproduces lines of the upstream WNO file (whitespace ignored), beyond <= 3 generic lines."""
    up = set(_nontrivial(open(_deps.WNO_PATH).read()))
    worst = {}
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in (".git", "external", "__pycache__", ".pytest_cache")]
        for fn in files:
            if fn.endswith((".py", ".md", ".txt", ".json", ".cff")):
                p = os.path.join(root, fn)
                n = sum(1 for x in _nontrivial(open(p, errors="ignore").read()) if x in up)
                if n:
                    worst[os.path.relpath(p, REPO)] = n
    assert max(worst.values(), default=0) <= 3, worst


@pytest.mark.parametrize("arm", ["V2J2", "FNOM"])
def test_parameter_count(arm):
    assert PW.count_params(PW.build_model(arm, 0)) == PARAMS


def _synthetic_batch(n=3000, nmax=3200, k=6, seed=0):
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)).astype(np.float32)
    xyz = np.zeros((1, nmax, 3), np.float32)
    xyz[0, :n, :2] = xy
    xyz[0, n:, :2] = 10.0                                    # padding convention of the loader
    mask = np.zeros((1, nmax), np.float32)
    mask[0, :n] = 1.0
    y = np.zeros((1, nmax, 3), np.float32)
    y[0, :n] = rng.random((n, 3))
    kidx = np.zeros((1, nmax, k), np.int64)
    kidx[0, :n] = cKDTree(xy).query(xy, k=k + 1)[1][:, 1:]
    return [torch.from_numpy(a) for a in (xyz, y, mask, kidx)]


@pytest.mark.parametrize("arm", ["V2J2", "FNOM"])
def test_synthetic_forward_and_loss(arm):
    """Forward pass on a synthetic 3,000-point cloud, masked loss and one backward step."""
    torch.manual_seed(0)
    model = PW.build_model(arm, 0)
    grid = PW.make_vrg("cpu")
    xyz, y, mask, kidx = _synthetic_batch()
    trainer = TN.OS.MaskedGradTrainer(model, lr=1e-3, grad_weight=0.0)
    total, data, _, pred = trainer.eval(xyz, grid, y, mask, kidx)
    assert pred.shape == (1, xyz.shape[1], 3) and np.isfinite(total) and total == pytest.approx(data)
    ref = TN.OS.masked_mse_loss(pred, y, mask)
    assert float(ref) == pytest.approx(total, rel=1e-6)
    loss, _, _ = trainer.step(xyz, grid, y, mask, kidx)
    assert np.isfinite(loss)


def test_vrg_shape():
    g = PW.make_vrg("cpu")
    TN.assert_grid_shape(g, 128, 96)


def test_shock_profile_metrics_identity_and_width():
    """P2 (|ln thickness ratio|) is 0 for a prediction equal to the GT and ln 2 for a shock twice as wide."""
    sys.path.insert(0, os.path.join(PW.api.BM_PKG, "shock"))
    import profile_metrics as PM
    s = np.linspace(0.3, 0.7, 401)
    g = np.tanh((s - 0.5) / 0.005)
    p = np.tanh((s - 0.5) / 0.010)
    m0, _, _ = PM.compare(s, g, g, 0.5)
    m1, _, _ = PM.compare(s, p, g, 0.5)
    assert m0["thickness_abs_log"] == pytest.approx(0.0, abs=1e-12)
    assert m1["thickness_ratio"] == pytest.approx(2.0, rel=0.02)
    assert m1["thickness_abs_log"] == pytest.approx(np.log(2.0), abs=0.02)


@pytest.mark.parametrize("arm,seed", RUNS)
def test_checkpoint_loads_strictly(arm, seed):
    p = os.path.join(CKPT, f"BM_NACA_{arm}_s{seed}", "pcwno_best.pt")
    if not os.path.isfile(p):
        pytest.skip("checkpoint not downloaded")
    comp = json.load(open(os.path.join(CKPT, f"BM_NACA_{arm}_s{seed}", "complete.json")))
    assert sha(p) == comp["best_sha256"]
    PW.load_checkpoint(PW.build_model(arm, seed), p)


def test_split_and_dataset():
    if not os.path.isfile(NPZ):
        pytest.skip("dataset not downloaded")
    assert sha(NPZ) == B.NACA_NPZ_SHA256
    split = json.load(open(os.path.join(REPO, "splits", "2d_orig_rng0_forced60_118.json")))
    reg = B.read_json(B.NACA_FNO_REGISTRY)["split"]
    for k in ("train", "val", "test"):
        assert [int(i) for i in split[k]] == reg[f"{k}_idx"]
    assert (len(reg["train_idx"]), len(reg["val_idx"]), len(reg["test_idx"])) == (96, 12, 12)
    assert {60, 118} <= set(reg["test_idx"])
