"""Release bootstrap: the external WNO dependency and the two research files that are not redistributed.

The research code under ``code/`` is byte-identical to the code that trained the released models, with two files
removed because they derive from the upstream WNO repository, which declares no license:

1. ``code/third_party/pointwno2d_original/wavelet_convolution_v3.py`` — a modified copy of upstream
   ``wavelet_convolution_v3.py`` (commit fa6b5a7). The released models import it and call ``WaveConv2d.__init__`` six
   times while being constructed (global-RNG draws). Every one of these layers is then replaced (by ``V2Conv`` in
   PointWNO, by ``FNO2dGridCore`` in PointFNO), so its ``forward`` is never executed. The local modifications concern
   only ``WaveConv2d.forward`` / ``spectralconv``. Here the UNMODIFIED upstream file is used instead; it was verified to
   give bit-identical state dicts, post-construction RNG states and outputs for all six released models
   (results/release_verification_20261116/).
2. ``code/revision_2026/2d_recovery/r1_exact_backward/wno2d_r1.py`` — exact-backward sub-band WNO layer. It is imported
   at module level by ``train_orig2d.py`` but used only by sub-band WNO (C3 / V3) runs, which are not released. A
   placeholder module that raises on any use is registered under its name.

``activate()`` must run before any research module is imported. It:
  * checks that the upstream file is present and has the pinned SHA-256 (otherwise stops with instructions),
  * puts its directory first on sys.path and checks that ``import wavelet_convolution_v3`` resolves to it,
  * registers the ``wno2d_r1`` placeholder,
  * wraps ``bm_common.verify_pins`` so that the registry pins of the two files are checked against the substitution
    above (upstream SHA-256 / placeholder present) while every other pinned file is still checked unchanged.
"""
from __future__ import annotations

import hashlib
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WNO_REPOSITORY = "https://github.com/csccm-iitd/WNO"
WNO_COMMIT = "fa6b5a79ad97ad5c2f8ee3773fe3c732d344dee3"
WNO_URL = f"https://raw.githubusercontent.com/csccm-iitd/WNO/{WNO_COMMIT}/wavelet_convolution_v3.py"
WNO_SHA256 = "5b80065afdcf2a24ce1b9e60d946132fa4d7a3da54a285bfa2055d32bac0e3b6"
WNO_GIT_BLOB = "9bf0a9426253a8c81bab5342cca7de5fd2c7d8e1"
WNO_DIR = os.path.join(REPO, "external", "csccm-iitd_WNO_fa6b5a7")
WNO_PATH = os.path.join(WNO_DIR, "wavelet_convolution_v3.py")

# Registry pins of the two research files that are not distributed (values recorded by the training runs).
VENDORED_V3_REL = "code/third_party/pointwno2d_original/wavelet_convolution_v3.py"
VENDORED_V3_SHA256 = "50b9fcdde89d63a2c73252e828149fc32b81ebe8e533b22f81125285a702f69b"
WNO2D_R1_REL = "code/revision_2026/2d_recovery/r1_exact_backward/wno2d_r1.py"
WNO2D_R1_SHA256 = "3cb8193d7f1a45950eeaf574e7083ca0348c39421f8cae32da6071a3dcf481a3"
SUBSTITUTED = {VENDORED_V3_REL: VENDORED_V3_SHA256, WNO2D_R1_REL: WNO2D_R1_SHA256}

_ACTIVE = False


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class DependencyError(RuntimeError):
    pass


def check_upstream():
    if not os.path.isfile(WNO_PATH):
        raise DependencyError(
            f"external WNO file missing: {os.path.relpath(WNO_PATH, REPO)}\n"
            f"  It is not distributed with this repository (the upstream repository declares no license).\n"
            f"  Run:  python -B tools/fetch_wno_upstream.py   (downloads {WNO_REPOSITORY} @ {WNO_COMMIT[:7]})")
    got = sha256_file(WNO_PATH)
    if got != WNO_SHA256:
        raise DependencyError(f"{os.path.relpath(WNO_PATH, REPO)}: sha256 {got} != {WNO_SHA256} (expected the unmodified "
                              f"upstream file of commit {WNO_COMMIT[:7]}); re-run tools/fetch_wno_upstream.py")
    for rel in SUBSTITUTED:
        if os.path.exists(os.path.join(REPO, rel)):
            raise DependencyError(f"{rel} must not exist in this release (not redistributed); remove it")


def _placeholder_wno2d_r1():
    m = types.ModuleType("wno2d_r1", "Placeholder: the exact-backward sub-band WNO layer (wno2d_r1.py) is not part of "
                                     "this release; it is used only by sub-band WNO (C3 / V3) runs, which are not released.")
    m.__file__ = None
    m.__release_placeholder__ = True

    def __getattr__(name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise DependencyError(f"wno2d_r1.{name}: the sub-band WNO r1 layer is not part of this release "
                              f"(only PointWNO V2J2 and PointFNO FNOM are released)")
    m.__getattr__ = __getattr__
    return m


def activate():
    """Idempotent. Call before importing bm_common / train_naca_bm / eval_naca_bm."""
    global _ACTIVE
    if _ACTIVE:
        return
    check_upstream()
    if "wavelet_convolution_v3" in sys.modules and os.path.realpath(sys.modules["wavelet_convolution_v3"].__file__) != os.path.realpath(WNO_PATH):
        raise DependencyError("another wavelet_convolution_v3 was imported before pointwno._deps.activate()")
    if WNO_DIR not in sys.path:
        sys.path.insert(0, WNO_DIR)
    sys.modules.setdefault("wno2d_r1", _placeholder_wno2d_r1())
    os.environ.setdefault("MPLBACKEND", "Agg")
    bm = os.path.join(REPO, "code", "revision_2026", "pointwno_v2_shock_benchmark_r1")
    if bm not in sys.path:
        sys.path.insert(0, bm)
    import bm_common as B
    orig = B.verify_pins

    def verify_pins_release(pins: dict, base):
        rest = {k: v for k, v in pins.items() if k not in SUBSTITUTED}
        for k, v in pins.items():
            if k in SUBSTITUTED and v != SUBSTITUTED[k]:
                raise RuntimeError(f"registry pin of {k} is {v}, not the recorded training value {SUBSTITUTED[k]}")
        check_upstream()
        if not getattr(sys.modules.get("wno2d_r1"), "__release_placeholder__", False):
            raise RuntimeError("wno2d_r1 is not the release placeholder")
        return orig(rest, base)

    B.verify_pins = verify_pins_release
    _ACTIVE = True
    import wavelet_convolution_v3 as WC
    if os.path.realpath(WC.__file__) != os.path.realpath(WNO_PATH):
        raise DependencyError(f"wavelet_convolution_v3 resolved to {WC.__file__}, not {WNO_PATH}")


def record() -> dict:
    """What was substituted, for logs and output records."""
    return {"wno_upstream": {"repository": WNO_REPOSITORY, "commit": WNO_COMMIT, "path": "wavelet_convolution_v3.py",
                             "sha256": WNO_SHA256, "used_for": VENDORED_V3_REL,
                             "training_used_modified_copy_sha256": VENDORED_V3_SHA256},
            "wno2d_r1": {"status": "placeholder (not released; unused by V2J2 / FNOM)", "training_sha256": WNO2D_R1_SHA256}}
