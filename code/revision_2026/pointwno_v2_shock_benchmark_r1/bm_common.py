"""Shared definitions of the Canonical PointWNO V2 vs PointFNO shock benchmark (package pointwno_v2_shock_benchmark_r1).

Question (pre-registered, configs/eval_protocol.json): does the canonical WNO V2 kernel (official csccm-iitd WNO
Version 2.0.0 WaveConv2d: DWT -> per-coefficient real channel mixing of LL_J and the three level-J detail bands,
finer details = 0 -> IDWT) give PointWNO a shock-capturing behaviour that is distinguishable from a Fourier PointFNO,
all else equal, on NACA (M 0.65, transonic, variable geometry), RAE2822 and the nozzle plume?

Matching rule (all three datasets): the FNO comparator of a V2 arm has Fourier modes (m1, m2) equal to the V2
coefficient grid (X_J, Y_J). Then both cores hold exactly 4 * width^2 * X_J * Y_J real kernel weights per layer and
every other module is identical -> exactly equal trainable parameter counts (checked in code, never assumed).
Existing FNO runs are reused only if every protocol item matches (audit/fno_reuse_audit.py); the HPO-selected /
incumbent FNO of each dataset is always reported as a context comparator, never dropped.

This module holds paths, sha256 pins, the run registry access, the statistics rule and small I/O helpers only.
Nothing here trains, evaluates or touches test data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REV = os.path.dirname(HERE)                          # code/revision_2026
CODE = os.path.dirname(REV)                          # code
EAAI = os.path.dirname(CODE)                         # Journal/EAAI
PACKAGE = "pointwno_v2_shock_benchmark_r1"
BM_VERSION = "pointwno_v2_shock_benchmark_r1_v1"
OUT_ROOT_DEFAULT = os.path.join(EAAI, "results", "revision_2026", PACKAGE)
REGISTRY = os.path.join(HERE, "configs", "benchmark_registry.json")
PROTOCOL = os.path.join(HERE, "configs", "eval_protocol.json")
GT_CASES = os.path.join(HERE, "configs", "gt_shock_cases.json")
SEEDS = (0, 1, 2)
V2_INIT_SEED_BASE = 20261110        # = nozzle_point_wno_variant_r1 / wno_local_singlecase_r1 (same V2 init rule)

# ----------------------------------------------------------------------------------- read-only dependencies
VAR_PKG = os.path.join(REV, "nozzle_point_wno_variant_r1")
VAR_MANIFEST_SHA256 = "6c7f17435dc9c4494590ff5d8a04f23ec36055a585cbb61518de8c2dd192ff45"
ABL_PKG = os.path.join(REV, "nozzle_point_ablation_r1_v2")
ABL_MANIFEST_SHA256 = "e0c4090658f2854ac6eb2f0eb8bf3934025517c1b34b9feb54de2b1087a0deb1"
PA_PKG = os.path.join(REV, "nozzle_point_physics_audit_r1")
PA_MANIFEST_SHA256 = "1e0a89fa03de900fb9efc538729cb171e7a41137c95b3d7aaf76077a67638867"
WNO_VARIANTS = os.path.join(VAR_PKG, "wno_variants.py")
WNO_VARIANTS_SHA256 = "5a8c8a3c24909441f40276d40db2f8a705ae63ec58aaceed425b1b17f990c4f8"
OFFICIAL_V2_REF = os.path.join(VAR_PKG, "official_ref", "official_V2.0.0_wavelet_convolution.py")
OFFICIAL_V2_REF_SHA256 = "6f7b9a4cba243f967c3679f358032dac3db58926e1508e5c7c9d46112762c701"

RAE_ROOT = os.path.join(EAAI, "RAE2822")
RAE_CODE = os.path.join(RAE_ROOT, "code", "revision_2026")
RAE_DATA_DEFAULT = os.path.join(RAE_ROOT, "data_derived", "rae2822_grid256_r1")
RAE_DATA_SUMS_SHA256 = "e2641ce7fc8a9f1d0f2ba81740cda76bba3161cbdd9f72e4c9c9bb0db705688d"
RAE_POINT_PKG = os.path.join(RAE_CODE, "rae2822_point_r1")
RAE_POINT_MANIFEST_SHA256 = "7c466f31fefe7bc8edf46033689ab3a6112dd20a84c55f70158229aa4c933ede"   # paths relative to RAE_CODE
RAE_EVAL_PKG = os.path.join(RAE_CODE, "rae2822_point_eval_local_r2")
RAE_EVAL_MANIFEST_SHA256 = "f93719e6e557b4bb28bcb9b0a219c78071503dfba7015c5c109d8859d64eb40f"    # paths relative to RAE_CODE
RAE_PURE_PINS = {
    "rae2822_pure_r1/train_pure.py": "d57836e9f7abd5d46a3462621a4e27ec84e9b19efd1a2ec01e6d28dc6b024926",
    "rae2822_pure_r1/rae_data.py": "14f66b99c787e3af536c315ab73d74247c3564c11601b0cefe59a5aca9754736",
    "rae2822_pure_r1/rae_metrics.py": "34913de430d7e292c8a39a8480a9761dcf7df3817ef5fff0b8acb1529e3e3ce7",
    "rae2822_pure_r1/rae_shock.py": "30499b677d7c8f283b080daba4ecc7e4116410e4fd751cb8620156e76c00cde8",
}

NACA_FNO_PKG = os.path.join(REV, "2d_fno_cmp_r1")
NACA_FNO_MANIFEST_SHA256 = "ad36c2d89551f39c34e62c9f76df713cfe9217a22884bcb6f06b00db57e06030"   # SHA256SUMS_fno.txt, EAAI-rooted
NACA_FNO_REGISTRY = os.path.join(NACA_FNO_PKG, "configs", "fno_registry.json")
NACA_FNO_REGISTRY_SHA256 = "8014affd3fbefb74812f1cce6ff19b310e56038a7f38000157e5048bc1779f7b"
NACA_NPZ_SHA256 = "af48dba038e033d62ecb3bd0dd4cbea32f3ccbcf37d3f36f0a3848e19fce8963"
NACA_MINF = 0.65                      # docs/A1_PREPARATION.md 6.2: all 120 cases M_inf = 0.6500 (inverted from p, Cp)

NOZZLE_GRID_REL = "data_derived/nozzle_grid_r1/nozzle_grid_1024x128.npz"
NOZZLE_GRID_SHA256 = "803aad87033e3536e535deb4d6ef1c9595ad10fb08a00fc28cb771bd49608c78"
NOZZLE_VAR_OUT_REL = "results/revision_2026/nozzle_point_wno_variant_r1"      # predecessor root (never written here)
NOZZLE_HPO_OUT_REL = "results/revision_2026/nozzle_point_hpo_r1"               # D8C (read-only)
NOZZLE_F10M_OUT_REL = "results/revision_2026/nozzle_f10_matched_completion_r1"  # F10 seeds 1/2 (read-only)

# Directories this package may read but never write into (out-root guard).
PROTECTED_ROOTS_REL = [
    "results/fno2d_cmp_r1", "results/capacity_r1", "results/capacity_r1_seeds", "results/orig2d_r1",
    "RAE2822/results", "results/revision_2026/nozzle_point_r1", "results/revision_2026/nozzle_point_convergence_r1",
    "results/revision_2026/nozzle_point_capacity_r1", NOZZLE_HPO_OUT_REL, NOZZLE_F10M_OUT_REL, NOZZLE_VAR_OUT_REL,
    "results/revision_2026/nozzle_point_ablation_r1_v2", "results/revision_2026/nozzle_point_hpo_f10ms_r1",
    "code", "data_derived", "RAE2822/data_derived", "RAE2822/code",
]


# ----------------------------------------------------------------------------------- small helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes):
    return hashlib.sha256(b).hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, **kw):
    kw.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
    kw.setdefault("pbs_job", os.environ.get("PBS_JOBID"))
    with open(path, "a") as f:
        f.write(json.dumps(kw, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def inside(p, root):
    p, root = os.path.realpath(p), os.path.realpath(root)
    return p == root or p.startswith(root + os.sep)


def assert_out_root_safe(out_root, eaai=EAAI):
    """The benchmark writes only below its own results root."""
    own = os.path.join(eaai, "results", "revision_2026", PACKAGE)
    if not inside(out_root, own):
        raise SystemExit(f"STOP out_root {out_root} is not below {own}")
    for rel in PROTECTED_ROOTS_REL:
        p = os.path.join(eaai, rel)
        if inside(out_root, p) or inside(p, out_root):
            raise SystemExit(f"STOP out_root {out_root} overlaps protected directory {p}")


def verify_manifest(pkg_dir, want_sha, tag, base=None):
    """Manifest file sha256 == want and every listed file matches. base = directory the listed paths are relative to."""
    man = os.path.join(pkg_dir, "MANIFEST_SHA256.txt") if os.path.isdir(pkg_dir) else pkg_dir
    got = sha256_file(man)
    if got != want_sha:
        raise RuntimeError(f"{tag}=FAIL manifest {man} sha256 {got} != {want_sha}")
    base = base or os.path.dirname(man)
    bad = []
    for line in open(man):
        if not line.strip():
            continue
        s, rel = line.rstrip("\n").split("  ", 1)
        p = os.path.join(base, rel)
        if not os.path.isfile(p) or sha256_file(p) != s:
            bad.append(rel)
    if bad:
        raise RuntimeError(f"{tag}=FAIL files differ from the manifest: {bad}")
    return got


def verify_pins(pins: dict, base):
    bad = [rel for rel, s in pins.items() if not os.path.isfile(os.path.join(base, rel)) or sha256_file(os.path.join(base, rel)) != s]
    if bad:
        raise RuntimeError(f"pinned files differ: {bad}")


def add_path(*ps):
    for p in ps:
        if p not in sys.path:
            sys.path.insert(0, p)


def load_wno_variants():
    """wno_variants.py of the variant package, sha256-pinned (the SAME file the RAE single-case V2 arms imported)."""
    if sha256_file(WNO_VARIANTS) != WNO_VARIANTS_SHA256:
        raise RuntimeError("wno_variants.py differs from the pinned sha256")
    add_path(os.path.join(REV, "3d_recovery", "r3"), VAR_PKG)
    import wno_variants as WV  # noqa: E402
    return WV


def registry():
    return read_json(REGISTRY)


def protocol():
    return read_json(PROTOCOL)


def registry_sha():
    return sha256_file(REGISTRY)


def protocol_sha():
    return sha256_file(PROTOCOL)


def manifest_sha():
    return sha256_file(os.path.join(HERE, "MANIFEST_SHA256.txt"))


def arm(dataset, arm_id):
    for a in registry()["datasets"][dataset]["arms"]:
        if a["arm"] == arm_id:
            return a
    raise KeyError((dataset, arm_id))


def model_for_seed(arm_model, seed):
    """Registry arm model block -> the run's model block (V2: init generator seed = 20261110 + seed)."""
    m = json.loads(json.dumps(arm_model))
    if m.get("kernel") == "v2":
        m["init_generator_seed"] = V2_INIT_SEED_BASE + seed
    return m


def run_id(dataset, arm_id, seed, smoke=False):
    return f"BM_{dataset.upper()}_{'SMOKE_' if smoke else ''}{arm_id}_s{seed}"


def run_config_path(dataset, arm_id, seed):
    return os.path.join(HERE, "configs", dataset, f"{run_id(dataset, arm_id, seed)}.json")


# ----------------------------------------------------------------------------------- run status (NACA / RAE runners)
def read_status(run_dir):
    p = os.path.join(run_dir, "status.json")
    return read_json(p) if os.path.isfile(p) else None


def write_status(run_dir, status, **kw):
    atomic_json({"status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pbs_job": os.environ.get("PBS_JOBID"), **kw},
                os.path.join(run_dir, "status.json"))


class Lock:
    """O_EXCL lock file; a stale lock (dead PID on this host) must be broken explicitly with --break_stale_lock <path>."""

    def __init__(self, path, break_stale=None):
        self.path, self.break_stale = path, break_stale

    def __enter__(self):
        if self.break_stale and os.path.abspath(self.break_stale) == os.path.abspath(self.path) and os.path.exists(self.path):
            os.replace(self.path, self.path + f".broken_{int(time.time())}")
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"STOP locked: {self.path} exists ({open(self.path).read().strip()}); another job may be "
                             f"running. If it is stale, rerun with --break_stale_lock {self.path}")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "pbs_job": os.environ.get("PBS_JOBID"),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        return self

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
        return False


# ----------------------------------------------------------------------------------- statistics rule (frozen)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def t_ci95(d):
    d = [x for x in d if x is not None and math.isfinite(x)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    h = T95[n - 1] * sd / math.sqrt(n)
    return mu - h, mu + h


def verdict(d, a, b, n_required=3):
    """'consistent' only if all n_required same-seed differences (a - b) share the sign AND the 95 % t interval of the
    mean difference excludes 0. Otherwise 'no detectable difference'. Never a significance claim (n = 3)."""
    d = [x for x in d if x is not None and math.isfinite(x)]
    if len(d) < n_required:
        return f"incomplete (n = {len(d)})"
    lo, hi = t_ci95(d)
    if all(x < 0 for x in d) and hi < 0:
        return f"{a} consistently lower"
    if all(x > 0 for x in d) and lo > 0:
        return f"{b} consistently lower"
    return "no detectable difference at n = 3"


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True, default=str)
