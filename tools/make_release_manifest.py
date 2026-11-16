"""Write SHA256SUMS.txt for every file of the release (repository files and the release assets data/naca/*.npz,
checkpoints/naca/*/pcwno_best.pt), excluding reproduction outputs, caches, SHA256SUMS.txt itself and the downloaded
external WNO file (external/, which is not part of the release; only external/README.md is listed).

  python -B tools/make_release_manifest.py
"""
import hashlib
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}
SKIP_PREFIX = ("results/revision_2026/pointwno_v2_shock_benchmark_r1/reproduced",
               "results/revision_2026/pointwno_v2_shock_benchmark_r1/naca")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


rows = []
for root, dirs, files in os.walk(REPO):
    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
    for fn in sorted(files):
        rel = os.path.relpath(os.path.join(root, fn), REPO).replace(os.sep, "/")
        if rel == "SHA256SUMS.txt" or rel.startswith(SKIP_PREFIX):
            continue
        if rel.startswith("external/") and rel != "external/README.md":
            continue
        if rel.endswith(("wavelet_convolution_v3.py", "wno2d_r1.py")):
            raise SystemExit(f"STOP {rel} must not be part of the release")
        rows.append(f"{sha(os.path.join(REPO, rel))}  {rel}")
with open(os.path.join(REPO, "SHA256SUMS.txt"), "w") as f:
    f.write("\n".join(rows) + "\n")
print(f"{len(rows)} files")
