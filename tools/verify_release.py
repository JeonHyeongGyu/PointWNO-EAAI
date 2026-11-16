"""Check every file listed in SHA256SUMS.txt (repository root). Large release assets (dataset, checkpoints) that have
not been downloaded are reported as 'absent' instead of failing, unless --require_assets is given.

  python -B tools/verify_release.py [--require_assets]
"""
import argparse
import hashlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = (".npz", ".pt")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--require_assets", action="store_true")
a = ap.parse_args()
ok = bad = absent = 0
for line in open(os.path.join(REPO, "SHA256SUMS.txt")):
    s, rel = line.rstrip("\n").split("  ", 1)
    p = os.path.join(REPO, rel)
    if not os.path.isfile(p):
        if rel.endswith(ASSETS) and not a.require_assets:
            absent += 1
            print(f"absent  {rel}")
            continue
        bad += 1
        print(f"MISSING {rel}")
    elif sha(p) != s:
        bad += 1
        print(f"DIFFERS {rel}")
    else:
        ok += 1
print(f"ok {ok}, absent release assets {absent}, failed {bad}")
sys.exit(1 if bad else 0)
