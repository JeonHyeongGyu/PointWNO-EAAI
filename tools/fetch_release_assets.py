"""Download the release assets (NACA dataset + six checkpoints) of GitHub release v1.0.0, verify each against
SHA256SUMS.txt and put it where the code expects it.

  python -B tools/fetch_release_assets.py [--only data|checkpoints] [--timeout 600]

Release: https://github.com/JeonHyeongGyu/PointWNO-EAAI/releases/tag/v1.0.0
Asset names: Data_allpoints_xymomcp_from_dat.npz -> data/naca/Data_allpoints_xymomcp_from_dat.npz
             BM_NACA_<arm>_s<seed>_pcwno_best.pt -> checkpoints/naca/BM_NACA_<arm>_s<seed>/pcwno_best.pt
An existing file with the correct SHA-256 is kept; a file with a wrong SHA-256 is never written.
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = "v1.0.0"
BASE = f"https://github.com/JeonHyeongGyu/PointWNO-EAAI/releases/download/{TAG}/"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def assets():
    out = []
    for line in open(os.path.join(REPO, "SHA256SUMS.txt")):
        s, rel = line.rstrip("\n").split("  ", 1)
        if rel.startswith("data/naca/") and rel.endswith(".npz"):
            out.append(("data", os.path.basename(rel), rel, s))
        elif rel.startswith("checkpoints/naca/") and rel.endswith("/pcwno_best.pt"):
            out.append(("checkpoints", rel.split("/")[2] + "_pcwno_best.pt", rel, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["data", "checkpoints"], default=None)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    bad = 0
    for kind, name, rel, want in assets():
        if a.only and kind != a.only:
            continue
        dest = os.path.join(REPO, rel)
        if os.path.isfile(dest) and sha(dest) == want:
            print(f"ok (present) {rel}")
            continue
        tmp = dest + ".part"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            with urllib.request.urlopen(BASE + name, timeout=a.timeout) as r, open(tmp, "wb") as f:
                for b in iter(lambda: r.read(1 << 22), b""):
                    f.write(b)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"DOWNLOAD FAILED {BASE + name}: {e!r}")
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        got = sha(tmp)
        if got != want:
            bad += 1
            os.remove(tmp)
            print(f"SHA-256 MISMATCH {name}: {got} != {want} (not written)")
            continue
        os.replace(tmp, dest)
        print(f"ok {rel}  sha256 {got[:16]}…")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
