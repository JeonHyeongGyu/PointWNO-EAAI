"""Download the ONE upstream WNO file the released models need, from its fixed commit, and verify it.

  python -B tools/fetch_wno_upstream.py                 # download from GitHub (raw.githubusercontent.com)
  python -B tools/fetch_wno_upstream.py --from_file F   # use a copy you obtained yourself (e.g. from a git clone)

Source: https://github.com/csccm-iitd/WNO, commit fa6b5a79ad97ad5c2f8ee3773fe3c732d344dee3, path wavelet_convolution_v3.py
(the file was removed from the default branch on 2025-08-19 but remains in the commit history).
The file is used UNMODIFIED. It is not part of this repository and must not be committed (see .gitignore): the upstream
repository declares no license, so this repository does not redistribute it. Downloading it for your own use is a
matter between you and the upstream authors; see THIRD_PARTY_NOTICES.md.
Destination: external/csccm-iitd_WNO_fa6b5a7/wavelet_convolution_v3.py
"""
import argparse
import hashlib
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("pointwno_deps", os.path.join(REPO, "pointwno", "_deps.py"))
D = importlib.util.module_from_spec(_spec)   # loaded by path: importing the pointwno package needs this file already
_spec.loader.exec_module(D)


def git_blob_sha1(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_file", default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    if a.from_file:
        data = open(a.from_file, "rb").read()
        src = os.path.abspath(a.from_file)
    else:
        try:
            with urllib.request.urlopen(D.WNO_URL, timeout=a.timeout) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001
            sys.exit(f"DOWNLOAD FAILED: {D.WNO_URL}\n  {e!r}\n"
                     f"Download the file at commit {D.WNO_COMMIT} by other means and run: "
                     f"python -B tools/fetch_wno_upstream.py --from_file <path>")
        src = D.WNO_URL
    got, blob = hashlib.sha256(data).hexdigest(), git_blob_sha1(data)
    if got != D.WNO_SHA256 or blob != D.WNO_GIT_BLOB:
        sys.exit(f"VERIFICATION FAILED for {src}\n  sha256 {got} (expected {D.WNO_SHA256})\n"
                 f"  git blob {blob} (expected {D.WNO_GIT_BLOB})\nNothing was written.")
    os.makedirs(os.path.dirname(D.WNO_PATH), exist_ok=True)
    tmp = D.WNO_PATH + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, D.WNO_PATH)
    print(f"OK {os.path.relpath(D.WNO_PATH, REPO)}  sha256 {got}  (git blob {blob}, commit {D.WNO_COMMIT})")


if __name__ == "__main__":
    main()
