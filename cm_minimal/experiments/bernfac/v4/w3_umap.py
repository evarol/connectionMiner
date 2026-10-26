"""W3: UMAPs with the v3 settings (n_neighbors 15, min_dist 0.3, random_state 0, input = the raw embedding,
whose Euclidean geometry equals that of the centred-SVD unwhitened principal coordinates).

Reused from v3 (identical input): R1, R2, R6, R7, R9, and R8 unless the R8 hook points elsewhere.
Recomputed: R3-R5 (train-only fits), R10, R11, and R8 if W3_R8_EMBEDDING / --r8 is set.
Output: v4/cache/w3_{code}_umap.npy
"""
import argparse
import os
import shutil

import numpy as np

from w3_common import CACHE, ORDER, R8_DEFAULT, V3CACHE, r8_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", default=ORDER)
    ap.add_argument("--r8", default=None)
    a = ap.parse_args()
    if a.r8:
        os.environ["W3_R8_EMBEDDING"] = a.r8
    import umap
    for code in a.codes:
        out = CACHE / f"w3_{code}_umap.npy"
        reuse = code in ("R1", "R2", "R6", "R7", "R9") or (code == "R8" and r8_path() == R8_DEFAULT)
        if reuse:
            shutil.copy(V3CACHE / f"{code}_umap.npy", out)
            print(code, "reused v3 UMAP", flush=True)
            continue
        f = CACHE / f"w3_{code}_raw.npz"
        if not f.exists():
            print(code, "no raw embedding; skipped", flush=True)
            continue
        E = np.load(f, allow_pickle=True)["E"].astype(np.float32)
        Z = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0).fit_transform(E).astype(np.float32)
        np.save(out, Z)
        print(code, "UMAP computed", flush=True)


if __name__ == "__main__":
    main()
