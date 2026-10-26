"""W3 step 1: build raw embeddings R1-R10 (R11 comes from w3_scvi.py), then the common convention:
centre over cells, thin SVD, keep principal coordinates P = U S (sorted by singular value).

For factor models the embedding of Psi = U V' is E = Q_u W S (thin SVD via QR; n x m never formed).
Centring E over cells and taking its SVD gives the principal coordinates of the column-centred Psi,
because E and Psi have the same row Gram matrix up to an orthogonal map.

Usage: python w3_build.py [--r8 PATH] [--only R8 ...]
Outputs: v4/cache/w3_{code}_raw.npz (E, info), v4/cache/w3_{code}_pc.npz (P, s, info)
"""
import argparse
import json
import os
import time

import numpy as np

from w3_common import CACHE, MODELS, ORDER, TRAIN_ONLY, V3CACHE, hvg_index, load_counts, r8_path

import umap_compare as uc  # v3 helpers (embed_r3 = QR/SVD embedding of U V')


def raw_embedding(code, X):
    f = CACHE / f"w3_{code}_raw.npz"
    if code in ("R1", "R2"):
        z = np.load(V3CACHE / f"{code}_embedding.npz", allow_pickle=True)
        return z["E"].astype(np.float64), {"source": f"v3 cache {code}_embedding.npz", **json.loads(str(z["info"]))}
    if code in TRAIN_ONLY:
        E, info = uc.embed_r3(TRAIN_ONLY[code])
        info["note"] = "train-only fit that was scored (review B finding 5)"
        return E, info
    if code == "R6":
        return np.load(MODELS / "counts/embedding_nb.npy").astype(np.float64), {"source": "models/counts/embedding_nb.npy"}
    if code == "R7":
        return np.load(MODELS / "mfa/embedding.npy").astype(np.float64), {"source": "models/mfa/embedding.npy"}
    if code == "R8":
        p = r8_path()
        return np.load(p).astype(np.float64), {"source": str(p), "hook": "W3_R8_EMBEDDING / --r8"}
    if code == "R9":
        return np.load(MODELS / "lbm/embedding.npy").astype(np.float64), {"source": "models/lbm/embedding.npy"}
    if code == "R10":
        if f.exists():
            z = np.load(f, allow_pickle=True)
            return z["E"].astype(np.float64), json.loads(str(z["info"]))
        import anndata as ad
        import scanpy as sc
        idx = hvg_index(X)
        a = ad.AnnData(X[:, idx].astype(np.float32).tocsr())
        sc.experimental.pp.normalize_pearson_residuals(a, theta=100)   # clip = sqrt(n) (default)
        sc.pp.pca(a, n_comps=50, svd_solver="arpack", random_state=0)
        E = a.obsm["X_pca"].astype(np.float64)
        info = {"method": "analytic Pearson residuals, NB theta=100, clip sqrt(n), on the 2,000 R1 HVGs "
                          "(totals over those genes), then PCA 50 (Lause et al. 2021 recipe)",
                "var_ratio_sum": float(a.uns["pca"]["variance_ratio"].sum())}
        np.savez_compressed(f, E=E.astype(np.float32), info=json.dumps(info))
        return E, info
    if code == "R11":
        if not f.exists():
            raise FileNotFoundError("run w3_scvi.py first")
        z = np.load(f, allow_pickle=True)
        return z["E"].astype(np.float64), json.loads(str(z["info"]))
    raise KeyError(code)


def principal_coords(E):
    Ec = E - E.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(Ec, full_matrices=False)
    keep = s > 1e-8 * s[0]
    U, s = U[:, keep], s[keep]
    sgn = np.sign(U[np.abs(U).argmax(0), np.arange(U.shape[1])])
    return U * s * sgn, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r8", default=None, help="alternative R8 embedding (n x d .npy), e.g. W2's refit")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    if args.r8:
        os.environ["W3_R8_EMBEDDING"] = args.r8
    X, _ = load_counts()
    for code in args.only or ORDER:
        if code == "R11" and not (CACHE / "w3_R11_raw.npz").exists():
            print("R11 not available yet; skipped", flush=True)
            continue
        t0 = time.time()
        E, info = raw_embedding(code, X)
        if code != "R10" and code != "R11":
            np.savez_compressed(CACHE / f"w3_{code}_raw.npz", E=E.astype(np.float32), info=json.dumps(info, default=str))
        P, s = principal_coords(E)
        info.update(native_dim=int(E.shape[1]), rank=int(P.shape[1]), sv_top5=s[:5].round(3).tolist(),
                    var_frac_top20=float((s[:20] ** 2).sum() / (s ** 2).sum()))
        np.savez_compressed(CACHE / f"w3_{code}_pc.npz", P=P.astype(np.float32), s=s, info=json.dumps(info, default=str))
        print(code, P.shape, f"{time.time() - t0:.0f}s", info["sv_top5"], flush=True)


if __name__ == "__main__":
    main()
