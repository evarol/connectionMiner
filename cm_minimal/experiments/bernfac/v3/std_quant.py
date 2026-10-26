"""Whole-matrix quantification of depth standardization (sampled; no n x m array is formed or saved).

Sample: ALL 16,460 cells x 3,000 random genes (seed 0) plus the 28 heat_select marker and
depth-sensitive genes (flagged; "all genes" summaries use the 3,000 random genes only). All cells is
the cheaper and better choice here: per-gene, within-hemilineage statistics need every cell of a
hemilineage, and every predictor evaluates a gene chunk for all cells in one block call.

Sources: binary Y (all entries, as data reference), p^ of R3-R9 and PO, p~ of R4, R5, R6
(predictors.standardized, c_bar = median c_i).

Per gene and source
  rho_hl    Spearman rho between p_ij (over cells i) and log total UMI, within each hemilineage with
            >= 30 cells, then averaged over hemilineages (unweighted)
  eta2      between-hemilineage variance / total variance of p_ij over all cells
  hl_mean   mean p per hemilineage (for the marker contrast)
  lo / hi   mean p over low / high depth-quartile cells (within-hemilineage quartiles, as heat_select)
Optional embedding check (R4, R5, R6): 10-PC embedding of the column-centred p^, p~ and of u_i'v_j on the
same genes; 15-NN overlap and mean canonical correlation vs the UV' embedding, and R^2 of log UMI
(total and within hemilineage) from each embedding (residual depth inside UV').

Outputs: v3/cache/std_quant.npz (per-gene vectors only), v3/cache/std_embed.json
"""
import json, os, sys, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
import predictors as pr

OUT = pr.CACHE / "std_quant.npz"
EMB = pr.CACHE / "std_embed.json"
NG, CHUNK, NPC, KNN = 3000, 250, 10, 15
SOURCES = ["Y", "R3", "R4", "R4~", "R5", "R5~", "R6", "R6~", "PO", "R7", "R8", "R9"]

sel = json.loads((pr.CACHE / "heat_selection.json").read_text())
cells = pd.read_csv(pr.ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
hl = cells.hemilineage.astype(str).values
HLS = sorted(set(hl))
depth = np.load(pr.CACHE / "cell_depth.npy")
ld = np.log(depth)
dq = np.load(pr.CACHE / "depth_quartile.npy")
rate = np.load(pr.CACHE / "gene_train_rate.npy")
m = len(rate)
n = len(hl)

rng = np.random.default_rng(0)
gg = sel["gene_groups"]
special = [j for k, v in gg.items() if k.startswith("marker") or k == "depth" for j in v]
rand = np.sort(rng.choice(np.setdiff1d(np.arange(m), special), NG, replace=False))
genes = np.r_[rand, special]
is_rand = np.r_[np.ones(NG, bool), np.zeros(len(special), bool)]

hl_rows = {h: np.flatnonzero(hl == h) for h in HLS}
hl_big = [h for h in HLS if len(hl_rows[h]) >= 30]
ld_rank = {h: rankdata(ld[hl_rows[h]]) for h in hl_big}
H_onehot = np.stack([hl == h for h in HLS], 1).astype(np.float64)        # n x n_hl
n_h = H_onehot.sum(0)


def gene_stats(P):
    """P: n x g block -> rho_hl (g), eta2 (g), hl_mean (n_hl x g), lo (g), hi (g)."""
    P = np.asarray(P, np.float64)
    rho = np.zeros((len(hl_big), P.shape[1]))
    for k, h in enumerate(hl_big):
        R = rankdata(P[hl_rows[h]], axis=0)
        r = ld_rank[h] - ld_rank[h].mean()
        R = R - R.mean(0)
        den = np.sqrt((R ** 2).sum(0) * (r ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            rho[k] = np.where(den > 0, r @ R / den, np.nan)
    hm = (H_onehot.T @ P) / n_h[:, None]
    tot = P.var(0)
    between = (n_h[:, None] * (hm - P.mean(0)) ** 2).sum(0) / n
    with np.errstate(invalid="ignore", divide="ignore"):
        eta2 = np.where(tot > 0, between / tot, np.nan)
    return np.nanmean(rho, 0), eta2, hm, P[dq == 1].mean(0), P[dq == 2].mean(0)


def get_pred(src):
    if src == "Y":
        return None
    return pr.standardized(src[:2]) if src.endswith("~") else pr.load(src)


def embed(A):
    """10-PC embedding of the column-centred n x g matrix A (float32)."""
    A = A - A.mean(0)
    U, s, _ = np.linalg.svd(A.T @ A)                                     # g x g Gram, cheap for g=3k
    return A @ U[:, :NPC]


def r2_depth(E, within=False):
    y = ld.copy(); X = E.copy()
    if within:                                                           # remove hemilineage means
        for h in HLS:
            r = hl_rows[h]; y[r] -= y[r].mean(); X[r] -= X[r].mean(0)
    X = np.c_[np.ones(n), X]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def knn_overlap(E1, E2):
    a = NearestNeighbors(n_neighbors=KNN + 1).fit(E1).kneighbors(E1, return_distance=False)[:, 1:]
    b = NearestNeighbors(n_neighbors=KNN + 1).fit(E2).kneighbors(E2, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / KNN for x, y in zip(a, b)]))


def cancorr(E1, E2):
    q1 = np.linalg.qr(E1 - E1.mean(0))[0]; q2 = np.linalg.qr(E2 - E2.mean(0))[0]
    return float(np.linalg.svd(q1.T @ q2, compute_uv=False).mean())


def main():
    res = {k: {} for k in ("rho", "eta2", "hlmean", "lo", "hi")}
    emb = {}
    Yall = np.load(pr.DATA / "Y.npy", mmap_mode="r")
    for src in SOURCES:
        t0 = time.time()
        p = get_pred(src)
        keepA = src[:2] in pr.STD_MODELS and src != "Y"
        A = np.empty((n, len(rand)), np.float32) if keepA else None
        acc = {k: [] for k in res}
        for s in range(0, len(genes), CHUNK):
            cols = genes[s:s + CHUNK]
            if p is None:
                P = np.asarray(Yall[:, cols], np.float64)
            else:
                P = p.block(np.arange(n), cols)[0]
            for k, v in zip(res, gene_stats(P)):
                acc[k].append(v)
            if keepA:
                w = np.flatnonzero(s + np.arange(len(cols)) < len(rand))
                A[:, s + w] = P[:, w]
            del P
        for k in res:
            res[k][src] = np.concatenate(acc[k], axis=-1)
        if keepA:
            e = embed(A); del A
            emb[src] = e
            if not src.endswith("~"):                                     # UV' embedding once per model
                UV = (p.U @ p.V[rand].T).astype(np.float32)
                emb[src[:2] + "_UV"] = embed(UV); del UV
        del p
        print(f"{src}: {time.time() - t0:.0f}s  median|rho| (random genes) "
              f"{np.nanmedian(np.abs(res['rho'][src][is_rand])):.3f}", flush=True)
    np.savez(OUT, genes=genes, is_rand=is_rand, rate=rate[genes], hls=np.array(HLS), hl_big=np.array(hl_big),
             sources=np.array(SOURCES),
             **{f"{k}__{src}": res[k][src].astype(np.float32) for k in res for src in SOURCES})
    out = {}
    for c in pr.STD_MODELS:
        uv = emb[c + "_UV"]
        out[c] = {lab: dict(knn15_overlap_vs_UV=knn_overlap(emb[key], uv), mean_cancorr_vs_UV=cancorr(emb[key], uv),
                            r2_logUMI=r2_depth(emb[key]), r2_logUMI_within_hl=r2_depth(emb[key], True))
                  for lab, key in (("p_hat", c), ("p_tilde", c + "~"))}
        out[c]["UV"] = dict(r2_logUMI=r2_depth(uv), r2_logUMI_within_hl=r2_depth(uv, True))
        # Spearman of each UV' PC with log UMI (max |rho|), the residual depth inside the factors
        out[c]["UV"]["max_abs_spearman_pc_logUMI"] = float(max(abs(pd.Series(uv[:, k]).corr(pd.Series(ld), method="spearman"))
                                                            for k in range(NPC)))
    EMB.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
