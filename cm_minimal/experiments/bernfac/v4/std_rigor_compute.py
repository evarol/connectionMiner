"""W5: rigor of the depth-standardized probability p~ on the real data (R4, R5, R6).

p~ = predictors.standardized(code, ref): the cell offset c_i (a_i for R4/R5, o_i + a_i for R6) is
replaced by one reference value c_bar. Here c_bar is the 10th, 50th (v3 default = median) or 90th
percentile of c_i.

No n x m array is formed or saved. Gene sample: the v3 3,000 random genes (seed 0, identical to
v3/std_quant.py) + 115 hemilineage markers (top 5 per hemilineage by train-rate enrichment, rate in
the hemilineage >= 0.3) + the v3 20 heat markers + the union of the depth-sensitive genes chosen in
the held-out split experiment (task 2). Matrices n x ~3.4k are held in memory only.

Sources: Y (binary data), PC (processed counts: log1p(1e4 x / UMI)), p^ of R4/R5/R6, p~ of R4/R5/R6
at three references.

Task 1  residual depth
  * 10-PC embedding (column-centred PCA on the 3,000 random genes) of logit(p) and of p
    (for Y: Y itself; for PC: the log-normalised counts; R1 = the v3 HVG-PCA embedding, top 10).
  * 5-fold CV linear R^2 of log UMI from the 10 PCs (total), and the partial CV R^2 beyond
    hemilineage x birth-time stratum dummies: 1 - SSE(strata + PCs) / SSE(strata only), out of fold.
  * same with log UMI permuted within strata (reference: an embedding that carries biology only).
  * per-gene Spearman rho with log UMI within hemilineage x birth-time strata (>= 30 cells),
    averaged over strata; median |rho| over the random genes. References: the same with depth
    permuted within strata; a binary draw Y* ~ Bernoulli(p) (puts p on the data's noise scale).
Task 2  depth-sensitive genes chosen on a random half of the cells (stratified by hemilineage),
  gap (mean over high-depth minus low-depth quartile cells, quartiles within hemilineage) evaluated
  on the other half; 20 splits; top 8 (as v3) and top 50.
Task 3  references 10/50/90th percentile of c_i: absolute level vs rank structure.
Task 4  retention p~ vs p^ on marker genes: between-hemilineage variance ratio, eta^2, correlation of
  hemilineage mean profiles; and within-hemilineage between-birth-time variance ratio (primary vs
  secondary biology).

Outputs: outputs/bernfac_20261020/v4/cache/std_rigor.npz (per-gene / per-cell vectors, n x 10
embeddings), std_rigor_*.json; tables are written by std_rigor_tables.py.
"""
import json, os, sys, time
from pathlib import Path
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "v3"))
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
import predictors as pr

V4 = pr.STUDY / "v4"
CACHE = V4 / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
NG, CHUNK, NPC, NSPLIT = 3000, 250, 10, 20
REFS = {"p10": 10, "p50": 50, "p90": 90}
rng0 = np.random.default_rng(0)

# --------------------------------------------------------------------------- metadata
cells = pd.read_csv(pr.ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
hl = cells.hemilineage.astype(str).values
bt = cells.birthtime.fillna("precursor").astype(str).values
strat = np.char.add(np.char.add(hl.astype(str), "|"), bt.astype(str))
HLS = sorted(set(hl))
depth = np.load(pr.CACHE / "cell_depth.npy")
ld = np.log(depth)
n = len(hl)
sel = json.loads((pr.CACHE / "heat_selection.json").read_text())
rate = np.load(pr.CACHE / "gene_train_rate.npy")
m = len(rate)

# same random genes as v3/std_quant.py
gg = sel["gene_groups"]
special = [j for k, v in gg.items() if k.startswith("marker") or k == "depth" for j in v]
rand = np.sort(np.random.default_rng(0).choice(np.setdiff1d(np.arange(m), special), NG, replace=False))
heat_markers = [j for k, v in gg.items() if k.startswith("marker") for j in v]
heat_depth = list(gg["depth"])

# strata
S_codes, S_idx = np.unique(strat, return_inverse=True)
S_size = np.bincount(S_idx)
big_strata = [np.flatnonzero(S_idx == s) for s in range(len(S_codes)) if S_size[s] >= 30]
hl_rows = {h: np.flatnonzero(hl == h) for h in HLS}
hl_big = [hl_rows[h] for h in HLS if len(hl_rows[h]) >= 30]
H1 = np.stack([hl == h for h in HLS], 1).astype(np.float64)
n_h = H1.sum(0)


def permute_within(x, groups, rng):
    y = x.copy()
    for g in np.unique(groups):
        r = np.flatnonzero(groups == g)
        y[r] = x[rng.permutation(r)]
    return y


ld_perm = permute_within(ld, S_idx, np.random.default_rng(1))

# --------------------------------------------------------------------------- Y, train mask, markers, task-2 selection
t0 = time.time()
Y = np.load(pr.DATA / "Y.npy")
Tr = ~np.load(pr.DATA / "heldout.npy")
print(f"loaded Y {time.time() - t0:.0f}s", flush=True)


def train_rate(rows):
    return (Y[rows] * Tr[rows]).sum(0) / np.maximum(Tr[rows].sum(0), 1)


rate_hl = np.stack([train_rate(hl_rows[h]) for h in HLS])
markers = []
for k, h in enumerate(HLS):
    others = np.delete(rate_hl, k, 0).mean(0)
    score = np.where(rate_hl[k] >= 0.3, rate_hl[k] - others, -np.inf)
    markers += [int(j) for j in np.argsort(-score)[:5] if j not in markers]
markers = np.array(markers)
lg = lambda p: np.log((p + 0.01) / (1 - p + 0.01))


def quartiles(rows_by_group):
    """lo / hi masks: bottom / top depth quartile within each group (list of index arrays)."""
    lo = np.zeros(n, bool); hi = np.zeros(n, bool)
    for r in rows_by_group:
        if len(r) < 8:
            continue
        q1, q3 = np.quantile(depth[r], [0.25, 0.75])
        lo[r[depth[r] <= q1]] = True; hi[r[depth[r] >= q3]] = True
    return lo, hi


splits = []
for s in range(NSPLIT):
    rng = np.random.default_rng(100 + s)
    A = np.zeros(n, bool)
    for h in HLS:
        r = rng.permutation(hl_rows[h]); A[r[:len(r) // 2]] = True
    B = ~A
    # selection on A exactly as heat_select.py (train entries, within-hemilineage quartiles)
    loA, hiA = quartiles([hl_rows[h][A[hl_rows[h]]] for h in HLS])
    overA = train_rate(np.flatnonzero(A))
    ds = np.mean([lg(train_rate(hl_rows[h][hiA[hl_rows[h]]])) - lg(train_rate(hl_rows[h][loA[hl_rows[h]]]))
                  for h in HLS], 0)
    broad = np.flatnonzero((overA > 0.25) & (overA < 0.75))
    top = broad[np.argsort(-ds[broad])][:50]
    splits.append(dict(A=A, top=top.astype(int), broad=broad))
    print(f"split {s}: top8 {top[:8].tolist()}  {time.time() - t0:.0f}s", flush=True)
depth_union = np.unique(np.concatenate([sp_["top"] for sp_ in splits] + [heat_depth]))
# reference set: all broad genes on the full data (for the "typical broad gene" gap)
over_all = rate
broad_all = np.flatnonzero((over_all > 0.25) & (over_all < 0.75))
broad_sample = np.sort(np.random.default_rng(2).choice(broad_all, min(300, len(broad_all)), replace=False))

extra = np.setdiff1d(np.unique(np.r_[markers, heat_markers, depth_union, broad_sample]), rand)
genes = np.r_[rand, extra]
pos = {int(j): k for k, j in enumerate(genes)}
ix = lambda arr: np.array([pos[int(j)] for j in arr])
I_rand = np.arange(NG)
I_mark, I_heat, I_dep, I_broad = ix(markers), ix(heat_markers), ix(depth_union), ix(broad_sample)
Yg = Y[:, genes].astype(np.float32)
del Y, Tr
print(f"genes: {len(genes)} (markers {len(markers)}, depth union {len(depth_union)}) {time.time() - t0:.0f}s",
      flush=True)

# processed counts on the same genes
keep = np.load(pr.DATA / "genes_kept.npy")
X = sp.load_npz(pr.ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz").tocsc()[:, keep]
PCg = X[:, genes].toarray().astype(np.float32)
del X
PCg = np.log1p(1e4 * PCg / depth[:, None]).astype(np.float32)


# --------------------------------------------------------------------------- metrics
def pca10(A):
    A = np.asarray(A, np.float64); A = A - A.mean(0)
    w, V = np.linalg.eigh(A.T @ A)
    return (A @ V[:, ::-1][:, :NPC]).astype(np.float32)


fold_lab = np.where(S_size[S_idx] >= 10, S_idx, -1)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), fold_lab))
D_strata = np.stack([fold_lab == s for s in np.unique(fold_lab)], 1).astype(np.float64)   # incl. intercept span


def cv_r2(E, y):
    """(total CV R^2 from E, partial CV R^2 of E beyond strata dummies, CV R^2 of strata alone)."""
    E = np.asarray(E, np.float64)
    pt, pf, ps = np.empty(n), np.empty(n), np.empty(n)
    for tr, te in FOLDS:
        X1 = np.c_[np.ones(len(tr)), E[tr]]
        b = np.linalg.lstsq(X1, y[tr], rcond=None)[0]; pt[te] = np.c_[np.ones(len(te)), E[te]] @ b
        Xs = D_strata[tr]; b = np.linalg.lstsq(Xs, y[tr], rcond=None)[0]; ps[te] = D_strata[te] @ b
        Xf = np.c_[D_strata[tr], E[tr]]; b = np.linalg.lstsq(Xf, y[tr], rcond=None)[0]
        pf[te] = np.c_[D_strata[te], E[te]] @ b
    sst = ((y - y.mean()) ** 2).sum()
    sse_s = ((y - ps) ** 2).sum()
    return 1 - ((y - pt) ** 2).sum() / sst, 1 - ((y - pf) ** 2).sum() / sse_s, 1 - sse_s / sst


LDR = {k: [rankdata(v[r]) for r in big_strata] for k, v in (("obs", ld), ("perm", ld_perm))}
LDR_HL = [rankdata(ld[r]) for r in hl_big]


def rho_within(P, groups, ldr_list):
    """Per-gene Spearman rho with each depth-rank vector, within groups, averaged (nan-mean)."""
    out = np.zeros((len(ldr_list), P.shape[1])); cnt = np.zeros((len(ldr_list), P.shape[1]))
    for gi, r in enumerate(groups):
        R = rankdata(P[r], axis=0); R = R - R.mean(0); rr = (R ** 2).sum(0)
        for k, ldr in enumerate(ldr_list):
            x = ldr[gi] - ldr[gi].mean()
            den = np.sqrt(rr * (x ** 2).sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(den > 0, x @ R / den, np.nan)
            ok = np.isfinite(v); out[k, ok] += v[ok]; cnt[k, ok] += 1
    with np.errstate(invalid="ignore"):
        return out / cnt


def var_parts(P):
    """between-HL variance, total variance, eta^2, HL means, within-HL between-birth-time variance."""
    P = np.asarray(P, np.float64)
    mu = P.mean(0)
    hm = (H1.T @ P) / n_h[:, None]
    between = (n_h[:, None] * (hm - mu) ** 2).sum(0) / n
    tot = P.var(0)
    wbt = np.zeros(P.shape[1])                       # within-HL, between-birth-time-class variance
    for s in range(len(S_codes)):
        r = np.flatnonzero(S_idx == s)
        h = HLS.index(S_codes[s].split("|")[0])
        wbt += len(r) * (P[r].mean(0) - hm[h]) ** 2
    return between, tot, hm, wbt / n


STRATA_ROWS = [np.flatnonzero(S_idx == k) for k in range(len(S_codes))]
QMASK = []                                            # per split and half: (lo, hi) within HL, within HL x BT
for spl in splits:
    d = {}
    for half, msk in (("selection_half", spl["A"]), ("heldout_half", ~spl["A"])):
        lo, hi = quartiles([hl_rows[h][msk[hl_rows[h]]] for h in HLS])
        lo2, hi2 = quartiles([r[msk[r]] for r in STRATA_ROWS])
        d[half] = [np.flatnonzero(x) for x in (lo, hi, lo2, hi2)]
    QMASK.append(d)


def split_gaps(P):
    """Task 2 on source P (n x G): per split, gap (hi - lo, quartiles within HL of the half) of the top-8 /
    top-50 genes in-sample (half A) and held out (half B); plus a within HL x birth-time variant."""
    rows = []
    for s, spl in enumerate(splits):
        for half, (lo, hi, lo2, hi2) in QMASK[s].items():
            for ntop in (8, 50):
                c = ix(spl["top"][:ntop])
                g = P[np.ix_(hi, c)].mean(0) - P[np.ix_(lo, c)].mean(0)
                g2 = P[np.ix_(hi2, c)].mean(0) - P[np.ix_(lo2, c)].mean(0)
                rows.append(dict(split=s, half=half, ntop=ntop, gap=float(g.mean()), gap_hlbt=float(g2.mean())))
            gb = P[np.ix_(hi, I_broad)].mean(0) - P[np.ix_(lo, I_broad)].mean(0)
            rows.append(dict(split=s, half=half, ntop=0, gap=float(gb.mean()), gap_hlbt=np.nan))
    return rows


res, emb, gap_rows, summ = {}, {}, [], {}


def analyse(name, P, logit_ok=True, draw=False):
    t = time.time()
    P = np.asarray(P, np.float32)
    r = {}
    rh = rho_within(P, big_strata, [LDR["obs"], LDR["perm"]])
    r["rho_hlbt"], r["rho_hlbt_perm"] = rh[0], rh[1]
    r["rho_hl"] = rho_within(P[:, I_rand], hl_big, [LDR_HL])[0]
    if draw:
        Ys = (np.random.default_rng(3).random((n, NG), dtype=np.float32) < P[:, I_rand]).astype(np.float32)
        r["rho_hlbt_draw"] = rho_within(Ys, big_strata, [LDR["obs"]])[0]
        del Ys
    b, tot, hm, wbt = var_parts(P)
    r.update(between=b, total=tot, wbt=wbt); r["hlmean"] = hm
    r["mean_p"] = P[:, I_rand].mean(0)
    r["cell_mean"] = P[:, I_rand].mean(1)
    res[name] = r
    e_p = pca10(P[:, I_rand])
    emb[name + "|p"] = e_p
    if logit_ok:
        L = np.log(np.clip(P[:, I_rand], 1e-6, 1 - 1e-6) / np.clip(1 - P[:, I_rand], 1e-6, 1))
        emb[name + "|logit"] = pca10(L); del L
    gr = split_gaps(P)
    for g in gr:
        g["source"] = name
    gap_rows.extend(gr)
    print(f"{name}: med|rho_hlbt| {np.nanmedian(np.abs(r['rho_hlbt'][I_rand])):.3f} "
          f"(perm {np.nanmedian(np.abs(r['rho_hlbt_perm'][I_rand])):.3f})  {time.time() - t:.0f}s", flush=True)


analyse("Y", Yg, logit_ok=False)
del Yg
analyse("PC", PCg, logit_ok=False)
del PCg
R1 = np.load(pr.STUDY / "full/umap/cache/R1_embedding.npz")["E"][:, :NPC].astype(np.float32)
emb["R1|p"] = R1

offsets = {}
for code in pr.STD_MODELS:
    p = pr.load(code)
    c = pr.cell_offset(p)
    offsets[code] = c
    blocks = lambda: np.concatenate([p.block(np.arange(n), genes[s:s + CHUNK])[0].astype(np.float32)
                                     for s in range(0, len(genes), CHUNK)], 1)
    analyse(code, blocks(), draw=True)
    for lab, q in REFS.items():
        p.a = np.full_like(c, np.percentile(c, q))
        analyse(f"{code}~{lab}", blocks(), draw=(lab == "p50"))
    if code == "R4":                                    # u_i'v_j alone (= centred logit p~ of R4)
        emb["R4_UV|p"] = pca10(p.U @ p.V[rand].T)
    del p

# --------------------------------------------------------------------------- embeddings -> R^2
r2 = []
for key, E in emb.items():
    for yname, y in (("logUMI", ld), ("logUMI_perm_within_strata", ld_perm)):
        tot, part, s_only = cv_r2(E, y)
        src, scale = key.split("|")
        r2.append(dict(source=src, scale=scale, target=yname, r2_cv_total=tot, r2_cv_partial_beyond_strata=part,
                       r2_cv_strata_only=s_only))
pd.DataFrame(r2).to_csv(CACHE / "std_rigor_r2.csv", index=False)
pd.DataFrame(gap_rows).to_csv(CACHE / "std_rigor_gaps.csv", index=False)
save = dict(genes=genes, rand_idx=I_rand, marker_idx=I_mark, heat_marker_idx=I_heat, depth_union_idx=I_dep,
            broad_idx=I_broad, hls=np.array(HLS), sources=np.array(list(res)), ld=ld, ld_perm=ld_perm,
            strata=strat, rate=rate[genes])
for name, r in res.items():
    for k, v in r.items():
        save[f"{k}__{name}"] = np.asarray(v, np.float32)
for code, c in offsets.items():
    save[f"c__{code}"] = c
for key, E in emb.items():
    save[f"emb__{key}"] = E
np.savez(CACHE / "std_rigor.npz", **save)
print(pd.DataFrame(r2).round(3).to_string())
print(f"done {time.time() - t0:.0f}s")
