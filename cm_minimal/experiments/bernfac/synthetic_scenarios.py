"""Synthetic ground-truth scenarios for the BernFac model-selection study.

Every scenario is calibrated to the real 1000 x 100 preMN subset
(outputs/bernfac_20261020/data: 4 libraries x 5 hemilineages x 50 cells) and
returns a known latent probability matrix P_true plus one binary sample Y.

S1  well-specified : logit P_true = U* V*'  of exact rank k* (3 or 8).
    U*, V* come from a rank-k* SVD truncation of a BernFac fit to the real
    Y_ge1 (so gene intercepts and hemilineage structure look real); rows are
    bootstrapped within library x hemilineage strata per seed, and the logit
    scale is calibrated so the spread of per-gene detection matches the data.
S2  misspecified count model :
        mu_ij = s_i * r_{c(i) j} * exp(a_i' b_j - |b_j|^2/2),  X ~ Poisson(mu),
        Y = 1[X >= 1],  P_true = 1 - exp(-mu)
    r = per-hemilineage gene rates estimated from the real counts, s_i = depth
    (real total_umi, library medians removed, bootstrapped), a_i' b_j = rank-2
    within-cluster continuous variation.  logit(P_true) is NOT low rank.
S3  batch : S2 x library depth factor L_l x gene-specific library effect
    exp(beta_lj).  Two targets: P_true (with batch) and P_bio (L = 1, beta = 0,
    i.e. library effects removed, per-cell depth kept).

For each dataset we also freeze all resampling used by the selection criteria
so that every grid point sees the same splits:
    Y_rep      independent Bernoulli replicate of P_true          (C3)
    Y_A, Y_B   count splitting: X_A ~ Binomial(X, 1/2), X_B = X - X_A,
               binarised (Poisson -> independent Poisson(mu/2) halves;
               Neufeld, Gao & Witten 2022)                         (C2)
    entry_fold random entry partition into 10 groups (0..4 used)  (C1)
    row_fold   5-fold cell partition, stratified by stratum        (C4)
    gene_half  per-cell random half of the genes for fold-in       (C4)

Run:  python synthetic_scenarios.py   (writes synthetic/data/*.npz + calibration.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import bernfac as bf

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/bernfac_20261020/data"
OUT = ROOT / "outputs/bernfac_20261020/synthetic/data"

SCENARIOS = ["S1_k3", "S1_k8", "S2", "S3"]
SEEDS = [0, 1, 2]
LIB_DEPTH = np.array([0.6, 0.85, 1.2, 1.65])   # S3 multiplicative library depth factors
BETA_SD = 0.5                                   # S3 gene-specific library log-effect sd
ETA_SD = 0.4                                    # S2/S3 within-cluster rank-2 effect sd (per factor)


def load_real():
    counts = np.load(DATA / "counts.npy").astype(float)
    Y = np.load(DATA / "Y_ge1.npy").astype(float)
    cells = pd.read_csv(DATA / "cells.csv")
    lib = pd.Categorical(cells.library).codes
    clu = pd.Categorical(cells.hemilineage).codes
    return counts, Y, cells, lib, clu


def stratified_bootstrap(lib, clu, rng):
    """Resample cell indices with replacement inside each library x cluster stratum."""
    idx = np.empty(len(lib), dtype=int)
    for s in np.unique(lib * 10 + clu):
        rows = np.flatnonzero(lib * 10 + clu == s)
        idx[rows] = rng.choice(rows, len(rows), replace=True)
    return idx


# ----------------------------------------------------------------------------
# S1: well-specified logistic low rank
# ----------------------------------------------------------------------------
_REAL_FIT = {}


def real_logits(Y):
    """BernFac fit to the real Y (k=12, lambda=1); cached, defines S1 structure."""
    if "psi" not in _REAL_FIT:
        r = bf.fit(Y, 12, 1.0, 1.0, max_iter=400, init="svd", seed=0)
        _REAL_FIT["psi"] = r.U @ r.V.T
    return _REAL_FIT["psi"]


def make_s1(kstar, seed, Y, lib, clu):
    rng = np.random.default_rng(1000 * kstar + seed)
    Us, s, Vt = np.linalg.svd(real_logits(Y), full_matrices=False)
    U0, V0 = Us[:, :kstar] * np.sqrt(s[:kstar]), Vt[:kstar].T * np.sqrt(s[:kstar])
    U0 = U0[stratified_bootstrap(lib, clu, rng)]
    U0 = U0 + 0.1 * U0.std(0) * rng.standard_normal(U0.shape)   # avoid exact duplicates
    # calibrate a global logit scale c: sd of per-gene mean P == sd of real detection
    target = Y.mean(0).std()
    cs = np.linspace(0.5, 3.0, 101)
    sds = [bf.sigmoid(c * U0 @ V0.T).mean(0).std() for c in cs]
    c = cs[np.argmin(np.abs(np.array(sds) - target))]
    U, V = np.sqrt(c) * U0, np.sqrt(c) * V0
    P = bf.sigmoid(U @ V.T)
    Ysim = (rng.random(P.shape) < P).astype(np.int8)
    Yrep = (rng.random(P.shape) < P).astype(np.int8)
    return dict(P_true=P, Y=Ysim, Y_rep=Yrep, U_true=U, V_true=V, logit_scale=c), rng


# ----------------------------------------------------------------------------
# S2 / S3: Poisson count model
# ----------------------------------------------------------------------------
def calibrate_rates(det, s, eta, clu, n_bisect=60):
    """Bisection on log r_cj so mean_{i in c} [1 - exp(-s_i r_cj e^eta_ij)] = det_cj."""
    lo, hi = np.full(det.shape, -12.0), np.full(det.shape, 8.0)
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        pred = np.stack([np.mean(1 - np.exp(-s[clu == c, None] * np.exp(mid[c] + eta[clu == c])), 0)
                         for c in range(det.shape[0])])
        lo, hi = np.where(pred < det, mid, lo), np.where(pred < det, hi, mid)
    return np.exp(0.5 * (lo + hi))


def make_counts(seed, counts, cells, lib, clu, batch):
    rng = np.random.default_rng(5000 + 100 * int(batch) + seed)
    n, m = counts.shape
    depth = cells.total_umi.to_numpy(float)
    for l in np.unique(lib):                          # remove real library depth differences
        depth[lib == l] /= np.median(depth[lib == l])
    depth /= depth.mean()
    boot = stratified_bootstrap(lib, clu, rng)
    s = depth[boot]
    a = rng.standard_normal((n, 2))
    b = ETA_SD * rng.standard_normal((m, 2))
    eta = a @ b.T
    # per-hemilineage gene rates r_cj, calibrated so that the expected detection
    # rate in each (cluster, gene) equals the real one (real counts are
    # overdispersed, so plugging in mean rates would over-detect by ~20 points)
    det = np.stack([(counts[clu == c] >= 1).mean(0) for c in range(clu.max() + 1)])
    r = calibrate_rates(np.clip(det, 0.005, 0.995), s, eta, clu)
    log_bio = np.log(r[clu]) + eta
    mu_bio = s[:, None] * np.exp(log_bio)
    if batch:
        beta = BETA_SD * rng.standard_normal((lib.max() + 1, m))
        beta -= beta.mean(0, keepdims=True)           # centred per gene over libraries
        mu = mu_bio * LIB_DEPTH[lib][:, None] / LIB_DEPTH.mean() * np.exp(beta[lib])
    else:
        mu = mu_bio
    X = rng.poisson(mu)
    Xrep = rng.poisson(mu)
    XA = rng.binomial(X, 0.5)
    XB = X - XA
    d = dict(mu=mu, P_true=1 - np.exp(-mu), counts=X.astype(np.int32),
             Y=(X >= 1).astype(np.int8), Y_rep=(Xrep >= 1).astype(np.int8),
             Y_A=(XA >= 1).astype(np.int8), Y_B=(XB >= 1).astype(np.int8),
             P_half=1 - np.exp(-mu / 2))
    if batch:
        d["P_bio"] = 1 - np.exp(-mu_bio)
    return d, rng


def add_splits(d, lib, clu, rng, n_entry_groups=10, n_row_folds=5):
    n, m = d["Y"].shape
    d["entry_fold"] = rng.permutation(np.arange(n * m) % n_entry_groups).reshape(n, m).astype(np.int8)
    row_fold = np.empty(n, dtype=np.int8)
    for s in np.unique(lib * 10 + clu):
        rows = np.flatnonzero(lib * 10 + clu == s)
        row_fold[rng.permutation(rows)] = np.arange(len(rows)) % n_row_folds
    d["row_fold"] = row_fold
    d["gene_half"] = np.stack([rng.permutation(np.arange(m) % 2) for _ in range(n)]).astype(bool)
    d["library"], d["cluster"] = lib.astype(np.int8), clu.astype(np.int8)
    return d


def make_dataset(scen, seed):
    counts, Y, cells, lib, clu = load_real()
    if scen.startswith("S1"):
        d, rng = make_s1(int(scen.split("_k")[1]), seed, Y, lib, clu)
    else:
        d, rng = make_counts(seed, counts, cells, lib, clu, batch=(scen == "S3"))
    return add_splits(d, lib, clu, rng)


def describe(Ymat):
    g, c = Ymat.mean(0), Ymat.mean(1)
    return dict(density=float(Ymat.mean()),
                gene_det_q=[float(x) for x in np.quantile(g, [0.05, 0.25, 0.5, 0.75, 0.95])],
                cell_det_q=[float(x) for x in np.quantile(c, [0.05, 0.25, 0.5, 0.75, 0.95])])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts, Y, cells, lib, clu = load_real()
    calib = {"real_Y_ge1": describe(Y)}
    for scen in SCENARIOS:
        for seed in SEEDS:
            d = make_dataset(scen, seed)
            np.savez_compressed(OUT / f"{scen}_seed{seed}.npz", **d)
            info = describe(d["Y"])
            info["mean_P_true"] = float(d["P_true"].mean())
            info["sd_logit_P_true"] = float(np.std(np.log(np.clip(d["P_true"], 1e-6, 1 - 1e-6) / np.clip(1 - d["P_true"], 1e-6, 1))))
            if "P_bio" in d:
                info["rms_P_true_minus_P_bio"] = float(np.sqrt(np.mean((d["P_true"] - d["P_bio"]) ** 2)))
            if "logit_scale" in d:
                info["logit_scale"] = float(d["logit_scale"])
            calib[f"{scen}_seed{seed}"] = info
            print(scen, seed, {k: (round(v, 3) if isinstance(v, float) else [round(x, 2) for x in v])
                               for k, v in info.items()})
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
