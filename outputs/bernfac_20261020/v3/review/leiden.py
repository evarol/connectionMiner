import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "4"
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from sklearn.metrics import adjusted_rand_score as ARI, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
R = "/Users/erdem/Documents/github/connectionMiner/outputs/"
cells = pd.read_csv(R + "connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
hl = cells["hemilineage"].astype(str).values
lib = cells["library"].astype(str).values
sex = cells["Sex"].astype(str).values
rows = []
for code in ["R1", "R3", "R4", "R5", "R6", "R7", "R8"]:
    z = np.load(R + f"bernfac_20261020/full/umap/cache/{code}_embedding.npz")
    E = z[list(z.keys())[0]].astype(np.float32)
    a = ad.AnnData(np.zeros((len(E), 1), np.float32)); a.obsm["E"] = E
    sc.pp.neighbors(a, n_neighbors=15, use_rep="E", random_state=0)
    for res in [0.05, 0.1, 0.25, 0.5]:
        for seed in range(5):
            sc.tl.leiden(a, resolution=res, key_added="cl", random_state=seed, flavor="igraph", n_iterations=2, directed=False)
            cl = a.obs["cl"].values
            rows.append(dict(code=code, res=res, seed=seed, ncl=len(set(cl)), ari=ARI(hl, cl)))
    # kNN with whitened components (scale-invariant check)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    Ew = (E - E.mean(0)) / E.std(0)
    for nm, X in [("asis", E), ("whitened", Ew)]:
        p = cross_val_predict(KNeighborsClassifier(15), X, hl, cv=cv)
        rows.append(dict(code=code, res=nm, seed=-1, ncl=-1, ari=balanced_accuracy_score(hl, p)))
    # sex decoding within mixed-sex libraries only, kNN restricted per library
    accs = []
    for L in np.unique(lib):
        m = lib == L
        if len(set(sex[m])) < 2 or min((sex[m] == s).sum() for s in set(sex[m])) < 50:
            continue
        p = cross_val_predict(KNeighborsClassifier(15), E[m], sex[m], cv=StratifiedKFold(5, shuffle=True, random_state=0))
        accs.append(balanced_accuracy_score(sex[m], p))
    rows.append(dict(code=code, res="sex_within_lib", seed=-1, ncl=len(accs), ari=float(np.mean(accs))))
    print(code, flush=True)
df = pd.DataFrame(rows)
df.to_csv("leiden_extra.csv", index=False)
g = df[df.seed >= 0].groupby(["code", "res"]).agg(ncl=("ncl", "mean"), ari_mean=("ari", "mean"), ari_sd=("ari", "std"), ari_min=("ari", "min"), ari_max=("ari", "max"))
print(g.round(4).to_string())
print(df[df.seed < 0].round(4).to_string())
