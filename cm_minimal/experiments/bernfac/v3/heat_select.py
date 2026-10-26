"""Choose the cells and genes for the v3 heatmap snippets (cached to v3/cache/heat_selection.json).

Cells: 5 hemilineages = the 3 best separated + the most mutually confused pair, judged by the
15-NN hemilineage confusion averaged over the nine embeddings (R1-R9); within each, 12 low-depth
(bottom quartile of total UMI) and 12 high-depth (top quartile) cells, random with a fixed seed.
Genes (all rates on TRAIN entries only):
  markers   4 per hemilineage: largest detection-rate enrichment vs the mean of the other
            22 hemilineages, rate in the hemilineage >= 0.5
  depth     8 broadly detected genes (train rate 0.25-0.75) whose detection rises most from the
            bottom to the top depth quartile within hemilineage (mean log-odds difference)
  rare      4 genes with train rate 0.004-0.02 overall but the largest rate in one chosen
            hemilineage (rare but structured)
  housekeeping 2 genes with the highest overall rate + 2 near-ubiquitous (rate 0.90-0.95) with the
            smallest depth sensitivity
"""
import json, os, sys
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import predictors as pr

rng = np.random.default_rng(0)
cells = pd.read_csv(pr.ROOT / "outputs/connectionminer_ready_inputs/bundles/motor/premn_24h/cells.csv.gz")
keep = np.load(pr.DATA / "genes_kept.npy")
genes = pd.read_csv(pr.ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/genes.csv").gene_id.values[keep]
X = sp.load_npz(pr.ROOT / "outputs/multisystem_data_audit/bundles/motor_premn_24h/G_prime.npz").tocsc()[:, keep].tocsr()
depth = np.asarray(X.sum(1)).ravel()
Y = np.load(pr.DATA / "Y.npy")
Tr = ~np.load(pr.DATA / "heldout.npy")
hl = cells.hemilineage.values
HLS = sorted(set(hl))

# ---- kNN hemilineage confusion over the nine embeddings
conf = {}
for c in [f"R{i}" for i in range(1, 10)]:
    E = np.load(pr.STUDY / f"full/umap/cache/{c}_embedding.npz")["E"]
    nn = NearestNeighbors(n_neighbors=16).fit(E)
    idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    lab = pd.Series(hl[idx].ravel())
    C = pd.crosstab(np.repeat(hl, 15), hl[idx].ravel(), normalize="index").reindex(index=HLS, columns=HLS, fill_value=0)
    conf[c] = C
    C.to_csv(pr.CACHE / f"knn_conf_{c}.csv")
Cm = sum(conf.values()) / len(conf)
acc = pd.Series(np.diag(Cm), index=HLS)
mut = (Cm + Cm.T).values / 2
np.fill_diagonal(mut, 0)
a, b = np.unravel_index(mut.argmax(), mut.shape)
pair = [HLS[a], HLS[b]]
sep = [h for h in acc.sort_values(ascending=False).index if h not in pair][:3]
chosen = sep + pair
print("kNN accuracy (mean over R1-R9):\n", acc.sort_values().round(3).to_string())
print("chosen separated", sep, "confused pair", pair, f"mutual conf {mut[a, b]:.3f}")
per_method = {c: dict(acc_sep=float(np.mean([conf[c].loc[h, h] for h in sep])),
                      acc_pair=float(np.mean([conf[c].loc[h, h] for h in pair])),
                      conf_pair=float((conf[c].loc[pair[0], pair[1]] + conf[c].loc[pair[1], pair[0]]) / 2))
              for c in conf}

# ---- train detection rates per hemilineage and per hemilineage x depth quartile
def rate(rows):
    return (Y[rows] * Tr[rows]).sum(0) / np.maximum(Tr[rows].sum(0), 1)

rate_hl = {h: rate(np.flatnonzero(hl == h)) for h in HLS}
overall = (Y * Tr).sum(0) / Tr.sum(0)
lo_hi = {}
dq = np.zeros(len(hl), int)
for h in HLS:
    r = np.flatnonzero(hl == h)
    q1, q3 = np.quantile(depth[r], [0.25, 0.75])
    lo, hi = r[depth[r] <= q1], r[depth[r] >= q3]
    dq[lo], dq[hi] = 1, 2
    lo_hi[h] = (rate(lo), rate(hi))
lg = lambda p: np.log((p + 0.01) / (1 - p + 0.01))
dsens = np.mean([lg(lo_hi[h][1]) - lg(lo_hi[h][0]) for h in HLS], 0)

markers = {}
used = set()
for h in chosen:
    others = np.mean([rate_hl[g] for g in HLS if g != h], 0)
    score = np.where(rate_hl[h] >= 0.5, rate_hl[h] - others, -np.inf)
    order = [j for j in np.argsort(-score) if j not in used][:4]
    markers[h] = [int(j) for j in order]; used.update(order)
broad = np.flatnonzero((overall > 0.25) & (overall < 0.75))
depth_genes = [int(j) for j in broad[np.argsort(-dsens[broad])] if j not in used][:8]
used.update(depth_genes)
rare_c = np.flatnonzero((overall > 0.004) & (overall < 0.02))
rmax = np.max([rate_hl[h][rare_c] for h in chosen], 0)
rare = [int(j) for j in rare_c[np.argsort(-rmax)] if j not in used][:4]
used.update(rare)
hk = [int(j) for j in np.argsort(-overall) if j not in used][:2]
near = np.flatnonzero((overall > 0.9) & (overall < 0.95))            # near-ubiquitous, not saturated
hk += [int(j) for j in near[np.argsort(np.abs(dsens[near]))] if j not in used][:2]

cell_groups = []
for h in chosen:
    for d, nm in ((1, "low"), (2, "high")):
        r = np.flatnonzero((hl == h) & (dq == d))
        pick = np.sort(rng.choice(r, 12, replace=False))
        pick = pick[np.argsort(depth[pick])]
        cell_groups.append(dict(hemilineage=h, depth=nm, rows=[int(i) for i in pick]))

sel = dict(hemilineages=chosen, separated=sep, confused_pair=pair, knn_acc=acc[chosen].round(4).to_dict(),
           knn_per_method=per_method, cell_groups=cell_groups,
           gene_groups=dict(**{f"marker {h}": markers[h] for h in chosen}, depth=depth_genes, rare=rare,
                            housekeeping=hk),
           gene_names={int(j): str(genes[j]) for j in sorted(used | set(hk))},
           depth_quartile_rule="within hemilineage, bottom vs top quartile of total UMI (all kept genes)")
(pr.CACHE / "heat_selection.json").write_text(json.dumps(sel, indent=1))
np.save(pr.CACHE / "cell_depth.npy", depth)
np.save(pr.CACHE / "gene_train_rate.npy", overall)
np.save(pr.CACHE / "depth_quartile.npy", dq)
for k, v in sel["gene_groups"].items():
    print(k, [(genes[j], round(float(overall[j]), 3), round(float(dsens[j]), 2)) for j in v])
