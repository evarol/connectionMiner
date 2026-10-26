"""How every method was tuned on the synthetic benchmark (reads stored runs only, no refitting).

    python syn_select_fig.py

Reads v3/synthetic/runs/<G>_s<seed>_<method>.json (selected config + full validation grid) and writes
    v3/tables/syn_selection.csv     one row per method x world x seed
    v3/figures/fig_syn_select.png   validation-NLL grid per method (heatmap: configs x worlds)

Heatmap value: mean over the 3 seeds of the validation Bernoulli NLL of each config, minus the
lowest mean in that (method, world) column, in units of 1e-3 nats (0 = best config).
Outlined cells = the config most seeds selected (label "n/3" = seed agreement). An orange outline
means the selected config sits at the edge of a grid axis.
The runs store validation NLL per config but test KL only for the selected config, so the
oracle-best config by test KL cannot be read from disk (oracle_* columns are left empty).
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import v3style as vs  # noqa: E402

RUNS = vs.V3 / "synthetic" / "runs"
TABLES = vs.V3 / "tables"
WORLDS = ["G1", "G2", "G3", "G4", "G5"]
WTITLE = {"G1": "G1\nlow-rank", "G2": "G2\nNB counts", "G3": "G3\ncell types", "G4": "G4\nmanifold",
          "G5": "G5\nblocks"}
METHODS = ["R3", "R4", "R5", "R6", "PO", "R7", "R8", "R9"]
NAMES = {**{c: vs.name(c) for c in vs.METHODS}, "PO": "Poisson counts"}
COLOURS = {**{c: vs.colour(c) for c in vs.METHODS}, "PO": "#8C6D1F"}

# exact grids from syn_methods.py, in the order they are fitted (axis name -> values)
BF_AXES = {"k": [5, 10, 20], "lam": [30.0, 10.0, 1.0]}
AXES = {"R3": BF_AXES, "R4": BF_AXES, "R5": BF_AXES, "R6": BF_AXES, "PO": BF_AXES,
        "R7": {"K": [1, 4], "d": [5, 10]},
        "R8": {"d": [2, 5, 10], "lamV": [1.0, 10.0, 100.0]},
        "R9": {"Kr": [5, 10, 20, 40]}}          # Kc = Kr (tied), so one axis
PRETTY = {"lam": "λ", "lamV": "λ_V", "Kr": "K"}


def parse(cfg):
    return {k: float(v) for k, v in re.findall(r"(\w+)=([\d.eE+-]+)", cfg)}


def label(method, cfg):
    p = parse(cfg)
    if method == "R9":
        return f"K_r=K_c={int(p['Kr'])}"
    return ", ".join(f"{PRETTY.get(k, k)}={v:g}" for k, v in p.items())


def edge_info(method, cfg):
    """List of 'axis=min/max' for every grid axis (with >= 3 values or 2) where cfg sits at an end."""
    p = parse(cfg)
    out = []
    for ax, vals in AXES[method].items():
        lo, hi = min(vals), max(vals)
        if p[ax] == lo:
            out.append(f"{ax}=min")
        elif p[ax] == hi:
            out.append(f"{ax}=max")
    return out


def load():
    rows, grid = [], []
    for m in METHODS:
        for g in WORLDS:
            for s in range(3):
                d = json.load(open(RUNS / f"{g}_s{s}_{m}.json"))
                rr = json.loads(d["rows"]) if isinstance(d["rows"], str) else d["rows"]
                (r,) = [x for x in rr if x["method"] == m]
                gl = json.loads(r["grid"])
                for c, v, sec, it, cv in gl:
                    grid.append(dict(method=m, world=g, seed=s, config=c, val_nll=v, seconds=sec, iterations=it,
                                     converged=cv))
                e = edge_info(m, r["config"])
                rows.append(dict(method=m, world=g, seed=s, grid_size=len(gl), selected_config=r["config"],
                                 at_grid_edge=bool(e), edge_axes=";".join(e),
                                 val_nll_selected=r["val_nll"], test_kl_selected=r["kl"],
                                 test_nll_selected=r["test_nll"],
                                 n_failed_configs=sum(v is None for _, v, *_ in gl),
                                 oracle_config_by_test_kl="", oracle_test_kl=""))
    return pd.DataFrame(rows), pd.DataFrame(grid)


def main():
    sel, grid = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    sel.to_csv(TABLES / "syn_selection.csv", index=False)
    grid.to_csv(TABLES / "syn_selection_grid.csv", index=False)

    mean = grid.groupby(["method", "world", "config"], sort=False)["val_nll"].mean().reset_index()
    mean["delta"] = 1e3 * (mean["val_nll"] - mean.groupby(["method", "world"])["val_nll"].transform("min"))

    fs.apply_publication_style()
    fig = plt.figure(figsize=(18, 10.1))
    gs = fig.add_gridspec(2, 4, height_ratios=[9, 9], hspace=0.42, wspace=1.05, left=0.085, right=0.93,
                          top=0.855, bottom=0.06)
    norm = LogNorm(vmin=0.1, vmax=1000)
    cmap = plt.get_cmap(fs.CMAP_SEQ_R).copy()
    pos = {m: (i // 4, i % 4) for i, m in enumerate(METHODS)}
    for m in METHODS:
        r, c = pos[m]
        cfgs = list(dict.fromkeys(grid.loc[grid.method == m, "config"]))
        M = np.full((len(cfgs), len(WORLDS)), np.nan)
        for j, g in enumerate(WORLDS):
            mm = mean[(mean.method == m) & (mean.world == g)].set_index("config")["delta"]
            M[:, j] = [mm[k] for k in cfgs]
        # R7/R9 have 4 rows: put them in a shorter axes at the top of the 9-row slot
        sub = gs[r, c].subgridspec(9, 1)
        ax = fig.add_subplot(sub[:len(cfgs), 0])
        ax.imshow(np.clip(M, 0.1, None), cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                txt = "0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")
                dark = norm(max(v, 0.1)) > 0.55
                ax.text(j, i, txt, ha="center", va="center", fontsize=fs.FS_TINY,
                        color="white" if dark else "#272727")
        agree = []
        for j, g in enumerate(WORLDS):
            picks = sel[(sel.method == m) & (sel.world == g)]["selected_config"].tolist()
            cnt = Counter(picks)
            for cfg, n in cnt.items():
                i = cfgs.index(cfg)
                edge = bool(edge_info(m, cfg))
                mode = n == max(cnt.values()) and cfg == cnt.most_common(1)[0][0]
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False,
                                       ec="#E66100" if edge else "black", lw=3.0 if mode else 1.6,
                                       ls="-" if mode else (0, (2, 1.5))))
            agree.append(f"{cnt.most_common(1)[0][1]}/3")
        ax.set_xticks(range(len(WORLDS)))
        ax.set_xticklabels([f"{g}\n{a}" for g, a in zip(WORLDS, agree)], fontsize=fs.FS_TINY + 0.5)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks(range(len(cfgs)))
        ax.set_yticklabels([label(m, k) for k in cfgs], fontsize=fs.FS_TINY + 0.5)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{m} {NAMES[m]}", color=COLOURS[m], fontweight="bold", fontsize=fs.FS_TITLE - 1,
                     pad=34, loc="left", x=-0.02)
    # shared colour bar + legend
    cax = fig.add_axes([0.945, 0.30, 0.012, 0.42])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label("validation ΔNLL vs best config in world  (×10⁻³ nats)", fontsize=fs.FS_SMALL)
    cb.ax.tick_params(labelsize=fs.FS_TINY)
    fig.text(0.01, 1.0, "How each method was tuned on the synthetic worlds (mean of 3 seeds; "
             "column header n/3 = seeds that chose the boxed config)", fontsize=fs.FS_SMALL + 1, va="top")
    fig.text(0.01, 0.968, "Box: selected config (thick = most common, dashed = other seeds).   "
             "Black box: interior of grid.   Orange box: at the edge of a grid axis.",
             fontsize=fs.FS_SMALL, va="top", color=fs.PALETTE["grey"])
    fig.text(0.01, 0.005, "R7 grid has two values per axis, so every R7 config is at an edge.  "
             "R9 ties K_c = K_r.  Selection = lowest validation Bernoulli NLL; no refits.",
             fontsize=fs.FS_SMALL, va="bottom", color=fs.PALETTE["grey"])
    fs.finalize_figure(fig, vs.FIGS / "fig_syn_select")

    # console summary
    for m in METHODS:
        s = sel[sel.method == m]
        ag = [Counter(s[s.world == g].selected_config).most_common(1)[0][1] for g in WORLDS]
        print(f"{m}: edge {int(s.at_grid_edge.sum())}/{len(s)}  agreement {ag}  "
              f"edge axes {Counter(x for e in s.edge_axes for x in e.split(';') if x)}")


if __name__ == "__main__":
    main()
