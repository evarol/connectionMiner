"""Schematic figures for the BernFac deck (matplotlib, vector).

1. schematic_model.png    : counts -> binary Y -> U, V (Gaussian priors) -> sigmoid -> P
2. schematic_pgem.png     : the PG-EM loop (E-step weights, M-step ridge solves)
3. schematic_realdesign.png: 4 libraries x 5 hemilineages balanced subset and the lambda x k grid
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bernfac import pg_mean, sigmoid

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/bernfac_20261020/data"
OUT = ROOT / "outputs/bernfac_20261020/schematics"
BLUE, ORANGE, GREY, DARK = "#0072B2", "#E69F00", "#9a9a9a", "#222222"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})


def arrow(ax, p, q, text=None, color=DARK, dy=0.035, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, text, ha="center",
                va="bottom", fontsize=10, color=color)


def box(ax, x, y, w, h, text, fc="#f4f4f4", ec=GREY, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def matrix(fig, rect, M, cmap, title=None, vmin=None, vmax=None):
    a = fig.add_axes(rect)
    a.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color(GREY)
    if title:
        a.set_title(title, fontsize=11)
    return a


def schematic_model():
    C = np.load(DATA / "counts.npy")[:60, :30]
    Y = (C >= 1).astype(float)
    rng = np.random.default_rng(0)
    U, V = rng.standard_normal((60, 3)), rng.standard_normal((30, 3))
    fig = plt.figure(figsize=(14, 4.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    matrix(fig, [0.02, 0.22, 0.11, 0.56], np.log1p(C), "magma_r", "UMI counts\n(cells x genes)")
    arrow(ax, (0.14, 0.5), (0.2, 0.5), "count >= 1")
    matrix(fig, [0.21, 0.22, 0.11, 0.56], Y, "Greys", "binary Y", 0, 1)
    ax.text(0.265, 0.13, r"$y_{ij}\in\{0,1\}$", ha="center", fontsize=12)
    arrow(ax, (0.335, 0.5), (0.395, 0.5), "infer")
    # factors with priors
    matrix(fig, [0.41, 0.22, 0.03, 0.56], U, "RdBu_r", "U", -2.5, 2.5)
    ax.text(0.425, 0.13, r"$u_i\sim\mathcal{N}(0,\lambda^{-1}I_k)$", ha="center", fontsize=11)
    matrix(fig, [0.47, 0.62, 0.11, 0.06], V.T, "RdBu_r", None, -2.5, 2.5)
    ax.text(0.525, 0.71, r"$V^\top$", ha="center", fontsize=12)
    ax.text(0.525, 0.53, r"$v_j\sim\mathcal{N}(0,\lambda^{-1}I_k)$", ha="center", fontsize=11)
    ax.text(0.525, 0.36, r"$\psi_{ij}=u_i^\top v_j$" "\n(rank k)", ha="center", fontsize=12)
    # sigmoid inset
    s = fig.add_axes([0.655, 0.32, 0.09, 0.36])
    x = np.linspace(-6, 6, 200)
    s.plot(x, sigmoid(x), color=BLUE, lw=2.2)
    s.set_xlabel(r"$\psi$"); s.set_ylabel(r"$\sigma(\psi)$"); s.set_title("sigmoid", fontsize=11)
    s.spines[["top", "right"]].set_visible(False)
    arrow(ax, (0.75, 0.5), (0.79, 0.5))
    matrix(fig, [0.80, 0.22, 0.11, 0.56], sigmoid(U @ V.T), "viridis",
           "latent P\n" r"$p_{ij}=\sigma(u_i^\top v_j)$", 0, 1)
    ax.text(0.855, 0.13, r"$y_{ij}\sim\mathrm{Bernoulli}(p_{ij})$", ha="center", fontsize=11)
    ax.add_patch(FancyArrowPatch((0.855, 0.84), (0.265, 0.84), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.4, color=ORANGE,
                                 connectionstyle="arc3,rad=0.12"))
    ax.text(0.56, 0.95, "generative model: each binary entry is one Bernoulli trial",
            ha="center", color=ORANGE, fontsize=11)
    ax.text(0.95, 0.5, "goal:\ncontinuous\n[0,1]\nrepresentation\nof a {0,1}\nmatrix",
            ha="center", va="center", fontsize=10.5, color=DARK)
    fig.savefig(OUT / "schematic_model.png", dpi=200); fig.savefig(OUT / "schematic_model.pdf")


def schematic_pgem():
    fig = plt.figure(figsize=(14, 5.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.93, "Polya-Gamma identity (Polson, Scott & Windle 2013)", fontsize=13,
            weight="bold")
    ax.text(0.02, 0.81, r"$\frac{e^{y\psi}}{1+e^{\psi}}=\frac{1}{2}e^{\kappa\psi}\,"
            r"\mathbb{E}_{\omega\sim PG(1,0)}\!\left[e^{-\omega\psi^2/2}\right],"
            r"\quad\kappa=y-\frac{1}{2}$", fontsize=15)
    ax.text(0.02, 0.69, "given $\\omega$, the logistic likelihood is a Gaussian in $\\psi$\n"
            "-> each factor row is a ridge-weighted least-squares problem", fontsize=11)
    # PG mean curve
    a = fig.add_axes([0.05, 0.1, 0.22, 0.42])
    x = np.linspace(-10, 10, 400)
    a.plot(x, pg_mean(x), color=ORANGE, lw=2.2)
    a.set_xlabel(r"logit $\psi_{ij}$"); a.set_ylabel(r"$\bar\omega_{ij}=E[\omega|\psi]$")
    a.set_title(r"E-step weight $\tanh(\psi/2)/2\psi$", fontsize=11)
    a.spines[["top", "right"]].set_visible(False)
    a.text(0, 0.255, "1/4 at psi=0", ha="center", fontsize=9)
    # loop
    box(ax, 0.38, 0.6, 0.27, 0.2, "E-step  (old U, V)\n" r"$W=\bar\omega(UV^\top)$" "\nfrozen for the whole M-step",
        fc="#fdf1dc", ec=ORANGE)
    box(ax, 0.72, 0.6, 0.26, 0.2, "M-step A  (rows)\n"
        r"$[\lambda I+\sum_j w_{ij}v_jv_j^\top]\,u_i=\sum_j\kappa_{ij}v_j$", fc="#e3eef7", ec=BLUE)
    box(ax, 0.72, 0.22, 0.26, 0.2, "M-step B  (cols, new U, same W)\n"
        r"$[\lambda I+\sum_i w_{ij}u_iu_i^\top]\,v_j=\sum_i\kappa_{ij}u_i$", fc="#e3eef7", ec=BLUE)
    box(ax, 0.38, 0.22, 0.27, 0.2, "monitor ORIGINAL objective\n"
        r"$J=\sum y\psi-\log(1+e^\psi)-\frac{\lambda}{2}(\|U\|^2+\|V\|^2)$"
        "\nnondecreasing; stop on rel. change", fc="#eeeeee")
    arrow(ax, (0.655, 0.7), (0.715, 0.7))
    arrow(ax, (0.85, 0.595), (0.85, 0.425))
    arrow(ax, (0.715, 0.32), (0.655, 0.32))
    arrow(ax, (0.515, 0.425), (0.515, 0.595))
    ax.text(0.68, 0.08, "k x k linear solves only; no PG sampling; nonconvex -> local optimum, "
            "multiple restarts;\nfactors identifiable only up to rotation (UR, VR): compare P, not U",
            ha="center", fontsize=10.5, color=DARK)
    fig.savefig(OUT / "schematic_pgem.png", dpi=200); fig.savefig(OUT / "schematic_pgem.pdf")


def schematic_realdesign():
    import pandas as pd
    cells = pd.read_csv(DATA / "cells.csv")
    libs = ["SITTA9", "SITTB9", "SITTG8", "SITTC8"]
    hls = ["14A", "04B", "09A", "12B", "03A"]
    fig = plt.figure(figsize=(14, 4.8))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.02, 0.92, "Real data: motor preMN, 24 h APF (raw UMI counts, 16,460 cells)",
            fontsize=13, weight="bold")
    g = fig.add_axes([0.1, 0.12, 0.22, 0.62])
    tab = pd.crosstab(cells.library, cells.hemilineage).loc[libs, hls]
    g.imshow(tab.values, cmap="Blues", vmin=0, vmax=80)
    for i in range(4):
        for j in range(5):
            g.text(j, i, tab.values[i, j], ha="center", va="center", fontsize=11)
    sex = cells.groupby("library").Sex.first()
    g.set_yticks(range(4), [f"{l} ({sex[l]})" for l in libs]); g.set_xticks(range(5), hls)
    g.set_xlabel("hemilineage (biology)"); g.set_ylabel("library (batch)")
    g.set_title("balanced 4 x 5 x 50 = 1000 cells", fontsize=11)
    box(ax, 0.37, 0.42, 0.19, 0.3, "100 genes\ntop HVGs with\n5-95% detection\n\nY = 1[count >= 1]\n(sensitivity: >= 2)",
        fc="#f4f4f4")
    arrow(ax, (0.565, 0.57), (0.62, 0.57), "fit BernFac")
    gg = fig.add_axes([0.64, 0.14, 0.3, 0.62])
    lam = [0.03, 0.1, 0.3, 1, 3, 10]; ks = [1, 2, 3, 5, 10, 20]
    gg.imshow(np.zeros((6, 6)), cmap="Greys", vmin=0, vmax=1)
    gg.set_xticks(range(6), ks); gg.set_yticks(range(6), lam)
    gg.set_xticks(np.arange(-0.5, 6), minor=True); gg.set_yticks(np.arange(-0.5, 6), minor=True)
    gg.grid(which="minor", color=GREY); gg.tick_params(which="minor", length=0)
    gg.set_xlabel("rank k"); gg.set_ylabel(r"prior precision $\lambda$")
    gg.set_title("36 fits: SVD init + 3 random restarts", fontsize=11)
    ax.text(0.46, 0.2, "batch and biology are\northogonal by design:\ncan P keep hemilineage\n"
            "and lose library?", ha="center", fontsize=10.5, color=BLUE)
    fig.savefig(OUT / "schematic_realdesign.png", dpi=200); fig.savefig(OUT / "schematic_realdesign.pdf")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    schematic_model(); schematic_pgem(); schematic_realdesign()
