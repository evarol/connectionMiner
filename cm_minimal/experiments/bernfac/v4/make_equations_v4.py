"""LaTeX panels for deck v4 (after the strict review): corrected metrics, exact tuning table,
errata and the revised recommendation. Unchanged v3 panels are reused from v3/equations.

Renders outputs/bernfac_20261020/v4/equations/<name>.png via make_equations.render.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_equations as me  # noqa: E402

me.OUT = Path(__file__).resolve().parents[4] / "outputs/bernfac_20261020/v4/equations"

PANELS = {
    "v4_metrics_pred": r"""
\hd{Held-out prediction (same entries for every model)}
\[
\mathcal O=\text{train}\ (95\%),\quad \mathcal V=\text{validation}\ (2.5\%),\quad \mathcal T=\text{test}\ (2.5\%)\quad
\text{(one random split of matrix entries, not cells)}
\]
\[
\mathrm{NLL}=-\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}\Big[y_{ij}\log\hat p_{ij}+(1-y_{ij})\log(1-\hat p_{ij})\Big],\qquad
\mathrm{Brier}=\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}(y_{ij}-\hat p_{ij})^2
\]
\hd{Uncertainty: paired two-way bootstrap over cells and genes}
\[
\Delta=\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}\big(\ell^{A}_{ij}-\ell^{B}_{ij}\big),\qquad
\Delta^{*}=\frac{\sum_{(i,j)\in\mathcal T}w_iw'_j(\ell^{A}_{ij}-\ell^{B}_{ij})}{\sum_{(i,j)\in\mathcal T}w_iw'_j},\quad
w_i,w'_j\stackrel{iid}{\sim}\mathrm{Poisson}(1)
\]
\hd{Synthetic data, where the truth is known}
\[
\mathrm{KL}(P_{\mathrm{true}}\,\|\,\hat P)=\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}\Big[p_{ij}\log\frac{p_{ij}}{\hat p_{ij}}+(1-p_{ij})\log\frac{1-p_{ij}}{1-\hat p_{ij}}\Big]
=\mathbb E[\mathrm{NLL}]-H(P_{\mathrm{true}})
\]
\nt{Settings are chosen on $\mathcal V$. Test NLL was also computed for every configuration, and the R3--R5 grids were first chosen on $\mathcal V\cup\mathcal T$; re-selecting on $\mathcal V$ alone gives the same configurations. The bootstrap resamples cells and genes, so it captures their correlation, but not refit or split variance.}
""",
    "v4_metrics_rep": r"""
\hd{One protocol for every cell representation}
\[
\bm E\;\xrightarrow{\ \text{centre}\ }\;\bm E-\bm 1\bar{\bm e}^{\top}=\bm A\bm S\bm B^{\top}\ \ (\text{SVD})\;\longrightarrow\;
\underbrace{\bm A_{:,1:d}\bm S_{1:d}}_{\text{unwhitened}}\ \ \text{or}\ \ \underbrace{\sqrt n\,\bm A_{:,1:d}}_{\text{whitened}},\qquad d\in\{20,40\}
\]
\hd{Biology}
\[
\text{linear probe (logistic regression, 5-fold, balanced accuracy)},\qquad
\text{15-NN balanced accuracy},\qquad
\text{Leiden ARI/AMI at exactly 23 clusters, 5 seeds}
\]
\hd{Nuisance}
\[
R^2_{\text{depth}}=\text{CV-}R^2\big(\log s_i\sim\text{top-10 axes}\big)\ \text{within hemilineage}\times\text{birth time},\qquad
\text{library 15-NN accuracy within hemilineage}
\]
\nt{The probe measures whether information is there; kNN and Leiden measure geometry, which depends on scaling. Depth is partly biology: hemilineage and birth time explain 33\% of the variance of $\log s_i$, so depth metrics are taken within those strata. The Sex label is almost exactly the roX1/roX2 signal (AUC 0.998) and is not used as validation.}
""",
    "v4_hyperparams": r"""
\hd{What is tuned, on which grid, and what validation chose (real data)}
\renewcommand{\arraystretch}{1.22}
\small
\begin{tabular}{llll}
\textbf{model} & \textbf{grid run} & \textbf{chosen} & \textbf{status}\\\hline
R1, R2 & none (R1: 2,000 HVGs, 50 PCs; R2: 50 PCs) & --- & ---\\
R3 BernFac logit & $k\in\{10,20,40,80\}\times\lambda\in\{1,\dots,300\}$ & $k=40,\ \lambda=10$ & interior\\
R4 BernFac + offsets & $k\in\{20,40,80,120\}\times\lambda\in\{3,10,30,100\}$ (partial) & $k=40,\ \lambda=10$ & interior\\
R5 BernFac cloglog & $k\in\{0,20,40,80\}\times\lambda\in\{1,\dots,100\}$ & $k=40,\ \lambda=30$ & interior\\
R6 NB counts & $k\in\{20,\dots,160\}\times\lambda\in\{3,\dots,100\}$ (partial) & $k=120,\ \lambda=10$ & $\lambda$ at edge for $k=120$\\
Poisson counts & $k\in\{20,\dots,160\}\times\lambda\in\{3,\dots,100\}$ (partial) & $k=120,\ \lambda=30$ & edge\\
R7 local-factor mixture & $K\in\{1,4,8,16\}$, $d\in\{5,\dots,80\}$, residual, $\lambda$ & $K=1,\ d=40$ & interior in $d$\\
R8 neural manifold & $d\in\{5,10,20,40\}$, $\lambda_V\in\{1,10,30\}$, width, head & $d=40,\ \lambda_V=10$ & corner; not converged\\
R9 latent blocks & $K_r,K_c\in\{1,\dots,200\}\times\lambda\in\{0.1,1\}$ & $200\times200$ & edge\\
\end{tabular}
\par\medskip
\nt{Fixed: NB dispersion $\phi_j$ fitted per gene (not tuned); offset prior $\lambda_0=10^{-3}$; R9 Dirichlet $\alpha=1.1$; R8 budget 250 outer iterations. Synthetic grids include the true rank ($k\in\{4,8,16\}$), and validation picks the KL-optimal setting almost everywhere.}
""",
    "v4_errata": r"""
\hd{What changed after the strict review (v3 $\rightarrow$ v4)}
\renewcommand{\arraystretch}{1.28}
\small
\begin{tabular}{p{0.40\linewidth}p{0.55\linewidth}}
\textbf{v3 said} & \textbf{v4 finds}\\\hline
NB's edge over Poisson comes from rank; $\phi$ runs to the Poisson limit & NB beats Poisson at every matched rank ($4$--$9\times10^{-3}$ nats); the gain is zeros of bursty, frequently detected genes. The large median $\phi$ comes from rare genes where $\phi$ cannot be estimated.\\
Five binary models tie & R7 $<$ R4 $<$ R3 are resolvable but within $0.2\times10^{-3}$; R4, R5, R8 are statistically tied.\\
Binary embeddings lose cell-type signal & All carry the information (linear probe 0.97--0.99); the gap was coordinate scaling.\\
R6 best Leiden ARI (0.67 vs 0.65) & Single seed, label-tuned resolution. At 23 clusters, 5 seeds: R6 best only at $d=40$ whitened (0.74); R1 leads at $d=20$.\\
R8 keeps depth off its first axis & Depth is recoverable from R8's top axes ($R^2\approx0.9$).\\
Depth-standardized $\tilde p$ is ``depth-free''; markers keep 84--90\% & $\tilde p$ is depth-\emph{adjusted} (R6 $R^2$ 0.41 vs 0.61 for log counts); marker variance kept 98--102\% (84--90\% was $\eta^2$).\\
Sex separates in every method; maybe single-sex libraries & Sex label tracks roX expression (AUC 0.998); within mixed libraries only all-gene factor models decode it.\\
R7 wins the NB synthetic world & Grid artefact: with the true rank in the grid, R6 wins its own world.\\
Depth shifts logits of rare genes more & Backwards: abundant genes shift most.\\
\end{tabular}
""",
    "v4_recommendation": r"""
\hd{What to work with, and how sure we are}
\renewcommand{\arraystretch}{1.28}
\begin{tabular}{p{0.25\linewidth}p{0.70\linewidth}}
\textbf{R6 NB counts} ($k=120$) & \textbf{Primary choice for probabilities.} Best held-out prediction by $1.0\times10^{-3}$ nats (95\% CI excludes 0 against every model; the lead over R4 holds in every depth decile). About 230 bursty wiring and TF genes carry half the gain. Wins its own synthetic world and recovers dispersion.\\
\textbf{R6 embedding} & Best Leiden agreement at $d=40$ whitened (ARI 0.74) and the least depth among factor models; this ranking depends on $d$ and scaling.\\
\textbf{R6 $\tilde p$} & Use for depth-adjusted displays. Recovers the true adjusted probabilities on synthetic data; on real data it also removes much birth-time variance, so over-correction cannot be excluded.\\
\textbf{R1 / scVI} & kNN maps as good as R6 (0.98) with the least library signal; no $\hat p_{ij}$.\\
\textbf{R5, R7, R8} & Binary-detection alternatives; differences among them are $\le0.2\times10^{-3}$ nats. R8 is at a grid corner and not converged.\\
\textbf{R9, Poisson} & Not competitive.\\
\end{tabular}
\hd{Still open}
\nt{Refit and split variance (one fit per model); NB $k=160$ with stronger priors; R8 beyond its grid corner; a depth model that separates birth-time biology from sequencing depth.}
""",
}


if __name__ == "__main__":
    me.OUT.mkdir(parents=True, exist_ok=True)
    for k, body in PANELS.items():
        me.render(k, body)
        print("rendered", k)
