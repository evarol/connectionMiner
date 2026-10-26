"""LaTeX panels for the v3 deck preamble: one notation for all nine representations.

Renders outputs/bernfac_20261020/v3/equations/<name>.png (tectonic + pdftoppm), reusing the
standalone preamble from make_equations.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_equations as me  # noqa: E402

me.OUT = Path(__file__).resolve().parents[4] / "outputs/bernfac_20261020/v3/equations"

PANELS = {
    "v3_notation": r"""
\hd{Data and notation}
\[
\bm X\in\mathbb N^{n\times m}\ \ (\text{UMI counts}),\qquad
\bm Y=\mathbf 1[\bm X\ge1]\in\{0,1\}^{n\times m}\ \ (\text{detection}),\qquad
n=16{,}460\ \text{cells},\ m=19{,}543\ \text{genes}
\]
\hd{Every model gives two things we can compare}
\[
\text{(1) a detection probability per entry: }\ \hat p_{ij}=\widehat{\Pr}(y_{ij}=1),\qquad
\text{(2) a cell representation: }\ \bm e_i\in\mathbb R^{d}
\]
\hd{Shared ingredients}
\[
\psi_{ij}=\text{logit},\quad \sigma(\psi)=\frac{1}{1+e^{-\psi}},\quad
a_i=\text{cell offset (depth)},\quad b_j=\text{gene offset (abundance)},\quad
\bm u_i,\bm v_j\in\mathbb R^{k}\ \text{low-rank factors}
\]
\[
\text{Gaussian priors on all factors: } \bm u_i,\bm v_j\sim\mathcal N(\bm 0,\lambda^{-1}\bm I);\qquad
\lambda,\ k\ (\text{or } d, K)\ \text{chosen on validation entries}
\]
\nt{Processed counts (normalise, log, 2,000 HVGs, PCA) and binary PCA give a representation but no $\hat p_{ij}$; they are the reference points.}
""",
    "v3_models": r"""
\hd{Nine representations, one table}
\renewcommand{\arraystretch}{1.35}
\begin{tabular}{lll}
\textbf{model} & \textbf{probability of detection} & \textbf{cell representation}\\\hline
R1 processed counts & --- & PCA of $\log(1+10^4 x_{ij}/s_i)$\\
R2 binary PCA & --- & PCA of $\bm Y$\\
R3 BernFac logit & $\sigma(\bm u_i^{\top}\bm v_j)$ & SVD of $\bm U\bm V^{\top}$\\
R4 BernFac + offsets & $\sigma(a_i+b_j+\bm u_i^{\top}\bm v_j)$ & SVD of $\bm U\bm V^{\top}$\\
R5 BernFac cloglog & $1-\exp\!\big(-e^{a_i+b_j+\bm u_i^{\top}\bm v_j}\big)$ & SVD of $\bm U\bm V^{\top}$\\
R6 NB counts & $1-(1+\mu_{ij}/\phi_j)^{-\phi_j},\ \ \mu_{ij}=s_ie^{a_i+b_j+\bm u_i^{\top}\bm v_j}$ & SVD of $\bm U\bm V^{\top}$\\
R7 local-factor mixture & $\sum_\ell r_{i\ell}\,\mathbb E\big[\sigma(\psi_{ij})\big],\ \bm\psi_i\sim\mathcal N(\bm\mu_\ell+\bm B_\ell\bm z_i,t_\ell\bm I)$ & mean logits $\mathbb E[\bm\psi_i]$\\
R8 neural manifold & $\sigma\big(g_\theta(\bm z_i)_j\big)$, \ shared decoder $g_\theta$ & $\bm z_i$\\
R9 latent blocks & $\sum_{a,b}R_{ia}C_{jb}\,\sigma(B_{ab})$ & $\bm R\bm B$ (expected block log-odds)\\
\end{tabular}
\par\medskip
\nt{Poisson counts ($\hat p=1-e^{-\mu}$) is fit too and reported in the prediction tables. R4--R6 also give a depth-standardized $\tilde p_{ij}$, with the cell offset set to a common value; its definition comes after the model slides.}
""",
    "v3_inference": r"""
\hd{Fitting: the same Pólya--Gamma trick in four places, Fisher scoring where it does not apply}
\[
\frac{(e^{\psi})^{y}}{1+e^{\psi}}=\tfrac12e^{\kappa\psi}\,\mathbb E_{\omega\sim PG(1,0)}\big[e^{-\omega\psi^2/2}\big],\qquad
\kappa=y-\tfrac12,\qquad \mathbb E[\omega\mid\psi]=f(\psi)=\frac{\tanh(\psi/2)}{2\psi}
\]
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{lll}
\textbf{model} & \textbf{inference} & \textbf{what PG buys}\\\hline
R3, R4 BernFac & PG-EM (MAP) & each factor row is a $k\times k$ ridge solve\\
R7 local-factor mixture & variational EM, $\xi^2=\mathbb E_q[\psi^2]$ & Gaussian posteriors for $\bm z_i$ per component\\
R8 neural manifold & PG-GEM: frozen $w=f(\psi^{\mathrm{old}})$ & quadratic surrogate in the logits; exact head solve\\
R9 latent blocks & mean-field labels + PG block step & $B_{ab}\leftarrow(T_{ab}-N_{ab}/2)/(\lambda+N_{ab}f(B_{ab}))$\\
R5 cloglog, R6 counts & block Fisher scoring + line search & (no PG identity for these likelihoods)\\
\end{tabular}
\par\medskip
\nt{Every method monitors its own objective (MAP objective, variational bound or ELBO) and checks it never decreases. Objectives differ across families, so models are compared on held-out prediction, never on training objectives.}
""",
    "v3_metrics_pred": r"""
\hd{Held-out prediction (same entries for every model)}
\[
\mathcal O=\text{train}\ (95\%),\quad \mathcal V=\text{validation}\ (2.5\%),\quad \mathcal T=\text{test}\ (2.5\%);\qquad
\text{choose }(\lambda,k,\dots)\text{ on }\mathcal V,\ \text{report once on }\mathcal T
\]
\nt{One random split of matrix entries (not $K$-fold): every model is fitted on $\mathcal O$ only, so every cell and gene is seen in training while the scored entries are not. Scores use the observed $\hat p_{ij}$, including each cell's own depth.}
\[
\mathrm{NLL}=-\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}\Big[y_{ij}\log\hat p_{ij}+(1-y_{ij})\log(1-\hat p_{ij})\Big],\qquad
\mathrm{Brier}=\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}(y_{ij}-\hat p_{ij})^2
\]
\[
\mathrm{AUC}=\Pr\big(\hat p_{ij}>\hat p_{i'j'}\ \big|\ y_{ij}=1,\ y_{i'j'}=0\big),\qquad
\text{calibration: }\ \bar y\ \text{vs}\ \bar{\hat p}\ \text{in quantile bins of }\hat p
\]
\hd{Synthetic data, where the truth is known}
\[
\mathrm{KL}(P_{\mathrm{true}}\,\|\,\hat P)=\frac{1}{|\mathcal T|}\sum_{(i,j)\in\mathcal T}\Big[p_{ij}\log\frac{p_{ij}}{\hat p_{ij}}+(1-p_{ij})\log\frac{1-p_{ij}}{1-\hat p_{ij}}\Big]
=\mathbb E[\mathrm{NLL}]-H(P_{\mathrm{true}})
\]
\nt{KL is the held-out NLL we would see with infinitely many replicate measurements, minus a constant that does not depend on the model.}
""",
    "v3_metrics_rep": r"""
\hd{Representation quality (computed in the embedding $\bm e_i$, not in the 2-D UMAP)}
\[
\text{kNN accuracy: } \mathrm{BA}=\frac{1}{C}\sum_{c=1}^{C}\frac{\#\{i\in c:\ \hat c_{15\text{-NN}}(i)=c\}}{\#\{i\in c\}}\ \ (\text{5-fold CV}),\qquad
\text{purity}=\frac{1}{n}\sum_i\frac{\#\{j\in\mathcal N_{15}(i):c_j=c_i\}}{15}
\]
\[
\text{Leiden ARI: agreement with hemilineage, best of 6 resolutions (label-tuned, same for all)},\qquad
\text{silhouette } s_i=\frac{b_i-a_i}{\max(a_i,b_i)}\ \ (\text{cosine})
\]
\hd{Nuisance structure we do not want}
\[
\text{library kNN accuracy (chance }1/21),\qquad
\text{mixing}=\frac{\overline{H\big(\text{library}\mid\mathcal N_{15}(i)\big)}}{\overline{H_{\text{random}}}},\qquad
\text{depth: }\ \max_{c\le2}\big|\rho_{\mathrm{Spearman}}(\bm e_{\cdot c},\log s)\big|
\]
\nt{Good representations score high on hemilineage and low on library and depth. Each metric is also recomputed after regressing $\log s_i$ out of $\bm e_i$.}
""",
    "v3_depth": r"""
\hd{One linear predictor, three parts}
\[
\psi_{ij}\;=\;\underbrace{c_i}_{\text{cell offset: depth}}\;+\;\underbrace{b_j}_{\text{gene abundance}}\;+\;\underbrace{\bm u_i^{\top}\bm v_j}_{\text{cell state}},\qquad
c_i=\begin{cases}a_i & \text{R4 (logit), R5 (cloglog)}\\ \log s_i + a_i & \text{R6 (NB counts)}\end{cases}
\]
\hd{Three outputs from the same fit}
\renewcommand{\arraystretch}{1.35}
\begin{tabular}{lll}
\textbf{output} & \textbf{definition} & \textbf{used for}\\\hline
observed probability & $\hat p_{ij}=g^{-1}\big(c_i+b_j+\bm u_i^{\top}\bm v_j\big)$ & held-out scoring, calibration (each cell at its own depth)\\
depth-standardized probability & $\tilde p_{ij}=g^{-1}\big(\bar c+b_j+\bm u_i^{\top}\bm v_j\big),\ \ \bar c=\mathrm{median}_i\,c_i$ & comparing cells as if sequenced equally deep\\
cell embedding & $\bm e_i=\text{SVD of }\bm U\bm V^{\top}$ (no $c_i$, no $b_j$) & UMAPs and representation metrics\\
\end{tabular}
\par\medskip
\[
g^{-1}(\psi)=\sigma(\psi)\ \text{(R4)},\qquad 1-e^{-e^{\psi}}\ \text{(R5)},\qquad 1-\big(1+e^{\psi}/\phi_j\big)^{-\phi_j}\ \text{(R6)}
\]
\nt{$\tilde p$ is not a prediction of what was observed and is never scored by NLL. R3, R7, R8 and R9 have no $c_i$: depth can only enter through their latent factors, so they have $\hat p$ only. Depth that leaks into $\bm U\bm V^{\top}$ survives standardization.}
""",
    "v3_hyperparams": r"""
\hd{What is tuned, on which grid, and what was chosen (validation entries)}
\renewcommand{\arraystretch}{1.25}
\small
\begin{tabular}{llll}
\textbf{model} & \textbf{real-data grid} & \textbf{chosen} & \textbf{synthetic grid (per world, per seed)}\\\hline
R1 processed counts & none (2,000 HVGs, 50 PCs fixed) & --- & --- \\
R2 binary PCA & none (50 PCs fixed) & --- & --- \\
R3 BernFac logit & $k\in\{10,20,40,80\}\times\lambda\in\{1,3,10,30,100,300\}$ & $k=40,\ \lambda=10$ & $k\in\{5,10,20\}\times\lambda\in\{1,10,30\}$\\
R4 BernFac + offsets & $k\in\{20,40,80\}\times\lambda\in\{3,10,30\}$ & $k=40,\ \lambda=10$ & same as R3\\
R5 BernFac cloglog & $k\in\{0,20,40,80\}\times\lambda\in\{1,3,10,30,100\}$ & $k=40,\ \lambda=30$ & same as R3\\
R6 NB counts & $k\in\{20,40,80,120,160\}\times\lambda\in\{3,10,30,100\}$ & $k=120,\ \lambda=10$ & same as R3\\
Poisson counts & $k\in\{20,40,80\}\times\lambda\in\{3,10,30,100\}$ & $k=80,\ \lambda=10$ (edge) & same as R3\\
R7 local-factor mixture & $K\in\{1,4,8,16\}$, $d\in\{5,\dots,80\}$, residual, $\lambda$ & $K=1,\ d=40$, none & $K\in\{1,4\}\times d\in\{5,10\}$, no residual\\
R8 neural manifold & $d\in\{5,10,20\}$, $\lambda_V\in\{1,10,30\}$, width, head & affine, $d=20$, $\lambda_V=1$ & $d\in\{2,5,10\}\times\lambda_V\in\{1,10,100\}$, affine\\
R9 latent blocks & $K_r,K_c\in\{1,\dots,200\}\times\lambda\in\{0.1,1\}$ & $200\times200$ (edge) & $K_r=K_c\in\{5,10,20,40\}$, $\lambda=1$\\
\end{tabular}
\par\medskip
\nt{Fixed everywhere: NB dispersion $\phi_j$ is fitted by maximum likelihood, not tuned; R4/R5 offset prior $\lambda_0=10^{-3}$; R9 Dirichlet $\alpha=1.1$; kNN uses 15 neighbours. Leiden resolution is the one setting chosen against the labels. R8 restarts were chosen by training objective.}
""",
    "v3_recommendation": r"""
\hd{What to work with}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{p{0.24\linewidth}p{0.70\linewidth}}
\textbf{R6 NB counts} ($k=120$) & Primary choice. Best held-out prediction (test NLL 0.1766), best Leiden ARI (0.67), weakest depth axis; within 0.01 of processed counts on kNN accuracy. Its fitted dispersion runs to the Poisson limit, so its edge over Poisson probably comes from the higher rank (120 vs 80). On synthetic data it is mid-pack for probabilities but recovers cell types well. Its depth-standardized $\tilde p$ is the cleanest depth-free probability (median depth correlation 0.11, markers kept).\\
\textbf{R1 processed counts} & Keep as the reference map: best kNN accuracy and purity, cheapest, but gives no $\hat p_{ij}$.\\
\textbf{R5 cloglog + offsets} & If the object must be binary detection: same prediction as the logit models, half their depth axis, and a clean $\tilde p$ (0.19). R4's $\tilde p$ keeps more depth (0.29), because depth leaks into its factors.\\
\textbf{R8 neural manifold} & Promising (library kNN 0.14, depth off its first axis) but every run hit the time budget; rerun longer before judging.\\
\textbf{R3, R4, R7} & Equivalent predictions on real data; their embeddings are dominated by depth. R7 recovers synthetic probabilities best (3 of 5 worlds) but not cell types.\\
\textbf{R9, Poisson} & Not competitive here; both selected at the edge of their grids.\\
\end{tabular}
\hd{Next steps}
\nt{Unit tests for the offsets and cloglog fits. Longer R8 budget. Check whether the sex signal is sex genes or single-sex libraries. Separate primary-neuron biology from depth before regressing depth out.}
""",
}


if __name__ == "__main__":
    me.OUT.mkdir(parents=True, exist_ok=True)
    for k, body in PANELS.items():
        me.render(k, body)
        print("rendered", k)
