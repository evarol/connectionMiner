"""Typeset the BernFac equations with LaTeX (tectonic) as slide-ready PNG panels.

Each panel is a standalone LaTeX document -> PDF (tectonic) -> PNG (pdftoppm, 300 dpi).
Output: outputs/bernfac_20261020/equations/eq_*.png
"""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/bernfac_20261020/equations"

PREAMBLE = r"""\documentclass[border=14pt,varwidth=24cm]{standalone}
\usepackage{amsmath,amssymb,bm,xcolor}
\usepackage[scaled]{helvet}
\definecolor{bmain}{HTML}{0F4D92}\definecolor{orng}{HTML}{C9731E}\definecolor{grey}{HTML}{555555}
\newcommand{\hd}[1]{{\sffamily\bfseries\color{bmain}#1}\par\smallskip}
\newcommand{\nt}[1]{{\sffamily\small\color{grey}#1}}
\begin{document}\large
"""

PANELS = {
    "eq_model": r"""
\hd{Generative model}
\[
y_{ij}\mid \bm u_i,\bm v_j \;\sim\; \mathrm{Bernoulli}\!\big(\sigma(\psi_{ij})\big),\qquad
\psi_{ij}=\bm u_i^{\top}\bm v_j,\qquad \sigma(x)=\frac{1}{1+e^{-x}}
\]
\[
\bm u_i\sim\mathcal N\!\big(\bm 0,\lambda_u^{-1}\bm I_k\big),\qquad
\bm v_j\sim\mathcal N\!\big(\bm 0,\lambda_v^{-1}\bm I_k\big),\qquad
\bm U\in\mathbb R^{n\times k},\;\bm V\in\mathbb R^{m\times k}
\]
\hd{MAP objective (what we maximise and monitor)}
\[
J(\bm U,\bm V)=\sum_{(i,j)\in\mathcal O}\Big[y_{ij}\psi_{ij}-\log\!\big(1+e^{\psi_{ij}}\big)\Big]
-\frac{\lambda_u}{2}\lVert\bm U\rVert_F^2-\frac{\lambda_v}{2}\lVert\bm V\rVert_F^2
\]
\[
\lambda_u=\lambda_v=\lambda:\qquad
\min_{\bm U\bm V^{\top}=\bm\Psi}\tfrac{\lambda}{2}\big(\lVert\bm U\rVert_F^2+\lVert\bm V\rVert_F^2\big)=\lambda\,\lVert\bm\Psi\rVert_{*}
\quad\text{(nuclear-norm logistic PCA, rank}\le k)
\]
\nt{Output: continuous latent probabilities $\hat P_{ij}=\sigma(\hat{\bm u}_i^{\top}\hat{\bm v}_j)\in[0,1]$ for a $\{0,1\}$ matrix.}
""",
    "eq_pg": r"""
\hd{P\'olya--Gamma identity (Polson, Scott \& Windle 2013)}
\[
\frac{\big(e^{\psi}\big)^{y}}{1+e^{\psi}}
=\tfrac12\,e^{\kappa\psi}\int_0^\infty e^{-\omega\psi^2/2}\,p_{\mathrm{PG}}(\omega\mid 1,0)\,d\omega,
\qquad \kappa=y-\tfrac12
\]
\hd{Conditional on $\omega$ the likelihood is Gaussian in $\psi$}
\[
\omega_{ij}\mid y_{ij},\psi_{ij}\sim\mathrm{PG}(1,\psi_{ij}),\qquad
\textcolor{orng}{\bar\omega_{ij}=\mathbb E[\omega_{ij}\mid\psi_{ij}]=\frac{\tanh(\psi_{ij}/2)}{2\psi_{ij}}}\quad(\tfrac14\text{ at }\psi=0)
\]
\hd{EM surrogate (expected complete-data log joint)}
\[
Q(\bm U,\bm V;\bar{\bm\Omega})=\sum_{(i,j)\in\mathcal O}\Big[\kappa_{ij}\,\bm u_i^{\top}\bm v_j-\frac{\bar\omega_{ij}}{2}\big(\bm u_i^{\top}\bm v_j\big)^2\Big]
-\frac{\lambda_u}{2}\lVert\bm U\rVert_F^2-\frac{\lambda_v}{2}\lVert\bm V\rVert_F^2
\]
\nt{$Q$ is a concave quadratic in each $\bm u_i$ (fixed $\bm V$) and in each $\bm v_j$ (fixed $\bm U$), but not jointly.}
""",
    "eq_mstep": r"""
\hd{One PG-EM iteration}
\[
\textbf{E-step:}\quad \bar\omega_{ij}\leftarrow\frac{\tanh(\psi_{ij}/2)}{2\psi_{ij}},\quad \psi_{ij}=\bm u_i^{\top}\bm v_j\ \text{(old factors; frozen for the M-step)}
\]
\[
\textbf{M-step A:}\quad
\Big[\lambda_u\bm I_k+\sum_{j\in\mathcal O_i}\bar\omega_{ij}\,\bm v_j\bm v_j^{\top}\Big]\bm u_i=\sum_{j\in\mathcal O_i}\kappa_{ij}\,\bm v_j
\qquad i=1,\dots,n
\]
\[
\textbf{M-step B:}\quad
\Big[\lambda_v\bm I_k+\sum_{i\in\mathcal O^j}\bar\omega_{ij}\,\bm u_i\bm u_i^{\top}\Big]\bm v_j=\sum_{i\in\mathcal O^j}\kappa_{ij}\,\bm u_i
\qquad j=1,\dots,m\ \ \text{(new }\bm U\text{, same }\bar{\bm\Omega})
\]
\hd{Guarantees and gauge}
\[
J\big(\bm U^{(t+1)},\bm V^{(t+1)}\big)\;\ge\;J\big(\bm U^{(t)},\bm V^{(t)}\big),
\qquad
\big(\bm U\bm R,\bm V\bm R\big),\ \bm R\in O(k)\ \text{leaves }J\text{ and }\hat P\text{ unchanged}
\]
\nt{Only $k\times k$ linear solves; no PG sampling. Nonconvex: local optima, multiple restarts; compare $\hat P$, never raw $\bm U$.}
""",
    "eq_eval": r"""
\hd{Held-out entry CV (C1): mask a random 10\% set $\mathcal M$, fit on the rest}
\[
\mathrm{NLL}_{\mathcal M}=-\frac{1}{|\mathcal M|}\sum_{(i,j)\in\mathcal M}\Big[y_{ij}\log\hat p_{ij}+(1-y_{ij})\log(1-\hat p_{ij})\Big]
\]
\hd{Oracle on synthetic data: held-out NLL with infinitely many replicates}
\[
\mathrm{KL}\big(P_{\mathrm{true}}\,\|\,\hat P\big)=\frac{1}{nm}\sum_{i,j}\Big[p_{ij}\log\frac{p_{ij}}{\hat p_{ij}}+(1-p_{ij})\log\frac{1-p_{ij}}{1-\hat p_{ij}}\Big],
\qquad \text{regret}=\mathrm{KL}(\hat\theta_{\mathrm{sel}})-\min_{\theta}\mathrm{KL}(\theta)
\]
\hd{Count splitting (C2; Neufeld, Gao \& Witten 2022)}
\[
X^{A}_{ij}\sim\mathrm{Binomial}\big(X_{ij},\tfrac12\big),\quad X^{B}_{ij}=X_{ij}-X^{A}_{ij}
\;\Rightarrow\; X^{A},X^{B}\overset{\text{ind}}{\sim}\mathrm{Poisson}(\mu_{ij}/2)\ \ \text{if }X\sim\mathrm{Poisson}(\mu)
\]
\[
p^{\mathrm{half}}_{ij}=1-e^{-\mu_{ij}/2}\qquad\Rightarrow\qquad p_{ij}=1-\big(1-p^{\mathrm{half}}_{ij}\big)^{2}
\]
\nt{Fit on $\mathbf 1[X^A\ge1]$, score on $\mathbf 1[X^B\ge1]$ to select $(\lambda,k)$; then refit on the full data.}
""",
    "eq_question": r"""
{\sffamily
\hd{Data}
A cell $\times$ gene detection matrix $Y\in\{0,1\}^{n\times m}$ from scRNA-seq: every entry is one Bernoulli trial.\par\medskip
\hd{Goal}
Infer the latent probability behind every entry,
\[
\hat p_{ij}=\Pr(y_{ij}=1\mid\bm u_i,\bm v_j)=\sigma\big(\bm u_i^{\top}\bm v_j\big)\in[0,1],
\]
turning a discrete $\{0,1\}$ matrix into a continuous one.\par\medskip
\hd{Questions}
\begin{enumerate}\itemsep4pt
\item Does $\hat P$ faithfully carry the information in the data (held-out likelihood, expression level)?
\item Does it keep biology (hemilineage) and drop batch (library, depth)?
\item Can $(\lambda,k)$ be chosen from the data alone so that $\hat P\approx P_{\mathrm{true}}$?
\end{enumerate}
\nt{Name: \textbf{BernFac} (Bernoulli factorization).}}
""",
    "eq_validation": r"""
{\sffamily
\hd{Solver validation: 11/11 tests pass}
\begin{itemize}\itemsep5pt
\item PG mean: $\bar\omega(0)=\tfrac14$, even, positive, $\bar\omega(\psi)\sim 1/(2|\psi|)$; continuous at the series switch $|\psi|=10^{-4}$.
\item Analytic gradients $\nabla_{\bm U}J=[M\odot(Y-P)]\bm V-\lambda_u\bm U$ match centred finite differences.
\item Block optimality: after each solve $\lVert\nabla_{\bm u_i}Q\rVert_\infty<10^{-10}$ (and likewise for $\bm v_j$).
\item Monotonicity: $J^{(t+1)}\ge J^{(t)}$ in every test, all 45 real fits and all 504 synthetic grid points.
\item Missing data: masked cells excluded, observed zeros counted; empty rows/columns $\to\bm 0$ under the prior.
\item Zero/zero initialisation is a fixed point: $P\equiv\tfrac12$, $\log p(Y)=-nm\log 2$; default init escapes.
\item Synthetic recovery ($200\times150$, $k=5$): $\operatorname{corr}(\hat P,P_{\mathrm{true}})>0.85$.
\end{itemize}}
""",
    "eq_offsets": r"""
\hd{Offset model: separate depth and abundance from biology}
\[
\psi_{ij}=a_i+b_j+\bm u_i^{\top}\bm v_j,\qquad
a_i,b_j\sim\mathcal N(0,\lambda_0^{-1})\ (\lambda_0=10^{-3},\ \text{near-flat}),\qquad
\bm u_i,\bm v_j\sim\mathcal N(\bm 0,\lambda^{-1}\bm I_k)
\]
\hd{Same PG-EM, augmented blocks (offset of the other side moved to the right-hand side)}
\[
\Big[\mathrm{diag}(\lambda\bm 1_k,\lambda_0)+\sum_j\bar\omega_{ij}\tilde{\bm v}_j\tilde{\bm v}_j^{\top}\Big]
\begin{pmatrix}\bm u_i\\ a_i\end{pmatrix}=\sum_j\big(\kappa_{ij}-\bar\omega_{ij}b_j\big)\tilde{\bm v}_j,
\qquad \tilde{\bm v}_j=\begin{pmatrix}\bm v_j\\ 1\end{pmatrix}
\]
\nt{and symmetrically for $(\bm v_j,b_j)$ with $\tilde{\bm u}_i=(\bm u_i;1)$ and offset $a_i$. Monotone in $J$; recovers synthetic offsets ($r=0.98$).}\par\medskip
\hd{Full data result ($k=40,\ \lambda=10$)}
\begin{itemize}\itemsep3pt
\item Held-out NLL $0.1777$ (vs $0.1778$ without offsets, $0.1854$ additive only); converges in 81 vs 188 iterations.
\item $\hat a_i$ absorbs depth: $\rho(\hat a,\log\text{UMI})=0.95$. The cell embedding of $\bm U\bm V^{\top}$ improves (hemilineage kNN $0.87\to0.92$, purity $0.78\to0.86$, ARI $0.49\to0.54$).
\item A depth axis remains ($|\rho|=0.80$). Under Poisson sampling $p=1-e^{-x}$, $x=s\mu_j$, so $\partial\,\mathrm{logit}\,p/\partial\log s = x e^{x}/(e^{x}-1)$: about 1 for rare genes and growing with $x$ for abundant ones. Depth moves the logits of abundant genes most, a depth\,$\times$\,gene interaction that an additive logit offset cannot absorb.
\end{itemize}
""",
    "eq_cloglog": r"""
\hd{Complementary log-log link: depth enters multiplicatively}
\[
\log\!\big(-\log(1-p_{ij})\big)=\eta_{ij}=a_i+b_j+\bm u_i^{\top}\bm v_j
\iff p_{ij}=1-e^{-\mu_{ij}},\ \ \mu_{ij}=e^{a_i}\,e^{b_j}\,e^{\bm u_i^{\top}\bm v_j},
\qquad y_{ij}=\mathbf 1[z_{ij}\ge1],\ z_{ij}\sim\mathrm{Poisson}(\mu_{ij})
\]
\hd{Inference: block Fisher scoring (no PG identity for this link), monotone by line search}
\[
g_{ij}=y_{ij}\frac{\mu_{ij}}{e^{\mu_{ij}}-1}-(1-y_{ij})\,\mu_{ij},\qquad
w_{ij}=\frac{\mu_{ij}^2}{e^{\mu_{ij}}-1},\qquad
\begin{pmatrix}\bm u_i\\ a_i\end{pmatrix}\leftarrow\begin{pmatrix}\bm u_i\\ a_i\end{pmatrix}
+t_i\Big[\mathrm{diag}(\lambda\bm 1_k,\lambda_0)+\sum_j w_{ij}\tilde{\bm v}_j\tilde{\bm v}_j^{\top}\Big]^{-1}\Big(\sum_j g_{ij}\tilde{\bm v}_j-\mathrm{diag}(\lambda\bm 1_k,\lambda_0)\begin{pmatrix}\bm u_i\\ a_i\end{pmatrix}\Big)
\]
\nt{The cloglog Bernoulli log-likelihood is concave in $\eta$, so each row (column) subproblem is concave; per-row backtracking ($t_i=1,\tfrac12,\dots$) makes $J$ nondecreasing.}\par\medskip
\hd{Full data result ($k=40,\ \lambda=30$)}
\begin{itemize}\itemsep3pt
\item Held-out NLL $0.17772$ (logit + offsets $0.17767$); additive-only cloglog $0.1845$ beats additive logistic $0.1854$.
\item $\rho(\hat a,\log\text{UMI})=0.96$; residual depth in the $\bm U\bm V^{\top}$ embedding halves ($|\rho|$ top-2: $0.80\to0.41$).
\item Best BernFac embedding: hemilineage kNN $0.93$, purity $0.89$, silhouette $0.12$, library mixing $0.90$ \textemdash\ still below processed counts ($0.98$, $0.97$, $0.30$, $0.94$).
\end{itemize}
""",
    "eq_counts": r"""
\hd{Count factorization: model the UMI counts, not just detection}
\[
x_{ij}\sim\mathrm{Poisson}(\mu_{ij})\ \ \text{or}\ \ \mathrm{NB}(\mu_{ij},\phi_j),\qquad
\log\mu_{ij}=\underbrace{\log s_i}_{\text{depth offset}}+a_i+b_j+\bm u_i^{\top}\bm v_j,\qquad
\mathrm{Var}_{\mathrm{NB}}=\mu+\mu^2/\phi_j
\]
\[
\bm u_i,\bm v_j\sim\mathcal N(\bm 0,\lambda^{-1}\bm I_k),\qquad a_i,b_j\sim\mathcal N(0,\lambda_0^{-1}),\qquad
s_i=\textstyle\sum_{j\in\text{train}}x_{ij}
\]
\hd{Fit: block Fisher scoring with per-row line search (MAP objective never decreases)}
\[
\text{Poisson: } g=x-\mu,\ w=\mu;\qquad
\text{NB: } g=\frac{x-\mu}{1+\mu/\phi_j},\ w=\frac{\mu}{1+\mu/\phi_j};\qquad \phi_j\ \text{by 1-D profile maximisation}
\]
\hd{Comparable prediction of detection on held-out entries}
\[
\Pr(y_{ij}=1)=\Pr(x_{ij}\ge1)=1-e^{-\mu_{ij}}\ \ (\text{Poisson}),\qquad 1-\big(1+\mu_{ij}/\phi_j\big)^{-\phi_j}\ \ (\text{NB})
\]
""",
    "eq_mfa": r"""
\hd{Bernoulli mixture of local Gaussian factor models}
\[
g_i\sim\mathrm{Cat}(\bm\pi),\quad \bm z_i\sim\mathcal N(\bm 0,\bm I_d),\quad
\bm\psi_i\mid\bm z_i,g_i{=}\ell\sim\mathcal N\big(\bm\mu_\ell+\bm B_\ell\bm z_i,\ t_\ell\bm I_m\big),\quad
y_{ij}\sim\mathrm{Bernoulli}\big(\sigma(\psi_{ij})\big)
\]
\nt{Logits live near $K$ local affine subspaces ($t_\ell=0$: exactly on them). Priors: $\bm\mu_\ell\sim\mathcal N(\bm 0,\lambda_\mu^{-1}\bm I)$, $\mathrm{vec}\,\bm B_\ell\sim\mathcal N(\bm 0,\lambda_B^{-1}\bm I)$, $\bm\pi\sim\mathrm{Dir}(\alpha)$.}\par\medskip
\hd{Variational E-step for every cell $i$ and component $\ell$ (PG quadratic bound, $w_j=f(\xi_{i\ell j})$, $a_j=1+t w_j$)}
\[
\bm A=\bm I_d+\bm B^{\top}\mathrm{diag}(w/a)\bm B,\quad
\bm h=\bm B^{\top}\big[(\kappa-w\odot\bm\mu)/a\big],\quad
q(\bm z)=\mathcal N(\bm A^{-1}\bm h,\bm A^{-1}),\quad
\xi_{i\ell j}^2=\mathbb E_q[\psi_{ij}^2]
\]
\[
r_{i\ell}=\mathrm{softmax}_\ell\big\{\log\pi_\ell+\ell^{\mathrm{bound}}_{i\ell}\big\},\qquad
\hat p_{ij}=\sum_\ell r_{i\ell}\,\mathbb E_{\psi\sim\mathcal N(\bar\psi_{i\ell j},\,t_\ell+\bm b_{\ell j}^{\top}\bm S_{i\ell}\bm b_{\ell j})}\big[\sigma(\psi)\big]
\]
\nt{M-step: $(\bm G_{\ell j}+t_\ell\bm\Lambda)\bm\theta_{\ell j}=\bm g_{\ell j}$ per component and gene; $\pi_\ell\propto N_\ell+\alpha-1$. The monitored bound never decreases. Predictions average probabilities (Gauss--Hermite), not logits.}
""",
    "eq_neural": r"""
\hd{Nonlinear latent manifold with a neural decoder (auto-decoder, joint MAP)}
\[
\bm z_i\sim\mathcal N(\bm 0,\lambda_z^{-1}\bm I_d),\qquad
\bm\psi_i=g_\theta(\bm z_i)=\bm b+\bm V\,h_\eta(\bm z_i)\ \ (\text{affine head})\quad\text{or}\quad \bm t+\rho\tanh(\bm t)\ \ (\text{nonlinear head})
\]
\[
F=\sum_{(i,j)\in\mathcal O}\big[\mathrm{softplus}(\psi_{ij})-y_{ij}\psi_{ij}\big]
+\tfrac{\lambda_z}{2}\lVert\bm Z\rVert_F^2+\tfrac{\lambda_\eta}{2}\lVert\eta\rVert^2+\tfrac{\lambda_V}{2}\lVert\bm V\rVert_F^2+\tfrac{\lambda_b}{2}\lVert\bm b\rVert^2
\]
\hd{PG-GEM: freeze $w_{ij}=f(\psi^{\mathrm{old}}_{ij})$, then decrease the quadratic surrogate}
\[
S(\theta;w)=\sum_{(i,j)\in\mathcal O}\big[\tfrac12 w_{ij}\psi_{ij}^2-\kappa_{ij}\psi_{ij}\big]+\text{penalties},\qquad
F(\theta)\le S(\theta;w)+C,\ \ \text{equality at }\theta^{\mathrm{old}}
\]
\[
\text{affine head, per gene: }\Big[\bm\Lambda_H+\sum_i w_{ij}\tilde{\bm h}_i\tilde{\bm h}_i^{\top}\Big]\bm\vartheta_j=\sum_i\kappa_{ij}\tilde{\bm h}_i;\qquad
\bm Z,\eta:\ \text{Armijo-checked gradient steps}
\]
\nt{Quadratic in the logits only, not in the network weights. Affine head: $\mathrm{rank}\,\bm\Psi\le H+1$. Predictions are MAP plug-in $\sigma(g_{\hat\theta}(\hat{\bm z}_i))$.}
""",
    "eq_lbm": r"""
\hd{Bernoulli latent block model (co-clustering)}
\[
g_i\sim\mathrm{Cat}(\bm\pi),\quad h_j\sim\mathrm{Cat}(\bm\rho),\quad
\psi_{ij}=B_{g_i h_j},\quad y_{ij}\sim\mathrm{Bernoulli}\big(\sigma(B_{g_i h_j})\big),\quad B_{ab}\sim\mathcal N(0,\lambda^{-1})
\]
\hd{Mean-field updates (rows, then columns with the new rows)}
\[
\log R_{ia}=\log\pi_a+\sum_{j}\sum_b C_{jb}\big[y_{ij}B_{ab}-\mathrm{softplus}(B_{ab})\big]+\mathrm{const},\qquad
\log C_{jb}=\log\rho_b+\sum_i\sum_a R_{ia}\big[\cdots\big]+\mathrm{const}
\]
\hd{Block log-odds: expected counts and a Pólya--Gamma step}
\[
\bm N=\bm R^{\top}\bm M\bm C,\quad \bm T=\bm R^{\top}(\bm M\odot\bm Y)\bm C,\qquad
B_{ab}\leftarrow\frac{T_{ab}-N_{ab}/2}{\lambda+N_{ab}\,f(B_{ab})}
\]
\[
\hat p_{ij}=\sum_{a,b}R_{ia}C_{jb}\,\sigma(B_{ab})=\big[\bm R\,\sigma(\bm B)\,\bm C^{\top}\big]_{ij}\ \neq\ \sigma\big(\bm R\bm B\bm C^{\top}\big)_{ij}
\]
""",
    "eq_conclusions": r"""
{\sffamily
\hd{Conclusions}
\begin{itemize}\itemsep4pt
\item Every model was fit on the same training entries, selected on the same validation half and scored once on the same 8.0 M test entries of the full fly dataset ($16{,}460\times19{,}543$).
\item For predicting detection, the negative-binomial count model is best (test NLL $0.1766$, AUC $0.9657$). The binary models tie within $0.0003$: mixture of local factors $0.1775$, logit + offsets $0.1777$, neural manifold $0.1777$, cloglog $0.1777$. The latent block model ($0.1830$) and Poisson counts ($0.1839$) are clearly worse.
\item For cell types, the NB count factors give the best clustering (hemilineage Leiden ARI $0.67$ vs $0.65$ for processed counts) and nearly match processed counts on kNN accuracy ($0.97$ vs $0.98$). The neural manifold mixes libraries almost as well as processed counts.
\item Binary-only models lose part of the cell-type signal: their leading axes track sequencing depth ($|\rho|$ up to $0.98$). Offsets, a cloglog link and modelling counts shrink this depth axis step by step.
\item Mixture components and co-clusters add little here: the mixture selects $K=1$, and block groups follow depth rather than hemilineage (ARI $0.06$).
\end{itemize}
\hd{Next steps}
\nt{Batch covariates $\bm X_{\mathrm{batch}}\bm\Gamma$ in the NB model; posterior uncertainty (PG Gibbs, Laplace); feed NB-derived detection probabilities into ConnectionMiner.\\
Caveats: one dataset; time-limited fits for the neural and mixture models; NB best rank $k=120$ (interior optimum; $k=160$ worse); latent block model best at the $200\times200$ grid corner.}}
""",

}


def render(name, body):
    with tempfile.TemporaryDirectory() as d:
        tex = Path(d) / f"{name}.tex"
        tex.write_text(PREAMBLE + body + "\n\\end{document}\n")
        subprocess.run(["tectonic", "-X", "compile", "--outdir", d, str(tex)], check=True,
                       capture_output=True)
        subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                        str(Path(d) / f"{name}.pdf"), str(OUT / name)], check=True)
        (OUT / f"{name}.pdf").write_bytes((Path(d) / f"{name}.pdf").read_bytes())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, body in PANELS.items():
        render(name, body)
        print("rendered", name)
