from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .models import CmResult, PrepData


def cm_solve(prep: PrepData, cfg: dict[str, Any]) -> CmResult:
    np.random.seed(cfg["seed"])
    t0 = time.time()

    G_full = prep.G_metacell_p
    if prep.G_metacell_p_solve is not None and prep.G_metacell_p_solve.size > 0:
        G = prep.G_metacell_p_solve.astype(float)
        ng_solver = G.shape[1]
        print(f"  Solver base gene count (Ng_solver) = {ng_solver} (of {G_full.shape[1]} total)")
    else:
        G = G_full.astype(float)
        ng_solver = G.shape[1]
        print(f"  Solver using all {ng_solver} genes as base solver genes")

    use_complement = bool(cfg["solver"].get("use_complement", False))
    if use_complement:
        G_base = G
        G_comp = np.clip(1.0 - G_base, 0.0, 1.0)
        G = np.hstack([G_base, G_comp])
        ng_solve = G.shape[1]
        print(f"  Using complement features: Ng_solver={ng_solver}, Ng_eff_raw={ng_solve} (=2*Ng_solver)")
    else:
        ng_solve = G.shape[1]
        print(f"  Complements disabled: Ng_solver={ng_solver}")

    beta_rank = int(cfg["solver"].get("beta_rank", 0) or 0)
    is_low_rank = False
    U_r = None
    S_r = None

    if 0 < beta_rank < ng_solve:
        print(f"  Low-rank beta: projecting G to rank {beta_rank}")
        G_centered = G - np.mean(G, axis=0, keepdims=True)
        U_r, s_vals, _ = svds(G_centered, k=beta_rank)
        ord_idx = np.argsort(-s_vals)
        s_vals = s_vals[ord_idx]
        U_r = U_r[:, ord_idx]
        S_r = np.diag(s_vals)
        G_proj = U_r @ S_r
        ng_eff = beta_rank
        is_low_rank = True
        print(f"  Effective gene dimension: {ng_eff} (was {ng_solve})")
    else:
        G_proj = G
        ng_eff = ng_solve

    C = prep.C_counts.astype(float).copy()
    C_mask = prep.C_mask.astype(float)
    if cfg["solver"].get("use_binary_connectome", True):
        C = (C > 0).astype(float)
    W = ((C_mask > 0) & (~np.isnan(C))).astype(float)
    C[W == 0] = 0.0

    P_constraints = prep.P_constraints_metacell.astype(float)
    D = P_constraints

    P = cm_init_P(P_constraints, cfg["solver"].get("P_init", "blend"))
    print(f"  P init: {cfg['solver'].get('P_init', 'blend')}")

    beta_init = str(cfg["solver"].get("beta_init", "random")).lower()
    if beta_init == "identity":
        beta = np.eye(ng_eff)
    elif beta_init == "ones":
        beta = np.ones((ng_eff, ng_eff), dtype=float)
    else:
        beta = np.random.rand(ng_eff, ng_eff)
    beta_max = np.full((ng_eff, ng_eff), np.inf)
    print(f"  Beta init: {beta_init}")

    beta_mask = None
    interactome_mode = str(cfg["solver"].get("interactome_constraint", "none"))
    if interactome_mode == "hard" and hasattr(prep, "beta_mask"):
        if is_low_rank:
            print("  Warning: interactome hard mask not applied in low-rank mode")
        else:
            beta_mask = getattr(prep, "beta_mask")
            beta[~beta_mask] = 0.0
            beta_max[~beta_mask] = 0.0

    num_iter = int(cfg["solver"]["num_iter"])
    lamb = float(cfg["solver"]["lambda_sparsity"])
    epsilon = float(cfg["solver"]["optimal_transport_epsilon"])
    step_size = float(cfg["solver"]["optimal_transport_step"])
    ot_max_iter = int(cfg["solver"]["optimal_transport_iterations"])
    reg_max_iter = int(cfg["solver"]["regression_iterations"])
    time_limit = float(cfg["solver"].get("time_limit_per_step", 30.0))

    loss = np.zeros(num_iter, dtype=float)
    obj_beta = np.zeros(num_iter, dtype=float)
    obj_P_fit = np.zeros(num_iter, dtype=float)
    obj_P_ent = np.zeros(num_iter, dtype=float)

    print(
        f"  Solver: {num_iter} outer iterations, Ng_eff={ng_eff}, beta size={ng_eff}x{ng_eff}"
    )

    for it in range(num_iter):
        t_iter = time.time()

        PG = P @ G_proj
        beta, train_loss, _ = cm_beta_update(
            A=PG,
            B=PG.T,
            C=C,
            W=W,
            beta=beta,
            beta_max=beta_max,
            lamb=lamb,
            max_iter=reg_max_iter,
            time_limit=time_limit,
            beta_mask=beta_mask,
        )
        obj_beta[it] = train_loss

        Z = G_proj @ beta @ G_proj.T
        P = cm_P_update(
            P=P,
            Z=Z,
            C=C,
            W=W,
            D=D,
            epsilon=epsilon,
            step_size=step_size,
            max_iter=ot_max_iter,
            time_limit=time_limit,
        )

        recon = P @ G_proj @ beta @ G_proj.T @ P.T
        obj_P_fit[it] = np.linalg.norm(W * (recon - C), ord="fro") ** 2
        mask_P = P > 0
        obj_P_ent[it] = epsilon * np.sum(P[mask_P] * np.log(P[mask_P]))
        loss[it] = obj_P_fit[it] + obj_P_ent[it]

        print(
            "  Solver iter {}/{}: obj_beta={:.6e}  obj_P_fit={:.6e}  obj_P_ent={:.6e}  total={:.6e} ({:.1f}s)".format(
                it + 1,
                num_iter,
                obj_beta[it],
                obj_P_fit[it],
                obj_P_ent[it],
                loss[it],
                time.time() - t_iter,
            )
        )

    final_recon = P @ G_proj @ beta @ G_proj.T @ P.T

    meta: dict[str, Any] = {}
    if is_low_rank:
        meta["U_r"] = U_r
        meta["S_r"] = S_r

    cm = CmResult(
        P=P,
        beta=beta,
        G_proj=G_proj,
        loss=loss,
        obj_beta=obj_beta,
        obj_P_fit=obj_P_fit,
        obj_P_ent=obj_P_ent,
        P_constraints=P_constraints,
        C=C,
        C_mask=W,
        C_recon=final_recon,
        elapsed_sec=time.time() - t0,
        Ng_solve=ng_solver,
        Ng_eff=ng_eff,
        is_low_rank=is_low_rank,
        meta=meta,
    )

    if cfg.get("run_dir"):
        obj_path = f"{cfg['run_dir']}/solver_objectives.txt"
        with open(obj_path, "w", encoding="utf-8") as f:
            f.write("iter\tobj_beta\tobj_P_fit\tobj_P_ent\ttotal_loss\n")
            for i in range(num_iter):
                f.write(
                    f"{i + 1}\t{obj_beta[i]:.6e}\t{obj_P_fit[i]:.6e}\t{obj_P_ent[i]:.6e}\t{loss[i]:.6e}\n"
                )
        print(f"  Wrote {obj_path}")

    print(f"  Solver done in {cm.elapsed_sec:.1f} s. Final loss: {loss[-1]:.6e}")
    return cm


def cm_init_P(D: np.ndarray, init_type: str = "blend") -> np.ndarray:
    init_type = (init_type or "blend").lower()

    if init_type == "uniform":
        P = D / np.maximum(D.sum(axis=1, keepdims=True), 1e-16)
        P[np.isnan(P)] = 0.0
        return P

    if init_type == "binary":
        P = _random_binary_init(D)
        return _normalize_rows(P)

    if init_type == "random_proportional":
        P = np.zeros_like(D, dtype=float)
        idx = D > 0
        P[idx] = np.random.rand(np.sum(idx))
        P = _normalize_rows(P)
        P[~idx] = 0.0
        return P

    P1 = _normalize_rows(_random_binary_init(D))
    P2 = D / np.maximum(D.sum(axis=1, keepdims=True), 1e-16)
    P2[np.isnan(P2)] = 0.0
    return 0.5 * P1 + 0.5 * P2


def _random_binary_init(D: np.ndarray) -> np.ndarray:
    N, M = D.shape
    P = np.zeros_like(D, dtype=float)

    groups: dict[tuple[int, ...], list[int]] = {}
    for c in range(M):
        rows = tuple(np.where(D[:, c] > 0)[0].tolist())
        groups.setdefault(rows, []).append(c)

    for rows, cols in groups.items():
        if not rows or not cols:
            continue
        rows_arr = np.array(rows, dtype=int)
        cols_arr = np.array(cols, dtype=int)
        np.random.shuffle(rows_arr)
        np.random.shuffle(cols_arr)
        for j, col in enumerate(cols_arr):
            P[rows_arr[j % rows_arr.size], col] = 1.0

    return P


def _normalize_rows(P: np.ndarray) -> np.ndarray:
    row_sums = np.sum(P, axis=1, keepdims=True)
    inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    return P * inv


def cm_beta_update(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    beta: np.ndarray,
    beta_max: np.ndarray,
    lamb: float,
    max_iter: int,
    time_limit: float,
    beta_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    tol = 1e-6
    t0 = time.time()

    Wsq = W ** 2
    Numer = A.T @ (Wsq * C) @ B.T

    beta = np.clip(beta, 0.0, beta_max)
    if beta_mask is not None:
        beta = np.where(beta_mask, beta, 0.0)

    prev_obj = _beta_obj(A, B, C, Wsq, beta)

    for _ in range(max_iter):
        M_recon = A @ (beta @ B)
        Denom = A.T @ (Wsq * M_recon) @ B.T
        beta_raw = beta * (Numer / (Denom + lamb))

        beta = np.clip(beta_raw, 0.0, beta_max)
        if beta_mask is not None:
            beta = np.where(beta_mask, beta, 0.0)

        curr_obj = _beta_obj(A, B, C, Wsq, beta)
        rel_change = abs(curr_obj - prev_obj) / (prev_obj + np.finfo(float).eps)

        if rel_change < tol or (time.time() - t0) > time_limit:
            break
        prev_obj = curr_obj

    beta = beta + 100 * np.finfo(float).eps * np.random.rand(*beta.shape)
    if beta_mask is not None:
        beta = np.where(beta_mask, beta, 0.0)

    train_loss = _beta_obj(A, B, C, Wsq, beta)
    val_loss = _beta_obj(A, B, C, 1.0 - Wsq, beta)
    return beta, train_loss, val_loss


def _beta_obj(A: np.ndarray, B: np.ndarray, C: np.ndarray, Wsq: np.ndarray, X: np.ndarray) -> float:
    R = A @ (X @ B) - C
    return float(np.sum(Wsq * (R ** 2)))


def cm_P_update(
    P: np.ndarray,
    Z: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    D: np.ndarray,
    epsilon: float,
    step_size: float,
    max_iter: int,
    time_limit: float,
) -> np.ndarray:
    D_norm = D / np.maximum(np.sum(D, axis=1, keepdims=True), 1e-16)
    D_norm[np.isnan(D_norm)] = 0.0
    row_c = np.sum(D_norm, axis=1)
    col_c = np.sum(D_norm, axis=0)

    A = Z @ P.T
    B = C
    unfixed = ~np.all(B == 0, axis=1)
    P_tmp = _entropic_sinkhorn(A, B, W, row_c, col_c, D, epsilon, step_size, max_iter, time_limit, P)
    P[unfixed, :] = P_tmp[unfixed, :]

    A = Z.T @ P.T
    B = C.T
    unfixed = ~np.all(B == 0, axis=1)
    P_tmp = _entropic_sinkhorn(A, B, W.T, row_c, col_c, D, epsilon, step_size, max_iter, time_limit, P)
    P[unfixed, :] = P_tmp[unfixed, :]

    return P


def _entropic_sinkhorn(
    A: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    D: np.ndarray,
    epsilon: float,
    step_size: float,
    max_iter: int,
    time_limit: float,
    P0: np.ndarray | None,
) -> np.ndarray:
    row_col_iters = 10
    max_backtrack = 20
    tol = 1e-6

    if P0 is not None:
        P = P0.copy()
    else:
        P = 0.5 * D
    P = _normalize_clip(P, a, b, D, row_col_iters)

    W2 = W ** 2
    prev_obj = _sinkhorn_obj(P, A, B, W2, epsilon)

    t0 = time.time()
    for _ in range(max_iter):
        PA = P @ A
        residual = PA - B
        WR = W2 * residual
        grad_ls = 2.0 * (WR @ A.T)
        grad_ent = epsilon * (1.0 + _safe_log(P))
        G = grad_ls + grad_ent

        trial_step = step_size
        new_obj = prev_obj
        P_trial = P
        for bt in range(max_backtrack + 1):
            P_trial = P * np.exp(-trial_step * G)
            P_trial = np.clip(P_trial, 0.0, D)
            P_trial = _normalize_clip(P_trial, a, b, D, row_col_iters)
            new_obj = _sinkhorn_obj(P_trial, A, B, W2, epsilon)
            if new_obj <= prev_obj or bt >= max_backtrack:
                break
            trial_step /= 2.0

        P = P_trial
        rel_change = abs(new_obj - prev_obj) / max(1.0, abs(prev_obj))

        if rel_change < tol:
            P = _normalize_clip(P, a, b, D, 1000)
            break
        if (time.time() - t0) > time_limit:
            break

        prev_obj = new_obj

    return P


def _sinkhorn_obj(P: np.ndarray, A: np.ndarray, B: np.ndarray, W2: np.ndarray, epsilon: float) -> float:
    R = P @ A - B
    val_ls = np.sum(W2 * (R ** 2))
    mask = P > 0
    val_ent = np.sum(P[mask] * np.log(P[mask])) if np.any(mask) else 0.0
    return float(val_ls + epsilon * val_ent)


def _safe_log(X: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(X, 1e-30))


def _normalize_clip(P: np.ndarray, a: np.ndarray, b: np.ndarray, D: np.ndarray, passes: int) -> np.ndarray:
    for _ in range(passes):
        rs = np.sum(P, axis=1, keepdims=True)
        scale_r = a[:, None] / np.maximum(rs, 1e-16)
        P = np.clip(P * scale_r, 0.0, D)

        cs = np.sum(P, axis=0, keepdims=True)
        scale_c = b[None, :] / np.maximum(cs, 1e-16)
        P = np.clip(P * scale_c, 0.0, D)
    return P
