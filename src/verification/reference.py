"""Независимый математический эталон для golden-тестов (REBUILD_SPEC §6).

Эти функции НЕ используют продакшн-код из ``src/`` (кроме numpy) и реализуют те
же величины другим путём — чтобы сверка ловила регрессии в продакшн-модулях:

  * OLS через QR-разложение            (эквив. R ``lm`` / ``mixexp::MixModel``);
  * D-критерий/эффективность через eig (эквив. ``AlgDesign`` det-критерий);
  * I-критерий через trace((XᵀX)⁻¹·M);
  * desirability Деррингера–Сюича      (эквив. R-пакет ``desirability``);
  * GP-постериор Matérn5/2 ARD / RBF   (эквив. ``DiceKriging::km`` при фикс. θ).
"""
from __future__ import annotations

from itertools import combinations
from math import gamma
from typing import List, Sequence, Tuple

import numpy as np


# ======================================================================
# Модельная матрица Шеффе (независимая реализация, без импорта linalg)
# ======================================================================
def scheffe_terms(q: int, order: int) -> List[Tuple[int, ...]]:
    terms: List[Tuple[int, ...]] = []
    for k in range(1, order + 1):
        terms.extend(combinations(range(q), k))
    return terms


def scheffe_design_matrix(X: np.ndarray, order: int) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    n, q = X.shape
    cols = []
    for idx in scheffe_terms(q, order):
        col = np.ones(n)
        for i in idx:
            col = col * X[:, i]
        cols.append(col)
    return np.column_stack(cols)


# ======================================================================
# OLS через QR (эталон для ScheffeModel)
# ======================================================================
def ref_ols(M: np.ndarray, y: np.ndarray) -> dict:
    """Решение МНК β=(MᵀM)⁻¹Mᵀy через QR (численно стабильно, как R lm)."""
    M = np.asarray(M, float)
    y = np.asarray(y, float).ravel()
    n, p = M.shape
    Q, R = np.linalg.qr(M)
    beta = np.linalg.solve(R, Q.T @ y)
    fitted = M @ beta
    resid = y - fitted
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    df_resid = n - p
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    rmse = float(np.sqrt(sse / df_resid)) if df_resid > 0 else 0.0
    adj_r2 = (1.0 - (sse / df_resid) / (sst / (n - 1))
              if df_resid > 0 and sst > 0 else float("nan"))
    return {"coefficients": beta, "fitted": fitted, "residuals": resid,
            "sse": sse, "sst": sst, "r2": r2, "adj_r2": adj_r2, "rmse": rmse}


# ======================================================================
# D-/I-критерии (эталон для core.linalg)
# ======================================================================
def ref_d_criterion(M: np.ndarray) -> float:
    """det(MᵀX) через произведение собственных значений (Gram = symmetric)."""
    M = np.asarray(M, float)
    w = np.linalg.eigvalsh(M.T @ M)
    return float(np.prod(w))


def ref_d_efficiency(M: np.ndarray) -> float:
    """(det(MᵀM))^(1/p) / n через собственные значения."""
    M = np.asarray(M, float)
    n, p = M.shape
    w = np.linalg.eigvalsh(M.T @ M)
    if np.any(w <= 0):
        return 0.0
    log_det = float(np.sum(np.log(w)))
    return float(np.exp(log_det / p) / n)


def ref_i_criterion(M: np.ndarray, moments: np.ndarray) -> float:
    """trace((MᵀM)⁻¹ · moments) через решение линейной системы."""
    M = np.asarray(M, float)
    moments = np.asarray(moments, float)
    XtX = M.T @ M
    inv_mom = np.linalg.solve(XtX, moments)
    return float(np.trace(inv_mom))


# ======================================================================
# Desirability Деррингера–Сюича (эталон для optimize.desirability)
# ======================================================================
def ref_desirability(y: np.ndarray, kind: str, low: float, high: float,
                     target: float = None, s: float = 1.0,
                     s2: float = None) -> np.ndarray:
    y = np.asarray(y, float)
    s2 = s if s2 is None else s2
    if kind == "max":
        d = np.where(y <= low, 0.0,
                     np.where(y >= high, 1.0, ((y - low) / (high - low)) ** s))
    elif kind == "min":
        d = np.where(y <= low, 1.0,
                     np.where(y >= high, 0.0, ((high - y) / (high - low)) ** s))
    elif kind == "target":
        lower = (y - low) / (target - low)
        upper = (high - y) / (high - target)
        d = np.where((y < low) | (y > high), 0.0,
                     np.where(y <= target,
                              np.clip(lower, 0.0, 1.0) ** s,
                              np.clip(upper, 0.0, 1.0) ** s2))
    else:
        raise ValueError(kind)
    return np.clip(d, 0.0, 1.0)


def ref_overall(d_list: Sequence[np.ndarray],
                weights: Sequence[float]) -> np.ndarray:
    """Взвешенное геометрическое среднее; любое d_i==0 => 0 (veto)."""
    D = np.vstack([np.atleast_1d(np.asarray(d, float)) for d in d_list])
    w = np.asarray(weights, float)
    w = w / w.sum()
    out = np.zeros(D.shape[1])
    veto = np.any(D <= 0.0, axis=0)
    safe = ~veto
    if np.any(safe):
        log_d = (w[:, None] * np.log(np.clip(D[:, safe], 1e-300, 1.0))).sum(axis=0)
        out[safe] = np.exp(log_d)
    return out


# ======================================================================
# GP-постериор при ФИКСИРОВАННЫХ гиперпараметрах (эталон для GPExpert)
# Ядро = const * base(ARD) + white,  как в продакшн-конфигурации.
# ======================================================================
def _ard_r(Xa: np.ndarray, Xb: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """Матрица масштабированных расстояний r_ij = ||(xa-xb)/ls||."""
    Xa = np.atleast_2d(Xa) / ls
    Xb = np.atleast_2d(Xb) / ls
    a2 = (Xa ** 2).sum(1)[:, None]
    b2 = (Xb ** 2).sum(1)[None, :]
    d2 = np.clip(a2 + b2 - 2 * Xa @ Xb.T, 0.0, None)
    return np.sqrt(d2)


def matern52(Xa, Xb, ls) -> np.ndarray:
    r = _ard_r(Xa, Xb, np.asarray(ls, float))
    s5 = np.sqrt(5.0)
    return (1.0 + s5 * r + 5.0 / 3.0 * r ** 2) * np.exp(-s5 * r)


def rbf(Xa, Xb, ls) -> np.ndarray:
    r = _ard_r(Xa, Xb, np.asarray(ls, float))
    return np.exp(-0.5 * r ** 2)


def gp_posterior(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                 const: float, length_scale, noise: float,
                 kernel: str = "matern52") -> dict:
    """Постериор GP: μ, σ при фиксированных θ (sklearn-совместимая семантика).

    K   = const·base(Xtr,Xtr) + noise·I
    k*  = const·base(Xtr,Xte)
    μ   = k*ᵀ K⁻¹ y
    σ²  = (const + noise) − diag(k*ᵀ K⁻¹ k*)   [base(x,x)=1, white на диагонали]
    """
    kfun = matern52 if kernel == "matern52" else rbf
    X_train = np.atleast_2d(np.asarray(X_train, float))
    X_test = np.atleast_2d(np.asarray(X_test, float))
    y_train = np.asarray(y_train, float).ravel()
    ls = np.asarray(length_scale, float)

    K = const * kfun(X_train, X_train, ls) + noise * np.eye(len(X_train))
    Ks = const * kfun(X_train, X_test, ls)            # (n_tr, n_te)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mean = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)                         # (n_tr, n_te)
    var = (const + noise) - np.sum(v ** 2, axis=0)
    var = np.clip(var, 0.0, None)
    return {"mean": mean, "std": np.sqrt(var)}


# ======================================================================
# Моменты на произведении области (эталон для design.block_moments, §13.5)
# Независимый путь: через Γ-функцию (vs factorial в продакшене).
#   симплекс S^{q-1}, Дирихле(1):  E[∏x_i^{a_i}] = Γ(q)∏Γ(a_i+1)/Γ(q+Σa)
#   куб [0,1]^d, равномерно:        E[∏z_k^{b_k}] = ∏ 1/(b_k+1)
#   произведение области ⇒ множители независимы.
# ======================================================================
def ref_simplex_moment(a: Sequence[int], q: int) -> float:
    """E[∏ x_i^{a_i}] на стандартном симплексе S^{q-1} (Дирихле(1,…,1))."""
    if q == 0:
        return 1.0
    num = gamma(q)
    for ai in a:
        num *= gamma(ai + 1)
    return num / gamma(q + sum(a))


def ref_cube_moment(b: Sequence[int]) -> float:
    """E[∏ z_k^{b_k}] на [0,1]^d (равномерно). d==0 → 1."""
    v = 1.0
    for bk in b:
        v *= 1.0 / (bk + 1.0)
    return v


def ref_product_moment_matrix(term_exponents: Sequence[Tuple[Sequence[int],
                                                              Sequence[int]]],
                              q: int, d: int) -> np.ndarray:
    """Матрица моментов M[r,s]=E[f_r f_s] по списку (a,b)-степеней термов.

    ``term_exponents[r] = (a_r, b_r)`` — степени r-го терма по x (len q) и z (len d).
    """
    p = len(term_exponents)
    M = np.zeros((p, p))
    for r, (ar, br) in enumerate(term_exponents):
        for s, (as_, bs) in enumerate(term_exponents):
            a = [ar[i] + as_[i] for i in range(q)]
            b = [br[k] + bs[k] for k in range(d)]
            M[r, s] = ref_simplex_moment(a, q) * ref_cube_moment(b)
    return M
