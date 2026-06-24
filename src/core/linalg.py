"""
core/linalg.py — Linear algebra & model-matrix utilities (numpy/scipy).

Replaces the legacy hand-rolled pure-Python matrix routines in
``src/utils/math_utils.py`` with vectorised numpy/scipy implementations.

Provides:
  * Numerically stable matrix metrics (logdet, determinant, inverse, solve).
  * Scheffe mixture model-matrix builder (explicit model = input).
  * Standard (factorial) polynomial model-matrix builder.
  * D- and I-efficiency / optimality criteria.

All public functions accept ``np.ndarray`` (or array-likes) and return numpy
arrays / floats.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence]

# ---------------------------------------------------------------------------
# Matrix metrics
# ---------------------------------------------------------------------------

def gram(X: ArrayLike) -> np.ndarray:
    """Return the information matrix XᵀX."""
    X = np.asarray(X, dtype=float)
    return X.T @ X


def slogdet(M: ArrayLike) -> float:
    """Signed log-determinant; returns -inf for singular matrices."""
    M = np.asarray(M, dtype=float)
    sign, logabs = np.linalg.slogdet(M)
    if sign <= 0:
        return float("-inf")
    return float(logabs)


def determinant(M: ArrayLike) -> float:
    """Determinant via numpy (LU). Returns 0.0 for singular matrices."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 1.0
    try:
        return float(np.linalg.det(M))
    except np.linalg.LinAlgError:
        return 0.0


def inverse(M: ArrayLike, ridge: float = 0.0) -> np.ndarray:
    """Matrix inverse with optional ridge regularisation (M + ridge·I)⁻¹."""
    M = np.asarray(M, dtype=float)
    if ridge:
        M = M + ridge * np.eye(M.shape[0])
    return np.linalg.inv(M)


def solve(A: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Solve A x = b, falling back to least squares if A is singular."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def condition_number(M: ArrayLike) -> float:
    """2-norm condition number (large => near-singular / collinear)."""
    M = np.asarray(M, dtype=float)
    try:
        return float(np.linalg.cond(M))
    except np.linalg.LinAlgError:
        return float("inf")


# ---------------------------------------------------------------------------
# Scheffe mixture model matrix  (NO intercept, NO pure-quadratic xi^2)
# ---------------------------------------------------------------------------

def scheffe_term_indices(q: int, model: Union[str, int]) -> List[Tuple[int, ...]]:
    """Return the list of component-index tuples defining Scheffe terms.

    ``model`` is the EXPLICIT model specification (per REBUILD_SPEC rule #1):
      * 'linear'    / 1 -> first-order terms x_i
      * 'quadratic' / 2 -> + x_i x_j
      * 'cubic'     / 3 -> + x_i x_j x_k          (special cubic / full = same set here)
      * 'quartic'   / 4 -> + 4-way
      * 'quintic'   / 5 -> + 5-way
    """
    order_map = {
        "linear": 1, "quadratic": 2, "cubic": 3, "quartic": 4, "quintic": 5,
    }
    if isinstance(model, str):
        if model not in order_map:
            raise ValueError(f"Unknown model '{model}'. Use one of {list(order_map)} or int.")
        order = order_map[model]
    else:
        order = int(model)
    if not 1 <= order <= q:
        raise ValueError(f"Model order {order} out of range for q={q}.")

    terms: List[Tuple[int, ...]] = []
    for k in range(1, order + 1):
        terms.extend(combinations(range(q), k))
    return terms


def scheffe_term_names(q: int, model: Union[str, int],
                       names: Sequence[str] | None = None) -> List[str]:
    """Human-readable Scheffe term names, e.g. ['A','B','A*B', ...]."""
    if names is None:
        names = [chr(ord("A") + i) for i in range(q)]
    out = []
    for idx in scheffe_term_indices(q, model):
        out.append("*".join(names[i] for i in idx))
    return out


def scheffe_matrix(X: ArrayLike, model: Union[str, int]) -> np.ndarray:
    """Build the Scheffe mixture model matrix for design ``X`` (n x q).

    Columns are products of the component values for each term tuple.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, q = X.shape
    cols = []
    for idx in scheffe_term_indices(q, model):
        col = np.ones(n)
        for i in idx:
            col = col * X[:, i]
        cols.append(col)
    return np.column_stack(cols) if cols else np.empty((n, 0))


def n_scheffe_params(q: int, model: Union[str, int]) -> int:
    """Number of parameters in the given Scheffe model."""
    return len(scheffe_term_indices(q, model))


def scheffe_active_terms(q: int, active: Sequence[int],
                         model: Union[str, int]) -> List[Tuple[int, ...]]:
    """Scheffe terms restricted to the ACTIVE components (q_eff reduction).

    Heredity-consistent: keep only terms whose component indices are ALL in
    ``active`` (linear ``(i,)`` if ``i`` active; interaction ``(i,j,...)`` only
    if every index is active). Inactive components drop out of the model — this
    is how M5 works on ``q_eff`` instead of full ``q`` (FinalCheckList §5.5.1),
    while the design points themselves stay full-``q`` mixtures.
    """
    aset = {int(a) for a in active}
    return [idx for idx in scheffe_term_indices(q, model)
            if all(i in aset for i in idx)]


def scheffe_matrix_terms(X: ArrayLike,
                         terms: Sequence[Tuple[int, ...]]) -> np.ndarray:
    """Scheffe model matrix from an EXPLICIT term list (e.g. q_eff-reduced).

    Each column is the product of component values for the term's indices.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    cols = []
    for idx in terms:
        col = np.ones(n)
        for i in idx:
            col = col * X[:, i]
        cols.append(col)
    return np.column_stack(cols) if cols else np.empty((n, 0))



# ---------------------------------------------------------------------------
# Standard (factorial) polynomial model matrix  (WITH intercept)
# ---------------------------------------------------------------------------

def standard_matrix(X: ArrayLike, model: Union[str, int]) -> np.ndarray:
    """Build a standard polynomial model matrix with intercept + main + 2fi (+ squares)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, k = X.shape
    order_map = {"linear": 1, "quadratic": 2, "cubic": 3}
    order = order_map.get(model, model) if isinstance(model, str) else int(model)

    cols = [np.ones(n)]                       # intercept
    for i in range(k):                        # main effects
        cols.append(X[:, i])
    if order >= 2:
        for i, j in combinations(range(k), 2):  # two-factor interactions
            cols.append(X[:, i] * X[:, j])
        for i in range(k):                      # pure quadratics
            cols.append(X[:, i] ** 2)
    if order >= 3:
        for i, j, l in combinations(range(k), 3):
            cols.append(X[:, i] * X[:, j] * X[:, l])
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Optimality criteria
# ---------------------------------------------------------------------------

def d_criterion(M: ArrayLike) -> float:
    """D-criterion = det(MᵀM) for a model matrix M (larger is better)."""
    return determinant(gram(M))


def d_efficiency(M: ArrayLike) -> float:
    """Normalised D-efficiency = (det(MᵀM)^(1/p)) / n  in [0, 1]-ish scale."""
    M = np.asarray(M, dtype=float)
    n, p = M.shape
    if n < p:
        return 0.0
    ld = slogdet(gram(M))
    if not np.isfinite(ld):
        return 0.0
    return float(np.exp(ld / p) / n)


def i_criterion(M: ArrayLike, moments: ArrayLike) -> float:
    """I-criterion = trace((MᵀM)⁻¹ · moments).  Smaller is better.

    ``moments`` is the region moment matrix E[f(x) f(x)ᵀ] over the design space
    (same column ordering as M). Returns +inf if MᵀM is singular.
    """
    M = np.asarray(M, dtype=float)
    moments = np.asarray(moments, dtype=float)
    XtX = gram(M)
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.trace(inv @ moments))


def prediction_variance(M: ArrayLike, x_row: ArrayLike) -> float:
    """Scaled prediction variance ν(x) = xᵀ(MᵀM)⁻¹x for a single model-row x."""
    M = np.asarray(M, dtype=float)
    x_row = np.asarray(x_row, dtype=float).ravel()
    XtX = gram(M)
    try:
        inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(x_row @ inv @ x_row)
