"""
DOE Sequential Workflow: From Screening to Competitive Model
=============================================================

A comprehensive step-by-step interface that guides the user through:

  Stage 1 – Problem Setup          : factors/components, bounds, model type
  Stage 2 – Initial Design         : generate D-optimal / simplex design, download
  Stage 3 – Data Entry             : enter or upload experimental responses
  Stage 4 – Screening Analysis     : p-values, ANOVA, half-normal plot,
                                     INTERACTIVE factor selection
  Stage 5 – Fold-Over / Augment    : generate de-aliasing runs, enter responses
  Stage 6 – Final Model            : competitive model with full diagnostics
  Stage 7 – Report                 : export results

Key improvements over the previous sequential interface:
  * Real user data (not synthetic)
  * Statistically-driven factor selection UI (user picks significant terms)
  * Fractional fold-over for de-aliasing confounded factorial effects
  * D-optimal augmentation for mixture designs
  * Full diagnostic suite matching JMP/Design-Expert output
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from itertools import combinations
import sys, os, io, json

# ── path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from algorithms.candidate_generation import create_candidate_generator, MixtureCandidateGenerator
    from algorithms.d_optimal_algorithm import MixtureDOptimalAlgorithm, create_d_optimal_algorithm
    from utils.math_utils import normalize_to_simplex, latin_hypercube_sampling
    _USE_MODULAR = True
except Exception:
    _USE_MODULAR = False

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DOE Sequential Workflow",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Statistical model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_matrix(design: np.ndarray, model_type: str, design_kind: str) -> tuple:
    """
    Build the (expanded) model matrix and return (X_matrix, term_names).

    design_kind : 'factorial' | 'mixture'
    model_type  : 'linear' | 'quadratic' | 'cubic'
    """
    n_runs, n_factors = design.shape
    term_names = []
    cols = []

    if design_kind == "factorial":
        # Intercept
        cols.append(np.ones(n_runs))
        term_names.append("Intercept")
        # Main effects
        for i in range(n_factors):
            cols.append(design[:, i])
            term_names.append(f"X{i+1}")
        # 2-way interactions
        if model_type in ("quadratic", "cubic"):
            for i, j in combinations(range(n_factors), 2):
                cols.append(design[:, i] * design[:, j])
                term_names.append(f"X{i+1}×X{j+1}")
        # Pure quadratic (X²)
        if model_type in ("quadratic", "cubic"):
            for i in range(n_factors):
                cols.append(design[:, i] ** 2)
                term_names.append(f"X{i+1}²")
        # 3-way interactions
        if model_type == "cubic":
            for i, j, k in combinations(range(n_factors), 3):
                cols.append(design[:, i] * design[:, j] * design[:, k])
                term_names.append(f"X{i+1}×X{j+1}×X{k+1}")

    else:  # mixture – Scheffé (no intercept, sum = 1 constraint)
        # Linear terms
        for i in range(n_factors):
            cols.append(design[:, i])
            term_names.append(f"x{i+1}")
        # 2-way interactions (β_ij * xi*xj)
        if model_type in ("quadratic", "cubic"):
            for i, j in combinations(range(n_factors), 2):
                cols.append(design[:, i] * design[:, j])
                term_names.append(f"x{i+1}·x{j+1}")
        # 3-way interactions
        if model_type == "cubic":
            for i, j, k in combinations(range(n_factors), 3):
                cols.append(design[:, i] * design[:, j] * design[:, k])
                term_names.append(f"x{i+1}·x{j+1}·x{k+1}")

    X = np.column_stack(cols)
    return X, term_names


def fit_model_statistics(design: np.ndarray, responses: np.ndarray,
                         model_type: str, design_kind: str,
                         factor_names: list[str]) -> dict:
    """
    Fit OLS model and return a comprehensive statistics dict.
    """
    X, raw_term_names = _build_model_matrix(design, model_type, design_kind)

    # Replace generic xi / Xi with actual factor names
    term_names = []
    for tn in raw_term_names:
        t = tn
        for idx, fname in enumerate(factor_names):
            t = t.replace(f"X{idx+1}", fname).replace(f"x{idx+1}", fname)
        term_names.append(t)

    n_obs, n_params = X.shape
    df_res = n_obs - n_params

    # OLS via least-squares (no intercept for mixture; intercept column included for factorial)
    try:
        coeffs, _residuals_ss, _rank, _sv = np.linalg.lstsq(X, responses, rcond=None)
    except Exception:
        return {"error": "Model matrix is singular – insufficient runs for selected model."}

    y_pred = X @ coeffs
    residuals = responses - y_pred
    ss_tot = np.sum((responses - np.mean(responses)) ** 2)
    ss_res = np.sum(residuals ** 2)
    ss_reg = ss_tot - ss_res
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (ss_res / max(df_res, 1)) / (ss_tot / max(n_obs - 1, 1)) if ss_tot > 0 else 0.0
    mse = ss_res / max(df_res, 1)
    rmse = np.sqrt(mse)

    # Standard errors & statistics
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        var_coeffs = mse * np.diag(XtX_inv)
        se = np.sqrt(np.abs(var_coeffs))
        t_stats = coeffs / (se + 1e-300)
        p_values = 2.0 * (1.0 - scipy_stats.t.cdf(np.abs(t_stats), max(df_res, 1)))
        logworth = -np.log10(np.clip(p_values, 1e-300, 1.0))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)
        t_stats = np.full(n_params, np.nan)
        p_values = np.full(n_params, 1.0)
        logworth = np.zeros(n_params)

    # 95% CI
    t_crit = scipy_stats.t.ppf(0.975, max(df_res, 1))
    ci_lower = coeffs - t_crit * se
    ci_upper = coeffs + t_crit * se

    # F-statistic for overall model
    df_reg = n_params - 1 if design_kind == "factorial" else n_params
    ms_reg = ss_reg / max(df_reg, 1)
    f_stat = ms_reg / mse if mse > 0 else 0.0
    f_p = 1.0 - scipy_stats.f.cdf(f_stat, max(df_reg, 1), max(df_res, 1))

    # ANOVA table
    anova_df = pd.DataFrame({
        "Source": ["Regression", "Residual", "Total"],
        "DF": [df_reg, df_res, n_obs - 1],
        "SS": [ss_reg, ss_res, ss_tot],
        "MS": [ms_reg, mse, np.nan],
        "F": [f_stat, np.nan, np.nan],
        "p-Value": [f_p, np.nan, np.nan],
    })

    # Coefficients table
    coeff_df = pd.DataFrame({
        "Term": term_names,
        "Coefficient": coeffs,
        "Std Error": se,
        "t-Stat": t_stats,
        "p-Value": p_values,
        "LogWorth": logworth,
        "CI Lower 95%": ci_lower,
        "CI Upper 95%": ci_upper,
        "Significant (α=0.05)": p_values < 0.05,
    })

    return {
        "X_matrix": X,
        "term_names": term_names,
        "coefficients": coeffs,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "logworth": logworth,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "y_pred": y_pred,
        "residuals": residuals,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mse": mse,
        "f_stat": f_stat,
        "f_p": f_p,
        "anova_df": anova_df,
        "coeff_df": coeff_df,
        "n_obs": n_obs,
        "n_params": n_params,
        "df_res": df_res,
    }


def build_reduced_model(design: np.ndarray, responses: np.ndarray,
                        selected_term_indices: list[int],
                        model_type: str, design_kind: str,
                        factor_names: list[str]) -> dict:
    """
    Fit reduced model using only the selected term indices from the full model matrix.
    Returns same dict structure as fit_model_statistics.
    """
    X_full, term_names_full = _build_model_matrix(design, model_type, design_kind)
    # Replace Xi -> factor names
    term_names_all = []
    for tn in term_names_full:
        t = tn
        for idx, fname in enumerate(factor_names):
            t = t.replace(f"X{idx+1}", fname).replace(f"x{idx+1}", fname)
        term_names_all.append(t)

    X_reduced = X_full[:, selected_term_indices]
    term_names_reduced = [term_names_all[i] for i in selected_term_indices]

    n_obs, n_params = X_reduced.shape
    df_res = n_obs - n_params

    try:
        coeffs_r, _, _, _ = np.linalg.lstsq(X_reduced, responses, rcond=None)
    except Exception:
        return {"error": "Singular matrix in reduced model."}

    y_pred = X_reduced @ coeffs_r
    residuals = responses - y_pred
    ss_tot = np.sum((responses - np.mean(responses)) ** 2)
    ss_res = np.sum(residuals ** 2)
    ss_reg = ss_tot - ss_res
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (ss_res / max(df_res, 1)) / (ss_tot / max(n_obs - 1, 1)) if ss_tot > 0 else 0.0
    mse = ss_res / max(df_res, 1)
    rmse = np.sqrt(mse)

    try:
        XtX_inv = np.linalg.inv(X_reduced.T @ X_reduced)
        var_c = mse * np.diag(XtX_inv)
        se = np.sqrt(np.abs(var_c))
        t_stats = coeffs_r / (se + 1e-300)
        p_values = 2.0 * (1.0 - scipy_stats.t.cdf(np.abs(t_stats), max(df_res, 1)))
        logworth = -np.log10(np.clip(p_values, 1e-300, 1.0))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)
        t_stats = np.full(n_params, np.nan)
        p_values = np.ones(n_params)
        logworth = np.zeros(n_params)

    t_crit = scipy_stats.t.ppf(0.975, max(df_res, 1))
    ci_lower = coeffs_r - t_crit * se
    ci_upper = coeffs_r + t_crit * se

    df_reg = n_params - 1 if design_kind == "factorial" else n_params
    ms_reg = ss_reg / max(df_reg, 1)
    f_stat = ms_reg / mse if mse > 0 else 0.0
    f_p = 1.0 - scipy_stats.f.cdf(f_stat, max(df_reg, 1), max(df_res, 1))

    anova_df = pd.DataFrame({
        "Source": ["Regression (reduced)", "Residual", "Total"],
        "DF": [df_reg, df_res, n_obs - 1],
        "SS": [ss_reg, ss_res, ss_tot],
        "MS": [ms_reg, mse, np.nan],
        "F": [f_stat, np.nan, np.nan],
        "p-Value": [f_p, np.nan, np.nan],
    })

    coeff_df = pd.DataFrame({
        "Term": term_names_reduced,
        "Coefficient": coeffs_r,
        "Std Error": se,
        "t-Stat": t_stats,
        "p-Value": p_values,
        "LogWorth": logworth,
        "CI Lower 95%": ci_lower,
        "CI Upper 95%": ci_upper,
    })

    return {
        "X_matrix": X_reduced,
        "term_names": term_names_reduced,
        "coefficients": coeffs_r,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "logworth": logworth,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "y_pred": y_pred,
        "residuals": residuals,
        "r2": r2,
        "adj_r2": adj_r2,
        "rmse": rmse,
        "mse": mse,
        "f_stat": f_stat,
        "f_p": f_p,
        "anova_df": anova_df,
        "coeff_df": coeff_df,
        "n_obs": n_obs,
        "n_params": n_params,
        "df_res": df_res,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Aliasing detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_aliasing(design: np.ndarray, model_type: str, design_kind: str,
                    factor_names: list[str], threshold: float = 0.75) -> pd.DataFrame:
    """
    Detect near-aliased / fully-aliased terms in the model matrix.
    Returns a DataFrame of pairs with |correlation| >= threshold.
    """
    X, term_names_raw = _build_model_matrix(design, model_type, design_kind)
    term_names = []
    for tn in term_names_raw:
        t = tn
        for idx, fname in enumerate(factor_names):
            t = t.replace(f"X{idx+1}", fname).replace(f"x{idx+1}", fname)
        term_names.append(t)

    n_terms = X.shape[1]
    norms = np.linalg.norm(X, axis=0)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    X_norm = X / norms

    aliased_rows = []
    for i in range(n_terms):
        for j in range(i + 1, n_terms):
            corr = float(X_norm[:, i] @ X_norm[:, j])
            if abs(corr) >= threshold:
                aliased_rows.append({
                    "Term A": term_names[i],
                    "Term B": term_names[j],
                    "Correlation": round(corr, 4),
                    "Severity": "Complete aliasing" if abs(corr) > 0.99
                                else "Severe" if abs(corr) > 0.90
                                else "Moderate",
                })
    if aliased_rows:
        return pd.DataFrame(aliased_rows)
    return pd.DataFrame(columns=["Term A", "Term B", "Correlation", "Severity"])


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Design generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_factorial_d_optimal(n_factors: int, n_runs: int, model_type: str) -> np.ndarray:
    """Generate D-optimal factorial design in [-1, 1]."""
    if _USE_MODULAR:
        try:
            lhs = latin_hypercube_sampling(max(n_runs * 10, 200), n_factors)
            candidates = lhs * 2 - 1  # map [0,1] → [-1,1]
            alg = create_d_optimal_algorithm("standard", model_type=model_type)
            design, _, _ = alg.optimize_factorial_design(candidates=candidates, n_runs=n_runs)
            return np.array(design)
        except Exception:
            pass
    # Fallback: proper LHS in [-1, 1]
    rng = np.random.default_rng(42)
    n_cand = max(n_runs * 10, 300)
    # Build a Latin-hypercube candidate set: each row is one candidate point
    # Each column uses n_cand strata; pick one random point per stratum then shuffle
    strata_edges = np.linspace(-1.0, 1.0, n_cand + 1)
    lhs_cols = []
    for _ in range(n_factors):
        # For each stratum i, sample uniformly in [strata_edges[i], strata_edges[i+1]]
        lower = strata_edges[:-1]
        upper = strata_edges[1:]
        col = rng.uniform(lower, upper)   # shape (n_cand,)
        rng.shuffle(col)
        lhs_cols.append(col)
    candidates = np.column_stack(lhs_cols)   # (n_cand, n_factors)
    # Greedy D-optimal selection from candidates
    selected = [int(rng.integers(0, n_cand))]
    remaining = list(range(n_cand))
    remaining.remove(selected[0])
    while len(selected) < n_runs and remaining:
        # Pick candidate that maximises minimum distance to already selected
        sel_pts = candidates[selected]
        best_idx = max(remaining,
                       key=lambda i: np.min(np.sum((candidates[i] - sel_pts) ** 2, axis=1)))
        selected.append(best_idx)
        remaining.remove(best_idx)
    return np.clip(candidates[selected], -1.0, 1.0)


def _simplex_lattice_base(n_comp: int, model_type: str) -> np.ndarray:
    """
    Generate classical simplex lattice / centroid base points for a mixture design.

    Returns structured points (all rows sum to 1.0):
      • Pure component vertices    : q rows  (xi = 1, rest = 0)
      • Binary blends (quadratic+) : C(q,2) rows  (xi = xj = 0.5)
      • Ternary blends (cubic)     : C(q,3) rows  (xi = xj = xk = 1/3)
      • Overall centroid           : 1 row   (all xi = 1/q)
    """
    pts = []
    # 1. Pure vertices
    for i in range(n_comp):
        v = np.zeros(n_comp); v[i] = 1.0
        pts.append(v)
    # 2. Binary blends
    if model_type in ("quadratic", "cubic"):
        for i, j in combinations(range(n_comp), 2):
            v = np.zeros(n_comp); v[i] = v[j] = 0.5
            pts.append(v)
    # 3. Ternary blends
    if model_type == "cubic":
        for i, j, k in combinations(range(n_comp), 3):
            v = np.zeros(n_comp); v[i] = v[j] = v[k] = 1.0/3.0
            pts.append(v)
    # 4. Overall centroid
    pts.append(np.ones(n_comp) / n_comp)
    return np.array(pts)


def _simplex_lattice_point_types(n_comp: int, model_type: str) -> list[str]:
    """Return a list of human-readable labels for each row from _simplex_lattice_base."""
    labels = []
    for i in range(n_comp):
        labels.append(f"Vertex {i+1}")
    if model_type in ("quadratic", "cubic"):
        for i, j in combinations(range(n_comp), 2):
            labels.append(f"Binary blend {i+1}–{j+1}")
    if model_type == "cubic":
        for i, j, k in combinations(range(n_comp), 3):
            labels.append(f"Ternary blend {i+1}–{j+1}–{k+1}")
    labels.append("Overall centroid")
    return labels


def _generate_mixture_design(n_components: int, n_runs: int, model_type: str,
                              lows: list | None = None,
                              highs: list | None = None) -> np.ndarray:
    """
    Generate a structured mixture design (simplex lattice base + D-optimal augmentation).

    When `lows` / `highs` are provided (constrained simplex), uses the pseudocomponent
    transformation:
        w_i = (x_i - L_i) / (1 - Σ Lⱼ)   →  unconstrained simplex in w-space
        x_i = L_i + w_i * (1 - Σ Lⱼ)       →  back-transform to real proportions

    Base points in real-proportion space:
      • Constrained vertices    : x_i = L_i + scale (i-th), x_j = L_j (j≠i)
      • Constrained binary blends and centroid (analogously)

    If n_runs > n_base: interior points added via greedy max-min distance.
    If n_runs < n_base: D-optimal subset of base points selected.
    """
    L = np.array(lows if lows is not None else [0.0] * n_components, dtype=float)
    U = np.array(highs if highs is not None else [1.0] * n_components, dtype=float)
    L_sum = float(L.sum())
    scale = max(1.0 - L_sum, 1e-9)    # pseudocomponent scale factor

    def _pseudo_to_real(w_pts: np.ndarray) -> np.ndarray:
        """Convert pseudocomponent proportions → real proportions."""
        return L + w_pts * scale

    # Generate unconstrained base in pseudocomponent space, back-transform
    base_w = _simplex_lattice_base(n_components, model_type)   # in [0,1]
    base   = _pseudo_to_real(base_w)                            # in real proportions

    # ── Modular path: only for UNCONSTRAINED simplex (all Lᵢ = 0) ────────────
    # When lower bounds are present the modular algorithm may renormalise points
    # back to the unconstrained simplex, losing the constraint.  Use the
    # structured pseudocomponent fallback instead.
    if _USE_MODULAR and L_sum < 1e-9:
        try:
            gen = create_candidate_generator("lhs", n_components=n_components,
                                             component_names=[f"x{i+1}" for i in range(n_components)])
            cands_w = np.array([normalize_to_simplex(c) for c in gen.generate_candidates(500)])
            cands = np.vstack([base, cands_w])          # unconstrained: base == base_w
            alg = MixtureDOptimalAlgorithm(model_type=model_type)
            design, _, _ = alg.optimize_mixture_design(candidates=cands, n_runs=n_runs,
                                                        strategy="balanced", max_iterations=200)
            return np.array(design)
        except Exception:
            pass

    # ── Structured fallback (always used when lower bounds > 0) ─────────────
    rng = np.random.default_rng(42)
    n_base = len(base)

    if n_runs <= n_base:
        # Greedy max-min-distance selection from the structured base
        sel = [0]
        remaining = list(range(1, n_base))
        while len(sel) < n_runs and remaining:
            sel_pts = base[sel]
            best = max(remaining,
                       key=lambda i: np.min(np.sum((base[i] - sel_pts) ** 2, axis=1)))
            sel.append(best); remaining.remove(best)
        return base[sel]

    # n_runs > n_base: augment with interior points in the constrained region
    extra = n_runs - n_base
    # Sample in pseudocomponent space (Dirichlet away from vertices), back-transform
    cands_w = rng.dirichlet(np.ones(n_components) * 2, extra * 20)
    # Filter by upper bounds
    cands_w = cands_w[np.all(cands_w * scale + L <= U + 1e-6, axis=1)]
    if len(cands_w) < extra:
        # Loosen if too few pass
        cands_w = rng.dirichlet(np.ones(n_components) * 2, extra * 100)
        cands_w = np.clip(cands_w * scale + L, L, U)  # clip and renorm
        row_sums = (cands_w - L).sum(axis=1, keepdims=True)
        cands_w = L + (cands_w - L) / np.maximum(row_sums, 1e-12) * scale
        cands_w = (cands_w - L) / scale  # back to pseudocomponent
    cands = _pseudo_to_real(cands_w)

    current = base.copy()
    added = []
    remaining = list(range(len(cands)))
    while len(added) < extra and remaining:
        best_i = max(remaining,
                     key=lambda i: np.min(np.sum((cands[i] - current) ** 2, axis=1)))
        added.append(best_i)
        current = np.vstack([current, cands[best_i]])
        remaining.remove(best_i)

    aug = cands[added] if added else cands[:extra]
    return np.vstack([base, aug])


def generate_foldover_factorial(design_coded: np.ndarray, fold_cols: list[int] | str = "all") -> np.ndarray:
    """
    Generate fold-over block for a coded factorial design.

    fold_cols : 'all'  → negate every column (full fold-over, resolution +1)
                [i, …] → negate only listed column indices (partial fold-over)
    """
    fo = design_coded.copy()
    if fold_cols == "all":
        fo = -fo
    else:
        for c in fold_cols:
            fo[:, c] = -fo[:, c]
    return fo


def generate_augmentation_mixture(existing_design: np.ndarray,
                                  n_augment: int,
                                  model_type: str) -> np.ndarray:
    """
    Generate D-optimal augmentation points for a mixture design.
    Selects new runs that maximise |X^T X| of the combined design.
    """
    n_comp = existing_design.shape[1]
    if _USE_MODULAR:
        try:
            gen = create_candidate_generator("lhs", n_components=n_comp,
                                             component_names=[f"x{i+1}" for i in range(n_comp)])
            cands = np.array([normalize_to_simplex(c) for c in gen.generate_candidates(600)])
        except Exception:
            rng = np.random.default_rng(42)
            cands = rng.dirichlet(np.ones(n_comp), 600)
    else:
        rng = np.random.default_rng(42)
        cands = rng.dirichlet(np.ones(n_comp), 600)

    # Greedy augmentation: pick candidate that most increases det(X^T X)
    aug_points = []
    current = existing_design.copy()

    for _ in range(n_augment):
        best_det = -np.inf
        best_pt = None
        X_cur, _ = _build_model_matrix(current, model_type, "mixture")
        base_m = X_cur.T @ X_cur

        for cand in cands:
            x_new, _ = _build_model_matrix(cand.reshape(1, -1), model_type, "mixture")
            m_new = base_m + x_new.T @ x_new
            try:
                d = np.linalg.det(m_new)
            except Exception:
                d = 0.0
            if d > best_det:
                best_det = d
                best_pt = cand

        if best_pt is not None:
            aug_points.append(best_pt)
            current = np.vstack([current, best_pt.reshape(1, -1)])
            # Remove chosen point from candidates
            cands = cands[~np.all(np.isclose(cands, best_pt), axis=1)]

    return np.array(aug_points) if aug_points else cands[:n_augment]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Visualization (shared)
# ─────────────────────────────────────────────────────────────────────────────

def half_normal_plot(coeff_df: pd.DataFrame) -> go.Figure:
    """Half-normal probability plot of |coefficients| using Lenth's method."""
    terms = coeff_df[~coeff_df["Term"].isin(["Intercept"])].copy()
    abs_eff = terms["Coefficient"].abs().values
    names = terms["Term"].values

    idx = np.argsort(abs_eff)
    sorted_abs = abs_eff[idx]
    sorted_names = names[idx]
    n = len(sorted_abs)

    probs = (np.arange(1, n + 1) - 0.5) / n
    quantiles = scipy_stats.norm.ppf((probs + 1) / 2)

    # Lenth's PSE
    cutoff = max(3, n // 3)
    pse = float(np.median(sorted_abs[:cutoff] / (quantiles[:cutoff] + 1e-12)))
    df_l = max(1, cutoff - 1)
    ME = scipy_stats.t.ppf(0.975, df_l) * pse
    SME = scipy_stats.t.ppf(1.0 - 0.05 / (2 * max(n, 1)), df_l) * pse

    colors = ["red" if v > SME else "orange" if v > ME else "steelblue" for v in sorted_abs]

    fig = go.Figure()
    # Reference line
    xmax = quantiles.max() * 1.1
    fig.add_trace(go.Scatter(x=[0, xmax], y=[0, pse * xmax],
                             mode="lines", line=dict(color="gray", dash="dash", width=1.5),
                             name="Non-sig reference"))
    fig.add_trace(go.Scatter(x=[0, xmax], y=[ME, ME],
                             mode="lines", line=dict(color="orange", dash="dot", width=1),
                             name=f"ME={ME:.3f}"))
    fig.add_trace(go.Scatter(x=[0, xmax], y=[SME, SME],
                             mode="lines", line=dict(color="red", dash="dot", width=1),
                             name=f"SME={SME:.3f}"))
    fig.add_trace(go.Scatter(
        x=quantiles, y=sorted_abs, mode="markers+text",
        marker=dict(size=9, color=colors, line=dict(width=1, color="black")),
        text=sorted_names, textposition="top right", textfont=dict(size=9),
        hovertemplate="<b>%{text}</b><br>|Effect|=%{y:.4f}<extra></extra>",
        name="Effects",
    ))
    fig.update_layout(title="Half-Normal Plot of Effects (Lenth's Method)",
                      xaxis_title="Half-Normal Quantile", yaxis_title="|Coefficient|",
                      height=420, showlegend=True)
    return fig


def pareto_chart(coeff_df: pd.DataFrame) -> go.Figure:
    """Pareto chart of |t-statistics| sorted descending."""
    terms = coeff_df[~coeff_df["Term"].isin(["Intercept"])].copy()
    terms = terms.dropna(subset=["t-Stat"])
    terms["abs_t"] = terms["t-Stat"].abs()
    terms = terms.sort_values("abs_t", ascending=True)

    colors = ["crimson" if p < 0.05 else "steelblue" for p in terms["p-Value"]]
    fig = go.Figure(go.Bar(
        x=terms["abs_t"], y=terms["Term"], orientation="h",
        marker_color=colors, text=terms["p-Value"].apply(lambda p: f"p={p:.3f}"),
        textposition="outside",
    ))
    # Significance threshold line (t-value at α=0.05)
    df_res = max(1, int(coeff_df.shape[0] - 1))
    t_crit = scipy_stats.t.ppf(0.975, df_res)
    fig.add_vline(x=t_crit, line_dash="dash", line_color="orange",
                  annotation_text=f"α=0.05 (t={t_crit:.2f})", annotation_position="top right")
    fig.update_layout(title="Pareto Chart of |t-Statistics|",
                      xaxis_title="|t-Statistic|", yaxis_title="",
                      height=max(350, len(terms) * 30), showlegend=False)
    return fig


def residual_plots(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """2×2 panel: residuals vs fitted, histogram, Q-Q, actual vs predicted."""
    resid = y_actual - y_pred
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Residuals vs Fitted", "Residuals Histogram",
                                        "Normal Q-Q", "Actual vs Predicted"))
    # Residuals vs fitted
    fig.add_trace(go.Scatter(x=y_pred, y=resid, mode="markers",
                             marker=dict(color="steelblue", size=7, opacity=0.7),
                             name="Residuals"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # Histogram
    fig.add_trace(go.Histogram(x=resid, nbinsx=12, name="Residuals",
                               marker_color="lightblue", opacity=0.8), row=1, col=2)

    # Q-Q
    sorted_resid = np.sort(resid)
    n = len(sorted_resid)
    probs = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical_q = scipy_stats.norm.ppf(probs)
    fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_resid, mode="markers",
                             marker=dict(color="steelblue", size=6), name="Q-Q"), row=2, col=1)
    mn, mx = theoretical_q.min(), theoretical_q.max()
    slope, intercept = np.polyfit(theoretical_q, sorted_resid, 1)
    fig.add_trace(go.Scatter(x=[mn, mx], y=[slope * mn + intercept, slope * mx + intercept],
                             mode="lines", line=dict(color="red", dash="dash"), name="Normal line"),
                  row=2, col=1)

    # Actual vs predicted
    lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect fit"), row=2, col=2)
    fig.add_trace(go.Scatter(x=y_actual, y=y_pred, mode="markers",
                             marker=dict(color="steelblue", size=7, opacity=0.7),
                             name="Data"), row=2, col=2)

    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Residuals", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantile", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantile", row=2, col=1)
    fig.update_xaxes(title_text="Actual", row=2, col=2)
    fig.update_yaxes(title_text="Predicted", row=2, col=2)
    return fig


def _coeff_df_styled(coeff_df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready version of coeff_df (formatted strings)."""
    df = coeff_df.copy()
    for col in ["Coefficient", "Std Error", "CI Lower 95%", "CI Upper 95%"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    for col in ["t-Stat"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    if "p-Value" in df.columns:
        df["p-Value"] = df["p-Value"].apply(
            lambda x: ("<0.001" if x < 0.001 else f"{x:.4f}") if pd.notna(x) else "")
    if "LogWorth" in df.columns:
        df["LogWorth"] = df["LogWorth"].apply(
            lambda x: "∞" if x == np.inf else (f"{x:.2f}" if pd.notna(x) else ""))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "workflow_stage": 1,
        "design_kind": "factorial",
        "n_factors": 3,
        "factor_names": ["A", "B", "C"],
        "factor_lows": [-1.0, -1.0, -1.0],
        "factor_highs": [1.0, 1.0, 1.0],
        "response_name": "Response",
        "model_type": "quadratic",
        # Stage 2
        "initial_design_coded": None,
        "initial_design_natural": None,
        "n_initial_runs": 12,
        # Stage 3
        "initial_responses": None,
        # Stage 4
        "screening_results": None,
        "aliasing_df": None,
        "selected_term_indices": None,
        # Stage 5
        "foldover_design_coded": None,
        "foldover_design_natural": None,
        "foldover_responses": None,
        "fold_columns": "all",
        # Stage 6
        "combined_design": None,
        "combined_responses": None,
        "final_results": None,
        # Mixture parts mode
        "mixture_parts_mode": False,
        "mixture_batch_total": 100.0,
        "mixture_units": "g",
        "mixture_parts_mins": [],
        "mixture_parts_maxs": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
s = st.session_state  # shorthand

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – PROGRESS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

stage_labels = {
    1: "⚙️ Problem Setup",
    2: "🎯 Initial Design",
    3: "📋 Data Entry",
    4: "📊 Screening Analysis",
    5: "🔄 Fold-Over / Augment",
    6: "🏆 Final Model",
}

st.sidebar.title("🔬 DOE Sequential Workflow")
st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Progress")
for idx, label in stage_labels.items():
    if idx < s.workflow_stage:
        st.sidebar.markdown(f"✅ {label}")
    elif idx == s.workflow_stage:
        st.sidebar.markdown(f"**▶ {label}**")
    else:
        st.sidebar.markdown(f"⬜ {label}")

st.sidebar.markdown("---")
if st.sidebar.button("🔁 Restart Workflow"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "This app guides you through:\n"
    "1. Initial screening design\n"
    "2. Significance-based factor selection\n"
    "3. Fold-over for de-aliasing (factorial)\n"
    "   OR D-optimal augmentation (mixture)\n"
    "4. Competitive final model"
)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 DOE Sequential Workflow")
st.markdown(
    "Build a **competitive DOE model from scratch** using sequential experimentation:\n"
    "screen → identify significant factors → de-alias if needed → final model."
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1: PROBLEM SETUP
# ─────────────────────────────────────────────────────────────────────────────

if s.workflow_stage == 1:
    st.header("⚙️ Stage 1 — Problem Setup")

    col1, col2 = st.columns(2)
    with col1:
        design_kind = st.selectbox(
            "Experiment Type",
            ["factorial", "mixture"],
            format_func=lambda x: "🔢 Factorial / Standard DOE" if x == "factorial" else "🧪 Mixture Design",
            index=0 if s.design_kind == "factorial" else 1,
        )
        response_name = st.text_input("Response variable name", value=s.response_name)
        model_type = st.selectbox("Model order",
                                   ["linear", "quadratic", "cubic"],
                                   index=["linear", "quadratic", "cubic"].index(s.model_type))

    with col2:
        if design_kind == "factorial":
            n_factors = st.number_input("Number of factors", 2, 10, s.n_factors, step=1)
        else:
            n_factors = st.number_input("Number of components", 2, 8, s.n_factors, step=1)

    # Factor / component names and bounds
    st.subheader("Factor / Component Details")
    factor_names = []
    factor_lows = []
    factor_highs = []

    if design_kind == "factorial":
        st.info("Enter names and the **actual (natural) range** for each factor. "
                "The app codes them to [−1, +1] internally.")
        header_cols = st.columns([2, 2, 2])
        header_cols[0].markdown("**Factor Name**")
        header_cols[1].markdown("**Low level**")
        header_cols[2].markdown("**High level**")
        for i in range(n_factors):
            c1, c2, c3 = st.columns([2, 2, 2])
            prev_name = s.factor_names[i] if i < len(s.factor_names) else f"X{i+1}"
            prev_low  = s.factor_lows[i]  if i < len(s.factor_lows)  else -1.0
            prev_high = s.factor_highs[i] if i < len(s.factor_highs) else  1.0
            name = c1.text_input(f"Factor {i+1} name", value=prev_name, key=f"fn_{i}", label_visibility="collapsed")
            lo   = c2.number_input(f"Low {i+1}",  value=float(prev_low),  key=f"fl_{i}", label_visibility="collapsed")
            hi   = c3.number_input(f"High {i+1}", value=float(prev_high), key=f"fh_{i}", label_visibility="collapsed")
            factor_names.append(name)
            factor_lows.append(lo)
            factor_highs.append(hi)
    else:
        # ── Parts mode toggle ──────────────────────────────────────────────────
        parts_mode = st.toggle(
            "🧮 Parts mode — enter component amounts in real units (g, mL, parts by weight…)",
            value=s.mixture_parts_mode,
            key="mix_parts_toggle",
            help=(
                "Instead of proportions (0–1), enter each component's range in actual "
                "units (grams, mL, % w/w, phr …) and a fixed batch total.  "
                "The app computes the proportion bounds AND tightens each component's "
                "achievable range by accounting for the constraints of all other components."
            ),
        )
        s.mixture_parts_mode = parts_mode

        if parts_mode:
            # ── Parts mode UI (no batch total — variable-sum normalization) ───
            units_label = st.text_input(
                "Units label (for display only)",
                value=s.mixture_units,
                help="e.g. g, mL, phr, wt%",
            )
            s.mixture_units = units_label

            st.info(
                f"Enter the **minimum and maximum amounts** for each component "
                f"in **{units_label}**. "
                f"The proportion of each component in any blend is computed as "
                f"**amount ÷ sum of all components** — no fixed batch total needed. "
                f"The table below shows the achievable proportion range for each component "
                f"knowing the ranges of all other components."
            )

            hdr = st.columns([2, 1.5, 1.5])
            hdr[0].markdown(f"**Component Name**")
            hdr[1].markdown(f"**Min ({units_label})**")
            hdr[2].markdown(f"**Max ({units_label})**")

            parts_names: list[str] = []
            parts_mins:  list[float] = []
            parts_maxs:  list[float] = []

            for i in range(n_factors):
                c1, c2, c3 = st.columns([2, 1.5, 1.5])
                prev_name = s.factor_names[i] if i < len(s.factor_names) else f"C{i+1}"
                _def_min = (s.mixture_parts_mins[i]
                            if i < len(s.mixture_parts_mins) else 0.0)
                _def_max = (s.mixture_parts_maxs[i]
                            if i < len(s.mixture_parts_maxs) else 100.0)
                name  = c1.text_input(
                    f"Component {i+1}", value=prev_name,
                    key=f"fn_{i}", label_visibility="collapsed",
                )
                p_min = c2.number_input(
                    f"Min {i+1}", value=float(_def_min),
                    min_value=0.0, step=0.5, format="%.3f",
                    key=f"pmin_{i}", label_visibility="collapsed",
                )
                p_max = c3.number_input(
                    f"Max {i+1}", value=float(_def_max),
                    min_value=0.0, step=0.5, format="%.3f",
                    key=f"pmax_{i}", label_visibility="collapsed",
                )
                parts_names.append(name)
                parts_mins.append(p_min)
                parts_maxs.append(p_max)

            # Persist for next render
            s.mixture_parts_mins = parts_mins
            s.mixture_parts_maxs = parts_maxs

            # ── Derive proportion bounds via variable-sum normalization ────────
            # Proportion of component i in any blend = parts_i / sum(all parts)
            #
            # prop_min_i: component i at its minimum, all others at their maximum
            #   prop_min_i = a_i / (a_i + Σ_{j≠i} b_j)
            #
            # prop_max_i: component i at its maximum, all others at their minimum
            #   prop_max_i = b_i / (b_i + Σ_{j≠i} a_j)
            a     = np.array(parts_mins, dtype=float)
            b     = np.array(parts_maxs, dtype=float)
            sum_a = float(a.sum())
            sum_b = float(b.sum())

            # Avoid division by zero
            denom_min = a + (sum_b - b)   # a_i + Σ_{j≠i} b_j
            denom_max = b + (sum_a - a)   # b_i + Σ_{j≠i} a_j
            denom_min = np.where(denom_min < 1e-12, 1e-12, denom_min)
            denom_max = np.where(denom_max < 1e-12, 1e-12, denom_max)

            prop_low  = np.clip(a / denom_min, 0.0, 1.0)
            prop_high = np.clip(b / denom_max, 0.0, 1.0)

            # Reference total for Stage 2 parts view (sum of midpoints)
            T_ref = float(((a + b) / 2).sum())
            s.mixture_batch_total = T_ref   # store for Stage 2 display

            # Summary table
            summary_rows = []
            for i in range(n_factors):
                mid = (a[i] + b[i]) / 2
                summary_rows.append({
                    "Component":                parts_names[i],
                    f"Min ({units_label})":     f"{a[i]:.3f}",
                    f"Max ({units_label})":     f"{b[i]:.3f}",
                    f"Midpoint ({units_label})": f"{mid:.3f}",
                    "Proportion min":           f"{prop_low[i]:.4f}",
                    "Proportion max":           f"{prop_high[i]:.4f}",
                    "Proportion at midpoints":  f"{mid / T_ref:.4f}" if T_ref > 0 else "—",
                })
            summary_df = pd.DataFrame(summary_rows)

            # Validation — only hard-block on truly impossible inputs (negative amounts)
            _hard_errors = []
            _warnings    = []
            for i in range(n_factors):
                if a[i] < 0:
                    _hard_errors.append(f"**{parts_names[i]}**: min amount is negative")
                    continue
                if b[i] < a[i]:
                    # Treat reversed bounds as a warning — may be intentional (filler component)
                    _warnings.append(
                        f"**{parts_names[i]}**: max ({b[i]:.3f}) < min ({a[i]:.3f}) — "
                        f"this component will act as a fixed point in the design"
                    )
                if abs(prop_high[i] - prop_low[i]) < 1e-6:
                    _warnings.append(
                        f"**{parts_names[i]}**: proportion range is essentially zero "
                        f"({prop_low[i]:.4f}) — consider widening its range or "
                        f"narrowing other components"
                    )

            if _hard_errors:
                for msg in _hard_errors:
                    st.error(f"❌ {msg}")
            else:
                st.success(
                    f"✅ Valid.  "
                    f"Batch total range: **{sum_a:.3f} – {sum_b:.3f} {units_label}** "
                    f"(reference midpoint total = {T_ref:.3f} {units_label})"
                )
                for msg in _warnings:
                    st.warning(f"⚠️ {msg}")

            with st.expander("📊 Derived proportion bounds (accounting for other components)", expanded=True):
                st.dataframe(summary_df, use_container_width=True)
                st.caption(
                    "**Proportion min/max** = achievable proportion range for each component "
                    "when it is at its extreme value and all other components are at the opposite extreme.  "
                    "Proportion at midpoints = each component at its midpoint amount ÷ sum of all midpoints.  "
                    "These proportion bounds are used for design generation."
                )

            # Set factor lists for downstream stages
            factor_names = parts_names
            factor_lows  = [float(max(prop_low[i],  0.0)) for i in range(n_factors)]
            factor_highs = [float(min(prop_high[i], 1.0)) for i in range(n_factors)]

        else:
            # ── Proportion mode (original) ─────────────────────────────────────
            st.info(
                "Enter names and optional **lower / upper bounds** for each component. "
                "Proportions must sum to 1.  "
                "Leaving bounds at 0 / 1 gives an unconstrained simplex.  "
                "Set lower bounds > 0 to enforce a minimum proportion for that component."
            )
            header_c = st.columns([2, 1.5, 1.5])
            header_c[0].markdown("**Component Name**")
            header_c[1].markdown("**Min proportion** (≥ 0)")
            header_c[2].markdown("**Max proportion** (≤ 1)")

            for i in range(n_factors):
                c1, c2, c3 = st.columns([2, 1.5, 1.5])
                prev_name = s.factor_names[i] if i < len(s.factor_names) else f"C{i+1}"
                # Clamp previous values to mixture-valid range [0,1] in case session
                # still holds factorial values (e.g. -1 … 1) from a prior run
                prev_low  = max(0.0, min(0.999, float(s.factor_lows[i])))  if i < len(s.factor_lows)  else 0.0
                prev_high = max(0.001, min(1.0,  float(s.factor_highs[i]))) if i < len(s.factor_highs) else 1.0
                name = c1.text_input(f"Component {i+1}", value=prev_name,
                                      key=f"fn_{i}", label_visibility="collapsed")
                lo   = c2.number_input(f"Min {i+1}", value=float(prev_low),
                                       min_value=0.0, max_value=0.999, step=0.01,
                                       format="%.3f", key=f"fl_{i}", label_visibility="collapsed")
                hi   = c3.number_input(f"Max {i+1}", value=float(prev_high),
                                       min_value=0.001, max_value=1.0, step=0.01,
                                       format="%.3f", key=f"fh_{i}", label_visibility="collapsed")
                factor_names.append(name)
                factor_lows.append(lo)
                factor_highs.append(hi)

            # Validate bounds
            _L_sum = sum(factor_lows)
            _U_sum = sum(factor_highs)
            if _L_sum >= 1.0:
                st.error(f"❌ Sum of lower bounds = {_L_sum:.3f} ≥ 1.0 — no feasible region exists. "
                          f"Reduce at least one lower bound.")
            elif _U_sum <= 1.0:
                st.error(f"❌ Sum of upper bounds = {_U_sum:.3f} ≤ 1.0 — no feasible region exists. "
                          f"Increase at least one upper bound.")
            elif _L_sum > 0:
                _rem = 1.0 - _L_sum
                st.success(
                    f"✅ Bounds valid. Σ lower = {_L_sum:.3f}, Σ upper = {_U_sum:.3f}.  "
                    f"Free range = {_rem:.3f}  "
                    f"(pseudocomponent scaling: each wᵢ = (xᵢ − Lᵢ) / {_rem:.3f})"
                )

        # ── Scheffé model explanation (shown for both modes) ───────────────────
        _n = n_factors
        _lin_terms = _n
        _int_terms = _n * (_n - 1) // 2 if model_type in ("quadratic","cubic") else 0
        _tri_terms = _n*(_n-1)*(_n-2)//6 if model_type == "cubic" else 0
        _total_terms = _lin_terms + _int_terms + _tri_terms
        _n_base_pts  = len(_simplex_lattice_base(_n, model_type))
        with st.expander("📐 Scheffé Mixture Model — Explanation", expanded=True):
            if model_type == "linear":
                st.markdown(
                    f"**Linear Scheffé model** ({_total_terms} terms, no intercept):\n\n"
                    r"$$\hat{y} = \sum_{i=1}^{q} \beta_i x_i$$\n\n"
                    f"- {_lin_terms} linear terms (βᵢ)\n"
                    f"- Design base: {_n} vertices + 1 centroid = **{_n_base_pts} structured points**"
                )
            elif model_type == "quadratic":
                st.markdown(
                    f"**Quadratic Scheffé model** ({_total_terms} terms, no intercept):\n\n"
                    r"$$\hat{y} = \sum_{i} \beta_i x_i + \sum_{i<j} \beta_{ij} x_i x_j$$\n\n"
                    f"- {_lin_terms} linear terms (βᵢ) + {_int_terms} binary interaction terms (βᵢⱼ)\n"
                    f"- **Note:** There are no pure quadratic xᵢ² terms — because Σxᵢ = 1 "
                    f"creates a linear dependency and xᵢ² is absorbed into the linear form.\n"
                    f"- Design base: {_n} vertices + {_int_terms} binary blends (xᵢ=xⱼ=0.5) "
                    f"+ 1 centroid = **{_n_base_pts} structured points**\n"
                    f"- Minimum runs needed: **{_total_terms + 2}** "
                    f"(= {_total_terms} parameters + 2 df for error)"
                )
            else:  # cubic
                st.markdown(
                    f"**Cubic Scheffé model** ({_total_terms} terms, no intercept):\n\n"
                    r"$$\hat{y} = \sum_{i} \beta_i x_i + \sum_{i<j} \beta_{ij} x_i x_j "
                    r"+ \sum_{i<j<k} \beta_{ijk} x_i x_j x_k$$\n\n"
                    f"- {_lin_terms} linear + {_int_terms} binary + {_tri_terms} ternary terms\n"
                    f"- Design base: vertices + binary blends + ternary blends + centroid "
                    f"= **{_n_base_pts} structured points**\n"
                    f"- Minimum runs needed: **{_total_terms + 2}**"
                )

    # Number of initial runs
    # Compute minimum
    _, tmp_names = _build_model_matrix(np.zeros((1, n_factors)), model_type, design_kind)
    min_runs = len(tmp_names) + 2
    n_initial = st.number_input(
        f"Number of initial screening runs (min {min_runs})",
        min_value=min_runs, max_value=100, value=max(min_runs, s.n_initial_runs), step=1,
        help=f"Minimum {min_runs} runs needed to estimate {len(tmp_names)} parameters with {2} degrees of freedom for error.",
    )

    if st.button("✅ Confirm Setup & Proceed to Initial Design", type="primary"):
        s.design_kind = design_kind
        s.n_factors = n_factors
        s.factor_names = factor_names
        s.factor_lows = factor_lows
        s.factor_highs = factor_highs
        s.response_name = response_name
        s.model_type = model_type
        s.n_initial_runs = n_initial
        s.workflow_stage = 2
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: INITIAL DESIGN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

elif s.workflow_stage == 2:
    st.header("🎯 Stage 2 — Initial Screening Design")

    st.info(
        f"Generating a D-optimal **{s.model_type}** screening design "
        f"for **{s.n_factors}** {'factors' if s.design_kind == 'factorial' else 'components'} "
        f"with **{s.n_initial_runs}** runs."
    )

    if st.button("🔄 Generate Initial Design", type="primary"):
        with st.spinner("Generating design..."):
            if s.design_kind == "factorial":
                coded = _generate_factorial_d_optimal(s.n_factors, s.n_initial_runs, s.model_type)
                # Convert coded [-1,1] → natural units
                lows  = np.array(s.factor_lows)
                highs = np.array(s.factor_highs)
                natural = lows + (coded + 1) / 2 * (highs - lows)
                s.initial_design_coded   = coded
                s.initial_design_natural = natural
            else:
                mix = _generate_mixture_design(
                    s.n_factors, s.n_initial_runs, s.model_type,
                    lows=s.factor_lows, highs=s.factor_highs,
                )
                s.initial_design_coded   = mix
                s.initial_design_natural = mix

        st.success(f"✅ Design generated with {s.n_initial_runs} runs.")

    if s.initial_design_natural is not None:
        design_df = pd.DataFrame(s.initial_design_natural, columns=s.factor_names)
        design_df.insert(0, "Run", range(1, len(design_df) + 1))
        design_df.insert(1, s.response_name, [None] * len(design_df))

        st.subheader("📋 Design Matrix (download this for experiments)")

        # For mixture designs: annotate point types
        if s.design_kind == "mixture":
            # Compute constrained base points (pseudocomponent back-transform)
            _L = np.array(s.factor_lows)
            _scale = 1.0 - float(_L.sum())
            base_w   = _simplex_lattice_base(s.n_factors, s.model_type)
            base_pts = _L + base_w * _scale   # real proportions
            n_base   = len(base_pts)
            base_labels = _simplex_lattice_point_types(s.n_factors, s.model_type)
            pt_types = []
            for row in s.initial_design_natural:
                matched = False
                for bi, bp in enumerate(base_pts):
                    if np.allclose(row, bp, atol=1e-4):
                        pt_types.append(base_labels[bi]); matched = True; break
                if not matched:
                    pt_types.append("Interior point")
            display_df = design_df.copy()
            display_df.insert(2, "Point Type", pt_types)
            st.dataframe(display_df.style.format(
                {c: "{:.4f}" for c in s.factor_names}
            ), use_container_width=True)
            # Explain point type breakdown
            from collections import Counter
            pt_counts = Counter(pt_types)
            pt_summary = ", ".join(f"{v} × {k}" for k, v in pt_counts.items())
            st.caption(f"🧩 Design composition: {pt_summary}")
        else:
            st.dataframe(design_df.style.format(
                {c: "{:.4f}" for c in s.factor_names}
            ), use_container_width=True)

        # Show sum check for mixture
        if s.design_kind == "mixture":
            row_sums = s.initial_design_natural.sum(axis=1)
            if np.allclose(row_sums, 1.0, atol=1e-4):
                st.success("✅ All rows sum to 1.0")
            else:
                st.warning(f"⚠️ Row sums range {row_sums.min():.4f}–{row_sums.max():.4f}")

            # ── Parts view when parts mode was used ──────────────────────────
            if s.mixture_parts_mode and s.mixture_batch_total > 0:
                T   = float(s.mixture_batch_total)
                ul  = s.mixture_units
                parts_view = pd.DataFrame(s.initial_design_natural * T,
                                          columns=[f"{n} ({ul})" for n in s.factor_names])
                parts_view.insert(0, "Run", range(1, len(parts_view) + 1))
                parts_view.insert(1, s.response_name, [None] * len(parts_view))
                parts_view[f"Total ({ul})"] = (s.initial_design_natural * T).sum(axis=1)
                with st.expander(f"📦 Parts view ({ul}) — batch total = {T:.3f} {ul}", expanded=True):
                    st.dataframe(parts_view.style.format(
                        {c: "{:.3f}" for c in parts_view.columns
                         if c not in ("Run", s.response_name)}
                    ), use_container_width=True)
                    buf_parts = io.BytesIO()
                    with pd.ExcelWriter(buf_parts, engine="openpyxl") as _w:
                        parts_view.to_excel(_w, sheet_name="Parts Design", index=False)
                    st.download_button(
                        f"📥 Download Parts Design (Excel)",
                        buf_parts.getvalue(),
                        file_name="initial_design_parts.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        # Download – Excel primary, CSV secondary
        buf_xlsx = io.BytesIO()
        with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as writer:
            design_df.to_excel(writer, sheet_name="Initial Design", index=False)
        dl1, dl2 = st.columns(2)
        dl1.download_button("📥 Download Design (Excel)", buf_xlsx.getvalue(),
                            file_name="initial_design.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        dl2.download_button("📥 Download Design (CSV)", design_df.to_csv(index=False, float_format="%.6f"),
                            file_name="initial_design.csv", mime="text/csv")

        # Ternary plot for 3-component mixture
        if s.design_kind == "mixture" and s.n_factors == 3:
            fig = go.Figure(go.Scatterternary(
                a=s.initial_design_natural[:, 0],
                b=s.initial_design_natural[:, 1],
                c=s.initial_design_natural[:, 2],
                mode="markers+text",
                text=[f"R{i+1}" for i in range(s.n_initial_runs)],
                textposition="top center",
                marker=dict(size=10, color="steelblue"),
            ))
            fig.update_layout(
                title="Initial Design – Ternary Plot",
                ternary=dict(sum=1,
                             aaxis=dict(title=s.factor_names[0]),
                             baxis=dict(title=s.factor_names[1]),
                             caxis=dict(title=s.factor_names[2])),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        col_back, col_fwd = st.columns(2)
        with col_back:
            if st.button("◀ Back to Setup"):
                s.workflow_stage = 1; st.rerun()
        with col_fwd:
            if st.button("▶ Proceed to Data Entry", type="primary"):
                s.workflow_stage = 3; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3: DATA ENTRY
# ─────────────────────────────────────────────────────────────────────────────

elif s.workflow_stage == 3:
    st.header("📋 Stage 3 — Enter Experimental Responses")

    st.info(
        "Run your experiments according to the design generated in Stage 2, "
        "then enter the responses below. Alternatively, upload a CSV/Excel file "
        "with the completed design (the response column must be present)."
    )

    tab_manual, tab_upload = st.tabs(["✏️ Manual Entry", "📁 Upload File"])

    with tab_manual:
        design_df = pd.DataFrame(s.initial_design_natural, columns=s.factor_names)
        design_df.insert(0, "Run", range(1, len(design_df) + 1))

        # Pre-fill with any previously entered responses
        if s.initial_responses is not None:
            design_df[s.response_name] = s.initial_responses
        else:
            design_df[s.response_name] = np.nan

        st.write("**Fill in the Response column (click a cell to edit):**")
        edited = st.data_editor(
            design_df,
            column_config={
                s.response_name: st.column_config.NumberColumn(
                    s.response_name, help="Enter the measured response value",
                    required=True,
                ),
                "Run": st.column_config.NumberColumn("Run", disabled=True),
                **{c: st.column_config.NumberColumn(c, disabled=True) for c in s.factor_names},
            },
            hide_index=True, use_container_width=True,
        )
        if st.button("💾 Save Manual Responses", type="primary"):
            resp = edited[s.response_name].values.astype(float)
            if np.any(np.isnan(resp)):
                st.error("❌ Some responses are still empty. Please complete all rows.")
            else:
                s.initial_responses = resp
                st.success("✅ Responses saved!")

    with tab_upload:
        uploaded = st.file_uploader("Upload completed design file (CSV or Excel)",
                                     type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)
                st.write("Preview:", df_up.head())
                resp_col = st.selectbox("Response column", df_up.columns.tolist())
                if st.button("💾 Load Responses from File"):
                    resp = df_up[resp_col].values.astype(float)
                    if len(resp) != s.n_initial_runs:
                        st.error(f"❌ Expected {s.n_initial_runs} rows, got {len(resp)}.")
                    elif np.any(np.isnan(resp)):
                        st.error("❌ Some response values are NaN.")
                    else:
                        s.initial_responses = resp
                        st.success("✅ Responses loaded from file!")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Status & navigation
    if s.initial_responses is not None:
        st.success(f"✅ {len(s.initial_responses)} responses ready. "
                   f"Range: [{s.initial_responses.min():.3f}, {s.initial_responses.max():.3f}]")

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Design"):
            s.workflow_stage = 2; st.rerun()
    with col_fwd:
        if st.button("▶ Proceed to Screening Analysis", type="primary",
                     disabled=(s.initial_responses is None)):
            s.workflow_stage = 4; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4: SCREENING ANALYSIS + FACTOR SELECTION
# ─────────────────────────────────────────────────────────────────────────────

elif s.workflow_stage == 4:
    st.header("📊 Stage 4 — Screening Analysis & Factor Selection")

    design = s.initial_design_coded
    responses = s.initial_responses

    # Include combined data if fold-over was done previously and user returns
    if s.foldover_responses is not None and s.foldover_design_coded is not None:
        design = np.vstack([s.initial_design_coded, s.foldover_design_coded])
        responses = np.concatenate([s.initial_responses, s.foldover_responses])
        st.info(f"ℹ️ Using **combined data** ({len(responses)} runs): "
                f"{s.n_initial_runs} initial + {len(s.foldover_responses)} fold-over/augmentation.")

    # ── Fit full model ────────────────────────────────────────────────────────
    results = fit_model_statistics(design, responses, s.model_type,
                                   s.design_kind, s.factor_names)

    if "error" in results:
        st.error(f"❌ {results['error']}")
        if st.button("◀ Back"):
            s.workflow_stage = 3; st.rerun()
        st.stop()

    s.screening_results = results

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{results['r2']:.4f}")
    col2.metric("Adj R²", f"{results['adj_r2']:.4f}")
    col3.metric("RMSE", f"{results['rmse']:.4f}")
    col4.metric("Overall F p-value",
                "<0.001" if results["f_p"] < 0.001 else f"{results['f_p']:.4f}")

    # ── ANOVA table ───────────────────────────────────────────────────────────
    with st.expander("📋 ANOVA Table", expanded=False):
        st.dataframe(results["anova_df"].style.format({
            "SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-Value": "{:.4f}",
        }, na_rep=""), use_container_width=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(half_normal_plot(results["coeff_df"]), use_container_width=True)
    with chart_col2:
        st.plotly_chart(pareto_chart(results["coeff_df"]), use_container_width=True)

    # ── Aliasing detection ────────────────────────────────────────────────────
    aliasing_df = detect_aliasing(design, s.model_type, s.design_kind, s.factor_names)
    s.aliasing_df = aliasing_df
    if not aliasing_df.empty:
        st.warning(f"⚠️ **{len(aliasing_df)} aliased term pair(s) detected** "
                   f"(|correlation| ≥ 0.75). See table below.")
        with st.expander("🔗 Aliasing Structure", expanded=True):
            st.dataframe(aliasing_df, use_container_width=True)
            if s.design_kind == "factorial":
                st.info(
                    "**Tip:** If you selected terms from aliased pairs, run the "
                    "**fold-over design** (Stage 5) to de-alias them before building "
                    "the final model."
                )
            else:
                st.info(
                    "**Tip:** For mixture designs, run the **D-optimal augmentation** "
                    "(Stage 5) to add runs that help separate aliased terms."
                )
    else:
        st.success("✅ No significant aliasing detected (all |correlations| < 0.75).")

    # ── INTERACTIVE FACTOR SELECTION ─────────────────────────────────────────
    st.subheader("🎛️ Interactive Factor Selection")
    st.markdown(
        "Review the full model results below and **check the terms you want to keep** "
        "in the final fitted model. The app will update the reduced-model R² live."
    )

    coeff_df_full = results["coeff_df"].copy()
    # Default selection: p-value < 0.05 OR term is always needed (Intercept / main effects)
    if "Include in Model" not in coeff_df_full.columns:
        coeff_df_full["Include in Model"] = coeff_df_full["Significant (α=0.05)"].fillna(False)
        # Always include intercept for factorial designs
        if s.design_kind == "factorial":
            coeff_df_full.loc[coeff_df_full["Term"] == "Intercept", "Include in Model"] = True

    # Pretty display for the editable table
    display_for_edit = coeff_df_full[["Term", "Coefficient", "Std Error", "t-Stat",
                                      "p-Value", "LogWorth", "Significant (α=0.05)",
                                      "Include in Model"]].copy()
    display_for_edit["Coefficient"] = display_for_edit["Coefficient"].round(4)
    display_for_edit["Std Error"]   = display_for_edit["Std Error"].round(4)
    display_for_edit["t-Stat"]      = display_for_edit["t-Stat"].round(3)
    display_for_edit["p-Value"]     = display_for_edit["p-Value"].round(5)
    display_for_edit["LogWorth"]    = display_for_edit["LogWorth"].round(2)

    edited_coeff = st.data_editor(
        display_for_edit,
        column_config={
            "Include in Model": st.column_config.CheckboxColumn(
                "✅ Include", help="Check to include this term in the final model",
            ),
            "Term":          st.column_config.TextColumn("Term",      disabled=True),
            "Coefficient":   st.column_config.NumberColumn("Coeff",   disabled=True, format="%.4f"),
            "Std Error":     st.column_config.NumberColumn("SE",      disabled=True, format="%.4f"),
            "t-Stat":        st.column_config.NumberColumn("t",       disabled=True, format="%.3f"),
            "p-Value":       st.column_config.NumberColumn("p",       disabled=True, format="%.5f"),
            "LogWorth":      st.column_config.NumberColumn("LogWorth",disabled=True, format="%.2f"),
            "Significant (α=0.05)": st.column_config.CheckboxColumn("Sig?", disabled=True),
        },
        hide_index=True, use_container_width=True, num_rows="fixed",
    )

    # Get selected term indices (map back to full model matrix columns)
    selected_term_names = edited_coeff.loc[edited_coeff["Include in Model"] == True, "Term"].tolist()
    all_term_names = results["term_names"]
    selected_indices = [i for i, t in enumerate(all_term_names) if t in selected_term_names]

    if selected_indices:
        s.selected_term_indices = selected_indices
        # Live preview of reduced model
        red = build_reduced_model(design, responses, selected_indices,
                                  s.model_type, s.design_kind, s.factor_names)
        if "error" not in red:
            st.markdown("#### 📉 Reduced Model Preview")
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Reduced R²", f"{red['r2']:.4f}",
                       delta=f"{red['r2']-results['r2']:+.4f} vs full")
            pm2.metric("Reduced Adj R²", f"{red['adj_r2']:.4f}")
            pm3.metric("Reduced RMSE", f"{red['rmse']:.4f}")
        else:
            st.warning(f"Reduced model: {red['error']}")
    else:
        st.warning("⚠️ No terms selected — please select at least one term.")
        s.selected_term_indices = None

    # ── Residuals for current full model ──────────────────────────────────────
    with st.expander("📉 Model Diagnostics (full model)"):
        st.plotly_chart(residual_plots(responses, results["y_pred"]),
                        use_container_width=True)

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("---")
    if not aliasing_df.empty:
        st.markdown(
            "**🔄 Aliasing detected.** You can either:\n"
            "- Proceed to **Stage 5** to run fold-over / augmentation experiments, OR\n"
            "- Skip straight to **Stage 6** if you're satisfied with the current analysis."
        )

    col_back, col_fo, col_final = st.columns(3)
    with col_back:
        if st.button("◀ Back to Data Entry"):
            s.workflow_stage = 3; st.rerun()
    with col_fo:
        if st.button("🔄 Go to Fold-Over / Augment (Stage 5)",
                     help="Run additional experiments to de-alias selected terms"):
            s.workflow_stage = 5; st.rerun()
    with col_final:
        if st.button("🏆 Skip to Final Model (Stage 6)", type="primary",
                     disabled=(s.selected_term_indices is None)):
            s.workflow_stage = 6; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: FOLD-OVER / AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

elif s.workflow_stage == 5:
    st.header("🔄 Stage 5 — Fold-Over / Augmentation for De-aliasing")

    if s.design_kind == "factorial":
        st.markdown(
            "### Fractional Fold-Over\n"
            "A **fold-over** design negates factor signs so that aliased "
            "main effects and interactions become separated in the combined analysis.\n\n"
            "**Full fold-over** (negate ALL factors): lifts resolution by 1 — "
            "all main effects become clear of 2FI.\n\n"
            "**Partial fold-over** (negate SELECTED factors): de-aliases only the "
            "targeted factors more economically."
        )

        fold_option = st.radio("Fold-over strategy",
                               ["Full fold-over (all factors)", "Partial fold-over (select factors)"])

        fold_cols = "all"
        if fold_option.startswith("Partial"):
            selected_fold = st.multiselect(
                "Select factors to fold (negate)",
                s.factor_names,
                default=s.factor_names[:1],
            )
            if selected_fold:
                fold_cols = [s.factor_names.index(f) for f in selected_fold]
            else:
                st.warning("Select at least one factor to fold.")
                fold_cols = "all"

        s.fold_columns = fold_cols

        if st.button("⚙️ Generate Fold-Over Design"):
            fo_coded = generate_foldover_factorial(s.initial_design_coded, fold_cols)
            lows  = np.array(s.factor_lows)
            highs = np.array(s.factor_highs)
            fo_natural = lows + (fo_coded + 1) / 2 * (highs - lows)
            s.foldover_design_coded   = fo_coded
            s.foldover_design_natural = fo_natural
            st.success(f"✅ Fold-over design generated: {len(fo_coded)} additional runs.")

    else:  # mixture
        st.markdown(
            "### D-Optimal Augmentation\n"
            "For mixture designs the classical fold-over does not apply (proportions must sum to 1). "
            "Instead we add **D-optimal augmentation runs** chosen to maximise information "
            "in the combined design and best separate any confounded mixture terms."
        )
        n_aug = st.number_input(
            "Number of augmentation runs", min_value=3,
            max_value=s.n_initial_runs, value=max(3, s.n_initial_runs // 2), step=1,
        )
        if st.button("⚙️ Generate Augmentation Design"):
            with st.spinner("Computing D-optimal augmentation..."):
                aug = generate_augmentation_mixture(s.initial_design_coded,
                                                    int(n_aug), s.model_type)
            s.foldover_design_coded   = aug
            s.foldover_design_natural = aug
            st.success(f"✅ {len(aug)} augmentation runs generated.")

    # ── Show new runs ─────────────────────────────────────────────────────────
    if s.foldover_design_natural is not None:
        fo_df = pd.DataFrame(s.foldover_design_natural, columns=s.factor_names)
        fo_df.insert(0, "Run", range(s.n_initial_runs + 1,
                                     s.n_initial_runs + len(fo_df) + 1))
        fo_df.insert(1, s.response_name, [None] * len(fo_df))

        if s.foldover_responses is not None:
            fo_df[s.response_name] = s.foldover_responses

        st.subheader(f"📋 Additional Experimental Runs ({len(fo_df)} new runs)")
        st.dataframe(fo_df.style.format(
            {c: "{:.4f}" for c in s.factor_names}
        ), use_container_width=True)

        buf_fo = io.BytesIO()
        with pd.ExcelWriter(buf_fo, engine="openpyxl") as writer:
            fo_df.to_excel(writer, sheet_name="Fold-Over Runs", index=False)
        dl_fo1, dl_fo2 = st.columns(2)
        dl_fo1.download_button(
            "📥 Download New Runs (Excel)", buf_fo.getvalue(),
            file_name="foldover_augmentation_runs.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        dl_fo2.download_button(
            "📥 Download New Runs (CSV)", fo_df.to_csv(index=False, float_format="%.6f"),
            file_name="foldover_augmentation_runs.csv", mime="text/csv",
        )

        # ── Enter fold-over / augmentation responses ──────────────────────────
        st.subheader("✏️ Enter Responses for New Runs")

        tab_fo_manual, tab_fo_upload = st.tabs(["✏️ Manual Entry", "📁 Upload File"])

        with tab_fo_manual:
            if s.foldover_responses is not None:
                fo_df[s.response_name] = s.foldover_responses

            st.write("**Fill in the Response column (click a cell to edit):**")
            edited_fo = st.data_editor(
                fo_df,
                column_config={
                    s.response_name: st.column_config.NumberColumn(
                        s.response_name, help="Enter measured response value", required=True
                    ),
                    "Run": st.column_config.NumberColumn("Run", disabled=True),
                    **{c: st.column_config.NumberColumn(c, disabled=True) for c in s.factor_names},
                },
                hide_index=True, use_container_width=True,
            )

            if st.button("💾 Save Manual Responses", type="primary", key="save_fo_manual"):
                fo_resp = edited_fo[s.response_name].values.astype(float)
                if np.any(np.isnan(fo_resp)):
                    st.error("❌ Some responses are still empty. Please complete all rows.")
                else:
                    s.foldover_responses = fo_resp
                    s.combined_design = np.vstack([s.initial_design_coded,
                                                   s.foldover_design_coded])
                    s.combined_responses = np.concatenate([s.initial_responses, fo_resp])
                    st.success(
                        f"✅ Responses saved! Combined dataset: "
                        f"{len(s.combined_responses)} runs total."
                    )

        with tab_fo_upload:
            st.info(
                "Upload a CSV or Excel file that contains the fold-over / augmentation runs "
                "with a **response column** filled in. The number of rows must match the "
                f"number of new runs (**{len(fo_df)}**)."
            )
            fo_uploaded = st.file_uploader(
                "Upload completed fold-over / augmentation file (CSV or Excel)",
                type=["csv", "xlsx", "xls"],
                key="fo_upload_widget",
            )
            if fo_uploaded is not None:
                try:
                    if fo_uploaded.name.endswith(".csv"):
                        df_fo_up = pd.read_csv(fo_uploaded)
                    else:
                        df_fo_up = pd.read_excel(fo_uploaded)
                    st.write("Preview:", df_fo_up.head())
                    fo_resp_col = st.selectbox(
                        "Response column", df_fo_up.columns.tolist(), key="fo_resp_col"
                    )
                    if st.button("💾 Load Responses from File", key="save_fo_upload"):
                        fo_resp = df_fo_up[fo_resp_col].values.astype(float)
                        n_expected = len(fo_df)
                        if len(fo_resp) != n_expected:
                            st.error(
                                f"❌ Expected {n_expected} rows "
                                f"(one per new run), got {len(fo_resp)}."
                            )
                        elif np.any(np.isnan(fo_resp)):
                            st.error("❌ Some response values are NaN.")
                        else:
                            s.foldover_responses = fo_resp
                            s.combined_design = np.vstack([s.initial_design_coded,
                                                           s.foldover_design_coded])
                            s.combined_responses = np.concatenate(
                                [s.initial_responses, fo_resp]
                            )
                            st.success(
                                f"✅ Responses loaded from file! "
                                f"Combined dataset: {len(s.combined_responses)} runs total."
                            )
                except Exception as exc:
                    st.error(f"Error reading file: {exc}")

        # Status banner
        if s.foldover_responses is not None:
            st.success(
                f"✅ {len(s.foldover_responses)} fold-over / augmentation responses ready.  "
                f"Range: [{s.foldover_responses.min():.3f}, {s.foldover_responses.max():.3f}]"
            )

    # ── Navigation ───────────────────────────────────────────────────────────
    st.markdown("---")
    col_back, col_re_analyze, col_final = st.columns(3)
    with col_back:
        if st.button("◀ Back to Analysis"):
            s.workflow_stage = 4; st.rerun()
    with col_re_analyze:
        if st.button("🔁 Re-Analyze Combined Data (Stage 4)",
                     disabled=(s.foldover_responses is None),
                     help="Return to Stage 4 to re-run analysis with the combined design"):
            s.workflow_stage = 4; st.rerun()
    with col_final:
        if st.button("🏆 Build Final Model (Stage 6)", type="primary",
                     disabled=(s.selected_term_indices is None)):
            s.workflow_stage = 6; st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 6: FINAL COMPETITIVE MODEL
# ─────────────────────────────────────────────────────────────────────────────

elif s.workflow_stage == 6:
    st.header("🏆 Stage 6 — Final Competitive Model")

    # Use combined data if available, otherwise initial only
    if s.combined_design is not None and s.combined_responses is not None:
        design    = s.combined_design
        responses = s.combined_responses
        data_source = f"Combined data ({len(responses)} runs: {s.n_initial_runs} initial + {len(responses)-s.n_initial_runs} fold-over/augment)"
    else:
        design    = s.initial_design_coded
        responses = s.initial_responses
        data_source = f"Initial screening data ({len(responses)} runs)"

    st.info(f"📊 Building final model from: **{data_source}**")

    # ── Fit full model to get all terms / stats ────────────────────────────
    full_results = fit_model_statistics(design, responses, s.model_type,
                                        s.design_kind, s.factor_names)
    if "error" in full_results:
        st.error(f"❌ {full_results['error']}")
        if st.button("◀ Back to Analysis"):
            s.workflow_stage = 4; st.rerun()
        st.stop()

    all_term_names = full_results["term_names"]
    n_all = len(all_term_names)

    # ── Interactive Factor Selection ───────────────────────────────────────
    st.subheader("🎛️ Interactive Factor Selection")
    st.markdown(
        "Check/uncheck terms below — the final model is **rebuilt instantly** "
        "with every change. Start from the Stage 4 selection or adjust freely."
    )

    # Pre-populate checkboxes from Stage 4 selection (or all if none)
    prev_sel = s.selected_term_indices or list(range(n_all))
    sel_flags = [i in prev_sel for i in range(n_all)]

    edit_rows = pd.DataFrame({
        "Term":           all_term_names,
        "Coefficient":    [round(c, 4) for c in full_results["coefficients"]],
        "p-Value":        [round(p, 5) for p in full_results["p_values"]],
        "Significant":    full_results["p_values"] < 0.05,
        "Include":        sel_flags,
    })
    edited6 = st.data_editor(
        edit_rows,
        column_config={
            "Include":      st.column_config.CheckboxColumn("✅ Include"),
            "Term":         st.column_config.TextColumn("Term",        disabled=True),
            "Coefficient":  st.column_config.NumberColumn("Coeff",     disabled=True, format="%.4f"),
            "p-Value":      st.column_config.NumberColumn("p-Value",   disabled=True, format="%.5f"),
            "Significant":  st.column_config.CheckboxColumn("Sig?",    disabled=True),
        },
        hide_index=True, use_container_width=True, num_rows="fixed",
    )

    selected_indices = [i for i, row in edited6.iterrows() if row["Include"]]

    if not selected_indices:
        st.error("❌ No terms selected — check at least one term to build the model.")
        if st.button("◀ Back to Analysis"):
            s.workflow_stage = 4; st.rerun()
        st.stop()

    # Persist updated selection
    s.selected_term_indices = selected_indices

    # ── Build final reduced model (live) ──────────────────────────────────
    final = build_reduced_model(design, responses, selected_indices,
                                s.model_type, s.design_kind, s.factor_names)
    s.final_results = final

    if "error" in final:
        st.error(f"❌ {final['error']}")
        if st.button("◀ Back to Stage 4"):
            s.workflow_stage = 4; st.rerun()
        st.stop()

    # ── Key quality metrics ────────────────────────────────────────────────
    st.subheader("📈 Model Quality Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R²",          f"{final['r2']:.4f}")
    m2.metric("Adjusted R²", f"{final['adj_r2']:.4f}")
    m3.metric("RMSE",        f"{final['rmse']:.4f}")
    m4.metric("F p-value",   "<0.001" if final["f_p"] < 0.001 else f"{final['f_p']:.4f}")
    m5.metric("Runs used",   str(final["n_obs"]))

    # R² quality badge
    if final["r2"] >= 0.95:
        st.success(f"✅ **Excellent model fit** (R² = {final['r2']:.4f})")
    elif final["r2"] >= 0.85:
        st.info(f"ℹ️ **Good model fit** (R² = {final['r2']:.4f})")
    elif final["r2"] >= 0.70:
        st.warning(f"⚠️ **Moderate model fit** (R² = {final['r2']:.4f}) — consider adding more runs.")
    else:
        st.error(f"❌ **Poor model fit** (R² = {final['r2']:.4f}) — model may need revision.")

    # ── Coefficients table ─────────────────────────────────────────────────
    st.subheader("🔢 Final Model Coefficients")
    styled_coeff = _coeff_df_styled(final["coeff_df"])
    st.dataframe(styled_coeff, use_container_width=True)

    # ── ANOVA table ────────────────────────────────────────────────────────
    with st.expander("📋 ANOVA Table (Final Model)", expanded=True):
        st.dataframe(final["anova_df"].style.format({
            "SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-Value": "{:.4f}",
        }, na_rep=""), use_container_width=True)

    # ── Model equation ─────────────────────────────────────────────────────
    st.subheader("📐 Model Equation")
    terms_eq = final["term_names"]
    coeffs_eq = final["coefficients"]

    parts = []
    for t, c in zip(terms_eq, coeffs_eq):
        sign = "+" if c >= 0 else "−"
        if isinstance(c, float) and abs(c) >= 1e-6:
            parts.append(f"{sign} {abs(c):.4f} · {t}")

    if parts:
        first = parts[0].lstrip("+ ").replace("−", "−")
        equation = f"**{s.response_name}** = {first}"
        if len(parts) > 1:
            equation += "  \n  ".join([""] + parts[1:])
        st.markdown(equation)
    else:
        st.warning("All coefficients are zero or near-zero.")

    # ── Shapiro-Wilk normality test ────────────────────────────────────────
    st.subheader("🔬 Residuals Normality Test")
    residuals = final["residuals"]
    if len(residuals) >= 3:
        stat_sw, p_sw = scipy_stats.shapiro(residuals)
        c1, c2, c3 = st.columns(3)
        c1.metric("Shapiro-Wilk W", f"{stat_sw:.4f}")
        c2.metric("p-Value",        f"{p_sw:.4f}")
        c3.metric("Residuals Normal?",
                  "✅ Yes" if p_sw > 0.05 else "⚠️ Marginal" if p_sw > 0.01 else "❌ No")

    # ── Lack-of-fit test (if replicates exist) ─────────────────────────────
    st.subheader("📊 Lack-of-Fit Test")
    y_pred = final["y_pred"]
    design_rows = [tuple(np.round(r, 6)) for r in design]
    unique_pts, inv_idx = np.unique(design_rows, axis=0, return_inverse=True)
    n_unique = len(unique_pts)
    n_total  = len(responses)
    n_rep = n_total - n_unique

    if n_rep >= 1:
        # Pure error SS
        ss_pe = 0.0
        for ui in range(n_unique):
            mask = inv_idx == ui
            y_grp = responses[mask]
            ss_pe += np.sum((y_grp - np.mean(y_grp)) ** 2)
        ss_res = np.sum(residuals ** 2)
        ss_lof = max(0.0, ss_res - ss_pe)
        df_lof = max(0, n_unique - final["n_params"])
        df_pe  = n_rep

        if df_lof > 0 and df_pe > 0:
            ms_lof = ss_lof / df_lof
            ms_pe  = ss_pe  / df_pe
            f_lof  = ms_lof / ms_pe if ms_pe > 0 else 0.0
            p_lof  = 1.0 - scipy_stats.f.cdf(f_lof, df_lof, df_pe)

            lof_df = pd.DataFrame({
                "Source": ["Lack of Fit", "Pure Error", "Total Residual"],
                "DF":     [df_lof, df_pe, df_lof + df_pe],
                "SS":     [ss_lof, ss_pe, ss_res],
                "MS":     [ms_lof, ms_pe, np.nan],
                "F":      [f_lof,  np.nan, np.nan],
                "p-Value":[p_lof,  np.nan, np.nan],
            })
            st.dataframe(lof_df.style.format({
                "SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-Value": "{:.4f}",
            }, na_rep=""), use_container_width=True)

            if p_lof > 0.10:
                st.success(f"✅ Lack-of-Fit p = {p_lof:.4f} — **no significant LOF** (model fits well).")
            elif p_lof > 0.05:
                st.warning(f"⚠️ Lack-of-Fit p = {p_lof:.4f} — marginal LOF.")
            else:
                st.error(f"❌ Lack-of-Fit p = {p_lof:.4f} — **significant LOF** — model structure may be inadequate.")
        else:
            st.info("ℹ️ LOF test requires replicated runs and sufficient degrees of freedom.")
    else:
        st.info("ℹ️ No replicated runs — Lack-of-Fit test not applicable.")

    # ── Full diagnostic plots ──────────────────────────────────────────────
    st.subheader("📉 Full Residual Diagnostics")
    st.plotly_chart(residual_plots(responses, y_pred), use_container_width=True)

    # ── Response surface (mixture: ternary contour) ─────────────────────────
    if s.design_kind == "mixture" and s.n_factors == 3:
        st.subheader("🌈 Response Surface — Ternary Contour")
        grid_n = 40
        a = np.linspace(0, 1, grid_n)
        b = np.linspace(0, 1, grid_n)
        A, B = np.meshgrid(a, b)
        C = 1.0 - A - B
        mask = (C >= 0) & (C <= 1)
        pts = np.column_stack([A[mask], B[mask], C[mask]])
        X_grid, _ = _build_model_matrix(pts, s.model_type, "mixture")
        X_reduced_g = X_grid[:, selected_indices]
        y_grid = X_reduced_g @ final["coefficients"]

        fig_tern = go.Figure(go.Contour(
            x=pts[:, 0], y=pts[:, 1], z=y_grid,
            colorscale="Viridis",
            contours=dict(showlabels=True),
        ))
        fig_tern.update_layout(
            title=f"{s.response_name} Response Surface (x1 vs x2, x3=1-x1-x2)",
            xaxis_title=s.factor_names[0], yaxis_title=s.factor_names[1],
            height=450,
        )
        st.plotly_chart(fig_tern, use_container_width=True)

    elif s.design_kind == "factorial" and s.n_factors >= 2:
        st.subheader("🌈 Response Surface (first two factors, others at centre)")
        grid_n = 30
        x1g = np.linspace(-1, 1, grid_n)
        x2g = np.linspace(-1, 1, grid_n)
        X1, X2 = np.meshgrid(x1g, x2g)
        flat = np.column_stack([X1.ravel(), X2.ravel()])
        if s.n_factors > 2:
            extra = np.zeros((len(flat), s.n_factors - 2))
            flat = np.column_stack([flat, extra])
        X_grid, _ = _build_model_matrix(flat, s.model_type, "factorial")
        X_red_g = X_grid[:, selected_indices]
        y_grid = (X_red_g @ final["coefficients"]).reshape(grid_n, grid_n)

        # Natural units for axis
        lo1, hi1 = s.factor_lows[0], s.factor_highs[0]
        lo2, hi2 = s.factor_lows[1], s.factor_highs[1]
        xnat1 = lo1 + (x1g + 1) / 2 * (hi1 - lo1)
        xnat2 = lo2 + (x2g + 1) / 2 * (hi2 - lo2)
        X1n, X2n = np.meshgrid(xnat1, xnat2)

        fig_rs = go.Figure(go.Surface(
            x=X1n, y=X2n, z=y_grid,
            colorscale="Viridis", opacity=0.85,
        ))
        fig_rs.update_layout(
            title=f"{s.response_name} Response Surface ({s.factor_names[0]} × {s.factor_names[1]})",
            scene=dict(
                xaxis_title=s.factor_names[0],
                yaxis_title=s.factor_names[1],
                zaxis_title=s.response_name,
            ),
            height=480,
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    # ── Export ─────────────────────────────────────────────────────────────
    st.subheader("📥 Download Final Results")

    # Build predictions dataframe (shared between downloads)
    _design_nat = s.initial_design_natural
    if s.combined_design is not None and s.foldover_design_natural is not None:
        _design_nat = np.vstack([s.initial_design_natural,
                                 s.foldover_design_natural])[:len(responses)]
    pred_df = pd.DataFrame(_design_nat, columns=s.factor_names)
    pred_df[s.response_name] = responses
    pred_df["Predicted"]     = y_pred
    pred_df["Residual"]      = final["residuals"]
    pred_df["Std Residual"]  = final["residuals"] / (final["rmse"] + 1e-12)

    # ── Primary: full Excel workbook (Coefficients + ANOVA + Predictions) ──
    buf_report = io.BytesIO()
    with pd.ExcelWriter(buf_report, engine="openpyxl") as writer:
        final["coeff_df"].to_excel(writer, sheet_name="Coefficients", index=False)
        final["anova_df"].to_excel(writer, sheet_name="ANOVA", index=False)
        pred_df.to_excel(writer, sheet_name="Predictions", index=False)
    st.download_button(
        "📂 Download Full Report (Excel — all sheets)",
        buf_report.getvalue(),
        file_name="final_model_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown("**Individual sheet downloads:**")
    ex1, ex2, ex3, ex4 = st.columns(4)

    with ex1:
        buf_c = io.BytesIO()
        with pd.ExcelWriter(buf_c, engine="openpyxl") as w:
            final["coeff_df"].to_excel(w, sheet_name="Coefficients", index=False)
        ex1.download_button("📊 Coefficients (.xlsx)", buf_c.getvalue(),
                            file_name="coefficients.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with ex2:
        buf_p = io.BytesIO()
        with pd.ExcelWriter(buf_p, engine="openpyxl") as w:
            pred_df.to_excel(w, sheet_name="Predictions", index=False)
        ex2.download_button("📉 Predictions (.xlsx)", buf_p.getvalue(),
                            file_name="predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with ex3:
        ex3.download_button("📊 Coefficients (.csv)",
                            final["coeff_df"].to_csv(index=False, float_format="%.6f"),
                            file_name="coefficients.csv", mime="text/csv")

    with ex4:
        ex4.download_button("📉 Predictions (.csv)",
                            pred_df.to_csv(index=False, float_format="%.6f"),
                            file_name="predictions.csv", mime="text/csv")

    # ── Summary banner ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("✅ Workflow Complete")
    n_sig = int((final["coeff_df"]["p-Value"].apply(
        lambda x: float(x.replace("<", "")) if isinstance(x, str) else x
    ) < 0.05).sum()) if "p-Value" in final["coeff_df"].columns else 0

    st.success(
        f"🎉 **Final competitive model built successfully!**\n\n"
        f"- Data used: {data_source}\n"
        f"- Selected terms: {final['n_params']}\n"
        f"- Model R² = **{final['r2']:.4f}**, Adj R² = **{final['adj_r2']:.4f}**\n"
        f"- RMSE = {final['rmse']:.4f}\n"
        f"- Fold-over / augmentation: {'✅ performed' if s.foldover_responses is not None else '➖ not needed'}"
    )

    col_back, col_restart = st.columns(2)
    with col_back:
        if st.button("◀ Revise Factor Selection (Stage 4)"):
            s.workflow_stage = 4; st.rerun()
    with col_restart:
        if st.button("🔁 Start New Experiment"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
