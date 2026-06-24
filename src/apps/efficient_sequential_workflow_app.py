"""
Efficient Sequential Mixture Workflow — Streamlit App
======================================================
Rebuilt sequential_workflow using Smart Simplex Centroid point-generation
and replication algorithms from streamlit_app.py, with an adaptive 3-phase
strategy that minimises total experiments.

Key improvements over doe_sequential_workflow_app.py
------------------------------------------------------
  1. Point generation: Smart Simplex Centroid (extracted from streamlit_app.py)
     - Structured centroid blends (vertices / binary / ternary / quaternary / overall)
     - JMP-validated replication counts per blend order
  2. Adaptive staging: stops requesting new runs once R² target is reached
     - Phase 1: vertices + binary blends + centroid  (~21 runs for q=5)
     - Phase 2: ternary blends (guided by Phase 1 significance) + centroid replication
     - Phase 3: quaternary blends (guided by Phase 2 significance)
  3. Efficiency: typically 30–53% fewer runs than the JMP 45-run fixed design

Reference: Smart Simplex Centroid algorithm  (smart_simplex_centroid.py)
           DOE Sequential Workflow UI         (doe_sequential_workflow_app.py)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from itertools import combinations
from math import comb
from typing import Dict, List, Optional, Tuple
import sys, os, io

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Efficient Sequential Mixture DOE",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# ── CORE ALGORITHMS  (extracted from streamlit_app.py + smart_simplex_centroid)
# ═══════════════════════════════════════════════════════════════════════════════

def _centroid_point(q: int, active_indices: Tuple) -> np.ndarray:
    """Equal-proportion blend for active component indices; rest = 0."""
    pt = np.zeros(q)
    for i in active_indices:
        pt[i] = 1.0 / len(active_indices)
    return pt


def generate_points_for_order(q: int, order: int) -> List[np.ndarray]:
    """All centroid blends of a given order for q components."""
    return [_centroid_point(q, combo) for combo in combinations(range(q), order)]


def smart_default_replicates(q: int) -> Dict[int, int]:
    """
    JMP-validated replication counts per blend order
    (extracted from smart_simplex_centroid.py).
    q=5 → {1:2, 2:1, 3:1, 4:1, 5:3}
    """
    k_ref = 2
    max_pts = comb(q, min(k_ref, q))
    reps: Dict[int, int] = {}
    for k in range(1, q + 1):
        if k == 1:
            reps[k] = 2
        elif k == q:
            reps[k] = 3
        elif k >= q - 1:
            reps[k] = 1
        else:
            n_pts = comb(q, k)
            ratio = max_pts / n_pts
            reps[k] = max(1, min(3, round(ratio)))
    return reps


def jmp_augmented_run_count(q: int) -> int:
    """Total runs including JMP power-based augmentation."""
    reps = smart_default_replicates(q)
    total = 0
    for k in range(1, q + 1):
        n_blends = comb(q, k)
        base_runs = n_blends * reps[k]
        if k >= 2 and k < q - 1:
            aug_rate = (q - k - 1) / (q - 1)
            n_extra = round(n_blends * aug_rate) if aug_rate > 0 else 0
            n_extra = max(1, n_extra) if n_extra > 0 else 0
        else:
            n_extra = 0
        total += base_runs + n_extra
    return total


def replicate_points(points: List[np.ndarray], n_reps: int) -> List[np.ndarray]:
    """Replicate a list of design points n_reps times."""
    return [pt.copy() for pt in points for _ in range(n_reps)]


def build_phase1_design(q: int, component_names: List[str]) -> pd.DataFrame:
    """
    Phase 1 design: vertices × 2 + all binary blends × 1 + centroid × 1.
    Returns a DataFrame ready for download with an empty Response column.
    """
    reps = smart_default_replicates(q)
    pts: List[np.ndarray] = []
    labels: List[str] = []
    orders: List[int] = []

    # Vertices × reps[1]
    for combo in combinations(range(q), 1):
        pt = _centroid_point(q, combo)
        name = component_names[combo[0]]
        for _ in range(reps[1]):
            pts.append(pt.copy())
            labels.append(f"Vertex {name}")
            orders.append(1)

    # Binary blends × reps[2]
    for combo in combinations(range(q), 2):
        pt = _centroid_point(q, combo)
        name = "+".join(component_names[i] for i in combo)
        for _ in range(max(reps[2], 1)):
            pts.append(pt.copy())
            labels.append(f"Binary {name}")
            orders.append(2)

    # Overall centroid × 1 (will be replicated in phase 2)
    pt_c = _centroid_point(q, tuple(range(q)))
    pts.append(pt_c.copy())
    labels.append("Centroid (all)")
    orders.append(q)

    df = pd.DataFrame(pts, columns=component_names)
    df.insert(0, "Run", range(1, len(df) + 1))
    df.insert(1, "Blend_Type", labels)
    df.insert(2, "Blend_Order", orders)
    df["Response"] = np.nan
    return df


def build_phase2_design(q: int, component_names: List[str],
                        sig_pair_indices: List[int],
                        existing_run_count: int) -> pd.DataFrame:
    """
    Phase 2 design: ternary blends (guided by significant pairs) + 2 extra centroid reps.
    """
    reps = smart_default_replicates(q)
    binary_combos = list(combinations(range(q), 2))
    ternary_all = list(combinations(range(q), 3))

    # Select ternary blends covering significant pairs
    if sig_pair_indices:
        sig_set = set(sig_pair_indices)
        ternary_sel = [t for t in ternary_all
                       if any(binary_combos.index((i, j)) in sig_set
                              for i, j in combinations(t, 2)
                              if (i, j) in binary_combos)]
        if not ternary_sel:
            ternary_sel = ternary_all
    else:
        ternary_sel = ternary_all

    pts: List[np.ndarray] = []
    labels: List[str] = []
    orders: List[int] = []

    # Ternary blends × reps[3]
    for combo in ternary_sel:
        pt = _centroid_point(q, combo)
        name = "+".join(component_names[i] for i in combo)
        for _ in range(max(reps[3], 1)):
            pts.append(pt.copy())
            labels.append(f"Ternary {name}")
            orders.append(3)

    # 2 extra centroid replicates (total becomes 3 across phases)
    pt_c = _centroid_point(q, tuple(range(q)))
    for _ in range(2):
        pts.append(pt_c.copy())
        labels.append("Centroid (all) — replicate for LOF")
        orders.append(q)

    df = pd.DataFrame(pts, columns=component_names)
    start_run = existing_run_count + 1
    df.insert(0, "Run", range(start_run, start_run + len(df)))
    df.insert(1, "Blend_Type", labels)
    df.insert(2, "Blend_Order", orders)
    df["Response"] = np.nan
    return df


def build_phase3_design(q: int, component_names: List[str],
                        sig_triple_indices: List[int],
                        sig_binary_for_rep: List[int],
                        existing_run_count: int) -> pd.DataFrame:
    """
    Phase 3 design: quaternary blends (guided by significant triples)
                    + replicate top binary blends + replicate top ternary blends.
    """
    reps = smart_default_replicates(q)
    ternary_combos = list(combinations(range(q), 3))
    quat_all = list(combinations(range(q), 4))
    binary_combos = list(combinations(range(q), 2))

    # Select quaternary blends covering significant triples
    if sig_triple_indices:
        sig_set = set(sig_triple_indices)
        quat_sel = [qb for qb in quat_all
                    if any(ternary_combos.index(triple) in sig_set
                           for triple in combinations(qb, 3)
                           if triple in ternary_combos)]
        if not quat_sel:
            quat_sel = quat_all
    else:
        quat_sel = quat_all

    pts: List[np.ndarray] = []
    labels: List[str] = []
    orders: List[int] = []

    # Quaternary × reps[4]
    for combo in quat_sel:
        pt = _centroid_point(q, combo)
        name = "+".join(component_names[i] for i in combo)
        for _ in range(max(reps[4], 1)):
            pts.append(pt.copy())
            labels.append(f"Quaternary {name}")
            orders.append(4)

    # Replicate top binary blends (pure-error for binary space)
    for bi in sig_binary_for_rep[:2]:
        if bi < len(binary_combos):
            combo = binary_combos[bi]
            pt = _centroid_point(q, combo)
            name = "+".join(component_names[i] for i in combo)
            pts.append(pt.copy())
            labels.append(f"Binary replicate {name}")
            orders.append(2)

    # Replicate top 3 significant ternary blends
    for ti in sig_triple_indices[:3]:
        if ti < len(ternary_combos):
            combo = ternary_combos[ti]
            pt = _centroid_point(q, combo)
            name = "+".join(component_names[i] for i in combo)
            pts.append(pt.copy())
            labels.append(f"Ternary replicate {name}")
            orders.append(3)

    df = pd.DataFrame(pts, columns=component_names)
    start_run = existing_run_count + 1
    df.insert(0, "Run", range(start_run, start_run + len(df)))
    df.insert(1, "Blend_Type", labels)
    df.insert(2, "Blend_Order", orders)
    df["Response"] = np.nan
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ── MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════════════

def build_scheffe_matrix(design: np.ndarray, component_names: List[str],
                         max_order: int = 5) -> Tuple[np.ndarray, List[str]]:
    """Scheffé model matrix (no intercept; mixture constraint)."""
    q = design.shape[1]
    cols, names = [], []
    for i in range(q):
        cols.append(design[:, i])
        names.append(component_names[i])
    for order in range(2, min(max_order, q) + 1):
        for combo in combinations(range(q), order):
            col = np.ones(len(design))
            for idx in combo:
                col *= design[:, idx]
            cols.append(col)
            names.append("·".join(component_names[j] for j in combo))
    return np.column_stack(cols), names


def fit_scheffe_model(design: np.ndarray, responses: np.ndarray,
                      component_names: List[str], max_order: int = 5) -> Dict:
    """Fit OLS Scheffé model; return full statistics dict."""
    X, term_names = build_scheffe_matrix(design, component_names, max_order)

    # If under-determined, reduce order
    while X.shape[1] > X.shape[0] and max_order > 1:
        max_order -= 1
        X, term_names = build_scheffe_matrix(design, component_names, max_order)

    n_obs, n_params = X.shape
    df_res = max(n_obs - n_params, 1)

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, responses, rcond=None)
    except Exception as e:
        return {"error": str(e), "max_order": max_order}

    y_pred = X @ coeffs
    residuals = responses - y_pred
    ss_tot = np.sum((responses - np.mean(responses)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (ss_res / df_res) / (ss_tot / max(n_obs - 1, 1)) if ss_tot > 0 else 0.0
    mse = ss_res / df_res
    rmse = np.sqrt(mse)

    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.abs(mse * np.diag(XtX_inv)))
        t_stats = coeffs / (se + 1e-300)
        p_values = 2.0 * (1.0 - scipy_stats.t.cdf(np.abs(t_stats), df_res))
        t_crit = scipy_stats.t.ppf(0.975, df_res)
        ci_lower = coeffs - t_crit * se
        ci_upper = coeffs + t_crit * se
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)
        t_stats = np.full(n_params, np.nan)
        p_values = np.ones(n_params)
        ci_lower = ci_upper = np.full(n_params, np.nan)

    # Overall F
    df_reg = n_params
    ms_reg = (ss_tot - ss_res) / max(df_reg, 1)
    f_stat = ms_reg / mse if mse > 0 else 0.0
    f_p = 1.0 - scipy_stats.f.cdf(f_stat, df_reg, df_res)

    coeff_df = pd.DataFrame({
        "Term":           term_names,
        "Coefficient":    coeffs,
        "Std Error":      se,
        "t-Stat":         t_stats,
        "p-Value":        p_values,
        "LogWorth":       -np.log10(np.clip(p_values, 1e-300, 1.0)),
        "CI Lower 95%":   ci_lower,
        "CI Upper 95%":   ci_upper,
        "Significant":    p_values < 0.05,
    })

    anova_df = pd.DataFrame({
        "Source":  ["Regression", "Residual", "Total"],
        "DF":      [df_reg, df_res, n_obs - 1],
        "SS":      [ss_tot - ss_res, ss_res, ss_tot],
        "MS":      [ms_reg, mse, np.nan],
        "F":       [f_stat, np.nan, np.nan],
        "p-Value": [f_p, np.nan, np.nan],
    })

    # Identify significant binary term indices (for Phase 2 guidance)
    q_comp = design.shape[1]
    n_binary = comb(q_comp, 2)
    n_ternary = comb(q_comp, 3) if max_order >= 3 else 0
    binary_slice = slice(q_comp, q_comp + n_binary)
    ternary_slice = slice(q_comp + n_binary, q_comp + n_binary + n_ternary)

    sig_binary_idx = [i for i, p in enumerate(p_values[binary_slice]) if p < 0.05]
    sig_ternary_idx = [i for i, p in enumerate(p_values[ternary_slice]) if p < 0.05]

    return {
        "term_names": term_names,
        "coefficients": coeffs,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
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
        "n_obs": n_obs,
        "n_params": n_params,
        "df_res": df_res,
        "max_order": max_order,
        "coeff_df": coeff_df,
        "anova_df": anova_df,
        "sig_binary_idx": sig_binary_idx,
        "sig_ternary_idx": sig_ternary_idx,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def residual_plots_fig(y_actual: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    resid = y_actual - y_pred
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residuals vs Fitted", "Actual vs Predicted"))
    fig.add_trace(go.Scatter(x=y_pred, y=resid, mode="markers",
                             marker=dict(color="steelblue", size=8, opacity=0.7),
                             name="Residuals"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect"), row=1, col=2)
    fig.add_trace(go.Scatter(x=y_actual, y=y_pred, mode="markers",
                             marker=dict(color="steelblue", size=8, opacity=0.7),
                             name="Data"), row=1, col=2)
    fig.update_layout(height=380, showlegend=False)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Actual", row=1, col=2)
    fig.update_yaxes(title_text="Predicted", row=1, col=2)
    return fig


def pareto_fig(coeff_df: pd.DataFrame) -> go.Figure:
    terms = coeff_df.dropna(subset=["t-Stat"]).copy()
    terms["abs_t"] = terms["t-Stat"].abs()
    terms = terms.sort_values("abs_t", ascending=True)
    colors = ["crimson" if p < 0.05 else "steelblue" for p in terms["p-Value"]]
    fig = go.Figure(go.Bar(
        x=terms["abs_t"], y=terms["Term"], orientation="h",
        marker_color=colors,
        text=terms["p-Value"].apply(lambda p: f"p={p:.3f}"),
        textposition="outside",
    ))
    fig.update_layout(title="Pareto Chart of |t-Statistics| (red = p < 0.05)",
                      xaxis_title="|t-Stat|", height=max(300, len(terms) * 28))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ── SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "eff_stage": 1,
        "q": 5,
        "component_names": ["A", "B", "C", "D", "E"],
        "response_name": "Response",
        "r2_target": 0.97,
        # Phase designs (DataFrames with Run/Blend_Type/Blend_Order/components/Response)
        "p1_design": None,
        "p2_design": None,
        "p3_design": None,
        # Accumulated design (numpy arrays) and responses (numpy arrays)
        "design_all": None,
        "responses_all": None,
        # Model fit results per phase
        "fit_p1": None,
        "fit_p2": None,
        "fit_p3": None,
        # Guidance: significant indices from previous phase
        "sig_binary_for_p2": [],
        "sig_ternary_for_p3": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()
s = st.session_state

# ═══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

stage_labels = {
    1: "⚙️ Setup",
    2: "🎯 Phase 1 Design",
    3: "📋 Phase 1 Data Entry",
    4: "📊 Phase 1 Analysis",
    5: "🔬 Phase 2 Design",
    6: "📋 Phase 2 Data Entry",
    7: "📊 Phase 2 Analysis",
    8: "⚗️ Phase 3 Design",
    9: "📋 Phase 3 Data Entry",
    10: "🏆 Final Model",
}

st.sidebar.title("🎯 Efficient Sequential DOE")
st.sidebar.markdown("---")

# Efficiency summary
q = s.q
n_jmp = jmp_augmented_run_count(q)
n_p1 = q * smart_default_replicates(q)[1] + comb(q, 2) + 1
n_p2_est = comb(q, 3) + 2
n_p3_est = comb(q, 4)

st.sidebar.subheader("📊 Experiment Budget")
st.sidebar.markdown(f"""
| Phase | Runs | Cumulative |
|---|---|---|
| Phase 1 (Linear+Binary) | ~{n_p1} | ~{n_p1} |
| Phase 2 (Ternary) | ~{n_p2_est} | ~{n_p1+n_p2_est} |
| Phase 3 (Quaternary) | ~{n_p3_est} | ~{n_p1+n_p2_est+n_p3_est} |
| **JMP Fixed (reference)** | **{n_jmp}** | **{n_jmp}** |
""")
st.sidebar.success(
    f"✅ Save **{n_jmp - n_p1}–{n_jmp - n_p1 - n_p2_est}** runs vs JMP  "
    f"if model is good after Phase 1 or 2"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Workflow Progress")
for idx, label in stage_labels.items():
    if idx < s.eff_stage:
        st.sidebar.markdown(f"✅ {label}")
    elif idx == s.eff_stage:
        st.sidebar.markdown(f"**▶ {label}**")
    else:
        st.sidebar.markdown(f"⬜ {label}")

st.sidebar.markdown("---")
if st.sidebar.button("🔁 Restart"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ── MAIN TITLE
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🎯 Efficient Sequential Mixture Workflow")
st.markdown(
    "**Adaptive 3-phase strategy** using Smart Simplex Centroid point generation.  \n"
    "Stops requesting new experiments as soon as the model meets your quality target — "
    "saving 30–53% of runs vs a fixed JMP design."
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — SETUP
# ═══════════════════════════════════════════════════════════════════════════════

if s.eff_stage == 1:
    st.header("⚙️ Stage 1 — Problem Setup")

    col1, col2 = st.columns(2)
    with col1:
        q_new = st.number_input("Number of mixture components", 2, 8, s.q, step=1)
        response_name = st.text_input("Response variable name", value=s.response_name)
        r2_target = st.slider(
            "Target model R² (stop adding runs when reached)",
            0.80, 0.9999, s.r2_target, step=0.01,
            help="Higher = more runs but better model. 0.97 is a good default."
        )

    with col2:
        st.subheader("Component Names")
        names = []
        for i in range(q_new):
            prev = s.component_names[i] if i < len(s.component_names) else f"C{i+1}"
            names.append(st.text_input(f"Component {i+1}", value=prev, key=f"cname_{i}"))

    # Show expected run counts
    reps = smart_default_replicates(q_new)
    n_p1_new = q_new * reps[1] + comb(q_new, 2) + 1
    n_p2_new = comb(q_new, 3) + 2
    n_p3_new = comb(q_new, 4)
    n_jmp_new = jmp_augmented_run_count(q_new)

    st.subheader("📊 Expected Experiment Budget")
    budget_df = pd.DataFrame({
        "Phase": ["Phase 1 — Vertices + Binary", "Phase 2 — Ternary", "Phase 3 — Quaternary", "JMP Fixed (reference)"],
        "New Runs": [str(n_p1_new), str(n_p2_new), str(n_p3_new), "—"],
        "Cumulative": [str(n_p1_new), str(n_p1_new + n_p2_new), str(n_p1_new + n_p2_new + n_p3_new), str(n_jmp_new)],
        "Stop if R2 >=":  [str(r2_target), str(r2_target), "—", "—"],
    })
    st.dataframe(budget_df, use_container_width=True, hide_index=True)

    st.info(
        f"💡 **Potential savings**: If your model reaches R² ≥ {r2_target:.2f} after Phase 1 "
        f"({n_p1_new} runs), you save **{n_jmp_new - n_p1_new} runs** vs the JMP fixed design. "
        f"Even Phase 2 completion ({n_p1_new+n_p2_new} runs) saves "
        f"**{n_jmp_new - n_p1_new - n_p2_new} runs**."
    )

    # Model explanation
    with st.expander("📐 Scheffé Mixture Model — How it works", expanded=False):
        for order in range(1, q_new + 1):
            n_blends = comb(q_new, order)
            n_reps_o = reps[order]
            label = {1: "pure vertices", 2: "binary blends",
                     3: "ternary blends", 4: "quaternary blends",
                     5: "quinary blend (centroid)"}.get(order, f"order-{order} blends")
            st.write(f"**Order {order}** ({label}): {n_blends} × "
                     f"{n_reps_o} reps = {n_blends * n_reps_o} base runs")
        st.write(f"**JMP augmented total**: {n_jmp_new} runs (with power-based augmentation)")

    if st.button("✅ Confirm Setup → Generate Phase 1 Design", type="primary"):
        s.q = q_new
        s.component_names = names
        s.response_name = response_name
        s.r2_target = r2_target
        # Reset downstream state
        for k in ["p1_design", "p2_design", "p3_design",
                  "design_all", "responses_all",
                  "fit_p1", "fit_p2", "fit_p3",
                  "sig_binary_for_p2", "sig_ternary_for_p3"]:
            s[k] = None
        s["sig_binary_for_p2"] = []
        s["sig_ternary_for_p3"] = []
        s.eff_stage = 2
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — PHASE 1 DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 2:
    st.header("🎯 Stage 2 — Phase 1 Screening Design")

    q = s.q
    reps = smart_default_replicates(q)

    st.markdown(f"""
    **Phase 1 point generation** (extracted from Smart Simplex Centroid):
    - **{q} pure vertices × {reps[1]}** = {q*reps[1]} runs  (pure-error estimation at vertices)
    - **{comb(q,2)} binary blends × 1** = {comb(q,2)} runs  (all 2-way interaction effects)
    - **1 overall centroid × 1** = 1 run    (LOF seed; replicated in Phase 2)
    
    **Total Phase 1: {q*reps[1] + comb(q,2) + 1} runs**  
    (JMP fixed design would use {jmp_augmented_run_count(q)} runs total)
    """)

    if s.p1_design is None:
        s.p1_design = build_phase1_design(q, s.component_names)

    df1 = s.p1_design

    st.subheader("📋 Phase 1 Design Matrix")

    # Annotate point types for mixture design
    st.dataframe(
        df1.style.format({c: "{:.4f}" for c in s.component_names}),
        use_container_width=True
    )

    # Sum check
    sums = df1[s.component_names].sum(axis=1)
    if np.allclose(sums, 1.0, atol=1e-6):
        st.success("✅ All rows sum to 1.0")

    # Ternary plot for q=3
    if q == 3:
        vals = df1[s.component_names].values
        fig = go.Figure(go.Scatterternary(
            a=vals[:, 0], b=vals[:, 1], c=vals[:, 2],
            mode="markers+text",
            text=[f"R{i+1}" for i in range(len(vals))],
            textposition="top center",
            marker=dict(size=12, color="steelblue"),
        ))
        fig.update_layout(
            title="Phase 1 Design — Ternary Plot",
            ternary=dict(sum=1,
                         aaxis=dict(title=s.component_names[0]),
                         baxis=dict(title=s.component_names[1]),
                         caxis=dict(title=s.component_names[2])),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Downloads
    dl1, dl2 = st.columns(2)
    buf_x = io.BytesIO()
    with pd.ExcelWriter(buf_x, engine="openpyxl") as w:
        df1.to_excel(w, sheet_name="Phase1_Design", index=False)
    dl1.download_button("📥 Download Phase 1 (Excel)", buf_x.getvalue(),
                        file_name="phase1_design.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    dl2.download_button("📥 Download Phase 1 (CSV)", df1.to_csv(index=False, float_format="%.6f"),
                        file_name="phase1_design.csv", mime="text/csv")

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Setup"):
            s.eff_stage = 1; st.rerun()
    with col_fwd:
        if st.button("▶ Proceed to Data Entry", type="primary"):
            s.eff_stage = 3; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — PHASE 1 DATA ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 3:
    st.header("📋 Stage 3 — Phase 1: Enter Experimental Responses")
    st.info("Run experiments from Phase 1. Enter measured response values below or upload a file.")

    tab_manual, tab_upload = st.tabs(["✏️ Manual Entry", "📁 Upload File"])

    df1 = s.p1_design.copy()
    if s.responses_all is not None and len(s.responses_all) == len(df1):
        df1[s.response_name] = s.responses_all[:len(df1)]
    else:
        df1[s.response_name] = df1["Response"]

    with tab_manual:
        edited = st.data_editor(
            df1,
            column_config={
                s.response_name: st.column_config.NumberColumn(s.response_name, required=True),
                **{c: st.column_config.NumberColumn(c, disabled=True) for c in s.component_names},
                "Run":        st.column_config.NumberColumn("Run",         disabled=True),
                "Blend_Type": st.column_config.TextColumn("Blend Type",    disabled=True),
                "Blend_Order":st.column_config.NumberColumn("Order",       disabled=True),
                "Response":   st.column_config.NumberColumn("Response",    disabled=True),
            },
            hide_index=True, use_container_width=True,
        )
        if st.button("💾 Save Phase 1 Responses", type="primary"):
            resp_col = s.response_name if s.response_name in edited.columns else "Response"
            resp = edited[resp_col].values.astype(float)
            if np.any(np.isnan(resp)):
                st.error("❌ Some responses are empty. Complete all rows.")
            else:
                s.design_all = df1[s.component_names].values
                s.responses_all = resp
                s.p1_design[s.response_name] = resp
                st.success("✅ Phase 1 responses saved!")

    with tab_upload:
        up = st.file_uploader("Upload completed Phase 1 file (CSV / Excel)", type=["csv", "xlsx"])
        if up:
            try:
                df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
                st.dataframe(df_up.head())
                resp_col = st.selectbox("Response column", df_up.columns.tolist())
                if st.button("💾 Load Phase 1 from File"):
                    resp = df_up[resp_col].values[:len(df1)].astype(float)
                    if np.any(np.isnan(resp)):
                        st.error("Some NaN responses.")
                    else:
                        s.design_all = df1[s.component_names].values
                        s.responses_all = resp
                        s.p1_design[s.response_name] = resp
                        st.success("✅ Loaded!")
            except Exception as e:
                st.error(str(e))

    if s.responses_all is not None:
        st.success(f"✅ {len(s.responses_all)} Phase 1 responses ready.")

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Design"):
            s.eff_stage = 2; st.rerun()
    with col_fwd:
        if st.button("▶ Analyse Phase 1", type="primary",
                     disabled=(s.responses_all is None or
                               len(s.responses_all) != len(s.p1_design))):
            s.eff_stage = 4; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — PHASE 1 ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 4:
    st.header("📊 Stage 4 — Phase 1 Analysis")

    design = s.design_all
    responses = s.responses_all

    fit = fit_scheffe_model(design, responses, s.component_names, max_order=2)

    if "error" in fit:
        st.error(f"❌ {fit['error']}")
    else:
        s.fit_p1 = fit
        s["sig_binary_for_p2"] = fit["sig_binary_idx"]

        # ── Key metrics ──────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Runs (Phase 1)", len(responses))
        c2.metric("R²", f"{fit['r2']:.4f}")
        c3.metric("Adj-R²", f"{fit['adj_r2']:.4f}")
        c4.metric("RMSE", f"{fit['rmse']:.4f}")
        c5.metric("Target R²", f"{s.r2_target:.2f}")

        # Quality banner
        if fit["r2"] >= s.r2_target:
            st.success(
                f"🎉 **R² = {fit['r2']:.4f} ≥ target {s.r2_target:.2f}** — "
                f"Model quality target met after Phase 1!  "
                f"You can **skip directly to Final Model** and save "
                f"{jmp_augmented_run_count(s.q) - len(responses)} runs vs JMP."
            )
        else:
            deficit = s.r2_target - fit["r2"]
            st.warning(
                f"⚠️ **R² = {fit['r2']:.4f}** (need {s.r2_target:.2f}).  "
                f"Gap = {deficit:.4f}.  Proceed to Phase 2 (ternary blends) to improve."
            )

        # ── Coefficients table ────────────────────────────────────────
        st.subheader("🔢 Coefficients (Phase 1 Model — Linear + 2-way Scheffé)")
        cdf = fit["coeff_df"].copy()
        cdf["p-Value"] = cdf["p-Value"].apply(
            lambda x: "<0.001" if float(x) < 0.001 else f"{float(x):.4f}")
        cdf["LogWorth"] = cdf["LogWorth"].apply(lambda x: f"{float(x):.2f}")
        st.dataframe(cdf.style.format({
            "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
            "t-Stat": "{:.3f}", "CI Lower 95%": "{:.4f}", "CI Upper 95%": "{:.4f}",
        }), use_container_width=True)

        # Significant terms
        n_sig = int(fit["coeff_df"]["Significant"].sum())
        sig_terms = fit["coeff_df"][fit["coeff_df"]["Significant"]]["Term"].tolist()
        st.info(f"  **{n_sig} significant terms** (α=0.05): {', '.join(sig_terms)}")

        # Pareto + residual side-by-side
        col_p, col_r = st.columns(2)
        with col_p:
            st.plotly_chart(pareto_fig(fit["coeff_df"]), use_container_width=True)
        with col_r:
            st.plotly_chart(residual_plots_fig(responses, fit["y_pred"]),
                            use_container_width=True)

        # ── ANOVA ─────────────────────────────────────────────────────
        with st.expander("📋 ANOVA Table"):
            st.dataframe(fit["anova_df"].style.format(
                {"SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-Value": "{:.4f}"},
                na_rep=""), use_container_width=True)

        # ── Navigation ────────────────────────────────────────────────
        st.markdown("---")
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            if st.button("◀ Back to Data Entry"):
                s.eff_stage = 3; st.rerun()
        with nc2:
            if st.button("🔬 Phase 2 — Add Ternary Blends",
                         help="Add ternary blends to detect 3-way interactions",
                         disabled=(fit["r2"] >= s.r2_target)):
                s.eff_stage = 5; st.rerun()
        with nc3:
            if st.button("🏆 Skip to Final Model", type="primary",
                         help="Model already meets quality target"):
                s.eff_stage = 10; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — PHASE 2 DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 5:
    st.header("🔬 Stage 5 — Phase 2: Ternary Augmentation Design")

    q = s.q
    sig_pairs = s["sig_binary_for_p2"]
    binary_combos = list(combinations(range(q), 2))
    sig_pair_names = [f"{s.component_names[i]}·{s.component_names[j]}"
                      for idx in sig_pairs if idx < len(binary_combos)
                      for i, j in [binary_combos[idx]]]

    st.markdown(f"""
    **Phase 2 augmentation** (guided by Phase 1 significance):
    - Significant 2-way pairs from Phase 1: **{', '.join(sig_pair_names) if sig_pair_names else 'none (using all)'}**
    - Ternary blends covering these pairs are prioritised
    - 2 extra centroid replicates added (total = 3 across phases — LOF detection)
    """)

    existing_count = len(s.p1_design)
    if s.p2_design is None:
        s.p2_design = build_phase2_design(q, s.component_names, sig_pairs, existing_count)

    df2 = s.p2_design

    st.subheader("📋 Phase 2 Augmentation Runs")
    st.dataframe(
        df2.style.format({c: "{:.4f}" for c in s.component_names}),
        use_container_width=True
    )
    st.info(f"  **{len(df2)} new runs** in Phase 2  "
            f"(cumulative total: {existing_count + len(df2)})")

    dl1, dl2 = st.columns(2)
    buf_x2 = io.BytesIO()
    with pd.ExcelWriter(buf_x2, engine="openpyxl") as w:
        df2.to_excel(w, sheet_name="Phase2_Design", index=False)
    dl1.download_button("📥 Phase 2 Design (Excel)", buf_x2.getvalue(),
                        file_name="phase2_design.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    dl2.download_button("📥 Phase 2 Design (CSV)", df2.to_csv(index=False, float_format="%.6f"),
                        file_name="phase2_design.csv", mime="text/csv")

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Phase 1 Analysis"):
            s.eff_stage = 4; st.rerun()
    with col_fwd:
        if st.button("▶ Proceed to Phase 2 Data Entry", type="primary"):
            s.eff_stage = 6; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — PHASE 2 DATA ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 6:
    st.header("📋 Stage 6 — Phase 2: Enter Responses")
    st.info("Run the Phase 2 experiments and enter the results below.")

    df2 = s.p2_design.copy()
    df2[s.response_name] = np.nan

    tab_m, tab_u = st.tabs(["✏️ Manual Entry", "📁 Upload File"])

    with tab_m:
        edited2 = st.data_editor(
            df2,
            column_config={
                s.response_name: st.column_config.NumberColumn(s.response_name, required=True),
                **{c: st.column_config.NumberColumn(c, disabled=True) for c in s.component_names},
                "Run":        st.column_config.NumberColumn("Run",      disabled=True),
                "Blend_Type": st.column_config.TextColumn("Blend Type", disabled=True),
                "Blend_Order":st.column_config.NumberColumn("Order",    disabled=True),
                "Response":   st.column_config.NumberColumn("Response", disabled=True),
            },
            hide_index=True, use_container_width=True,
        )
        if st.button("💾 Save Phase 2 Responses", type="primary"):
            resp2 = edited2[s.response_name].values.astype(float)
            if np.any(np.isnan(resp2)):
                st.error("❌ Complete all responses.")
            else:
                new_design = df2[s.component_names].values
                s.design_all = np.vstack([s.design_all, new_design])
                s.responses_all = np.concatenate([s.responses_all, resp2])
                s.p2_design[s.response_name] = resp2
                st.success(f"✅ Phase 2 responses saved! Cumulative runs: {len(s.responses_all)}")

    with tab_u:
        up2 = st.file_uploader("Upload Phase 2 file", type=["csv", "xlsx"], key="p2_up")
        if up2:
            try:
                df_up2 = pd.read_csv(up2) if up2.name.endswith(".csv") else pd.read_excel(up2)
                rc = st.selectbox("Response column", df_up2.columns.tolist(), key="p2_rc")
                if st.button("💾 Load Phase 2", key="p2_load"):
                    resp2 = df_up2[rc].values[:len(df2)].astype(float)
                    new_design = df2[s.component_names].values
                    s.design_all = np.vstack([s.design_all, new_design])
                    s.responses_all = np.concatenate([s.responses_all, resp2])
                    s.p2_design[s.response_name] = resp2
                    st.success("✅ Loaded!")
            except Exception as e:
                st.error(str(e))

    if s.responses_all is not None:
        n_p1 = len(s.p1_design)
        n_total = len(s.responses_all)
        if n_total > n_p1:
            st.success(f"✅ {n_total - n_p1} Phase 2 responses ready (cumulative: {n_total} runs).")

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Phase 2 Design"):
            s.eff_stage = 5; st.rerun()
    n_needed = len(s.p1_design) + len(s.p2_design)
    with col_fwd:
        if st.button("▶ Analyse Phase 2", type="primary",
                     disabled=(s.responses_all is None or len(s.responses_all) < n_needed)):
            s.eff_stage = 7; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — PHASE 2 ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 7:
    st.header("📊 Stage 7 — Phase 2 Analysis")

    design = s.design_all
    responses = s.responses_all

    fit2 = fit_scheffe_model(design, responses, s.component_names, max_order=3)

    if "error" in fit2:
        st.error(f"❌ {fit2['error']}")
    else:
        s.fit_p2 = fit2
        s["sig_ternary_for_p3"] = fit2["sig_ternary_idx"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Runs (cumulative)", len(responses))
        c2.metric("R²", f"{fit2['r2']:.4f}")
        c3.metric("Adj-R²", f"{fit2['adj_r2']:.4f}")
        c4.metric("RMSE", f"{fit2['rmse']:.4f}")
        c5.metric("Target R²", f"{s.r2_target:.2f}")

        if fit2["r2"] >= s.r2_target:
            st.success(
                f"🎉 **R² = {fit2['r2']:.4f} ≥ {s.r2_target:.2f}** — Quality target met!  "
                f"Saved {jmp_augmented_run_count(s.q) - len(responses)} runs vs JMP."
            )
        else:
            st.warning(f"⚠️ R² = {fit2['r2']:.4f} — gap {s.r2_target - fit2['r2']:.4f}. "
                       "Proceed to Phase 3 (quaternary blends).")

        st.subheader("🔢 Coefficients (Phase 2 — Linear + 2-way + 3-way Scheffé)")
        cdf2 = fit2["coeff_df"].copy()
        cdf2["p-Value"] = cdf2["p-Value"].apply(
            lambda x: "<0.001" if float(x) < 0.001 else f"{float(x):.4f}")
        st.dataframe(cdf2.style.format({
            "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
            "t-Stat": "{:.3f}", "CI Lower 95%": "{:.4f}", "CI Upper 95%": "{:.4f}",
        }), use_container_width=True)

        n_sig2 = int(fit2["coeff_df"]["Significant"].sum())
        sig2 = fit2["coeff_df"][fit2["coeff_df"]["Significant"]]["Term"].tolist()
        st.info(f"**{n_sig2} significant terms**: {', '.join(sig2)}")

        col_p, col_r = st.columns(2)
        with col_p:
            st.plotly_chart(pareto_fig(fit2["coeff_df"]), use_container_width=True)
        with col_r:
            st.plotly_chart(residual_plots_fig(responses, fit2["y_pred"]),
                            use_container_width=True)

        with st.expander("📋 ANOVA Table"):
            st.dataframe(fit2["anova_df"].style.format(
                {"SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-Value": "{:.4f}"},
                na_rep=""), use_container_width=True)

        st.markdown("---")
        nc1, nc2, nc3 = st.columns(3)
        with nc1:
            if st.button("◀ Back to Data Entry"):
                s.eff_stage = 6; st.rerun()
        with nc2:
            if st.button("⚗️ Phase 3 — Quaternary Blends",
                         disabled=(fit2["r2"] >= s.r2_target)):
                s.eff_stage = 8; st.rerun()
        with nc3:
            if st.button("🏆 Skip to Final Model", type="primary"):
                s.eff_stage = 10; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — PHASE 3 DESIGN
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 8:
    st.header("⚗️ Stage 8 — Phase 3: Quaternary Augmentation Design")

    q = s.q
    sig_triples = s["sig_ternary_for_p3"]
    sig_bins_for_rep = s["sig_binary_for_p2"]
    ternary_combos = list(combinations(range(q), 3))
    sig_triple_names = ["+".join(s.component_names[j] for j in ternary_combos[i])
                        for i in sig_triples if i < len(ternary_combos)]

    st.markdown(f"""
    **Phase 3 augmentation** (guided by Phase 2 significance):
    - Significant 3-way terms from Phase 2: **{', '.join(sig_triple_names) if sig_triple_names else 'none (using all)'}**
    - Quaternary (4-component) blends added for 4-way interaction coverage
    - Top binary and ternary blends replicated for robust error estimation
    """)

    existing_count = len(s.design_all)
    if s.p3_design is None:
        s.p3_design = build_phase3_design(q, s.component_names,
                                          sig_triples, sig_bins_for_rep,
                                          existing_count)

    df3 = s.p3_design

    st.subheader("📋 Phase 3 Augmentation Runs")
    st.dataframe(
        df3.style.format({c: "{:.4f}" for c in s.component_names}),
        use_container_width=True
    )
    st.info(f"  **{len(df3)} new runs** in Phase 3  "
            f"(cumulative total: {existing_count + len(df3)})")

    dl1, dl2 = st.columns(2)
    buf_x3 = io.BytesIO()
    with pd.ExcelWriter(buf_x3, engine="openpyxl") as w:
        df3.to_excel(w, sheet_name="Phase3_Design", index=False)
    dl1.download_button("📥 Phase 3 Design (Excel)", buf_x3.getvalue(),
                        file_name="phase3_design.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    dl2.download_button("📥 Phase 3 Design (CSV)", df3.to_csv(index=False, float_format="%.6f"),
                        file_name="phase3_design.csv", mime="text/csv")

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Phase 2 Analysis"):
            s.eff_stage = 7; st.rerun()
    with col_fwd:
        if st.button("▶ Proceed to Phase 3 Data Entry", type="primary"):
            s.eff_stage = 9; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — PHASE 3 DATA ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 9:
    st.header("📋 Stage 9 — Phase 3: Enter Responses")

    df3 = s.p3_design.copy()
    df3[s.response_name] = np.nan

    tab_m, tab_u = st.tabs(["✏️ Manual Entry", "📁 Upload File"])

    with tab_m:
        edited3 = st.data_editor(
            df3,
            column_config={
                s.response_name: st.column_config.NumberColumn(s.response_name, required=True),
                **{c: st.column_config.NumberColumn(c, disabled=True) for c in s.component_names},
                "Run":        st.column_config.NumberColumn("Run",      disabled=True),
                "Blend_Type": st.column_config.TextColumn("Blend Type", disabled=True),
                "Blend_Order":st.column_config.NumberColumn("Order",    disabled=True),
                "Response":   st.column_config.NumberColumn("Response", disabled=True),
            },
            hide_index=True, use_container_width=True,
        )
        if st.button("💾 Save Phase 3 Responses", type="primary"):
            resp3 = edited3[s.response_name].values.astype(float)
            if np.any(np.isnan(resp3)):
                st.error("❌ Complete all responses.")
            else:
                new_design3 = df3[s.component_names].values
                s.design_all = np.vstack([s.design_all, new_design3])
                s.responses_all = np.concatenate([s.responses_all, resp3])
                s.p3_design[s.response_name] = resp3
                st.success(f"✅ Phase 3 responses saved! Total runs: {len(s.responses_all)}")

    with tab_u:
        up3 = st.file_uploader("Upload Phase 3 file", type=["csv", "xlsx"], key="p3_up")
        if up3:
            try:
                df_up3 = pd.read_csv(up3) if up3.name.endswith(".csv") else pd.read_excel(up3)
                rc = st.selectbox("Response column", df_up3.columns.tolist(), key="p3_rc")
                if st.button("💾 Load Phase 3", key="p3_load"):
                    resp3 = df_up3[rc].values[:len(df3)].astype(float)
                    new_design3 = df3[s.component_names].values
                    s.design_all = np.vstack([s.design_all, new_design3])
                    s.responses_all = np.concatenate([s.responses_all, resp3])
                    st.success("✅ Loaded!")
            except Exception as e:
                st.error(str(e))

    col_back, col_fwd = st.columns(2)
    with col_back:
        if st.button("◀ Back to Phase 3 Design"):
            s.eff_stage = 8; st.rerun()
    n_needed3 = (len(s.p1_design) +
                 (len(s.p2_design) if s.p2_design is not None else 0) +
                 len(s.p3_design))
    with col_fwd:
        if st.button("▶ Build Final Model", type="primary",
                     disabled=(s.responses_all is None or
                               len(s.responses_all) < n_needed3)):
            s.eff_stage = 10; st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 10 — FINAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif s.eff_stage == 10:
    st.header("🏆 Stage 10 — Final Competitive Model")

    design = s.design_all
    responses = s.responses_all
    q = s.q

    # Fit the highest-order model that is identified
    max_phase = 1
    if s.p3_design is not None and len(responses) >= len(s.p1_design) + len(s.p2_design) + len(s.p3_design):
        max_phase = 3
    elif s.p2_design is not None and len(responses) >= len(s.p1_design) + len(s.p2_design):
        max_phase = 2
    max_order_final = {1: 2, 2: 3, 3: q}[max_phase]

    fit_final = fit_scheffe_model(design, responses, s.component_names,
                                  max_order=max_order_final)

    if "error" in fit_final:
        st.error(f"❌ {fit_final['error']}")
        if st.button("◀ Back"):
            s.eff_stage = max(s.eff_stage - 1, 1); st.rerun()
        st.stop()

    s.fit_p3 = fit_final

    # ── Efficiency banner ─────────────────────────────────────────────────────
    n_used = len(responses)
    n_jmp = jmp_augmented_run_count(q)
    savings = n_jmp - n_used
    pct = 100.0 * savings / n_jmp if n_jmp > 0 else 0

    st.subheader("📊 Efficiency Comparison")
    eff_col1, eff_col2, eff_col3, eff_col4 = st.columns(4)
    eff_col1.metric("Runs Used", n_used)
    eff_col2.metric("JMP Fixed (reference)", n_jmp)
    eff_col3.metric("Runs Saved", savings, delta=f"-{pct:.0f}%")
    eff_col4.metric("Phases Completed", max_phase)

    if savings > 0:
        st.success(f"✅ Completed in **{n_used} runs** — saved **{savings} experiments** "
                   f"({pct:.0f}%) vs the JMP fixed design.")

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.subheader("📈 Final Model Quality")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R²",          f"{fit_final['r2']:.4f}")
    m2.metric("Adjusted R²", f"{fit_final['adj_r2']:.4f}")
    m3.metric("RMSE",        f"{fit_final['rmse']:.4f}")
    m4.metric("F p-value",   "<0.001" if fit_final["f_p"] < 0.001 else f"{fit_final['f_p']:.4f}")
    m5.metric("Model Order",  f"{fit_final['max_order']}-way")

    if fit_final["r2"] >= 0.95:
        st.success(f"✅ Excellent model fit (R² = {fit_final['r2']:.4f})")
    elif fit_final["r2"] >= 0.85:
        st.info(f"ℹ️ Good model fit (R² = {fit_final['r2']:.4f})")
    else:
        st.warning(f"⚠️ Moderate fit (R² = {fit_final['r2']:.4f}) — consider more runs")

    # ── Coefficients ──────────────────────────────────────────────────────────
    st.subheader("🔢 Final Model Coefficients")
    cdf_final = fit_final["coeff_df"].copy()
    cdf_final["p-Value"] = cdf_final["p-Value"].apply(
        lambda x: "<0.001" if float(x) < 0.001 else f"{float(x):.4f}")
    st.dataframe(cdf_final.style.format({
        "Coefficient": "{:.4f}", "Std Error": "{:.4f}",
        "t-Stat": "{:.3f}", "CI Lower 95%": "{:.4f}", "CI Upper 95%": "{:.4f}",
    }), use_container_width=True)

    # ── Model equation ────────────────────────────────────────────────────────
    st.subheader("📐 Model Equation")
    parts = []
    for term, coef in zip(fit_final["term_names"], fit_final["coefficients"]):
        if abs(coef) > 1e-6:
            sign = "+" if coef >= 0 else "−"
            parts.append(f"{sign} {abs(coef):.4f} · {term}")
    if parts:
        eq = f"**{s.response_name}** = " + "  \n".join(parts)
        st.markdown(eq)

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    with st.expander("📋 ANOVA Table"):
        st.dataframe(fit_final["anova_df"].style.format(
            {"SS": "{:.4f}", "MS": "{:.4f}", "F": "{:.3f}", "p-Value": "{:.4f}"},
            na_rep=""), use_container_width=True)

    # ── Residual diagnostics ──────────────────────────────────────────────────
    st.subheader("📉 Residual Diagnostics")
    st.plotly_chart(residual_plots_fig(responses, fit_final["y_pred"]),
                    use_container_width=True)

    # ── Pareto chart ──────────────────────────────────────────────────────────
    st.subheader("📊 Effect Significance (Pareto)")
    st.plotly_chart(pareto_fig(fit_final["coeff_df"]), use_container_width=True)

    # ── Normality test ────────────────────────────────────────────────────────
    residuals = fit_final["residuals"]
    if len(residuals) >= 3:
        sw_stat, sw_p = scipy_stats.shapiro(residuals)
        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Shapiro-Wilk W", f"{sw_stat:.4f}")
        nc2.metric("p-Value",        f"{sw_p:.4f}")
        nc3.metric("Residuals Normal?",
                   "✅ Yes" if sw_p > 0.05 else "⚠️ Marginal" if sw_p > 0.01 else "❌ No")

    # ── Experiment budget summary ─────────────────────────────────────────────
    st.subheader("📊 Experiment Budget Summary")
    p1r = len(s.p1_design)
    p2r = len(s.p2_design) if s.p2_design is not None else 0
    p3r = len(s.p3_design) if s.p3_design is not None else 0

    budget_rows = [
        {"Phase": "Phase 1 (Vertices + Binary)",
         "Runs": p1r, "Note": "All vertices × 2 + all binary blends × 1 + centroid × 1"},
    ]
    if p2r:
        budget_rows.append({
            "Phase": "Phase 2 (Ternary + LOF centroid)",
            "Runs": p2r, "Note": "Guided by Phase 1 significance"})
    if p3r:
        budget_rows.append({
            "Phase": "Phase 3 (Quaternary)",
            "Runs": p3r, "Note": "Guided by Phase 2 significance"})
    budget_rows += [
        {"Phase": "TOTAL (this workflow)", "Runs": n_used, "Note": ""},
        {"Phase": "JMP Fixed (reference)",  "Runs": n_jmp,  "Note": ""},
        {"Phase": f"Savings",              "Runs": savings, "Note": f"{pct:.0f}% reduction"},
    ]
    st.dataframe(pd.DataFrame(budget_rows), use_container_width=True, hide_index=True)

    # ── Download full results ─────────────────────────────────────────────────
    st.subheader("📥 Download Final Results")

    pred_df = pd.DataFrame(s.design_all, columns=s.component_names)
    pred_df[s.response_name] = s.responses_all
    pred_df["Predicted"]         = fit_final["y_pred"]
    pred_df["Residual"]          = fit_final["residuals"]
    pred_df["Std Residual"]      = (fit_final["residuals"]
                                    / (fit_final["rmse"] + 1e-12))

    buf_final = io.BytesIO()
    with pd.ExcelWriter(buf_final, engine="openpyxl") as writer:
        fit_final["coeff_df"].to_excel(writer, sheet_name="Coefficients", index=False)
        fit_final["anova_df"].to_excel(writer,  sheet_name="ANOVA",        index=False)
        pred_df.to_excel(writer,                sheet_name="Predictions",   index=False)
        pd.DataFrame(budget_rows).to_excel(writer, sheet_name="Budget",    index=False)

    st.download_button(
        "📂 Download Full Report (Excel — all sheets)",
        buf_final.getvalue(),
        file_name="efficient_sequential_final_model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    dl_c1, dl_c2 = st.columns(2)
    dl_c1.download_button(
        "📊 Coefficients (CSV)",
        fit_final["coeff_df"].to_csv(index=False, float_format="%.6f"),
        file_name="final_coefficients.csv", mime="text/csv")
    dl_c2.download_button(
        "📉 Predictions (CSV)",
        pred_df.to_csv(index=False, float_format="%.6f"),
        file_name="final_predictions.csv", mime="text/csv")

    # ── Completion banner ─────────────────────────────────────────────────────
    st.markdown("---")
    n_sig_final = int(fit_final["coeff_df"]["Significant"].sum())
    st.success(
        f"🎉 **Efficient Sequential Workflow complete!**\n\n"
        f"- Components: {', '.join(s.component_names)}\n"
        f"- Runs used: **{n_used}** (saved {savings} vs JMP fixed design, {pct:.0f}%)\n"
        f"- Significant terms recovered: {n_sig_final}\n"
        f"- Model R² = **{fit_final['r2']:.4f}**, "
        f"Adj-R² = **{fit_final['adj_r2']:.4f}**\n"
        f"- RMSE = {fit_final['rmse']:.4f}"
    )

    col_restart, _ = st.columns(2)
    with col_restart:
        if st.button("🔁 Start New Experiment"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
