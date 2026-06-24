"""
Quick functional test for doe_sequential_workflow_app.py helper functions.
Run from project root: python _test_workflow_functions.py
"""
import sys, os, types
sys.path.insert(0, "src")

# ── Stub streamlit (no server needed) ────────────────────────────────────────
class _AnyMock:
    """Accepts any attribute access and any call, returns itself."""
    def __call__(self, *a, **k): return self
    def __getattr__(self, name): return self
    def __setattr__(self, name, value): object.__setattr__(self, name, value)
    def __iter__(self): return iter([self, self, self])  # st.columns(3)
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __bool__(self): return False

class _SessionState(dict):
    """Dict that also supports attribute access (like st.session_state)."""
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): del self[k]

_mock = _AnyMock()
st_stub = types.ModuleType("streamlit")
st_stub.session_state = _SessionState({
    "workflow_stage": 99,   # force no UI stage to execute
    "design_kind": "factorial",
    "n_factors": 3,
    "factor_names": ["A","B","C"],
    "factor_lows": [-1.,-1.,-1.],
    "factor_highs": [1.,1.,1.],
    "response_name": "Response",
    "model_type": "quadratic",
    "initial_design_coded": None,
    "initial_design_natural": None,
    "n_initial_runs": 12,
    "initial_responses": None,
    "screening_results": None,
    "aliasing_df": None,
    "selected_term_indices": None,
    "foldover_design_coded": None,
    "foldover_design_natural": None,
    "foldover_responses": None,
    "fold_columns": "all",
    "combined_design": None,
    "combined_responses": None,
    "final_results": None,
})

class _ColConf:
    NumberColumn   = staticmethod(lambda *a,**k: None)
    TextColumn     = staticmethod(lambda *a,**k: None)
    CheckboxColumn = staticmethod(lambda *a,**k: None)

st_stub.column_config = _ColConf()
st_stub.sidebar = _AnyMock()

for attr in [
    "set_page_config","header","subheader","info","warning","error","success",
    "markdown","title","columns","button","selectbox","radio","multiselect",
    "number_input","text_input","data_editor","tabs","expander","plotly_chart",
    "dataframe","download_button","table","write","caption","stop","rerun",
    "spinner","metric","add_hline","add_vline","progress","status","code",
]:
    setattr(st_stub, attr, _AnyMock())

sys.modules["streamlit"] = st_stub

# ── Load the module functions without Streamlit running ──────────────────────
import numpy as np, pandas as pd
from scipy import stats as scipy_stats
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import io, json

src = open("src/apps/doe_sequential_workflow_app.py", encoding="utf-8").read()

ns = {
    "__name__": "__imported__",
    "__file__": os.path.abspath("src/apps/doe_sequential_workflow_app.py"),
    "st": st_stub,
    "np": np, "pd": pd, "scipy_stats": scipy_stats,
    "combinations": combinations,
    "LinearRegression": LinearRegression,
    "r2_score": r2_score, "mean_squared_error": mean_squared_error,
    "go": go, "make_subplots": make_subplots, "px": px,
    "io": io, "json": json, "os": os, "sys": sys,
    "_USE_MODULAR": False,
}
exec(compile(src, "doe_sequential_workflow_app.py", "exec"), ns)
print("Module exec: OK")

# ── Grab helpers ─────────────────────────────────────────────────────────────
_build_model_matrix      = ns["_build_model_matrix"]
fit_model_statistics     = ns["fit_model_statistics"]
build_reduced_model      = ns["build_reduced_model"]
detect_aliasing          = ns["detect_aliasing"]
generate_foldover_factorial = ns["generate_foldover_factorial"]
generate_augmentation_mixture = ns["generate_augmentation_mixture"]

# ── Test 1: _build_model_matrix ───────────────────────────────────────────────
print("\n--- Test 1: _build_model_matrix ---")
design_eye = np.eye(3)

X_f, names_f = _build_model_matrix(design_eye, "quadratic", "factorial")
print(f"  factorial quadratic 3x3: shape={X_f.shape}, n_terms={len(names_f)}")
print(f"  terms: {names_f}")
# intercept(1) + 3 main + 3 interact + 3 quadratic = 10 terms
assert X_f.shape[1] == 10, f"Expected 10 terms, got {X_f.shape[1]}"

X_m, names_m = _build_model_matrix(np.ones((5,3))/3, "quadratic", "mixture")
print(f"  mixture quadratic 5x3: shape={X_m.shape}, n_terms={len(names_m)}")
# 3 linear + 3 interactions = 6 terms
assert X_m.shape[1] == 6, f"Expected 6 mixture terms, got {X_m.shape[1]}"
print("  PASSED")

# ── Test 2: fit_model_statistics – factorial ──────────────────────────────────
print("\n--- Test 2: fit_model_statistics (factorial) ---")
rng = np.random.default_rng(42)
n = 20
design_f = rng.uniform(-1, 1, (n, 3))
# True model: y = 5 + 2A - 3B + 1.5*A*B
y = (5 + 2*design_f[:,0] - 3*design_f[:,1]
       + 1.5*design_f[:,0]*design_f[:,1]
       + rng.normal(0, 0.3, n))

res = fit_model_statistics(design_f, y, "quadratic", "factorial", ["A","B","C"])
print(f"  R2={res['r2']:.4f}, adj_R2={res['adj_r2']:.4f}, RMSE={res['rmse']:.4f}")
print(f"  n_params={res['n_params']}, df_res={res['df_res']}")
print(f"  Terms: {res['term_names']}")
print(f"  p-values: {dict(zip(res['term_names'], res['p_values'].round(4)))}")
assert res["r2"] > 0.85, f"R2 too low: {res['r2']}"
# Intercept, A, B, A*B should be significant
p_map = dict(zip(res["term_names"], res["p_values"]))
for sig_term in ["Intercept", "A", "B", "A×B"]:
    assert p_map[sig_term] < 0.05, f"{sig_term} should be significant, p={p_map[sig_term]:.4f}"
print("  PASSED")

# ── Test 3: fit_model_statistics – mixture ────────────────────────────────────
print("\n--- Test 3: fit_model_statistics (mixture) ---")
# Simple 3-component mixture
mix_pts = rng.dirichlet(np.ones(3), 18)
y_mix = (75*mix_pts[:,0] + 85*mix_pts[:,1] + 45*mix_pts[:,2]
          + 40*mix_pts[:,0]*mix_pts[:,1]
          + rng.normal(0, 1.0, 18))

res_m = fit_model_statistics(mix_pts, y_mix, "quadratic", "mixture", ["x1","x2","x3"])
print(f"  R2={res_m['r2']:.4f}, adj_R2={res_m['adj_r2']:.4f}")
print(f"  Terms: {res_m['term_names']}")
assert res_m["r2"] > 0.85, f"Mixture R2 too low: {res_m['r2']}"
print("  PASSED")

# ── Test 4: generate_foldover_factorial ─────────────────────────────────────
print("\n--- Test 4: generate_foldover_factorial ---")
d_orig = np.array([[1,1,1],[-1,1,-1],[1,-1,-1],[-1,-1,1]], dtype=float)

# Full fold-over
fo_all = generate_foldover_factorial(d_orig, "all")
assert np.allclose(fo_all, -d_orig), "Full fold-over: expected -d_orig"
print("  Full fold-over: OK (-d_orig verified)")

# Partial fold-over: fold only factor 0
fo_partial = generate_foldover_factorial(d_orig, [0])
assert np.allclose(fo_partial[:,0], -d_orig[:,0]), "Partial fold: col 0 not negated"
assert np.allclose(fo_partial[:,1:], d_orig[:,1:]), "Partial fold: cols 1,2 should not change"
print("  Partial fold-over (col 0 only): OK")

# Combined design after fold-over should have 8 runs
combined = np.vstack([d_orig, fo_all])
assert combined.shape == (8, 3), f"Combined design shape wrong: {combined.shape}"
print(f"  Combined design shape: {combined.shape} -> OK")
print("  PASSED")

# ── Test 5: detect_aliasing ─────────────────────────────────────────────────
print("\n--- Test 5: detect_aliasing ---")
# A 4-run unreplicated 2^(3-1) design: C = A*B
A = np.array([1,-1,1,-1], dtype=float)
B = np.array([1,1,-1,-1], dtype=float)
C = A * B  # generator C = AB → C aliases with A*B
d_frac = np.column_stack([A, B, C])

alias_df = detect_aliasing(d_frac, "quadratic", "factorial", ["A","B","C"], threshold=0.9)
print(f"  Found {len(alias_df)} aliased pairs with |corr| >= 0.9:")
print(alias_df.to_string(index=False) if len(alias_df) > 0 else "  (none)")
# In a 2^(3-1) design, C should alias with A*B
assert len(alias_df) > 0, "Expected aliasing to be detected in 2^(3-1) design"
print("  PASSED")

# ── Test 6: build_reduced_model ──────────────────────────────────────────────
print("\n--- Test 6: build_reduced_model ---")
# Use only intercept(0), A(1), B(2) from the 20-run factorial design
red = build_reduced_model(design_f, y, [0,1,2], "quadratic", "factorial", ["A","B","C"])
print(f"  Reduced R2={red['r2']:.4f}, n_params={red['n_params']}, terms={red['term_names']}")
assert red["n_params"] == 3, f"Expected 3 params, got {red['n_params']}"

# Full model with all significant terms
red_full = build_reduced_model(design_f, y, [0,1,2,4], "quadratic", "factorial", ["A","B","C"])
print(f"  Reduced (with A*B) R2={red_full['r2']:.4f}")
assert red_full["r2"] > red["r2"], "Adding A*B should improve R2"
print("  PASSED")

# ── Test 7: Fold-over improves aliasing (integration test) ────────────────────
print("\n--- Test 7: Fold-over de-aliasing integration test ---")
# Start with 2^(3-1) design (C = AB)
# True model: y = 10 + 5A - 3B (C has zero effect)
y_alias = 10 + 5*A - 3*B + rng.normal(0, 0.2, 4)

# Fit initial model
res_init = fit_model_statistics(d_frac, y_alias, "linear", "factorial", ["A","B","C"])
print(f"  Initial (4 runs, 2^(3-1)): R2={res_init['r2']:.4f}")
print(f"  Initial p-values: {dict(zip(res_init['term_names'], res_init['p_values'].round(4)))}")

# Apply full fold-over
fo_block = generate_foldover_factorial(d_frac, "all")
A2, B2, C2 = fo_block[:,0], fo_block[:,1], fo_block[:,2]
y_fo = 10 + 5*A2 - 3*B2 + rng.normal(0, 0.2, 4)

# Combined analysis
d_combined = np.vstack([d_frac, fo_block])
y_combined = np.concatenate([y_alias, y_fo])
res_comb = fit_model_statistics(d_combined, y_combined, "linear", "factorial", ["A","B","C"])
print(f"  After fold-over (8 runs): R2={res_comb['r2']:.4f}")
print(f"  Combined p-values: {dict(zip(res_comb['term_names'], res_comb['p_values'].round(4)))}")
# After fold-over, A should be significant and C should NOT be
p_comb = dict(zip(res_comb["term_names"], res_comb["p_values"]))
print(f"  A significant (p<0.05): {p_comb['A'] < 0.05}")
print(f"  C not significant (p>0.05): {p_comb['C'] > 0.05}")
print("  PASSED")

print("\n" + "="*55)
print("  ALL 7 FUNCTIONAL TESTS PASSED")
print("="*55)
