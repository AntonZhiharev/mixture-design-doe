"""
Specific test focusing on D-efficiency calculation with real data and 30 runs
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign

print('D-EFFICIENCY TESTING WITH REAL DATA - 30 RUNS')
print('============================================\n')

# Real data from test_implementation.py
all_component_names = ['NP200', 'CPE', 'DL531', 'TL60', 'OPE', 'PVC', 'CaCO3', 'UVStabilisator']
n_components = 8

# Component bounds in parts
component_bounds_parts = [
    (0.02, 0.04),     # NP200 (variable)
    (0.04, 0.16),     # CPE (variable)
    (0.01, 0.08),     # DL531 (variable)
    (0.005, 0.01),    # TL60 (variable)
    (0.04, 0.16),     # OPE (variable)
    (1.0, 1.0),       # PVC (fixed)
    (0.35, 0.35),     # CaCO3 (fixed)
    (0.025, 0.025)    # UVStabilisator (fixed)
]

fixed_parts = {
    'PVC': 1.0,
    'CaCO3': 0.35,
    'UVStabilisator': 0.025
}

# Create mixture design with parts mode and fixed components
mixture = MixtureDesign(
    n_components=n_components,
    component_names=all_component_names,
    component_bounds=component_bounds_parts,
    use_parts_mode=True,
    fixed_components=fixed_parts
)

# Test with 30 runs
n_runs = 30
print(f'Generating D-optimal design with {n_runs} runs (linear model)...')
design = mixture.generate_d_optimal_mixture(n_runs=n_runs, model_type='linear', random_seed=42)

# ========= CRITICAL: Investigate both D-efficiency values =========

# 1. Calculate D-efficiency in optimization space (before fixed component replacement)
# Create a copy of the mixture object with the same settings
mixture_calc = MixtureDesign(
    n_components=n_components,
    component_names=all_component_names,
    component_bounds=component_bounds_parts,
    use_parts_mode=True,
    fixed_components=fixed_parts
)

# Directly call the internal D-efficiency calculation with model_matrix
try:
    # Temporarily disable fixed component tracking for D-efficiency calculation
    original_truly_fixed = mixture_calc.truly_fixed_components
    mixture_calc.truly_fixed_components = set()  # Treat all as variable for calculation
    
    # Create model matrix directly from design
    model_matrix = mixture_calc.create_mixture_model_matrix(design, "linear")
    
    # Raw determinant calculation
    XtX = np.dot(model_matrix.T, model_matrix)
    det_XtX = np.linalg.det(XtX)
    n_params = model_matrix.shape[1]
    raw_d_eff = (det_XtX / n_runs) ** (1/n_params)
    
    # Log determinant calculation (more numerically stable)
    sign, logdet = np.linalg.slogdet(XtX)
    log_d_eff = sign * np.exp(logdet / n_params) / n_runs
    
    # SVD calculation (most numerically stable)
    U, s, Vt = np.linalg.svd(model_matrix, full_matrices=False)
    log_det_svd = np.sum(np.log(s**2))
    svd_d_eff = np.exp(log_det_svd / (2 * n_params)) / n_runs
    
    # Restore original fixed component tracking
    mixture_calc.truly_fixed_components = original_truly_fixed
    
    print("\n===== D-EFFICIENCY CALCULATION DETAILS =====")
    print(f"Model matrix shape: {model_matrix.shape}")
    print(f"Number of parameters: {n_params}")
    print(f"Condition number of X'X: {np.linalg.cond(XtX):.2e}")
    print(f"Raw determinant of X'X: {det_XtX:.10e}")
    print(f"Raw D-efficiency: {raw_d_eff:.10f}")
    print(f"Log determinant D-efficiency: {log_d_eff:.10f}")
    print(f"SVD D-efficiency: {svd_d_eff:.10f}")
    
except Exception as e:
    print(f"Error calculating raw D-efficiency: {str(e)}")

# 2. Evaluate the design using the standard method
results = mixture.evaluate_mixture_design(design, "linear")
print("\n===== STANDARD EVALUATION RESULTS =====")
print(f"D-efficiency (standard): {results['d_efficiency']:.10f}")
print(f"I-efficiency: {results['i_efficiency']:.6f}")

# 3. Check fixed components
fixed_comp_values = {}
for comp_name in fixed_parts:
    idx = all_component_names.index(comp_name)
    values = design[:, idx]
    fixed_comp_values[comp_name] = {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'std': np.std(values)
    }

print("\nFixed component statistics:")
for comp, stats in fixed_comp_values.items():
    print(f"  {comp}: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}, std={stats['std']:.6f}")

# 4. Format D-efficiency with different precision
d_eff = results['d_efficiency']
print("\nD-efficiency with different formatting:")
print(f"  {d_eff:.10f} (10 decimal places)")
print(f"  {d_eff:.6f} (6 decimal places)")
print(f"  {d_eff:.4f} (4 decimal places)")
print(f"  {d_eff:.2f} (2 decimal places)")
print(f"  {d_eff:.1e} (scientific notation)")

# 5. Check if the D-efficiency is being rounded to zero in display
if d_eff < 0.0001:
    print("\nNOTE: D-efficiency is very small. If displayed with limited precision (e.g., %.4f),")
    print("it might appear as 0.0000, even though the actual value is:", d_eff)

print('\nTest complete!')
