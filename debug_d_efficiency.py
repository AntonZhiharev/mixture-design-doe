"""
Debug D-efficiency calculation to find the root cause
"""

from mixture_designs import MixtureDesign
import numpy as np

print("=== Debugging D-Efficiency Calculation ===\n")

# Create same setup as in your Streamlit app
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']
component_bounds = [(0, 1), (0, 1), (0, 0.1), (0, 0.1), (0, 0.1), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)]
fixed_components = {'Fixed_1': 0.05, 'Fixed_2': 0.05, 'Fixed_3': 0.05}

# Create mixture design with parts mode
mixture = MixtureDesign(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    use_parts_mode=True,
    fixed_components=fixed_components
)

print("Testing single D-optimal design generation...")
design = mixture.generate_d_optimal_mixture(n_runs=15, model_type="quadratic", random_seed=42)

print("\nDesign shape:", design.shape)
print("First few rows:")
print(design[:3])

print("\nTesting model matrix creation...")
model_matrix = mixture.create_mixture_model_matrix(design, "quadratic")
print("Model matrix shape:", model_matrix.shape)
print("First few rows of model matrix:")
print(model_matrix[:3])

print("\nTesting XtX calculation...")
XtX = np.dot(model_matrix.T, model_matrix)
print("XtX shape:", XtX.shape)
print("XtX condition number:", np.linalg.cond(XtX))

print("\nTesting determinant calculation...")
try:
    det_XtX = np.linalg.det(XtX)
    print("Determinant:", det_XtX)
except Exception as e:
    print("Determinant calculation failed:", e)

print("\nStep-by-step D-efficiency calculation...")

# Manual D-efficiency calculation with debugging
try:
    n_runs = design.shape[0]
    n_params = model_matrix.shape[1]
    print(f"n_runs: {n_runs}, n_params: {n_params}")
    
    if n_runs < n_params:
        print("❌ PROBLEM: Not enough runs for parameters")
    else:
        print("✅ Enough runs for parameters")
    
    cond_num = np.linalg.cond(XtX)
    print(f"Condition number: {cond_num}")
    if cond_num > 1e12:
        print("❌ PROBLEM: Matrix is ill-conditioned")
    else:
        print("✅ Matrix condition is acceptable")
    
    det_XtX = np.linalg.det(XtX)
    print(f"Determinant: {det_XtX}")
    if det_XtX <= 1e-15:
        print("❌ PROBLEM: Determinant is too small")
    else:
        print("✅ Determinant is acceptable")
    
    # Calculate D-efficiency
    d_eff = (det_XtX / n_runs) ** (1/n_params)
    print(f"Raw D-efficiency: {d_eff}")
    
    if np.isnan(d_eff) or np.isinf(d_eff) or d_eff <= 0:
        print("❌ PROBLEM: D-efficiency is NaN, Inf, or negative")
    else:
        print(f"✅ Final D-efficiency: {min(d_eff, 1.0)}")
        
except Exception as e:
    print(f"❌ EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting fixed components identification...")
print("Fixed components:", mixture.fixed_components)
if hasattr(mixture, 'original_fixed_components_proportions'):
    print("Original fixed components (proportions):", mixture.original_fixed_components_proportions)
print("Component names:", mixture.component_names)

# Check which components are considered variable
variable_indices = []
for i, name in enumerate(mixture.component_names):
    if name not in mixture.fixed_components:
        variable_indices.append(i)
print("Variable component indices:", variable_indices)
print("Number of variable components:", len(variable_indices))
