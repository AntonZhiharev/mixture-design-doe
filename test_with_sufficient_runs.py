"""
Test with sufficient runs for all parameters
"""

from mixture_designs import MixtureDesign
import numpy as np

# Create same setup as in your Streamlit app
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']
component_bounds = [(0, 1), (0, 1), (0, 0.1), (0, 0.1), (0, 0.1), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)]
fixed_components = {'Fixed_1': 0.05, 'Fixed_2': 0.05, 'Fixed_3': 0.05}

mixture = MixtureDesign(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    use_parts_mode=True,
    fixed_components=fixed_components
)

print("=== Testing with Sufficient Runs ===\n")

# Test with 40 runs (sufficient for 36 parameters)
print("Testing with 40 runs (sufficient for 36 parameters):")
design_40 = mixture.generate_d_optimal_mixture(40, "quadratic", random_seed=42)
d_eff_40 = mixture._calculate_d_efficiency(design_40, "quadratic")
i_eff_40 = mixture._calculate_i_efficiency(design_40, "quadratic")
print(f"  D-efficiency: {d_eff_40:.6f}")
print(f"  I-efficiency: {i_eff_40:.2f}")

# Check if we have good vertex coverage
vertex_points = sum(1 for point in design_40 if np.max(point[:5]) > 0.5)
edge_points = sum(1 for point in design_40 if np.max(point[:5]) > 0.4 and np.max(point[:5]) <= 0.5)
interior_points = len(design_40) - vertex_points - edge_points

print(f"\nPoint distribution:")
print(f"  Vertex-like points (max > 0.5): {vertex_points}")
print(f"  Edge-like points (0.4 < max ≤ 0.5): {edge_points}")
print(f"  Interior points (max ≤ 0.4): {interior_points}")

if d_eff_40 > 0.001:
    print("✅ D-efficiency improved with sufficient runs!")
else:
    print("❌ D-efficiency still at minimum even with sufficient runs")

print(f"\n=== Comparison ===")
print(f"With 25 runs (insufficient): D-eff = 0.001000")
print(f"With 40 runs (sufficient):   D-eff = {d_eff_40:.6f}")
