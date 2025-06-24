"""
Test the improved candidate generation with vertex and edge points
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

print("=== Testing Improved Candidate Generation ===\n")

# Test with 25 runs (sufficient for 15 parameters)
print("Testing with 25 runs:")
design_25 = mixture.generate_d_optimal_mixture(25, "quadratic", random_seed=42)
d_eff_25 = mixture._calculate_d_efficiency(design_25, "quadratic")
i_eff_25 = mixture._calculate_i_efficiency(design_25, "quadratic")
print(f"  D-efficiency: {d_eff_25:.6f}")
print(f"  I-efficiency: {i_eff_25:.2f}")

print(f"\nFirst 5 design points to verify diversity:")
for i in range(min(5, len(design_25))):
    point = design_25[i]
    # Check if any component is dominant (>0.5)
    max_component = np.max(point[:5])  # Only variable components
    dominant = "VERTEX" if max_component > 0.5 else "INTERIOR" 
    print(f"  Point {i+1}: max_var = {max_component:.3f} ({dominant})")
    print(f"    Components: {point[:5].round(3)}")

print(f"\n=== Summary ===")
print(f"25 runs: D-eff = {d_eff_25:.6f}, I-eff = {i_eff_25:.2f}")

# Check if we have good vertex coverage
vertex_points = sum(1 for point in design_25 if np.max(point[:5]) > 0.5)
edge_points = sum(1 for point in design_25 if np.max(point[:5]) > 0.4 and np.max(point[:5]) <= 0.5)
interior_points = len(design_25) - vertex_points - edge_points

print(f"\nPoint distribution:")
print(f"  Vertex-like points (max > 0.5): {vertex_points}")
print(f"  Edge-like points (0.4 < max ≤ 0.5): {edge_points}")
print(f"  Interior points (max ≤ 0.4): {interior_points}")

if vertex_points >= 3 and edge_points >= 2:
    print("✅ Good vertex and edge coverage!")
else:
    print("❌ Need better vertex/edge coverage")

if d_eff_25 > 0.001:
    print("✅ D-efficiency improved!")
else:
    print("❌ D-efficiency still at minimum")
