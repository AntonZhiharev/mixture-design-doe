"""
Compare fixed space solution with original approach
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign

print('COMPARISON OF DESIGNS WITH AND WITHOUT FIXED COMPONENTS')
print('=====================================================\n')

# Component names and bounds similar to the issue in the image
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5']

component_bounds = [
    (0.0, 0.5),     # Component_1
    (0.0, 0.5),     # Component_2
    (0.0, 0.5),     # Component_3
    (0.0, 0.5),     # Component_4
    (0.0, 0.5),     # Component_5
]

# Test 1: Without fixed components (should be diverse)
print("1. WITHOUT FIXED COMPONENTS TEST")
print("--------------------------------")
mixture_no_fixed = MixtureDesign(
    n_components=5,
    component_names=component_names,
    component_bounds=component_bounds
)

print("\nGenerating D-optimal design with 4 runs...")
design_no_fixed = mixture_no_fixed.generate_d_optimal_mixture(n_runs=4, model_type='linear', random_seed=42)

# Evaluate diversity
print("\nDesign points:")
df_no_fixed = pd.DataFrame(design_no_fixed, columns=component_names).round(4)
print(df_no_fixed)

# Calculate point-to-point distances
distances = []
for i in range(len(design_no_fixed)):
    for j in range(i+1, len(design_no_fixed)):
        dist = np.linalg.norm(design_no_fixed[i] - design_no_fixed[j])
        distances.append(dist)

print(f"\nPoint-to-point distances:")
print(f"  Minimum distance: {min(distances):.6f}")
print(f"  Maximum distance: {max(distances):.6f}")
print(f"  Average distance: {sum(distances)/len(distances):.6f}")
print(f"  Number of unique designs: {len(set(tuple(row) for row in np.round(design_no_fixed, 4)))}")

# Test 2: With fixed components (for comparison)
print("\n\n2. WITH FIXED COMPONENTS TEST")
print("-----------------------------")

# Use fixed components as in the image (similar values)
fixed_components = {
    'Component_3': 0.04,
    'Component_4': 0.04,
    'Component_5': 0.04,
}

mixture_fixed = MixtureDesign(
    n_components=5,
    component_names=component_names,
    component_bounds=component_bounds,
    fixed_components=fixed_components
)

print("\nGenerating D-optimal design with 4 runs...")
design_fixed = mixture_fixed.generate_d_optimal_mixture(n_runs=4, model_type='linear', random_seed=42)

# Evaluate diversity
print("\nDesign points:")
df_fixed = pd.DataFrame(design_fixed, columns=component_names).round(4)
print(df_fixed)

# Calculate point-to-point distances for fixed design
distances_fixed = []
for i in range(len(design_fixed)):
    for j in range(i+1, len(design_fixed)):
        dist = np.linalg.norm(design_fixed[i] - design_fixed[j])
        distances_fixed.append(dist)

print(f"\nPoint-to-point distances:")
print(f"  Minimum distance: {min(distances_fixed) if distances_fixed else 'N/A':.6f}")
print(f"  Maximum distance: {max(distances_fixed) if distances_fixed else 'N/A':.6f}")
print(f"  Average distance: {sum(distances_fixed)/len(distances_fixed) if distances_fixed else 'N/A':.6f}")
print(f"  Number of unique designs: {len(set(tuple(row) for row in np.round(design_fixed, 4)))}")

print("\nTest complete!")
