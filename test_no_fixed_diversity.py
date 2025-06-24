"""
Test for design diversity when not using fixed components
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign

print('DESIGN DIVERSITY TEST (NO FIXED COMPONENTS)')
print('==========================================\n')

# Component names
component_names = ['Comp1', 'Comp2', 'Comp3', 'Comp4', 'Comp5']

# Component bounds in proportions (no fixed components)
component_bounds = [
    (0.1, 0.5),     # Comp1
    (0.1, 0.5),     # Comp2
    (0.0, 0.1),     # Comp3
    (0.0, 0.1),     # Comp4
    (0.0, 0.1),     # Comp5
]

# Create mixture design WITHOUT fixed components
print("Creating mixture design without fixed components...")
mixture = MixtureDesign(
    n_components=5,
    component_names=component_names,
    component_bounds=component_bounds
)

# Generate D-optimal design with 10 runs
print("\nGenerating D-optimal design with 10 runs...")
design = mixture.generate_d_optimal_mixture(n_runs=10, model_type='linear', random_seed=42)

# Evaluate diversity
print("\nChecking design diversity...")
df = pd.DataFrame(design, columns=component_names).round(4)
print(df)

# Calculate point-to-point distances
distances = []
for i in range(len(design)):
    for j in range(i+1, len(design)):
        dist = np.linalg.norm(design[i] - design[j])
        distances.append(dist)

print(f"\nPoint-to-point distances:")
print(f"  Minimum distance: {min(distances):.6f}")
print(f"  Maximum distance: {max(distances):.6f}")
print(f"  Average distance: {sum(distances)/len(distances):.6f}")
print(f"  Number of unique designs: {len(set(tuple(row) for row in np.round(design, 4)))}")

# Count similar points (distance < 0.01)
similar_count = sum(1 for d in distances if d < 0.01)
print(f"  Number of very similar points (distance < 0.01): {similar_count}")

# Verify row sums
print("\nVerifying row sums:")
for i in range(len(design)):
    print(f"  Row {i+1} sum: {np.sum(design[i]):.6f}")

print("\nTest complete!")
