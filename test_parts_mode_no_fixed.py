"""
Test for parts mode without fixed components
"""

import numpy as np
from mixture_designs import MixtureDesign

print('PARTS MODE WITHOUT FIXED COMPONENTS TEST')
print('====================================\n')

# Component names
component_names = ['A', 'B', 'C', 'D', 'E']

# Component bounds in parts (no fixed components)
component_bounds_parts = [
    (1.0, 2.0),     # A
    (2.0, 4.0),     # B
    (1.5, 3.0),     # C
    (0.5, 1.0),     # D
    (3.0, 6.0),     # E
]

# Create mixture design with parts mode but NO fixed components
print("Creating mixture design with parts mode but NO fixed components...")
mixture = MixtureDesign(
    n_components=5,
    component_names=component_names,
    component_bounds=component_bounds_parts,
    use_parts_mode=True  # No fixed_components parameter
)

# Generate D-optimal design
print("\nGenerating D-optimal design with 10 runs...")
design = mixture.generate_d_optimal_mixture(n_runs=10, model_type='linear', random_seed=42)

# Evaluate the design
results = mixture.evaluate_mixture_design(design, "linear")
print(f"\nD-efficiency: {results['d_efficiency']:.6f}")
print(f"I-efficiency: {results['i_efficiency']:.6f}")

# Display the design
print("\nDesign points (first 3):")
import pandas as pd
df = pd.DataFrame(design[:3], columns=component_names).round(4)
print(df)

# Check sums
print("\nVerifying sums of components:")
for i in range(min(3, len(design))):
    print(f"Row {i+1} sum: {np.sum(design[i]):.6f}")

print("\nTest complete!")
