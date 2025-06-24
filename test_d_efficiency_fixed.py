"""
Test script to verify the fixed D-efficiency calculation with higher run counts
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign

# 1. Create a test case with fixed components
print("TEST 1: 4-component mixture with 1 fixed component")
component_names = ['A', 'B', 'C', 'D']
component_bounds = [(0.1, 0.6), (0.1, 0.5), (0.1, 0.4), (0.1, 0.3)]
fixed_components = {'D': 0.2}  # Fix component D at 20%

# Create design with fixed component
mixture = MixtureDesign(4, component_names, component_bounds, fixed_components=fixed_components)

# Test with increasing run counts
for n_runs in [10, 20, 30, 40, 50]:
    print(f"\nGenerating D-optimal design with {n_runs} runs...")
    design = mixture.generate_d_optimal_mixture(n_runs, "quadratic", random_seed=42)
    
    # Evaluate the design
    results = mixture.evaluate_mixture_design(design, "quadratic")
    print(f"D-efficiency: {results['d_efficiency']:.6f}")
    print(f"I-efficiency: {results['i_efficiency']:.6f}")
    
    # Print a sample of the design
    df = pd.DataFrame(design[:3], columns=component_names).round(4)
    print("\nSample design points:")
    print(df)
    
    # Verify component D is fixed at 0.2
    d_vals = design[:, 3]  # Component D is at index 3
    print(f"Component D values - min: {np.min(d_vals):.4f}, max: {np.max(d_vals):.4f}, mean: {np.mean(d_vals):.4f}")

# 2. Test with parts mode
print("\n\nTEST 2: 3-component mixture in parts mode")
component_names = ['X', 'Y', 'Z']
component_bounds = [(1, 5), (3, 8), (2, 6)]  # Parts bounds
fixed_components = {'Z': 3}  # Fix Z at 3 parts

# Create design with parts mode
mixture_parts = MixtureDesign(3, component_names, component_bounds, 
                             use_parts_mode=True, fixed_components=fixed_components)

# Test with 30 runs
n_runs = 30
print(f"\nGenerating D-optimal design with {n_runs} runs in parts mode...")
parts_design = mixture_parts.generate_d_optimal_mixture(n_runs, "quadratic", random_seed=42)

# Evaluate the design
results = mixture_parts.evaluate_mixture_design(parts_design, "quadratic")
print(f"D-efficiency: {results['d_efficiency']:.6f}")
print(f"I-efficiency: {results['i_efficiency']:.6f}")

# Print a sample of the design
df = pd.DataFrame(parts_design[:5], columns=component_names).round(4)
print("\nSample design points (proportions):")
print(df)

# Calculate batch quantities (to parts)
batch_size = 100
parts_quantities = mixture_parts.convert_to_batch_quantities(parts_design, batch_size)
df_parts = pd.DataFrame(parts_quantities[:5], columns=component_names).round(2)
print("\nSample design points (parts for 100-unit batch):")
print(df_parts)

print("\nTest complete!")
