"""
Comprehensive test of the fixed space solution implementation using real data
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign

print('COMPREHENSIVE TESTING WITH REAL DATA')
print('===================================\n')

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

print('1. TESTING WITH INCREASING RUN COUNTS')
print('------------------------------------\n')

# Create mixture design with parts mode and fixed components
mixture = MixtureDesign(
    n_components=n_components,
    component_names=all_component_names,
    component_bounds=component_bounds_parts,
    use_parts_mode=True,
    fixed_components=fixed_parts
)

# Test with various run counts to verify D-efficiency calculation
run_counts = [5, 10, 20, 30]

for n_runs in run_counts:
    print(f'\nGenerating D-optimal design with {n_runs} runs (linear model)...')
    design = mixture.generate_d_optimal_mixture(n_runs=n_runs, model_type='linear', random_seed=42)
    
    # Evaluate the design
    results = mixture.evaluate_mixture_design(design, "linear")
    print(f"D-efficiency: {results['d_efficiency']:.6f}")
    print(f"I-efficiency: {results['i_efficiency']:.6f}")
    
    # Check fixed components
    fixed_comp_values = {}
    for comp_name in fixed_parts:
        idx = all_component_names.index(comp_name)
        values = design[:, idx]
        fixed_comp_values[comp_name] = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values)
        }
    
    print("\nFixed component statistics:")
    for comp, stats in fixed_comp_values.items():
        print(f"  {comp}: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f}")

# Test with quadratic model (more parameters)
print('\n2. TESTING WITH QUADRATIC MODEL')
print('-----------------------------\n')

n_runs = 20  # Need more runs for quadratic model
print(f'Generating D-optimal design with {n_runs} runs (quadratic model)...')
design_quad = mixture.generate_d_optimal_mixture(n_runs=n_runs, model_type='quadratic', random_seed=42)

# Evaluate the design
results_quad = mixture.evaluate_mixture_design(design_quad, "quadratic")
print(f"D-efficiency: {results_quad['d_efficiency']:.6f}")
print(f"I-efficiency: {results_quad['i_efficiency']:.6f}")

# Output sample design points
print("\nSample design points (proportions):")
df = pd.DataFrame(design_quad[:3], columns=all_component_names).round(4)
print(df)

# Convert to batch quantities
print('\nSample design points (batch size = 100):')
batch_quantities = mixture.convert_to_batch_quantities(design_quad[:3], 100.0)
df_batch = pd.DataFrame(batch_quantities, columns=all_component_names).round(2)
print(df_batch)

print('\n3. TESTING WITH SWAPPED BOUNDS')
print('----------------------------\n')

# Deliberately swap bounds for one component
swapped_bounds = component_bounds_parts.copy()
swapped_bounds[0] = (0.04, 0.02)  # Swap NP200 bounds

print('Creating design with swapped bounds for NP200...')
mixture_swapped = MixtureDesign(
    n_components=n_components,
    component_names=all_component_names,
    component_bounds=swapped_bounds,
    use_parts_mode=True,
    fixed_components=fixed_parts
)

print('\nGenerating D-optimal design with 10 runs...')
design_swapped = mixture_swapped.generate_d_optimal_mixture(n_runs=10, model_type='linear', random_seed=42)

# Check if bounds were correctly applied
np200_idx = all_component_names.index('NP200')
np200_values = design_swapped[:, np200_idx]

print(f"\nNP200 values after automatic bound correction:")
print(f"  Min: {np200_values.min():.4f}")
print(f"  Max: {np200_values.max():.4f}")
print(f"  Mean: {np200_values.mean():.4f}")
print(f"  Expected bounds after correction: (0.02, 0.04)")

print('\n4. FINAL VERIFICATION')
print('-------------------\n')

# Create final batch quantities for review
final_design = mixture.generate_d_optimal_mixture(n_runs=10, model_type='linear', random_seed=42)
final_batch = mixture.convert_to_batch_quantities(final_design, 100.0)

print('Final design with batch quantities (batch size = 100):')
for i, row in enumerate(final_batch):
    print(f'\nMix {i+1}:')
    for j, comp_name in enumerate(all_component_names):
        if comp_name in fixed_parts:
            print(f'  {comp_name}: {row[j]:.2f} (FIXED)')
        else:
            print(f'  {comp_name}: {row[j]:.2f}')
    print(f'  TOTAL: {sum(row):.2f}')

print('\nTest complete!')
