"""
Test that D-efficiency from optimization space is correctly preserved
"""

import numpy as np
from mixture_designs import MixtureDesign

print('D-EFFICIENCY PRESERVATION TEST')
print('=============================\n')

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

# Evaluate the design - should use the stored optimization space D-efficiency
results = mixture.evaluate_mixture_design(design, "linear")
print(f"\nD-efficiency from evaluate_mixture_design: {results['d_efficiency']:.6f}")
print(f"This should match the optimization space value printed during generation.")

# Verify it works with different run counts
for runs in [10, 20]:
    print(f"\nTesting with {runs} runs:")
    design_new = mixture.generate_d_optimal_mixture(n_runs=runs, model_type='linear', random_seed=42)
    results_new = mixture.evaluate_mixture_design(design_new, "linear")
    print(f"D-efficiency from evaluate_mixture_design: {results_new['d_efficiency']:.6f}")
    
print("\nTest complete!")
