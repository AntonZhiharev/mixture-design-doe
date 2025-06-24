"""
Test the fixed space solution implementation
"""

from mixture_designs import MixtureDesign

# Example from test_fixed_space_solution.py
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
    (0.35, 0.35),       # CaCO3 (fixed)
    (0.025, 0.025)      # UVStabilisator (fixed)
]

fixed_parts = {
    'PVC': 1.0,
    'CaCO3': 0.35,
    'UVStabilisator': 0.025
}

print('Testing fixed space solution implementation...')

# Create mixture design with parts mode and fixed components
mixture = MixtureDesign(
    n_components=n_components,
    component_names=all_component_names,
    component_bounds=component_bounds_parts,
    use_parts_mode=True,
    fixed_components=fixed_parts
)

print('\nGenerating D-optimal design with 5 runs...')
design = mixture.generate_d_optimal_mixture(n_runs=5, model_type='linear', random_seed=42)

print(f'\nFinal design shape: {design.shape}')
print('Final design with batch quantities (batch size = 100):')
batch_quantities = mixture.convert_to_batch_quantities(design, 100.0)

for i, row in enumerate(batch_quantities):
    print(f'\nMix {i+1}:')
    for j, comp_name in enumerate(all_component_names):
        if comp_name in fixed_parts:
            print(f'  {comp_name}: {row[j]:.2f} (FIXED)')
        else:
            print(f'  {comp_name}: {row[j]:.2f}')
    print(f'  TOTAL: {sum(row):.2f}')
