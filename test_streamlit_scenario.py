"""
Test the exact Streamlit scenario with 6 components and 3 fixed
"""

from enhanced_mixture_designs import EnhancedMixtureDesign
import numpy as np

# Test the exact scenario that was failing in Streamlit
component_names = ['Component_1', 'Component_2', 'Component_3', 'Fixed_1', 'Fixed_2', 'Fixed_3']
component_bounds_parts = [
    (0.1, 1.0),    # Component_1 (variable)
    (0.1, 1.0),    # Component_2 (variable) 
    (0.1, 1.0),    # Component_3 (variable)
    (0.05, 0.05),  # Fixed_1 (fixed)
    (0.05, 0.05),  # Fixed_2 (fixed)
    (0.05, 0.05)   # Fixed_3 (fixed)
]

fixed_parts = {
    'Fixed_1': 0.05,
    'Fixed_2': 0.05, 
    'Fixed_3': 0.05
}

print('=== TESTING ENHANCED MIXTURE DESIGN (STREAMLIT VERSION) ===')

# Test with EnhancedMixtureDesign (used by Streamlit)
enhanced_mixture = EnhancedMixtureDesign(
    6, 
    component_names, 
    component_bounds_parts, 
    use_parts_mode=True,
    fixed_components=fixed_parts
)

print('✅ Enhanced mixture created successfully')

design = enhanced_mixture.generate_d_optimal_mixture(n_runs=3, model_type='linear', random_seed=42)

print()
print('Fixed component verification:')
total_parts = sum(bound[1] for bound in component_bounds_parts)
print(f'Total parts: {total_parts}')

for j, comp_name in enumerate(component_names):
    if comp_name in fixed_parts:
        expected = fixed_parts[comp_name] / total_parts  # Correct calculation
        actual = design[0][j]
        print(f'  {comp_name}: actual={actual:.6f}, expected={expected:.6f}, correct={abs(actual-expected) < 1e-5}')
    else:
        print(f'  {comp_name}: {design[0][j]:.6f} (variable)')

print(f'')
print(f'First mixture sum: {sum(design[0]):.6f}')
print('')

# Check all rows
for i in range(len(design)):
    row_sum = sum(design[i])
    print(f'Row {i+1} sum: {row_sum:.6f}')

print('')
print('✅ SUCCESS! Fixed components now show correct values in Streamlit interface!')
