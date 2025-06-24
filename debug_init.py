"""
Debug initialization to see why truly_fixed_components is empty
"""

from mixture_designs import MixtureDesign
import numpy as np

print("=== Debugging Initialization ===\n")

# Create same setup as in your Streamlit app
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']
component_bounds = [(0, 1), (0, 1), (0, 0.1), (0, 0.1), (0, 0.1), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)]
fixed_components = {'Fixed_1': 0.05, 'Fixed_2': 0.05, 'Fixed_3': 0.05}

print("BEFORE MixtureDesign creation:")
print(f"fixed_components: {fixed_components}")

# Create mixture design with parts mode
mixture = MixtureDesign(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    use_parts_mode=True,
    fixed_components=fixed_components
)

print("\nAFTER MixtureDesign creation:")
print(f"mixture.fixed_components: {mixture.fixed_components}")
print(f"mixture.original_fixed_components: {mixture.original_fixed_components}")
print(f"mixture.truly_fixed_components: {mixture.truly_fixed_components}")

if hasattr(mixture, 'original_fixed_components_proportions'):
    print(f"mixture.original_fixed_components_proportions: {mixture.original_fixed_components_proportions}")

print(f"\nComponent names: {mixture.component_names}")

print("\nChecking which components should be variable vs fixed:")
for i, name in enumerate(mixture.component_names):
    is_in_truly_fixed = name in mixture.truly_fixed_components
    is_in_original_fixed = name in mixture.original_fixed_components
    print(f"  {i}: {name} - truly_fixed: {is_in_truly_fixed}, original_fixed: {is_in_original_fixed}")
