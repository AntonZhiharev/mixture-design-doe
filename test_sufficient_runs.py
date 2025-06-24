"""
Test D-optimal with sufficient runs for the model parameters
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

print("=== Testing with Sufficient Runs ===\n")

# Test with 20 runs (should be enough for 15 parameters)
print("Testing with 20 runs (sufficient for 15 parameters):")
design_20 = mixture.generate_d_optimal_mixture(20, "quadratic", random_seed=42)
d_eff_20 = mixture._calculate_d_efficiency(design_20, "quadratic")
i_eff_20 = mixture._calculate_i_efficiency(design_20, "quadratic")
print(f"  D-efficiency: {d_eff_20:.6f}")
print(f"  I-efficiency: {i_eff_20:.2f}")

# Test with 25 runs (even more comfortable)
print(f"\nTesting with 25 runs (generous for 15 parameters):")
design_25 = mixture.generate_d_optimal_mixture(25, "quadratic", random_seed=42)
d_eff_25 = mixture._calculate_d_efficiency(design_25, "quadratic")
i_eff_25 = mixture._calculate_i_efficiency(design_25, "quadratic")
print(f"  D-efficiency: {d_eff_25:.6f}")
print(f"  I-efficiency: {i_eff_25:.2f}")

# Test with 30 runs (very comfortable)
print(f"\nTesting with 30 runs (very comfortable for 15 parameters):")
design_30 = mixture.generate_d_optimal_mixture(30, "quadratic", random_seed=42)
d_eff_30 = mixture._calculate_d_efficiency(design_30, "quadratic")
i_eff_30 = mixture._calculate_i_efficiency(design_30, "quadratic")
print(f"  D-efficiency: {d_eff_30:.6f}")
print(f"  I-efficiency: {i_eff_30:.2f}")

print(f"\n=== Summary ===")
print(f"20 runs: D-eff = {d_eff_20:.6f}, I-eff = {i_eff_20:.2f}")
print(f"25 runs: D-eff = {d_eff_25:.6f}, I-eff = {i_eff_25:.2f}")
print(f"30 runs: D-eff = {d_eff_30:.6f}, I-eff = {i_eff_30:.2f}")

# Check if D-efficiency improves with more runs
if d_eff_30 > d_eff_25 > d_eff_20:
    print("✅ D-efficiency improves with more runs!")
elif d_eff_30 > 0.01 and d_eff_25 > 0.01 and d_eff_20 > 0.01:
    print("✅ D-efficiency is reasonable for all run counts!")
else:
    print("❌ D-efficiency still problematic")

# Check if I-efficiency improves (decreases) with more runs  
if i_eff_30 < i_eff_25 < i_eff_20:
    print("✅ I-efficiency improves (decreases) with more runs!")
else:
    print("❌ I-efficiency pattern unexpected")
