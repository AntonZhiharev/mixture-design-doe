"""
Test to identify the specific D-efficiency issue
"""

from mixture_designs import MixtureDesign
import numpy as np

# Test the D-efficiency issue
print("=== Testing D-Efficiency Issue ===\n")

# Create same setup as in your Streamlit app
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']
component_bounds = [(0, 1), (0, 1), (0, 0.1), (0, 0.1), (0, 0.1), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)]
fixed_components = {'Fixed_1': 0.05, 'Fixed_2': 0.05, 'Fixed_3': 0.05}

# Create mixture design with parts mode
mixture = MixtureDesign(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    use_parts_mode=True,
    fixed_components=fixed_components
)

print("Testing D-optimal designs with different run counts:")
print("-" * 60)

# Test different run counts
run_counts = [15, 20, 25, 30]
results = []

for n_runs in run_counts:
    print(f"\nGenerating D-optimal design with {n_runs} runs...")
    design = mixture.generate_d_optimal_mixture(n_runs=n_runs, model_type="quadratic", random_seed=42)
    eval_results = mixture.evaluate_mixture_design(design, "quadratic")
    
    d_eff = eval_results['d_efficiency']
    i_eff = eval_results['i_efficiency']
    
    print(f"  Actual runs: {len(design)}")
    print(f"  D-efficiency: {d_eff}")
    print(f"  I-efficiency: {i_eff}")
    
    results.append({
        'n_runs': n_runs,
        'actual_runs': len(design),
        'd_efficiency': d_eff,
        'i_efficiency': i_eff
    })

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
for result in results:
    print(f"Runs: {result['n_runs']:2d} → D-eff: {result['d_efficiency']:10.6f}, I-eff: {result['i_efficiency']:10.6f}")

# Check if D-efficiencies are identical
d_efficiencies = [r['d_efficiency'] for r in results]
if len(set(d_efficiencies)) == 1:
    print(f"\n❌ PROBLEM: All D-efficiencies are identical: {d_efficiencies[0]}")
else:
    print(f"\n✅ SUCCESS: D-efficiencies are different: {d_efficiencies}")

# Check if we're getting the extreme values
if any(abs(d) > 100 for d in d_efficiencies):
    print("❌ PROBLEM: Extreme D-efficiency values detected")
    
i_efficiencies = [r['i_efficiency'] for r in results]
if any(abs(i) > 100 for i in i_efficiencies):
    print("❌ PROBLEM: Extreme I-efficiency values detected")
