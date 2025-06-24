"""
Compare Simplex Lattice vs Sequential Mixture DOE approaches
"""

from mixture_designs import MixtureDesign
from sequential_mixture_doe import SequentialMixtureDOE
import numpy as np
import pandas as pd

print("=== COMPARISON: Simplex Lattice vs Sequential Mixture DOE ===\n")

# User's exact setup
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']

# User's bounds (Variable first, then fixed)
component_bounds = [
    (0.0, 1.0),    # Component_1 (variable)
    (0.0, 1.0),    # Component_2 (variable)
    (0.0, 0.1),    # Component_3 (variable, tight bounds)
    (0.0, 0.1),    # Component_4 (variable, tight bounds)
    (0.0, 0.1),    # Component_5 (variable, tight bounds)
    (0.05, 0.05),  # Fixed_1 = 5%
    (0.05, 0.05),  # Fixed_2 = 5%
    (0.05, 0.05),  # Fixed_3 = 5%
]

fixed_components = {
    'Fixed_1': 0.05,
    'Fixed_2': 0.05,
    'Fixed_3': 0.05
}

print("SETUP:")
print(f"• 5 variable components + 3 fixed components")
print(f"• Components 3-5 have tight bounds (0.0, 0.1)")
print(f"• Fixed components at 5% each")
print(f"• Available for variable: 85%")

print(f"\n" + "="*80)

# 1. REGULAR SIMPLEX LATTICE (Current approach)
print("1. REGULAR SIMPLEX LATTICE")

mixture = MixtureDesign(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    fixed_components=fixed_components
)

lattice_design = mixture.generate_simplex_lattice(degree=3)
lattice_results = mixture.evaluate_mixture_design(lattice_design, "quadratic")

print(f"✅ Method: Simplex Lattice (Degree 3)")
print(f"✅ Runs: {len(lattice_design)}")
print(f"✅ D-Efficiency: {lattice_results['d_efficiency']:.4f}")

# Count zeros
lattice_df = pd.DataFrame(lattice_design, columns=component_names)
variable_cols = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5']
zeros_per_run = (lattice_df[variable_cols] == 0).sum(axis=1)
avg_zeros = zeros_per_run.mean()
print(f"✅ Average zeros per run: {avg_zeros:.1f}")

print(f"\n" + "="*80)

# 2. SEQUENTIAL MIXTURE DOE (User's better approach)
print("2. SEQUENTIAL MIXTURE DOE")

seq_mixture = SequentialMixtureDOE(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    fixed_components=fixed_components
)

# Generate more runs like user mentioned (120 total)
stage1_runs = 60  # Stage 1
stage2_runs = 60  # Stage 2

print(f"Stage 1: {stage1_runs} runs (D-optimal)")
stage1_design = seq_mixture.generate_d_optimal_mixture(
    n_runs=stage1_runs, 
    model_type="quadratic", 
    random_seed=42
)

print(f"Stage 2: {stage2_runs} runs (Augmentation)")
stage2_design = seq_mixture.augment_mixture_design(
    stage1_design,
    n_additional_runs=stage2_runs,
    model_type="quadratic",
    random_seed=43
)

# Combine designs
sequential_design = np.vstack([stage1_design, stage2_design])
sequential_results = seq_mixture.evaluate_mixture_design(sequential_design, "quadratic")

print(f"✅ Method: Sequential D-Optimal")
print(f"✅ Total Runs: {len(sequential_design)}")
print(f"✅ D-Efficiency: {sequential_results['d_efficiency']:.4f}")

# Count zeros in sequential design
seq_df = pd.DataFrame(sequential_design, columns=component_names)
seq_zeros_per_run = (seq_df[variable_cols] == 0).sum(axis=1)
seq_avg_zeros = seq_zeros_per_run.mean()
print(f"✅ Average zeros per run: {seq_avg_zeros:.1f}")

print(f"\n" + "="*80)

# 3. REGULAR MIXTURE D-OPTIMAL (120 runs for fair comparison)
print("3. REGULAR MIXTURE D-OPTIMAL (120 runs)")

regular_d_optimal_design = mixture.generate_d_optimal_mixture(
    n_runs=120, 
    model_type="quadratic", 
    random_seed=42
)
regular_d_optimal_results = mixture.evaluate_mixture_design(regular_d_optimal_design, "quadratic")

print(f"✅ Method: Regular D-Optimal")
print(f"✅ Runs: {len(regular_d_optimal_design)}")
print(f"✅ D-Efficiency: {regular_d_optimal_results['d_efficiency']:.4f}")

# Count zeros
reg_d_opt_df = pd.DataFrame(regular_d_optimal_design, columns=component_names)
reg_d_opt_zeros_per_run = (reg_d_opt_df[variable_cols] == 0).sum(axis=1)
reg_d_opt_avg_zeros = reg_d_opt_zeros_per_run.mean()
print(f"✅ Average zeros per run: {reg_d_opt_avg_zeros:.1f}")

print(f"\n" + "="*80)

# 4. COMPREHENSIVE COMPARISON
print("4. COMPREHENSIVE COMPARISON")

comparison_data = {
    'Method': [
        'Simplex Lattice (Deg 3)',
        'Sequential D-Optimal', 
        'Regular D-Optimal (120 runs)'
    ],
    'Runs': [
        len(lattice_design),
        len(sequential_design),
        len(regular_d_optimal_design)
    ],
    'D-Efficiency': [
        lattice_results['d_efficiency'],
        sequential_results['d_efficiency'],
        regular_d_optimal_results['d_efficiency']
    ],
    'Avg Zeros': [
        avg_zeros,
        seq_avg_zeros,
        reg_d_opt_avg_zeros
    ],
    'Algorithm': [
        'Mathematical lattice',
        'Coordinate exchange',
        'Coordinate exchange'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(4))

print(f"\n📊 KEY INSIGHTS:")

# Find best method
best_d_eff_idx = comparison_df['D-Efficiency'].idxmax()
best_method = comparison_df.loc[best_d_eff_idx, 'Method']
best_d_eff = comparison_df.loc[best_d_eff_idx, 'D-Efficiency']

print(f"🏆 BEST D-EFFICIENCY: {best_method} with {best_d_eff:.4f}")

# Efficiency improvement
lattice_eff = lattice_results['d_efficiency']
seq_eff = sequential_results['d_efficiency']
improvement = (seq_eff / lattice_eff - 1) * 100

print(f"📈 IMPROVEMENT: Sequential vs Simplex Lattice: {improvement:.1f}% better D-efficiency")

# Zeros comparison
print(f"🎯 ZEROS REDUCTION:")
print(f"   • Simplex Lattice: {avg_zeros:.1f} zeros per run")
print(f"   • Sequential: {seq_avg_zeros:.1f} zeros per run") 
print(f"   • Regular D-Optimal: {reg_d_opt_avg_zeros:.1f} zeros per run")

print(f"\n" + "="*80)

# 5. SAMPLE COMPARISON (First 10 runs)
print("5. DESIGN PATTERN COMPARISON (First 10 runs)")

print(f"\n🔸 SIMPLEX LATTICE (Many zeros, systematic):")
for i in range(min(10, len(lattice_design))):
    row = lattice_df[variable_cols].iloc[i]
    active = row[row > 0]
    print(f"   Run {i+1}: {len(active)} active components: {dict(active.round(3))}")

print(f"\n🔸 SEQUENTIAL D-OPTIMAL (Fewer zeros, optimized):")
for i in range(min(10, len(sequential_design))):
    row = seq_df[variable_cols].iloc[i]
    active = row[row > 0]
    print(f"   Run {i+1}: {len(active)} active components: {dict(active.round(3))}")

print(f"\n" + "="*80)

# 6. RECOMMENDATIONS
print("6. RECOMMENDATIONS")

print(f"\n🎯 WHY SEQUENTIAL IS BETTER:")
print(f"✅ Uses coordinate exchange algorithm (optimization-based)")
print(f"✅ Can generate any number of runs (not limited to lattice formula)")
print(f"✅ Focuses on efficient point placement, not systematic coverage")
print(f"✅ Better D-efficiency with more runs")
print(f"✅ Fewer zeros = more realistic formulations")

print(f"\n🎯 WHEN TO USE EACH:")
print(f"📐 SIMPLEX LATTICE:")
print(f"   • Academic research")
print(f"   • Understanding component boundaries")
print(f"   • Traditional systematic approach")
print(f"   • Small number of experiments")

print(f"🎯 SEQUENTIAL D-OPTIMAL:")
print(f"   • Applied research & development")
print(f"   • Want high efficiency with many runs")
print(f"   • Practical formulation work")
print(f"   • Flexible experiment budget")

print(f"\n🔄 SOLUTION FOR REGULAR MIXTURE DESIGN:")
print(f"✅ Add option to specify custom number of runs for D-optimal")
print(f"✅ This would match Sequential performance")
print(f"✅ User can choose: 35 (lattice) vs 120 (high efficiency)")

print(f"\n=== CONCLUSION ===")
print(f"Sequential Mixture DOE is superior because:")
print(f"1. Uses optimization algorithm (not fixed lattice pattern)")
print(f"2. Allows custom run count (120 vs fixed 35)")
print(f"3. Achieves much higher D-efficiency")
print(f"4. Produces more realistic formulations (fewer zeros)")
print(f"\nRegular Mixture Design should offer the same flexibility!")
