"""
Demonstration of flexible run number control in mixture designs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enhanced_mixture_designs import EnhancedMixtureDesign

def main():
    # Setup
    component_names = ['Component A', 'Component B', 'Component C']
    component_bounds = [(0.2, 0.6), (0.2, 0.6), (0.2, 0.6)]
    
    enhanced_design = EnhancedMixtureDesign(
        n_components=3, 
        component_names=component_names,
        component_bounds=component_bounds
    )
    
    print("=== FLEXIBLE MIXTURE DESIGN RUN NUMBERS ===\n")
    
    # 1. Show default runs for different design types
    print("1. DEFAULT NUMBER OF RUNS FOR EACH DESIGN TYPE:")
    print("=" * 60)
    
    design_info = []
    for design_type in ["simplex-lattice", "simplex-centroid", "d-optimal", "i-optimal"]:
        design = enhanced_design.generate_mixture_design(
            design_type=design_type,
            n_runs=None,  # Use default
            model_type="quadratic",
            random_seed=42
        )
        design_info.append({
            'Design Type': design_type,
            'Default Runs': len(design),
            'Formula/Logic': get_formula_description(design_type, enhanced_design)
        })
    
    print(pd.DataFrame(design_info).to_string(index=False))
    
    # 2. Show how to set exact number of runs
    print("\n\n2. SETTING EXACT NUMBER OF RUNS:")
    print("=" * 60)
    
    target_runs = [10, 15, 20, 25, 30]
    
    for n_runs in target_runs:
        print(f"\nTarget: {n_runs} runs")
        print("-" * 30)
        
        # Generate D-optimal with exact runs
        d_opt = enhanced_design.generate_mixture_design(
            design_type="d-optimal",
            n_runs=n_runs,
            model_type="quadratic",
            random_seed=42
        )
        
        # Generate adjusted simplex lattice
        lattice = enhanced_design.generate_mixture_design(
            design_type="simplex-lattice",
            n_runs=n_runs,
            model_type="quadratic",
            augment_strategy="d-optimal",
            random_seed=42
        )
        
        print(f"D-optimal: {len(d_opt)} runs (exact)")
        print(f"Adjusted Simplex Lattice: {len(lattice)} runs")
    
    # 3. Show augmentation strategies
    print("\n\n3. AUGMENTATION STRATEGIES FOR SIMPLEX LATTICE:")
    print("=" * 60)
    
    # Generate base simplex lattice
    base_degree = 2
    base_design = enhanced_design.generate_simplex_lattice(degree=base_degree)
    base_runs = len(base_design)
    
    print(f"Base Simplex Lattice (degree {base_degree}): {base_runs} runs")
    print(f"Formula: N = C({enhanced_design.n_components}+{base_degree}-1, {base_degree}) = {base_runs}")
    
    # Show different augmentation strategies
    target = 20
    print(f"\nAugmenting to {target} runs using different strategies:")
    print("-" * 50)
    
    strategies = ["centroid", "replicate", "d-optimal", "space-filling"]
    results = []
    
    for strategy in strategies:
        augmented = enhanced_design.generate_mixture_design(
            design_type="simplex-lattice",
            n_runs=target,
            augment_strategy=strategy,
            random_seed=42
        )
        
        eval_result = enhanced_design.evaluate_mixture_design(augmented, "quadratic")
        
        results.append({
            'Strategy': strategy,
            'Actual Runs': len(augmented),
            'D-efficiency': f"{eval_result['d_efficiency']:.4f}",
            'Description': get_strategy_description(strategy)
        })
    
    print(pd.DataFrame(results).to_string(index=False))
    
    # 4. Show reduction strategies
    print("\n\n4. REDUCTION STRATEGIES FOR SIMPLEX LATTICE:")
    print("=" * 60)
    
    # Generate larger base design
    base_degree = 4
    base_design = enhanced_design.generate_simplex_lattice(degree=base_degree)
    base_runs = len(base_design)
    
    print(f"Base Simplex Lattice (degree {base_degree}): {base_runs} runs")
    
    target = 15
    print(f"\nReducing to {target} runs using different strategies:")
    print("-" * 50)
    
    reduction_strategies = ["subset", "d-optimal"]
    results = []
    
    for strategy in reduction_strategies:
        reduced = enhanced_design.generate_mixture_design(
            design_type="simplex-lattice",
            n_runs=target,
            augment_strategy=strategy,
            random_seed=42
        )
        
        eval_result = enhanced_design.evaluate_mixture_design(reduced, "quadratic")
        
        results.append({
            'Strategy': strategy,
            'Actual Runs': len(reduced),
            'D-efficiency': f"{eval_result['d_efficiency']:.4f}",
            'Description': get_reduction_description(strategy)
        })
    
    print(pd.DataFrame(results).to_string(index=False))
    
    # 5. Recommended runs for different models
    print("\n\n5. RECOMMENDED NUMBER OF RUNS BY MODEL COMPLEXITY:")
    print("=" * 60)
    
    model_recommendations = []
    
    for model_type in ["linear", "quadratic", "cubic"]:
        min_runs = enhanced_design._get_minimum_runs(model_type)
        recommended_1x = min_runs
        recommended_1_5x = int(min_runs * 1.5)
        recommended_2x = min_runs * 2
        
        model_recommendations.append({
            'Model Type': model_type,
            'Minimum Runs': min_runs,
            'Recommended (1x)': recommended_1x,
            'Recommended (1.5x)': recommended_1_5x,
            'Recommended (2x)': recommended_2x
        })
    
    print(pd.DataFrame(model_recommendations).to_string(index=False))
    
    # 6. Practical example: Finding optimal number of runs
    print("\n\n6. FINDING OPTIMAL NUMBER OF RUNS:")
    print("=" * 60)
    
    print("Evaluating D-efficiency for different run numbers...")
    
    run_numbers = range(6, 31, 3)
    efficiencies = []
    
    for n_runs in run_numbers:
        design = enhanced_design.generate_mixture_design(
            design_type="d-optimal",
            n_runs=n_runs,
            model_type="quadratic",
            random_seed=42
        )
        
        eval_result = enhanced_design.evaluate_mixture_design(design, "quadratic")
        efficiencies.append(eval_result['d_efficiency'])
    
    # Plot efficiency vs runs
    plt.figure(figsize=(10, 6))
    plt.plot(run_numbers, efficiencies, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Runs', fontsize=12)
    plt.ylabel('D-efficiency', fontsize=12)
    plt.title('D-efficiency vs Number of Runs for Quadratic Model', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add reference lines
    min_runs = enhanced_design._get_minimum_runs("quadratic")
    plt.axvline(x=min_runs, color='r', linestyle='--', label=f'Minimum runs ({min_runs})')
    plt.axvline(x=int(min_runs * 1.5), color='g', linestyle='--', label=f'Recommended ({int(min_runs * 1.5)})')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('d_efficiency_vs_runs.png', dpi=150)
    plt.close()
    
    print("\nPlot saved as 'd_efficiency_vs_runs.png'")
    
    # Summary table
    efficiency_df = pd.DataFrame({
        'Runs': run_numbers,
        'D-efficiency': efficiencies
    })
    
    print("\nEfficiency Summary:")
    print(efficiency_df.to_string(index=False))
    
    # 7. Special cases
    print("\n\n7. SPECIAL CASES:")
    print("=" * 60)
    
    # Case 1: Very few runs
    print("\nCase 1: Minimal design (5 runs for quadratic model)")
    minimal_design = enhanced_design.generate_mixture_design(
        design_type="d-optimal",
        n_runs=5,
        model_type="quadratic",
        random_seed=42
    )
    print(f"Generated: {len(minimal_design)} runs")
    print("Note: This is below the recommended minimum for a quadratic model")
    
    # Case 2: Fixed components with custom runs
    print("\nCase 2: Fixed component with custom runs")
    fixed_design = EnhancedMixtureDesign(
        n_components=3,
        component_names=component_names,
        component_bounds=component_bounds,
        fixed_components={'Component C': 0.3}
    )
    
    custom_fixed = fixed_design.generate_mixture_design(
        design_type="d-optimal",
        n_runs=12,
        model_type="quadratic",
        random_seed=42
    )
    print(f"Generated: {len(custom_fixed)} runs with Component C fixed at 0.3")
    
    print("\n=== DEMONSTRATION COMPLETE ===")


def get_formula_description(design_type, enhanced_design):
    """Get formula description for design type"""
    n = enhanced_design.n_components
    
    if design_type == "simplex-lattice":
        return "N = C(q+m-1, m) where m=degree"
    elif design_type == "simplex-centroid":
        return f"N = 2^{n} - 1 = {2**n - 1}"
    elif design_type in ["d-optimal", "i-optimal"]:
        return "User-specified or 2×components"
    else:
        return "Custom algorithm"


def get_strategy_description(strategy):
    """Get description of augmentation strategy"""
    descriptions = {
        "centroid": "Add centroids of point subsets",
        "replicate": "Replicate important points",
        "d-optimal": "Add D-optimal points",
        "space-filling": "Add space-filling points"
    }
    return descriptions.get(strategy, "Unknown strategy")


def get_reduction_description(strategy):
    """Get description of reduction strategy"""
    descriptions = {
        "subset": "Select diverse subset",
        "d-optimal": "Select D-optimal subset"
    }
    return descriptions.get(strategy, "Unknown strategy")


if __name__ == "__main__":
    main()
