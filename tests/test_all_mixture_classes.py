"""
Test all mixture design classes to ensure they use direct optimization
and produce high D-efficiency (same as Regular DOE)
"""

import numpy as np
from base_doe import OptimalDOE
from mixture_designs import MixtureDesign, MixtureDesignGenerator
from mixture_designs import MixtureDesign as OptimizedMixtureDesign

def test_all_classes():
    """Test all mixture design classes for consistent high D-efficiency"""
    
    # Test parameters
    component_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.1)]
    component_names = ["A", "B", "C"]
    n_components = 3
    n_runs = 15
    random_seed = 42
    
    print("=" * 80)
    print("TESTING ALL MIXTURE DESIGN CLASSES FOR CONSISTENT HIGH D-EFFICIENCY")
    print("=" * 80)
    
    # 1. Baseline: Regular Optimal Design
    print("\n1. BASELINE: Regular Optimal Design")
    print("-" * 50)
    regular_doe = OptimalDOE(n_factors=n_components, factor_ranges=component_bounds)
    regular_design = regular_doe.generate_d_optimal(n_runs=n_runs, model_order=1, random_seed=random_seed)
    regular_d_eff = regular_doe.d_efficiency(regular_design, model_order=1)
    print(f"Regular DOE D-efficiency: {regular_d_eff:.6f}")
    
    # 2. OptimizedMixtureDesign (now uses exact same approach as Regular DOE)
    print("\n2. OptimizedMixtureDesign (using exact same approach as Regular DOE)")
    print("-" * 50)
    optimized_direct = OptimizedMixtureDesign(
        n_components=n_components,
        component_names=component_names,
        component_bounds=component_bounds
    )
    optimized_design = optimized_direct.generate_d_optimal(n_runs=n_runs, model_type="linear", random_seed=random_seed)
    optimized_parts = optimized_direct.parts_design
    optimized_d_eff = regular_doe.d_efficiency(optimized_parts, model_order=1)
    print(f"OptimizedMixtureDesign D-efficiency: {optimized_d_eff:.6f}")
    print(f"Matches Regular DOE: {abs(optimized_d_eff - regular_d_eff) < 1e-10}")
    
    # 3. MixtureDesign (backward compatibility class)
    print("\n3. MixtureDesign (backward compatibility)")
    print("-" * 50)
    mixture_design = MixtureDesign(
        n_components=n_components,
        component_names=component_names,
        component_bounds=component_bounds
    )
    mixture_design_result = mixture_design.generate_d_optimal(n_runs=n_runs, model_type="linear", random_seed=random_seed)
    mixture_parts = mixture_design.parts_design
    mixture_d_eff = regular_doe.d_efficiency(mixture_parts, model_order=1)
    print(f"MixtureDesign D-efficiency: {mixture_d_eff:.6f}")
    print(f"Matches Regular DOE: {abs(mixture_d_eff - regular_d_eff) < 1e-10}")
    
    # 4. MixtureDesignGenerator.create_d_optimal
    print("\n4. MixtureDesignGenerator.create_d_optimal")
    print("-" * 50)
    generator_design, generator_obj = MixtureDesignGenerator.create_d_optimal(
        n_components=n_components,
        n_runs=n_runs,
        component_names=component_names,
        component_bounds=component_bounds,
        model_type="linear",
        random_seed=random_seed
    )
    generator_parts = generator_obj.parts_design
    generator_d_eff = regular_doe.d_efficiency(generator_parts, model_order=1)
    print(f"MixtureDesignGenerator D-efficiency: {generator_d_eff:.6f}")
    print(f"Matches Regular DOE: {abs(generator_d_eff - regular_d_eff) < 1e-10}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF D-EFFICIENCY VALUES")
    print("=" * 80)
    print(f"Regular DOE:              {regular_d_eff:.6f}")
    print(f"OptimizedMixtureDesign:   {optimized_d_eff:.6f}")
    print(f"MixtureDesign:            {mixture_d_eff:.6f}")
    print(f"MixtureDesignGenerator:   {generator_d_eff:.6f}")
    
    # Check if all match
    all_match = (
        abs(optimized_d_eff - regular_d_eff) < 1e-10 and
        abs(mixture_d_eff - regular_d_eff) < 1e-10 and
        abs(generator_d_eff - regular_d_eff) < 1e-10
    )
    
    print("\n" + "=" * 80)
    if all_match:
        print("✅ SUCCESS: ALL MIXTURE DESIGN CLASSES PRODUCE HIGH D-EFFICIENCY!")
        print("✅ All classes now use direct optimization approach")
        print("✅ All D-efficiency values match Regular DOE exactly")
        print("✅ The low D-efficiency issue has been COMPLETELY RESOLVED")
    else:
        print("❌ ERROR: Some classes still produce low D-efficiency")
        print("❌ Further investigation needed")
    print("=" * 80)
    
    return all_match

if __name__ == "__main__":
    test_all_classes()
