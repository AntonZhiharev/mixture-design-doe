"""
Test to verify that Direct Optimization produces EXACTLY the same results as Regular Optimal Design
"""

import numpy as np
from base_doe import OptimalDOE
from mixture_design_optimization import OptimizedMixtureDesign

def main():
    # Component bounds
    component_bounds = [
        (0.0, 1.0),  # Component A: 0 to 1 parts
        (0.0, 1.0),  # Component B: 0 to 1 parts  
        (0.0, 0.1)   # Component C: 0 to 0.1 parts
    ]
    
    component_names = ["A", "B", "C"]
    n_components = len(component_names)
    n_runs = 15
    
    print("=" * 70)
    print("VERIFYING EXACT SAME APPROACH")
    print("=" * 70)
    
    # 1. REGULAR OPTIMAL DESIGN
    print("\n1. REGULAR OPTIMAL DESIGN")
    print("-" * 50)
    
    regular_doe = OptimalDOE(
        n_factors=n_components,
        factor_ranges=component_bounds
    )
    
    regular_design = regular_doe.generate_d_optimal(
        n_runs=n_runs,
        model_order=1,
        random_seed=42
    )
    
    regular_d_eff = regular_doe.d_efficiency(regular_design, model_order=1)
    print(f"Regular D-efficiency: {regular_d_eff:.6f}")
    print(f"Regular design:\n{regular_design}")
    
    # 2. DIRECT MIXTURE APPROACH - SHOULD BE IDENTICAL
    print("\n2. DIRECT MIXTURE APPROACH")
    print("-" * 50)
    
    # Let's manually do what the direct optimization should do
    direct_mixture = OptimizedMixtureDesign(
        n_components=n_components,
        component_names=component_names,
        component_bounds=component_bounds,
        use_parts_mode=True,
        direct_optimization=True
    )
    
    # Generate using direct optimization
    direct_design = direct_mixture.generate_d_optimal(
        n_runs=n_runs,
        model_type="linear",
        random_seed=42
    )
    
    # Get the parts design
    parts_design = direct_mixture.get_parts_design()
    parts_d_eff = regular_doe.d_efficiency(parts_design, model_order=1)
    
    print(f"Direct D-efficiency (parts): {parts_d_eff:.6f}")
    print(f"Parts design:\n{parts_design}")
    
    # 3. COMPARISON
    print("\n3. COMPARISON")
    print("-" * 50)
    
    print(f"Regular D-efficiency:    {regular_d_eff:.6f}")
    print(f"Direct D-efficiency:     {parts_d_eff:.6f}")
    print(f"Designs are identical:   {np.allclose(regular_design, parts_design, atol=1e-10)}")
    print(f"D-efficiencies match:    {abs(regular_d_eff - parts_d_eff) < 1e-10}")
    
    if not np.allclose(regular_design, parts_design, atol=1e-10):
        print("\nDIFFERENCES FOUND:")
        print("Regular design points that differ:")
        for i, (reg_point, direct_point) in enumerate(zip(regular_design, parts_design)):
            if not np.allclose(reg_point, direct_point, atol=1e-10):
                print(f"Point {i}: Regular {reg_point} vs Direct {direct_point}")
    else:
        print("\n✅ SUCCESS: Direct optimization produces EXACTLY the same results as Regular Optimal Design!")

if __name__ == "__main__":
    main()
