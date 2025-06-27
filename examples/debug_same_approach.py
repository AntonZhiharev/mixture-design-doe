"""
Debug why direct optimization doesn't produce the same results as Regular Optimal Design
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
    
    n_components = 3
    n_runs = 15
    random_seed = 42
    
    print("=" * 70)
    print("DEBUGGING THE ISSUE")
    print("=" * 70)
    
    # 1. REGULAR OPTIMAL DESIGN - FIRST CALL
    print("\n1. REGULAR OPTIMAL DESIGN - FIRST CALL")
    print("-" * 50)
    
    np.random.seed(random_seed)  # Reset seed
    regular_doe_1 = OptimalDOE(
        n_factors=n_components,
        factor_ranges=component_bounds
    )
    
    regular_design_1 = regular_doe_1.generate_d_optimal(
        n_runs=n_runs,
        model_order=1,
        random_seed=random_seed
    )
    
    regular_d_eff_1 = regular_doe_1.d_efficiency(regular_design_1, model_order=1)
    print(f"First call D-efficiency: {regular_d_eff_1:.6f}")
    print(f"First call design hash: {hash(str(regular_design_1))}")
    
    # 2. REGULAR OPTIMAL DESIGN - SECOND CALL (SHOULD BE IDENTICAL)
    print("\n2. REGULAR OPTIMAL DESIGN - SECOND CALL")
    print("-" * 50)
    
    np.random.seed(random_seed)  # Reset seed again
    regular_doe_2 = OptimalDOE(
        n_factors=n_components,
        factor_ranges=component_bounds
    )
    
    regular_design_2 = regular_doe_2.generate_d_optimal(
        n_runs=n_runs,
        model_order=1,
        random_seed=random_seed
    )
    
    regular_d_eff_2 = regular_doe_2.d_efficiency(regular_design_2, model_order=1)
    print(f"Second call D-efficiency: {regular_d_eff_2:.6f}")
    print(f"Second call design hash: {hash(str(regular_design_2))}")
    print(f"Designs identical: {np.allclose(regular_design_1, regular_design_2)}")
    
    # 3. MIXTURE DESIGN CALL - SAME EXACT PARAMETERS
    print("\n3. MIXTURE DESIGN CALL")
    print("-" * 50)
    
    # Reset random seed before mixture call
    np.random.seed(random_seed)
    
    # Create mixture design object
    mixture_design = OptimizedMixtureDesign(
        n_components=n_components,
        component_names=["A", "B", "C"],
        component_bounds=component_bounds,
        use_parts_mode=True,
        direct_optimization=True
    )
    
    # This should call the exact same Regular DOE algorithm
    mixture_result = mixture_design.generate_d_optimal(
        n_runs=n_runs,
        model_type="linear",
        random_seed=random_seed
    )
    
    mixture_parts = mixture_design.get_parts_design()
    mixture_d_eff = regular_doe_1.d_efficiency(mixture_parts, model_order=1)
    
    print(f"Mixture D-efficiency: {mixture_d_eff:.6f}")
    print(f"Mixture design hash: {hash(str(mixture_parts))}")
    print(f"Mixture vs Regular1 identical: {np.allclose(mixture_parts, regular_design_1)}")
    print(f"Mixture vs Regular2 identical: {np.allclose(mixture_parts, regular_design_2)}")
    
    # 4. DEBUG: DIRECT CALL INSIDE MIXTURE CLASS
    print("\n4. DEBUG: DIRECT CALL INSIDE MIXTURE CLASS")
    print("-" * 50)
    
    # Reset seed
    np.random.seed(random_seed)
    
    # Create Regular DOE exactly like in the mixture class
    debug_regular_doe = OptimalDOE(
        n_factors=n_components,
        factor_ranges=component_bounds
    )
    
    debug_design = debug_regular_doe.generate_d_optimal(
        n_runs=n_runs,
        model_order=1,
        max_iter=1000,  # Default value
        random_seed=random_seed
    )
    
    debug_d_eff = debug_regular_doe.d_efficiency(debug_design, model_order=1)
    print(f"Debug D-efficiency: {debug_d_eff:.6f}")
    print(f"Debug design hash: {hash(str(debug_design))}")
    print(f"Debug vs Regular1 identical: {np.allclose(debug_design, regular_design_1)}")
    
    # 5. SUMMARY
    print("\n5. SUMMARY")
    print("-" * 50)
    print(f"Regular 1:  {regular_d_eff_1:.6f}")
    print(f"Regular 2:  {regular_d_eff_2:.6f}")
    print(f"Mixture:    {mixture_d_eff:.6f}")
    print(f"Debug:      {debug_d_eff:.6f}")
    print("\nAll should be identical if we're truly using the same approach!")

if __name__ == "__main__":
    main()
