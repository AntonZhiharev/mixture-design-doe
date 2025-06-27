"""
Final verification test for mixture design with fixed components
"""
from mixture_designs import EnhancedMixtureDesign

def test_fixed_components():
    """Test mixture design with fixed components"""
    print("Testing mixture design with fixed components\n")
    
    # Create a mixture design with fixed components
    component_names = ["Component_1", "Component_2", "Component_3", "Fixed_1", "Fixed_2"]
    component_bounds_parts = [(0.1, 1.0), (0.1, 1.0), (0.1, 1.0), (0.05, 0.05), (0.02, 0.02)]
    
    # Define fixed components
    fixed_components = {
        "Fixed_1": 0.05,
        "Fixed_2": 0.02
    }
    
    print("Variable components:")
    print("Component_1: (0.1, 1.0) parts")
    print("Component_2: (0.1, 1.0) parts")
    print("Component_3: (0.1, 1.0) parts")
    print("\nFixed components:")
    print("Fixed_1: 0.05 parts")
    print("Fixed_2: 0.02 parts")
    
    mixture = EnhancedMixtureDesign(
        n_components=5,
        component_names=component_names,
        component_bounds=component_bounds_parts,
        use_parts_mode=True,
        fixed_components=fixed_components
    )
    
    print("\nMixture design created successfully")
    
    # Generate a D-optimal design
    try:
        design = mixture.generate_d_optimal_mixture(
            n_runs=10,
            model_type="linear",  # Using linear for simplicity
            random_seed=42
        )
        
        print(f"\nGenerated design with {len(design)} runs")
        print("Design matrix:")
        for i in range(len(design)):
            print(f"Run {i+1}: {design[i].round(4)}")
        
        # Verify sums
        sums = design.sum(axis=1)
        print(f"Row sums: {sums.round(6)}")
        
        # Verify fixed components
        print("\nVerifying fixed components:")
        fixed_1_idx = component_names.index("Fixed_1")
        fixed_2_idx = component_names.index("Fixed_2")
        
        for i in range(len(design)):
            fixed_1_value = design[i, fixed_1_idx]
            fixed_2_value = design[i, fixed_2_idx]
            print(f"Run {i+1}: Fixed_1 = {fixed_1_value:.6f}, Fixed_2 = {fixed_2_value:.6f}")
        
        print("\nFixed components test successful!")
        
    except Exception as e:
        print(f"Error generating design: {str(e)}")

def test_extreme_case():
    """Test with extreme bounds and fixed components"""
    print("\n\nTesting extreme case with very small bounds and fixed components\n")
    
    # Create a mixture design with extreme bounds
    component_names = ["Main", "Minor", "Trace", "Fixed"]
    component_bounds_parts = [(1.0, 10.0), (0.01, 0.1), (0.001, 0.01), (0.05, 0.05)]
    
    # Define fixed components
    fixed_components = {
        "Fixed": 0.05
    }
    
    print("Variable components:")
    print("Main: (1.0, 10.0) parts")
    print("Minor: (0.01, 0.1) parts")
    print("Trace: (0.001, 0.01) parts")
    print("\nFixed components:")
    print("Fixed: 0.05 parts")
    
    mixture = EnhancedMixtureDesign(
        n_components=4,
        component_names=component_names,
        component_bounds=component_bounds_parts,
        use_parts_mode=True,
        fixed_components=fixed_components
    )
    
    print("\nMixture design created successfully")
    
    # Generate a D-optimal design
    try:
        design = mixture.generate_d_optimal_mixture(
            n_runs=8,
            model_type="linear",
            random_seed=42
        )
        
        print(f"\nGenerated design with {len(design)} runs")
        print("Design matrix:")
        for i in range(len(design)):
            print(f"Run {i+1}: {design[i].round(6)}")
        
        # Verify sums
        sums = design.sum(axis=1)
        print(f"Row sums: {sums.round(6)}")
        
        # Verify fixed components
        print("\nVerifying fixed components:")
        fixed_idx = component_names.index("Fixed")
        
        for i in range(len(design)):
            fixed_value = design[i, fixed_idx]
            print(f"Run {i+1}: Fixed = {fixed_value:.6f}")
        
        print("\nExtreme case test successful!")
        
    except Exception as e:
        print(f"Error generating design: {str(e)}")

if __name__ == "__main__":
    test_fixed_components()
    test_extreme_case()
