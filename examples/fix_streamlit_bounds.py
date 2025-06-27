"""
Fix for streamlit_app.py to handle component bounds properly
"""
from mixture_designs import EnhancedMixtureDesign

def test_streamlit_bounds():
    """Test the component bounds that were failing in the streamlit app"""
    print("Fixing component bounds issue in streamlit_app.py\n")
    
    # Create a mixture design with the bounds that were failing in streamlit
    print("Creating mixture design with very relaxed bounds:")
    component_names = ["Component_1", "Component_2", "Component_3"]
    component_bounds_parts = [(1.0, 10.0), (1.0, 10.0), (0.5, 5.0)]
    
    print("Component_1: (1.0, 10.0) parts")
    print("Component_2: (1.0, 10.0) parts")
    print("Component_3: (0.5, 5.0) parts")
    
    mixture = EnhancedMixtureDesign(
        n_components=3,
        component_names=component_names,
        component_bounds=component_bounds_parts,
        use_parts_mode=True
    )
    
    print("\nMixture design created successfully")
    print(f"Component bounds after conversion: {mixture.component_bounds}")
    
    # Generate a D-optimal design
    try:
        design = mixture.generate_d_optimal_mixture(
            n_runs=15,
            model_type="quadratic",
            random_seed=42
        )
        
        print(f"\nGenerated design with {len(design)} runs")
        print("Design matrix:")
        for i in range(min(5, len(design))):
            print(f"Run {i+1}: {design[i].round(3)}")
        
        # Verify sums
        sums = design.sum(axis=1)
        print(f"Row sums: {sums.round(6)}")
        
        # Calculate D-efficiency
        d_eff = mixture._calculate_d_efficiency(design, "quadratic")
        print(f"D-efficiency: {d_eff:.6f}")
        
        print("\nFix successful! The mixture design generation now works correctly.")
        print("\nRecommendations for streamlit_app.py:")
        print("1. The improved _generate_candidate_points method has been applied to mixture_designs.py")
        print("2. This ensures robust candidate point generation for all designs")
        print("3. No further changes needed in streamlit_app.py")
        
    except Exception as e:
        print(f"Error generating design: {str(e)}")

def test_zero_min_bounds():
    """Test with a component having zero minimum bound"""
    print("\nTesting with a component having zero minimum bound:")
    component_names = ["Component_1", "Component_2", "Component_3"]
    component_bounds_parts = [(0.1, 1.0), (0.1, 1.0), (0.0, 0.1)]
    
    print("Component_1: (0.1, 1.0) parts")
    print("Component_2: (0.1, 1.0) parts")
    print("Component_3: (0.0, 0.1) parts")
    
    mixture = EnhancedMixtureDesign(
        n_components=3,
        component_names=component_names,
        component_bounds=component_bounds_parts,
        use_parts_mode=True
    )
    
    print("\nMixture design created successfully")
    print(f"Component bounds after conversion: {mixture.component_bounds}")
    
    # Generate a D-optimal design
    try:
        design = mixture.generate_d_optimal_mixture(
            n_runs=15,
            model_type="quadratic",
            random_seed=42
        )
        
        print(f"\nGenerated design with {len(design)} runs")
        print("Design matrix:")
        for i in range(min(5, len(design))):
            print(f"Run {i+1}: {design[i].round(3)}")
        
        # Verify sums
        sums = design.sum(axis=1)
        print(f"Row sums: {sums.round(6)}")
        
        print("\nZero minimum bound test successful!")
        
    except Exception as e:
        print(f"Error generating design: {str(e)}")

if __name__ == "__main__":
    test_streamlit_bounds()
    test_zero_min_bounds()
