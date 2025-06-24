"""
Debug script to identify the exact issue with fixed space solution
"""

import numpy as np
from mixture_designs import MixtureDesign

def test_fixed_space_issue():
    print("=== DEBUGGING FIXED SPACE SOLUTION ===")
    
    # Simple test case that matches the screenshot issue
    component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']
    component_bounds_parts = [
        (0.1, 1.0),    # Component_1 (variable)
        (0.1, 1.0),    # Component_2 (variable) 
        (0.1, 1.0),    # Component_3 (variable)
        (0.1, 1.0),    # Component_4 (variable)
        (0.1, 1.0),    # Component_5 (variable)
        (0.05, 0.05),  # Fixed_1 (fixed)
        (0.05, 0.05),  # Fixed_2 (fixed)
        (0.05, 0.05)   # Fixed_3 (fixed)
    ]
    
    fixed_parts = {
        'Fixed_1': 0.05,
        'Fixed_2': 0.05, 
        'Fixed_3': 0.05
    }
    
    try:
        # Create mixture design
        mixture = MixtureDesign(
            n_components=8,
            component_names=component_names,
            component_bounds=component_bounds_parts,
            use_parts_mode=True,
            fixed_components=fixed_parts
        )
        
        print("\n1. After initialization:")
        print(f"Fixed components cleared: {mixture.fixed_components}")
        if hasattr(mixture, 'original_fixed_components_proportions'):
            print(f"Original fixed proportions stored: {mixture.original_fixed_components_proportions}")
        
        # Generate a simple design
        print("\n2. Generating D-optimal design...")
        design = mixture.generate_d_optimal_mixture(n_runs=1, model_type='linear', random_seed=42)
        
        print(f"\n3. Generated design shape: {design.shape}")
        print("Design values:")
        for i, comp_name in enumerate(component_names):
            print(f"  {comp_name}: {design[0, i]:.6f}")
        print(f"  Sum: {np.sum(design[0]):.6f}")
        
        # Check what's happening in post-processing
        print("\n4. Checking post-processing variables...")
        if hasattr(mixture, 'original_fixed_components_proportions'):
            print("✅ Has original_fixed_components_proportions")
            for name, value in mixture.original_fixed_components_proportions.items():
                print(f"  {name}: {value:.6f}")
        else:
            print("❌ Missing original_fixed_components_proportions")
            
        if hasattr(mixture, 'original_fixed_components'):
            print("✅ Has original_fixed_components")
            for name, value in mixture.original_fixed_components.items():
                print(f"  {name}: {value:.6f}")
        else:
            print("❌ Missing original_fixed_components")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_space_issue()
