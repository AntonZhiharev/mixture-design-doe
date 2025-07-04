"""
Test i-optimal design generation to reproduce and fix the error
"""

import numpy as np
from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

def test_ioptimal_error():
    """Test i-optimal design generation that's causing the error."""
    
    print("🧪 Testing I-OPTIMAL Design Generation Error")
    print("=" * 50)
    
    # Test scenario: Paint formulation with fixed components
    component_names = ["Pigment_A", "Pigment_B", "Solvent", "Binder", "Additive"]
    
    # Fixed components (constant parts)
    fixed_parts = {
        "Binder": 15.0,      # Always 15 parts binder
        "Additive": 2.5      # Always 2.5 parts additive
    }
    
    # Variable component bounds (parts)
    variable_bounds = {
        "Pigment_A": (5, 25),    # 5-25 parts
        "Pigment_B": (5, 25),    # 5-25 parts  
        "Solvent": (10, 40)      # 10-40 parts
    }
    
    # Create design generator
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    print("Testing D-optimal (should work):")
    try:
        d_design = designer.generate_design(
            n_runs=10,
            design_type="d-optimal",
            model_type="quadratic",
            random_seed=42
        )
        print("✅ D-optimal generation successful!")
    except Exception as e:
        print(f"❌ D-optimal generation failed: {e}")
    
    print("\nTesting I-optimal (should now work):")
    try:
        i_design = designer.generate_design(
            n_runs=10,
            design_type="i-optimal",
            model_type="quadratic",
            random_seed=42
        )
        print("✅ I-optimal generation successful!")
        print(f"Generated design shape: {i_design.shape}")
        print("\nFirst few rows of I-optimal design:")
        print(i_design.head())
        
        # Verify design properties
        print(f"\nI-optimal design verification:")
        parts_cols = [col for col in i_design.columns if col.endswith('_Parts')]
        prop_cols = [col for col in i_design.columns if col.endswith('_Prop')]
        
        # Check fixed components
        print(f"Binder parts: {i_design['Binder_Parts'].unique()}")
        print(f"Additive parts: {i_design['Additive_Parts'].unique()}")
        
        # Check proportions sum to 1
        prop_sums = i_design[prop_cols].sum(axis=1)
        print(f"Proportion sums: {prop_sums.min():.6f} to {prop_sums.max():.6f}")
        
    except Exception as e:
        print(f"❌ I-optimal generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ioptimal_error()
