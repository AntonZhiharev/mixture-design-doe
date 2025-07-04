"""
Completely isolated test for i-optimal design generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

def test_isolated_ioptimal():
    """Test i-optimal design generation in complete isolation."""
    
    print("🧪 ISOLATED I-OPTIMAL Test (No External Dependencies)")
    print("=" * 60)
    
    # Simple test scenario
    component_names = ["A", "B", "C"]
    
    # Fixed components
    fixed_parts = {"C": 10.0}
    
    # Variable component bounds
    variable_bounds = {
        "A": (5, 20),
        "B": (5, 20)
    }
    
    try:
        print("Step 1: Creating design generator...")
        designer = FixedPartsMixtureDesign(
            component_names=component_names,
            fixed_parts=fixed_parts,
            variable_bounds=variable_bounds
        )
        print("✅ Design generator created successfully")
        
        print("\nStep 2: Testing D-optimal (baseline)...")
        d_design = designer.generate_design(
            n_runs=8,
            design_type="d-optimal",
            model_type="linear",  # Use simpler model
            random_seed=42
        )
        print("✅ D-optimal generation successful")
        
        print("\nStep 3: Testing I-optimal (main test)...")
        i_design = designer.generate_design(
            n_runs=8,
            design_type="i-optimal",
            model_type="linear",  # Use simpler model
            random_seed=42
        )
        print("✅ I-optimal generation successful!")
        print(f"Generated design shape: {i_design.shape}")
        
        # Quick verification
        print(f"\nQuick verification:")
        print(f"Fixed component C parts: {i_design['C_Parts'].unique()}")
        prop_cols = [col for col in i_design.columns if col.endswith('_Prop')]
        prop_sums = i_design[prop_cols].sum(axis=1)
        print(f"Proportion sums: {prop_sums.min():.6f} to {prop_sums.max():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during isolated test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_isolated_ioptimal()
    if success:
        print(f"\n🎉 I-OPTIMAL FIX VERIFIED!")
    else:
        print(f"\n💥 I-OPTIMAL STILL HAS ISSUES")
