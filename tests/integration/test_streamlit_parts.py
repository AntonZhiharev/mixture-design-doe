"""
Test to verify the fix for "No parts columns found in design!" error
This test simulates the exact scenario that was failing in the Streamlit app
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

def test_streamlit_parts_columns_fix():
    """Test that mimics the exact Streamlit app scenario that was failing."""
    
    print("🧪 TESTING STREAMLIT PARTS COLUMNS FIX")
    print("=" * 60)
    
    # Test scenario: D-optimal with parts mode and fixed components
    component_names = ["A", "B", "C"]
    fixed_components = {"C": 10.0}
    variable_bounds = {"A": (5, 20), "B": (5, 20)}
    
    try:
        print("Step 1: Creating FixedPartsMixtureDesign...")
        fixed_designer = FixedPartsMixtureDesign(
            component_names=component_names,
            fixed_parts=fixed_components,
            variable_bounds=variable_bounds
        )
        print("✅ Designer created successfully")
        
        print("\nStep 2: Generating D-optimal design...")
        design_df = fixed_designer.generate_design(
            n_runs=8,
            design_type="d-optimal",
            model_type="linear",
            random_seed=42
        )
        print("✅ Design generated successfully")
        
        print("\nStep 3: Testing get_parts_design() method...")
        # This is the exact call that was failing
        full_design = fixed_designer.get_parts_design()
        print(f"✅ get_parts_design() returned: {type(full_design)}")
        print(f"   Shape: {full_design.shape}")
        print(f"   Columns: {list(full_design.columns)}")
        
        print("\nStep 4: Simulating Streamlit app column search...")
        # This is the exact code from the Streamlit app that was failing
        if hasattr(full_design, 'columns'):
            # It's a DataFrame
            parts_cols = [col for col in full_design.columns if '_Parts' in col]
            print(f"   Found parts columns: {parts_cols}")
            
            if parts_cols:  # Only proceed if we found parts columns
                parts_design = full_design[parts_cols].values  # Extract as numpy array
                print(f"✅ Parts design extracted successfully!")
                print(f"   Parts design shape: {parts_design.shape}")
                print(f"   Sample values:")
                for i, col in enumerate(parts_cols):
                    print(f"     {col}: [{parts_design[0, i]:.3f}, {parts_design[1, i]:.3f}, ...]")
                
                # Verify the design properties
                print(f"\nStep 5: Verifying design properties...")
                
                # Check that fixed component C has constant parts
                c_col_idx = parts_cols.index('C_Parts')
                c_parts = parts_design[:, c_col_idx]
                is_constant = np.allclose(c_parts, 10.0, atol=1e-6)
                print(f"   Fixed component C constant: {is_constant} (values: {c_parts})")
                
                # Check that variable components are within bounds
                a_col_idx = parts_cols.index('A_Parts')
                b_col_idx = parts_cols.index('B_Parts')
                a_parts = parts_design[:, a_col_idx]
                b_parts = parts_design[:, b_col_idx]
                
                a_within_bounds = np.all((a_parts >= 5 - 1e-6) & (a_parts <= 20 + 1e-6))
                b_within_bounds = np.all((b_parts >= 5 - 1e-6) & (b_parts <= 20 + 1e-6))
                
                print(f"   A within bounds [5, 20]: {a_within_bounds} (range: [{a_parts.min():.3f}, {a_parts.max():.3f}])")
                print(f"   B within bounds [5, 20]: {b_within_bounds} (range: [{b_parts.min():.3f}, {b_parts.max():.3f}])")
                
                if is_constant and a_within_bounds and b_within_bounds:
                    print(f"\n🎉 SUCCESS: All constraints satisfied!")
                    return True
                else:
                    print(f"\n❌ FAILURE: Constraints not satisfied!")
                    return False
                
            else:
                print(f"❌ FAILURE: No parts columns found in design!")
                print(f"   Available columns: {list(full_design.columns)}")
                return False
        else:
            print(f"❌ FAILURE: get_parts_design() did not return a DataFrame")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consistency_with_generate_design():
    """Test that get_parts_design() is consistent with generate_design() output."""
    
    print(f"\n" + "=" * 60)
    print("🧪 TESTING CONSISTENCY BETWEEN METHODS")
    print("=" * 60)
    
    component_names = ["Component_A", "Component_B", "Component_C"]
    fixed_components = {"Component_C": 15.0}
    variable_bounds = {"Component_A": (10, 30), "Component_B": (5, 25)}
    
    try:
        print("Step 1: Creating designer and generating design...")
        designer = FixedPartsMixtureDesign(
            component_names=component_names,
            fixed_parts=fixed_components,
            variable_bounds=variable_bounds
        )
        
        # Generate design - this creates comprehensive DataFrame
        full_results_df = designer.generate_design(
            n_runs=6,
            design_type="d-optimal",
            model_type="linear",
            random_seed=123
        )
        
        print(f"✅ Full results DataFrame shape: {full_results_df.shape}")
        print(f"   Columns: {list(full_results_df.columns)}")
        
        # Get parts design using backward compatibility method
        parts_only_df = designer.get_parts_design()
        
        print(f"✅ Parts-only DataFrame shape: {parts_only_df.shape}")
        print(f"   Columns: {list(parts_only_df.columns)}")
        
        print(f"\nStep 2: Comparing consistency...")
        
        # Extract parts columns from full results
        parts_cols_from_full = [col for col in full_results_df.columns if col.endswith('_Parts')]
        parts_data_from_full = full_results_df[parts_cols_from_full].values
        
        # Get parts data from parts-only method
        parts_data_from_method = parts_only_df.values
        
        # Compare
        are_equal = np.allclose(parts_data_from_full, parts_data_from_method, atol=1e-10)
        print(f"   Parts data consistency: {are_equal}")
        
        if are_equal:
            print(f"✅ Both methods return identical parts data!")
            
            # Show sample comparison
            print(f"\nSample comparison (first 2 runs):")
            print(f"From generate_design():  {parts_data_from_full[:2].tolist()}")
            print(f"From get_parts_design(): {parts_data_from_method[:2].tolist()}")
            
            return True
        else:
            print(f"❌ Parts data inconsistency detected!")
            print(f"Max difference: {np.max(np.abs(parts_data_from_full - parts_data_from_method))}")
            return False
            
    except Exception as e:
        print(f"❌ Error during consistency test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TESTING FIX FOR 'No parts columns found in design!' ERROR")
    print("=" * 70)
    
    # Test 1: Main Streamlit scenario
    success1 = test_streamlit_parts_columns_fix()
    
    # Test 2: Consistency check
    success2 = test_consistency_with_generate_design()
    
    print(f"\n" + "=" * 70)
    print("FINAL RESULTS:")
    print(f"  ✅ Streamlit scenario fix: {'PASSED' if success1 else 'FAILED'}")
    print(f"  ✅ Method consistency: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print(f"\n🎉 ALL TESTS PASSED - 'No parts columns found' error is FIXED!")
    else:
        print(f"\n💥 SOME TESTS FAILED - Issue may still exist!")
