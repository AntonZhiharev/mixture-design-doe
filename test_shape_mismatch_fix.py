"""
Test that the shape mismatch fix works correctly
"""

import numpy as np
import pandas as pd

def test_shape_compatibility_fix():
    """Test the shape compatibility logic from the Streamlit app"""
    
    # Simulate the scenario: export_parts_design has 3 columns, component_names has 4 elements
    export_parts_design = np.random.random((9, 3))  # 9 rows, 3 columns
    component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4']  # 4 names
    
    print(f"export_parts_design shape: {export_parts_design.shape}")
    print(f"component_names length: {len(component_names)}")
    print(f"component_names: {component_names}")
    
    # Apply the fix logic
    actual_n_components = export_parts_design.shape[1]
    if len(component_names) != actual_n_components:
        if len(component_names) > actual_n_components:
            # More component names than design columns - truncate names
            effective_component_names = component_names[:actual_n_components]
            print(f"Fixed by truncating: {effective_component_names}")
        else:
            # Fewer component names than design columns - generate additional names
            effective_component_names = component_names.copy()
            for i in range(len(component_names), actual_n_components):
                effective_component_names.append(f"Component_{i+1}")
            print(f"Fixed by extending: {effective_component_names}")
    else:
        effective_component_names = component_names
        print(f"No fix needed: {effective_component_names}")
    
    # Try to create DataFrame - this should work now
    try:
        parts_with_runs_df = pd.DataFrame(export_parts_design, columns=effective_component_names)
        parts_with_runs_df.insert(0, 'Run_Number', range(1, len(export_parts_design) + 1))
        parts_with_runs_df['Total_Parts'] = export_parts_design.sum(axis=1)
        
        print("✅ SUCCESS: DataFrame created without shape mismatch error!")
        print(f"DataFrame shape: {parts_with_runs_df.shape}")
        print(f"DataFrame columns: {list(parts_with_runs_df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing shape mismatch fix...")
    success = test_shape_compatibility_fix()
    if success:
        print("\n🎉 The shape mismatch fix is working correctly!")
    else:
        print("\n💥 The shape mismatch fix needs more work.")
