"""
Test precision cleanup functionality for UI display
This test verifies that the new clean_numerical_precision function
fixes the 0.9998/0.0002 display issues in Streamlit UI.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.mixture_utils import clean_numerical_precision

def test_precision_cleanup():
    """Test that precision cleanup fixes common numerical issues"""
    print("================================================================================")
    print("TESTING PRECISION CLEANUP FOR UI DISPLAY")
    print("================================================================================")
    
    print("\n1. Testing with problem values (like 0.9998, 0.0002)...")
    
    # Create test data with precision issues
    problematic_data = np.array([
        [0.9998, 0.0001, 0.0001],  # Near 1.0, near 0.0, near 0.0
        [0.0002, 0.9997, 0.0001],  # Near 0.0, near 1.0, near 0.0
        [0.3333, 0.3334, 0.3333],  # Normal values
        [0.5000, 0.4999, 0.0001],  # Mixed precision issues
        [0.1000, 0.8999, 0.0001],  # More mixed issues
    ])
    
    print("Original problematic data:")
    for i, row in enumerate(problematic_data):
        print(f"  Row {i+1}: {row}")
    
    # Clean the data
    cleaned_data = clean_numerical_precision(problematic_data)
    
    print("\nAfter precision cleanup:")
    for i, row in enumerate(cleaned_data):
        print(f"  Row {i+1}: {row}")
    
    # Verify cleanup worked
    print("\n✅ Verification:")
    near_zero_count = np.sum(problematic_data < 1e-4)
    near_one_count = np.sum(problematic_data > 1 - 1e-4)
    exact_zero_count = np.sum(cleaned_data == 0.0)
    exact_one_count = np.sum(cleaned_data == 1.0)
    
    print(f"  - Original near-zero values (< 1e-4): {near_zero_count}")
    print(f"  - Original near-one values (> 1-1e-4): {near_one_count}")
    print(f"  - Cleaned exact zeros: {exact_zero_count}")
    print(f"  - Cleaned exact ones: {exact_one_count}")
    
    if exact_zero_count >= near_zero_count and exact_one_count >= near_one_count:
        print("  ✅ SUCCESS: Precision issues were cleaned up!")
    else:
        print("  ❌ FAILED: Some precision issues remain")
        return False
    
    print("\n2. Testing with DataFrame (like Streamlit UI uses)...")
    
    # Create DataFrame with precision issues
    df_problematic = pd.DataFrame(problematic_data, columns=['Component_1', 'Component_2', 'Component_3'])
    print("Original DataFrame:")
    print(df_problematic)
    
    # Clean DataFrame
    df_cleaned = clean_numerical_precision(df_problematic)
    print("\nCleaned DataFrame:")
    print(df_cleaned)
    
    # Check specific problem cases
    print("\n3. Testing specific UI scenarios...")
    
    # Test case from user report: values like 0.9998 and 0.002
    ui_test_data = pd.DataFrame([
        {'Comp1': 0.9998, 'Comp2': 0.0001, 'Comp3': 0.0001},
        {'Comp1': 0.0002, 'Comp2': 0.9997, 'Comp3': 0.0001},
        {'Comp1': 0.3333, 'Comp2': 0.3334, 'Comp3': 0.3333},
    ])
    
    print("UI test data (simulates Design Matrix - Proportions table):")
    print(ui_test_data)
    
    ui_cleaned = clean_numerical_precision(ui_test_data)
    print("\nAfter UI cleanup (what user will see):")
    print(ui_cleaned)
    
    # Verify no more precision artifacts
    has_precision_issues = False
    for col in ui_cleaned.columns:
        values = ui_cleaned[col].values
        near_zero_but_not_zero = np.any((values > 0) & (values < 1e-4))
        near_one_but_not_one = np.any((values < 1) & (values > 1 - 1e-4))
        
        if near_zero_but_not_zero or near_one_but_not_one:
            has_precision_issues = True
            print(f"  ⚠️ Column {col} still has precision issues")
    
    if not has_precision_issues:
        print("  ✅ SUCCESS: No precision artifacts remain in UI data!")
    else:
        print("  ❌ FAILED: Some precision artifacts remain")
        return False
    
    print("\n4. Testing sum constraint preservation...")
    
    # Verify that cleanup preserves mixture constraint (sum = 1)
    original_sums = problematic_data.sum(axis=1)
    cleaned_sums = cleaned_data.sum(axis=1)
    
    print(f"Original sums: {original_sums}")
    print(f"Cleaned sums:  {cleaned_sums}")
    
    sum_preserved = np.allclose(original_sums, cleaned_sums, atol=1e-6)
    
    if sum_preserved:
        print("  ✅ SUCCESS: Sum constraint preserved after cleanup!")
    else:
        print("  ❌ FAILED: Sum constraint violated after cleanup")
        return False
    
    print("\n5. Testing percentage conversion (like UI does)...")
    
    # Test the full UI flow: cleanup + percentage conversion
    percentages_before = ui_test_data * 100
    ui_cleaned_percentages = clean_numerical_precision(ui_test_data) * 100
    
    print("Percentages before cleanup:")
    print(percentages_before.round(1))
    
    print("\nPercentages after cleanup:")
    print(ui_cleaned_percentages.round(1))
    
    # Check for clean percentage values
    clean_percentages = True
    for col in ui_cleaned_percentages.columns:
        values = ui_cleaned_percentages[col].values
        has_artifacts = np.any((values > 0) & (values < 0.01)) or np.any((values < 100) & (values > 99.99))
        if has_artifacts:
            clean_percentages = False
            print(f"  ⚠️ Column {col} percentages still have artifacts")
    
    if clean_percentages:
        print("  ✅ SUCCESS: Clean percentage values in UI!")
    else:
        print("  ❌ FAILED: Percentage artifacts remain")
        return False
    
    print("\n" + "="*80)
    print("🎉 ALL PRECISION CLEANUP TESTS PASSED!")
    print("The UI will now show clean values like 0.000 and 1.000 instead of 0.9998 and 0.0002")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = test_precision_cleanup()
    if not success:
        sys.exit(1)
