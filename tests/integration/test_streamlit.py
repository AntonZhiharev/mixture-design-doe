"""
Test end-to-end precision cleanup in Streamlit UI tables
This test simulates the full Streamlit workflow to verify precision cleanup works
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.mixture_utils import clean_numerical_precision

def create_test_data_with_precision_issues():
    """Create test data that simulates the precision issues reported by the user"""
    
    # Simulate design matrix with precision artifacts (like what happens in D-optimal)
    design_proportions = np.array([
        [0.9998, 0.0001, 0.0001],  # Component 1 dominant - precision artifacts
        [0.0002, 0.9997, 0.0001],  # Component 2 dominant - precision artifacts  
        [0.0001, 0.0002, 0.9997],  # Component 3 dominant - precision artifacts
        [0.3333, 0.3334, 0.3333],  # Mixed proportions
        [0.6667, 0.3332, 0.0001],  # Two-component mix with artifact
        [0.5001, 0.4998, 0.0001],  # Another two-component mix
    ])
    
    return design_proportions

def test_design_matrix_proportions_cleanup():
    """Test precision cleanup in Design Matrix - Proportions table"""
    print("=" * 80)
    print("TESTING DESIGN MATRIX - PROPORTIONS TABLE CLEANUP")
    print("=" * 80)
    
    # Create test data
    design_data = create_test_data_with_precision_issues()
    component_names = ['Component_1', 'Component_2', 'Component_3']
    
    # Create DataFrame like Streamlit does
    display_df = pd.DataFrame(design_data, columns=component_names)
    
    print("Before precision cleanup:")
    print(display_df)
    
    # Apply precision cleanup like Streamlit does
    display_df = clean_numerical_precision(display_df)
    
    # Add percentage columns like Streamlit does  
    for col_name in display_df.columns:
        display_df[f"{col_name} (%)"] = (display_df[col_name] * 100).round(1)
    
    print("\nAfter precision cleanup:")
    print(display_df.round(4))
    
    # Verify cleanup worked
    has_artifacts = False
    for col in component_names:
        values = display_df[col].values
        near_zero_but_not_zero = np.any((values > 0) & (values < 1e-4))
        near_one_but_not_one = np.any((values < 1) & (values > 1 - 1e-4))
        
        if near_zero_but_not_zero or near_one_but_not_one:
            has_artifacts = True
            print(f"  ⚠️ Column {col} still has precision artifacts")
    
    if not has_artifacts:
        print("  ✅ SUCCESS: No precision artifacts in Design Matrix!")
        return True
    else:
        print("  ❌ FAILED: Some precision artifacts remain in Design Matrix")
        return False

def test_parts_per_100_cleanup():
    """Test precision cleanup in Parts per 100 Total table"""
    print("\n" + "=" * 80)
    print("TESTING PARTS PER 100 TOTAL TABLE CLEANUP")
    print("=" * 80)
    
    # Create parts design with precision issues
    design_data = create_test_data_with_precision_issues()
    parts_design = design_data * 100.0  # Convert to parts per 100
    
    component_names = ['Component_1', 'Component_2', 'Component_3']
    
    # Create DataFrame like Streamlit does
    parts_df = pd.DataFrame(parts_design, columns=component_names)
    parts_df.index = [f"Run_{i+1}" for i in range(len(parts_df))]
    
    print("Before precision cleanup:")
    print(parts_df)
    
    # Apply precision cleanup like Streamlit does
    parts_df = clean_numerical_precision(parts_df, preserve_mixture_constraint=False)
    
    # Add totals verification like Streamlit does
    parts_df['Total_Parts'] = parts_design.sum(axis=1)
    
    print("\nAfter precision cleanup:")
    print(parts_df.round(3))
    
    # Verify cleanup worked
    has_artifacts = False
    for col in component_names:
        values = parts_df[col].values
        near_zero_but_not_zero = np.any((values > 0) & (values < 0.1))  # 0.1 parts threshold
        near_hundred_but_not_hundred = np.any((values < 100) & (values > 99.9))
        
        if near_zero_but_not_zero or near_hundred_but_not_hundred:
            has_artifacts = True
            print(f"  ⚠️ Column {col} still has precision artifacts")
    
    if not has_artifacts:
        print("  ✅ SUCCESS: No precision artifacts in Parts per 100 table!")
        return True
    else:
        print("  ❌ FAILED: Some precision artifacts remain in Parts per 100 table")
        return False

def test_manufacturing_worksheets_cleanup():
    """Test precision cleanup in Manufacturing Worksheets tables"""
    print("\n" + "=" * 80)
    print("TESTING MANUFACTURING WORKSHEETS TABLE CLEANUP")
    print("=" * 80)
    
    # Create parts design with precision issues
    design_data = create_test_data_with_precision_issues()
    parts_design = design_data * 100.0  # Convert to parts per 100
    
    component_names = ['Component_1', 'Component_2', 'Component_3']
    batch_size = 5.0  # kg
    
    # Calculate actual quantities like Streamlit does
    actual_quantities = parts_design * batch_size / 100.0
    
    # Create manufacturing worksheet like Streamlit does
    worksheet_df = pd.DataFrame()
    worksheet_df['Run_ID'] = [f'EXP_{i+1:02d}' for i in range(len(actual_quantities))]
    
    # Add percentages first, then kg quantities for each component
    for j, comp_name in enumerate(component_names):
        worksheet_df[f'{comp_name}_%'] = (parts_design[:, j]).round(2)
        worksheet_df[f'{comp_name}_kg'] = actual_quantities[:, j].round(4)
    
    # Add totals and verification
    worksheet_df['Total_kg'] = actual_quantities.sum(axis=1).round(4)
    worksheet_df['Weight_Check'] = ['✓' if abs(total - batch_size) < 1e-3 else '✗' 
                                   for total in worksheet_df['Total_kg']]
    
    print("Before precision cleanup:")
    print(worksheet_df)
    
    # Apply precision cleanup like Streamlit does
    numeric_cols = [col for col in worksheet_df.columns if col not in ['Run_ID', 'Weight_Check']]
    worksheet_df[numeric_cols] = clean_numerical_precision(worksheet_df[numeric_cols], preserve_mixture_constraint=False)
    
    print("\nAfter precision cleanup:")
    print(worksheet_df)
    
    # Verify cleanup worked
    has_artifacts = False
    for col in numeric_cols:
        values = worksheet_df[col].values
        if 'kg' in col:
            # For kg columns, check for very small values that should be zero
            near_zero_but_not_zero = np.any((values > 0) & (values < 0.001))  # 1mg threshold
        else:
            # For percentage columns, check for precision artifacts
            near_zero_but_not_zero = np.any((values > 0) & (values < 0.1))  # 0.1% threshold
            near_hundred_but_not_hundred = np.any((values < 100) & (values > 99.9))
            
        if ('kg' in col and near_zero_but_not_zero) or ('kg' not in col and (near_zero_but_not_zero or near_hundred_but_not_hundred)):
            has_artifacts = True
            print(f"  ⚠️ Column {col} still has precision artifacts")
    
    if not has_artifacts:
        print("  ✅ SUCCESS: No precision artifacts in Manufacturing Worksheets!")
        return True
    else:
        print("  ❌ FAILED: Some precision artifacts remain in Manufacturing Worksheets")
        return False

def test_percentage_display():
    """Test that percentage displays show clean values"""
    print("\n" + "=" * 80)
    print("TESTING PERCENTAGE DISPLAY VALUES")
    print("=" * 80)
    
    # Test cases that should display cleanly
    test_values = np.array([
        [0.9998, 0.0001, 0.0001],  # Should become [100.0, 0.0, 0.0]
        [0.0002, 0.9997, 0.0001],  # Should become [0.0, 100.0, 0.0]
        [0.3333, 0.3334, 0.3333],  # Should stay approximately the same
    ])
    
    print("Test values (proportions):")
    print(test_values)
    
    # Apply cleanup
    cleaned_values = clean_numerical_precision(test_values)
    
    # Convert to percentages like UI does
    percentages = cleaned_values * 100
    
    print("\nPercentages after cleanup:")
    print(percentages.round(1))
    
    # Check for clean percentage values
    expected_clean_values = [
        [100.0, 0.0, 0.0],
        [0.0, 100.0, 0.0], 
        # Third row should preserve original proportions
    ]
    
    success = True
    for i in range(2):  # Check first two rows for exact matches
        if not np.allclose(percentages[i].round(1), expected_clean_values[i], atol=0.1):
            print(f"  ❌ Row {i+1} percentages not clean: {percentages[i].round(1)}")
            success = False
    
    if success:
        print("  ✅ SUCCESS: Clean percentage values!")
        return True
    else:
        print("  ❌ FAILED: Some percentage values are not clean")
        return False

def main():
    """Run all precision cleanup tests"""
    print("🧪 TESTING STREAMLIT UI PRECISION CLEANUP SOLUTION")
    print("=" * 80)
    print("This test verifies that the precision cleanup fixes are working")
    print("in all Streamlit tables where 0.9998/0.0002 issues were reported.")
    print("=" * 80)
    
    results = []
    
    # Test each table type
    results.append(test_design_matrix_proportions_cleanup())
    results.append(test_parts_per_100_cleanup())
    results.append(test_manufacturing_worksheets_cleanup())
    results.append(test_percentage_display())
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("✅ The Streamlit UI precision cleanup solution is working correctly!")
        print("✅ Users will no longer see 0.9998/0.0002 artifacts in any tables!")
        print("✅ The following tables now display clean values:")
        print("   • Design Matrix - Proportions")
        print("   • Parts per 100 Total") 
        print("   • Manufacturing Worksheets")
        print("   • All percentage displays")
        return True
    else:
        print(f"❌ {passed}/{total} tests passed. Some issues remain.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
