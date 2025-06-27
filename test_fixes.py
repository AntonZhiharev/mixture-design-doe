"""
Test script to verify the fixes for:
1. KeyError in streamlit app (column name mismatch)
2. D-optimal design generating fewer runs than requested
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.simplified_mixture_design import create_mixture_design

def test_d_optimal_runs():
    """Test that D-optimal design generates the requested number of runs"""
    print("Testing D-optimal design with 24 runs...")
    
    # Test with 3 components, 24 runs
    design = create_mixture_design(
        method='d-optimal',
        n_components=3,
        n_runs=24,
        include_interior=True,
        component_names=['Component_A', 'Component_B', 'Component_C']
    )
    
    print(f"Requested runs: 24")
    print(f"Generated runs: {len(design)}")
    print(f"Column names: {design.columns.tolist()}")
    print(f"Design preview:")
    print(design.head())
    
    # Verify all mixtures sum to 1
    sums = design.values.sum(axis=1)
    all_sum_to_one = all(abs(s - 1.0) < 1e-10 for s in sums)
    print(f"All mixtures sum to 1.0: {all_sum_to_one}")
    
    return len(design) == 24

def test_column_names():
    """Test that custom component names are properly used"""
    print("\nTesting custom component names...")
    
    custom_names = ['Polymer_A', 'Polymer_B', 'Polymer_C']
    design = create_mixture_design(
        method='simplex-lattice',
        n_components=3,
        degree=2,
        component_names=custom_names
    )
    
    print(f"Expected column names: {custom_names}")
    print(f"Actual column names: {design.columns.tolist()}")
    
    return list(design.columns) == custom_names

def test_parts_mode():
    """Test parts mode functionality"""
    print("\nTesting parts mode...")
    
    # Test with fixed components in parts mode
    component_names = ['Polymer_A', 'Polymer_B', 'Additive']
    component_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.1)]  # Parts bounds
    fixed_components = {'Additive': 0.5}  # Fixed at 0.5 parts
    
    design = create_mixture_design(
        method='d-optimal',
        n_components=3,
        n_runs=12,
        component_names=component_names,
        use_parts_mode=True,
        component_bounds=component_bounds,
        fixed_components=fixed_components,
        include_interior=True
    )
    
    print(f"Design shape: {design.shape}")
    print(f"Column names: {design.columns.tolist()}")
    print("Design preview:")
    print(design.head())
    
    # Verify the design sums to 1 (proportions)
    sums = design.values.sum(axis=1)
    all_sum_to_one = all(abs(s - 1.0) < 1e-10 for s in sums)
    print(f"All mixtures sum to 1.0: {all_sum_to_one}")
    
    return len(design) > 0 and all_sum_to_one

def test_fixed_parts_design():
    """Test the dedicated FixedPartsMixtureDesign class"""
    print("\nTesting FixedPartsMixtureDesign...")
    
    from core.simplified_mixture_design import FixedPartsMixtureDesign
    
    component_names = ['Base_Resin', 'Hardener', 'Catalyst']
    component_bounds = [(10.0, 50.0), (1.0, 10.0), (0.1, 2.0)]  # Parts bounds
    fixed_components = {'Catalyst': 0.5}  # Always 0.5 parts catalyst
    
    designer = FixedPartsMixtureDesign(
        n_components=3,
        component_names=component_names,
        component_bounds=component_bounds,
        fixed_components=fixed_components
    )
    
    design = designer.generate_design(
        n_runs=15,
        design_type='d-optimal',
        include_interior=True
    )
    
    print(f"Design shape: {design.shape}")
    print("Design preview:")
    print(design.head())
    
    # Check if parts design is available
    parts_design = designer.get_parts_design()
    has_parts = parts_design is not None
    print(f"Parts design available: {has_parts}")
    
    if has_parts:
        print("Parts design preview:")
        print(parts_design[:3])  # First 3 rows
    
    return len(design) > 0 and has_parts

def test_i_optimal_design():
    """Test I-optimal design functionality"""
    print("\nTesting I-optimal design...")
    
    # Test 1: Basic I-optimal design generation
    print("  - Testing basic I-optimal design...")
    design = create_mixture_design(
        method='i-optimal',
        n_components=3,
        n_runs=12,
        component_names=['Component_A', 'Component_B', 'Component_C'],
        include_interior=True,
        model_type='quadratic'
    )
    
    print(f"  Design shape: {design.shape}")
    print(f"  Column names: {design.columns.tolist()}")
    print("  Design preview:")
    print(design.head(3))
    
    # Verify the design sums to 1 (proportions)
    sums = design.values.sum(axis=1)
    all_sum_to_one = all(abs(s - 1.0) < 1e-10 for s in sums)
    print(f"  All mixtures sum to 1.0: {all_sum_to_one}")
    
    # Verify we got the requested number of runs
    correct_runs = len(design) == 12
    print(f"  Generated correct number of runs (12): {correct_runs}")
    
    basic_test_passed = len(design) > 0 and all_sum_to_one and correct_runs
    
    # Test 2: I-optimal with constraints (parts mode)
    print("  - Testing I-optimal with constraints...")
    try:
        constrained_design = create_mixture_design(
            method='i-optimal',
            n_components=3,
            n_runs=10,
            component_names=['Polymer_A', 'Polymer_B', 'Additive'],
            use_parts_mode=True,
            component_bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 0.1)],
            include_interior=True,
            model_type='linear'
        )
        
        print(f"  Constrained design shape: {constrained_design.shape}")
        print("  Constrained design preview:")
        print(constrained_design.head(3))
        
        # Verify constraints
        constrained_sums = constrained_design.values.sum(axis=1)
        constrained_all_sum_to_one = all(abs(s - 1.0) < 1e-10 for s in constrained_sums)
        print(f"  Constrained mixtures sum to 1.0: {constrained_all_sum_to_one}")
        
        constrained_test_passed = len(constrained_design) > 0 and constrained_all_sum_to_one
    except Exception as e:
        print(f"  Constrained I-optimal test failed: {e}")
        constrained_test_passed = False
    
    # Test 3: I-optimal with different model types
    print("  - Testing I-optimal with different model types...")
    try:
        for model_type in ['linear', 'quadratic', 'cubic']:
            model_design = create_mixture_design(
                method='i-optimal',
                n_components=3,
                n_runs=8,
                include_interior=True,
                model_type=model_type
            )
            print(f"    {model_type} model: {len(model_design)} runs generated")
        
        model_test_passed = True
    except Exception as e:
        print(f"  Model type test failed: {e}")
        model_test_passed = False
    
    overall_passed = basic_test_passed and constrained_test_passed and model_test_passed
    print(f"  I-optimal overall test: {'PASSED' if overall_passed else 'FAILED'}")
    
    return overall_passed

if __name__ == "__main__":
    print("=" * 50)
    print("Testing fixes and new features for mixture design")
    print("=" * 50)
    
    # Test 1: D-optimal runs count
    test1_passed = test_d_optimal_runs()
    
    # Test 2: Column names
    test2_passed = test_column_names()
    
    # Test 3: Parts mode functionality
    test3_passed = test_parts_mode()
    
    # Test 4: Fixed parts design
    test4_passed = test_fixed_parts_design()
    
    # Test 5: I-optimal design
    test5_passed = test_i_optimal_design()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"✅ D-optimal runs count fix: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"✅ Custom component names fix: {'PASSED' if test2_passed else 'FAILED'}")
    print(f"✅ Parts mode functionality: {'PASSED' if test3_passed else 'FAILED'}")
    print(f"✅ Fixed parts design: {'PASSED' if test4_passed else 'FAILED'}")
    print(f"✅ I-optimal design: {'PASSED' if test5_passed else 'FAILED'}")
    
    all_passed = all([test1_passed, test2_passed, test3_passed, test4_passed, test5_passed])
    
    if all_passed:
        print("\n🎉 All tests passed! Fixes and parts mode working correctly!")
    else:
        print("\n❌ Some tests failed. Check the implementation.")
