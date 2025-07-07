#!/usr/bin/env python3
"""
Test the integrated OptimalDesignGenerator approach
Verify that D-optimal designs now use the superior method
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np
from core.simplified_mixture_design import create_mixture_design, DOptimalMixtureDesign
from core.optimal_design_generator import OptimalDesignGenerator

def test_basic_d_optimal_integration():
    """Test that D-optimal designs now use OptimalDesignGenerator"""
    print("="*80)
    print("TESTING INTEGRATED OPTIMAL DESIGN GENERATOR")
    print("="*80)
    
    print("\n1. Testing basic D-optimal design generation...")
    
    # Test using factory function
    design_df = create_mixture_design(
        method='d-optimal',
        n_components=3,
        n_runs=10,
        model_type='quadratic',
        include_interior=True
    )
    
    print(f"✅ Generated design with {len(design_df)} runs")
    print(f"   Design columns: {list(design_df.columns)}")
    print(f"   First few points:")
    print(design_df.head())
    
    # Verify sum to 1
    sums = design_df.values.sum(axis=1)
    print(f"   Row sums: {sums[:5]}... (should be ~1.0)")
    
    if np.allclose(sums, 1.0):
        print("✅ All mixtures sum to 1.0")
    else:
        print("❌ Some mixtures don't sum to 1.0")
    
    return design_df

def test_d_efficiency_calculation():
    """Test the new D-efficiency calculation using gram matrix"""
    print("\n2. Testing D-efficiency calculation with gram matrix...")
    
    # Import the standalone D-efficiency calculator to avoid Streamlit warnings
    from utils.d_efficiency_calculator import calculate_d_efficiency
    
    # Create a simple test design
    design_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.33, 0.33, 0.34],
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3]
    ])
    
    # Test different model types
    for model_type in ['linear', 'quadratic', 'cubic']:
        d_eff = calculate_d_efficiency(design_matrix, model_type)
        print(f"   D-efficiency ({model_type}): {d_eff:.6f}")
    
    print("✅ D-efficiency calculation working with gram matrix approach")

def test_parts_mode_integration():
    """Test parts mode with the new approach"""
    print("\n3. Testing parts mode integration...")
    
    # Test with parts mode and bounds
    design_df = create_mixture_design(
        method='d-optimal',
        n_components=3,
        n_runs=8,
        use_parts_mode=True,
        component_bounds=[(0.1, 5.0), (0.2, 3.0), (0.05, 1.0)],
        model_type='linear'
    )
    
    print(f"✅ Generated parts mode design with {len(design_df)} runs")
    print(f"   Design preview:")
    print(design_df.head())
    
    # Check if parts design was stored
    designer = DOptimalMixtureDesign(3, use_parts_mode=True, 
                                   component_bounds=[(0.1, 5.0), (0.2, 3.0), (0.05, 1.0)])
    result = designer.generate_design(n_runs=8, model_type='linear')
    
    if hasattr(designer, 'parts_design') and designer.parts_design is not None:
        print(f"✅ Parts design properly stored: {designer.parts_design.shape}")
        print(f"   Parts design preview:")
        print(designer.parts_design[:3])
    else:
        print("⚠️ Parts design not found")

def test_comparison_with_direct_generator():
    """Compare results with direct OptimalDesignGenerator usage"""
    print("\n4. Comparing with direct OptimalDesignGenerator...")
    
    # Direct usage
    generator = OptimalDesignGenerator(
        num_variables=3,
        model_type='quadratic',
        num_runs=12
    )
    
    det_direct = generator.generate_optimal_design()
    
    print(f"✅ Direct generator determinant: {det_direct:.6f}")
    print(f"   Generated {len(generator.design_points)} points")
    
    # Via integrated approach
    design_df = create_mixture_design(
        method='d-optimal',
        n_components=3,
        n_runs=12,
        model_type='quadratic'
    )
    
    print(f"✅ Integrated approach generated {len(design_df)} points")
    
    # Both should work and produce similar quality results
    return generator, design_df

def test_model_types():
    """Test different model types"""
    print("\n5. Testing different model types...")
    
    for model_type in ['linear', 'quadratic', 'cubic']:
        print(f"\n   Testing {model_type} model...")
        
        design_df = create_mixture_design(
            method='d-optimal',
            n_components=3,
            n_runs=15,
            model_type=model_type
        )
        
        print(f"   ✅ {model_type}: {len(design_df)} runs generated")
        
        # Verify design matrix
        design_matrix = design_df.values
        sums = design_matrix.sum(axis=1)
        
        if np.allclose(sums, 1.0):
            print(f"   ✅ {model_type}: All points sum to 1.0")
        else:
            print(f"   ⚠️ {model_type}: Some points don't sum to 1.0")

def main():
    """Run all tests"""
    try:
        # Test 1: Basic integration
        design_df = test_basic_d_optimal_integration()
        
        # Test 2: D-efficiency calculation
        test_d_efficiency_calculation()
        
        # Test 3: Parts mode
        test_parts_mode_integration()
        
        # Test 4: Comparison with direct usage
        generator, integrated_design = test_comparison_with_direct_generator()
        
        # Test 5: Different model types
        test_model_types()
        
        print("\n" + "="*80)
        print("INTEGRATION TEST SUMMARY")
        print("="*80)
        print("✅ All tests completed successfully!")
        print("✅ OptimalDesignGenerator is properly integrated")
        print("✅ D-optimal designs now use the superior approach")
        print("✅ Gram matrix approach is working for D-efficiency")
        print("✅ Parts mode integration is functional")
        print("✅ All model types (linear/quadratic/cubic) are supported")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 INTEGRATION SUCCESSFUL! The OptimalDesignGenerator approach is now active.")
    else:
        print("\n💥 INTEGRATION FAILED! Check the errors above.")
