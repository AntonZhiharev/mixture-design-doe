"""
Test standard mixture design for numerical precision issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.optimal_design_generator import OptimalDesignGenerator

def test_standard_mixture_precision():
    """Test that standard mixture designs (without parts mode) have clean values"""
    
    print("="*80)
    print("TESTING STANDARD MIXTURE DESIGN PRECISION")
    print("="*80)
    
    # Create standard mixture design WITHOUT parts mode (no component_ranges)
    print("\nCreating standard mixture design (no parts mode)...")
    generator = OptimalDesignGenerator(
        num_variables=3,
        num_runs=10,
        design_type="mixture",
        model_type="quadratic",
        component_ranges=None  # No parts mode - this should trigger precision cleanup
    )
    
    print(f"  Proportional ranges: {generator.proportional_ranges}")
    
    # Generate design
    print("\nGenerating design...")
    det = generator.generate_optimal_design()
    
    print(f"\nFinal determinant: {det:.6e}")
    
    # Check precision of values
    print("\nDesign points and precision analysis:")
    precision_issues = 0
    clean_zeros = 0
    clean_ones = 0
    
    for i, point in enumerate(generator.design_points):
        point_str = [f'{x:7.3f}' for x in point]
        print(f"  Point {i+1}: {point_str}")
        
        # Check for precision issues
        for j, value in enumerate(point):
            if 0.0001 <= value <= 0.001:  # Very small but not zero
                print(f"    ⚠️  Component {j+1}: {value:.6f} (should be 0.0)")
                precision_issues += 1
            elif 0.999 <= value <= 0.9999:  # Very close to 1 but not exactly 1
                print(f"    ⚠️  Component {j+1}: {value:.6f} (should be 1.0)")
                precision_issues += 1
            elif value == 0.0:
                clean_zeros += 1
            elif value == 1.0:
                clean_ones += 1
    
    print(f"\nPrecision Analysis:")
    print(f"  Precision issues found: {precision_issues}")
    print(f"  Clean zeros (0.0): {clean_zeros}")
    print(f"  Clean ones (1.0): {clean_ones}")
    
    if precision_issues == 0:
        print("✅ SUCCESS: No precision issues found - all values are clean!")
    else:
        print("❌ FAILURE: Precision issues still present")
    
    return precision_issues == 0

if __name__ == "__main__":
    test_standard_mixture_precision()
