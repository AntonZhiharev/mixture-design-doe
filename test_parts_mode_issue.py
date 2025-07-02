"""
Test to demonstrate the issue with Parts Mode vs Standard Mode
in D-optimal design generation
"""

import numpy as np
import pandas as pd

# Import the simplified mixture design classes
from src.core.simplified_mixture_design import DOptimalMixtureDesign

def test_parts_mode_vs_standard_mode():
    """Test to show different points generated in first stage between parts mode and standard mode"""
    
    print("="*80)
    print("TESTING PARTS MODE vs STANDARD MODE - D-OPTIMAL DESIGN")
    print("="*80)
    
    # Test configuration
    n_components = 3
    n_runs = 10
    component_names = ['Component_A', 'Component_B', 'Component_C']
    
    # Component ranges for parts mode
    component_ranges = [
        (0.1, 5.0),   # Component A: 0.1 to 5.0 parts
        (0.2, 3.0),   # Component B: 0.2 to 3.0 parts  
        (0.1, 2.0)    # Component C: 0.1 to 2.0 parts
    ]
    
    print("\n1. STANDARD MODE (use_parts_mode=False)")
    print("-" * 50)
    
    # Create designer in standard mode
    designer_standard = DOptimalMixtureDesign(
        n_components=n_components,
        component_names=component_names,
        use_parts_mode=False  # Standard mode
    )
    
    # Generate design in standard mode
    design_standard = designer_standard.generate_design(
        n_runs=n_runs,
        include_interior=True,
        model_type="quadratic"
    )
    
    print(f"Generated {len(design_standard)} points in standard mode:")
    for i, row in design_standard.iterrows():
        values = [f"{row[col]:.6f}" for col in design_standard.columns]
        sum_val = sum(row.values)
        print(f"  Point {i+1}: [{', '.join(values)}] (sum = {sum_val:.6f})")
    
    # Get determinant from standard mode
    det_standard = designer_standard._last_generator.determinant_history[-1]
    print(f"\nStandard mode determinant: {det_standard:.6e}")
    
    print("\n" + "="*80)
    print("\n2. PARTS MODE (use_parts_mode=True)")
    print("-" * 50)
    
    # Create designer in parts mode
    designer_parts = DOptimalMixtureDesign(
        n_components=n_components,
        component_names=component_names,
        use_parts_mode=True,  # Parts mode
        component_bounds=component_ranges
    )
    
    # Generate design in parts mode
    design_parts = designer_parts.generate_design(
        n_runs=n_runs,
        include_interior=True,
        model_type="quadratic"
    )
    
    print(f"Generated {len(design_parts)} points in parts mode:")
    for i, row in design_parts.iterrows():
        values = [f"{row[col]:.6f}" for col in design_parts.columns]
        sum_val = sum(row.values)
        print(f"  Point {i+1}: [{', '.join(values)}] (sum = {sum_val:.6f})")
    
    # Get determinant from parts mode
    det_parts = designer_parts._last_generator.determinant_history[-1]
    print(f"\nParts mode determinant: {det_parts:.6e}")
    
    # Show parts design if available
    if hasattr(designer_parts, 'parts_design') and designer_parts.parts_design is not None:
        print(f"\nCorresponding parts design:")
        parts_design = designer_parts.parts_design
        for i, point in enumerate(parts_design):
            values = [f"{point[j]:.3f}" for j in range(len(point))]
            total_parts = sum(point)
            print(f"  Point {i+1}: [{', '.join(values)}] (total parts = {total_parts:.3f})")
    
    print("\n" + "="*80)
    print("\n3. COMPARISON ANALYSIS")
    print("-" * 50)
    
    # Compare the designs
    print("First few points comparison:")
    print("\nStandard Mode Points:")
    for i in range(min(5, len(design_standard))):
        row = design_standard.iloc[i]
        values = [f"{row[col]:.6f}" for col in design_standard.columns]
        print(f"  Point {i+1}: [{', '.join(values)}]")
    
    print("\nParts Mode Points:")
    for i in range(min(5, len(design_parts))):
        row = design_parts.iloc[i]
        values = [f"{row[col]:.6f}" for col in design_parts.columns]
        print(f"  Point {i+1}: [{', '.join(values)}]")
    
    # Calculate differences
    if len(design_standard) == len(design_parts):
        print(f"\nPoint-by-point differences (absolute):")
        max_diff = 0
        for i in range(min(5, len(design_standard))):
            row_std = design_standard.iloc[i].values
            row_parts = design_parts.iloc[i].values
            diff = np.abs(row_std - row_parts)
            max_component_diff = np.max(diff)
            max_diff = max(max_diff, max_component_diff)
            print(f"  Point {i+1}: max diff = {max_component_diff:.6f}")
        
        if max_diff > 1e-6:
            print(f"\n❌ ISSUE CONFIRMED: Points are different between modes!")
            print(f"   Maximum difference: {max_diff:.6f}")
        else:
            print(f"\n✅ Points are essentially the same (max diff: {max_diff:.6f})")
    
    # Compare determinants
    det_ratio = det_parts / det_standard if det_standard != 0 else float('inf')
    print(f"\nDeterminant comparison:")
    print(f"  Standard mode: {det_standard:.6e}")
    print(f"  Parts mode: {det_parts:.6e}")
    print(f"  Ratio (parts/standard): {det_ratio:.6f}")
    
    if abs(det_ratio - 1.0) > 0.01:
        print(f"❌ DETERMINANTS ARE DIFFERENT! This suggests different optimization paths.")
    else:
        print(f"✅ Determinants are similar.")
    
    print("\n" + "="*80)
    print("\n4. ROOT CAUSE ANALYSIS")
    print("-" * 50)
    
    print("The issue appears to be in the OptimalDesignGenerator.convert_to_parts() method.")
    print("Let's examine what happens:")
    
    # Get the raw design points from the OptimalDesignGenerator
    print(f"\nRaw design points from standard mode OptimalDesignGenerator:")
    raw_points_standard = designer_standard._last_generator.design_points
    for i, point in enumerate(raw_points_standard[:3]):
        values = [f"{point[j]:.6f}" for j in range(len(point))]
        print(f"  Raw point {i+1}: [{', '.join(values)}]")
    
    print(f"\nRaw design points from parts mode OptimalDesignGenerator:")
    raw_points_parts = designer_parts._last_generator.design_points  
    for i, point in enumerate(raw_points_parts[:3]):
        values = [f"{point[j]:.6f}" for j in range(len(point))]
        print(f"  Raw point {i+1}: [{', '.join(values)}]")
    
    # Check if the raw points are the same
    if len(raw_points_standard) == len(raw_points_parts):
        raw_max_diff = 0
        for i in range(len(raw_points_standard)):
            diff = np.abs(np.array(raw_points_standard[i]) - np.array(raw_points_parts[i]))
            raw_max_diff = max(raw_max_diff, np.max(diff))
        
        if raw_max_diff > 1e-6:
            print(f"\n❌ RAW POINTS ARE DIFFERENT! Max diff: {raw_max_diff:.6f}")
            print("This means the OptimalDesignGenerator itself generates different points")
            print("when component_ranges are provided vs when they're not.")
        else:
            print(f"\n✅ Raw points are the same. Difference is in post-processing.")
    
    return design_standard, design_parts

if __name__ == "__main__":
    design_standard, design_parts = test_parts_mode_vs_standard_mode()
