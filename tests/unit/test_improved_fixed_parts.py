"""
Test the improved fixed parts mixture design implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

def test_improved_implementation():
    """Test the improved implementation with better space-filling."""
    
    print("🧪 Testing IMPROVED Fixed Parts Mixture Design Implementation")
    print("=" * 70)
    
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
    
    # Create improved design generator
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate improved design
    design_df = designer.generate_design(
        n_runs=15,
        design_type="d-optimal",
        model_type="quadratic",
        random_seed=42
    )
    
    print(f"\n📊 Design Results:")
    print(design_df)
    
    # Test space-filling by checking distribution of variable components
    print(f"\n🎯 Analyzing space-filling properties:")
    
    # Extract variable component data
    pigment_a_parts = design_df['Pigment_A_Parts'].values
    pigment_b_parts = design_df['Pigment_B_Parts'].values
    solvent_parts = design_df['Solvent_Parts'].values
    
    # Calculate coefficient of variation (CV) - lower is more uniform
    cv_a = np.std(pigment_a_parts) / np.mean(pigment_a_parts)
    cv_b = np.std(pigment_b_parts) / np.mean(pigment_b_parts)
    cv_s = np.std(solvent_parts) / np.mean(solvent_parts)
    
    print(f"   Pigment_A distribution: mean={np.mean(pigment_a_parts):.2f}, std={np.std(pigment_a_parts):.2f}, CV={cv_a:.3f}")
    print(f"   Pigment_B distribution: mean={np.mean(pigment_b_parts):.2f}, std={np.std(pigment_b_parts):.2f}, CV={cv_b:.3f}")
    print(f"   Solvent distribution: mean={np.mean(solvent_parts):.2f}, std={np.std(solvent_parts):.2f}, CV={cv_s:.3f}")
    
    # Test boundary coverage
    bounds_coverage = {
        'Pigment_A': (min(pigment_a_parts), max(pigment_a_parts), variable_bounds['Pigment_A']),
        'Pigment_B': (min(pigment_b_parts), max(pigment_b_parts), variable_bounds['Pigment_B']),
        'Solvent': (min(solvent_parts), max(solvent_parts), variable_bounds['Solvent'])
    }
    
    print(f"\n🎯 Boundary coverage analysis:")
    for comp, (actual_min, actual_max, (bound_min, bound_max)) in bounds_coverage.items():
        coverage_min = (actual_min - bound_min) / (bound_max - bound_min) * 100
        coverage_max = (actual_max - bound_max) / (bound_max - bound_min) * 100
        print(f"   {comp}: actual=[{actual_min:.2f}, {actual_max:.2f}], bounds=[{bound_min}, {bound_max}]")
        print(f"      Coverage: {coverage_min:.1f}% from min, {coverage_max:.1f}% from max")
    
    # Test fixed components consistency
    print(f"\n🔒 Fixed components verification:")
    for comp, expected_value in fixed_parts.items():
        actual_values = design_df[f'{comp}_Parts'].values
        is_constant = np.allclose(actual_values, expected_value, atol=1e-6)
        print(f"   {comp}: expected={expected_value}, actual=[{actual_values.min():.6f}, {actual_values.max():.6f}], constant={is_constant}")
    
    # Test proportions sum to 1
    prop_cols = [col for col in design_df.columns if col.endswith('_Prop')]
    prop_sums = design_df[prop_cols].sum(axis=1)
    props_valid = np.allclose(prop_sums, 1.0, atol=1e-6)
    print(f"\n✅ Proportions sum to 1: {props_valid} (range: {prop_sums.min():.6f} to {prop_sums.max():.6f})")
    
    print(f"\n🎉 IMPROVED Implementation Test Complete!")
    return design_df

if __name__ == "__main__":
    test_improved_implementation()
