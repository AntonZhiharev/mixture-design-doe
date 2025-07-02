"""
Show the actual design matrix produced by our Fixed Components implementation
"""

import numpy as np
import pandas as pd
from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

def show_current_design_matrix():
    """Generate and display the actual design matrix."""
    print("=" * 80)
    print("CURRENT DESIGN MATRIX - FIXED COMPONENTS IMPLEMENTATION")
    print("=" * 80)
    
    # Test case 1: Simple 3-component mixture
    print("\n🧪 TEST CASE 1: Simple 3-component mixture")
    print("Components: A_Fixed, B_Var, C_Var")
    print("Fixed: A_Fixed = 10.0 parts")
    print("Variable: B_Var = 0-20 parts, C_Var = 0-15 parts")
    
    component_names = ['A_Fixed', 'B_Var', 'C_Var']
    fixed_parts = {'A_Fixed': 10.0}
    variable_bounds = {'B_Var': (0.1, 10.0), 'C_Var': (0.1, 10.0)}
    
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate design
    design_df = designer.generate_design(
        n_runs=8,
        design_type="d-optimal",
        model_type="quadratic",
        random_seed=42
    )
    
    print(f"\n📊 DESIGN MATRIX - PARTS:")
    parts_cols = [col for col in design_df.columns if '_Parts' in col]
    print(design_df[parts_cols].round(3))
    
    print(f"\n📊 DESIGN MATRIX - PROPORTIONS:")
    prop_cols = [col for col in design_df.columns if '_Prop' in col]
    props_df = design_df[prop_cols].round(3)
    print(props_df)
    
    print(f"\n🔍 ANALYSIS:")
    print(f"A_Fixed parts range: {design_df['A_Fixed_Parts'].min():.3f} to {design_df['A_Fixed_Parts'].max():.3f}")
    print(f"A_Fixed proportions range: {design_df['A_Fixed_Prop'].min():.3f} to {design_df['A_Fixed_Prop'].max():.3f}")
    print(f"B_Var proportions range: {design_df['B_Var_Prop'].min():.3f} to {design_df['B_Var_Prop'].max():.3f}")
    print(f"C_Var proportions range: {design_df['C_Var_Prop'].min():.3f} to {design_df['C_Var_Prop'].max():.3f}")
    
    # Check for the specific issues the user mentioned
    zero_proportions = (props_df == 0.0).any(axis=1).sum()
    less_than_third = (props_df < 0.333).any(axis=1).sum()
    
    print(f"\n📋 USER'S CONCERNS:")
    print(f"Runs with proportion = 0: {zero_proportions} out of {len(props_df)}")
    print(f"Runs with proportion < 0.333: {less_than_third} out of {len(props_df)}")
    
    # Show specific examples
    print(f"\n🎯 SPECIFIC EXAMPLES:")
    for i, row in props_df.iterrows():
        if any(row == 0.0):
            print(f"Run {i+1}: {row.values} (has zero proportion)")
        elif any(row < 0.333):
            print(f"Run {i+1}: {row.values} (has < 0.333 proportion)")
    
    return design_df

def show_polymer_example():
    """Show the polymer example that was in the test."""
    print("\n" + "=" * 80)
    print("POLYMER FORMULATION EXAMPLE")
    print("=" * 80)
    
    print("\n🧪 TEST CASE 2: Polymer Formulation")
    print("Components: Base_Polymer, Catalyst, Solvent, Additive")
    print("Fixed: Base_Polymer = 50.0 parts, Catalyst = 2.5 parts")
    print("Variable: Solvent = 0-40 parts, Additive = 0-15 parts")
    
    component_names = ["Base_Polymer", "Catalyst", "Solvent", "Additive"]
    fixed_parts = {
        "Base_Polymer": 50.0,
        "Catalyst": 2.5
    }
    variable_bounds = {
        "Solvent": (0.0, 40.0),
        "Additive": (0.0, 15.0)
    }
    
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate design
    design_df = designer.generate_design(
        n_runs=12,
        design_type="d-optimal",
        model_type="quadratic",
        random_seed=42
    )
    
    print(f"\n📊 DESIGN MATRIX - PROPORTIONS (first 8 runs):")
    prop_cols = [col for col in design_df.columns if '_Prop' in col]
    props_df = design_df[prop_cols].head(8).round(3)
    print(props_df)
    
    print(f"\n🔍 ANALYSIS:")
    print(f"Base_Polymer proportions range: {design_df['Base_Polymer_Prop'].min():.3f} to {design_df['Base_Polymer_Prop'].max():.3f}")
    print(f"Catalyst proportions range: {design_df['Catalyst_Prop'].min():.3f} to {design_df['Catalyst_Prop'].max():.3f}")
    print(f"Solvent proportions range: {design_df['Solvent_Prop'].min():.3f} to {design_df['Solvent_Prop'].max():.3f}")
    print(f"Additive proportions range: {design_df['Additive_Prop'].min():.3f} to {design_df['Additive_Prop'].max():.3f}")
    
    return design_df

if __name__ == "__main__":
    print("🔍 SHOWING CURRENT DESIGN MATRIX")
    print("Let's see exactly what our implementation produces")
    
    design1 = show_current_design_matrix()
    design2 = show_polymer_example()
    
    print(f"\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print("Above are the ACTUAL design matrices our implementation produces.")
    print("Please review and let me know what specific behavior concerns you.")
