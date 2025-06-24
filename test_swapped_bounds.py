"""
Test script to verify the automatic correction of swapped bounds
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign

print("TEST: Automatic correction of swapped bounds")

# 1. Test with swapped proportion bounds
print("\nTest 1: Swapped proportion bounds")
component_names = ['A', 'B', 'C']
# Deliberately swap the bounds for component A
component_bounds = [(0.6, 0.1), (0.1, 0.5), (0.1, 0.4)]

try:
    # Create design with swapped bounds - should auto-correct
    mixture = MixtureDesign(3, component_names, component_bounds)
    
    # Print the corrected bounds
    print("Corrected bounds:")
    for i, (name, bounds) in enumerate(zip(component_names, mixture.component_bounds)):
        print(f"  {name}: {bounds}")
    
    # Generate a design to test it works
    design = mixture.generate_d_optimal_mixture(10, random_seed=42)
    
    # Verify the design respects the corrected bounds
    print("\nDesign points:")
    df = pd.DataFrame(design[:3], columns=component_names).round(4)
    print(df)
    
    # Check bounds compliance
    for col, (lower, upper) in zip(component_names, mixture.component_bounds):
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"Component {col} range: {min_val:.4f} - {max_val:.4f} (bounds: {lower:.4f} - {upper:.4f})")
        
except Exception as e:
    print(f"Error: {str(e)}")

# 2. Test with swapped parts bounds in parts mode
print("\nTest 2: Swapped parts bounds")
component_names = ['X', 'Y', 'Z']
# Deliberately swap the bounds for component X
component_bounds = [(5.0, 1.0), (3.0, 8.0), (2.0, 6.0)]
fixed_components = {'Z': 3}  # Fix Z at 3 parts

try:
    # Create design with swapped parts bounds - should auto-correct
    mixture_parts = MixtureDesign(3, component_names, component_bounds, 
                                 use_parts_mode=True, fixed_components=fixed_components)
    
    # Generate a design to test it works
    design = mixture_parts.generate_d_optimal_mixture(10, random_seed=42)
    
    # Verify the design
    print("\nDesign points (proportions):")
    df = pd.DataFrame(design[:3], columns=component_names).round(4)
    print(df)
    
    # Calculate batch quantities (to parts)
    batch_size = 100
    parts_quantities = mixture_parts.convert_to_batch_quantities(design, batch_size)
    df_parts = pd.DataFrame(parts_quantities[:3], columns=component_names).round(2)
    print("\nDesign points (parts for 100-unit batch):")
    print(df_parts)
    
    # Verify Z is fixed correctly
    z_vals = df['Z'].values
    print(f"\nComponent Z values - min: {min(z_vals):.4f}, max: {max(z_vals):.4f}, mean: {np.mean(z_vals):.4f}")
    
except Exception as e:
    print(f"Error: {str(e)}")

print("\nTest complete!")
