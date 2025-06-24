"""
Solution: Allocate free space for fixed components proportionally
"""

import numpy as np
from enhanced_mixture_designs import EnhancedMixtureDesign

print("="*60)
print("SOLUTION: PROPORTIONAL SPACE ALLOCATION FOR FIXED COMPONENTS")
print("="*60)

# Example user case with realistic bounds
all_component_names = ['NP200', 'CPE', 'DL531', 'TL60', 'OPE', 'PVC', 'CaCO3', 'UVStabilisator']
n_components = 8

# Component bounds in parts
component_bounds_parts = [
    (0.02, 0.04),     # NP200 (variable)
    (0.04, 0.16),     # CPE (variable)
    (0.01, 0.08),     # DL531 (variable)
    (0.005, 0.01),    # TL60 (variable)
    (0.04, 0.16),     # OPE (variable)
    (1.0, 1.0),       # PVC (fixed)
    (0.5, 0.5),       # CaCO3 (fixed)
    (0.02, 0.02)      # UVStabilisator (fixed)
]

fixed_parts = {
    'PVC': 1.0,
    'CaCO3': 0.5,
    'UVStabilisator': 0.02
}

fixed_component_names = list(fixed_parts.keys())

print("\nStep 1: Calculate total parts")
total_parts = sum(bound[1] for bound in component_bounds_parts)
print(f"Total parts = {total_parts:.3f}")

print("\nStep 2: Initial normalization - convert to proportions")
component_bounds_props = []
fixed_components_props_original = {}
variable_indices = []
fixed_indices = []

for i, (comp_name, (min_parts, max_parts)) in enumerate(zip(all_component_names, component_bounds_parts)):
    min_prop = min_parts / total_parts
    max_prop = max_parts / total_parts
    component_bounds_props.append((min_prop, max_prop))
    
    if comp_name in fixed_component_names:
        fixed_components_props_original[comp_name] = max_prop
        fixed_indices.append(i)
        print(f"{comp_name}: {max_parts:.3f} parts → {max_prop:.4f} prop (FIXED)")
    else:
        variable_indices.append(i)
        print(f"{comp_name}: ({min_parts:.3f}-{max_parts:.3f}) parts → ({min_prop:.4f}-{max_prop:.4f}) props")

print("\nStep 3: Calculate flexibility space")
# Sum of max variable components
sum_max_variable = sum(component_bounds_props[i][1] for i in variable_indices)
# Sum of min variable components
sum_min_variable = sum(component_bounds_props[i][0] for i in variable_indices)

# Free space is the flexibility of variable components
free_space = sum_max_variable - sum_min_variable

print(f"Sum of MIN variable components = {sum_min_variable:.4f}")
print(f"Sum of MAX variable components = {sum_max_variable:.4f}")
print(f"Variable flexibility (free space) = {free_space:.4f}")

# Space available for fixed components
space_when_var_at_min = 1.0 - sum_min_variable  # Max space for fixed
space_when_var_at_max = 1.0 - sum_max_variable  # Min space for fixed

print(f"\nSpace for fixed when variables at MIN = {space_when_var_at_min:.4f}")
print(f"Space for fixed when variables at MAX = {space_when_var_at_max:.4f}")

# Calculate total fixed components (original)
total_fixed_original = sum(fixed_components_props_original.values())
print(f"\nOriginal fixed components sum = {total_fixed_original:.4f}")

print("\nStep 4: Create bounds for fixed components with flexibility")
fixed_components_props_adjusted = {}
component_bounds_adjusted = component_bounds_props.copy()

for comp_name, original_value in fixed_components_props_original.items():
    # Calculate proportion of this fixed component relative to all fixed
    fraction = original_value / total_fixed_original
    
    # When variables are at MIN, fixed can be at MAX
    max_value = space_when_var_at_min * fraction
    # When variables are at MAX, fixed must be at MIN
    min_value = space_when_var_at_max * fraction
    
    fixed_components_props_adjusted[comp_name] = original_value  # Store original for later
    
    # Update bounds for algorithm with range
    comp_idx = all_component_names.index(comp_name)
    component_bounds_adjusted[comp_idx] = (min_value, max_value)
    
    print(f"{comp_name}:")
    print(f"  Original: {original_value:.4f} ({original_value*100:.1f}%)")
    print(f"  Fraction of fixed total: {fraction:.4f}")
    print(f"  Adjusted bounds: ({min_value:.4f}, {max_value:.4f})")
    print(f"  Range: {max_value - min_value:.4f}")

# Verify the adjusted bounds
sum_min_adjusted = sum(bound[0] for bound in component_bounds_adjusted)
sum_max_adjusted = sum(bound[1] for bound in component_bounds_adjusted)
print(f"\n✓ Sum of adjusted MIN bounds = {sum_min_adjusted:.6f}")
print(f"✓ Sum of adjusted MAX bounds = {sum_max_adjusted:.6f}")
print(f"✓ Flexibility range = {sum_max_adjusted - sum_min_adjusted:.6f}")

if sum_min_adjusted >= 1.0:
    print("⚠️ WARNING: Sum of minimums >= 1.0 - No flexibility!")
elif sum_max_adjusted <= 1.0:
    print("⚠️ WARNING: Sum of maximums <= 1.0 - Infeasible!")
else:
    print("✅ Bounds are valid for mixture optimization")

print("\nStep 5: Run DOE algorithms with adjusted bounds")
doe_methods = ["d-optimal", "simplex-lattice", "extreme-vertices"]

for method in doe_methods[:1]:  # Test just d-optimal for now
    print(f"\n{'='*40}")
    print(f"Testing {method.upper()}")
    print(f"{'='*40}")
    
    try:
        # Create mixture design with adjusted bounds
        mixture = EnhancedMixtureDesign(
            n_components=n_components,
            component_names=all_component_names,
            component_bounds=component_bounds_adjusted,
            use_parts_mode=False  # Already converted
        )
        
        # Generate design
        design = mixture.generate_mixture_design(
            design_type=method,
            n_runs=5,
            model_type="linear" if method == "d-optimal" else None
        )
        
        print(f"✅ Generated {len(design)} points")
        print("\nRaw output from algorithm:")
        for i, point in enumerate(design):
            print(f"Mix {i+1}: {point.round(4)} (sum = {sum(point):.6f})")
            
        # Step 6: Replace fixed components with original values
        print("\nStep 6: Replace fixed components with original proportions")
        design_corrected = design.copy()
        
        for row_idx in range(len(design_corrected)):
            for comp_name, original_value in fixed_components_props_original.items():
                comp_idx = all_component_names.index(comp_name)
                design_corrected[row_idx, comp_idx] = original_value
        
        print("\nAfter replacing fixed components:")
        for i, point in enumerate(design_corrected):
            print(f"Mix {i+1}: {point.round(4)} (sum = {sum(point):.6f})")
        
        # Step 7: Final normalization
        print("\nStep 7: Final normalization to ensure sum = 1.0")
        design_normalized = design_corrected / design_corrected.sum(axis=1)[:, np.newaxis]
        
        print("\nFinal normalized design:")
        for i, point in enumerate(design_normalized):
            print(f"Mix {i+1}:")
            for j, comp_name in enumerate(all_component_names):
                if comp_name in fixed_component_names:
                    print(f"  {comp_name}: {point[j]:.4f} (FIXED)")
                else:
                    print(f"  {comp_name}: {point[j]:.4f}")
            print(f"  SUM: {sum(point):.6f}")
        
        # Step 8: Convert to batch quantities
        print("\nStep 8: Convert to batch quantities")
        batch_size = 100.0
        batch_quantities = design_normalized * batch_size
        
        print(f"\nBatch quantities (batch size = {batch_size}):")
        for i, row in enumerate(batch_quantities):
            print(f"\nMix {i+1}:")
            for j, comp_name in enumerate(all_component_names):
                print(f"  {comp_name}: {row[j]:.2f}")
            print(f"  TOTAL: {sum(row):.2f}")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

print("\n" + "="*60)
print("SUMMARY OF SOLUTION:")
print("="*60)
print("1. Calculate free space = sum_max_variable - sum_min_variable")
print("2. Distribute free space among fixed components proportionally")
print("3. Pass adjusted bounds to DOE algorithm")
print("4. Replace fixed components with original values after generation")
print("5. Final normalization ensures sum = 1.0")
print("6. Convert to batch quantities")
print("\nThis allows the algorithm to work while maintaining fixed ratios!")
