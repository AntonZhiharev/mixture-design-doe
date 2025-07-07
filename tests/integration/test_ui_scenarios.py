"""
Test the exact UI scenario to reproduce the boundary violation issue
"""

import numpy as np
import pandas as pd
from src.core.simplified_mixture_design import DOptimalMixtureDesign


def test_exact_ui_scenario():
    """Test the exact scenario from the UI that's causing boundary violations"""
    print("="*80)
    print("TESTING EXACT UI SCENARIO - PARTS MODE WITH COMPONENT BOUNDS")
    print("="*80)
    
    # Exact parameters from UI
    n_components = 3
    component_names = ['Component_A', 'Component_B', 'Component_C']
    use_parts_mode = True
    
    # Typical component bounds that cause the issue
    component_bounds = [
        (0.1, 5.0),   # Component A: 0.1 to 5.0 parts
        (0.2, 3.0),   # Component B: 0.2 to 3.0 parts  
        (0.1, 2.0)    # Component C: 0.1 to 2.0 parts
    ]
    
    print(f"UI Parameters:")
    print(f"  Components: {n_components}")
    print(f"  Names: {component_names}")
    print(f"  Parts mode: {use_parts_mode}")
    print(f"  Bounds: {component_bounds}")
    
    # Create exactly like the UI does
    print(f"\nCreating DOptimalMixtureDesign exactly like UI...")
    
    d_optimal_designer = DOptimalMixtureDesign(
        n_components=n_components, 
        component_names=component_names, 
        use_parts_mode=use_parts_mode, 
        component_bounds=component_bounds, 
        fixed_components={}  # No fixed components
    )
    
    print(f"✅ Created DOptimalMixtureDesign")
    print(f"   use_parts_mode: {d_optimal_designer.use_parts_mode}")
    print(f"   component_bounds: {d_optimal_designer.component_bounds}")
    print(f"   original_bounds: {getattr(d_optimal_designer, 'original_bounds', 'Not set')}")
    
    # Generate design exactly like UI does
    print(f"\nGenerating design...")
    
    design_params = {
        'n_runs': 10,
        'include_interior': True,
        'model_type': 'quadratic'
    }
    
    design_df = d_optimal_designer.generate_design(**design_params)
    
    print(f"✅ Generated design with shape: {design_df.shape}")
    print(f"✅ Design columns: {list(design_df.columns)}")
    
    # Check if OptimalDesignGenerator was used with component_ranges
    if hasattr(d_optimal_designer, '_last_generator'):
        generator = d_optimal_designer._last_generator
        print(f"\nOptimalDesignGenerator details:")
        print(f"   component_ranges passed: {generator.component_ranges}")
        print(f"   proportional_ranges available: {generator.proportional_ranges is not None}")
        
        if generator.proportional_ranges is not None:
            print(f"   proportional_ranges: {generator.proportional_ranges}")
            print(f"✅ Proportional parts functionality WAS activated")
        else:
            print(f"❌ Proportional parts functionality was NOT activated")
            
        # Test actual boundary violations
        print(f"\nTesting boundary violations...")
        design_points = generator.design_points
        
        # Convert design points to parts and check boundaries
        try:
            parts_points, normalized_points = generator.convert_to_parts(component_bounds)
            
            print(f"Parts conversion successful:")
            print(f"  Parts shape: {np.array(parts_points).shape}")
            print(f"  Normalized shape: {np.array(normalized_points).shape}")
            
            # Check boundary violations in parts
            violations = 0
            for i, parts_point in enumerate(parts_points):
                for j, parts_val in enumerate(parts_point):
                    min_val, max_val = component_bounds[j]
                    if parts_val < min_val - 1e-10 or parts_val > max_val + 1e-10:
                        violations += 1
                        print(f"❌ VIOLATION: Run {i+1}, Component {j+1}: {parts_val:.6f} outside [{min_val}, {max_val}]")
            
            if violations == 0:
                print(f"✅ NO boundary violations found in parts conversion")
            else:
                print(f"❌ Found {violations} boundary violations!")
                
                # Show first few parts points for debugging
                print(f"\nFirst 5 parts points:")
                for i in range(min(5, len(parts_points))):
                    parts_str = [f"{val:.3f}" for val in parts_points[i]]
                    print(f"  Run {i+1}: [{', '.join(parts_str)}]")
                
        except Exception as e:
            print(f"❌ Parts conversion failed: {e}")
    else:
        print(f"❌ No OptimalDesignGenerator instance available")
    
    # Display first few design points for inspection
    print(f"\nFirst 5 design points (proportions):")
    for i in range(min(5, len(design_df))):
        props_str = [f"{val:.3f}" for val in design_df.iloc[i]]
        print(f"  Run {i+1}: [{', '.join(props_str)}]")
    
    return design_df, d_optimal_designer


if __name__ == "__main__":
    design, designer = test_exact_ui_scenario()
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if hasattr(designer, '_last_generator') and designer._last_generator.proportional_ranges is not None:
        print("✅ Proportional parts fix IS activated")
        print("✅ Component boundaries should be respected")
    else:
        print("❌ Proportional parts fix is NOT activated")
        print("❌ This explains the boundary violations in the UI!")
        
        print(f"\n🔧 DEBUGGING INFO:")
        print(f"   designer.use_parts_mode: {designer.use_parts_mode}")
        print(f"   designer.component_bounds: {designer.component_bounds}")
        print(f"   hasattr original_bounds: {hasattr(designer, 'original_bounds')}")
        if hasattr(designer, 'original_bounds'):
            print(f"   designer.original_bounds: {designer.original_bounds}")
