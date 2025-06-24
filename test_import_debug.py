"""
Debug import issues step by step
"""

print("Step 1: Testing numpy import...")
import numpy as np
print("✅ numpy imported successfully")

print("Step 2: Testing base mixture import...")
try:
    from mixture_designs import MixtureDesign
    print("✅ MixtureDesign imported successfully")
except Exception as e:
    print(f"❌ MixtureDesign import failed: {e}")
    exit(1)

print("Step 3: Testing enhanced mixture import...")
try:
    from enhanced_mixture_designs import EnhancedMixtureDesign
    print("✅ EnhancedMixtureDesign imported successfully")
except Exception as e:
    print(f"❌ EnhancedMixtureDesign import failed: {e}")
    exit(1)

print("Step 4: Testing simple instantiation...")
try:
    component_names = ['A', 'B', 'C']
    component_bounds = [(0.1, 0.7), (0.1, 0.7), (0.2, 0.8)]
    
    mixture = EnhancedMixtureDesign(3, component_names, component_bounds)
    print("✅ Simple instantiation successful")
except Exception as e:
    print(f"❌ Simple instantiation failed: {e}")
    exit(1)

print("Step 5: Testing with fixed components...")
try:
    component_names = ['Comp1', 'Comp2', 'Fixed1']
    component_bounds_parts = [(0.1, 1.0), (0.1, 1.0), (0.05, 0.05)]
    fixed_parts = {'Fixed1': 0.05}
    
    enhanced_mixture = EnhancedMixtureDesign(
        3, 
        component_names, 
        component_bounds_parts, 
        use_parts_mode=True,
        fixed_components=fixed_parts
    )
    print("✅ Fixed components instantiation successful")
except Exception as e:
    print(f"❌ Fixed components instantiation failed: {e}")
    print(f"Error details: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("All tests completed!")
