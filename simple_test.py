print("Starting simple test...")

try:
    print("Step 1: Importing...")
    from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign
    print("Step 1: ✓ Import successful")
    
    print("Step 2: Creating design object...")
    design = FixedPartsMixtureDesign(
        component_names=['A', 'B', 'C'],
        fixed_parts={'A': 45.0},
        variable_bounds={'B': (0, 20), 'C': (0, 20)}
    )
    print("Step 2: ✓ Design object created")
    
    print("Step 3: Checking anti-clustering setting...")
    print(f"Anti-clustering enabled: {design.enable_anti_clustering}")
    print("Step 3: ✓ Anti-clustering check complete")
    
    print("\n🎉 SIMPLE TEST PASSED!")
    
except Exception as e:
    print(f"❌ Error at step: {e}")
    import traceback
    traceback.print_exc()
