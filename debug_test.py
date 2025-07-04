"""Simple debug test for the integrated anti-clustering solution."""

try:
    print("Starting debug test...")
    
    from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign
    print("✓ Import successful")
    
    design = FixedPartsMixtureDesign(
        component_names=['A', 'B', 'C'],
        fixed_parts={'A': 45.0},
        variable_bounds={'B': (0, 20), 'C': (0, 20)}
    )
    print("✓ Design creation successful")
    print(f"✓ Anti-clustering enabled: {design.enable_anti_clustering}")
    
    df = design.generate_design(n_runs=8, design_type='d-optimal', random_seed=42)
    print("✓ Design generation successful")
    print(f"✓ Shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    print("\n🎉 DEBUG TEST PASSED!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
