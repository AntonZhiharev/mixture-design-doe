"""
Test script to verify that the import error is fixed
"""

# Try to import MixtureDesign from mixture_designs
try:
    from mixture_designs import MixtureDesign
    print("Successfully imported MixtureDesign from mixture_designs")
except ImportError as e:
    print(f"Import error: {e}")

# Try to import OptimizedMixtureDesign from mixture_design_optimization
try:
    from mixture_design_optimization import OptimizedMixtureDesign
    print("Successfully imported OptimizedMixtureDesign from mixture_design_optimization")
except ImportError as e:
    print(f"Import error: {e}")

# Try to use MixtureDesignGenerator to create a D-optimal design
try:
    from mixture_designs import MixtureDesignGenerator
    
    # Create a simple 3-component design
    n_components = 3
    n_runs = 10
    component_names = ["A", "B", "C"]
    component_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    
    print("\nCreating D-optimal design...")
    design, mixture_design = MixtureDesignGenerator.create_d_optimal(
        n_components=n_components,
        n_runs=n_runs,
        component_names=component_names,
        component_bounds=component_bounds,
        model_type="linear"
    )
    
    print(f"Design shape: {design.shape}")
    print("First 3 design points:")
    for i in range(min(3, len(design))):
        print(f"  {i+1}: {design[i]}")
    
    print("\nTest completed successfully!")
except Exception as e:
    print(f"Error: {e}")
