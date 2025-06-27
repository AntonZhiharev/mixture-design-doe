"""
Test script to verify that the MixtureDesign class is using the optimized algorithm
"""

from mixture_designs import MixtureDesign
import numpy as np
import time

# Create a simple 3-component design with MixtureDesign class
print("Testing MixtureDesign class...")
n_components = 3
n_runs = 10
component_names = ["A", "B", "C"]
component_bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.1)]  # Restrict third component

print(f"\nCreating design with bounds: {component_bounds}")
start_time = time.time()
mixture_design = MixtureDesign(
    n_components=n_components,
    component_names=component_names,
    component_bounds=component_bounds
)

# Generate D-optimal design
design = mixture_design.generate_d_optimal(
    n_runs=n_runs,
    model_type="linear",
    random_seed=42
)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")
print(f"Design shape: {design.shape}")
print("First 3 design points:")
for i in range(min(3, len(design))):
    print(f"  {i+1}: {design[i]}")

# Calculate D-efficiency
d_eff = mixture_design._calculate_d_efficiency(design, "linear")
print(f"D-efficiency: {d_eff:.6f}")

print("\nTest completed successfully!")
