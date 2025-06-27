"""
Test script to compare performance between regular DOE and optimized mixture design
"""

import numpy as np
import time
from base_doe import OptimalDOE
from mixture_design_optimization import OptimizedMixtureDesign

def test_performance(n_components=3, n_runs=15, model_type="linear", random_seed=42, custom_bounds=None):
    """
    Test performance difference between regular DOE and optimized mixture design
    """
    print(f"\nTesting with {n_components} components, {n_runs} runs, {model_type} model")
    
    # Create component bounds (all positive for fair comparison)
    if custom_bounds:
        component_bounds = custom_bounds
        print(f"Using custom bounds: {component_bounds}")
    else:
        component_bounds = [(0.0, 1.0)] * n_components
    
    component_names = [f"Component_{i+1}" for i in range(n_components)]
    
    # Create regular DOE design
    print("\n--- Regular Optimal Design ---")
    start_time = time.time()
    regular_doe = OptimalDOE(n_components, component_bounds)
    regular_result = regular_doe.generate_d_optimal(n_runs, 1 if model_type == "linear" else 2, random_seed=random_seed)
    
    # Normalize to ensure sum is exactly 1.0
    regular_result = regular_result / regular_result.sum(axis=1)[:, np.newaxis]
    
    regular_time = time.time() - start_time
    print(f"Time taken: {regular_time:.2f} seconds")
    
    # Create temporary mixture design to calculate efficiency
    temp_mixture = OptimizedMixtureDesign(n_components, component_names, component_bounds)
    regular_efficiency = temp_mixture._calculate_d_efficiency(regular_result, model_type)
    print(f"D-efficiency: {regular_efficiency:.6f}")
    
    # Also calculate using the regular DOE method for comparison
    regular_doe_efficiency = regular_doe.d_efficiency(regular_result, 1 if model_type == "linear" else 2)
    print(f"Regular DOE D-efficiency calculation: {regular_doe_efficiency:.6f}")
    
    # Create optimized mixture design
    print("\n--- Optimized Mixture Design ---")
    start_time = time.time()
    optimized_design = OptimizedMixtureDesign(n_components, component_names, component_bounds)
    optimized_result = optimized_design.generate_d_optimal(n_runs, model_type, random_seed=random_seed)
    optimized_time = time.time() - start_time
    print(f"Time taken: {optimized_time:.2f} seconds")
    
    # Calculate efficiency
    optimized_efficiency = optimized_design._calculate_d_efficiency(optimized_result, model_type)
    print(f"D-efficiency: {optimized_efficiency:.6f}")
    
    # Compare results
    print("\n--- Comparison ---")
    print(f"Speed improvement: {regular_time / optimized_time:.2f}x faster")
    
    # Safely calculate efficiency ratio
    if regular_efficiency > 0 and optimized_efficiency > 0:
        print(f"Efficiency ratio: {optimized_efficiency / regular_efficiency:.2f}x better")
    else:
        print(f"Efficiency comparison: Regular={regular_efficiency:.6f}, Optimized={optimized_efficiency:.6f}")
    
    # Print first few design points for comparison
    print("\nRegular DOE design (first 3 points):")
    for i in range(min(3, len(regular_result))):
        print(f"  {i+1}: {regular_result[i]}")
    
    print("\nOptimized design (first 3 points):")
    for i in range(min(3, len(optimized_result))):
        print(f"  {i+1}: {optimized_result[i]}")
    
    return regular_result, optimized_result, regular_efficiency, optimized_efficiency

if __name__ == "__main__":
    print("=== Comparing Regular DOE vs Optimized Mixture Design ===")
    
    # Test with standard bounds first for comparison
    print("\n*** TEST WITH STANDARD BOUNDS ***")
    test_performance(n_components=3, n_runs=15, model_type="linear")
    
    # Test with custom bounds for 3 components as requested
    print("\n*** TEST WITH CUSTOM BOUNDS ***")
    custom_bounds_3comp = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.1)]
    test_performance(n_components=3, n_runs=15, model_type="linear", custom_bounds=custom_bounds_3comp)
    
    # Test with 5 components
    print("\n*** TEST WITH 5 COMPONENTS ***")
    test_performance(n_components=5, n_runs=20, model_type="linear")
    # Commented out quadratic model test as requested
    # test_performance(n_components=3, n_runs=15, model_type="quadratic")
