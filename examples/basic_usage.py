"""
Example usage of the restructured mixture design code
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesignGenerator

def example_simplex_lattice():
    """Example of simplex lattice design"""
    print("\n=== Simplex Lattice Design Example ===")
    
    # Create simplex lattice design
    n_components = 3
    component_names = ["A", "B", "C"]
    component_bounds = [(0.0, 0.8), (0.0, 0.7), (0.0, 0.6)]
    
    design, mixture_design = MixtureDesignGenerator.create_simplex_lattice(
        n_components=n_components,
        degree=2,
        component_names=component_names,
        component_bounds=component_bounds
    )
    
    # Print design
    print("\nSimplex Lattice Design:")
    for i, point in enumerate(design):
        print(f"Run {i+1}: {', '.join([f'{name}={val:.3f}' for name, val in zip(component_names, point)])}")
    
    # Export design to CSV
    MixtureDesignGenerator.export_design_to_csv(design, mixture_design, "simplex_lattice_design.csv")

def example_d_optimal():
    """Example of D-optimal design"""
    print("\n=== D-Optimal Design Example ===")
    
    # Create D-optimal design
    n_components = 4
    component_names = ["W", "X", "Y", "Z"]
    component_bounds = [(0.2, 0.8), (0.1, 0.5), (0.05, 0.3), (0.05, 0.2)]
    
    design, mixture_design = MixtureDesignGenerator.create_d_optimal(
        n_components=n_components,
        n_runs=10,
        component_names=component_names,
        component_bounds=component_bounds,
        model_type="quadratic",
        random_seed=42
    )
    
    # Print design
    print("\nD-Optimal Design:")
    for i, point in enumerate(design):
        print(f"Run {i+1}: {', '.join([f'{name}={val:.3f}' for name, val in zip(component_names, point)])}")

def example_fixed_parts():
    """Example of fixed parts design"""
    print("\n=== Fixed Parts Design Example ===")
    
    # Create fixed parts design
    n_components = 5
    component_names = ["A", "B", "C", "D", "E"]
    component_bounds = [(5, 20), (10, 30), (5, 15), (1, 5), (0.5, 2)]
    fixed_components = {"D": 3, "E": 1}
    
    design, mixture_design = MixtureDesignGenerator.create_fixed_parts_design(
        n_components=n_components,
        n_runs=8,
        component_names=component_names,
        component_bounds=component_bounds,
        fixed_components=fixed_components,
        design_type="d-optimal",
        model_type="quadratic",
        random_seed=42
    )
    
    # Print design
    print("\nFixed Parts Design (Proportions):")
    for i, point in enumerate(design):
        print(f"Run {i+1}: {', '.join([f'{name}={val:.3f}' for name, val in zip(component_names, point)])}")

if __name__ == "__main__":
    example_simplex_lattice()
    example_d_optimal()
    example_fixed_parts()
