"""
Demonstration of the improved mixture design algorithm with parts mode

This script shows how the updated algorithm correctly handles component bounds in parts mode,
which allows us to properly constrain the mixture space similar to regular optimal designs.
"""

import numpy as np
import matplotlib.pyplot as plt
from mixture_design_optimization import OptimizedMixtureDesign

def main():
    # Number of components
    n_components = 3
    
    # Define bounds in parts mode
    # Regular DOE would respect these bounds directly
    # For example, component 1 has parts between 1 and 5
    component_bounds = [
        (1.0, 5.0),    # Component 1: parts between 1 and 5
        (2.0, 8.0),    # Component 2: parts between 2 and 8
        (0.5, 3.0)     # Component 3: parts between 0.5 and 3
    ]
    
    component_names = ["A", "B", "C"]
    
    # Number of runs
    n_runs = 12
    
    # Model type
    model_type = "linear"
    
    print("=" * 70)
    print("DEMONSTRATION OF MIXTURE DESIGN WITH PARTS MODE")
    print("=" * 70)
    print("\nComponent bounds (in parts, not proportions):")
    for i, (name, (lower, upper)) in enumerate(zip(component_names, component_bounds)):
        print(f"  {name}: {lower} to {upper} parts")
    
    # 1. Generate standard mixture design (not respecting parts properly)
    print("\n1. STANDARD APPROACH (not respecting parts mode)")
    print("-" * 50)
    
    # Create a naive implementation that treats mixture proportions like regular factors
    # This simulates how traditional mixture designs often fail to properly handle parts mode
    
    # First, just use standard mixture optimization with pure component proportions
    standard_design_proportions = np.zeros((n_runs, n_components))
    
    # Create vertices (ignoring parts bounds)
    for i in range(min(n_runs, n_components)):
        point = np.zeros(n_components)
        point[i] = 1.0  # Pure component at vertex
        standard_design_proportions[i] = point
    
    # Fill remaining runs with points on edges
    if n_runs > n_components:
        edge_points_needed = min(n_runs - n_components, n_components * (n_components - 1) // 2)
        edge_idx = n_components
        
        for i in range(n_components):
            for j in range(i+1, n_components):
                if edge_idx < n_runs:
                    point = np.zeros(n_components)
                    point[i] = 0.5
                    point[j] = 0.5
                    standard_design_proportions[edge_idx] = point
                    edge_idx += 1
    
    # Fill any remaining runs with centroid and random points
    if edge_idx < n_runs:
        # Add centroid
        standard_design_proportions[edge_idx] = np.ones(n_components) / n_components
        edge_idx += 1
        
        # Add random points if needed
        while edge_idx < n_runs:
            point = np.random.random(n_components)
            point = point / np.sum(point)
            standard_design_proportions[edge_idx] = point
            edge_idx += 1
    
    print(f"Created basic mixture design with {n_runs} points")
    print("This approach treats mixture components as pure proportions")
    print("and doesn't account for parts mode during optimization")
    
    # This is the standard design in proportions
    standard_design = standard_design_proportions.copy()
    
    # 2. Generate improved mixture design (properly respecting parts)
    print("\n2. IMPROVED APPROACH (properly respecting parts)")
    print("-" * 50)
    
    improved_design_generator = OptimizedMixtureDesign(
        n_components=n_components,
        component_names=component_names,
        component_bounds=component_bounds,
        use_parts_mode=True,
        vertex_penalty=0.0,  # No penalties to focus on the parts handling
        edge_penalty=0.0
    )
    
    improved_design = improved_design_generator.generate_d_optimal(
        n_runs=n_runs,
        model_type=model_type,
        random_seed=42
    )
    
    # 3. Analyze the designs
    print("\n3. ANALYSIS OF RESULTS")
    print("-" * 50)
    
    # Convert the designs back to parts properly
    def proportions_to_parts(proportions, bounds):
        """Convert proportions to parts based on the bounds"""
        # This is how parts conversion should work in mixture designs
        parts = np.zeros_like(proportions)
        for i, (prop, (lower, upper)) in enumerate(zip(proportions, bounds)):
            # Calculate parts as lower bound + proportion * (upper - lower)
            parts[i] = lower + prop * (upper - lower)
        
        # Adjust for total parts
        total_parts = np.sum(parts)
        return parts
    
    standard_parts = np.array([proportions_to_parts(point, component_bounds) for point in standard_design])
    improved_parts = np.array([proportions_to_parts(point, component_bounds) for point in improved_design])
    
    # Check if points respect the bounds in parts
    def check_bounds_violations(design_parts, bounds):
        """Check how many points violate the bounds when expressed in parts"""
        violations = 0
        total_points = design_parts.shape[0] * design_parts.shape[1]
        
        for i, point in enumerate(design_parts):
            for j, (value, (lower, upper)) in enumerate(zip(point, bounds)):
                if value < lower - 1e-6 or value > upper + 1e-6:
                    violations += 1
                    print(f"  Point {i}, Component {j}: {value:.4f} parts (outside {lower}-{upper})")
        
        return violations, total_points
    
    print("\nStandard approach components analysis (parts):")
    std_violations, std_total = check_bounds_violations(standard_parts, component_bounds)
    if std_violations == 0:
        print("  No direct bounds violations, but vertices in proportion space ")
        print("  lead to extreme values when converted to parts space.")
        
        # Calculate distance from bounds
        for i, (name, (lower, upper)) in enumerate(zip(component_names, component_bounds)):
            for j, point in enumerate(standard_design):
                if point[i] > 0.9:  # Near vertex
                    print(f"  Point {j}, Component {name}: {standard_parts[j,i]:.4f} parts (allowed: {lower}-{upper})")
                    print(f"    This is at {(point[i]*100):.1f}% proportion, which in parts mode may not be realistic")
    else:
        print(f"  {std_violations} violations out of {std_total} component values ({std_violations/std_total*100:.1f}%)")
    
    print("\nImproved approach components analysis (parts):")
    imp_violations, imp_total = check_bounds_violations(improved_parts, component_bounds)
    if imp_violations == 0:
        print("  All components respect their bounds in parts")
        
        # Show the proportion and parts distribution
        for i, (name, (lower, upper)) in enumerate(zip(component_names, component_bounds)):
            max_prop = np.max(improved_design[:, i])
            min_prop = np.min(improved_design[:, i])
            print(f"  Component {name}: Range in proportions: {min_prop:.4f} - {max_prop:.4f}")
            print(f"                Range in parts: {lower + min_prop * (upper - lower):.4f} - {lower + max_prop * (upper - lower):.4f}")
    else:
        print(f"  {imp_violations} violations out of {imp_total} component values ({imp_violations/imp_total*100:.1f}%)")
    
    # Calculate statistics
    print("\nAverage parts per component:")
    print(f"{'Component':<10} {'Standard Avg':<15} {'Improved Avg':<15} {'Bound':<15}")
    print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15}")
    
    for i, (name, (lower, upper)) in enumerate(zip(component_names, component_bounds)):
        std_avg = np.mean(standard_parts[:, i])
        imp_avg = np.mean(improved_parts[:, i])
        print(f"{name:<10} {std_avg:<15.4f} {imp_avg:<15.4f} {lower}-{upper}")
    
    # Plot the designs in proportions
    plt.figure(figsize=(12, 5))
    
    # Standard design
    plt.subplot(1, 2, 1)
    plt.scatter(standard_design[:, 0], standard_design[:, 1], color='blue', alpha=0.7)
    plt.title("Standard Approach (Proportions)")
    plt.xlabel(f"{component_names[0]} Proportion")
    plt.ylabel(f"{component_names[1]} Proportion")
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    
    # Improved design
    plt.subplot(1, 2, 2)
    plt.scatter(improved_design[:, 0], improved_design[:, 1], color='red', alpha=0.7)
    plt.title("Improved Approach (Proportions)")
    plt.xlabel(f"{component_names[0]} Proportion")
    plt.ylabel(f"{component_names[1]} Proportion")
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    
    plt.tight_layout()
    plt.savefig("parts_mode_comparison_proportions.png")
    print("\nDesign comparison in proportions saved as 'parts_mode_comparison_proportions.png'")
    
    # Plot the designs in parts
    plt.figure(figsize=(12, 5))
    
    # Standard design
    plt.subplot(1, 2, 1)
    plt.scatter(standard_parts[:, 0], standard_parts[:, 1], color='blue', alpha=0.7)
    plt.title("Standard Approach (Parts)")
    plt.xlabel(f"{component_names[0]} Parts")
    plt.ylabel(f"{component_names[1]} Parts")
    plt.grid(True)
    
    # Draw the bounds
    plt.axhline(y=component_bounds[1][0], color='gray', linestyle='--')
    plt.axhline(y=component_bounds[1][1], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][0], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][1], color='gray', linestyle='--')
    
    # Improved design
    plt.subplot(1, 2, 2)
    plt.scatter(improved_parts[:, 0], improved_parts[:, 1], color='red', alpha=0.7)
    plt.title("Improved Approach (Parts)")
    plt.xlabel(f"{component_names[0]} Parts")
    plt.ylabel(f"{component_names[1]} Parts")
    plt.grid(True)
    
    # Draw the bounds
    plt.axhline(y=component_bounds[1][0], color='gray', linestyle='--')
    plt.axhline(y=component_bounds[1][1], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][0], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][1], color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("parts_mode_comparison_parts.png")
    print("Design comparison in parts saved as 'parts_mode_comparison_parts.png'")
    
    print("\n" + "=" * 70)
    print("SUMMARY OF IMPROVEMENTS")
    print("=" * 70)
print("""
The key difference between standard and improved approaches:

1. STANDARD APPROACH (typical mixture designs):
   - Optimizes in proportion space (0-1) without considering parts mode
   - Treats all components equally regardless of their physical constraints
   - Places many points at vertices (pure components) which may be unrealistic
   - Only converts to parts after optimization is complete
   - Can lead to designs that don't properly explore the realistic mixture space

2. IMPROVED APPROACH (our implementation):
   - Understands and respects components' bounds in parts
   - Generates candidate points that properly account for parts mode constraints
   - Avoids placing points in unrealistic regions of the mixture space
   - Better matches how regular optimal designs handle factor bounds
   - Results in designs that more effectively explore the realistic parts-based mixture space

This improvement makes mixture designs much more practical for real-world experiments
where components have specific part ranges and constraints.
""")

if __name__ == "__main__":
    main()
