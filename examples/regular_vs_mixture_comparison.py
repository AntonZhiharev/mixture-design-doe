"""
Direct comparison between Regular Optimal Design and Mixture Design
Demonstrates how each handles components with parts constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from base_doe import OptimalDOE  # Regular Optimal Design
from mixture_design_optimization import OptimizedMixtureDesign  # Our improved Mixture Design

def main():
    # Number of components/factors
    n_components = 3
    
    # Define bounds in parts mode - these are the same for both designs
    component_bounds = [
        (0.0, 1.0),    # Component 1: parts between 1 and 5
        (0.0, 1.0),    # Component 2: parts between 2 and 8
        (0.0, 0.1)     # Component 3: parts between 0.5 and 3
    ]
    
    component_names = ["A", "B", "C"]
    
    # Number of runs
    n_runs = 15
    
    # Model type - use linear to ensure fair comparison
    model_type = "linear"
    
    print("=" * 70)
    print("DIRECT COMPARISON: REGULAR OPTIMAL DESIGN VS. MIXTURE DESIGN")
    print("=" * 70)
    print("\nComponent bounds (in parts):")
    for i, (name, (lower, upper)) in enumerate(zip(component_names, component_bounds)):
        print(f"  {name}: {lower} to {upper} parts")
    
    print("\n1. REGULAR OPTIMAL DESIGN")
    print("-" * 50)
    
    # Create a Regular Optimal Design (not a mixture design)
    # This treats factors independently without the constraint that they sum to 1
    regular_doe = OptimalDOE(
        n_factors=n_components,
        factor_ranges=component_bounds
    )
    
    # Generate a D-optimal design
    regular_design = regular_doe.generate_d_optimal(
        n_runs=n_runs,
        model_order=1 if model_type == "linear" else 2,
        random_seed=42
    )
    
    # Calculate D-efficiency directly in parts space
    regular_d_eff = regular_doe.d_efficiency(regular_design, model_order=1 if model_type == "linear" else 2)
    print(f"Regular D-optimal design D-efficiency: {regular_d_eff:.6f}")
    
    print("\n2. MIXTURE DESIGN WITH PARTS MODE")
    print("-" * 50)
    
    # Create Mixture Design with parts mode
    mixture_design_generator = OptimizedMixtureDesign(
        n_components=n_components,
        component_names=component_names,
        component_bounds=component_bounds,
        use_parts_mode=True,
        vertex_penalty=0.0,  # No penalties to focus on the core algorithm
        edge_penalty=0.0
    )
    
    # Generate a D-optimal mixture design
    mixture_design_parts = mixture_design_generator.generate_d_optimal(
        n_runs=n_runs,
        model_type=model_type,
        random_seed=42
    )
    
    # Get the design in parts (not proportions)
    def proportions_to_parts(proportions, bounds):
        """Convert proportions to parts based on the bounds"""
        parts = np.zeros_like(proportions)
        for i, (prop, (lower, upper)) in enumerate(zip(proportions, bounds)):
            parts[i] = lower + prop * (upper - lower)
        return parts
    
    # Convert mixture design proportions back to parts for comparison
    mixture_design = np.array([proportions_to_parts(point, component_bounds) 
                               for point in mixture_design_parts])
    
    # Analyze the designs
    print("\n3. ANALYSIS OF RESULTS")
    print("-" * 50)
    
    # Compare design spaces
    print("\nDesign Space Comparison:")
    print(f"{'Design Type':<20} {'Key Characteristics'}")
    print(f"{'-'*20} {'-'*50}")
    print(f"{'Regular Optimal':<20} {'- Factors are independent'}")
    print(f"{'':<20} {'- No constraint that factors sum to a constant'}")
    print(f"{'':<20} {'- Optimizes directly in parts space'}")
    print(f"{'':<20} {'- Can explore the full hypercube of the design space'}")
    
    print(f"\n{'Mixture Design':<20} {'- Components must sum to a constant'}")
    print(f"{'':<20} {'- Works in a constrained space (the simplex)'}")
    print(f"{'':<20} {'- Normalizes parts to proportions during optimization'}")
    print(f"{'':<20} {'- Explores only a slice of the full design space'}")
    
    # Compare design points distribution
    print("\nPoints Distribution in Parts Space:")
    
    # Calculate statistics
    print(f"\n{'Component':<10} {'Regular Min':<15} {'Regular Max':<15} {'Mixture Min':<15} {'Mixture Max':<15} {'Bound':<15}")
    print(f"{'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    for i, (name, (lower, upper)) in enumerate(zip(component_names, component_bounds)):
        reg_min = np.min(regular_design[:, i])
        reg_max = np.max(regular_design[:, i])
        mix_min = np.min(mixture_design[:, i])
        mix_max = np.max(mixture_design[:, i])
        print(f"{name:<10} {reg_min:<15.4f} {reg_max:<15.4f} {mix_min:<15.4f} {mix_max:<15.4f} {lower}-{upper}")
    
    # Calculate total parts for each design point
    regular_totals = np.sum(regular_design, axis=1)
    mixture_totals = np.sum(mixture_design, axis=1)
    
    print(f"\nTotal parts per point:")
    print(f"{'Design Type':<20} {'Min':<15} {'Max':<15} {'Average':<15} {'Std Dev':<15}")
    print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    print(f"{'Regular Optimal':<20} {np.min(regular_totals):<15.4f} {np.max(regular_totals):<15.4f} {np.mean(regular_totals):<15.4f} {np.std(regular_totals):<15.4f}")
    print(f"{'Mixture Design':<20} {np.min(mixture_totals):<15.4f} {np.max(mixture_totals):<15.4f} {np.mean(mixture_totals):<15.4f} {np.std(mixture_totals):<15.4f}")
    
    # Plot the designs - comparing components 1 and 2
    plt.figure(figsize=(12, 6))
    
    # Regular Optimal Design
    plt.subplot(1, 2, 1)
    plt.scatter(regular_design[:, 0], regular_design[:, 1], color='blue', alpha=0.7)
    plt.title("Regular Optimal Design (Parts Space)")
    plt.xlabel(f"{component_names[0]} Parts")
    plt.ylabel(f"{component_names[1]} Parts")
    plt.grid(True)
    
    # Draw the bounds
    plt.axhline(y=component_bounds[1][0], color='gray', linestyle='--')
    plt.axhline(y=component_bounds[1][1], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][0], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][1], color='gray', linestyle='--')
    
    # Mixture Design
    plt.subplot(1, 2, 2)
    plt.scatter(mixture_design[:, 0], mixture_design[:, 1], color='red', alpha=0.7)
    plt.title("Mixture Design (Parts Space)")
    plt.xlabel(f"{component_names[0]} Parts")
    plt.ylabel(f"{component_names[1]} Parts")
    plt.grid(True)
    
    # Draw the bounds
    plt.axhline(y=component_bounds[1][0], color='gray', linestyle='--')
    plt.axhline(y=component_bounds[1][1], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][0], color='gray', linestyle='--')
    plt.axvline(x=component_bounds[0][1], color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("regular_vs_mixture_comparison.png")
    print("\nComparison plot saved as 'regular_vs_mixture_comparison.png'")
    
    # 3D Scatterplot if possible
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 6))
        
        # Regular Optimal Design
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(regular_design[:, 0], regular_design[:, 1], regular_design[:, 2], color='blue', alpha=0.7)
        ax1.set_title("Regular Optimal Design (3D)")
        ax1.set_xlabel(component_names[0])
        ax1.set_ylabel(component_names[1])
        ax1.set_zlabel(component_names[2])
        
        # Mixture Design
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(mixture_design[:, 0], mixture_design[:, 1], mixture_design[:, 2], color='red', alpha=0.7)
        ax2.set_title("Mixture Design (3D)")
        ax2.set_xlabel(component_names[0])
        ax2.set_ylabel(component_names[1])
        ax2.set_zlabel(component_names[2])
        
        plt.tight_layout()
        plt.savefig("regular_vs_mixture_comparison_3d.png")
        print("3D comparison plot saved as 'regular_vs_mixture_comparison_3d.png'")
    except:
        print("Could not create 3D plot")
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES AND RECOMMENDATIONS")
    print("=" * 70)
    print("""
WHY REGULAR OPTIMAL DESIGN HAS HIGHER D-EFFICIENCY:

1. DESIGN SPACE FREEDOM
   - Regular Optimal Design operates in the unconstrained parts space
   - It can place points anywhere within the hypercube defined by the bounds
   - It doesn't have to enforce the constraint that components sum to a constant
   - This allows it to explore a larger design space and achieve higher D-efficiency

2. DIRECT OPTIMIZATION
   - Regular Optimal Design optimizes directly in the parts space
   - It doesn't need to convert between parts and proportions
   - The D-efficiency calculation is more straightforward
   - The algorithm can more directly maximize determinant-based criteria

3. INDEPENDENCE OF FACTORS
   - Regular Optimal Design treats each factor as independent
   - It can vary one factor without affecting others
   - This gives more degrees of freedom for optimization
   - The design matrix has better statistical properties

WHEN TO USE EACH APPROACH:

- Use Regular Optimal Design when:
  * Components don't need to sum to a constant
  * Factors can be varied independently
  * You need higher statistical efficiency
  * You're primarily interested in screening effects

- Use Mixture Design when:
  * Components must sum to a constant (true mixtures)
  * You need to model mixture-specific effects (synergism, antagonism)
  * The design space is inherently constrained to a simplex
  * You're studying formulations where proportions matter

POTENTIAL HYBRID APPROACH:

A hybrid approach could combine strengths of both methods:
1. Start with Regular Optimal Design to explore the parts space efficiently
2. Project or transform points to satisfy mixture constraints
3. Use mixture-specific model terms to capture interaction effects
4. Apply post-optimization to ensure mixture constraints are satisfied

This would maintain higher D-efficiency while respecting the mixture nature of the experiment.
""")

if __name__ == "__main__":
    main()
