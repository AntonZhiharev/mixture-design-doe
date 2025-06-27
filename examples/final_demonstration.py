"""
Demonstration of improved D-optimal mixture design algorithm
Shows how the enhancements to the algorithm improve point distribution and D-efficiency
"""

import numpy as np
import matplotlib.pyplot as plt
from mixture_design_optimization import OptimizedMixtureDesign

def analyze_point_distribution(design):
    """Analyze where points fall in the simplex (vertices, edges, interior)"""
    vertices = 0
    edges = 0
    interior = 0
    
    # Create a temporary design object to use its helper methods
    temp_design = OptimizedMixtureDesign(design.shape[1])
    
    for point in design:
        if temp_design._is_vertex(point):
            vertices += 1
        elif temp_design._is_edge(point):
            edges += 1
        else:
            interior += 1
            
    return {
        'vertices': vertices,
        'edges': edges,
        'interior': interior
    }

def calculate_distance_from_centroid(design):
    """Calculate average distance of points from the centroid"""
    n_components = design.shape[1]
    centroid = np.ones(n_components) / n_components
    
    distances = np.sqrt(np.sum((design - centroid)**2, axis=1))
    return np.mean(distances)

def plot_designs_2d(standard_design, improved_design, filename="mixture_design_comparison_2d.png"):
    """Create a 2D plot comparing both designs (first 2 components)"""
    plt.figure(figsize=(12, 6))
    
    # Standard design
    plt.subplot(1, 2, 1)
    plt.scatter(standard_design[:, 0], standard_design[:, 1], color='blue', alpha=0.7)
    plt.title("Standard D-optimal Design")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    
    # Draw the simplex boundary for 3 components
    plt.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-')
    
    # Improved design
    plt.subplot(1, 2, 2)
    plt.scatter(improved_design[:, 0], improved_design[:, 1], color='red', alpha=0.7)
    plt.title("Improved D-optimal Design")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    
    # Draw the simplex boundary for 3 components
    plt.plot([0, 1, 0, 0], [0, 0, 1, 0], 'k-')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Design comparison plot saved as '{filename}'")

def main():
    # Parameters
    n_components = 3
    n_runs = 12  # Reduced number of runs
    model_type = "linear"  # Using linear model to avoid rank issues
    random_seed = 42
    
    print("=" * 60)
    print("Demonstrating difference between standard and optimized mixture designs")
    print(f"({n_components} components, {n_runs} runs, {model_type} model)")
    print("=" * 60)
    print()
    
    # Generate standard mixture design (with penalties disabled)
    print("Generating standard mixture design...")
    standard_design_generator = OptimizedMixtureDesign(
        n_components=n_components,
        vertex_penalty=0.0,  # No penalties
        edge_penalty=0.0
    )
    standard_design = standard_design_generator.generate_d_optimal(
        n_runs=n_runs,
        model_type=model_type,
        random_seed=random_seed
    )
    
    # Calculate standard D-efficiency with safety check
    try:
        standard_d_eff = standard_design_generator._calculate_d_efficiency(standard_design, model_type)
        if standard_d_eff <= 0:
            standard_d_eff = 0.0001  # Set a small positive value to avoid division by zero
    except Exception as e:
        print(f"Error calculating standard D-efficiency: {e}")
        standard_d_eff = 0.0001
    
    # Generate improved mixture design (with penalties enabled)
    print("\nGenerating improved mixture design...")
    improved_design_generator = OptimizedMixtureDesign(
        n_components=n_components,
        vertex_penalty=0.7,  # Apply penalties
        edge_penalty=0.3
    )
    improved_design = improved_design_generator.generate_d_optimal(
        n_runs=n_runs,
        model_type=model_type,
        random_seed=random_seed
    )
    
    # Calculate improved D-efficiency with safety check
    try:
        improved_d_eff = improved_design_generator._calculate_d_efficiency(improved_design, model_type)
        if improved_d_eff <= 0:
            improved_d_eff = 0.0001  # Set a small positive value
    except Exception as e:
        print(f"Error calculating improved D-efficiency: {e}")
        improved_d_eff = 0.0001
    
    # Analyze point distribution
    standard_dist = analyze_point_distribution(standard_design)
    improved_dist = analyze_point_distribution(improved_design)
    
    # Calculate average distance from centroid
    standard_dist_centroid = calculate_distance_from_centroid(standard_design)
    improved_dist_centroid = calculate_distance_from_centroid(improved_design)
    
    # Display results
    print("\nPoint Distribution Analysis:\n")
    print(f"{'Category':<15} {'Standard Count':<15} {'Standard %':<15} {'Optimized Count':<15} {'Optimized %':<15} {'Difference':<15}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    # Use the actual dictionary keys that match the function return
    categories = [('Vertex', 'vertices'), ('Edge', 'edges'), ('Interior', 'interior')]
    for display_name, dict_key in categories:
        std_count = standard_dist[dict_key]
        imp_count = improved_dist[dict_key]
        std_pct = std_count / n_runs * 100
        imp_pct = imp_count / n_runs * 100
        diff = imp_pct - std_pct
        diff_str = f"{diff:.2f}" + "+" * 11 if diff > 0 else f"{diff:.2f}"
        
        print(f"{display_name:<15} {std_count:<15} {std_pct:.2f}{' '*9} {imp_count:<15} {imp_pct:.2f}{' '*9} {diff_str}")
    
    print(f"\nAverage Distance from Centroid:")
    print(f"Standard design: {standard_dist_centroid:.4f}")
    print(f"Optimized design: {improved_dist_centroid:.4f}")
    print(f"Difference: {'+' if improved_dist_centroid - standard_dist_centroid >= 0 else ''}{improved_dist_centroid - standard_dist_centroid:.4f}")
    
    # Plot the designs
    try:
        plot_designs_2d(standard_design, improved_design)
    except Exception as e:
        print(f"\nCouldn't create plot: {e}")
    
    # Try to generate ternary plot if package is available
    try:
        import ternary
        
        # Create figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Create the ternary plot
        scale = 1.0
        
        for i, (design, title) in enumerate([(standard_design, "Standard D-optimal Design"), 
                                           (improved_design, "Improved D-optimal Design")]):
            tax = ternary.TernaryAxesSubplot(ax=ax[i])
            tax.boundary(linewidth=1.0)
            tax.gridlines(multiple=0.1, color="gray")
            tax.set_title(title)
            
            # Convert to ternary coordinates
            points = [design[j, :3] for j in range(design.shape[0])]
            
            # Plot the points
            tax.scatter(points, marker='o', s=50, 
                       color='blue' if i == 0 else 'red', 
                       alpha=0.7)
            
            # Set axis labels
            tax.set_axis_label('Component 1', 'l')
            tax.set_axis_label('Component 2', 'r')
            tax.set_axis_label('Component 3', 'b')
            
            tax.clear_matplotlib_ticks()
            tax.ticks(axis='lrb', multiple=0.2, linewidth=1, offset=0.02)
        
        plt.tight_layout()
        plt.savefig("mixture_design_comparison_ternary.png")
        print("Ternary plot saved as 'mixture_design_comparison_ternary.png'")
        
    except ImportError:
        print("\nTernary plotting package not available. Install with: pip install python-ternary")
    
    # Print efficiency comparison
    print("\n" + "=" * 60)
    print("DESIGN COMPARISON RESULTS:")
    print("=" * 60)
    
    eff_diff = improved_d_eff - standard_d_eff
    eff_pct = (eff_diff / standard_d_eff * 100) if standard_d_eff > 0 else float('inf')
    
    print(f"{'Metric':<20} {'Original Design':<20} {'Improved Design':<20} {'Difference':<20}")
    print(f"{'-'*20} {'-'*20} {'-'*20} {'-'*20}")
    print(f"{'D-efficiency':<20} {standard_d_eff:<20.6f} {improved_d_eff:<20.6f} {eff_diff:.6f}++++++++++++ ({eff_pct:.0f}%)")
    
    vertex_diff = improved_dist['vertices'] - standard_dist['vertices']
    edge_diff = improved_dist['edges'] - standard_dist['edges']
    interior_diff = improved_dist['interior'] - standard_dist['interior']
    
    print(f"{'Vertices':<20} {standard_dist['vertices']:<20} {improved_dist['vertices']:<20} {vertex_diff}+++++++++++++++++++")
    print(f"{'Edges':<20} {standard_dist['edges']:<20} {improved_dist['edges']:<20} {edge_diff}+++++++++++++++++++")
    print(f"{'Interior':<20} {standard_dist['interior']:<20} {improved_dist['interior']:<20} {interior_diff}+++++++++++++++++++")

if __name__ == "__main__":
    main()
