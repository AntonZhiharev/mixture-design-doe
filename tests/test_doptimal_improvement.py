"""
Test to demonstrate and fix the D-optimal design issue where it only generates corner points
"""

import numpy as np
import pandas as pd
from src.core.refactored_mixture_design import MixtureDesign
import matplotlib.pyplot as plt


def analyze_design_points(design_df, design_name):
    """Analyze where the design points are located"""
    X = design_df.values
    n_components = X.shape[1]
    
    # Check if points are only at corners (vertices)
    corner_points = []
    for i in range(n_components):
        corner = np.zeros(n_components)
        corner[i] = 1.0
        corner_points.append(corner)
    
    # Count how many points are at corners
    n_corner_points = 0
    tolerance = 1e-6
    
    for point in X:
        for corner in corner_points:
            if np.allclose(point, corner, atol=tolerance):
                n_corner_points += 1
                break
    
    # Check for edge points (two components sum to 1)
    n_edge_points = 0
    for point in X:
        non_zero_components = np.sum(point > tolerance)
        if non_zero_components == 2:
            n_edge_points += 1
    
    # Check for interior points (all components > 0)
    n_interior_points = 0
    for point in X:
        if np.all(point > tolerance):
            n_interior_points += 1
    
    print(f"\n{design_name} Analysis:")
    print(f"Total points: {len(X)}")
    print(f"Corner points: {n_corner_points} ({n_corner_points/len(X)*100:.1f}%)")
    print(f"Edge points: {n_edge_points} ({n_edge_points/len(X)*100:.1f}%)")
    print(f"Interior points: {n_interior_points} ({n_interior_points/len(X)*100:.1f}%)")
    
    return n_corner_points, n_edge_points, n_interior_points


def test_current_doptimal():
    """Test the current D-optimal design to show it only generates corner points"""
    print("=" * 60)
    print("Testing Current D-Optimal Design")
    print("=" * 60)
    
    # Create mixture design with current D-optimal
    md = MixtureDesign(n_components=3)
    
    # Generate D-optimal design
    design_df = md.create_design(n_runs=10, design_type='d-optimal')
    print("\nCurrent D-optimal design points:")
    print(design_df)
    
    # Calculate D-efficiency
    X = design_df.values
    d_eff = md.calculate_d_efficiency(X)
    print(f"\nD-efficiency: {d_eff:.4f}")
    
    # Analyze point distribution
    analyze_design_points(design_df, "Current D-Optimal")
    
    return design_df, d_eff


def create_improved_doptimal(n_components, n_runs, include_interior=True, 
                           include_edges=True, augment_corners=True):
    """
    Create an improved D-optimal design that includes interior and edge points
    """
    md = MixtureDesign(n_components=n_components)
    
    # Start with a candidate set that includes various types of points
    candidate_points = []
    
    # 1. Add corner points (vertices)
    for i in range(n_components):
        corner = np.zeros(n_components)
        corner[i] = 1.0
        candidate_points.append(corner)
    
    # 2. Add edge centroids (midpoints of edges)
    if include_edges:
        for i in range(n_components):
            for j in range(i + 1, n_components):
                edge_point = np.zeros(n_components)
                edge_point[i] = 0.5
                edge_point[j] = 0.5
                candidate_points.append(edge_point)
    
    # 3. Add face centroids (for 3-component mixtures)
    if n_components >= 3 and include_interior:
        # Overall centroid
        centroid = np.ones(n_components) / n_components
        candidate_points.append(centroid)
        
        # Add points at various positions
        if n_components == 3:
            # Add points at 1/3, 2/3 positions
            positions = [1/3, 2/3]
            for p1 in positions:
                for p2 in positions:
                    p3 = 1 - p1 - p2
                    if p3 >= 0 and p3 <= 1:
                        candidate_points.append(np.array([p1, p2, p3]))
    
    # 4. Add augmented points if requested
    if augment_corners:
        # Add slightly interior points near corners
        epsilon = 0.1
        for i in range(n_components):
            augmented = np.ones(n_components) * (epsilon / (n_components - 1))
            augmented[i] = 1 - epsilon
            candidate_points.append(augmented)
    
    # Convert to numpy array
    candidate_set = np.array(candidate_points)
    
    # Use D-optimal algorithm to select best subset
    from src.algorithms.mixture_algorithms import MixtureAlgorithms
    algo = MixtureAlgorithms(n_components=n_components)
    
    # Select optimal points using coordinate exchange
    selected_indices = []
    selected_points = []
    
    # Start with a space-filling design
    initial_design = algo.simplex_lattice(degree=2)
    if len(initial_design) > n_runs:
        # Randomly select initial points
        indices = np.random.choice(len(initial_design), n_runs, replace=False)
        current_design = initial_design[indices]
    else:
        current_design = initial_design
        # Fill remaining spots from candidate set
        remaining = n_runs - len(current_design)
        if remaining > 0:
            unused_candidates = candidate_set[~np.isin(np.arange(len(candidate_set)), 
                                                      selected_indices)]
            if len(unused_candidates) > 0:
                add_indices = np.random.choice(len(unused_candidates), 
                                             min(remaining, len(unused_candidates)), 
                                             replace=False)
                current_design = np.vstack([current_design, unused_candidates[add_indices]])
    
    # Optimize using coordinate exchange
    max_iterations = 100
    for iteration in range(max_iterations):
        improved = False
        current_d_eff = md.calculate_d_efficiency(current_design)
        
        # Try to exchange each point
        for i in range(len(current_design)):
            best_d_eff = current_d_eff
            best_point = current_design[i].copy()
            
            # Try each candidate point
            for candidate in candidate_set:
                # Skip if already in design
                if any(np.allclose(candidate, point) for j, point in enumerate(current_design) if j != i):
                    continue
                
                # Try this candidate
                test_design = current_design.copy()
                test_design[i] = candidate
                test_d_eff = md.calculate_d_efficiency(test_design)
                
                if test_d_eff > best_d_eff:
                    best_d_eff = test_d_eff
                    best_point = candidate
                    improved = True
            
            current_design[i] = best_point
            current_d_eff = best_d_eff
        
        if not improved:
            break
    
    # Convert to DataFrame
    columns = [f'X{i+1}' for i in range(n_components)]
    design_df = pd.DataFrame(current_design, columns=columns)
    
    return design_df


def test_improved_doptimal():
    """Test the improved D-optimal design"""
    print("\n" + "=" * 60)
    print("Testing Improved D-Optimal Design")
    print("=" * 60)
    
    # Create improved design
    design_df = create_improved_doptimal(n_components=3, n_runs=10)
    print("\nImproved D-optimal design points:")
    print(design_df)
    
    # Calculate D-efficiency
    md = MixtureDesign(n_components=3)
    X = design_df.values
    d_eff = md.calculate_d_efficiency(X)
    print(f"\nD-efficiency: {d_eff:.4f}")
    
    # Analyze point distribution
    analyze_design_points(design_df, "Improved D-Optimal")
    
    return design_df, d_eff


def visualize_designs(design1_df, design2_df, title1="Current", title2="Improved"):
    """Visualize the two designs side by side"""
    if design1_df.shape[1] != 3:
        print("Visualization only available for 3-component mixtures")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot current design
    ax1.set_title(f"{title1} D-Optimal Design")
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    
    # Plot triangle boundaries
    triangle = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    ax1.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.3)
    
    # Plot design points
    ax1.scatter(design1_df['X1'], design1_df['X2'], s=100, c='red', alpha=0.7)
    
    # Plot improved design
    ax2.set_title(f"{title2} D-Optimal Design")
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    
    # Plot triangle boundaries
    ax2.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.3)
    
    # Plot design points
    ax2.scatter(design2_df['X1'], design2_df['X2'], s=100, c='blue', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('mixture-design-doe/doptimal_comparison.png', dpi=150)
    plt.close()
    print("\nVisualization saved as 'doptimal_comparison.png'")


if __name__ == "__main__":
    # Test current D-optimal
    current_design, current_d_eff = test_current_doptimal()
    
    # Test improved D-optimal
    improved_design, improved_d_eff = test_improved_doptimal()
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Current D-optimal D-efficiency: {current_d_eff:.4f}")
    print(f"Improved D-optimal D-efficiency: {improved_d_eff:.4f}")
    print(f"Improvement: {(improved_d_eff - current_d_eff) / current_d_eff * 100:.1f}%")
    
    # Visualize designs
    visualize_designs(current_design, improved_design)
