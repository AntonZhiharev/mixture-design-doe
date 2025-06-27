"""
Test D-optimal design issue with simplified architecture
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.simplified_mixture_design import create_mixture_design, DOptimalMixtureDesign


def analyze_design(design_df, design_name):
    """Analyze design point distribution"""
    X = design_df.values
    n_components = X.shape[1]
    
    # Count point types
    corner_points = 0
    edge_points = 0
    interior_points = 0
    
    tolerance = 1e-6
    
    for point in X:
        non_zero = np.sum(point > tolerance)
        
        if non_zero == 1:
            corner_points += 1
        elif non_zero == 2:
            edge_points += 1
        else:
            interior_points += 1
    
    print(f"\n{design_name} Analysis:")
    print(f"Total points: {len(X)}")
    print(f"Corner points (vertices): {corner_points} ({corner_points/len(X)*100:.1f}%)")
    print(f"Edge points: {edge_points} ({edge_points/len(X)*100:.1f}%)")
    print(f"Interior points: {interior_points} ({interior_points/len(X)*100:.1f}%)")
    
    return corner_points, edge_points, interior_points


def calculate_d_efficiency(design_matrix):
    """Calculate D-efficiency of a design"""
    X = design_matrix
    
    try:
        # For mixture designs, we use the design matrix without intercept
        information_matrix = X.T @ X
        det_value = np.linalg.det(information_matrix)
        
        # D-efficiency relative to regular simplex design
        n_components = X.shape[1]
        n_runs = X.shape[0]
        
        # Calculate relative efficiency
        d_efficiency = (det_value / n_runs) ** (1/n_components)
        
        return d_efficiency
    except:
        return 0.0


def test_doptimal_comparison():
    """Compare D-optimal with and without interior points"""
    
    print("=" * 60)
    print("D-OPTIMAL DESIGN COMPARISON")
    print("=" * 60)
    
    n_components = 3
    n_runs = 10
    
    # Test 1: D-optimal without interior points
    print("\n1. D-Optimal WITHOUT interior points:")
    designer1 = DOptimalMixtureDesign(n_components)
    design1 = designer1.generate_design(n_runs=n_runs, include_interior=False)
    print(design1)
    analyze_design(design1, "D-Optimal (no interior)")
    d_eff1 = calculate_d_efficiency(design1.values)
    print(f"D-efficiency: {d_eff1:.4f}")
    
    # Test 2: D-optimal with interior points
    print("\n2. D-Optimal WITH interior points:")
    designer2 = DOptimalMixtureDesign(n_components)
    design2 = designer2.generate_design(n_runs=n_runs, include_interior=True)
    print(design2)
    analyze_design(design2, "D-Optimal (with interior)")
    d_eff2 = calculate_d_efficiency(design2.values)
    print(f"D-efficiency: {d_eff2:.4f}")
    
    # Compare with other designs
    print("\n3. Simplex Lattice (degree 2) for comparison:")
    design3 = create_mixture_design('simplex-lattice', n_components, degree=2)
    print(design3)
    analyze_design(design3, "Simplex Lattice")
    d_eff3 = calculate_d_efficiency(design3.values)
    print(f"D-efficiency: {d_eff3:.4f}")
    
    print("\n4. Simplex Centroid for comparison:")
    design4 = create_mixture_design('simplex-centroid', n_components)
    print(design4)
    analyze_design(design4, "Simplex Centroid")
    d_eff4 = calculate_d_efficiency(design4.values)
    print(f"D-efficiency: {d_eff4:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"D-Optimal (no interior) D-efficiency: {d_eff1:.4f}")
    print(f"D-Optimal (with interior) D-efficiency: {d_eff2:.4f}")
    print(f"Simplex Lattice D-efficiency: {d_eff3:.4f}")
    print(f"Simplex Centroid D-efficiency: {d_eff4:.4f}")
    
    if d_eff2 > d_eff1:
        improvement = (d_eff2 - d_eff1) / d_eff1 * 100
        print(f"\n✅ Including interior points improved D-efficiency by {improvement:.1f}%")
    
    # Visualize if 3 components
    if n_components == 3:
        visualize_designs(design1, design2, design3, design4)


def visualize_designs(design1, design2, design3, design4):
    """Visualize the designs in 2D ternary plot"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    designs = [design1, design2, design3, design4]
    titles = ['D-Optimal (no interior)', 'D-Optimal (with interior)', 
              'Simplex Lattice', 'Simplex Centroid']
    colors = ['red', 'blue', 'green', 'purple']
    
    for ax, design, title, color in zip(axes.flat, designs, titles, colors):
        # Plot triangle boundaries
        triangle = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        ax.plot(triangle[:, 0], triangle[:, 1], 'k-', alpha=0.3)
        
        # Plot design points
        X = design.values
        ax.scatter(X[:, 0], X[:, 1], s=100, c=color, alpha=0.7, edgecolor='black')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        
        ax.set_title(title)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        
    plt.tight_layout()
    plt.savefig('output/doptimal_comparison_simplified.png', dpi=150)
    print("\n📊 Visualization saved as 'output/doptimal_comparison_simplified.png'")
    plt.close()


if __name__ == "__main__":
    test_doptimal_comparison()
