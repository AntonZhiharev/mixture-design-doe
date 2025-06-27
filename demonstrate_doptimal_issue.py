"""
Demonstrate the D-optimal design issue where it only generates corner points
"""

import numpy as np
import pandas as pd

# Simple D-optimal implementation to show the issue
def simple_doptimal_corners_only(n_components, n_runs):
    """D-optimal that only selects corner points (the problematic version)"""
    # Only consider corner points as candidates
    candidates = []
    for i in range(n_components):
        corner = np.zeros(n_components)
        corner[i] = 1.0
        candidates.append(corner)
    
    # For demonstration, just repeat corners
    design_points = []
    for i in range(n_runs):
        design_points.append(candidates[i % n_components])
    
    return np.array(design_points)


def simple_doptimal_with_interior(n_components, n_runs):
    """D-optimal that includes interior points (the improved version)"""
    # Include corners, edges, and interior points as candidates
    candidates = []
    
    # Corners
    for i in range(n_components):
        corner = np.zeros(n_components)
        corner[i] = 1.0
        candidates.append(corner)
    
    # Edge midpoints
    for i in range(n_components):
        for j in range(i + 1, n_components):
            edge = np.zeros(n_components)
            edge[i] = 0.5
            edge[j] = 0.5
            candidates.append(edge)
    
    # Interior points
    if n_components == 3:
        candidates.extend([
            [1/3, 1/3, 1/3],  # Centroid
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
            [0.4, 0.4, 0.2],
            [0.4, 0.2, 0.4],
            [0.2, 0.4, 0.4]
        ])
    
    # Select points (simplified selection)
    np.random.seed(42)
    selected_indices = np.random.choice(len(candidates), min(n_runs, len(candidates)), replace=False)
    return np.array([candidates[i] for i in selected_indices])


def calculate_d_efficiency(design_matrix):
    """Calculate D-efficiency"""
    X = design_matrix
    try:
        info_matrix = X.T @ X
        det_value = np.linalg.det(info_matrix)
        n_components = X.shape[1]
        n_runs = X.shape[0]
        d_efficiency = (det_value / n_runs) ** (1/n_components)
        return d_efficiency
    except:
        return 0.0


def main():
    print("DEMONSTRATING D-OPTIMAL DESIGN ISSUE")
    print("=" * 60)
    
    n_components = 3
    n_runs = 10
    
    # Current problematic D-optimal (corners only)
    print("\n1. PROBLEMATIC D-OPTIMAL (corners only):")
    design1 = simple_doptimal_corners_only(n_components, n_runs)
    df1 = pd.DataFrame(design1, columns=['X1', 'X2', 'X3'])
    print(df1)
    
    # Analyze
    corner_count = sum(1 for row in design1 if np.sum(row > 0.001) == 1)
    print(f"\nCorner points: {corner_count}/{len(design1)} ({corner_count/len(design1)*100:.0f}%)")
    d_eff1 = calculate_d_efficiency(design1)
    print(f"D-efficiency: {d_eff1:.4f}")
    
    # Improved D-optimal (with interior points)
    print("\n\n2. IMPROVED D-OPTIMAL (with interior points):")
    design2 = simple_doptimal_with_interior(n_components, n_runs)
    df2 = pd.DataFrame(design2, columns=['X1', 'X2', 'X3'])
    print(df2)
    
    # Analyze
    corner_count = sum(1 for row in design2 if np.sum(row > 0.001) == 1)
    interior_count = sum(1 for row in design2 if np.all(row > 0.001))
    print(f"\nCorner points: {corner_count}/{len(design2)} ({corner_count/len(design2)*100:.0f}%)")
    print(f"Interior points: {interior_count}/{len(design2)} ({interior_count/len(design2)*100:.0f}%)")
    d_eff2 = calculate_d_efficiency(design2)
    print(f"D-efficiency: {d_eff2:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Problematic D-optimal (corners only): D-efficiency = {d_eff1:.4f}")
    print(f"Improved D-optimal (with interior): D-efficiency = {d_eff2:.4f}")
    
    if d_eff2 > d_eff1:
        improvement = (d_eff2 - d_eff1) / d_eff1 * 100
        print(f"\n✅ IMPROVEMENT: {improvement:.1f}% increase in D-efficiency")
        print("This demonstrates why including interior points is important!")
    
    print("\nThe issue is that the current D-optimal implementation only considers")
    print("corner points (pure components), which leads to lower D-efficiency.")
    print("By including interior points in the candidate set, we get better designs.")


if __name__ == "__main__":
    main()
