"""
Test and analyze clustering in parts mode with fixed components.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

def test_clustering_issue():
    """Demonstrate clustering issue with high fixed parts."""
    
    print("=" * 80)
    print("CLUSTERING ANALYSIS: Parts Mode with Fixed Components")
    print("=" * 80)
    
    # Test Case 1: Moderate fixed parts
    print("\n1. MODERATE FIXED PARTS SCENARIO:")
    print("-" * 50)
    
    component_names = ["Component_A", "Component_B", "Component_C"]
    fixed_parts = {"Component_A": 5.0}  # Moderate fixed amount
    variable_bounds = {
        "Component_B": (0, 50), 
        "Component_C": (0, 50)
    }
    
    design1 = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    df1 = design1.generate_design(n_runs=12, design_type="d-optimal", random_seed=42)
    
    # Test Case 2: HIGH fixed parts (causing clustering)
    print("\n\n2. HIGH FIXED PARTS SCENARIO (CLUSTERING ISSUE):")
    print("-" * 50)
    
    fixed_parts_high = {"Component_A": 45.0}  # Very high fixed amount
    variable_bounds_limited = {
        "Component_B": (0, 20), 
        "Component_C": (0, 20)
    }
    
    design2 = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts_high,
        variable_bounds=variable_bounds_limited
    )
    
    df2 = design2.generate_design(n_runs=12, design_type="d-optimal", random_seed=42)
    
    # Analyze clustering
    print("\n\n3. CLUSTERING ANALYSIS:")
    print("-" * 50)
    
    def analyze_clustering(df, scenario_name):
        print(f"\n{scenario_name}:")
        
        # Extract variable component parts
        var_b_parts = df[f"Component_B_Parts"].values
        var_c_parts = df[f"Component_C_Parts"].values
        
        # Calculate spread and clustering metrics
        b_range = var_b_parts.max() - var_b_parts.min()
        c_range = var_c_parts.max() - var_c_parts.min()
        
        # Calculate minimum distances between points
        points = np.column_stack([var_b_parts, var_c_parts])
        min_distances = []
        
        for i in range(len(points)):
            distances = []
            for j in range(len(points)):
                if i != j:
                    dist = np.linalg.norm(points[i] - points[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        avg_min_distance = np.mean(min_distances) if min_distances else 0
        min_min_distance = min(min_distances) if min_distances else 0
        
        # Count clustered points (points with very small distances)
        clustered_count = sum(1 for d in min_distances if d < 1.0)
        
        print(f"  Component B range: {b_range:.3f} parts")
        print(f"  Component C range: {c_range:.3f} parts")
        print(f"  Average minimum distance: {avg_min_distance:.3f}")
        print(f"  Minimum distance: {min_min_distance:.3f}")
        print(f"  Clustered points (dist < 1.0): {clustered_count}/{len(min_distances)}")
        
        # Batch size analysis
        batch_sizes = df["Batch_Size"].values
        print(f"  Batch size range: {batch_sizes.min():.1f} to {batch_sizes.max():.1f}")
        
        # Proportion analysis for variable components
        var_b_props = df[f"Component_B_Prop"].values
        var_c_props = df[f"Component_C_Prop"].values
        
        print(f"  Component B proportions: {var_b_props.min():.4f} to {var_b_props.max():.4f}")
        print(f"  Component C proportions: {var_c_props.min():.4f} to {var_c_props.max():.4f}")
        
        return {
            'b_range': b_range,
            'c_range': c_range,
            'avg_min_distance': avg_min_distance,
            'min_min_distance': min_min_distance,
            'clustered_count': clustered_count,
            'total_points': len(min_distances)
        }
    
    metrics1 = analyze_clustering(df1, "Moderate Fixed Parts")
    metrics2 = analyze_clustering(df2, "High Fixed Parts")
    
    # Problem identification
    print("\n\n4. PROBLEM IDENTIFICATION:")
    print("-" * 50)
    
    clustering_factor = metrics2['clustered_count'] / metrics2['total_points']
    
    if clustering_factor > 0.5:
        print("🚨 CLUSTERING DETECTED!")
        print(f"   {metrics2['clustered_count']}/{metrics2['total_points']} points are clustered")
        print(f"   Average minimum distance: {metrics2['avg_min_distance']:.3f}")
        print(f"   This indicates poor space-filling in the design")
    else:
        print("✅ No significant clustering detected")
    
    # Create visualization
    create_clustering_visualization(df1, df2)
    
    return df1, df2, metrics1, metrics2

def create_clustering_visualization(df1, df2):
    """Create visualization to show clustering issue."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Moderate fixed parts
    b1_parts = df1["Component_B_Parts"].values
    c1_parts = df1["Component_C_Parts"].values
    
    ax1.scatter(b1_parts, c1_parts, c='blue', s=100, alpha=0.7, edgecolors='black')
    ax1.set_xlabel('Component B (Parts)')
    ax1.set_ylabel('Component C (Parts)')
    ax1.set_title('Moderate Fixed Parts\n(Component A = 5.0 parts)')
    ax1.grid(True, alpha=0.3)
    
    # Add point labels
    for i, (b, c) in enumerate(zip(b1_parts, c1_parts)):
        ax1.annotate(f'{i+1}', (b, c), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: High fixed parts (clustering)
    b2_parts = df2["Component_B_Parts"].values
    c2_parts = df2["Component_C_Parts"].values
    
    ax2.scatter(b2_parts, c2_parts, c='red', s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Component B (Parts)')
    ax2.set_ylabel('Component C (Parts)')
    ax2.set_title('High Fixed Parts - CLUSTERING ISSUE\n(Component A = 45.0 parts)')
    ax2.grid(True, alpha=0.3)
    
    # Add point labels
    for i, (b, c) in enumerate(zip(b2_parts, c2_parts)):
        ax2.annotate(f'{i+1}', (b, c), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('mixture-design-doe/clustering_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as: clustering_analysis.png")
    plt.show()

def propose_solution():
    """Propose solution for clustering issue."""
    
    print("\n\n5. PROPOSED SOLUTION:")
    print("-" * 50)
    
    print("""
🔧 ANTI-CLUSTERING SOLUTION STRATEGIES:

1. ADAPTIVE CANDIDATE GENERATION:
   - Scale candidate generation based on available design space
   - Increase candidate density in constrained regions
   - Use adaptive sampling for small variable ranges

2. CONSTRAINT-AWARE SPACE-FILLING:
   - Modify space-filling algorithms for constrained spaces
   - Use relative coordinates in the available space
   - Implement minimum distance constraints

3. ENHANCED COORDINATE EXCHANGE:
   - Use relative improvement thresholds
   - Add diversity preservation mechanisms
   - Implement anti-clustering penalties

4. INTELLIGENT BOUNDS ADJUSTMENT:
   - Detect when variable space is too constrained
   - Suggest bounds modifications to users
   - Warn about clustering potential

5. ALTERNATIVE DESIGN STRATEGIES:
   - Switch to I-optimal for constrained spaces
   - Use minimax distance designs
   - Implement space-filling designs as fallback
    """)

if __name__ == "__main__":
    df1, df2, metrics1, metrics2 = test_clustering_issue()
    propose_solution()
