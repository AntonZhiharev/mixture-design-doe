"""
Diagnostic Test for Fixed Parts Mixture Design Distribution Issues

This script identifies and demonstrates the poor distribution problem
when using fixed components on points in mixture designs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import seaborn as sns
from src.core.fixed_parts_mixture_designs import FixedPartsMixtureDesign
import warnings
warnings.filterwarnings('ignore')

def test_distribution_quality():
    """Test the distribution quality of current fixed parts mixture designs."""
    
    print("=== Testing Fixed Parts Mixture Design Distribution ===")
    
    # Example setup: 4-component polymer formulation
    component_names = ["Base_Polymer", "Catalyst", "Solvent", "Additive"]
    
    # Fixed components (constant parts)
    fixed_parts = {
        "Base_Polymer": 50.0,  # Always 50 parts
        "Catalyst": 2.5        # Always 2.5 parts
    }
    
    # Variable components bounds (in parts) 
    variable_bounds = {
        "Solvent": (0.0, 40.0),    # 0-40 parts
        "Additive": (0.0, 15.0)    # 0-15 parts
    }
    
    print(f"Fixed parts: {fixed_parts}")
    print(f"Variable bounds: {variable_bounds}")
    print(f"Total fixed consumption: {sum(fixed_parts.values())} parts")
    
    # Create design generator
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate design with different sizes to see the distribution
    test_sizes = [10, 15, 20, 25]
    
    for n_runs in test_sizes:
        print(f"\n--- Testing with {n_runs} runs ---")
        
        # Generate design
        design_df = designer.generate_design(
            n_runs=n_runs,
            design_type="d-optimal", 
            model_type="quadratic",
            random_seed=42
        )
        
        # Analyze distribution
        analyze_distribution(design_df, component_names, fixed_parts, n_runs)

def analyze_distribution(design_df, component_names, fixed_parts, n_runs):
    """Analyze the distribution quality of the design."""
    
    print(f"\n📊 Distribution Analysis for {n_runs} runs:")
    
    # 1. Check variable component coverage
    variable_names = [name for name in component_names if name not in fixed_parts]
    
    for var_name in variable_names:
        parts_col = f"{var_name}_Parts"
        prop_col = f"{var_name}_Prop"
        
        if parts_col in design_df.columns:
            parts_values = design_df[parts_col].values
            prop_values = design_df[prop_col].values
            
            parts_range = parts_values.max() - parts_values.min()
            parts_std = np.std(parts_values)
            
            print(f"  {var_name}:")
            print(f"    Parts range: {parts_values.min():.2f} to {parts_values.max():.2f} (spread: {parts_range:.2f})")
            print(f"    Parts std: {parts_std:.2f}")
            print(f"    Proportion range: {prop_values.min():.3f} to {prop_values.max():.3f}")
    
    # 2. Check batch size distribution
    if 'Batch_Size' in design_df.columns:
        batch_sizes = design_df['Batch_Size'].values
        batch_range = batch_sizes.max() - batch_sizes.min()
        batch_std = np.std(batch_sizes)
        
        print(f"  Batch sizes:")
        print(f"    Range: {batch_sizes.min():.1f} to {batch_sizes.max():.1f} (spread: {batch_range:.1f})")
        print(f"    Std: {batch_std:.2f}")
    
    # 3. Calculate space-filling metrics
    calculate_space_filling_metrics(design_df, variable_names)

def calculate_space_filling_metrics(design_df, variable_names):
    """Calculate space-filling quality metrics."""
    
    # Extract variable component parts for space-filling analysis
    parts_cols = [f"{name}_Parts" for name in variable_names]
    
    if all(col in design_df.columns for col in parts_cols):
        X = design_df[parts_cols].values
        n_points, n_dims = X.shape
        
        # 1. Minimum distance between points
        min_distances = []
        for i in range(n_points):
            distances = []
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(X[i] - X[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        if min_distances:
            min_min_dist = min(min_distances)
            avg_min_dist = np.mean(min_distances)
            
            print(f"  Space-filling metrics:")
            print(f"    Minimum distance between points: {min_min_dist:.3f}")
            print(f"    Average minimum distance: {avg_min_dist:.3f}")
            
            # Check for clustered points (very small distances)
            clustered_threshold = 0.5  # Adjust based on scale
            clustered_pairs = sum(1 for d in min_distances if d < clustered_threshold)
            print(f"    Points with distance < {clustered_threshold}: {clustered_pairs}/{len(min_distances)}")
        
        # 2. Coverage of design space corners
        analyze_corner_coverage(X, variable_names)

def analyze_corner_coverage(X, variable_names):
    """Analyze how well the design covers the corners of the variable space."""
    
    print(f"  Corner coverage analysis:")
    
    # Normalize to [0,1] for analysis
    X_norm = np.zeros_like(X)
    bounds_info = []
    
    for i, var_name in enumerate(variable_names):
        col_min, col_max = X[:, i].min(), X[:, i].max()
        if col_max > col_min:
            X_norm[:, i] = (X[:, i] - col_min) / (col_max - col_min)
        bounds_info.append((col_min, col_max))
        print(f"    {var_name}: normalized from [{col_min:.2f}, {col_max:.2f}]")
    
    # Check proximity to corners
    n_dims = X_norm.shape[1]
    corners = []
    
    # Generate all corner combinations
    for i in range(2**n_dims):
        corner = []
        for j in range(n_dims):
            corner.append((i >> j) & 1)  # 0 or 1
        corners.append(corner)
    
    corner_distances = []
    for corner in corners:
        # Find closest design point to this corner
        min_dist_to_corner = float('inf')
        for point in X_norm:
            dist = np.linalg.norm(point - np.array(corner))
            min_dist_to_corner = min(min_dist_to_corner, dist)
        corner_distances.append(min_dist_to_corner)
    
    avg_corner_dist = np.mean(corner_distances)
    max_corner_dist = max(corner_distances)
    
    print(f"    Average distance to corners: {avg_corner_dist:.3f}")
    print(f"    Maximum distance to any corner: {max_corner_dist:.3f}")
    
    # Poor coverage indicator
    if max_corner_dist > 0.5:  # In normalized space
        print(f"    ⚠️  POOR CORNER COVERAGE - Some corners are far from design points!")
    
    if avg_corner_dist > 0.3:
        print(f"    ⚠️  POOR OVERALL COVERAGE - Design points don't spread well!")

def create_distribution_plots():
    """Create visualization plots showing the distribution issues."""
    
    print("\n=== Creating Distribution Plots ===")
    
    # Setup
    component_names = ["Base_Polymer", "Catalyst", "Solvent", "Additive"]
    fixed_parts = {"Base_Polymer": 50.0, "Catalyst": 2.5}
    variable_bounds = {"Solvent": (0.0, 40.0), "Additive": (0.0, 15.0)}
    
    designer = FixedPartsMixtureDesign(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate designs of different sizes
    sizes = [10, 15, 20]
    fig, axes = plt.subplots(1, len(sizes), figsize=(15, 5))
    
    for idx, n_runs in enumerate(sizes):
        design_df = designer.generate_design(
            n_runs=n_runs,
            design_type="d-optimal",
            model_type="quadratic", 
            random_seed=42
        )
        
        # Plot variable components in design space
        solvent_parts = design_df['Solvent_Parts'].values
        additive_parts = design_df['Additive_Parts'].values
        
        ax = axes[idx]
        ax.scatter(solvent_parts, additive_parts, alpha=0.7, s=50)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 15)
        ax.set_xlabel('Solvent (Parts)')
        ax.set_ylabel('Additive (Parts)')
        ax.set_title(f'{n_runs} Runs - Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add corner indicators
        corners_x = [0, 0, 40, 40]
        corners_y = [0, 15, 0, 15] 
        ax.scatter(corners_x, corners_y, marker='x', s=100, c='red', alpha=0.7)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    plt.savefig('output/fixed_parts_distribution_problem.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Distribution plots saved to: output/fixed_parts_distribution_problem.png")

def demonstrate_candidate_generation_issue():
    """Demonstrate the root cause: poor candidate generation."""
    
    print("\n=== Investigating Candidate Generation ===")
    
    component_names = ["Base_Polymer", "Catalyst", "Solvent", "Additive"]
    fixed_parts = {"Base_Polymer": 50.0, "Catalyst": 2.5}
    variable_bounds = {"Solvent": (0.0, 40.0), "Additive": (0.0, 15.0)}
    
    # Create the underlying designer to access internal methods
    from src.core.true_fixed_components_mixture import TrueFixedComponentsMixture
    
    core_designer = TrueFixedComponentsMixture(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate candidate set and analyze distribution
    print("Generating 1000 candidate points...")
    parts_candidates, prop_candidates, batch_candidates = core_designer.generate_candidate_set(1000)
    
    # Analyze variable component distribution in candidates
    solvent_idx = component_names.index("Solvent")
    additive_idx = component_names.index("Additive")
    
    solvent_parts = parts_candidates[:, solvent_idx]
    additive_parts = parts_candidates[:, additive_idx]
    
    print(f"\nCandidate Set Analysis:")
    print(f"  Solvent parts: {solvent_parts.min():.2f} to {solvent_parts.max():.2f}")
    print(f"  Additive parts: {additive_parts.min():.2f} to {additive_parts.max():.2f}")
    
    # Check corner coverage in candidates
    corners = [(0, 0), (0, 15), (40, 0), (40, 15)]
    corner_coverage = []
    
    for corner_x, corner_y in corners:
        distances = np.sqrt((solvent_parts - corner_x)**2 + (additive_parts - corner_y)**2)
        min_dist = distances.min()
        corner_coverage.append(min_dist)
        print(f"  Corner ({corner_x}, {corner_y}): closest candidate at distance {min_dist:.2f}")
    
    avg_corner_coverage = np.mean(corner_coverage)
    print(f"  Average corner coverage: {avg_corner_coverage:.2f}")
    
    if avg_corner_coverage > 2.0:
        print("  ⚠️  PROBLEM: Candidates poorly cover design space corners!")
        print("  This explains why final designs have poor distribution.")
    
    return solvent_parts, additive_parts

if __name__ == "__main__":
    # Run the diagnostic tests
    test_distribution_quality()
    create_distribution_plots()
    demonstrate_candidate_generation_issue()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("IDENTIFIED ISSUES:")
    print("1. Random candidate generation doesn't ensure space-filling")
    print("2. Limited structured points for corner coverage")
    print("3. No space-filling optimization in coordinate exchange")
    print("4. Independent variable sampling ignores correlation structure")
    print("\nRECOMMENDED FIXES:")
    print("1. Implement Latin Hypercube Sampling for candidates")
    print("2. Add more corner and edge points")
    print("3. Include space-filling criteria in optimization")
    print("4. Use stratified sampling for better coverage")
