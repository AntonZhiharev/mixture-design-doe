"""
Script to demonstrate the effect of improved candidate generation on D-optimal designs
"""

import numpy as np
import matplotlib.pyplot as plt
from mixture_design_optimization import OptimizedMixtureDesign
from compare_candidate_distribution import original_generate_candidates, improved_generate_candidates

def generate_design_from_candidates(candidates, n_runs=8, n_components=3, model_type="quadratic"):
    """
    Generate a D-optimal design from a set of candidate points
    
    Parameters:
    -----------
    candidates : np.ndarray
        Array of candidate points
    n_runs : int
        Number of runs in the design
    n_components : int
        Number of components
    model_type : str
        Model type ("linear", "quadratic", or "cubic")
        
    Returns:
    --------
    np.ndarray
        D-optimal design matrix
    float
        D-efficiency
    """
    # Create mixture design object
    mixture_design = OptimizedMixtureDesign(n_components=n_components)
    
    # Create initial design from diverse subset of candidates
    from mixture_utils import select_diverse_subset
    design = select_diverse_subset(candidates, n_runs)
    
    # Calculate initial D-efficiency
    d_eff = mixture_design._calculate_d_efficiency(design, model_type)
    print(f"Initial D-efficiency: {d_eff:.6f}")
    
    # Perform coordinate exchange algorithm
    for iteration in range(100):  # Limited iterations for demonstration
        improved = False
        
        for i in range(n_runs):
            best_point = design[i].copy()
            best_eff = d_eff
            
            # Try replacing point i with candidates
            for candidate in candidates[:min(100, len(candidates))]:  # Try subset of candidates
                temp_design = design.copy()
                temp_design[i] = candidate
                
                # Normalize to ensure sum is exactly 1.0
                temp_design = temp_design / temp_design.sum(axis=1)[:, np.newaxis]
                
                temp_eff = mixture_design._calculate_d_efficiency(temp_design, model_type)
                
                if temp_eff > best_eff:
                    best_eff = temp_eff
                    best_point = candidate.copy()
                    improved = True
            
            if improved:
                design[i] = best_point
                # Normalize to ensure sum is exactly 1.0
                design = design / design.sum(axis=1)[:, np.newaxis]
                d_eff = best_eff
        
        if not improved:
            print(f"Converged after {iteration+1} iterations")
            break
    
    final_d_eff = mixture_design._calculate_d_efficiency(design, model_type)
    print(f"Final D-efficiency: {final_d_eff:.6f}")
    
    return design, final_d_eff

def classify_design_points(design, n_components):
    """
    Classify design points as vertices, edges, or interior
    
    Parameters:
    -----------
    design : np.ndarray
        Design matrix
    n_components : int
        Number of components
        
    Returns:
    --------
    dict
        Dictionary with counts of points in each category
    """
    from compare_candidate_distribution import is_vertex, is_edge, is_face
    
    vertices = 0
    edges = 0
    faces = 0
    interior = 0
    
    for point in design:
        if is_vertex(point, n_components):
            vertices += 1
        elif is_edge(point, n_components):
            edges += 1
        elif n_components > 3 and is_face(point, n_components):
            faces += 1
        else:
            interior += 1
    
    return {
        'vertices': vertices,
        'edges': edges,
        'faces': faces,
        'interior': interior
    }

def compare_designs(n_runs=8, n_components=3, model_type="quadratic", n_candidates=500):
    """
    Compare D-optimal designs generated using original and improved candidate generation
    
    Parameters:
    -----------
    n_runs : int
        Number of runs in the design
    n_components : int
        Number of components
    model_type : str
        Model type ("linear", "quadratic", or "cubic")
    n_candidates : int
        Number of candidate points to generate
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print(f"\n{'='*60}")
    print(f"Comparing D-optimal designs for {n_components} components, {n_runs} runs, {model_type} model")
    print(f"{'='*60}")
    
    # Generate candidates using both methods
    print("\nGenerating candidate points...")
    original_candidates = original_generate_candidates(n_candidates, n_components)
    improved_candidates = improved_generate_candidates(n_candidates, n_components)
    
    # Generate designs
    print("\nGenerating D-optimal design using original candidate generation:")
    original_design, original_d_eff = generate_design_from_candidates(
        original_candidates, n_runs, n_components, model_type
    )
    
    print("\nGenerating D-optimal design using improved candidate generation:")
    improved_design, improved_d_eff = generate_design_from_candidates(
        improved_candidates, n_runs, n_components, model_type
    )
    
    # Classify design points
    original_classified = classify_design_points(original_design, n_components)
    improved_classified = classify_design_points(improved_design, n_components)
    
    # Print comparison results
    print(f"\n{'='*60}")
    print("DESIGN COMPARISON RESULTS:")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Original Design':<20} {'Improved Design':<20} {'Difference':<20}")
    print(f"{'-'*20:<20} {'-'*20:<20} {'-'*20:<20} {'-'*20:<20}")
    
    # D-efficiency
    d_eff_diff = improved_d_eff - original_d_eff
    d_eff_pct = d_eff_diff / original_d_eff * 100 if original_d_eff > 0 else float('inf')
    print(f"{'D-efficiency':<20} {original_d_eff:<20.6f} {improved_d_eff:<20.6f} {d_eff_diff:+<20.6f} ({d_eff_pct:+.2f}%)")
    
    # Point distribution
    categories = ['vertices', 'edges', 'faces', 'interior']
    for category in categories:
        orig_count = original_classified[category]
        impr_count = improved_classified[category]
        diff = impr_count - orig_count
        print(f"{category.capitalize():<20} {orig_count:<20} {impr_count:<20} {diff:+<20}")
    
    # Visualize the designs
    if n_components == 3:
        visualize_designs(original_design, improved_design, original_d_eff, improved_d_eff)

def visualize_designs(original_design, improved_design, original_d_eff, improved_d_eff):
    """
    Visualize both designs for comparison
    
    Parameters:
    -----------
    original_design : np.ndarray
        Original design matrix
    improved_design : np.ndarray
        Improved design matrix
    original_d_eff : float
        Original design D-efficiency
    improved_d_eff : float
        Improved design D-efficiency
    """
    try:
        import ternary
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original design
        tax1 = ternary.TernaryAxesSubplot(ax=ax1)
        tax1.boundary(linewidth=1.0)
        tax1.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax1.set_title(f"Original Design (D-eff: {original_d_eff:.4f})")
        tax1.scatter(original_design, marker='o', color='blue', s=50, zorder=5)
        
        # Plot improved design
        tax2 = ternary.TernaryAxesSubplot(ax=ax2)
        tax2.boundary(linewidth=1.0)
        tax2.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax2.set_title(f"Improved Design (D-eff: {improved_d_eff:.4f})")
        tax2.scatter(improved_design, marker='o', color='red', s=50, zorder=5)
        
        plt.tight_layout()
        plt.savefig("design_comparison_result.png", dpi=300, bbox_inches='tight')
        print("\nDesign comparison plot saved as 'design_comparison_result.png'")
        plt.close()
        
    except ImportError:
        print("\nTernary plotting package not available. Install with: pip install python-ternary")
        
        # Fallback to 2D plots for component pairs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Component pairs
        pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ["Comp1 vs Comp2", "Comp1 vs Comp3", "Comp2 vs Comp3"]
        
        for i, ((c1, c2), title) in enumerate(zip(pairs, titles)):
            ax = axes[i]
            ax.scatter(original_design[:, c1], original_design[:, c2], color='blue', label='Original Design')
            ax.scatter(improved_design[:, c1], improved_design[:, c2], color='red', label='Improved Design')
            ax.set_xlabel(f"Component {c1+1}")
            ax.set_ylabel(f"Component {c2+1}")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("design_comparison_2d.png", dpi=300, bbox_inches='tight')
        print("\nDesign comparison plot saved as 'design_comparison_2d.png'")
        plt.close()

if __name__ == "__main__":
    # Run the comparison
    compare_designs(n_runs=8, n_components=3, model_type="quadratic", n_candidates=500)
