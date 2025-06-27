"""
Test script to verify improvement in D-efficiency for mixture designs
Compares D-efficiency values before and after the enhanced candidate point generation
"""

import numpy as np
import matplotlib.pyplot as plt
from mixture_designs import MixtureDesignGenerator
from mixture_design_optimization import OptimizedMixtureDesign
from base_doe import OptimalDOE

def test_d_efficiency_comparison(n_components=3, n_runs=8, model_type="quadratic", random_seed=42):
    """
    Compare D-efficiency between mixture design and regular DOE
    
    Parameters:
    -----------
    n_components : int
        Number of mixture components
    n_runs : int
        Number of runs in the design
    model_type : str
        Model type ("linear", "quadratic", or "cubic")
    random_seed : int
        Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Testing D-efficiency for {n_components} components, {n_runs} runs, {model_type} model")
    print(f"{'='*60}")
    
    # Initialize mixture design
    bounds = [(0.0, 1.0)] * n_components
    
    # Generate D-optimal mixture design
    print("\nGenerating D-optimal mixture design...")
    design, mixture_design = MixtureDesignGenerator.create_d_optimal(
        n_components=n_components,
        n_runs=n_runs,
        model_type=model_type,
        max_iter=1000,
        random_seed=random_seed
    )
    
    # Evaluate design
    print("\nEvaluating mixture design...")
    mixture_metrics = MixtureDesignGenerator.evaluate_design(design, mixture_design, model_type)
    
    # Generate equivalent Regular DOE design for comparison
    print("\nGenerating regular DOE design for comparison...")
    optimal_doe = OptimalDOE(n_components, bounds)
    doe_design = optimal_doe.generate_d_optimal(n_runs, 2 if model_type == "quadratic" else 1, 1000, random_seed)
    
    # Evaluate regular DOE design using the mixture design metrics
    # We need to normalize the design to sum to 1
    normalized_doe_design = doe_design / doe_design.sum(axis=1)[:, np.newaxis]
    doe_mixture_metrics = mixture_design._calculate_d_efficiency(normalized_doe_design, model_type)
    
    # Print metrics
    print(f"\n{'='*60}")
    print(f"{'Metric':<20} {'Mixture Design':<20} {'Regular DOE':<20}")
    print(f"{'-'*20:<20} {'-'*20:<20} {'-'*20:<20}")
    print(f"{'D-efficiency':<20} {mixture_metrics['d_efficiency']:<20.6f} {doe_mixture_metrics:<20.6f}")
    
    # Visualize designs
    print("\nVisualizing designs...")
    visualize_designs(design, normalized_doe_design, n_components)
    
    return mixture_metrics, doe_mixture_metrics

def visualize_designs(mixture_design, regular_doe_design, n_components):
    """
    Visualize both designs for comparison
    
    Parameters:
    -----------
    mixture_design : np.ndarray
        Mixture design matrix
    regular_doe_design : np.ndarray
        Regular DOE design matrix (normalized)
    n_components : int
        Number of components
    """
    if n_components != 3:
        print("Visualization only supported for 3 components")
        return
    
    try:
        import ternary
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot mixture design
        tax1 = ternary.TernaryAxesSubplot(ax=ax1)
        tax1.boundary(linewidth=1.0)
        tax1.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax1.set_title("Optimized Mixture Design")
        tax1.scatter(mixture_design, marker='o', color='blue', s=50, zorder=5)
        
        # Plot regular DOE design
        tax2 = ternary.TernaryAxesSubplot(ax=ax2)
        tax2.boundary(linewidth=1.0)
        tax2.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax2.set_title("Regular DOE Design (Normalized)")
        tax2.scatter(regular_doe_design, marker='o', color='red', s=50, zorder=5)
        
        plt.tight_layout()
        plt.savefig("design_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("Ternary plot requires the 'python-ternary' package.")
        print("Install with: pip install python-ternary")
        
        # Fallback to 2D plots for component pairs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Component pairs
        pairs = [(0, 1), (0, 2), (1, 2)]
        titles = ["Comp1 vs Comp2", "Comp1 vs Comp3", "Comp2 vs Comp3"]
        
        for i, ((c1, c2), title) in enumerate(zip(pairs, titles)):
            ax = axes[i]
            ax.scatter(mixture_design[:, c1], mixture_design[:, c2], color='blue', label='Mixture Design')
            ax.scatter(regular_doe_design[:, c1], regular_doe_design[:, c2], color='red', label='Regular DOE')
            ax.set_xlabel(f"Component {c1+1}")
            ax.set_ylabel(f"Component {c2+1}")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("design_comparison_2d.png", dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Test with 3 components (ternary plot)
    test_d_efficiency_comparison(n_components=3, n_runs=8)
    
    # Test with 4 components
    test_d_efficiency_comparison(n_components=4, n_runs=10)
    
    # Test with 5 components
    test_d_efficiency_comparison(n_components=5, n_runs=15)
