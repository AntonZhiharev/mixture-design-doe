"""
Script to compare D-efficiency between original and improved mixture design approaches
"""

import numpy as np
from mixture_design_optimization import OptimizedMixtureDesign

def original_generate_candidates(n_points, n_components):
    """
    Original candidate point generation method
    """
    candidates = []
    
    # 1. Vertices (pure components)
    for i in range(n_components):
        vertex = np.zeros(n_components)
        vertex[i] = 1.0
        candidates.append(vertex)
    
    # 2. Centroid
    centroid = np.ones(n_components) / n_components
    candidates.append(centroid)
    
    # 3. Binary mixtures (midpoints of edges)
    for i in range(n_components):
        for j in range(i+1, n_components):
            midpoint = np.zeros(n_components)
            midpoint[i] = 0.5
            midpoint[j] = 0.5
            candidates.append(midpoint)
    
    # 4. Random points
    n_random = max(0, n_points - len(candidates))
    for _ in range(n_random):
        point = np.random.random(n_components)
        point = point / np.sum(point)  # Normalize
        candidates.append(point)
    
    return np.array(candidates)

def improved_generate_candidates(n_points, n_components):
    """
    Improved candidate point generation method
    """
    candidates = []
    
    # 1. Vertices (pure components)
    for i in range(n_components):
        vertex = np.zeros(n_components)
        vertex[i] = 1.0
        candidates.append(vertex)
    
    # 2. Centroid
    centroid = np.ones(n_components) / n_components
    candidates.append(centroid)
    
    # 3. Binary mixtures (midpoints of edges)
    for i in range(n_components):
        for j in range(i+1, n_components):
            midpoint = np.zeros(n_components)
            midpoint[i] = 0.5
            midpoint[j] = 0.5
            candidates.append(midpoint)
    
    # 4. Generate interior points using space-filling design
    n_interior = min(max(50, n_points // 2), n_points - len(candidates))
    
    # Method 1: Dirichlet distribution (concentrates more points in the interior)
    alpha = np.ones(n_components) * 2  # Alpha > 1 concentrates points toward centroid
    interior_points1 = np.random.dirichlet(alpha, size=n_interior // 2)
    
    # Method 2: Random points with normalization (more uniform coverage)
    interior_points2 = []
    for _ in range(n_interior - len(interior_points1)):
        point = np.random.random(n_components)
        point = point / np.sum(point)  # Normalize
        interior_points2.append(point)
        
    # Add interior points to candidates
    candidates.extend(interior_points1)
    candidates.extend(interior_points2)
    
    # 5. Additional random points if needed
    n_random = max(0, n_points - len(candidates))
    for _ in range(n_random):
        point = np.random.random(n_components)
        point = point / np.sum(point)  # Normalize
        candidates.append(point)
    
    return np.array(candidates)

def compare_d_optimal_designs(n_components=3, n_runs=8, model_type="quadratic"):
    """
    Compare D-optimal designs generated using original and improved candidate generation
    
    Parameters:
    -----------
    n_components : int
        Number of mixture components
    n_runs : int
        Number of runs in the design
    model_type : str
        Model type ("linear", "quadratic", or "cubic")
    """
    print(f"\n{'='*60}")
    print(f"Testing D-efficiency for {n_components} components, {n_runs} runs, {model_type} model")
    print(f"{'='*60}")
    
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    # Create a mixture design object
    mixture_design = OptimizedMixtureDesign(n_components=n_components)
    
    # Generate designs using both methods
    print("\nGenerating D-optimal design using original method...")
    # Generate candidate points using original method
    original_candidates = original_generate_candidates(1000, n_components)
    
    # Create initial design
    from mixture_utils import select_diverse_subset
    original_initial_design = select_diverse_subset(original_candidates, n_runs)
    
    # Optimize design
    original_design = original_initial_design.copy()
    best_d_eff_original = mixture_design._calculate_d_efficiency(original_design, model_type)
    
    # Coordinate exchange algorithm (simplified)
    for iteration in range(100):  # Reduced iterations for quick testing
        improved = False
        
        for i in range(n_runs):
            best_point = original_design[i].copy()
            best_eff = best_d_eff_original
            
            # Try replacing point i with candidates
            for candidate in original_candidates[:50]:  # Try subset of candidates
                temp_design = original_design.copy()
                temp_design[i] = candidate
                
                temp_eff = mixture_design._calculate_d_efficiency(temp_design, model_type)
                
                if temp_eff > best_eff:
                    best_eff = temp_eff
                    best_point = candidate.copy()
                    improved = True
            
            if improved:
                original_design[i] = best_point
                best_d_eff_original = best_eff
        
        if not improved:
            break
    
    print(f"Original method D-efficiency: {best_d_eff_original:.6f}")
    
    # Reset random seed
    np.random.seed(42)
    
    print("\nGenerating D-optimal design using improved method...")
    # Generate candidate points using improved method
    improved_candidates = improved_generate_candidates(1000, n_components)
    
    # Create initial design
    improved_initial_design = select_diverse_subset(improved_candidates, n_runs)
    
    # Optimize design
    improved_design = improved_initial_design.copy()
    best_d_eff_improved = mixture_design._calculate_d_efficiency(improved_design, model_type)
    
    # Coordinate exchange algorithm (simplified)
    for iteration in range(100):  # Reduced iterations for quick testing
        improved_flag = False
        
        for i in range(n_runs):
            best_point = improved_design[i].copy()
            best_eff = best_d_eff_improved
            
            # Try replacing point i with candidates
            for candidate in improved_candidates[:50]:  # Try subset of candidates
                temp_design = improved_design.copy()
                temp_design[i] = candidate
                
                temp_eff = mixture_design._calculate_d_efficiency(temp_design, model_type)
                
                if temp_eff > best_eff:
                    best_eff = temp_eff
                    best_point = candidate.copy()
                    improved_flag = True
            
            if improved_flag:
                improved_design[i] = best_point
                best_d_eff_improved = best_eff
        
        if not improved_flag:
            break
    
    print(f"Improved method D-efficiency: {best_d_eff_improved:.6f}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"RESULTS COMPARISON:")
    print(f"Original method D-efficiency: {best_d_eff_original:.6f}")
    print(f"Improved method D-efficiency: {best_d_eff_improved:.6f}")
    print(f"Improvement: {(best_d_eff_improved - best_d_eff_original) / best_d_eff_original * 100:.2f}%")
    
    # Visualize the designs
    try:
        if n_components == 3:
            import matplotlib.pyplot as plt
            import ternary
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot original design
            tax1 = ternary.TernaryAxesSubplot(ax=ax1)
            tax1.boundary(linewidth=1.0)
            tax1.gridlines(color="gray", multiple=0.1, linewidth=0.5)
            tax1.set_title(f"Original Method (D-eff: {best_d_eff_original:.4f})")
            tax1.scatter(original_design, marker='o', color='blue', s=50, zorder=5)
            
            # Plot improved design
            tax2 = ternary.TernaryAxesSubplot(ax=ax2)
            tax2.boundary(linewidth=1.0)
            tax2.gridlines(color="gray", multiple=0.1, linewidth=0.5)
            tax2.set_title(f"Improved Method (D-eff: {best_d_eff_improved:.4f})")
            tax2.scatter(improved_design, marker='o', color='red', s=50, zorder=5)
            
            plt.tight_layout()
            plt.savefig("d_efficiency_comparison.png", dpi=300, bbox_inches='tight')
            print("\nDesign comparison plot saved as 'd_efficiency_comparison.png'")
            plt.close()
    except ImportError:
        print("\nTernary plotting package not available. Install with: pip install python-ternary")

if __name__ == "__main__":
    # Run the comparison
    compare_d_optimal_designs(n_components=3, n_runs=8)
