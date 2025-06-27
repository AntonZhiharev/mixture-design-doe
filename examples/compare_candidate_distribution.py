"""
Script to compare the distribution of candidate points between the original and improved methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

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
    Improved candidate point generation method with more interior points
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

def is_vertex(point, n_components, tolerance=1e-6):
    """Check if a point is a vertex (one component = 1, others = 0)"""
    return any(abs(p - 1.0) < tolerance for p in point)

def is_edge(point, n_components, tolerance=1e-6):
    """Check if a point is on an edge (two components > 0, others = 0)"""
    nonzero = sum(p > tolerance for p in point)
    return nonzero == 2

def is_face(point, n_components, tolerance=1e-6):
    """Check if a point is on a face (three components > 0, others = 0)"""
    nonzero = sum(p > tolerance for p in point)
    return nonzero == 3

def distance_to_centroid(point, n_components):
    """Calculate distance from point to the centroid"""
    centroid = np.ones(n_components) / n_components
    return np.sqrt(np.sum((point - centroid) ** 2))

def classify_interior_by_distance(points, n_components):
    """Classify interior points by distance to centroid"""
    if len(points) == 0:
        return {
            'near_centroid': np.empty((0, n_components)),
            'mid_interior': np.empty((0, n_components)),
            'near_boundary': np.empty((0, n_components))
        }
    
    # Calculate distances to centroid
    distances = [distance_to_centroid(p, n_components) for p in points]
    
    # Calculate max possible distance in the simplex
    # (Distance from centroid to any vertex)
    centroid = np.ones(n_components) / n_components
    vertex = np.zeros(n_components)
    vertex[0] = 1.0
    max_distance = np.sqrt(np.sum((vertex - centroid) ** 2))
    
    # Classify based on distance
    near_centroid = []
    mid_interior = []
    near_boundary = []
    
    for i, point in enumerate(points):
        # Normalize distance to be between 0 and 1
        relative_dist = distances[i] / max_distance
        
        if relative_dist < 0.33:
            near_centroid.append(point)
        elif relative_dist < 0.66:
            mid_interior.append(point)
        else:
            near_boundary.append(point)
    
    return {
        'near_centroid': np.array(near_centroid) if near_centroid else np.empty((0, n_components)),
        'mid_interior': np.array(mid_interior) if mid_interior else np.empty((0, n_components)),
        'near_boundary': np.array(near_boundary) if near_boundary else np.empty((0, n_components))
    }

def classify_points(candidates, n_components):
    """Classify points as vertices, edges, faces, or interior"""
    vertices = []
    edges = []
    faces = []
    interior = []
    
    for point in candidates:
        if is_vertex(point, n_components):
            vertices.append(point)
        elif is_edge(point, n_components):
            edges.append(point)
        elif n_components > 3 and is_face(point, n_components):
            faces.append(point)
        else:
            interior.append(point)
    
    return {
        'vertices': np.array(vertices) if vertices else np.empty((0, n_components)),
        'edges': np.array(edges) if edges else np.empty((0, n_components)),
        'faces': np.array(faces) if faces else np.empty((0, n_components)),
        'interior': np.array(interior) if interior else np.empty((0, n_components))
    }

def compare_candidate_distributions(n_points=500, n_components=3):
    """
    Compare the distribution of candidate points between original and improved methods
    """
    # Set a fixed random seed for reproducibility
    np.random.seed(42)
    
    # Generate candidates using both methods
    original_candidates = original_generate_candidates(n_points, n_components)
    improved_candidates = improved_generate_candidates(n_points, n_components)
    
    # Classify points
    original_classified = classify_points(original_candidates, n_components)
    improved_classified = classify_points(improved_candidates, n_components)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Candidate Point Distribution Comparison ({n_points} points, {n_components} components)")
    print(f"{'='*60}")
    
    # Count points in each category
    original_counts = {k: len(v) for k, v in original_classified.items()}
    improved_counts = {k: len(v) for k, v in improved_classified.items()}
    
    # Calculate percentages
    original_percentages = {k: count/n_points*100 for k, count in original_counts.items()}
    improved_percentages = {k: count/n_points*100 for k, count in improved_counts.items()}
    
    # Print comparison table
    print(f"\n{'Category':<15} {'Original Count':<15} {'Original %':<15} {'Improved Count':<15} {'Improved %':<15} {'Difference':<15}")
    print(f"{'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}")
    
    categories = ['vertices', 'edges', 'faces', 'interior']
    for category in categories:
        orig_count = original_counts[category]
        orig_pct = original_percentages[category]
        impr_count = improved_counts[category]
        impr_pct = improved_percentages[category]
        diff = impr_pct - orig_pct
        
        print(f"{category:<15} {orig_count:<15} {orig_pct:<15.2f} {impr_count:<15} {impr_pct:<15.2f} {diff:+<15.2f}")
    
    # Further analyze interior points
    print("\nAnalyzing Interior Point Distribution...")
    original_interior_classified = classify_interior_by_distance(original_classified['interior'], n_components)
    improved_interior_classified = classify_interior_by_distance(improved_classified['interior'], n_components)
    
    # Count interior points by category
    original_interior_counts = {k: len(v) for k, v in original_interior_classified.items()}
    improved_interior_counts = {k: len(v) for k, v in improved_interior_classified.items()}
    
    # Calculate percentages relative to all interior points
    original_interior_total = sum(original_interior_counts.values())
    improved_interior_total = sum(improved_interior_counts.values())
    
    if original_interior_total > 0:
        original_interior_percentages = {k: count/original_interior_total*100 for k, count in original_interior_counts.items()}
    else:
        original_interior_percentages = {k: 0 for k in original_interior_counts}
        
    if improved_interior_total > 0:
        improved_interior_percentages = {k: count/improved_interior_total*100 for k, count in improved_interior_counts.items()}
    else:
        improved_interior_percentages = {k: 0 for k in improved_interior_counts}
    
    # Print interior point distribution
    print(f"\n{'Interior Region':<15} {'Original Count':<15} {'Original %':<15} {'Improved Count':<15} {'Improved %':<15} {'Difference':<15}")
    print(f"{'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15} {'-'*15:<15}")
    
    interior_categories = ['near_centroid', 'mid_interior', 'near_boundary']
    for category in interior_categories:
        orig_count = original_interior_counts[category]
        orig_pct = original_interior_percentages[category]
        impr_count = improved_interior_counts[category]
        impr_pct = improved_interior_percentages[category]
        diff = impr_pct - orig_pct
        
        print(f"{category:<15} {orig_count:<15} {orig_pct:<15.2f} {impr_count:<15} {impr_pct:<15.2f} {diff:+<15.2f}")
    
    # Plot the distributions if the components count allows visualization
    if n_components == 3:
        plot_ternary_comparison(original_classified, improved_classified)
    elif n_components == 2:
        plot_binary_comparison(original_classified, improved_classified)

def plot_ternary_comparison(original_classified, improved_classified):
    """
    Plot ternary visualization comparing original and improved candidate distributions
    """
    try:
        import ternary
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original distribution
        tax1 = ternary.TernaryAxesSubplot(ax=ax1)
        tax1.boundary(linewidth=1.0)
        tax1.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax1.set_title("Original Candidate Distribution")
        
        # Plot improved distribution
        tax2 = ternary.TernaryAxesSubplot(ax=ax2)
        tax2.boundary(linewidth=1.0)
        tax2.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax2.set_title("Improved Candidate Distribution")
        
        # Plot points by category with consistent colors
        # Vertices
        if len(original_classified['vertices']) > 0:
            tax1.scatter(original_classified['vertices'], marker='o', color='red', 
                         s=60, zorder=5, label='Vertices')
        if len(improved_classified['vertices']) > 0:
            tax2.scatter(improved_classified['vertices'], marker='o', color='red', 
                         s=60, zorder=5, label='Vertices')
            
        # Edges
        if len(original_classified['edges']) > 0:
            tax1.scatter(original_classified['edges'], marker='s', color='blue', 
                         s=40, zorder=4, label='Edges')
        if len(improved_classified['edges']) > 0:
            tax2.scatter(improved_classified['edges'], marker='s', color='blue', 
                         s=40, zorder=4, label='Edges')
            
        # Interior points - categorize by distance to centroid
        original_interior_classified = classify_interior_by_distance(original_classified['interior'], 3)
        improved_interior_classified = classify_interior_by_distance(improved_classified['interior'], 3)
        
        # Near centroid
        if len(original_interior_classified['near_centroid']) > 0:
            tax1.scatter(original_interior_classified['near_centroid'], marker='^', color='darkgreen', 
                         s=40, zorder=3, label='Near Centroid')
        if len(improved_interior_classified['near_centroid']) > 0:
            tax2.scatter(improved_interior_classified['near_centroid'], marker='^', color='darkgreen', 
                         s=40, zorder=3, label='Near Centroid')
        
        # Mid interior
        if len(original_interior_classified['mid_interior']) > 0:
            tax1.scatter(original_interior_classified['mid_interior'], marker='d', color='green', 
                         s=30, zorder=2, label='Mid Interior')
        if len(improved_interior_classified['mid_interior']) > 0:
            tax2.scatter(improved_interior_classified['mid_interior'], marker='d', color='green', 
                         s=30, zorder=2, label='Mid Interior')
            
        # Near boundary
        if len(original_interior_classified['near_boundary']) > 0:
            tax1.scatter(original_interior_classified['near_boundary'], marker='.', color='lightgreen', 
                         s=20, zorder=1, label='Near Boundary')
        if len(improved_interior_classified['near_boundary']) > 0:
            tax2.scatter(improved_interior_classified['near_boundary'], marker='.', color='lightgreen', 
                         s=20, zorder=1, label='Near Boundary')
        
        # Add legends
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig("candidate_distribution_comparison.png", dpi=300, bbox_inches='tight')
        print("\nTernary plot saved as 'candidate_distribution_comparison.png'")
        plt.close()
        
        # Also create a density heatmap for a more visual comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original distribution heatmap
        tax1 = ternary.TernaryAxesSubplot(ax=ax1)
        tax1.boundary(linewidth=1.0)
        tax1.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax1.set_title("Original Candidate Density")
        
        # Plot improved distribution heatmap
        tax2 = ternary.TernaryAxesSubplot(ax=ax2)
        tax2.boundary(linewidth=1.0)
        tax2.gridlines(color="gray", multiple=0.1, linewidth=0.5)
        tax2.set_title("Improved Candidate Density")
        
        # Generate heatmaps
        if len(original_classified['interior']) > 0:
            try:
                tax1.heatmap(original_classified['interior'], scale=5, cmap=plt.cm.viridis, 
                           colorbar=True, cbarlabel="Density")
            except:
                print("Could not generate heatmap for original distribution")
                
        if len(improved_classified['interior']) > 0:
            try:
                tax2.heatmap(improved_classified['interior'], scale=5, cmap=plt.cm.viridis, 
                           colorbar=True, cbarlabel="Density")
            except:
                print("Could not generate heatmap for improved distribution")
        
        plt.tight_layout()
        plt.savefig("candidate_density_comparison.png", dpi=300, bbox_inches='tight')
        print("\nDensity heatmap saved as 'candidate_density_comparison.png'")
        plt.close()
        
    except ImportError:
        print("\nTernary plotting package not available. Install with: pip install python-ternary")
        # Fallback to scatter plot
        plot_scatter_comparison(original_classified, improved_classified)

def plot_binary_comparison(original_classified, improved_classified):
    """
    Plot binary comparison for 2-component mixtures
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot original distribution
    ax1.set_title("Original Candidate Distribution")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.grid(True, alpha=0.3)
    
    # Plot improved distribution
    ax2.set_title("Improved Candidate Distribution")
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    ax2.grid(True, alpha=0.3)
    
    # Plot points by category with consistent colors
    # Vertices
    if len(original_classified['vertices']) > 0:
        ax1.scatter(original_classified['vertices'][:, 0], original_classified['vertices'][:, 1], 
                    marker='o', color='red', s=60, label='Vertices')
    if len(improved_classified['vertices']) > 0:
        ax2.scatter(improved_classified['vertices'][:, 0], improved_classified['vertices'][:, 1], 
                    marker='o', color='red', s=60, label='Vertices')
        
    # Edges (in 2D, these are the middle points)
    if len(original_classified['edges']) > 0:
        ax1.scatter(original_classified['edges'][:, 0], original_classified['edges'][:, 1], 
                    marker='s', color='blue', s=40, label='Edges')
    if len(improved_classified['edges']) > 0:
        ax2.scatter(improved_classified['edges'][:, 0], improved_classified['edges'][:, 1], 
                    marker='s', color='blue', s=40, label='Edges')
        
    # Interior
    if len(original_classified['interior']) > 0:
        ax1.scatter(original_classified['interior'][:, 0], original_classified['interior'][:, 1], 
                    marker='^', color='green', s=30, label='Interior')
    if len(improved_classified['interior']) > 0:
        ax2.scatter(improved_classified['interior'][:, 0], improved_classified['interior'][:, 1], 
                    marker='^', color='green', s=30, label='Interior')
    
    # Set consistent axis limits
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add constraint line
    ax1.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Sum = 1')
    ax2.plot([0, 1], [1, 0], 'k--', alpha=0.5, label='Sum = 1')
    
    # Add legends
    ax1.legend()
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("binary_distribution_comparison.png", dpi=300, bbox_inches='tight')
    print("\nBinary plot saved as 'binary_distribution_comparison.png'")
    plt.close()

def plot_scatter_comparison(original_classified, improved_classified):
    """
    Fallback scatter plot comparison (when ternary is not available)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Component pairs
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = ["Comp1 vs Comp2", "Comp1 vs Comp3", "Comp2 vs Comp3"]
    
    # Plot each pair for original candidates
    for i, ((c1, c2), title) in enumerate(zip(pairs, pair_names)):
        ax = axes[0, i]
        
        # Vertices
        if len(original_classified['vertices']) > 0:
            ax.scatter(original_classified['vertices'][:, c1], 
                      original_classified['vertices'][:, c2], 
                      marker='o', color='red', s=60, label='Vertices')
        
        # Edges
        if len(original_classified['edges']) > 0:
            ax.scatter(original_classified['edges'][:, c1], 
                      original_classified['edges'][:, c2], 
                      marker='s', color='blue', s=40, label='Edges')
        
        # Interior
        if len(original_classified['interior']) > 0:
            ax.scatter(original_classified['interior'][:, c1], 
                      original_classified['interior'][:, c2], 
                      marker='^', color='green', s=30, label='Interior')
        
        ax.set_xlabel(f"Component {c1+1}")
        ax.set_ylabel(f"Component {c2+1}")
        ax.set_title(f"Original: {title}")
        ax.grid(True, alpha=0.3)
        
        # Only show legend on the first subplot
        if i == 0:
            ax.legend()
    
    # Plot each pair for improved candidates
    for i, ((c1, c2), title) in enumerate(zip(pairs, pair_names)):
        ax = axes[1, i]
        
        # Vertices
        if len(improved_classified['vertices']) > 0:
            ax.scatter(improved_classified['vertices'][:, c1], 
                      improved_classified['vertices'][:, c2], 
                      marker='o', color='red', s=60, label='Vertices')
        
        # Edges
        if len(improved_classified['edges']) > 0:
            ax.scatter(improved_classified['edges'][:, c1], 
                      improved_classified['edges'][:, c2], 
                      marker='s', color='blue', s=40, label='Edges')
        
        # Interior
        if len(improved_classified['interior']) > 0:
            ax.scatter(improved_classified['interior'][:, c1], 
                      improved_classified['interior'][:, c2], 
                      marker='^', color='green', s=30, label='Interior')
        
        ax.set_xlabel(f"Component {c1+1}")
        ax.set_ylabel(f"Component {c2+1}")
        ax.set_title(f"Improved: {title}")
        ax.grid(True, alpha=0.3)
        
        # Only show legend on the first subplot
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig("scatter_distribution_comparison.png", dpi=300, bbox_inches='tight')
    print("\nScatter plot saved as 'scatter_distribution_comparison.png'")
    plt.close()

if __name__ == "__main__":
    # Run the comparison for 3 components
    compare_candidate_distributions(n_points=500, n_components=3)
