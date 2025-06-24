"""
Enhanced Mixture Design with Flexible Run Numbers
Allows setting custom number of runs for all design types
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import itertools
from mixture_designs import MixtureDesign


class EnhancedMixtureDesign(MixtureDesign):
    """
    Enhanced mixture design class with flexible run number options
    Implements the fixed space solution approach for fixed components
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None,
                 use_parts_mode: bool = False, fixed_components: Dict[str, float] = None):
        """
        Initialize enhanced mixture design with fixed space solution approach
        """
        super().__init__(n_components, component_names, component_bounds, 
                        use_parts_mode, fixed_components)
    
    def generate_mixture_design(self, 
                               design_type: str = "d-optimal",
                               n_runs: Optional[int] = None,
                               model_type: str = "quadratic",
                               augment_strategy: str = "centroid",
                               random_seed: Optional[int] = None) -> np.ndarray:
        """
        Unified interface to generate mixture designs with flexible run numbers
        
        Parameters:
        -----------
        design_type : str
            Type of design: "simplex-lattice", "simplex-centroid", "d-optimal", 
            "i-optimal", "space-filling", "custom"
        n_runs : int, optional
            Desired number of runs. If None, uses default for design type
        model_type : str
            Model type: "linear", "quadratic", "cubic"
        augment_strategy : str
            Strategy for adding/removing runs: "centroid", "replicate", 
            "d-optimal", "space-filling", "subset"
        random_seed : int, optional
            Random seed for reproducibility
        
        Returns:
        --------
        np.ndarray : Design matrix with specified number of runs
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Calculate minimum runs needed for model
        min_runs = self._get_minimum_runs(model_type)
        
        # Generate the design
        if design_type == "simplex-lattice":
            design = self._flexible_simplex_lattice(n_runs, model_type, augment_strategy)
        
        elif design_type == "simplex-centroid":
            design = self._flexible_simplex_centroid(n_runs, augment_strategy)
        
        elif design_type == "d-optimal":
            if n_runs is None:
                n_runs = max(min_runs, self.n_components * 2)
            design = self.generate_d_optimal_mixture(n_runs, model_type, random_seed=random_seed)
        
        elif design_type == "i-optimal":
            if n_runs is None:
                n_runs = max(min_runs, self.n_components * 2)
            design = self.generate_i_optimal_mixture(n_runs, model_type, random_seed=random_seed)
        
        elif design_type == "space-filling":
            if n_runs is None:
                n_runs = max(min_runs, self.n_components * 3)
            design = self._generate_space_filling_design(n_runs)
        
        elif design_type == "custom":
            if n_runs is None:
                n_runs = max(min_runs, self.n_components * 2)
            design = self._generate_custom_design(n_runs, model_type)
        
        else:
            raise ValueError(f"Unknown design type: {design_type}")
        
        # CRITICAL FIX: Don't apply post-processing again!
        # The parent class methods (generate_d_optimal_mixture, generate_i_optimal_mixture) 
        # already apply post-processing for fixed components.
        # Applying it again here causes double processing and corrupts the fixed component values.
        # Only apply post-processing for methods that don't do it internally.
        
        if design_type in ["space-filling", "custom"]:
            # These methods don't apply post-processing internally
            if self.fixed_components:
                design = self._post_process_design_fixed_components(design)
        
        # Note: simplex-lattice, simplex-centroid, d-optimal, i-optimal already apply post-processing
        
        return design
    
    def _get_minimum_runs(self, model_type: str) -> int:
        """Calculate minimum number of runs needed for model type"""
        n = self.n_components
        
        # Account for fixed components
        if self.fixed_components:
            n_variable = n - len(self.fixed_components)
        else:
            n_variable = n
        
        if model_type == "linear":
            return n_variable
        elif model_type == "quadratic":
            return n_variable + (n_variable * (n_variable - 1)) // 2
        elif model_type == "cubic":
            linear_terms = n_variable
            quadratic_terms = (n_variable * (n_variable - 1)) // 2
            cubic_terms = (n_variable * (n_variable - 1) * (n_variable - 2)) // 6
            return linear_terms + quadratic_terms + cubic_terms
        else:
            return n_variable
    
    def _flexible_simplex_lattice(self, n_runs: Optional[int], 
                                 model_type: str, 
                                 augment_strategy: str) -> np.ndarray:
        """Generate simplex lattice with flexible number of runs"""
        # Find appropriate degree for simplex lattice
        if n_runs is None:
            # Use default degree based on model type
            if model_type == "linear":
                degree = 1
            elif model_type == "quadratic":
                degree = 2
            else:
                degree = 3
        else:
            # Find degree that gives closest number of runs
            degree = self._find_best_lattice_degree(n_runs)
        
        # Generate base lattice
        base_design = self.generate_simplex_lattice(degree)
        
        if n_runs is None or len(base_design) == n_runs:
            return base_design
        
        # Adjust number of runs
        return self._adjust_design_runs(base_design, n_runs, augment_strategy)
    
    def _flexible_simplex_centroid(self, n_runs: Optional[int], 
                                  augment_strategy: str) -> np.ndarray:
        """Generate simplex centroid with flexible number of runs"""
        # Generate base centroid design
        base_design = self.generate_simplex_centroid()
        
        if n_runs is None or len(base_design) == n_runs:
            return base_design
        
        # Adjust number of runs
        return self._adjust_design_runs(base_design, n_runs, augment_strategy)
    
    def _find_best_lattice_degree(self, target_runs: int) -> int:
        """Find lattice degree that gives closest to target runs"""
        best_degree = 1
        min_diff = float('inf')
        
        for degree in range(1, 10):  # Test degrees 1-9
            n_runs = self._calculate_lattice_runs(degree)
            diff = abs(n_runs - target_runs)
            
            if diff < min_diff:
                min_diff = diff
                best_degree = degree
            
            # If we've exceeded target by a lot, stop searching
            if n_runs > target_runs * 2:
                break
        
        return best_degree
    
    def _calculate_lattice_runs(self, degree: int) -> int:
        """Calculate number of runs for simplex lattice of given degree"""
        from math import factorial
        n = self.n_components
        if self.fixed_components:
            n = n - len(self.fixed_components)
        return factorial(n + degree - 1) // (factorial(degree) * factorial(n - 1))
    
    def _adjust_design_runs(self, base_design: np.ndarray, 
                           target_runs: int, 
                           strategy: str) -> np.ndarray:
        """Adjust design to have target number of runs"""
        current_runs = len(base_design)
        
        if current_runs == target_runs:
            return base_design
        
        elif current_runs > target_runs:
            # Need to reduce runs
            if strategy == "subset":
                return self._select_optimal_subset(base_design, target_runs)
            elif strategy == "d-optimal":
                return self._select_d_optimal_subset(base_design, target_runs)
            else:
                # Default: select diverse subset
                return self._select_diverse_subset(base_design, target_runs)
        
        else:
            # Need to add runs
            if strategy == "centroid":
                return self._augment_with_centroids(base_design, target_runs)
            elif strategy == "replicate":
                return self._augment_with_replicates(base_design, target_runs)
            elif strategy == "d-optimal":
                return self._augment_d_optimal(base_design, target_runs)
            elif strategy == "space-filling":
                return self._augment_space_filling(base_design, target_runs)
            else:
                # Default: augment with space-filling points
                return self._augment_space_filling(base_design, target_runs)
    
    def _select_optimal_subset(self, design: np.ndarray, n_select: int) -> np.ndarray:
        """Select optimal subset of points maintaining good coverage"""
        from scipy.spatial import distance_matrix
        
        # Start with most extreme points
        selected_indices = []
        remaining_indices = list(range(len(design)))
        
        # Select first point (closest to pure component)
        min_purity = float('inf')
        first_idx = 0
        for i, point in enumerate(design):
            purity = 1 - np.max(point)
            if purity < min_purity:
                min_purity = purity
                first_idx = i
        
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Iteratively select points that maximize minimum distance
        while len(selected_indices) < n_select and remaining_indices:
            selected_points = design[selected_indices]
            max_min_dist = -1
            best_idx = remaining_indices[0]
            
            for idx in remaining_indices:
                point = design[idx]
                # Calculate minimum distance to selected points
                dists = np.linalg.norm(selected_points - point, axis=1)
                min_dist = np.min(dists)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return design[selected_indices]
    
    def _select_d_optimal_subset(self, design: np.ndarray, n_select: int) -> np.ndarray:
        """Select D-optimal subset"""
        # Use exchange algorithm to find D-optimal subset
        from itertools import combinations
        
        best_subset = None
        best_d_eff = -float('inf')
        
        # For small designs, try all combinations
        if len(design) <= 20:
            for indices in combinations(range(len(design)), n_select):
                subset = design[list(indices)]
                d_eff = self._calculate_d_efficiency(subset, "quadratic")
                if d_eff > best_d_eff:
                    best_d_eff = d_eff
                    best_subset = subset
        else:
            # For larger designs, use greedy selection
            best_subset = self._select_optimal_subset(design, n_select)
        
        return best_subset
    
    def _select_diverse_subset(self, design: np.ndarray, n_select: int) -> np.ndarray:
        """Select diverse subset using maximin distance criterion"""
        from sklearn.metrics import pairwise_distances
        
        # Calculate pairwise distances
        distances = pairwise_distances(design)
        
        # Start with two most distant points
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        selected = [i, j]
        
        # Iteratively add points that maximize minimum distance
        while len(selected) < n_select:
            remaining = [idx for idx in range(len(design)) if idx not in selected]
            if not remaining:
                break
            
            best_idx = remaining[0]
            best_min_dist = 0
            
            for idx in remaining:
                # Find minimum distance to selected points
                min_dist = np.min([distances[idx, s] for s in selected])
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            selected.append(best_idx)
        
        return design[selected]
    
    def _augment_with_centroids(self, design: np.ndarray, target_runs: int) -> np.ndarray:
        """Augment design with centroids of subsets"""
        augmented = design.copy()
        
        while len(augmented) < target_runs:
            # Generate centroids of random subsets
            n_points = min(3, len(design))  # Use 2-3 points for centroids
            indices = np.random.choice(len(design), n_points, replace=False)
            centroid = np.mean(design[indices], axis=0)
            
            # Ensure it's a valid mixture point
            centroid = centroid / np.sum(centroid)
            
            if self._check_bounds(centroid):
                augmented = np.vstack([augmented, centroid])
        
        # Return design without post-processing - will be done at the end of generate_mixture_design()
        return augmented[:target_runs]
    
    def _augment_with_replicates(self, design: np.ndarray, target_runs: int) -> np.ndarray:
        """Augment design with replicates of existing points"""
        n_additional = target_runs - len(design)
        
        if n_additional <= len(design):
            # Replicate most important points (vertices, centroid)
            importance_scores = []
            for point in design:
                # Higher score for pure components and centroid
                purity = np.max(point)
                uniformity = 1 - np.std(point)
                score = purity + uniformity
                importance_scores.append(score)
            
            # Select points to replicate based on importance
            indices = np.argsort(importance_scores)[-n_additional:]
            replicates = design[indices]
        else:
            # Need multiple replicates
            n_full_reps = n_additional // len(design)
            n_partial = n_additional % len(design)
            
            replicates = np.tile(design, (n_full_reps, 1))
            if n_partial > 0:
                replicates = np.vstack([replicates, design[:n_partial]])
        
        final_design = np.vstack([design, replicates])
        # Return design without premature post-processing
        return final_design
    
    def _augment_d_optimal(self, design: np.ndarray, target_runs: int) -> np.ndarray:
        """Augment design using D-optimality criterion"""
        augmented = design.copy()
        candidates = self._generate_candidate_points(100)
        
        while len(augmented) < target_runs:
            best_candidate = None
            best_d_eff = -float('inf')
            
            for candidate in candidates:
                # Skip if too close to existing points
                min_dist = np.min(np.linalg.norm(augmented - candidate, axis=1))
                if min_dist < 0.01:
                    continue
                
                # Evaluate D-efficiency with this candidate
                temp_design = np.vstack([augmented, candidate])
                d_eff = self._calculate_d_efficiency(temp_design, "quadratic")
                
                if d_eff > best_d_eff:
                    best_d_eff = d_eff
                    best_candidate = candidate
            
            if best_candidate is not None:
                augmented = np.vstack([augmented, best_candidate])
            else:
                # Fallback: add random feasible point
                random_point = candidates[np.random.randint(len(candidates))]
                augmented = np.vstack([augmented, random_point])
        
        # Return design without premature post-processing
        return augmented
    
    def _augment_space_filling(self, design: np.ndarray, target_runs: int) -> np.ndarray:
        """Augment design with space-filling points"""
        augmented = design.copy()
        
        while len(augmented) < target_runs:
            candidates = self._generate_candidate_points(50)
            
            # Find candidate that maximizes minimum distance to existing points
            best_candidate = candidates[0]
            best_min_dist = 0
            
            for candidate in candidates:
                distances = np.linalg.norm(augmented - candidate, axis=1)
                min_dist = np.min(distances)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate
            
            augmented = np.vstack([augmented, best_candidate])
        
        # Return design without premature post-processing
        return augmented
    
    def _generate_space_filling_design(self, n_runs: int) -> np.ndarray:
        """Generate space-filling design using maximin distance criterion"""
        # Start with random feasible points
        candidates = self._generate_candidate_points(n_runs * 10)
        
        # Select subset using maximin criterion
        selected = []
        remaining = list(range(len(candidates)))
        
        # Start with random point
        first_idx = np.random.randint(len(candidates))
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Iteratively add points
        while len(selected) < n_runs and remaining:
            best_idx = remaining[0]
            best_min_dist = 0
            
            for idx in remaining:
                # Calculate minimum distance to selected points
                point = candidates[idx]
                selected_points = candidates[selected]
                distances = np.linalg.norm(selected_points - point, axis=1)
                min_dist = np.min(distances)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx
            
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return candidates[selected]
    
    def _generate_custom_design(self, n_runs: int, model_type: str) -> np.ndarray:
        """Generate custom design combining multiple strategies"""
        designs = []
        
        # Add some lattice points if possible
        try:
            lattice = self.generate_simplex_lattice(2)
            if len(lattice) > 0:
                designs.append(lattice)
        except:
            pass
        
        # Add centroid points
        try:
            centroid = self.generate_simplex_centroid()
            if len(centroid) > 0:
                designs.append(centroid)
        except:
            pass
        
        # Combine all designs
        if designs:
            combined = np.vstack(designs)
            # Remove duplicates
            unique_design = []
            for point in combined:
                if not any(np.allclose(point, p, atol=1e-6) for p in unique_design):
                    unique_design.append(point)
            combined = np.array(unique_design)
        else:
            combined = self._generate_candidate_points(n_runs)
        
        # Adjust to target number of runs
        if len(combined) > n_runs:
            return self._select_d_optimal_subset(combined, n_runs)
        elif len(combined) < n_runs:
            return self._augment_d_optimal(combined, n_runs)
        else:
            return combined
    
    def compare_designs_fixed_runs(self, n_runs: int, model_type: str = "quadratic") -> pd.DataFrame:
        """Compare different design types with fixed number of runs"""
        design_types = ["simplex-lattice", "simplex-centroid", "d-optimal", 
                       "i-optimal", "space-filling", "custom"]
        
        results = []
        
        for design_type in design_types:
            try:
                # Generate design
                design = self.generate_mixture_design(
                    design_type=design_type,
                    n_runs=n_runs,
                    model_type=model_type,
                    random_seed=42
                )
                
                # Evaluate design
                eval_results = self.evaluate_mixture_design(design, model_type)
                
                # Calculate additional metrics
                min_point_distance = float('inf')
                if len(design) > 1:
                    for i in range(len(design)):
                        for j in range(i + 1, len(design)):
                            dist = np.linalg.norm(design[i] - design[j])
                            min_point_distance = min(min_point_distance, dist)
                
                results.append({
                    'Design Type': design_type,
                    'Actual Runs': len(design),
                    'D-efficiency': eval_results['d_efficiency'],
                    'I-efficiency': eval_results['i_efficiency'],
                    'Min Point Distance': min_point_distance
                })
                
            except Exception as e:
                results.append({
                    'Design Type': design_type,
                    'Actual Runs': 0,
                    'D-efficiency': -1,
                    'I-efficiency': -1,
                    'Min Point Distance': -1,
                    'Error': str(e)
                })
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    print("=== Enhanced Mixture Design with Flexible Run Numbers ===\n")
    
    # Create enhanced mixture design
    component_names = ['A', 'B', 'C']
    component_bounds = [(0.1, 0.7), (0.1, 0.7), (0.1, 0.7)]
    
    enhanced_design = EnhancedMixtureDesign(3, component_names, component_bounds)
    
    # Example 1: Generate designs with exactly 15 runs
    print("1. Generating designs with exactly 15 runs:")
    print("-" * 50)
    
    for design_type in ["simplex-lattice", "d-optimal", "space-filling"]:
        design = enhanced_design.generate_mixture_design(
            design_type=design_type,
            n_runs=15,
            model_type="quadratic",
            random_seed=42
        )
        print(f"\n{design_type.title()} Design (15 runs):")
        print(pd.DataFrame(design, columns=component_names).round(3).head())
        print(f"Shape: {design.shape}")
    
    # Example 2: Compare designs with fixed number of runs
    print("\n\n2. Comparing all design types with 20 runs:")
    print("-" * 50)
    comparison = enhanced_design.compare_designs_fixed_runs(n_runs=20, model_type="quadratic")
    print(comparison.round(4))
    
    # Example 3: Augmentation strategies
    print("\n\n3. Different augmentation strategies for Simplex Lattice:")
    print("-" * 50)
    
    base_lattice = enhanced_design.generate_simplex_lattice(degree=2)
    print(f"Base lattice has {len(base_lattice)} runs")
    
    for strategy in ["centroid", "replicate", "d-optimal", "space-filling"]:
        augmented = enhanced_design.generate_mixture_design(
            design_type="simplex-lattice",
            n_runs=12,
            augment_strategy=strategy,
            random_seed=42
        )
        print(f"\nAugmented with '{strategy}' to 12 runs:")
        eval_results = enhanced_design.evaluate_mixture_design(augmented, "quadratic")
        print(f"D-efficiency: {eval_results['d_efficiency']:.4f}")
    
    # Example 4: Custom run numbers for different models
    print("\n\n4. Optimal run numbers for different model complexities:")
    print("-" * 50)
    
    for model in ["linear", "quadratic", "cubic"]:
        min_runs = enhanced_design._get_minimum_runs(model)
        recommended_runs = int(min_runs * 1.5)
        
        design = enhanced_design.generate_mixture_design(
            design_type="d-optimal",
            n_runs=recommended_runs,
            model_type=model,
            random_seed=42
        )
        
        print(f"\n{model.title()} model:")
        print(f"Minimum runs: {min_runs}")
        print(f"Recommended runs: {recommended_runs}")
        print(f"Generated design shape: {design.shape}")
    
    print("\n=== Demo Complete ===")
