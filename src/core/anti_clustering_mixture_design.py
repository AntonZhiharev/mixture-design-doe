"""
Anti-Clustering Mixture Design Implementation
Solves clustering issues in parts mode with fixed components.

Key Features:
- Adaptive candidate generation based on available design space
- Constraint-aware space-filling with minimum distance enforcement
- Multi-objective optimization balancing D-efficiency and space-filling
- Intelligent fallback strategies for severely constrained spaces
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import warnings
from scipy.spatial.distance import pdist, squareform

class AntiClusteringMixtureDesign:
    """
    Anti-clustering mixture design generator that prevents point clustering
    in constrained design spaces typical of fixed parts mixtures.
    """
    
    def __init__(self, component_names: List[str], 
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None,
                 min_distance_factor: float = 0.1,
                 space_filling_weight: float = 0.3,
                 **kwargs):
        """
        Initialize anti-clustering mixture design generator.
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components in the mixture
        fixed_parts : Dict[str, float], optional
            Dictionary mapping fixed component names to their constant parts amount
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable components in parts
        min_distance_factor : float, default 0.1
            Minimum distance between points as fraction of space diagonal
        space_filling_weight : float, default 0.3
            Weight for space-filling objective vs D-efficiency (0=pure D-optimal, 1=pure space-filling)
        """
        
        # Handle backward compatibility arguments
        if kwargs:
            deprecated_args = list(kwargs.keys())
            warnings.warn(
                f"Arguments {deprecated_args} are deprecated and ignored. "
                f"Use 'fixed_parts' and 'variable_bounds' instead.",
                DeprecationWarning,
                stacklevel=2
            )
        
        self.component_names = component_names
        self.fixed_parts = fixed_parts or {}
        self.variable_bounds = variable_bounds or {}
        self.min_distance_factor = min_distance_factor
        self.space_filling_weight = space_filling_weight
        
        # Setup derived attributes
        self.variable_names = [name for name in self.component_names if name not in self.fixed_parts]
        self.fixed_names = list(self.fixed_parts.keys())
        self.n_components = len(self.component_names)
        self.n_fixed = len(self.fixed_parts)
        self.n_variable = len(self.variable_names)
        
        # Validate inputs
        self._validate_inputs()
        
        # Calculate design space properties
        self.total_fixed_parts = sum(self.fixed_parts.values())
        self.variable_total_range = self._calculate_variable_ranges()
        self._calculate_space_metrics()
        
        # Store last generated design
        self.last_design = None
        self.last_parts_design = None
        self.last_batch_sizes = None
    
    def _validate_inputs(self):
        """Validate and setup default bounds."""
        if not self.component_names:
            raise ValueError("component_names cannot be empty")
        
        if not self.fixed_parts:
            raise ValueError("At least one fixed component must be specified")
        
        # Check for unknown fixed components
        unknown_fixed = set(self.fixed_parts.keys()) - set(self.component_names)
        if unknown_fixed:
            raise ValueError(f"Unknown fixed components: {unknown_fixed}")
        
        # Set default bounds for variable components
        for name in self.variable_names:
            if name not in self.variable_bounds:
                self.variable_bounds[name] = (0.0, 100.0)
        
        print(f"\n🔧 Anti-Clustering Mixture Design Initialized:")
        print(f"   Total components: {len(self.component_names)}")
        print(f"   Fixed components: {len(self.fixed_parts)} {list(self.fixed_parts.keys())}")
        print(f"   Variable components: {len(self.variable_names)} {self.variable_names}")
        print(f"   Anti-clustering factor: {self.min_distance_factor}")
        print(f"   Space-filling weight: {self.space_filling_weight}")
    
    def _calculate_variable_ranges(self) -> Tuple[float, float]:
        """Calculate the range of total variable parts."""
        min_total = sum(bounds[0] for bounds in self.variable_bounds.values())
        max_total = sum(bounds[1] for bounds in self.variable_bounds.values())
        return min_total, max_total
    
    def _calculate_space_metrics(self):
        """Calculate design space metrics for clustering analysis."""
        # Calculate space diagonal for minimum distance scaling
        if self.n_variable >= 2:
            # Use first two variable components for space diagonal calculation
            var_ranges = []
            for name in self.variable_names[:2]:
                min_val, max_val = self.variable_bounds[name]
                var_ranges.append(max_val - min_val)
            
            self.space_diagonal = np.sqrt(sum(r**2 for r in var_ranges))
            self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
        else:
            self.space_diagonal = max(self.variable_bounds[self.variable_names[0]]) if self.variable_names else 1.0
            self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
        
        print(f"   Space diagonal: {self.space_diagonal:.3f}")
        print(f"   Minimum distance threshold: {self.min_distance_threshold:.3f}")
    
    def generate_design(self, n_runs: int, design_type: str = "d-optimal", 
                       model_type: str = "quadratic", max_iter: int = 1000, 
                       random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate anti-clustering mixture design.
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        design_type : str, default "d-optimal"
            Type of design ("d-optimal", "i-optimal", "space-filling")
        model_type : str, default "quadratic"
            Model type ("linear", "quadratic", "cubic")
        max_iter : int, default 1000
            Maximum iterations for optimization
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        pd.DataFrame
            Anti-clustering design matrix
        """
        print(f"\n🚀 Generating ANTI-CLUSTERING {design_type} design with {n_runs} runs")
        print(f"   Model type: {model_type}")
        print(f"   Space-filling weight: {self.space_filling_weight}")
        print(f"   Minimum distance enforcement: {self.min_distance_threshold:.3f}")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate design based on type
        if design_type.lower() == "d-optimal":
            parts_design, prop_design, batch_sizes = self._generate_anti_clustering_d_optimal(
                n_runs, model_type, max_iter, random_seed
            )
        elif design_type.lower() == "i-optimal":
            parts_design, prop_design, batch_sizes = self._generate_anti_clustering_i_optimal(
                n_runs, model_type, max_iter, random_seed
            )
        elif design_type.lower() == "space-filling":
            parts_design, prop_design, batch_sizes = self._generate_space_filling_design(
                n_runs, random_seed
            )
        else:
            raise ValueError(f"Unsupported design type: {design_type}")
        
        # Store results
        self.last_design = prop_design
        self.last_parts_design = parts_design
        self.last_batch_sizes = batch_sizes
        
        # Create results DataFrame
        results_df = self._create_results_dataframe(parts_design, prop_design, batch_sizes)
        
        # Analyze anti-clustering performance
        self._analyze_anti_clustering_performance(parts_design)
        
        return results_df
    
    def _generate_anti_clustering_d_optimal(self, n_runs: int, model_type: str, 
                                          max_iter: int, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate D-optimal design with anti-clustering constraints."""
        
        print(f"\n🔧 Anti-Clustering D-Optimal Generation:")
        
        # Generate enhanced candidate set with anti-clustering focus
        candidate_parts, candidate_props, candidate_batches = self._generate_anti_clustering_candidates(3000)
        
        # Select initial design with space-filling priority
        initial_indices = self._select_space_filling_initial_design(candidate_props, n_runs)
        current_design_indices = list(initial_indices)
        
        # Calculate initial metrics
        initial_design = candidate_props[current_design_indices]
        initial_d_eff = self._calculate_d_efficiency(initial_design, model_type)
        initial_space_fill = self._calculate_space_filling_score(candidate_parts[current_design_indices])
        
        print(f"  Initial D-efficiency: {initial_d_eff:.6e}")
        print(f"  Initial space-filling score: {initial_space_fill:.6f}")
        
        # Multi-objective coordinate exchange
        best_combined_score = self._calculate_combined_score(initial_d_eff, initial_space_fill)
        best_design_indices = current_design_indices.copy()
        
        print(f"\n🔄 Running anti-clustering coordinate exchange:")
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                current_idx = current_design_indices[i]
                best_replacement_idx = current_idx
                best_combined = best_combined_score
                
                # Try each candidate as replacement
                for candidate_idx in range(len(candidate_props)):
                    if candidate_idx in current_design_indices:
                        continue
                    
                    # Create test design
                    test_indices = current_design_indices.copy()
                    test_indices[i] = candidate_idx
                    test_design_props = candidate_props[test_indices]
                    test_design_parts = candidate_parts[test_indices]
                    
                    # Check minimum distance constraint
                    if not self._check_minimum_distance_constraint(test_design_parts):
                        continue
                    
                    # Calculate combined score
                    try:
                        d_eff = self._calculate_d_efficiency(test_design_props, model_type)
                        space_fill = self._calculate_space_filling_score(test_design_parts)
                        combined_score = self._calculate_combined_score(d_eff, space_fill)
                        
                        if combined_score > best_combined:
                            best_combined = combined_score
                            best_replacement_idx = candidate_idx
                            improved = True
                    except:
                        continue
                
                # Apply best replacement
                if best_replacement_idx != current_idx:
                    current_design_indices[i] = best_replacement_idx
                    best_combined_score = best_combined
            
            if improved:
                best_design_indices = current_design_indices.copy()
                if (iteration + 1) % 200 == 0:
                    print(f"  Iteration {iteration + 1}: Combined score = {best_combined_score:.6f}")
            else:
                break
        
        print(f"  Converged after {iteration + 1} iterations")
        print(f"  Final combined score: {best_combined_score:.6f}")
        
        # Extract final design
        final_parts = candidate_parts[best_design_indices]
        final_props = candidate_props[best_design_indices]
        final_batches = candidate_batches[best_design_indices]
        
        return final_parts, final_props, final_batches
    
    def _generate_anti_clustering_i_optimal(self, n_runs: int, model_type: str, 
                                          max_iter: int, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate I-optimal design with anti-clustering constraints."""
        
        print(f"\n🔧 Anti-Clustering I-Optimal Generation:")
        
        # For I-optimal, use similar approach but with I-efficiency
        candidate_parts, candidate_props, candidate_batches = self._generate_anti_clustering_candidates(3000)
        
        initial_indices = self._select_space_filling_initial_design(candidate_props, n_runs)
        current_design_indices = list(initial_indices)
        
        # Multi-objective coordinate exchange for I-optimal
        best_design_indices = current_design_indices.copy()
        
        print(f"🔄 Running anti-clustering I-optimal exchange:")
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                current_idx = current_design_indices[i]
                best_replacement_idx = current_idx
                best_score = -np.inf
                
                for candidate_idx in range(len(candidate_props)):
                    if candidate_idx in current_design_indices:
                        continue
                    
                    test_indices = current_design_indices.copy()
                    test_indices[i] = candidate_idx
                    test_design_props = candidate_props[test_indices]
                    test_design_parts = candidate_parts[test_indices]
                    
                    if not self._check_minimum_distance_constraint(test_design_parts):
                        continue
                    
                    try:
                        i_eff = self._calculate_i_efficiency(test_design_props, candidate_props, model_type)
                        space_fill = self._calculate_space_filling_score(test_design_parts)
                        combined_score = self._calculate_combined_score(i_eff, space_fill)
                        
                        if combined_score > best_score:
                            best_score = combined_score
                            best_replacement_idx = candidate_idx
                            improved = True
                    except:
                        continue
                
                if best_replacement_idx != current_idx:
                    current_design_indices[i] = best_replacement_idx
            
            if improved:
                best_design_indices = current_design_indices.copy()
            else:
                break
        
        final_parts = candidate_parts[best_design_indices]
        final_props = candidate_props[best_design_indices]
        final_batches = candidate_batches[best_design_indices]
        
        return final_parts, final_props, final_batches
    
    def _generate_space_filling_design(self, n_runs: int, random_seed: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate pure space-filling design for severely constrained cases."""
        
        print(f"\n🔧 Pure Space-Filling Design Generation:")
        
        candidate_parts, candidate_props, candidate_batches = self._generate_anti_clustering_candidates(5000)
        
        # Use maximin distance design
        selected_indices = self._maximin_design_selection(candidate_parts, n_runs)
        
        final_parts = candidate_parts[selected_indices]
        final_props = candidate_props[selected_indices]
        final_batches = candidate_batches[selected_indices]
        
        return final_parts, final_props, final_batches
    
    def _generate_anti_clustering_candidates(self, n_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate candidates optimized for anti-clustering."""
        
        print(f"  🎯 Generating anti-clustering candidates:")
        
        # Allocate candidates with anti-clustering focus
        n_structured = min(20, max(3**self.n_variable, 9))
        n_lhs = int(0.25 * n_candidates)  # Reduced LHS for more targeted sampling
        n_grid = int(0.35 * n_candidates)  # Grid sampling for even distribution
        n_random = n_candidates - n_structured - n_lhs - n_grid
        
        all_parts = []
        all_props = []
        all_batches = []
        
        # 1. Enhanced structured points
        structured_parts = self._generate_enhanced_structured_points()
        for parts in structured_parts:
            props = self._parts_to_proportions(parts)
            batch_size = np.sum(parts)
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(batch_size)
        
        print(f"    Structured points: {len(structured_parts)}")
        
        # 2. Grid sampling for even distribution
        grid_parts = self._generate_grid_candidates(n_grid)
        for parts in grid_parts:
            props = self._parts_to_proportions(parts)
            batch_size = np.sum(parts)
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(batch_size)
        
        print(f"    Grid candidates: {len(grid_parts)}")
        
        # 3. Latin Hypercube Sampling
        lhs_parts = self._generate_lhs_candidates(n_lhs)
        for parts in lhs_parts:
            props = self._parts_to_proportions(parts)
            batch_size = np.sum(parts)
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(batch_size)
        
        print(f"    LHS candidates: {len(lhs_parts)}")
        
        # 4. Random candidates for diversity
        random_parts = self._generate_random_candidates(n_random)
        for parts in random_parts:
            props = self._parts_to_proportions(parts)
            batch_size = np.sum(parts)
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(batch_size)
        
        print(f"    Random candidates: {len(random_parts)}")
        
        candidate_parts = np.array(all_parts)
        candidate_props = np.array(all_props)
        candidate_batches = np.array(all_batches)
        
        print(f"    Total anti-clustering candidates: {len(candidate_parts)}")
        
        return candidate_parts, candidate_props, candidate_batches
    
    def _generate_grid_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate grid-based candidates for even space coverage."""
        candidates = []
        
        if self.n_variable == 0:
            return candidates
        
        # Calculate grid resolution
        grid_points_per_dim = max(2, int(n_samples ** (1/self.n_variable)))
        
        # Generate grid coordinates for variable components
        if self.n_variable == 1:
            var_name = self.variable_names[0]
            min_val, max_val = self.variable_bounds[var_name]
            values = np.linspace(min_val, max_val, grid_points_per_dim)
            
            for val in values:
                parts = np.zeros(self.n_components)
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                    else:
                        parts[j] = val
                candidates.append(parts)
        
        elif self.n_variable == 2:
            var_names = self.variable_names[:2]
            bounds_0 = self.variable_bounds[var_names[0]]
            bounds_1 = self.variable_bounds[var_names[1]]
            
            values_0 = np.linspace(bounds_0[0], bounds_0[1], grid_points_per_dim)
            values_1 = np.linspace(bounds_1[0], bounds_1[1], grid_points_per_dim)
            
            for val_0 in values_0:
                for val_1 in values_1:
                    parts = np.zeros(self.n_components)
                    # Set fixed components
                    for j, name in enumerate(self.component_names):
                        if name in self.fixed_parts:
                            parts[j] = self.fixed_parts[name]
                    
                    # Set variable components
                    parts[self.component_names.index(var_names[0])] = val_0
                    parts[self.component_names.index(var_names[1])] = val_1
                    
                    candidates.append(parts)
        
        else:
            # For higher dimensions, use random grid sampling
            for _ in range(n_samples):
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                    else:
                        min_val, max_val = self.variable_bounds[name]
                        # Use grid-like sampling
                        grid_val = np.random.choice(np.linspace(min_val, max_val, grid_points_per_dim))
                        parts[j] = grid_val
                
                candidates.append(parts)
        
        return candidates[:n_samples]  # Limit to requested number
    
    def _check_minimum_distance_constraint(self, design_parts: np.ndarray) -> bool:
        """Check if design satisfies minimum distance constraints."""
        if len(design_parts) < 2:
            return True
        
        # Extract variable components for distance calculation
        if self.n_variable < 2:
            return True  # Skip constraint for low-dimensional cases
        
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_parts = design_parts[:, var_indices]
        
        # Calculate pairwise distances
        distances = pdist(var_parts)
        
        # Check if any distance is below threshold
        min_distance = np.min(distances) if len(distances) > 0 else np.inf
        
        return min_distance >= self.min_distance_threshold
    
    def _calculate_space_filling_score(self, design_parts: np.ndarray) -> float:
        """Calculate space-filling quality score."""
        if len(design_parts) < 2 or self.n_variable < 2:
            return 1.0
        
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_parts = design_parts[:, var_indices]
        
        # Calculate minimum distance criterion
        distances = pdist(var_parts)
        min_distance = np.min(distances) if len(distances) > 0 else 0
        
        # Normalize by space diagonal
        space_fill_score = min_distance / self.space_diagonal if self.space_diagonal > 0 else 0
        
        return space_fill_score
    
    def _calculate_combined_score(self, efficiency: float, space_filling: float) -> float:
        """Calculate combined objective score."""
        # Normalize efficiency (assuming typical range)
        norm_efficiency = max(0, min(1, efficiency / 1.0))  # Normalize to [0,1]
        
        # Combine with weights
        combined = (1 - self.space_filling_weight) * norm_efficiency + self.space_filling_weight * space_filling
        
        return combined
    
    def _select_space_filling_initial_design(self, candidates: np.ndarray, n_runs: int) -> List[int]:
        """Select initial design with space-filling priority."""
        return self._maximin_design_selection(candidates, n_runs)
    
    def _maximin_design_selection(self, candidates: np.ndarray, n_runs: int) -> List[int]:
        """Select design points using maximin distance criterion."""
        n_candidates = len(candidates)
        
        if n_candidates <= n_runs:
            return list(range(n_candidates))
        
        if self.n_variable < 2:
            # Fallback to random selection for low dimensions
            return list(np.random.choice(n_candidates, n_runs, replace=False))
        
        # Extract variable components for distance calculations
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_candidates = candidates[:, var_indices]
        
        selected_indices = []
        
        # Select first point randomly
        selected_indices.append(np.random.randint(n_candidates))
        
        # Greedily select remaining points to maximize minimum distance
        for _ in range(n_runs - 1):
            best_idx = -1
            best_min_distance = -1
            
            for candidate_idx in range(n_candidates):
                if candidate_idx in selected_indices:
                    continue
                
                # Calculate distances to all selected points
                candidate_point = var_candidates[candidate_idx]
                distances = []
                
                for selected_idx in selected_indices:
                    selected_point = var_candidates[selected_idx]
                    distance = np.linalg.norm(candidate_point - selected_point)
                    distances.append(distance)
                
                min_distance = min(distances) if distances else np.inf
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_idx = candidate_idx
            
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                # Fallback: select random remaining point
                remaining = [i for i in range(n_candidates) if i not in selected_indices]
                if remaining:
                    selected_indices.append(np.random.choice(remaining))
        
        return selected_indices
    
    def _analyze_anti_clustering_performance(self, parts_design: np.ndarray):
        """Analyze the anti-clustering performance of the generated design."""
        print(f"\n🎯 Anti-Clustering Performance Analysis:")
        
        if self.n_variable < 2:
            print(f"  Analysis not applicable for {self.n_variable} variable components")
            return
        
        # Extract variable components
        var_indices = [self.component_names.index(name) for name in self.variable_names[:2]]
        var_parts = parts_design[:, var_indices]
        
        # Calculate distance metrics
        distances = pdist(var_parts)
        if len(distances) == 0:
            print("  No distances to analyze")
            return
        
        min_distance = np.min(distances)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Count clustered points
        clustered_points = sum(1 for d in distances if d < self.min_distance_threshold)
        total_pairs = len(distances)
        clustering_ratio = clustered_points / total_pairs if total_pairs > 0 else 0
        
        print(f"  Minimum distance: {min_distance:.3f} (threshold: {self.min_distance_threshold:.3f})")
        print(f"  Average distance: {avg_distance:.3f}")
        print(f"  Maximum distance: {max_distance:.3f}")
        print(f"  Clustered pairs: {clustered_points}/{total_pairs} ({clustering_ratio:.1%})")
        
        # Performance assessment
        if min_distance >= self.min_distance_threshold:
            print(f"  ✅ ANTI-CLUSTERING SUCCESS: All points meet distance requirements")
        elif clustering_ratio < 0.2:
            print(f"  ⚠️ Minor clustering: {clustering_ratio:.1%} of point pairs clustered")
        else:
            print(f"  ❌ Significant clustering remains: {clustering_ratio:.1%} of point pairs clustered")
        
        # Space utilization
        space_utilization = min_distance / self.space_diagonal if self.space_diagonal > 0 else 0
        print(f"  Space utilization: {space_utilization:.1%}")
    
    # Helper methods (reuse from parent class)
    def _parts_to_proportions(self, parts: np.ndarray) -> np.ndarray:
        """Convert parts to proportions."""
        total = np.sum(parts)
        if total <= 0:
            raise ValueError("Total parts must be positive")
        return parts / total
    
    def _generate_enhanced_structured_points(self) -> List[np.ndarray]:
        """Generate enhanced structured points."""
        structured_points = []
        
        # Corner points
        if self.n_variable <= 3:
            for i in range(2**self.n_variable):
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                
                # Set variable components to corners
                var_idx = 0
                for j, name in enumerate(self.component_names):
                    if name not in self.fixed_parts:
                        min_val, max_val = self.variable_bounds[name]
                        if (i >> var_idx) & 1:
                            parts[j] = max_val
                        else:
                            parts[j] = min_val
                        var_idx += 1
                
                structured_points.append(parts)
        
        # Centroid
        parts = np.zeros(self.n_components)
        for j, name in enumerate(self.component_names):
            if name in self.fixed_parts:
                parts[j] = self.fixed_parts[name]
            else:
                min_val, max_val = self.variable_bounds[name]
                parts[j] = (min_val + max_val) / 2
        structured_points.append(parts)
        
        return structured_points
    
    def _generate_lhs_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate Latin Hypercube Sampling candidates."""
        if self.n_variable == 0:
            return []
        
        # Generate LHS in [0,1] space
        lhs_samples = self._latin_hypercube_sampling(n_samples, self.n_variable)
        
        candidates = []
        for sample in lhs_samples:
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
            
            # Map LHS samples to variable bounds
            var_idx = 0
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts:
                    min_val, max_val = self.variable_bounds[name]
                    parts[j] = min_val + sample[var_idx] * (max_val - min_val)
                    var_idx += 1
            
            candidates.append(parts)
        
        return candidates
    
    def _latin_hypercube_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """Generate Latin Hypercube Sampling points in [0,1]^n_dims."""
        samples = np.zeros((n_samples, n_dims))
        
        for i in range(n_dims):
            # Create stratified intervals
            intervals = np.arange(n_samples) / n_samples
            # Add random jitter within each interval
            jitter = np.random.uniform(0, 1/n_samples, n_samples)
            stratified_samples = intervals + jitter
            # Random permutation to break correlation between dimensions
            samples[:, i] = np.random.permutation(stratified_samples)
        
        return samples
    
    def _generate_random_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate random candidates using uniform sampling."""
        candidates = []
        
        for _ in range(n_samples):
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
                else:
                    min_val, max_val = self.variable_bounds[name]
                    parts[j] = np.random.uniform(min_val, max_val)
            
            candidates.append(parts)
        
        return candidates
    
    def _calculate_d_efficiency(self, design: np.ndarray, model_type: str) -> float:
        """Calculate D-efficiency of the design."""
        try:
            # Build model matrix
            X = self._build_model_matrix(design, model_type)
            
            # Calculate information matrix
            XTX = X.T @ X
            
            # Calculate determinant
            det = np.linalg.det(XTX)
            
            # Calculate D-efficiency
            n_runs, n_params = X.shape
            d_efficiency = (det / n_runs) ** (1 / n_params) if det > 0 and n_params > 0 else 0.0
            
            return d_efficiency
        except:
            return -np.inf
    
    def _calculate_i_efficiency(self, design: np.ndarray, candidates: np.ndarray, model_type: str) -> float:
        """Calculate I-efficiency of the design."""
        try:
            # Build model matrix for design
            X_design = self._build_model_matrix(design, model_type)
            
            # Calculate information matrix
            XTX = X_design.T @ X_design
            
            # Check if matrix is invertible
            try:
                XTX_inv = np.linalg.inv(XTX)
            except np.linalg.LinAlgError:
                return -np.inf
            
            # Use subset of candidates for efficiency
            candidate_subset = candidates[::10] if len(candidates) > 100 else candidates
            
            # Build model matrix for candidates
            X_pred = self._build_model_matrix(candidate_subset, model_type)
            
            # Calculate average prediction variance
            avg_pred_var = 0.0
            n_pred = len(X_pred)
            
            for i in range(n_pred):
                x_pred = X_pred[i:i+1, :]
                pred_var = x_pred @ XTX_inv @ x_pred.T
                avg_pred_var += pred_var[0, 0]
            
            avg_pred_var /= n_pred
            
            # I-efficiency is inverse of average prediction variance
            i_efficiency = 1.0 / avg_pred_var if avg_pred_var > 0 else 0.0
            
            return i_efficiency
        except Exception:
            return -np.inf
    
    def _build_model_matrix(self, design: np.ndarray, model_type: str) -> np.ndarray:
        """Build model matrix for the given design and model type."""
        n_runs, n_components = design.shape
        
        if model_type == "linear":
            return design
        
        elif model_type == "quadratic":
            X = []
            
            for i in range(n_runs):
                row = []
                
                # Linear terms
                for j in range(n_components):
                    row.append(design[i, j])
                
                # Two-way interactions
                for j in range(n_components):
                    for k in range(j + 1, n_components):
                        row.append(design[i, j] * design[i, k])
                
                X.append(row)
            
            return np.array(X)
        
        elif model_type == "cubic":
            X = []
            
            for i in range(n_runs):
                row = []
                
                # Linear terms
                for j in range(n_components):
                    row.append(design[i, j])
                
                # Two-way interactions
                for j in range(n_components):
                    for k in range(j + 1, n_components):
                        row.append(design[i, j] * design[i, k])
                
                # Three-way interactions
                for j in range(n_components):
                    for k in range(j + 1, n_components):
                        for l in range(k + 1, n_components):
                            row.append(design[i, j] * design[i, k] * design[i, l])
                
                X.append(row)
            
            return np.array(X)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_results_dataframe(self, parts_design: np.ndarray, prop_design: np.ndarray, batch_sizes: np.ndarray) -> pd.DataFrame:
        """Create comprehensive results DataFrame."""
        results = {}
        
        # Add run numbers
        results['Run'] = range(1, len(parts_design) + 1)
        
        # Add parts columns
        for i, name in enumerate(self.component_names):
            results[f'{name}_Parts'] = parts_design[:, i]
        
        # Add proportion columns
        for i, name in enumerate(self.component_names):
            results[f'{name}_Prop'] = prop_design[:, i]
        
        # Add batch sizes
        results['Batch_Size'] = batch_sizes
        
        return pd.DataFrame(results)
    
    # Backward compatibility methods
    def get_parts_design(self) -> pd.DataFrame:
        """Get the parts design from the last generated design."""
        if self.last_parts_design is None:
            raise ValueError("No design has been generated yet. Call generate_design() first.")
        
        parts_columns = [f"{name}_Parts" for name in self.component_names]
        parts_df = pd.DataFrame(self.last_parts_design, columns=parts_columns)
        parts_df.index = [f"Run_{i+1}" for i in range(len(parts_df))]
        return parts_df
    
    def get_proportions_design(self) -> pd.DataFrame:
        """Get the proportions design from the last generated design."""
        if self.last_design is None:
            raise ValueError("No design has been generated yet. Call generate_design() first.")
        
        props_df = pd.DataFrame(self.last_design, columns=self.component_names)
        props_df.index = [f"Run_{i+1}" for i in range(len(props_df))]
        return props_df
    
    @property
    def parts_design(self) -> np.ndarray:
        """Get parts design as numpy array for backward compatibility."""
        return self.last_parts_design
    
    @property
    def prop_design(self) -> np.ndarray:
        """Get proportions design as numpy array for backward compatibility."""
        return self.last_design
    
    @property
    def batch_sizes(self) -> np.ndarray:
        """Get batch sizes as numpy array for backward compatibility."""
        return self.last_batch_sizes
