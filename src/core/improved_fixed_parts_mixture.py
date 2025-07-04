"""
Improved Fixed Parts Mixture Design Implementation

This implementation fixes the poor distribution issues by adapting the sophisticated
algorithms from ProportionalPartsMixture for the fixed components case.

Key Improvements:
1. Latin Hypercube Sampling for variable components
2. Enhanced structured points generation 
3. Proportional relationship awareness
4. Space-filling optimization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings
from scipy import stats


class ImprovedFixedPartsMixture:
    """
    Improved mixture design class for fixed components with better distribution.
    
    This implementation addresses the poor distribution issues found in the original
    TrueFixedComponentsMixture by using sophisticated candidate generation strategies
    adapted from the successful ProportionalPartsMixture approach.
    """
    
    def __init__(self, 
                 component_names: List[str],
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize the improved fixed components mixture design.
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components in the mixture
        fixed_parts : Dict[str, float], optional
            Dictionary mapping fixed component names to their constant parts amount
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable components in parts
        """
        self.component_names = component_names
        self.n_components = len(component_names)
        self.fixed_parts = fixed_parts or {}
        self.variable_bounds = variable_bounds or {}
        
        # Identify fixed vs variable components
        self.fixed_names = list(self.fixed_parts.keys())
        self.variable_names = [name for name in component_names if name not in self.fixed_names]
        self.n_fixed = len(self.fixed_names)
        self.n_variable = len(self.variable_names)
        
        # Calculate total fixed parts
        self.total_fixed_parts = sum(self.fixed_parts.values())
        
        # Set default bounds for variable components if not provided
        for name in self.variable_names:
            if name not in self.variable_bounds:
                self.variable_bounds[name] = (0.0, 100.0)
        
        # Validate and calculate design space
        self._validate_setup()
        self._calculate_design_space()
        self._calculate_variable_proportional_ranges()
        
        print(f"\nImproved Fixed Components Mixture Design Setup:")
        print(f"  Total components: {self.n_components}")
        print(f"  Fixed components: {self.n_fixed} {self.fixed_names}")
        print(f"  Variable components: {self.n_variable} {self.variable_names}")
        print(f"  Total fixed parts: {self.total_fixed_parts}")
        print(f"  Variable space dimensions: {self.n_variable}D")
        print(f"  Batch size range: {self.min_batch_size:.1f} to {self.max_batch_size:.1f}")
    
    def _validate_setup(self):
        """Validate that the fixed components setup is feasible."""
        if not self.fixed_parts:
            raise ValueError("No fixed components specified. Use regular mixture design instead.")
        
        # Check for component name conflicts
        unknown_fixed = set(self.fixed_names) - set(self.component_names)
        if unknown_fixed:
            raise ValueError(f"Unknown fixed components: {unknown_fixed}")
        
        # Check that fixed parts are positive
        for name, parts in self.fixed_parts.items():
            if parts <= 0:
                raise ValueError(f"Fixed component {name} must have positive parts ({parts})")
        
        # Check variable bounds
        for name, (min_parts, max_parts) in self.variable_bounds.items():
            if min_parts < 0:
                raise ValueError(f"Variable component {name} minimum parts cannot be negative ({min_parts})")
            if max_parts <= min_parts:
                raise ValueError(f"Variable component {name} max parts ({max_parts}) must be > min parts ({min_parts})")
    
    def _calculate_design_space(self):
        """Calculate the design space constraints."""
        # Total variable parts range
        self.min_variable_parts = sum(bounds[0] for bounds in self.variable_bounds.values())
        self.max_variable_parts = sum(bounds[1] for bounds in self.variable_bounds.values())
        
        # Batch size range
        self.min_batch_size = self.total_fixed_parts + self.min_variable_parts
        self.max_batch_size = self.total_fixed_parts + self.max_variable_parts
        
        # Variable bounds as array for easier processing
        self.variable_bounds_array = np.array([
            self.variable_bounds[name] for name in self.variable_names
        ])
    
    def _calculate_variable_proportional_ranges(self):
        """
        Calculate proportional ranges for variable components.
        
        This adapts the ProportionalPartsMixture approach for the variable components
        in the context of fixed components consuming constant space.
        """
        if self.n_variable == 0:
            self.variable_proportional_ranges = []
            return
        
        # Calculate feasible total variable parts range
        min_var_total = self.min_variable_parts
        max_var_total = self.max_variable_parts
        
        print(f"  Variable component total range: [{min_var_total:.3f}, {max_var_total:.3f}] parts")
        
        # For each variable component, calculate its feasible proportion range
        # within the variable component space
        self.variable_proportional_ranges = []
        
        for i, var_name in enumerate(self.variable_names):
            min_parts, max_parts = self.variable_bounds[var_name]
            
            # Minimum proportion: when this variable is at minimum and others can be at maximum
            other_max_total = sum(
                self.variable_bounds[other_name][1] 
                for other_name in self.variable_names 
                if other_name != var_name
            )
            min_feasible_var_total = min_parts + other_max_total
            min_proportion = min_parts / min_feasible_var_total if min_feasible_var_total > 0 else 0.0
            
            # Maximum proportion: when this variable is at maximum and others are at minimum
            other_min_total = sum(
                self.variable_bounds[other_name][0] 
                for other_name in self.variable_names 
                if other_name != var_name
            )
            max_feasible_var_total = max_parts + other_min_total
            max_proportion = max_parts / max_feasible_var_total if max_feasible_var_total > 0 else 1.0
            
            # Ensure proportions are valid
            min_proportion = max(0.0, min(1.0, min_proportion))
            max_proportion = max(0.0, min(1.0, max_proportion))
            
            if min_proportion > max_proportion:
                min_proportion, max_proportion = max_proportion, min_proportion
            
            self.variable_proportional_ranges.append((min_proportion, max_proportion))
            
            print(f"    {var_name}: parts [{min_parts:.3f}, {max_parts:.3f}] -> var_props [{min_proportion:.6f}, {max_proportion:.6f}]")
    
    def generate_latin_hypercube_candidates(self, n_candidates: int) -> np.ndarray:
        """
        Generate Latin Hypercube Sampling candidates for variable components.
        
        This ensures better space-filling properties than random sampling.
        
        Parameters:
        -----------
        n_candidates : int
            Number of LHS candidates to generate
            
        Returns:
        --------
        np.ndarray
            Variable component parts matrix (n_candidates x n_variable)
        """
        if self.n_variable == 0:
            return np.zeros((n_candidates, 0))
        
        # Generate LHS samples in [0,1]^n_variable space
        lhs_samples = self._latin_hypercube_sampling(n_candidates, self.n_variable)
        
        # Scale to variable component bounds
        variable_parts = np.zeros((n_candidates, self.n_variable))
        
        for i, var_name in enumerate(self.variable_names):
            min_parts, max_parts = self.variable_bounds[var_name]
            variable_parts[:, i] = min_parts + lhs_samples[:, i] * (max_parts - min_parts)
        
        return variable_parts
    
    def _latin_hypercube_sampling(self, n_samples: int, n_dims: int) -> np.ndarray:
        """
        Generate Latin Hypercube Sampling points in [0,1]^n_dims.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples
        n_dims : int
            Number of dimensions
            
        Returns:
        --------
        np.ndarray
            LHS samples in [0,1]^n_dims
        """
        # Create stratified samples
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
    
    def generate_proportional_variable_candidates(self, n_candidates: int) -> np.ndarray:
        """
        Generate candidates using proportional relationships for variable components.
        
        This adapts the ProportionalPartsMixture approach for variable components.
        """
        if self.n_variable == 0:
            return np.zeros((n_candidates, 0))
        
        variable_parts = []
        
        for _ in range(n_candidates):
            # Generate proportional candidate for variable components
            var_proportions = self._generate_variable_proportional_candidate()
            
            # Convert to parts using multiple total variable parts candidates
            var_parts = self._convert_variable_proportions_to_parts(var_proportions)
            
            variable_parts.append(var_parts)
        
        return np.array(variable_parts)
    
    def _generate_variable_proportional_candidate(self) -> np.ndarray:
        """Generate a proportional candidate for variable components."""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Generate random proportions within calculated ranges
            var_proportions = np.zeros(self.n_variable)
            
            for i, (min_prop, max_prop) in enumerate(self.variable_proportional_ranges):
                if max_prop > min_prop:
                    var_proportions[i] = np.random.uniform(min_prop, max_prop)
                else:
                    var_proportions[i] = min_prop
            
            # Normalize to sum = 1
            current_sum = np.sum(var_proportions)
            if current_sum > 1e-10:
                var_proportions = var_proportions / current_sum
                
                # Check if this candidate can be converted to valid parts
                if self._is_valid_variable_proportional_candidate(var_proportions):
                    return var_proportions
        
        # Fallback: equal proportions
        return np.ones(self.n_variable) / self.n_variable
    
    def _is_valid_variable_proportional_candidate(self, var_proportions: np.ndarray) -> bool:
        """Check if variable proportional candidate can be converted to valid parts."""
        # Try different total variable parts values
        candidate_totals = []
        
        for i, prop in enumerate(var_proportions):
            if prop > 1e-10:
                min_parts, max_parts = self.variable_bounds_array[i]
                candidate_totals.extend([min_parts / prop, max_parts / prop])
        
        if not candidate_totals:
            return False
        
        # Try each candidate total
        for total_var_parts in candidate_totals:
            if total_var_parts <= 0:
                continue
            
            # Check if this total makes all variable components satisfy bounds
            var_parts = var_proportions * total_var_parts
            
            if np.all((var_parts >= self.variable_bounds_array[:, 0] - 1e-10) & 
                     (var_parts <= self.variable_bounds_array[:, 1] + 1e-10)):
                return True
        
        return False
    
    def _convert_variable_proportions_to_parts(self, var_proportions: np.ndarray) -> np.ndarray:
        """Convert variable proportions to parts while respecting bounds."""
        best_parts = None
        min_violation = float('inf')
        
        # Generate candidate total variable parts
        candidate_totals = []
        
        for i, prop in enumerate(var_proportions):
            if prop > 1e-10:
                min_parts, max_parts = self.variable_bounds_array[i]
                candidate_totals.extend([min_parts / prop, max_parts / prop])
        
        # Add some intermediate values
        if candidate_totals:
            min_total = min(candidate_totals)
            max_total = max(candidate_totals)
            for factor in [0.25, 0.5, 0.75]:
                candidate_totals.append(min_total + factor * (max_total - min_total))
        
        # Test each candidate total
        for total_var_parts in candidate_totals:
            if total_var_parts <= 0:
                continue
            
            var_parts = var_proportions * total_var_parts
            
            # Calculate violation
            violation = 0.0
            for i, parts_val in enumerate(var_parts):
                min_parts, max_parts = self.variable_bounds_array[i]
                if parts_val < min_parts:
                    violation += (min_parts - parts_val) ** 2
                elif parts_val > max_parts:
                    violation += (parts_val - max_parts) ** 2
            
            if violation < min_violation:
                min_violation = violation
                best_parts = var_parts.copy()
                
                if violation < 1e-10:  # Exact solution found
                    break
        
        # Fallback if no good solution found
        if best_parts is None:
            avg_total = np.mean([np.sum(self.variable_bounds_array[:, 0]), 
                               np.sum(self.variable_bounds_array[:, 1])])
            best_parts = var_proportions * avg_total
        
        return best_parts
    
    def generate_enhanced_structured_points(self) -> np.ndarray:
        """
        Generate enhanced structured points including corners, edges, and faces.
        
        This provides much better coverage than the basic structured points
        in the original implementation.
        """
        if self.n_variable == 0:
            return np.zeros((1, 0))
        
        structured_points = []
        
        # 1. All corner points of variable space (2^n_variable points)
        for i in range(2**self.n_variable):
            corner = np.zeros(self.n_variable)
            for j in range(self.n_variable):
                if (i >> j) & 1:
                    corner[j] = self.variable_bounds_array[j, 1]  # max
                else:
                    corner[j] = self.variable_bounds_array[j, 0]  # min
            structured_points.append(corner)
        
        # 2. Edge midpoints (for 2D and higher)
        if self.n_variable >= 2:
            for i in range(self.n_variable):
                for j in range(i + 1, self.n_variable):
                    # Midpoint of edge between corners
                    for corner_mask in range(2**(self.n_variable - 2)):
                        edge_point = np.zeros(self.n_variable)
                        
                        # Set the two edge dimensions to their midpoints
                        edge_point[i] = (self.variable_bounds_array[i, 0] + 
                                       self.variable_bounds_array[i, 1]) / 2
                        edge_point[j] = (self.variable_bounds_array[j, 0] + 
                                       self.variable_bounds_array[j, 1]) / 2
                        
                        # Set other dimensions based on corner mask
                        mask_idx = 0
                        for k in range(self.n_variable):
                            if k != i and k != j:
                                if (corner_mask >> mask_idx) & 1:
                                    edge_point[k] = self.variable_bounds_array[k, 1]
                                else:
                                    edge_point[k] = self.variable_bounds_array[k, 0]
                                mask_idx += 1
                        
                        structured_points.append(edge_point)
        
        # 3. Face centers (for 3D and higher)
        if self.n_variable >= 3:
            for i in range(self.n_variable):
                # Center of face perpendicular to dimension i
                for bound_idx in [0, 1]:  # min and max faces
                    face_center = np.zeros(self.n_variable)
                    face_center[i] = self.variable_bounds_array[i, bound_idx]
                    
                    for j in range(self.n_variable):
                        if j != i:
                            face_center[j] = (self.variable_bounds_array[j, 0] + 
                                            self.variable_bounds_array[j, 1]) / 2
                    
                    structured_points.append(face_center)
        
        # 4. Overall centroid
        centroid = np.mean(self.variable_bounds_array, axis=1)
        structured_points.append(centroid)
        
        # 5. Axis-aligned points (each dimension at max, others at center)
        center = np.mean(self.variable_bounds_array, axis=1)
        for i in range(self.n_variable):
            for bound_idx in [0, 1]:
                axis_point = center.copy()
                axis_point[i] = self.variable_bounds_array[i, bound_idx]
                structured_points.append(axis_point)
        
        structured_points = np.array(structured_points)
        
        # Remove duplicates
        unique_points = []
        for point in structured_points:
            is_duplicate = False
            for existing_point in unique_points:
                if np.allclose(point, existing_point, atol=1e-10):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        print(f"  Generated {len(unique_points)} enhanced structured points")
        
        return np.array(unique_points) if unique_points else np.zeros((1, self.n_variable))
    
    def generate_candidate_set(self, n_candidates: int = 2000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate an improved candidate set with better space-filling properties.
        
        Combines multiple strategies:
        1. Latin Hypercube Sampling 
        2. Proportional relationship candidates
        3. Enhanced structured points
        
        Parameters:
        -----------
        n_candidates : int
            Total number of candidates to generate
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (parts_matrix, proportions_matrix, batch_sizes)
        """
        all_variable_parts = []
        
        # 1. Enhanced structured points (high priority)
        structured_var_parts = self.generate_enhanced_structured_points()
        all_variable_parts.append(structured_var_parts)
        
        # 2. Latin Hypercube Sampling (30% of remaining candidates)
        n_remaining = n_candidates - len(structured_var_parts)
        n_lhs = max(1, int(0.3 * n_remaining))
        lhs_var_parts = self.generate_latin_hypercube_candidates(n_lhs)
        all_variable_parts.append(lhs_var_parts)
        
        # 3. Proportional relationship candidates (40% of remaining)
        n_remaining -= n_lhs
        n_prop = max(1, int(0.4 * n_remaining))
        prop_var_parts = self.generate_proportional_variable_candidates(n_prop)
        all_variable_parts.append(prop_var_parts)
        
        # 4. Pure random for diversity (remaining candidates)
        n_random = n_candidates - len(structured_var_parts) - n_lhs - n_prop
        if n_random > 0:
            random_var_parts = self._generate_random_variable_candidates(n_random)
            all_variable_parts.append(random_var_parts)
        
        # Combine all variable parts
        variable_parts_combined = np.vstack(all_variable_parts)
        
        # Convert to full parts matrix (including fixed components)
        parts_matrix = self._create_full_parts_matrix(variable_parts_combined)
        
        # Calculate batch sizes
        batch_sizes = np.sum(parts_matrix, axis=1)
        
        # Convert to proportions
        proportions_matrix = np.zeros_like(parts_matrix)
        for i in range(len(parts_matrix)):
            proportions_matrix[i] = parts_matrix[i] / batch_sizes[i]
        
        print(f"\nImproved candidate set generated:")
        print(f"  Total candidates: {len(parts_matrix)}")
        print(f"  Structured points: {len(structured_var_parts)}")
        print(f"  LHS candidates: {n_lhs}")
        print(f"  Proportional candidates: {n_prop}")
        print(f"  Random candidates: {n_random}")
        
        return parts_matrix, proportions_matrix, batch_sizes
    
    def _generate_random_variable_candidates(self, n_candidates: int) -> np.ndarray:
        """Generate pure random candidates for variable components (for diversity)."""
        if self.n_variable == 0:
            return np.zeros((n_candidates, 0))
        
        variable_parts = np.zeros((n_candidates, self.n_variable))
        
        for i, var_name in enumerate(self.variable_names):
            min_parts, max_parts = self.variable_bounds[var_name]
            variable_parts[:, i] = np.random.uniform(min_parts, max_parts, n_candidates)
        
        return variable_parts
    
    def _create_full_parts_matrix(self, variable_parts: np.ndarray) -> np.ndarray:
        """Create full parts matrix including fixed components."""
        n_candidates = len(variable_parts)
        parts_matrix = np.zeros((n_candidates, self.n_components))
        
        # Set fixed component values
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                parts_matrix[:, i] = self.fixed_parts[name]
        
        # Set variable component values
        for i, name in enumerate(self.component_names):
            if name in self.variable_names:
                var_idx = self.variable_names.index(name)
                parts_matrix[:, i] = variable_parts[:, var_idx]
        
        return parts_matrix
    
    def generate_d_optimal_design(self, n_runs: int, model_type: str = "quadratic", 
                                 max_iter: int = 1000, n_candidates: int = 2000,
                                 random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate D-optimal design using improved candidate generation.
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        model_type : str
            Type of model ("linear", "quadratic", "cubic")
        max_iter : int
            Maximum iterations for coordinate exchange algorithm
        n_candidates : int
            Number of candidate points to generate
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (parts_design, proportions_design, batch_sizes)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"\nGenerating improved D-optimal design for fixed components mixture:")
        print(f"  Runs: {n_runs}")
        print(f"  Model: {model_type}")
        print(f"  Fixed components: {self.fixed_names}")
        
        # Generate improved candidate set
        parts_candidates, prop_candidates, batch_candidates = self.generate_candidate_set(n_candidates)
        
        # Create model matrix for the candidates (use proportions for statistical modeling)
        X_candidates = self._create_model_matrix(prop_candidates, model_type)
        
        # Run coordinate exchange algorithm
        design_indices = self._coordinate_exchange_d_optimal(
            X_candidates, n_runs, max_iter
        )
        
        # Extract the selected design points
        parts_design = parts_candidates[design_indices]
        proportions_design = prop_candidates[design_indices]
        batch_sizes = batch_candidates[design_indices]
        
        # Verify the design
        self._verify_design(parts_design, proportions_design, batch_sizes)
        
        return parts_design, proportions_design, batch_sizes
    
    def _create_model_matrix(self, proportions: np.ndarray, model_type: str) -> np.ndarray:
        """Create the model matrix X for mixture model using proportions."""
        n_points = len(proportions)
        
        if model_type == "linear":
            # For mixture: just the component proportions (no intercept)
            X = proportions.copy()
        
        elif model_type == "quadratic":
            # Components + interaction terms
            n_terms = self.n_components + (self.n_components * (self.n_components - 1)) // 2
            X = np.zeros((n_points, n_terms))
            
            # Linear terms
            X[:, :self.n_components] = proportions
            
            # Interaction terms
            col_idx = self.n_components
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    X[:, col_idx] = proportions[:, i] * proportions[:, j]
                    col_idx += 1
        
        elif model_type == "cubic":
            # Components + quadratic interactions + cubic interactions
            n_linear = self.n_components
            n_quadratic = (self.n_components * (self.n_components - 1)) // 2
            n_cubic = (self.n_components * (self.n_components - 1) * (self.n_components - 2)) // 6
            n_terms = n_linear + n_quadratic + n_cubic
            
            X = np.zeros((n_points, n_terms))
            
            # Linear terms
            X[:, :n_linear] = proportions
            
            # Quadratic interaction terms
            col_idx = n_linear
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    X[:, col_idx] = proportions[:, i] * proportions[:, j]
                    col_idx += 1
            
            # Cubic interaction terms
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    for k in range(j + 1, self.n_components):
                        X[:, col_idx] = (proportions[:, i] * 
                                       proportions[:, j] * 
                                       proportions[:, k])
                        col_idx += 1
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return X
    
    def _coordinate_exchange_d_optimal(self, X_candidates: np.ndarray, 
                                     n_runs: int, max_iter: int) -> np.ndarray:
        """Coordinate exchange algorithm for D-optimal design."""
        n_candidates = len(X_candidates)
        
        # Initialize with random selection
        current_indices = np.random.choice(n_candidates, n_runs, replace=False)
        X_current = X_candidates[current_indices]
        
        try:
            current_det = np.linalg.det(X_current.T @ X_current)
        except:
            current_det = 1e-10
        
        print(f"  Initial D-efficiency: {current_det:.6e}")
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            
            for i in range(n_runs):
                best_det = current_det
                best_candidate = current_indices[i]
                
                # Try replacing point i with each candidate
                for candidate_idx in range(n_candidates):
                    if candidate_idx in current_indices:
                        continue
                    
                    # Create trial design
                    trial_indices = current_indices.copy()
                    trial_indices[i] = candidate_idx
                    X_trial = X_candidates[trial_indices]
                    
                    try:
                        trial_det = np.linalg.det(X_trial.T @ X_trial)
                        
                        if trial_det > best_det:
                            best_det = trial_det
                            best_candidate = candidate_idx
                            improved = True
                    except:
                        continue
                
                # Update if improvement found
                if best_candidate != current_indices[i]:
                    current_indices[i] = best_candidate
                    current_det = best_det
            
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: D-efficiency = {current_det:.6e}")
        
        print(f"  Final D-efficiency after {iteration} iterations: {current_det:.6e}")
        
        return current_indices
    
    def _verify_design(self, parts_design: np.ndarray, proportions_design: np.ndarray, batch_sizes: np.ndarray):
        """Verify that the generated design meets all constraints."""
        print(f"\nImproved Design Verification:")
        
        # Check proportions sum to 1
        prop_sums = np.sum(proportions_design, axis=1)
        prop_ok = np.allclose(prop_sums, 1.0, atol=1e-6)
        print(f"  Proportion sums correct: {prop_ok} (range: {prop_sums.min():.6f} to {prop_sums.max():.6f})")
        
        # Check fixed component parts are constant
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                col_values = parts_design[:, i]
                expected_parts = self.fixed_parts[name]
                parts_ok = np.allclose(col_values, expected_parts, atol=1e-6)
                print(f"  {name} parts constant: {parts_ok} "
                      f"(expected: {expected_parts}, actual: [{col_values.min():.6f}, {col_values.max():.6f}])")
        
        # Check variable component bounds
        for i, name in enumerate(self.component_names):
            if name in self.variable_names:
                col_values = parts_design[:, i]
                min_bound, max_bound = self.variable_bounds[name]
                bounds_ok = np.all((col_values >= min_bound - 1e-6) & (col_values <= max_bound + 1e-6))
                print(f"  {name} bounds [{min_bound:.1f}, {max_bound:.1f}]: {bounds_ok} "
                      f"(actual: [{col_values.min():.6f}, {col_values.max():.6f}])")
        
        # Check batch size range
        print(f"  Batch sizes: {batch_sizes.min():.1f} to {batch_sizes.max():.1f}")
        print(f"  Theoretical range: {self.min_batch_size:.1f} to {self.max_batch_size:.1f}")
        
        # Check space-filling properties
        self._analyze_space_filling(parts_design)
    
    def _analyze_space_filling(self, parts_design: np.ndarray):
        """Analyze space-filling properties of the design."""
        if self.n_variable < 2:
            return
        
        # Extract variable component parts
        var_parts = np.zeros((len(parts_design), self.n_variable))
        for i, name in enumerate(self.component_names):
            if name in self.variable_names:
                var_idx = self.variable_names.index(name)
                var_parts[:, var_idx] = parts_design[:, i]
        
        # Calculate minimum distances between points
        n_points = len(var_parts)
        min_distances = []
        
        for i in range(n_points):
            distances = []
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(var_parts[i] - var_parts[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        if min_distances:
            min_min_dist = min(min_distances)
            avg_min_dist = np.mean(min_distances)
            
            print(f"  Space-filling metrics:")
            print(f"    Minimum distance between points: {min_min_dist:.3f}")
            print(f"    Average minimum distance: {avg_min_dist:.3f}")
            
            # Check for clustered points
            clustered_threshold = 1.0  # Adjust based on scale
            clustered_pairs = sum(1 for d in min_distances if d < clustered_threshold)
            print(f"    Points with distance < {clustered_threshold}: {clustered_pairs}/{len(min_distances)}")
    
    def create_results_dataframe(self, parts_design: np.ndarray, 
                                proportions_design: np.ndarray, 
                                batch_sizes: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame."""
        # Parts DataFrame
        df_parts = pd.DataFrame(parts_design, columns=[f"{name}_Parts" for name in self.component_names])
        
        # Proportions DataFrame
        df_props = pd.DataFrame(proportions_design, columns=[f"{name}_Prop" for name in self.component_names])
        
        # Combine
        df = pd.concat([df_parts, df_props], axis=1)
        df['Batch_Size'] = batch_sizes
        df['Total_Parts'] = parts_design.sum(axis=1)
        df['Total_Props'] = proportions_design.sum(axis=1)
        
        df.index = [f"Run_{i+1}" for i in range(len(parts_design))]
        
        # Add verification columns
        for name in self.fixed_names:
            expected = self.fixed_parts[name]
            actual = parts_design[:, self.component_names.index(name)]
            df[f'{name}_FixedOK'] = np.abs(actual - expected) < 1e-6
        
        return df


# Example usage and testing
if __name__ == "__main__":
    # Test the improved implementation
    component_names = ["Base_Polymer", "Catalyst", "Solvent", "Additive"]
    fixed_parts = {"Base_Polymer": 50.0, "Catalyst": 2.5}
    variable_bounds = {"Solvent": (0.0, 40.0), "Additive": (0.0, 15.0)}
    
    print("=== Testing Improved Fixed Parts Mixture Design ===")
    
    # Create improved designer
    improved_designer = ImprovedFixedPartsMixture(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate improved design
    parts_design, prop_design, batch_sizes = improved_designer.generate_d_optimal_design(
        n_runs=15,
        model_type="quadratic",
        random_seed=42
    )
    
    # Create results dataframe
    results_df = improved_designer.create_results_dataframe(parts_design, prop_design, batch_sizes)
    print("\nImproved Design Results:")
    
    # Show key columns
    key_cols = ([f"{name}_Parts" for name in component_names] + 
                [f"{name}_Prop" for name in component_names] + 
                ['Batch_Size'])
    print(results_df[key_cols].round(3))
