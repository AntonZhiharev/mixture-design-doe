"""
Correct Fixed Components Mixture Design Implementation

This module provides the theoretically correct implementation of mixture designs 
with fixed components. Fixed components are components that:

1. MUST be present in every experimental run
2. Have VARIABLE proportions (within specified bounds) 
3. REDUCE the available design space for other components
4. Create a CONSTRAINED SIMPLEX for remaining components

Example: 4 components (A,B,C,D), A is fixed [0.1, 0.3]
- A will vary between 10-30% in every run
- B,C,D share the remaining 70-90% space
- This creates a much smaller, constrained design space
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from itertools import combinations_with_replacement
import warnings

class CorrectFixedComponentsMixture:
    """
    Theoretically correct implementation of mixture designs with fixed components.
    
    Fixed components reduce the available design space by creating constraints
    on the simplex where other components must operate.
    """
    
    def __init__(self, 
                 component_names: List[str],
                 fixed_components: Dict[str, Tuple[float, float]] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None,
                 total_mixture: float = 1.0):
        """
        Initialize the fixed components mixture design.
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components in the mixture
        fixed_components : Dict[str, Tuple[float, float]], optional
            Dictionary mapping fixed component names to their (min_prop, max_prop) bounds
            These components will be present in every run with varying proportions
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable (non-fixed) components
        total_mixture : float, default=1.0
            Total sum that all components must equal
        """
        self.component_names = component_names
        self.n_components = len(component_names)
        self.fixed_components = fixed_components or {}
        self.variable_bounds = variable_bounds or {}
        self.total_mixture = total_mixture
        
        # Identify fixed vs variable components
        self.fixed_names = list(self.fixed_components.keys())
        self.variable_names = [name for name in component_names if name not in self.fixed_names]
        self.n_fixed = len(self.fixed_names)
        self.n_variable = len(self.variable_names)
        
        # Validate the setup
        self._validate_setup()
        
        # Calculate the constrained design space
        self._calculate_constrained_space()
        
        print(f"\nFixed Components Mixture Design Setup:")
        print(f"  Total components: {self.n_components}")
        print(f"  Fixed components: {self.n_fixed} {self.fixed_names}")
        print(f"  Variable components: {self.n_variable} {self.variable_names}")
        print(f"  Available space for variables: {self.min_available_space:.3f} to {self.max_available_space:.3f}")
        print(f"  Space reduction: {(1.0 - self.max_available_space)*100:.1f}% of original space removed")
    
    def _validate_setup(self):
        """Validate that the fixed components setup is feasible."""
        if not self.fixed_components:
            raise ValueError("No fixed components specified. Use regular mixture design instead.")
        
        # Check if fixed component bounds are feasible
        min_fixed_total = sum(bounds[0] for bounds in self.fixed_components.values())
        max_fixed_total = sum(bounds[1] for bounds in self.fixed_components.values())
        
        if min_fixed_total >= self.total_mixture:
            raise ValueError(f"Fixed components minimum total ({min_fixed_total:.3f}) >= mixture total ({self.total_mixture})")
        
        if max_fixed_total >= self.total_mixture:
            print(f"Warning: Fixed components maximum total ({max_fixed_total:.3f}) >= mixture total. "
                  "Some combinations may be infeasible.")
        
        # Check for component name conflicts
        unknown_fixed = set(self.fixed_names) - set(self.component_names)
        if unknown_fixed:
            raise ValueError(f"Unknown fixed components: {unknown_fixed}")
    
    def _calculate_constrained_space(self):
        """Calculate the available space for variable components."""
        # Minimum space available (when fixed components are at maximum)
        max_fixed_total = sum(bounds[1] for bounds in self.fixed_components.values())
        self.min_available_space = max(0, self.total_mixture - max_fixed_total)
        
        # Maximum space available (when fixed components are at minimum)
        min_fixed_total = sum(bounds[0] for bounds in self.fixed_components.values())
        self.max_available_space = self.total_mixture - min_fixed_total
        
        # Store individual fixed bounds for easier access
        self.fixed_bounds_array = np.array([self.fixed_components[name] for name in self.fixed_names])
    
    def generate_random_feasible_point(self) -> np.ndarray:
        """
        Generate a single random feasible point respecting fixed component constraints.
        
        Returns:
        --------
        np.ndarray: A feasible mixture point with fixed components properly constrained
        """
        point = np.zeros(self.n_components)
        
        # Step 1: Generate fixed component values
        fixed_values = np.zeros(self.n_fixed)
        for i, (min_val, max_val) in enumerate(self.fixed_bounds_array):
            fixed_values[i] = np.random.uniform(min_val, max_val)
        
        # Check if this combination is feasible
        fixed_total = np.sum(fixed_values)
        if fixed_total >= self.total_mixture:
            # Retry with scaled-down fixed values
            scale_factor = (self.total_mixture * 0.95) / fixed_total
            fixed_values = fixed_values * scale_factor
            fixed_total = np.sum(fixed_values)
        
        # Step 2: Calculate remaining space for variable components
        remaining_space = self.total_mixture - fixed_total
        
        # Step 3: Generate variable component values in the constrained simplex
        if self.n_variable > 0:
            # Generate random proportions for variable components
            if self.n_variable == 1:
                variable_values = np.array([remaining_space])
            else:
                # Use Dirichlet distribution to generate points in simplex
                alphas = np.ones(self.n_variable)
                variable_proportions = np.random.dirichlet(alphas)
                variable_values = variable_proportions * remaining_space
        else:
            variable_values = np.array([])
        
        # Step 4: Combine fixed and variable components
        fixed_idx = 0
        variable_idx = 0
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                point[i] = fixed_values[fixed_idx]
                fixed_idx += 1
            else:
                point[i] = variable_values[variable_idx]
                variable_idx += 1
        
        return point
    
    def generate_candidate_set(self, n_candidates: int = 1000) -> np.ndarray:
        """
        Generate a set of feasible candidate points for optimal design algorithms.
        
        Parameters:
        -----------
        n_candidates : int
            Number of candidate points to generate
            
        Returns:
        --------
        np.ndarray: Matrix of candidate points (n_candidates x n_components)
        """
        candidates = np.zeros((n_candidates, self.n_components))
        
        print(f"Generating {n_candidates} candidate points for constrained mixture space...")
        
        for i in range(n_candidates):
            candidates[i] = self.generate_random_feasible_point()
        
        # Verify feasibility
        row_sums = np.sum(candidates, axis=1)
        if not np.allclose(row_sums, self.total_mixture, atol=1e-10):
            print(f"Warning: Some candidates have incorrect sums. Range: {row_sums.min():.6f} to {row_sums.max():.6f}")
        
        return candidates
    
    def generate_structured_points(self) -> np.ndarray:
        """
        Generate structured design points (vertices, edge centers, centroids) for the
        constrained mixture space.
        
        Returns:
        --------
        np.ndarray: Matrix of structured points
        """
        structured_points = []
        
        # 1. Generate extreme points where fixed components are at their bounds
        print("Generating structured points for constrained mixture space...")
        
        # Fixed at minimum, variable components get maximum space
        fixed_at_min = np.array([bounds[0] for bounds in self.fixed_bounds_array])
        if np.sum(fixed_at_min) < self.total_mixture:
            remaining = self.total_mixture - np.sum(fixed_at_min)
            
            # Generate vertices for variable components
            if self.n_variable > 0:
                for i in range(self.n_variable):
                    point = np.zeros(self.n_components)
                    
                    # Set fixed components to minimum
                    fixed_idx = 0
                    for j, name in enumerate(self.component_names):
                        if name in self.fixed_names:
                            point[j] = fixed_at_min[fixed_idx]
                            fixed_idx += 1
                    
                    # Set one variable component to maximum, others to minimum
                    variable_idx = 0
                    for j, name in enumerate(self.component_names):
                        if name not in self.fixed_names:
                            if variable_idx == i:
                                point[j] = remaining
                            else:
                                point[j] = 0.0
                            variable_idx += 1
                    
                    structured_points.append(point)
        
        # 2. Fixed at maximum, variable components get minimum space
        fixed_at_max = np.array([bounds[1] for bounds in self.fixed_bounds_array])
        if np.sum(fixed_at_max) < self.total_mixture:
            remaining = self.total_mixture - np.sum(fixed_at_max)
            
            if self.n_variable > 0:
                # Equal distribution among variable components
                point = np.zeros(self.n_components)
                
                # Set fixed components to maximum
                fixed_idx = 0
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_names:
                        point[j] = fixed_at_max[fixed_idx]
                        fixed_idx += 1
                
                # Distribute remaining equally among variable components
                for j, name in enumerate(self.component_names):
                    if name not in self.fixed_names:
                        point[j] = remaining / self.n_variable
                
                structured_points.append(point)
        
        # 3. Centroid of constrained space
        if len(structured_points) > 0:
            centroid = np.mean(structured_points, axis=0)
            # Ensure centroid sums to 1
            centroid = centroid / np.sum(centroid) * self.total_mixture
            structured_points.append(centroid)
        
        if not structured_points:
            print("Warning: No feasible structured points found. Using random points.")
            return self.generate_candidate_set(10)
        
        return np.array(structured_points)
    
    def generate_d_optimal_design(self, n_runs: int, model_type: str = "quadratic", 
                                 max_iter: int = 1000, n_candidates: int = 2000,
                                 random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate D-optimal design for the constrained mixture space.
        
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
        np.ndarray: D-optimal design matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"\nGenerating D-optimal design for fixed components mixture:")
        print(f"  Runs: {n_runs}")
        print(f"  Model: {model_type}")
        print(f"  Fixed components: {self.fixed_names}")
        print(f"  Available space reduction: {(1.0 - self.max_available_space)*100:.1f}%")
        
        # Generate candidate points in the constrained space
        candidates = self.generate_candidate_set(n_candidates)
        
        # Add structured points
        structured = self.generate_structured_points()
        all_candidates = np.vstack([structured, candidates])
        
        print(f"  Total candidates: {len(all_candidates)} (including {len(structured)} structured points)")
        
        # Create model matrix for the candidates
        X_candidates = self._create_model_matrix(all_candidates, model_type)
        
        # Run coordinate exchange algorithm
        design_indices = self._coordinate_exchange_d_optimal(
            X_candidates, n_runs, max_iter
        )
        
        # Extract the selected design points
        design = all_candidates[design_indices]
        
        # Verify the design
        self._verify_design(design)
        
        return design
    
    def _create_model_matrix(self, design_points: np.ndarray, model_type: str) -> np.ndarray:
        """Create the model matrix X for mixture model."""
        n_points = len(design_points)
        
        if model_type == "linear":
            # For mixture: just the component proportions (no intercept)
            X = design_points.copy()
        
        elif model_type == "quadratic":
            # Components + interaction terms
            n_terms = self.n_components + (self.n_components * (self.n_components - 1)) // 2
            X = np.zeros((n_points, n_terms))
            
            # Linear terms
            X[:, :self.n_components] = design_points
            
            # Interaction terms
            col_idx = self.n_components
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    X[:, col_idx] = design_points[:, i] * design_points[:, j]
                    col_idx += 1
        
        elif model_type == "cubic":
            # Components + interactions + cubic terms
            n_linear = self.n_components
            n_quadratic = (self.n_components * (self.n_components - 1)) // 2
            n_cubic = (self.n_components * (self.n_components - 1) * (self.n_components - 2)) // 6
            n_terms = n_linear + n_quadratic + n_cubic
            
            X = np.zeros((n_points, n_terms))
            
            # Linear terms
            X[:, :n_linear] = design_points
            
            # Quadratic interaction terms
            col_idx = n_linear
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    X[:, col_idx] = design_points[:, i] * design_points[:, j]
                    col_idx += 1
            
            # Cubic interaction terms
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    for k in range(j + 1, self.n_components):
                        X[:, col_idx] = (design_points[:, i] * 
                                       design_points[:, j] * 
                                       design_points[:, k])
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
    
    def _verify_design(self, design: np.ndarray):
        """Verify that the generated design meets all constraints."""
        print(f"\nDesign Verification:")
        
        # Check row sums
        row_sums = np.sum(design, axis=1)
        sum_ok = np.allclose(row_sums, self.total_mixture, atol=1e-6)
        print(f"  Row sums correct: {sum_ok} (range: {row_sums.min():.6f} to {row_sums.max():.6f})")
        
        # Check fixed component bounds
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                col_values = design[:, i]
                min_bound, max_bound = self.fixed_components[name]
                bounds_ok = np.all((col_values >= min_bound - 1e-6) & (col_values <= max_bound + 1e-6))
                print(f"  {name} bounds [{min_bound:.3f}, {max_bound:.3f}]: {bounds_ok} "
                      f"(actual: [{col_values.min():.6f}, {col_values.max():.6f}])")
        
        # Calculate space utilization
        fixed_totals = np.sum(design[:, [self.component_names.index(name) for name in self.fixed_names]], axis=1)
        variable_space = self.total_mixture - fixed_totals
        print(f"  Variable space utilization: {variable_space.min():.3f} to {variable_space.max():.3f}")
        print(f"  Theoretical range: {self.min_available_space:.3f} to {self.max_available_space:.3f}")
    
    def create_results_dataframe(self, design: np.ndarray) -> pd.DataFrame:
        """Create a comprehensive results DataFrame."""
        df = pd.DataFrame(design, columns=self.component_names)
        df.index = [f"Run_{i+1}" for i in range(len(design))]
        
        # Add verification columns
        df['Total'] = df.sum(axis=1)
        
        # Add fixed component indicators
        for name in self.fixed_names:
            min_bound, max_bound = self.fixed_components[name]
            df[f'{name}_InBounds'] = ((df[name] >= min_bound - 1e-6) & 
                                     (df[name] <= max_bound + 1e-6))
        
        # Add space utilization
        fixed_cols = [name for name in self.component_names if name in self.fixed_names]
        if fixed_cols:
            df['Fixed_Total'] = df[fixed_cols].sum(axis=1)
            df['Variable_Space'] = 1.0 - df['Fixed_Total']
        
        return df


# Example usage and demonstration
if __name__ == "__main__":
    # Example: 4-component mixture with 2 fixed components
    component_names = ["Polymer_A", "Polymer_B", "Solvent", "Additive"]
    
    # Polymer_A and Solvent are fixed components (must always be present)
    fixed_components = {
        "Polymer_A": (0.10, 0.30),    # 10-30%
        "Solvent": (0.20, 0.40)       # 20-40%
    }
    
    print("=== Fixed Components Mixture Design Example ===")
    print("Components:", component_names)
    print("Fixed components:", fixed_components)
    print("This reduces available design space significantly!")
    
    # Create the design generator
    designer = CorrectFixedComponentsMixture(
        component_names=component_names,
        fixed_components=fixed_components
    )
    
    # Generate a D-optimal design
    design = designer.generate_d_optimal_design(
        n_runs=12, 
        model_type="quadratic",
        random_seed=42
    )
    
    # Create results dataframe
    results_df = designer.create_results_dataframe(design)
    print("\nGenerated Design:")
    print(results_df.round(4))
