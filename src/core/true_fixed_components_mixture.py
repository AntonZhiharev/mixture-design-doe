"""
True Fixed Components Mixture Design Implementation

CORRECT UNDERSTANDING of Fixed Components:
- Fixed components have CONSTANT amounts in PARTS (e.g., always 10 parts)
- Variable components have VARIABLE amounts in PARTS (e.g., 0-20 parts)  
- Fixed components have VARIABLE PROPORTIONS (because total batch changes)
- They reduce design space by consuming fixed material amounts

Example:
- Component A (fixed): 10 parts (constant)
- Component B (variable): 0-20 parts
- Component C (variable): 0-15 parts

Run 1: A=10, B=5,  C=10 → Total=25 → Proportions: A=40.0%, B=20.0%, C=40.0%
Run 2: A=10, B=15, C=5  → Total=30 → Proportions: A=33.3%, B=50.0%, C=16.7%

The fixed component A always uses 10 parts but its proportion varies!
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import warnings

class TrueFixedComponentsMixture:
    """
    Correct implementation of mixture designs with fixed components.
    
    Fixed components consume a constant amount of material (in parts),
    which reduces the available design space for variable components.
    """
    
    def __init__(self, 
                 component_names: List[str],
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize the true fixed components mixture design.
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components in the mixture
        fixed_parts : Dict[str, float], optional
            Dictionary mapping fixed component names to their constant parts amount
            Example: {"Polymer_A": 10.0, "Catalyst": 2.5}
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable components in parts
            Example: {"Solvent": (0, 50), "Additive": (0, 20)}
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
                self.variable_bounds[name] = (0.0, 100.0)  # Default: 0-100 parts
        
        # Validate the setup
        self._validate_setup()
        
        # Calculate design space constraints
        self._calculate_design_space()
        
        print(f"\nTrue Fixed Components Mixture Design Setup:")
        print(f"  Total components: {self.n_components}")
        print(f"  Fixed components: {self.n_fixed} {self.fixed_names}")
        print(f"  Variable components: {self.n_variable} {self.variable_names}")
        print(f"  Total fixed parts: {self.total_fixed_parts}")
        print(f"  Minimum batch size: {self.min_batch_size}")
        print(f"  Maximum batch size: {self.max_batch_size}")
        print(f"  Variable parts range: {self.min_variable_parts} to {self.max_variable_parts}")
    
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
        
        # Calculate effective design space reduction
        if self.max_variable_parts > 0:
            self.space_reduction = self.total_fixed_parts / (self.total_fixed_parts + self.max_variable_parts)
        else:
            self.space_reduction = 1.0
    
    def generate_random_feasible_point(self) -> Tuple[np.ndarray, float]:
        """
        Generate a single random feasible point in parts and its total batch size.
        
        Returns:
        --------
        Tuple[np.ndarray, float]: (parts_vector, total_batch_size)
        """
        parts = np.zeros(self.n_components)
        
        # Step 1: Set fixed component parts (constant)
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                parts[i] = self.fixed_parts[name]
        
        # Step 2: Generate variable component parts
        for i, name in enumerate(self.component_names):
            if name in self.variable_names:
                min_parts, max_parts = self.variable_bounds[name]
                parts[i] = np.random.uniform(min_parts, max_parts)
        
        total_batch = np.sum(parts)
        
        return parts, total_batch
    
    def parts_to_proportions(self, parts: np.ndarray) -> np.ndarray:
        """Convert parts to proportions."""
        total = np.sum(parts)
        if total == 0:
            return np.zeros_like(parts)
        return parts / total
    
    def proportions_to_parts(self, proportions: np.ndarray, total_batch: float) -> np.ndarray:
        """Convert proportions to parts given a total batch size."""
        return proportions * total_batch
    
    def generate_candidate_set(self, n_candidates: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a set of feasible candidate points for optimal design algorithms.
        
        Parameters:
        -----------
        n_candidates : int
            Number of candidate points to generate
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (parts_matrix, proportions_matrix, batch_sizes)
        """
        parts_candidates = np.zeros((n_candidates, self.n_components))
        batch_sizes = np.zeros(n_candidates)
        
        print(f"Generating {n_candidates} candidate points for fixed components mixture...")
        
        for i in range(n_candidates):
            parts, batch_size = self.generate_random_feasible_point()
            parts_candidates[i] = parts
            batch_sizes[i] = batch_size
        
        # Convert to proportions for design optimization
        proportions_candidates = np.zeros_like(parts_candidates)
        for i in range(n_candidates):
            proportions_candidates[i] = self.parts_to_proportions(parts_candidates[i])
        
        return parts_candidates, proportions_candidates, batch_sizes
    
    def generate_structured_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate structured design points (extreme points, centroid) for the design space.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (parts_matrix, proportions_matrix, batch_sizes)
        """
        structured_parts = []
        
        print("Generating structured points for fixed components mixture...")
        
        # 1. Minimum batch size points (all variable components at minimum)
        parts = np.zeros(self.n_components)
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                parts[i] = self.fixed_parts[name]
            else:
                parts[i] = self.variable_bounds[name][0]
        structured_parts.append(parts.copy())
        
        # 2. Maximum batch size points (all variable components at maximum)
        parts = np.zeros(self.n_components)
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                parts[i] = self.fixed_parts[name]
            else:
                parts[i] = self.variable_bounds[name][1]
        structured_parts.append(parts.copy())
        
        # 3. Extreme points for each variable component
        for var_name in self.variable_names:
            # Variable component at maximum, others at minimum
            parts = np.zeros(self.n_components)
            for i, name in enumerate(self.component_names):
                if name in self.fixed_names:
                    parts[i] = self.fixed_parts[name]
                elif name == var_name:
                    parts[i] = self.variable_bounds[name][1]
                else:
                    parts[i] = self.variable_bounds[name][0]
            structured_parts.append(parts.copy())
        
        # 4. Centroid of variable space
        parts = np.zeros(self.n_components)
        for i, name in enumerate(self.component_names):
            if name in self.fixed_names:
                parts[i] = self.fixed_parts[name]
            else:
                min_parts, max_parts = self.variable_bounds[name]
                parts[i] = (min_parts + max_parts) / 2
        structured_parts.append(parts.copy())
        
        # Convert to arrays
        parts_matrix = np.array(structured_parts)
        batch_sizes = np.sum(parts_matrix, axis=1)
        
        # Convert to proportions
        proportions_matrix = np.zeros_like(parts_matrix)
        for i in range(len(parts_matrix)):
            proportions_matrix[i] = self.parts_to_proportions(parts_matrix[i])
        
        return parts_matrix, proportions_matrix, batch_sizes
    
    def generate_d_optimal_design(self, n_runs: int, model_type: str = "quadratic", 
                                 max_iter: int = 1000, n_candidates: int = 2000,
                                 random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate D-optimal design for the fixed components mixture.
        
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
        
        print(f"\nGenerating D-optimal design for fixed components mixture:")
        print(f"  Runs: {n_runs}")
        print(f"  Model: {model_type}")
        print(f"  Fixed components: {self.fixed_names}")
        print(f"  Fixed parts consumption: {self.total_fixed_parts}")
        print(f"  Space reduction: {self.space_reduction:.1%}")
        
        # Generate candidate points
        parts_candidates, prop_candidates, batch_candidates = self.generate_candidate_set(n_candidates)
        
        # Add structured points
        struct_parts, struct_props, struct_batches = self.generate_structured_points()
        all_parts = np.vstack([struct_parts, parts_candidates])
        all_props = np.vstack([struct_props, prop_candidates])
        all_batches = np.concatenate([struct_batches, batch_candidates])
        
        print(f"  Total candidates: {len(all_props)} (including {len(struct_props)} structured points)")
        
        # Create model matrix for the candidates (use proportions for statistical modeling)
        X_candidates = self._create_model_matrix(all_props, model_type)
        
        # Run coordinate exchange algorithm
        design_indices = self._coordinate_exchange_d_optimal(
            X_candidates, n_runs, max_iter
        )
        
        # Extract the selected design points
        parts_design = all_parts[design_indices]
        proportions_design = all_props[design_indices]
        batch_sizes = all_batches[design_indices]
        
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
        print(f"\nDesign Verification:")
        
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


# Example usage and demonstration
if __name__ == "__main__":
    # Example: 4-component mixture with 2 fixed components
    component_names = ["Polymer_Base", "Hardener", "Solvent", "Additive"]
    
    # Fixed components (constant parts)
    fixed_parts = {
        "Polymer_Base": 25.0,  # Always 25 parts
        "Hardener": 5.0        # Always 5 parts
    }
    
    # Variable components bounds (in parts)
    variable_bounds = {
        "Solvent": (0.0, 30.0),    # 0-30 parts
        "Additive": (0.0, 15.0)    # 0-15 parts
    }
    
    print("=== True Fixed Components Mixture Design Example ===")
    print("Components:", component_names)
    print("Fixed parts:", fixed_parts)
    print("Variable bounds:", variable_bounds)
    print("Fixed components consume constant material, reducing design space!")
    
    # Create the design generator
    designer = TrueFixedComponentsMixture(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate a D-optimal design
    parts_design, prop_design, batch_sizes = designer.generate_d_optimal_design(
        n_runs=10, 
        model_type="quadratic",
        random_seed=42
    )
    
    # Create results dataframe
    results_df = designer.create_results_dataframe(parts_design, prop_design, batch_sizes)
    print("\nGenerated Design (Parts and Proportions):")
    
    # Show key columns
    key_cols = ([f"{name}_Parts" for name in component_names] + 
                [f"{name}_Prop" for name in component_names] + 
                ['Batch_Size'])
    print(results_df[key_cols].round(3))
