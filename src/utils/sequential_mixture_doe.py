"""
Sequential Design of Experiments for Mixture Experiments
Handles mixture constraints where components must sum to 1
"""

import numpy as np
import pandas as pd
from mixture_designs import MixtureDesign
from typing import List, Tuple, Optional, Dict
import json
from datetime import datetime

class SequentialMixtureDOE(MixtureDesign):
    """
    Extended Mixture DOE class for sequential experimentation
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None,
                 fixed_components: Dict[str, float] = None,
                 use_parts_mode: bool = False):
        """
        Initialize sequential mixture DOE generator
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        component_names : List[str]
            Names of components
        component_bounds : List[Tuple[float, float]]
            Min/max bounds for each component
            - If use_parts_mode=False: bounds must be between 0 and 1 (proportions)
            - If use_parts_mode=True: bounds are in parts (any positive values)
        fixed_components : Dict[str, float]
            Components with fixed values
            - If use_parts_mode=False: values are proportions (must sum to < 1)
            - If use_parts_mode=True: values are parts (will be normalized)
        use_parts_mode : bool
            If True, work with parts that get normalized to proportions
        """
        # Store original bounds and fixed values if in parts mode
        self.use_parts_mode = use_parts_mode
        self.original_bounds = []
        self.original_fixed = {}
        
        if use_parts_mode:
            # Store original parts values
            self.original_bounds = component_bounds.copy() if component_bounds else []
            self.original_fixed = fixed_components.copy() if fixed_components else {}
            
            # For parent class, we'll use proportion bounds (0-1)
            if component_bounds:
                component_bounds = [(0.0, 1.0) for _ in component_bounds]
        
        # Initialize parent class
        super().__init__(n_components, component_names, component_bounds)
        self.experiment_history = []
        self.fixed_components = fixed_components or {}
        
        # Validate fixed components for proportion mode
        if not use_parts_mode and self.fixed_components:
            self._validate_fixed_components()
    
    def _validate_fixed_components(self):
        """Validate that fixed components are feasible"""
        fixed_sum = sum(self.fixed_components.values())
        if fixed_sum >= 1.0:
            raise ValueError(f"Fixed components sum to {fixed_sum}, leaving no room for variable components")
        
        # Check that fixed component names are valid
        for comp_name in self.fixed_components:
            if comp_name not in self.component_names:
                raise ValueError(f"Fixed component '{comp_name}' not in component names")
    
    def generate_d_optimal_mixture(self, n_runs: int, model_type: str = "quadratic", 
                                   random_seed: int = None) -> np.ndarray:
        """
        Generate D-optimal mixture design with proper parts handling
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        model_type : str
            Type of model to fit ("linear", "quadratic", "cubic")
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Design matrix (n_runs x n_components) with proportions
        """
        # Generate base design using parent method
        design = super().generate_d_optimal_mixture(n_runs, model_type, random_seed)
        
        # If in parts mode, we need to handle the conversion properly
        if self.use_parts_mode and self.original_bounds:
            # Convert design to parts space for optimization
            design_parts = self._proportions_to_parts_design(design)
            
            # Apply bounds in parts space
            for i in range(self.n_components):
                min_parts, max_parts = self.original_bounds[i]
                design_parts[:, i] = np.clip(design_parts[:, i], min_parts, max_parts)
            
            # Convert back to proportions
            design = self._parts_to_proportions_design(design_parts)
        
        # Apply fixed components
        design = self._adjust_for_fixed_components(design)
        
        return design
    
    def _proportions_to_parts_design(self, design_proportions: np.ndarray) -> np.ndarray:
        """
        Convert a design from proportions to parts based on bounds
        
        Parameters:
        -----------
        design_proportions : np.ndarray
            Design matrix with proportions
            
        Returns:
        --------
        np.ndarray
            Design matrix in parts space
        """
        design_parts = np.zeros_like(design_proportions)
        
        # Scale each component based on its bounds range
        for i in range(self.n_components):
            min_parts, max_parts = self.original_bounds[i]
            # Linear scaling from [0,1] to [min_parts, max_parts]
            design_parts[:, i] = design_proportions[:, i] * (max_parts - min_parts) + min_parts
        
        return design_parts
    
    def _parts_to_proportions_design(self, design_parts: np.ndarray) -> np.ndarray:
        """
        Convert a design matrix from parts to proportions
        
        Parameters:
        -----------
        design_parts : np.ndarray
            Design matrix with values in parts
            
        Returns:
        --------
        np.ndarray
            Design matrix with values as proportions (each row sums to 1)
        """
        # Calculate row sums
        row_sums = design_parts.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        # Normalize each row
        return design_parts / row_sums[:, np.newaxis]
    
    def parts_to_proportions(self, parts_dict: Dict[str, float], 
                           variable_parts: Dict[str, float] = None) -> Dict[str, float]:
        """
        Convert parts to proportions
        
        Parameters:
        -----------
        parts_dict : Dict[str, float]
            Component names and their parts
        variable_parts : Dict[str, float]
            Additional variable components and their parts
            
        Returns:
        --------
        Dict[str, float]
            Component names and their proportions (sum to 1)
        """
        all_parts = parts_dict.copy()
        if variable_parts:
            all_parts.update(variable_parts)
        
        total = sum(all_parts.values())
        if total == 0:
            raise ValueError("Total parts cannot be zero")
            
        return {name: value/total for name, value in all_parts.items()}
    
    def calculate_batch_quantities(self, design_proportions: np.ndarray, 
                                 batch_size: float = 100.0) -> pd.DataFrame:
        """
        Calculate actual quantities for a given batch size
        
        Parameters:
        -----------
        design_proportions : np.ndarray
            Design matrix with proportions (each row sums to 1)
        batch_size : float
            Total batch size (in whatever units you're using)
            
        Returns:
        --------
        pd.DataFrame
            Quantities for each component in each mixture
        """
        quantities = design_proportions * batch_size
        df = pd.DataFrame(quantities, columns=self.component_names)
        df.insert(0, 'Mixture', range(1, len(df) + 1))
        df['Total'] = df[self.component_names].sum(axis=1)
        return df
    
    def _adjust_for_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """
        Adjust design matrix to account for fixed components using fixed space solution
        This is now primarily handled by the parent class post-processing which implements
        the exact algorithm from test_fixed_space_solution.py
        """
        if not self.fixed_components:
            return design
        
        # Apply the parent class post-processing method which implements 
        # the fixed space solution from test_fixed_space_solution.py:
        # - Step 6: Replace fixed components with original values 
        # - Step 7: Final normalization to ensure sum = 1.0
        return self._post_process_design_fixed_components(design)
    
    def augment_mixture_design(self, existing_design: np.ndarray, 
                              n_additional_runs: int, 
                              model_type: str = "quadratic",
                              focus_components: List[str] = None,
                              avoid_region: Optional[np.ndarray] = None,
                              random_seed: int = None) -> np.ndarray:
        """
        Augment an existing mixture design with additional experiments
        
        Parameters:
        -----------
        existing_design : np.ndarray
            Current mixture design matrix (n_existing_runs x n_components)
        n_additional_runs : int
            Number of new mixture experiments to add
        model_type : str
            "linear", "quadratic", or "cubic"
        focus_components : List[str]
            Components to focus on based on screening results
        avoid_region : np.ndarray, optional
            Mixture compositions to avoid
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            New mixture experiments only (n_additional_runs x n_components)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Generate candidate mixtures
        n_candidates = max(1000, n_additional_runs * 50)
        candidates = self._generate_mixture_candidates(n_candidates, focus_components)
        
        # Apply fixed components if any
        candidates = self._adjust_for_fixed_components(candidates)
        
        # Remove points too close to existing design
        candidates = self._filter_mixture_candidates(candidates, existing_design)
        
        # Remove avoided regions
        if avoid_region is not None:
            candidates = self._remove_avoided_mixtures(candidates, avoid_region)
        
        # Select additional points using D-optimal criterion
        augmented_design = existing_design.copy()
        new_points = []
        min_distance = 0.001  # Start with strict distance requirement
        
        for iteration in range(n_additional_runs):
            best_candidate = None
            best_d_eff = -np.inf
            
            # If we're running low on candidates, generate more
            if len(candidates) < 50:
                additional_candidates = self._generate_mixture_candidates(
                    n_candidates * 2, focus_components
                )
                additional_candidates = self._adjust_for_fixed_components(additional_candidates)
                additional_candidates = self._filter_mixture_candidates(
                    additional_candidates, existing_design, min_distance
                )
                if avoid_region is not None:
                    additional_candidates = self._remove_avoided_mixtures(
                        additional_candidates, avoid_region
                    )
                candidates = np.vstack([candidates, additional_candidates]) if len(candidates) > 0 else additional_candidates
            
            # Test candidates for best D-efficiency
            n_test = min(len(candidates), 200)
            test_candidates = candidates[:n_test] if len(candidates) > n_test else candidates
            
            for candidate in test_candidates:
                # Create temporary augmented design
                temp_design = np.vstack([augmented_design, 
                                       np.array(new_points + [candidate])])
                
                # Calculate D-efficiency
                d_eff = self._calculate_mixture_d_efficiency(temp_design, model_type)
                
                if d_eff > best_d_eff:
                    best_d_eff = d_eff
                    best_candidate = candidate
            
            if best_candidate is not None:
                new_points.append(best_candidate)
                # Remove selected point and nearby points
                distances = np.sum((candidates - best_candidate)**2, axis=1)
                candidates = candidates[distances > min_distance]
            else:
                # If no candidate found, try relaxing the distance constraint
                if min_distance > 0.0001:
                    min_distance *= 0.5  # Relax distance requirement
                    print(f"Relaxing distance constraint to {min_distance:.6f}")
                    # Re-filter candidates with new distance
                    candidates = self._generate_mixture_candidates(n_candidates, focus_components)
                    candidates = self._adjust_for_fixed_components(candidates)
                    candidates = self._filter_mixture_candidates(
                        candidates, np.vstack([existing_design] + new_points) if new_points else existing_design, 
                        min_distance
                    )
                    if avoid_region is not None:
                        candidates = self._remove_avoided_mixtures(candidates, avoid_region)
                    continue  # Try again with relaxed constraints
                else:
                    # Last resort: generate a random valid mixture
                    print(f"Warning: Using random generation for point {len(new_points) + 1}")
                    random_candidate = self._generate_random_valid_mixture(
                        existing_design, new_points, focus_components
                    )
                    if random_candidate is not None:
                        new_points.append(random_candidate)
                    else:
                        print(f"Could not generate point {len(new_points) + 1}, stopping at {len(new_points)} points")
                        break
        
        return np.array(new_points) if new_points else np.array([]).reshape(0, self.n_components)
    
    def generate_simplex_lattice_mixture(self, degree: int = 3, random_seed: int = None) -> np.ndarray:
        """
        Generate Simplex Lattice mixture design
        
        Parameters:
        -----------
        degree : int
            Lattice degree (higher = more points)
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Design matrix with mixture proportions
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Use parent class method if available
        if hasattr(super(), 'generate_simplex_lattice'):
            design = super().generate_simplex_lattice(degree)
        else:
            # Implement simplex lattice for variable components
            # Generate lattice points for variable components only
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in self.fixed_components:
                    variable_indices.append(i)
            
            n_variable = len(variable_indices)
            if n_variable < 2:
                raise ValueError("Need at least 2 variable components for lattice design")
            
            # Generate simplex lattice points
            from itertools import combinations_with_replacement
            
            # Create lattice points
            lattice_points = []
            
            # Generate all combinations of lattice fractions
            for combo in combinations_with_replacement(range(degree + 1), n_variable):
                if sum(combo) == degree:
                    # Convert to proportions
                    mixture = np.array(combo) / degree
                    lattice_points.append(mixture)
            
            lattice_points = np.array(lattice_points)
            
            # Create full mixture design
            design = np.zeros((len(lattice_points), self.n_components))
            for i, var_idx in enumerate(variable_indices):
                design[:, var_idx] = lattice_points[:, i]
        
        # Apply fixed components
        design = self._adjust_for_fixed_components(design)
        
        return design
    
    def generate_simplex_centroid_mixture(self, random_seed: int = None) -> np.ndarray:
        """
        Generate Simplex Centroid mixture design
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Design matrix with mixture proportions
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Use parent class method if available
        if hasattr(super(), 'generate_simplex_centroid'):
            design = super().generate_simplex_centroid()
        else:
            # Implement simplex centroid for variable components
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in self.fixed_components:
                    variable_indices.append(i)
            
            n_variable = len(variable_indices)
            if n_variable < 2:
                raise ValueError("Need at least 2 variable components for centroid design")
            
            from itertools import combinations
            
            # Generate all possible subcombinations and their centroids
            centroid_points = []
            
            # Pure components (vertices)
            for i in range(n_variable):
                mixture = np.zeros(n_variable)
                mixture[i] = 1.0
                centroid_points.append(mixture)
            
            # Binary centroids
            for combo in combinations(range(n_variable), 2):
                mixture = np.zeros(n_variable)
                for idx in combo:
                    mixture[idx] = 1.0 / len(combo)
                centroid_points.append(mixture)
            
            # Ternary centroids (if enough components)
            if n_variable >= 3:
                for combo in combinations(range(n_variable), 3):
                    mixture = np.zeros(n_variable)
                    for idx in combo:
                        mixture[idx] = 1.0 / len(combo)
                    centroid_points.append(mixture)
            
            # Overall centroid (all components equal)
            if n_variable >= 2:
                mixture = np.ones(n_variable) / n_variable
                centroid_points.append(mixture)
            
            centroid_points = np.array(centroid_points)
            
            # Create full mixture design
            design = np.zeros((len(centroid_points), self.n_components))
            for i, var_idx in enumerate(variable_indices):
                design[:, var_idx] = centroid_points[:, i]
        
        # Apply fixed components
        design = self._adjust_for_fixed_components(design)
        
        return design
    
    def generate_extreme_vertices_mixture(self, random_seed: int = None) -> np.ndarray:
        """
        Generate Extreme Vertices mixture design
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        np.ndarray
            Design matrix with mixture proportions
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Use parent class method if available
        if hasattr(super(), 'generate_extreme_vertices'):
            design = super().generate_extreme_vertices()
        else:
            # Implement extreme vertices for variable components
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in self.fixed_components:
                    variable_indices.append(i)
            
            n_variable = len(variable_indices)
            if n_variable < 2:
                raise ValueError("Need at least 2 variable components for extreme vertices")
            
            # Get bounds for variable components
            variable_bounds = []
            for var_idx in variable_indices:
                if self.use_parts_mode:
                    min_parts, max_parts = self.original_bounds[var_idx]
                    variable_bounds.append((min_parts, max_parts))
                else:
                    variable_bounds.append(self.component_bounds[var_idx])
            
            # Generate extreme vertices of the feasible region
            vertices = []
            
            # Simple approach: generate combinations of min/max bounds
            from itertools import product
            
            for bounds_combo in product(*[[(bound[0], 0), (bound[1], 1)] for bound in variable_bounds]):
                # Extract values and indices
                values = [bc[0] for bc in bounds_combo]
                
                if self.use_parts_mode:
                    # Convert parts to proportions
                    total_parts = sum(values)
                    if total_parts > 0:
                        mixture = np.array(values) / total_parts
                    else:
                        continue
                else:
                    # Check if proportions sum to reasonable value
                    total_prop = sum(values)
                    if total_prop <= 0 or total_prop > 1:
                        continue
                    # Normalize to sum to available proportion
                    available = 1.0 - sum(self.fixed_components.values())
                    mixture = np.array(values) / total_prop * available
                
                vertices.append(mixture)
            
            if not vertices:
                # Fallback: use simplex vertices
                vertices = []
                for i in range(n_variable):
                    mixture = np.zeros(n_variable)
                    mixture[i] = 1.0
                    vertices.append(mixture)
            
            vertices = np.array(vertices)
            
            # Remove duplicates
            unique_vertices = []
            for vertex in vertices:
                is_duplicate = False
                for existing in unique_vertices:
                    if np.allclose(vertex, existing, atol=1e-6):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_vertices.append(vertex)
            
            vertices = np.array(unique_vertices) if unique_vertices else vertices
            
            # Create full mixture design
            design = np.zeros((len(vertices), self.n_components))
            for i, var_idx in enumerate(variable_indices):
                design[:, var_idx] = vertices[:, i]
        
        # Apply fixed components
        design = self._adjust_for_fixed_components(design)
        
        return design
    
    def _generate_random_valid_mixture(self, existing_design: np.ndarray,
                                     current_new_points: List, 
                                     focus_components: List[str] = None) -> Optional[np.ndarray]:
        """
        Generate a random valid mixture as a last resort
        
        Parameters:
        -----------
        existing_design : np.ndarray
            Existing design points
        current_new_points : List
            New points already selected
        focus_components : List[str]
            Components to focus on
            
        Returns:
        --------
        Optional[np.ndarray]
            A valid mixture point or None if failed
        """
        max_attempts = 1000
        
        for _ in range(max_attempts):
            if self.use_parts_mode:
                # Generate random parts
                candidate_parts = np.zeros(self.n_components)
                
                for j in range(self.n_components):
                    if self.component_names[j] in self.fixed_components:
                        candidate_parts[j] = self.original_fixed[self.component_names[j]]
                    else:
                        min_parts, max_parts = self.original_bounds[j]
                        candidate_parts[j] = np.random.uniform(min_parts, max_parts)
                
                # Convert to proportions
                candidate = self._parts_to_proportions_design(candidate_parts.reshape(1, -1))[0]
            else:
                # Generate random proportions
                variable_indices = []
                for i, name in enumerate(self.component_names):
                    if name not in self.fixed_components:
                        variable_indices.append(i)
                
                if not variable_indices:
                    return None
                
                # Generate random mixture for variable components
                n_variable = len(variable_indices)
                alpha = np.ones(n_variable)
                
                if focus_components:
                    for i, idx in enumerate(variable_indices):
                        if self.component_names[idx] in focus_components:
                            alpha[i] = 2.0  # Bias towards focus components
                
                variable_mixture = np.random.dirichlet(alpha)
                
                # Create full mixture
                candidate = np.zeros(self.n_components)
                for i, idx in enumerate(variable_indices):
                    candidate[idx] = variable_mixture[i]
                
                # Apply fixed components
                candidate = self._adjust_for_fixed_components(candidate.reshape(1, -1))[0]
            
            # Check if candidate is valid and sufficiently different
            if self._is_valid_mixture_candidate(candidate, existing_design, current_new_points):
                return candidate
        
        return None
    
    def _is_valid_mixture_candidate(self, candidate: np.ndarray, 
                                   existing_design: np.ndarray,
                                   current_new_points: List,
                                   min_distance: float = 0.0001) -> bool:
        """
        Check if a mixture candidate is valid
        
        Parameters:
        -----------
        candidate : np.ndarray
            Candidate mixture point
        existing_design : np.ndarray
            Existing design points
        current_new_points : List
            Current new points
        min_distance : float
            Minimum distance requirement
            
        Returns:
        --------
        bool
            True if candidate is valid
        """
        # Check basic mixture constraints
        if not np.isclose(candidate.sum(), 1.0, atol=1e-6):
            return False
        
        if np.any(candidate < -1e-6):  # Allow small numerical errors
            return False
        
        # Check distance to existing points
        if len(existing_design) > 0:
            distances = np.sqrt(np.sum((existing_design - candidate)**2, axis=1))
            if np.min(distances) < min_distance:
                return False
        
        # Check distance to current new points
        if current_new_points:
            new_points_array = np.array(current_new_points)
            distances = np.sqrt(np.sum((new_points_array - candidate)**2, axis=1))
            if np.min(distances) < min_distance:
                return False
        
        return True
    
    def _generate_mixture_candidates(self, n_candidates: int, 
                                   focus_components: List[str] = None) -> np.ndarray:
        """Generate candidate mixture compositions"""
        if self.use_parts_mode:
            # Generate candidates in parts space
            candidates_parts = np.zeros((n_candidates, self.n_components))
            
            # For each candidate
            for i in range(n_candidates):
                # Randomly assign parts within bounds
                for j in range(self.n_components):
                    if self.component_names[j] in self.fixed_components:
                        # Use fixed value
                        candidates_parts[i, j] = self.original_fixed[self.component_names[j]]
                    else:
                        # Random value within bounds
                        min_parts, max_parts = self.original_bounds[j]
                        candidates_parts[i, j] = np.random.uniform(min_parts, max_parts)
                
                # If focusing on certain components, bias towards them
                if focus_components:
                    for j, name in enumerate(self.component_names):
                        if name in focus_components and name not in self.fixed_components:
                            min_parts, max_parts = self.original_bounds[j]
                            # Bias towards higher values for focus components
                            candidates_parts[i, j] = np.random.uniform((min_parts + max_parts) / 2, max_parts)
            
            # Convert to proportions
            candidates = self._parts_to_proportions_design(candidates_parts)
        else:
            # Original proportion-based generation
            # Get variable components only
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in self.fixed_components:
                    variable_indices.append(i)
            
            n_variable = len(variable_indices)
            
            if n_variable == 0:
                raise ValueError("No variable components to optimize")
            
            # Generate random mixtures for variable components
            candidates = np.zeros((n_candidates, self.n_components))
            
            # Use Dirichlet distribution for mixture generation
            if focus_components:
                # Check if all variable components are focused
                all_focused = all(self.component_names[idx] in focus_components 
                                 for idx in variable_indices)
                
                if all_focused:
                    # If all components are focused, use diverse sampling strategies
                    strategies = ['uniform', 'vertices', 'edges', 'center']
                    variable_mixtures = []
                    
                    for i in range(n_candidates):
                        strategy = strategies[i % len(strategies)]
                        
                        if strategy == 'uniform':
                            # Uniform sampling
                            alpha = np.ones(n_variable)
                            mixture = np.random.dirichlet(alpha, 1)[0]
                        elif strategy == 'vertices':
                            # Near vertices
                            vertex_idx = i % n_variable
                            mixture = np.ones(n_variable) * 0.1
                            mixture[vertex_idx] = 1.0 - (n_variable - 1) * 0.1
                            # Add some noise
                            mixture += np.random.uniform(-0.05, 0.05, n_variable)
                            mixture = np.maximum(mixture, 0)
                            mixture = mixture / mixture.sum()
                        elif strategy == 'edges':
                            # Along edges
                            edge_idx = i % (n_variable * (n_variable - 1) // 2)
                            # Find which edge
                            edge_count = 0
                            for j in range(n_variable):
                                for k in range(j + 1, n_variable):
                                    if edge_count == edge_idx:
                                        mixture = np.zeros(n_variable)
                                        t = np.random.uniform(0.2, 0.8)
                                        mixture[j] = t
                                        mixture[k] = 1 - t
                                        # Add small amounts to other components
                                        for m in range(n_variable):
                                            if m != j and m != k:
                                                mixture[m] = 0.05
                                        mixture = mixture / mixture.sum()
                                        break
                                    edge_count += 1
                                if edge_count > edge_idx:
                                    break
                        else:  # center
                            # Near center
                            mixture = np.ones(n_variable) / n_variable
                            mixture += np.random.uniform(-0.1, 0.1, n_variable)
                            mixture = np.maximum(mixture, 0)
                            mixture = mixture / mixture.sum()
                        
                        variable_mixtures.append(mixture)
                    
                    variable_mixtures = np.array(variable_mixtures)
                else:
                    # Original logic - focus on specific components
                    alpha = np.ones(n_variable)
                    for i, idx in enumerate(variable_indices):
                        if self.component_names[idx] in focus_components:
                            alpha[i] = 3.0  # Higher weight for focus components
                    variable_mixtures = np.random.dirichlet(alpha, n_candidates)
            else:
                alpha = np.ones(n_variable)
                variable_mixtures = np.random.dirichlet(alpha, n_candidates)
            
            # Assign to full mixture array
            for i, idx in enumerate(variable_indices):
                candidates[:, idx] = variable_mixtures[:, i]
            
            # Apply bounds to each component
            for i in range(self.n_components):
                min_val, max_val = self.component_bounds[i]
                candidates[:, i] = np.clip(candidates[:, i], min_val, max_val)
            
            # Renormalize to ensure sum to 1
            candidates = self._normalize_mixtures(candidates)
        
        return candidates
    
    def _normalize_mixtures(self, mixtures: np.ndarray) -> np.ndarray:
        """Normalize mixtures to sum to 1"""
        row_sums = mixtures.sum(axis=1)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        return mixtures / row_sums[:, np.newaxis]
    
    def _filter_mixture_candidates(self, candidates: np.ndarray, 
                                 existing_design: np.ndarray, 
                                 min_distance: float = 0.02) -> np.ndarray:
        """Remove candidates too close to existing mixture points"""
        filtered = []
        
        for candidate in candidates:
            # Calculate distances to all existing points
            distances = np.sqrt(np.sum((existing_design - candidate)**2, axis=1))
            
            # Keep if far enough from all existing points
            if np.min(distances) > min_distance:
                filtered.append(candidate)
        
        return np.array(filtered) if filtered else candidates
    
    def _remove_avoided_mixtures(self, candidates: np.ndarray, 
                                avoided_regions: np.ndarray,
                                buffer: float = 0.05) -> np.ndarray:
        """Remove candidates near avoided mixture regions"""
        filtered = []
        
        for candidate in candidates:
            distances = np.sqrt(np.sum((avoided_regions - candidate)**2, axis=1))
            if np.min(distances) > buffer:
                filtered.append(candidate)
        
        return np.array(filtered) if filtered else candidates
    
    def _calculate_mixture_d_efficiency(self, design: np.ndarray, 
                                      model_type: str) -> float:
        """Calculate D-efficiency for mixture design"""
        # Create model matrix for mixture
        if model_type == "linear":
            # Scheffé linear model (no intercept)
            model_matrix = design
        elif model_type == "quadratic":
            # Scheffé quadratic model
            n_runs = design.shape[0]
            n_terms = self.n_components + (self.n_components * (self.n_components - 1)) // 2
            model_matrix = np.zeros((n_runs, n_terms))
            
            # Linear terms
            model_matrix[:, :self.n_components] = design
            
            # Interaction terms
            col = self.n_components
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    model_matrix[:, col] = design[:, i] * design[:, j]
                    col += 1
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Calculate D-efficiency
        try:
            XtX = model_matrix.T @ model_matrix
            det_XtX = np.linalg.det(XtX)
            n_params = model_matrix.shape[1]
            d_eff = (det_XtX / len(design)) ** (1 / n_params)
            return d_eff
        except:
            return -1e6
    
    def get_mixture_recommendations(self, n_variable_components: int) -> dict:
        """
        Get recommendations for sequential mixture experimentation
        
        Parameters:
        -----------
        n_variable_components : int
            Number of components that can vary (not fixed)
            
        Returns:
        --------
        dict
            Recommendations for Stage 1 and Stage 2
        """
        # Stage 1: Linear mixture model
        stage1_params = n_variable_components  # No intercept in mixture models
        stage1_min = stage1_params
        stage1_rec = max(int(np.ceil(stage1_params * 1.5)), stage1_params + 3)
        stage1_exc = max(stage1_params * 2, stage1_params + 5)
        
        # Stage 2: Quadratic mixture model
        quad_params = n_variable_components + (n_variable_components * (n_variable_components - 1)) // 2
        total_rec = max(int(np.ceil(quad_params * 1.5)), quad_params + 5)
        stage2_rec = max(total_rec - stage1_rec, 5)
        
        return {
            'stage1': {
                'purpose': 'Screening (Linear Mixture Model)',
                'minimum': stage1_min,
                'recommended': stage1_rec,
                'excellent': stage1_exc,
                'can_fit': f'Linear mixture model with {stage1_params} parameters',
                'note': 'No intercept in mixture models'
            },
            'stage2': {
                'purpose': 'Optimization (Quadratic Mixture Model)',
                'recommended_additional': stage2_rec,
                'total_runs': stage1_rec + stage2_rec,
                'total_parameters': quad_params,
                'efficiency_gain': 'Focuses on promising mixture regions'
            },
            'mixture_specific': {
                'fixed_components': len(self.fixed_components),
                'variable_components': n_variable_components,
                'constraints': 'All components sum to 1',
                'bounds': 'All components between 0 and 1'
            }
        }
    
    def save_mixture_stage(self, design: np.ndarray, stage_name: str, 
                         metadata: dict = None) -> str:
        """Save mixture experiment stage to history"""
        stage_info = {
            'stage_name': stage_name,
            'timestamp': datetime.now().isoformat(),
            'design': design.tolist(),
            'n_runs': len(design),
            'n_components': self.n_components,
            'component_names': self.component_names,
            'fixed_components': self.fixed_components,
            'metadata': metadata or {}
        }
        
        self.experiment_history.append(stage_info)
        
        # Save to file
        filename = f"sequential_mixture_{stage_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stage_info, f, indent=2)
        
        return filename


# Example usage
def sequential_mixture_example():
    """Example: Sequential mixture experimentation with parts mode"""
    print("=== Sequential Mixture DOE Example (Parts Mode) ===\n")
    
    # Define mixture with parts
    component_names = ['Polymer_A', 'Polymer_B', 'Filler', 'Additive_1', 'Additive_2']
    
    # Bounds in parts (not proportions)
    component_bounds_parts = [
        (1.0, 3.0),    # Polymer_A: 1-3 parts
        (1.0, 3.0),    # Polymer_B: 1-3 parts  
        (0.2, 1.0),    # Filler: 0.2-1 parts
        (0.02, 0.1),   # Additive_1: 0.02-0.1 parts
        (0.02, 0.1)    # Additive_2: 0.02-0.1 parts
    ]
    
    # Fix one component in parts
    fixed_components_parts = {'Additive_2': 0.05}  # Always 0.05 parts
    
    # Create sequential mixture DOE in parts mode
    seq_mix = SequentialMixtureDOE(
        n_components=5,
        component_names=component_names,
        component_bounds=component_bounds_parts,
        fixed_components=fixed_components_parts,
        use_parts_mode=True
    )
    
    # Get recommendations
    n_variable = 4  # 5 components - 1 fixed
    recommendations = seq_mix.get_mixture_recommendations(n_variable)
    
    print("Stage 1 Recommendations:")
    print(f"  Minimum: {recommendations['stage1']['minimum']} runs")
    print(f"  Recommended: {recommendations['stage1']['recommended']} runs")
    print(f"  Can fit: {recommendations['stage1']['can_fit']}")
    
    # Generate Stage 1 design
    stage1_design = seq_mix.generate_d_optimal_mixture(
        n_runs=recommendations['stage1']['recommended'],
        model_type="linear",
        random_seed=42
    )
    
    print(f"\nStage 1 Design Generated: {len(stage1_design)} mixtures")
    print("First 3 mixtures (proportions):")
    for i in range(min(3, len(stage1_design))):
        print(f"  Mix {i+1}: {stage1_design[i].round(3)}")
        print(f"    Sum: {stage1_design[i].sum():.3f}")
    
    # Show batch quantities for 100 unit batch
    batch_quantities = seq_mix.calculate_batch_quantities(stage1_design[:3], batch_size=100)
    print("\nBatch Quantities (100 unit batch):")
    print(batch_quantities.round(2))
    
    # Generate Stage 2 augmentation
    stage2_design = seq_mix.augment_mixture_design(
        stage1_design,
        n_additional_runs=recommendations['stage2']['recommended_additional'],
        model_type="quadratic",
        focus_components=['Polymer_A', 'Polymer_B'],  # Focus on main polymers
        random_seed=43
    )
    
    print(f"\nStage 2 Augmentation: {len(stage2_design)} additional mixtures")
    print("Focused on Polymer_A and Polymer_B based on Stage 1 results")
    
    # Verify all constraints
    all_designs = np.vstack([stage1_design, stage2_design])
    print(f"\nVerification:")
    print(f"  All sums close to 1: {np.allclose(all_designs.sum(axis=1), 1.0)}")
    print(f"  All values non-negative: {np.all(all_designs >= 0)}")
    
    return seq_mix, stage1_design, stage2_design


if __name__ == "__main__":
    seq_mix, stage1, stage2 = sequential_mixture_example()
