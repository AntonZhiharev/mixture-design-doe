"""
Simplified Mixture Design Implementation
One method - one class approach for better clarity and maintainability
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict
import warnings


class MixtureDesignBase(ABC):
    """Base class for all mixture design methods"""
    
    def __init__(self, n_components: int, component_names: Optional[List[str]] = None,
                 use_parts_mode: bool = False, component_bounds: Optional[List[Tuple[float, float]]] = None,
                 fixed_components: Optional[Dict[str, float]] = None):
        """
        Initialize base mixture design
        
        Parameters:
        -----------
        n_components : int
            Number of components in the mixture
        component_names : List[str], optional
            Names of components. If None, uses X1, X2, etc.
        use_parts_mode : bool, optional
            If True, work with parts that get normalized to proportions
        component_bounds : List[Tuple[float, float]], optional
            Lower and upper bounds for each component
        fixed_components : Dict[str, float], optional
            Components with fixed values (in parts if use_parts_mode=True)
        """
        self.n_components = n_components
        self.component_names = component_names or [f'X{i+1}' for i in range(n_components)]
        self.use_parts_mode = use_parts_mode
        self.fixed_components = fixed_components or {}
        
        # Set default bounds
        if component_bounds is None:
            if use_parts_mode:
                self.component_bounds = [(0.1, 10.0)] * n_components
            else:
                self.component_bounds = [(0.0, 1.0)] * n_components
        else:
            self.component_bounds = component_bounds
        
        # Store original values for parts mode
        self.original_bounds = self.component_bounds.copy() if use_parts_mode else None
        self.original_fixed_components = self.fixed_components.copy()
        
        # Keep parts mode - we'll generate directly in parts space and normalize
        # No need to convert bounds ahead of time
        
        # Storage for parts design
        self.parts_design = None
        
    def _convert_parts_to_proportions(self):
        """Convert parts to proportions using simple normalization"""
        if not self.use_parts_mode:
            return
        
        print("\nConverting from parts to proportions...")
        
        # Simple approach: normalize bounds by total max parts
        total_max_parts = sum(bound[1] for bound in self.component_bounds)
        if total_max_parts == 0:
            total_max_parts = 1.0
        
        component_bounds_props = []
        for min_parts, max_parts in self.component_bounds:
            min_prop = min_parts / total_max_parts
            max_prop = max_parts / total_max_parts
            component_bounds_props.append((min_prop, max_prop))
        
        self.component_bounds = component_bounds_props
        
        # Convert fixed components to proportions
        if self.fixed_components:
            total_fixed_parts = sum(self.fixed_components.values())
            if total_fixed_parts > 0:
                fixed_components_props = {}
                
                for name, parts in self.fixed_components.items():
                    fixed_components_props[name] = parts / total_fixed_parts
                
                # Store original fixed components proportions for later use
                self.original_fixed_components_proportions = fixed_components_props
                
                # Update fixed components
                self.fixed_components = fixed_components_props
        
        # Switch to proportion mode since we've converted from parts
        self.use_parts_mode = False
        
        print("✅ Conversion complete")
        
    def convert_design_to_parts(self, design: np.ndarray) -> np.ndarray:
        """Convert normalized design to parts"""
        if not hasattr(self, 'original_fixed_components') or not self.original_fixed_components:
            # No fixed components - just scale proportionally
            return design * 100.0  # Default scale factor
        
        # Calculate total parts for each run based on fixed components
        parts_design = np.zeros_like(design)
        
        for row_idx in range(len(design)):
            # Calculate total parts from fixed components
            total_parts = 0
            for comp_name, fixed_parts in self.original_fixed_components.items():
                comp_idx = self.component_names.index(comp_name)
                fixed_prop = design[row_idx, comp_idx]
                if fixed_prop > 0:
                    total_parts = fixed_parts / fixed_prop
                    break
            
            if total_parts == 0:
                total_parts = 100.0  # Default
            
            # Convert all components to parts
            for comp_idx, comp_name in enumerate(self.component_names):
                if comp_name in self.original_fixed_components:
                    parts_design[row_idx, comp_idx] = self.original_fixed_components[comp_name]
                else:
                    parts_design[row_idx, comp_idx] = design[row_idx, comp_idx] * total_parts
        
        return parts_design
    
    def get_parts_design(self) -> Optional[np.ndarray]:
        """Get the design in parts (if available)"""
        return self.parts_design
        
    @abstractmethod
    def generate_design(self, **kwargs) -> pd.DataFrame:
        """Generate the design matrix"""
        pass
    
    def _to_dataframe(self, design_array: np.ndarray) -> pd.DataFrame:
        """Convert design array to DataFrame with component names"""
        # Simple approach - just convert to DataFrame
        # If we were in parts mode, store the parts design for reference
        if hasattr(self, 'original_bounds') and self.original_bounds:
            self.parts_design = self.convert_design_to_parts(design_array)
        
        df = pd.DataFrame(design_array, columns=self.component_names)
        
        return df
    
    def validate_design(self, design: np.ndarray) -> bool:
        """Validate that design points sum to 1 and are non-negative"""
        sums = np.sum(design, axis=1)
        return np.allclose(sums, 1.0) and np.all(design >= -1e-10)


class SimplexLatticeDesign(MixtureDesignBase):
    """Simplex Lattice Design for mixture experiments"""
    
    def generate_design(self, degree: int = 2) -> pd.DataFrame:
        """
        Generate Simplex Lattice design
        
        Parameters:
        -----------
        degree : int
            Degree of the lattice (1, 2, 3, etc.)
            
        Returns:
        --------
        pd.DataFrame
            Design matrix with mixture proportions
        """
        points = []
        
        # Generate all combinations that sum to degree
        def generate_combinations(n_components, degree, current=[]):
            if len(current) == n_components - 1:
                # Last component is determined by the constraint
                last = degree - sum(current)
                if last >= 0:
                    points.append(current + [last])
                return
            
            for i in range(degree + 1):
                if sum(current) + i <= degree:
                    generate_combinations(n_components, degree, current + [i])
        
        generate_combinations(self.n_components, degree)
        
        # Convert to proportions
        design_array = np.array(points) / degree
        
        return self._to_dataframe(design_array)


class SimplexCentroidDesign(MixtureDesignBase):
    """Simplex Centroid Design for mixture experiments"""
    
    def generate_design(self) -> pd.DataFrame:
        """
        Generate Simplex Centroid design
        
        Returns:
        --------
        pd.DataFrame
            Design matrix with all centroids
        """
        from itertools import combinations
        
        points = []
        
        # Generate centroids for all subsets
        for k in range(1, self.n_components + 1):
            for combo in combinations(range(self.n_components), k):
                point = np.zeros(self.n_components)
                for idx in combo:
                    point[idx] = 1.0 / k
                points.append(point)
        
        design_array = np.array(points)
        
        return self._to_dataframe(design_array)


class DOptimalMixtureDesign(MixtureDesignBase):
    """D-Optimal Design for mixture experiments"""
    
    def __init__(self, n_components: int, component_names: Optional[List[str]] = None,
                 use_parts_mode: bool = False, component_bounds: Optional[List[Tuple[float, float]]] = None,
                 fixed_components: Optional[Dict[str, float]] = None):
        super().__init__(n_components, component_names, use_parts_mode, component_bounds, fixed_components)
        self.model_matrix = None
        
    def generate_design(self, n_runs: int, include_interior: bool = True) -> pd.DataFrame:
        """
        Generate D-Optimal design
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        include_interior : bool
            Whether to include interior points (not just vertices)
            
        Returns:
        --------
        pd.DataFrame
            D-optimal design matrix
        """
        # Generate candidate set
        candidates = self._generate_candidates(include_interior, n_runs)
        
        # Select D-optimal subset
        selected_indices = self._select_doptimal_subset(candidates, n_runs)
        
        design_array = candidates[selected_indices]
        
        return self._to_dataframe(design_array)
    
    def _generate_candidates(self, include_interior: bool, n_runs: int = None) -> np.ndarray:
        """Generate candidate points - work in parts space first, then normalize"""
        candidates = []
        
        if self.use_parts_mode and hasattr(self, 'original_bounds'):
            # Work in parts space directly!
            print(f"Generating candidates in parts space with bounds: {self.original_bounds}")
            parts_candidates = self._generate_parts_candidates(include_interior, n_runs)
            
            # Normalize each parts candidate to proportions
            for parts_point in parts_candidates:
                total_parts = np.sum(parts_point)
                if total_parts > 0:
                    normalized_point = parts_point / total_parts
                    candidates.append(normalized_point)
        else:
            # Work in proportion space (standard case)
            candidates = self._generate_proportion_candidates(include_interior, n_runs)
        
        candidates = np.array(candidates) if candidates else np.array([]).reshape(0, self.n_components)
        
        print(f"Generated {len(candidates)} candidates")
        return candidates
    
    def _generate_parts_candidates(self, include_interior: bool, n_runs: int = None) -> List[np.ndarray]:
        """Generate candidates directly in parts space"""
        parts_candidates = []
        
        if not hasattr(self, 'original_bounds'):
            return parts_candidates
        
        lower_parts = [bound[0] for bound in self.original_bounds]
        upper_parts = [bound[1] for bound in self.original_bounds]
        
        print(f"Parts bounds: min={lower_parts}, max={upper_parts}")
        
        # Generate parts vertices (each component at max, others at min)
        for i in range(self.n_components):
            vertex_parts = np.array(lower_parts, dtype=float)
            vertex_parts[i] = upper_parts[i]
            parts_candidates.append(vertex_parts)
        
        # Generate edge points in parts space
        from itertools import combinations
        for i, j in combinations(range(self.n_components), 2):
            for ratio in [0.3, 0.5, 0.7]:
                edge_parts = np.array(lower_parts, dtype=float)
                # Use ratio to distribute between components i and j
                total_available = (upper_parts[i] - lower_parts[i]) + (upper_parts[j] - lower_parts[j])
                comp_i_add = total_available * ratio
                comp_j_add = total_available * (1 - ratio)
                
                # Check bounds
                if (lower_parts[i] + comp_i_add <= upper_parts[i] and 
                    lower_parts[j] + comp_j_add <= upper_parts[j]):
                    edge_parts[i] = lower_parts[i] + comp_i_add
                    edge_parts[j] = lower_parts[j] + comp_j_add
                    parts_candidates.append(edge_parts)
        
        if include_interior:
            # Generate random interior points in parts space
            target_candidates = max(n_runs * 2 if n_runs else 20, 10)
            additional_needed = max(0, target_candidates - len(parts_candidates))
            
            np.random.seed(42)
            for _ in range(additional_needed):
                # Generate random point within parts bounds
                random_parts = np.random.uniform(lower_parts, upper_parts)
                parts_candidates.append(random_parts)
        
        print(f"Generated {len(parts_candidates)} parts candidates")
        return parts_candidates
    
    def _generate_proportion_candidates(self, include_interior: bool, n_runs: int = None) -> List[np.ndarray]:
        """Generate candidates in standard proportion space"""
        candidates = []
        
        # Standard vertices (pure components)
        for i in range(self.n_components):
            vertex = np.zeros(self.n_components)
            vertex[i] = 1.0
            candidates.append(vertex)
        
        # Standard centroid
        if include_interior:
            centroid = np.ones(self.n_components) / self.n_components
            candidates.append(centroid)
        
        # Edge points
        from itertools import combinations
        for i, j in combinations(range(self.n_components), 2):
            for ratio in [0.2, 0.5, 0.8]:
                edge_point = np.zeros(self.n_components)
                edge_point[i] = ratio
                edge_point[j] = 1.0 - ratio
                candidates.append(edge_point)
        
        # Random interior points
        if include_interior and n_runs:
            additional_needed = max(0, n_runs * 2 - len(candidates))
            np.random.seed(42)
            for _ in range(additional_needed):
                random_point = np.random.dirichlet([1.2] * self.n_components)
                candidates.append(random_point)
        
        return candidates
    
    def _get_effective_bounds(self) -> Tuple[List[float], List[float]]:
        """Get effective lower and upper bounds"""
        if hasattr(self, 'component_bounds') and self.component_bounds:
            lower_bounds = [bound[0] for bound in self.component_bounds]
            upper_bounds = [bound[1] for bound in self.component_bounds]
        else:
            lower_bounds = [0.0] * self.n_components
            upper_bounds = [1.0] * self.n_components
        
        return lower_bounds, upper_bounds
    
    def _is_feasible_point(self, point: np.ndarray) -> bool:
        """Check if a point satisfies all constraints"""
        # Check sum constraint
        if not np.isclose(np.sum(point), 1.0, atol=1e-10):
            return False
        
        # Check non-negativity
        if np.any(point < -1e-10):
            return False
        
        # Check bounds if they exist
        if hasattr(self, 'component_bounds') and self.component_bounds:
            lower_bounds, upper_bounds = self._get_effective_bounds()
            for i in range(self.n_components):
                if point[i] < lower_bounds[i] - 1e-10 or point[i] > upper_bounds[i] + 1e-10:
                    return False
        
        return True
    
    def _generate_random_feasible_point(self) -> Optional[np.ndarray]:
        """Generate a single random point that satisfies constraints"""
        lower_bounds, upper_bounds = self._get_effective_bounds()
        
        max_attempts = 100
        for attempt in range(max_attempts):
            # Generate random point using rejection sampling
            if hasattr(self, 'component_bounds') and self.component_bounds:
                # Use uniform sampling within bounds, then project to simplex
                point = np.random.uniform(lower_bounds, upper_bounds)
                point = point / np.sum(point)  # Project to simplex
                
                # Check if still within bounds after normalization
                if self._is_feasible_point(point):
                    return point
            else:
                # Use Dirichlet for unconstrained case
                point = np.random.dirichlet([1.2] * self.n_components)
                if self._is_feasible_point(point):
                    return point
        
        return None
    
    def _select_doptimal_subset(self, candidates: np.ndarray, n_runs: int) -> np.ndarray:
        """Select D-optimal subset from candidates using exchange algorithm"""
        n_candidates = len(candidates)
        
        if n_runs >= n_candidates:
            return np.arange(n_candidates)
        
        # Start with random selection
        selected = np.random.choice(n_candidates, n_runs, replace=False)
        
        # Build model matrix
        X = self._build_model_matrix(candidates[selected])
        
        # Exchange algorithm
        improved = True
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            current_det = np.linalg.det(X.T @ X)
            
            for i in range(n_runs):
                for j in range(n_candidates):
                    if j not in selected:
                        # Try exchange
                        trial_selected = selected.copy()
                        trial_selected[i] = j
                        
                        trial_X = self._build_model_matrix(candidates[trial_selected])
                        trial_det = np.linalg.det(trial_X.T @ trial_X)
                        
                        if trial_det > current_det * 1.001:  # Small improvement threshold
                            selected = trial_selected
                            X = trial_X
                            current_det = trial_det
                            improved = True
                            break
                
                if improved:
                    break
            
            iteration += 1
        
        return selected
    
    def _build_model_matrix(self, design: np.ndarray) -> np.ndarray:
        """Build model matrix for linear model"""
        n_runs = len(design)
        
        # For mixture model: includes main effects only (no intercept due to constraint)
        X = design.copy()
        
        return X


class IOptimalMixtureDesign(MixtureDesignBase):
    """I-Optimal Design for mixture experiments (minimizes average prediction variance)"""
    
    def __init__(self, n_components: int, component_names: Optional[List[str]] = None,
                 use_parts_mode: bool = False, component_bounds: Optional[List[Tuple[float, float]]] = None,
                 fixed_components: Optional[Dict[str, float]] = None):
        super().__init__(n_components, component_names, use_parts_mode, component_bounds, fixed_components)
        self.model_matrix = None
        
    def generate_design(self, n_runs: int, include_interior: bool = True, model_type: str = "linear") -> pd.DataFrame:
        """
        Generate I-Optimal design
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        include_interior : bool
            Whether to include interior points (not just vertices)
        model_type : str
            Model type for optimization ("linear", "quadratic", "cubic")
            
        Returns:
        --------
        pd.DataFrame
            I-optimal design matrix
        """
        # Generate candidate set
        candidates = self._generate_candidates(include_interior, n_runs)
        
        # Select I-optimal subset
        selected_indices = self._select_ioptimal_subset(candidates, n_runs, model_type)
        
        design_array = candidates[selected_indices]
        
        return self._to_dataframe(design_array)
    
    def _generate_candidates(self, include_interior: bool, n_runs: int = None) -> np.ndarray:
        """Generate candidate points - work in parts space first, then normalize"""
        candidates = []
        
        if self.use_parts_mode and hasattr(self, 'original_bounds'):
            # Work in parts space directly!
            print(f"I-optimal: Generating candidates in parts space with bounds: {self.original_bounds}")
            parts_candidates = self._generate_parts_candidates(include_interior, n_runs)
            
            # Normalize each parts candidate to proportions
            for parts_point in parts_candidates:
                total_parts = np.sum(parts_point)
                if total_parts > 0:
                    normalized_point = parts_point / total_parts
                    candidates.append(normalized_point)
        else:
            # Work in proportion space (standard case)
            candidates = self._generate_proportion_candidates_ioptimal(include_interior, n_runs)
        
        candidates = np.array(candidates) if candidates else np.array([]).reshape(0, self.n_components)
        
        print(f"I-optimal: Generated {len(candidates)} candidates")
        return candidates
    
    def _generate_parts_candidates(self, include_interior: bool, n_runs: int = None) -> List[np.ndarray]:
        """Generate candidates directly in parts space (same as D-optimal)"""
        parts_candidates = []
        
        if not hasattr(self, 'original_bounds'):
            return parts_candidates
        
        lower_parts = [bound[0] for bound in self.original_bounds]
        upper_parts = [bound[1] for bound in self.original_bounds]
        
        print(f"I-optimal parts bounds: min={lower_parts}, max={upper_parts}")
        
        # Generate parts vertices (each component at max, others at min)
        for i in range(self.n_components):
            vertex_parts = np.array(lower_parts, dtype=float)
            vertex_parts[i] = upper_parts[i]
            parts_candidates.append(vertex_parts)
        
        # Generate edge points in parts space
        from itertools import combinations
        for i, j in combinations(range(self.n_components), 2):
            for ratio in [0.3, 0.5, 0.7]:
                edge_parts = np.array(lower_parts, dtype=float)
                # Use ratio to distribute between components i and j
                total_available = (upper_parts[i] - lower_parts[i]) + (upper_parts[j] - lower_parts[j])
                comp_i_add = total_available * ratio
                comp_j_add = total_available * (1 - ratio)
                
                # Check bounds
                if (lower_parts[i] + comp_i_add <= upper_parts[i] and 
                    lower_parts[j] + comp_j_add <= upper_parts[j]):
                    edge_parts[i] = lower_parts[i] + comp_i_add
                    edge_parts[j] = lower_parts[j] + comp_j_add
                    parts_candidates.append(edge_parts)
        
        if include_interior:
            # Generate more random interior points for I-optimal (needs more diversity)
            target_candidates = max(n_runs * 3 if n_runs else 30, 15)
            additional_needed = max(0, target_candidates - len(parts_candidates))
            
            np.random.seed(42)
            for _ in range(additional_needed):
                # Generate random point within parts bounds
                random_parts = np.random.uniform(lower_parts, upper_parts)
                parts_candidates.append(random_parts)
        
        print(f"I-optimal: Generated {len(parts_candidates)} parts candidates")
        return parts_candidates
    
    def _generate_proportion_candidates_ioptimal(self, include_interior: bool, n_runs: int = None) -> List[np.ndarray]:
        """Generate candidates in standard proportion space for I-optimal"""
        candidates = []
        
        # Standard vertices (pure components)
        for i in range(self.n_components):
            vertex = np.zeros(self.n_components)
            vertex[i] = 1.0
            candidates.append(vertex)
        
        # Standard centroid
        centroid = np.ones(self.n_components) / self.n_components
        candidates.append(centroid)
        
        # Edge points
        from itertools import combinations
        for i, j in combinations(range(self.n_components), 2):
            for ratio in [0.2, 0.35, 0.5, 0.65, 0.8]:  # More ratios for I-optimal
                edge_point = np.zeros(self.n_components)
                edge_point[i] = ratio
                edge_point[j] = 1.0 - ratio
                candidates.append(edge_point)
        
        # Additional interior points for better I-optimality
        if include_interior and self.n_components == 3:
            # Add systematic interior points
            interior_points = [
                [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
                [0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.5, 0.3],
                [0.5, 0.2, 0.3], [0.3, 0.2, 0.5], [0.2, 0.3, 0.5],
                [0.4, 0.4, 0.2], [0.4, 0.2, 0.4], [0.2, 0.4, 0.4]
            ]
            for point in interior_points:
                candidates.append(point)
        
        # Random interior points for I-optimal (needs more diversity)
        if include_interior and n_runs:
            additional_needed = max(0, n_runs * 3 - len(candidates))
            np.random.seed(42)
            for _ in range(additional_needed):
                if self.n_components == 3:
                    random_point = np.random.dirichlet([1.5, 1.5, 1.5])  # Slightly favor interior
                else:
                    random_point = np.random.dirichlet([1.2] * self.n_components)
                candidates.append(random_point)
        
        return candidates
    
    def _get_effective_bounds(self) -> Tuple[List[float], List[float]]:
        """Get effective lower and upper bounds"""
        if hasattr(self, 'component_bounds') and self.component_bounds:
            lower_bounds = [bound[0] for bound in self.component_bounds]
            upper_bounds = [bound[1] for bound in self.component_bounds]
        else:
            lower_bounds = [0.0] * self.n_components
            upper_bounds = [1.0] * self.n_components
        
        return lower_bounds, upper_bounds
    
    def _is_feasible_point(self, point: np.ndarray) -> bool:
        """Check if a point satisfies all constraints"""
        # Check sum constraint
        if not np.isclose(np.sum(point), 1.0, atol=1e-10):
            return False
        
        # Check non-negativity
        if np.any(point < -1e-10):
            return False
        
        # Check bounds if they exist
        if hasattr(self, 'component_bounds') and self.component_bounds:
            lower_bounds, upper_bounds = self._get_effective_bounds()
            for i in range(self.n_components):
                if point[i] < lower_bounds[i] - 1e-10 or point[i] > upper_bounds[i] + 1e-10:
                    return False
        
        return True
    
    def _generate_random_feasible_point(self) -> Optional[np.ndarray]:
        """Generate a single random point that satisfies constraints"""
        lower_bounds, upper_bounds = self._get_effective_bounds()
        
        max_attempts = 100
        for attempt in range(max_attempts):
            # Generate random point using rejection sampling
            if hasattr(self, 'component_bounds') and self.component_bounds:
                # Use uniform sampling within bounds, then project to simplex
                point = np.random.uniform(lower_bounds, upper_bounds)
                point = point / np.sum(point)  # Project to simplex
                
                # Check if still within bounds after normalization
                if self._is_feasible_point(point):
                    return point
            else:
                # Use Dirichlet for unconstrained case
                if self.n_components == 3:
                    point = np.random.dirichlet([1.5, 1.5, 1.5])  # Slightly favor interior
                else:
                    point = np.random.dirichlet([1.2] * self.n_components)
                
                if self._is_feasible_point(point):
                    return point
        
        return None
    
    def _select_ioptimal_subset(self, candidates: np.ndarray, n_runs: int, model_type: str) -> np.ndarray:
        """Select I-optimal subset from candidates using exchange algorithm"""
        n_candidates = len(candidates)
        
        if n_runs >= n_candidates:
            return np.arange(n_candidates)
        
        # Start with random selection
        selected = np.random.choice(n_candidates, n_runs, replace=False)
        
        # Build model matrix
        X = self._build_model_matrix(candidates[selected], model_type)
        
        # Generate prediction points for I-optimality (average prediction variance)
        prediction_points = self._generate_prediction_points()
        
        # Exchange algorithm for I-optimality
        improved = True
        max_iterations = 150  # More iterations for I-optimal
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            current_i_criterion = self._calculate_i_criterion(X, prediction_points, model_type)
            
            for i in range(n_runs):
                for j in range(n_candidates):
                    if j not in selected:
                        # Try exchange
                        trial_selected = selected.copy()
                        trial_selected[i] = j
                        
                        trial_X = self._build_model_matrix(candidates[trial_selected], model_type)
                        trial_i_criterion = self._calculate_i_criterion(trial_X, prediction_points, model_type)
                        
                        # For I-optimality, we want to minimize average prediction variance
                        if trial_i_criterion < current_i_criterion * 0.999:  # Small improvement threshold
                            selected = trial_selected
                            X = trial_X
                            current_i_criterion = trial_i_criterion
                            improved = True
                            break
                
                if improved:
                    break
            
            iteration += 1
        
        return selected
    
    def _generate_prediction_points(self) -> np.ndarray:
        """Generate points for I-optimality calculation (where we want good predictions)"""
        prediction_points = []
        
        # Include vertices
        for i in range(self.n_components):
            vertex = np.zeros(self.n_components)
            vertex[i] = 1.0
            prediction_points.append(vertex)
        
        # Include centroid
        centroid = np.ones(self.n_components) / self.n_components
        prediction_points.append(centroid)
        
        # Include systematic grid points
        if self.n_components == 3:
            # Create a systematic grid in the simplex
            for i in range(5, 10):  # Different levels
                for j in range(i):
                    for k in range(i - j):
                        if i - j - k >= 0:
                            point = np.array([j, k, i - j - k]) / i
                            if np.all(point >= 0.05):  # Avoid boundary issues
                                prediction_points.append(point)
        else:
            # For other dimensions, use random points
            np.random.seed(123)  # Fixed seed for consistency
            for _ in range(50):
                random_point = np.random.dirichlet([1] * self.n_components)
                if np.all(random_point >= 0.05):
                    prediction_points.append(random_point)
        
        return np.array(prediction_points)
    
    def _calculate_i_criterion(self, X: np.ndarray, prediction_points: np.ndarray, model_type: str) -> float:
        """Calculate I-optimality criterion (average prediction variance)"""
        try:
            # Information matrix
            info_matrix = X.T @ X
            
            # Check if matrix is invertible
            if np.linalg.det(info_matrix) < 1e-12:
                return float('inf')  # Bad design
            
            inv_info = np.linalg.inv(info_matrix)
            
            # Calculate average prediction variance over prediction points
            total_variance = 0.0
            valid_points = 0
            
            for pred_point in prediction_points:
                # Build model vector for this prediction point
                pred_vector = self._build_model_vector(pred_point, model_type)
                
                # Prediction variance
                pred_variance = pred_vector.T @ inv_info @ pred_vector
                total_variance += pred_variance
                valid_points += 1
            
            if valid_points == 0:
                return float('inf')
            
            # Return average prediction variance
            return total_variance / valid_points
            
        except np.linalg.LinAlgError:
            return float('inf')
    
    def _build_model_vector(self, point: np.ndarray, model_type: str) -> np.ndarray:
        """Build model vector for a single point"""
        if model_type == "linear":
            return point
        elif model_type == "quadratic":
            # Linear terms + interactions
            terms = list(point)
            for i in range(len(point)):
                for j in range(i+1, len(point)):
                    terms.append(point[i] * point[j])
            return np.array(terms)
        elif model_type == "cubic":
            # Linear + quadratic + cubic terms
            terms = list(point)
            # Quadratic interactions
            for i in range(len(point)):
                for j in range(i+1, len(point)):
                    terms.append(point[i] * point[j])
            # Cubic interactions
            for i in range(len(point)):
                for j in range(i+1, len(point)):
                    for k in range(j+1, len(point)):
                        terms.append(point[i] * point[j] * point[k])
            return np.array(terms)
        else:
            return point
    
    def _build_model_matrix(self, design: np.ndarray, model_type: str = "linear") -> np.ndarray:
        """Build model matrix for given model type"""
        n_runs = len(design)
        
        if model_type == "linear":
            X = design.copy()
        elif model_type == "quadratic":
            # Linear terms + interactions
            model_terms = []
            # Linear terms
            for i in range(self.n_components):
                model_terms.append(design[:, i])
            # Interaction terms
            for i in range(self.n_components):
                for j in range(i+1, self.n_components):
                    model_terms.append(design[:, i] * design[:, j])
            X = np.column_stack(model_terms)
        elif model_type == "cubic":
            # All terms up to cubic
            model_terms = []
            # Linear terms
            for i in range(self.n_components):
                model_terms.append(design[:, i])
            # Quadratic interactions
            for i in range(self.n_components):
                for j in range(i+1, self.n_components):
                    model_terms.append(design[:, i] * design[:, j])
            # Cubic interactions
            for i in range(self.n_components):
                for j in range(i+1, self.n_components):
                    for k in range(j+1, self.n_components):
                        model_terms.append(design[:, i] * design[:, j] * design[:, k])
            X = np.column_stack(model_terms)
        else:
            X = design.copy()
        
        return X


class AugmentedDesign(MixtureDesignBase):
    """Augmented mixture design (adds axial points)"""
    
    def generate_design(self, base_design: pd.DataFrame, delta: float = 0.1) -> pd.DataFrame:
        """
        Generate augmented design by adding axial points
        
        Parameters:
        -----------
        base_design : pd.DataFrame
            Base design to augment
        delta : float
            Distance from vertices for axial points
            
        Returns:
        --------
        pd.DataFrame
            Augmented design matrix
        """
        base_array = base_design.values
        augmented_points = []
        
        # Add base design
        augmented_points.extend(base_array)
        
        # Add axial points near vertices
        for i in range(self.n_components):
            axial_point = np.ones(self.n_components) * (delta / (self.n_components - 1))
            axial_point[i] = 1 - delta
            augmented_points.append(axial_point)
        
        design_array = np.array(augmented_points)
        
        return self._to_dataframe(design_array)


class CustomMixtureDesign(MixtureDesignBase):
    """Custom mixture design from user-specified points"""
    
    def generate_design(self, design_matrix: np.ndarray) -> pd.DataFrame:
        """
        Create design from custom matrix
        
        Parameters:
        -----------
        design_matrix : np.ndarray
            User-specified design matrix
            
        Returns:
        --------
        pd.DataFrame
            Custom design as DataFrame
        """
        if not self.validate_design(design_matrix):
            warnings.warn("Design points do not sum to 1 or contain negative values")
        
        return self._to_dataframe(design_matrix)


class ExtremeVerticesDesign(MixtureDesignBase):
    """Extreme Vertices design for constrained mixture spaces"""
    
    def generate_design(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> pd.DataFrame:
        """
        Generate Extreme Vertices design for constrained mixture
        
        Parameters:
        -----------
        lower_bounds : np.ndarray
            Lower bounds for each component
        upper_bounds : np.ndarray
            Upper bounds for each component
            
        Returns:
        --------
        pd.DataFrame
            Extreme vertices of the constrained region
        """
        # This is a simplified version - full implementation would find all vertices
        # of the constrained simplex region
        vertices = []
        
        # Add vertices where one component is at max, others distributed
        for i in range(self.n_components):
            vertex = lower_bounds.copy()
            remaining = 1.0 - np.sum(lower_bounds)
            max_additional = min(upper_bounds[i] - lower_bounds[i], remaining)
            vertex[i] += max_additional
            
            # Distribute remaining to other components
            remaining -= max_additional
            for j in range(self.n_components):
                if j != i and remaining > 0:
                    additional = min(upper_bounds[j] - lower_bounds[j], remaining)
                    vertex[j] += additional
                    remaining -= additional
            
            if np.abs(np.sum(vertex) - 1.0) < 1e-10:
                vertices.append(vertex)
        
        # Add centroid of feasible region
        if len(vertices) > 0:
            centroid = np.mean(vertices, axis=0)
            vertices.append(centroid)
        
        design_array = np.array(vertices)
        
        return self._to_dataframe(design_array)


class FixedPartsMixtureDesign(MixtureDesignBase):
    """Specialized class for mixture designs with fixed components in parts mode"""
    
    def __init__(self, n_components: int, component_names: Optional[List[str]] = None,
                 component_bounds: Optional[List[Tuple[float, float]]] = None,
                 fixed_components: Optional[Dict[str, float]] = None):
        """Initialize with forced parts mode"""
        super().__init__(n_components, component_names, use_parts_mode=True, 
                        component_bounds=component_bounds, fixed_components=fixed_components)
    
    def generate_design(self, n_runs: int, design_type: str = "d-optimal", **kwargs) -> pd.DataFrame:
        """Generate design with fixed parts"""
        if design_type.lower() == "d-optimal":
            designer = DOptimalMixtureDesign(self.n_components, self.component_names)
            # Copy our parts configuration to the designer
            designer.original_bounds = self.original_bounds
            designer.original_fixed_components = self.original_fixed_components
            designer.parts_design = None
            
            result = designer.generate_design(n_runs=n_runs, **kwargs)
            
            # Store the parts design
            if designer.parts_design is not None:
                self.parts_design = designer.parts_design
            
            return result
        elif design_type.lower() == "simplex-lattice":
            designer = SimplexLatticeDesign(self.n_components, self.component_names,
                                          use_parts_mode=True, component_bounds=self.original_bounds,
                                          fixed_components=self.original_fixed_components)
            return designer.generate_design(**kwargs)
        elif design_type.lower() == "simplex-centroid":
            designer = SimplexCentroidDesign(self.n_components, self.component_names,
                                           use_parts_mode=True, component_bounds=self.original_bounds,
                                           fixed_components=self.original_fixed_components)
            return designer.generate_design(**kwargs)
        else:
            raise ValueError(f"Design type '{design_type}' not supported for fixed parts design")


# Factory function for easy access
def create_mixture_design(method: str, n_components: int, **kwargs) -> pd.DataFrame:
    """
    Factory function to create mixture designs
    
    Parameters:
    -----------
    method : str
        Design method: 'simplex-lattice', 'simplex-centroid', 'd-optimal', 
        'augmented', 'extreme-vertices', 'custom', 'fixed-parts'
    n_components : int
        Number of mixture components
    **kwargs : dict
        Additional parameters specific to each method. Supports:
        - component_names: List of component names
        - use_parts_mode: Boolean for parts mode
        - component_bounds: List of (min, max) tuples
        - fixed_components: Dict of fixed component values
        
    Returns:
    --------
    pd.DataFrame
        Generated design matrix
    """
    designers = {
        'simplex-lattice': SimplexLatticeDesign,
        'simplex-centroid': SimplexCentroidDesign,
        'd-optimal': DOptimalMixtureDesign,
        'i-optimal': IOptimalMixtureDesign,
        'augmented': AugmentedDesign,
        'extreme-vertices': ExtremeVerticesDesign,
        'custom': CustomMixtureDesign,
        'fixed-parts': FixedPartsMixtureDesign
    }
    
    if method not in designers:
        raise ValueError(f"Unknown method: {method}. Choose from {list(designers.keys())}")
    
    designer_class = designers[method]
    
    # Extract common parameters
    component_names = kwargs.pop('component_names', None)
    use_parts_mode = kwargs.pop('use_parts_mode', False)
    component_bounds = kwargs.pop('component_bounds', None)
    fixed_components = kwargs.pop('fixed_components', None)
    
    # Create designer with appropriate parameters
    if method == 'fixed-parts':
        designer = designer_class(n_components, component_names, component_bounds, fixed_components)
    elif use_parts_mode or component_bounds or fixed_components:
        designer = designer_class(n_components, component_names, use_parts_mode, 
                                component_bounds, fixed_components)
    else:
        designer = designer_class(n_components, component_names)
    
    return designer.generate_design(**kwargs)


# Example usage
if __name__ == "__main__":
    # Simplex Lattice
    design1 = create_mixture_design('simplex-lattice', 3, degree=2)
    print("Simplex Lattice Design:")
    print(design1)
    
    # D-Optimal
    design2 = create_mixture_design('d-optimal', 3, n_runs=10, include_interior=True)
    print("\nD-Optimal Design:")
    print(design2)
    
    # Simplex Centroid
    design3 = create_mixture_design('simplex-centroid', 3)
    print("\nSimplex Centroid Design:")
    print(design3)
