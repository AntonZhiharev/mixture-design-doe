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
    """D-Optimal Design for mixture experiments using OptimalDesignGenerator"""
    
    def __init__(self, n_components: int, component_names: Optional[List[str]] = None,
                 use_parts_mode: bool = False, component_bounds: Optional[List[Tuple[float, float]]] = None,
                 fixed_components: Optional[Dict[str, float]] = None):
        super().__init__(n_components, component_names, use_parts_mode, component_bounds, fixed_components)
        
        # Import the OptimalDesignGenerator
        from .optimal_design_generator import OptimalDesignGenerator
        self.OptimalDesignGenerator = OptimalDesignGenerator
        
    def generate_design(self, n_runs: int, include_interior: bool = True, model_type: str = "linear") -> pd.DataFrame:
        """
        Generate D-Optimal design using the superior OptimalDesignGenerator approach
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        include_interior : bool
            Whether to include interior points (not just vertices) - always True for OptimalDesignGenerator
        model_type : str
            Model type ("linear", "quadratic", "cubic")
            
        Returns:
        --------
        pd.DataFrame
            D-optimal design matrix
        """
        print(f"\n🚀 Using OptimalDesignGenerator for superior D-optimal design")
        print(f"   Model: {model_type}, Runs: {n_runs}, Components: {self.n_components}")
        
        # Prepare component ranges - should be passed whenever component bounds are provided
        # This enables the proportional parts mixture functionality in OptimalDesignGenerator
        component_ranges = None
        if hasattr(self, 'component_bounds') and self.component_bounds:
            # Use original bounds if available (for parts mode), otherwise use current bounds
            if hasattr(self, 'original_bounds') and self.original_bounds:
                component_ranges = self.original_bounds
                print(f"   Parts mode with original bounds: {component_ranges}")
            else:
                component_ranges = self.component_bounds
                print(f"   Component bounds: {component_ranges}")
        
        # Create OptimalDesignGenerator for mixture design
        generator = self.OptimalDesignGenerator(
            num_variables=self.n_components,
            num_runs=n_runs,
            design_type="mixture",  # This is crucial - tells it to work in simplex space!
            model_type=model_type,
            component_ranges=component_ranges
        )
        
        # Generate optimal design using new enhanced API
        final_det = generator.generate_optimal_design(method="d_optimal")
        
        # Store the generator instance for accessing determinant and other metrics
        self._last_generator = generator
        
        print(f"✅ Generated design with determinant: {final_det:.6f}")
        
        # Get design points
        design_points = generator.design_points
        
        # ALWAYS use the optimal design points directly from OptimalDesignGenerator
        # These are the mathematically optimal points in mixture space
        design_array = np.array(design_points)
        
        print(f"✅ Using optimal design points directly from OptimalDesignGenerator")
        print(f"   First few points: {design_array[:3] if len(design_array) > 0 else 'none'}")
        
        # Verify they sum to 1
        if len(design_array) > 0:
            sums = np.sum(design_array, axis=1)
            print(f"   Point sums: {sums[:5] if len(sums) > 5 else sums}")
            if not np.allclose(sums, 1.0, atol=1e-10):
                print(f"   ⚠️ Warning: Points don't sum to 1, normalizing...")
                # Normalize just in case
                for i in range(len(design_array)):
                    total = np.sum(design_array[i])
                    if total > 1e-10:
                        design_array[i] = design_array[i] / total
            else:
                print(f"   ✅ All points correctly sum to 1.0")
        
        # Convert to parts ONLY for display/reference purposes (don't change the design!)
        if component_ranges:
            print(f"✅ Converting to parts for display purposes (design points remain unchanged)")
            try:
                design_points_parts, _ = generator.convert_to_parts(component_ranges)
                # Store parts design for reference/display only
                self.parts_design = np.array(design_points_parts)
                print(f"✅ Parts conversion available for display")
            except Exception as e:
                print(f"⚠️ Parts conversion failed: {e}")
                self.parts_design = None
        
        # Calculate D-efficiency for verification
        d_efficiency = self._calculate_d_efficiency(design_array, model_type)
        print(f"✅ Final D-efficiency: {d_efficiency:.6f}")
        
        return self._to_dataframe(design_array)
    
    def _calculate_d_efficiency(self, design_matrix: np.ndarray, model_type: str = "linear") -> float:
        """Calculate D-efficiency of the design using proper gram matrix approach"""
        try:
            # Import the proper mathematical functions from optimal_design_generator
            from .optimal_design_generator import gram_matrix, calculate_determinant
            
            X = self._build_model_matrix(design_matrix, model_type)
            
            # Use gram matrix approach for triangular/constrained matrices
            info_matrix = gram_matrix(X.tolist())
            det_value = calculate_determinant(info_matrix)
            
            n_runs, n_params = X.shape
            d_efficiency = (det_value / n_runs) ** (1/n_params) if det_value > 0 else 0.0
            
            return d_efficiency
        except Exception:
            return 0.0
    
    def _build_model_matrix(self, design: np.ndarray, model_type: str = "linear") -> np.ndarray:
        """Build model matrix for given model type"""
        if model_type == "linear":
            return design.copy()
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
            return np.column_stack(model_terms)
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
            return np.column_stack(model_terms)
        else:
            return design.copy()


class IOptimalMixtureDesign(MixtureDesignBase):
    """I-Optimal Design for mixture experiments using OptimalDesignGenerator"""
    
    def __init__(self, n_components: int, component_names: Optional[List[str]] = None,
                 use_parts_mode: bool = False, component_bounds: Optional[List[Tuple[float, float]]] = None,
                 fixed_components: Optional[Dict[str, float]] = None):
        super().__init__(n_components, component_names, use_parts_mode, component_bounds, fixed_components)
        
        # Import the OptimalDesignGenerator
        from .optimal_design_generator import OptimalDesignGenerator
        self.OptimalDesignGenerator = OptimalDesignGenerator
        
    def generate_design(self, n_runs: int, include_interior: bool = True, model_type: str = "linear") -> pd.DataFrame:
        """
        Generate I-Optimal design using the superior OptimalDesignGenerator approach
        
        Parameters:
        -----------
        n_runs : int
            Number of experimental runs
        include_interior : bool
            Whether to include interior points (not just vertices) - always True for OptimalDesignGenerator
        model_type : str
            Model type ("linear", "quadratic", "cubic")
            
        Returns:
        --------
        pd.DataFrame
            I-optimal design matrix
        """
        print(f"\n🚀 Using OptimalDesignGenerator for superior I-optimal design")
        print(f"   Model: {model_type}, Runs: {n_runs}, Components: {self.n_components}")
        
        # Prepare component ranges - should be passed whenever component bounds are provided
        # This enables the proportional parts mixture functionality in OptimalDesignGenerator
        component_ranges = None
        if hasattr(self, 'component_bounds') and self.component_bounds:
            # Use original bounds if available (for parts mode), otherwise use current bounds
            if hasattr(self, 'original_bounds') and self.original_bounds:
                component_ranges = self.original_bounds
                print(f"   Parts mode with original bounds: {component_ranges}")
            else:
                component_ranges = self.component_bounds
                print(f"   Component bounds: {component_ranges}")
        
        # Create OptimalDesignGenerator for mixture design
        generator = self.OptimalDesignGenerator(
            num_variables=self.n_components,
            num_runs=n_runs,
            design_type="mixture",  # This is crucial - tells it to work in simplex space!
            model_type=model_type,
            component_ranges=component_ranges
        )
        
        # Generate optimal design using new enhanced I-optimal API
        final_det = generator.generate_optimal_design(method="i_optimal")
        
        # Store the generator instance for accessing determinant and other metrics
        self._last_generator = generator
        
        print(f"✅ Generated I-optimal design with determinant: {final_det:.6f}")
        
        # Get design points
        design_points = generator.design_points
        
        # Convert to parts if needed, then normalize to proportions
        if component_ranges:
            # Convert from simplex space to parts, then normalize
            design_points_parts, design_points_normalized = generator.convert_to_parts(component_ranges)
            design_array = np.array(design_points_normalized)
            
            # Store parts design for reference
            self.parts_design = np.array(design_points_parts)
            print(f"✅ Converted to parts and normalized to proportions")
        else:
            # OptimalDesignGenerator for mixture designs already returns points in simplex space (sum=1)!
            # No conversion needed - just use the points directly
            design_array = np.array(design_points)
            
            print(f"✅ Using optimal design points directly from OptimalDesignGenerator")
            print(f"   First few points: {design_array[:3] if len(design_array) > 0 else 'none'}")
            
            # Verify they sum to 1
            if len(design_array) > 0:
                sums = np.sum(design_array, axis=1)
                print(f"   Point sums: {sums[:5] if len(sums) > 5 else sums}")
                if not np.allclose(sums, 1.0, atol=1e-10):
                    print(f"   ⚠️ Warning: Points don't sum to 1, normalizing...")
                    # Normalize just in case
                    for i in range(len(design_array)):
                        total = np.sum(design_array[i])
                        if total > 1e-10:
                            design_array[i] = design_array[i] / total
                else:
                    print(f"   ✅ All points correctly sum to 1.0")
        
        # Calculate I-efficiency for verification
        i_efficiency = self._calculate_i_efficiency(design_array, model_type)
        print(f"✅ Final I-efficiency: {i_efficiency:.6f}")
        
        return self._to_dataframe(design_array)
    
    def _calculate_i_efficiency(self, design_matrix: np.ndarray, model_type: str = "linear") -> float:
        """Calculate I-efficiency of the design using proper gram matrix approach"""
        try:
            # Import the proper mathematical functions from optimal_design_generator
            from .optimal_design_generator import gram_matrix, calculate_determinant, matrix_inverse, matrix_trace
            
            X = self._build_model_matrix(design_matrix, model_type)
            
            # Use gram matrix approach for triangular/constrained matrices
            info_matrix = gram_matrix(X.tolist())
            
            # Calculate I-efficiency (1 / trace of inverse)
            try:
                inverse_matrix = matrix_inverse(info_matrix)
                trace_value = matrix_trace(inverse_matrix)
                i_efficiency = 1.0 / trace_value if trace_value > 1e-10 else 0.0
                return i_efficiency
            except:
                return 0.0
        except Exception:
            return 0.0
    
    def _build_model_matrix(self, design: np.ndarray, model_type: str = "linear") -> np.ndarray:
        """Build model matrix for given model type"""
        if model_type == "linear":
            return design.copy()
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
            return np.column_stack(model_terms)
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
            return np.column_stack(model_terms)
        else:
            return design.copy()
    


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
