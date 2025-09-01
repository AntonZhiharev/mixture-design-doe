"""
Candidate Generation Algorithms for DOE
=======================================

This module contains all candidate generation strategies used in optimal
design of experiments. Extracted from various classes to eliminate code
duplication and improve maintainability.

Strategies include:
- Latin Hypercube Sampling (LHS)
- Structured points (vertices, edges, centers)
- Proportional candidate generation
- Grid-based candidate generation
- Random candidate generation
- Anti-clustering candidate generation
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
import sys
import os

# Add parent directory to path for imports if needed
if 'src' not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.math_utils import latin_hypercube_sampling, normalize_to_simplex


class CandidateGenerator:
    """
    Base class for generating candidate points for optimal design selection
    """
    
    def __init__(self, n_components: int, component_names: List[str],
                 component_bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize candidate generator
        
        Parameters:
        -----------
        n_components : int
            Number of components/variables
        component_names : List[str]
            Names of components
        component_bounds : List[Tuple[float, float]], optional
            Bounds for each component
        """
        self.n_components = n_components
        self.component_names = component_names
        self.component_bounds = component_bounds or [(0.0, 1.0)] * n_components
    
    def generate_candidates(self, n_candidates: int, **kwargs) -> List[np.ndarray]:
        """
        Generate candidate points
        
        Parameters:
        -----------
        n_candidates : int
            Number of candidates to generate
        **kwargs : dict
            Additional parameters for specific generators
            
        Returns:
        --------
        List[np.ndarray]
            List of candidate points
        """
        raise NotImplementedError("Subclasses must implement generate_candidates")


class RandomCandidateGenerator(CandidateGenerator):
    """Generate random candidate points"""
    
    def generate_candidates(self, n_candidates: int, **kwargs) -> List[np.ndarray]:
        """Generate random candidates using uniform sampling"""
        candidates = []
        
        for _ in range(n_candidates):
            point = np.zeros(self.n_components)
            
            for j in range(self.n_components):
                min_val, max_val = self.component_bounds[j]
                point[j] = np.random.uniform(min_val, max_val)
            
            candidates.append(point)
        
        # Inject JMP-like ultra-high quartic hotspot candidates
        if self.n_components >= 5:
            from itertools import product
            hotspot_band = np.linspace(0.23, 0.27, 5)
            hotspot_candidates = []
            for vals in product(hotspot_band, repeat=4):
                quartic_value = np.prod(vals)
                if quartic_value >= 0.0038:  # Near theoretical max for 0.25 each
                    remaining = 1 - sum(vals)
                    if remaining >= 0:
                        point = np.array(list(vals) + [remaining])
                        if len(point) < self.n_components:
                            # Fill remaining comps minimally
                            extra = np.zeros(self.n_components - len(point))
                            point = np.concatenate([point, extra])
                        hotspot_candidates.append(point)
            candidates.extend(hotspot_candidates)
        
        return candidates


class LHSCandidateGenerator(CandidateGenerator):
    """Generate Latin Hypercube Sampling candidates"""
    
    def generate_candidates(self, n_candidates: int, **kwargs) -> List[np.ndarray]:
        """Generate LHS candidates for better space-filling properties"""
        if self.n_components == 0:
            return []
        
        # Generate LHS in [0,1] space
        lhs_samples = latin_hypercube_sampling(n_candidates, self.n_components)
        
        candidates = []
        for sample in lhs_samples:
            point = np.zeros(self.n_components)
            
            # Map LHS samples to component bounds
            for j in range(self.n_components):
                min_val, max_val = self.component_bounds[j]
                point[j] = min_val + sample[j] * (max_val - min_val)
            
            candidates.append(point)
        
        return candidates


class StructuredPointGenerator(CandidateGenerator):
    """Generate structured points (vertices, edges, centers)"""
    
    def generate_candidates(self, n_candidates: int = None, **kwargs) -> List[np.ndarray]:
        """Generate structured points - different behavior for mixture vs general designs"""
        # Check if this is a mixture design (bounds are [0,1] and sum should be 1)
        is_mixture_design = all(bound == (0.0, 1.0) for bound in self.component_bounds)
        
        if is_mixture_design:
            return self._generate_mixture_structured_points()
        else:
            return self._generate_general_structured_points()
    
    def _generate_mixture_structured_points(self) -> List[np.ndarray]:
        """Generate structured points specifically for mixture designs - different from Simplex Centroid"""
        structured_points = []
        
        # 1. Pure components (vertices)
        for i in range(self.n_components):
            point = np.zeros(self.n_components)
            point[i] = 1.0
            structured_points.append(point)
        
        # 2. Binary edge midpoints (only adjacent pairs - different from Simplex Centroid which has all pairs)
        if self.n_components >= 2:
            for i in range(self.n_components):
                j = (i + 1) % self.n_components  # Only adjacent pairs, not all combinations
                point = np.zeros(self.n_components)
                point[i] = 0.5
                point[j] = 0.5
                structured_points.append(point)
        
        # 3. Overall centroid
        centroid = np.ones(self.n_components) / self.n_components
        structured_points.append(centroid)
        
        # 4. Structured intermediate points (high-low blends - different approach from Simplex Centroid)
        if self.n_components == 3:
            # Add 80/10/10 type blends (one dominant component)
            for i in range(self.n_components):
                point = np.ones(self.n_components) * 0.1
                point[i] = 0.8
                structured_points.append(point)
            
            # Add 60/20/20 type blends
            for i in range(self.n_components):
                point = np.ones(self.n_components) * 0.2
                point[i] = 0.6
                structured_points.append(point)
        
        elif self.n_components == 4:
            # For 4 components: 70/10/10/10 type blends
            for i in range(self.n_components):
                point = np.ones(self.n_components) * 0.1
                point[i] = 0.7
                structured_points.append(point)
        
        elif self.n_components == 2:
            # For 2 components: add 70/30 and 30/70 points
            structured_points.append(np.array([0.7, 0.3]))
            structured_points.append(np.array([0.3, 0.7]))
        
        return structured_points
    
    def _generate_general_structured_points(self) -> List[np.ndarray]:
        """Generate structured points for general (non-mixture) designs"""
        structured_points = []
        
        # All corner points of design space
        if self.n_components <= 4:  # Only for reasonable dimensions
            for i in range(2**self.n_components):
                point = np.zeros(self.n_components)
                
                # Set components to corners using binary representation
                for j in range(self.n_components):
                    min_val, max_val = self.component_bounds[j]
                    if (i >> j) & 1:
                        point[j] = max_val
                    else:
                        point[j] = min_val
                
                structured_points.append(point)
        
        # Edge midpoints for 2D space
        if self.n_components == 2:
            bounds_0 = self.component_bounds[0]
            bounds_1 = self.component_bounds[1]
            
            # Edge midpoints
            edge_configs = [
                (bounds_0[0], (bounds_1[0] + bounds_1[1]) / 2),  # Left edge
                (bounds_0[1], (bounds_1[0] + bounds_1[1]) / 2),  # Right edge
                ((bounds_0[0] + bounds_0[1]) / 2, bounds_1[0]),  # Bottom edge
                ((bounds_0[0] + bounds_0[1]) / 2, bounds_1[1]),  # Top edge
            ]
            
            for val_0, val_1 in edge_configs:
                point = np.zeros(self.n_components)
                point[0] = val_0
                point[1] = val_1
                structured_points.append(point)
        
        # Overall centroid
        point = np.zeros(self.n_components)
        for j in range(self.n_components):
            min_val, max_val = self.component_bounds[j]
            point[j] = (min_val + max_val) / 2
        structured_points.append(point)
        
        return structured_points


class SimplexLatticeGenerator(CandidateGenerator):
    """Generate Simplex Lattice design points for mixture experiments"""
    
    def generate_candidates(self, n_candidates: int = None, degree: int = 3, **kwargs) -> List[np.ndarray]:
        """
        Generate Simplex Lattice design points
        
        Parameters:
        -----------
        degree : int
            Lattice degree (affects point density)
        """
        lattice_points = []
        
        # Generate all combinations of lattice coordinates that sum to degree
        def generate_lattice_coordinates(n_components, degree, current_coords=None, remaining_degree=None):
            if current_coords is None:
                current_coords = []
            if remaining_degree is None:
                remaining_degree = degree
            
            if len(current_coords) == n_components - 1:
                # Last component is determined by the constraint
                coords = current_coords + [remaining_degree]
                return [coords]
            
            result = []
            for i in range(remaining_degree + 1):
                new_coords = current_coords + [i]
                result.extend(generate_lattice_coordinates(
                    n_components, degree, new_coords, remaining_degree - i
                ))
            
            return result
        
        # Generate lattice coordinates
        lattice_coords = generate_lattice_coordinates(self.n_components, degree)
        
        # Convert to proportions
        for coords in lattice_coords:
            proportions = [coord / degree for coord in coords]
            lattice_points.append(np.array(proportions))
        
        return lattice_points


class SimplexCentroidGenerator(CandidateGenerator):
    """Generate Simplex Centroid design points for mixture experiments"""
    
    def generate_candidates(self, n_candidates: int = None, **kwargs) -> List[np.ndarray]:
        """Generate Simplex Centroid design points"""
        centroid_points = []
        
        # Generate all possible component subsets (1 to n_components)
        for subset_size in range(1, self.n_components + 1):
            # Generate all combinations of 'subset_size' components
            from itertools import combinations
            
            for component_indices in combinations(range(self.n_components), subset_size):
                # Create centroid point
                point = np.zeros(self.n_components)
                equal_proportion = 1.0 / subset_size
                
                for idx in component_indices:
                    point[idx] = equal_proportion
                
                centroid_points.append(point)
        
        return centroid_points


class ExtremeVerticesGenerator(CandidateGenerator):
    """Generate Extreme Vertices design points for constrained mixture experiments"""
    
    def generate_candidates(self, n_candidates: int = None, **kwargs) -> List[np.ndarray]:
        """Generate Extreme Vertices design points using the correct simplified algorithm"""
        # Get bounds for mixture proportions
        lower_bounds = np.array([bound[0] for bound in self.component_bounds])
        upper_bounds = np.array([bound[1] for bound in self.component_bounds])
        
        # Use the correct simplified algorithm from the old implementation
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
            
            # Only add if it sums to 1 (valid mixture point)
            if np.abs(np.sum(vertex) - 1.0) < 1e-10:
                vertices.append(vertex)
        
        # Add centroid of feasible region if we have vertices
        if len(vertices) > 0:
            centroid = np.mean(vertices, axis=0)
            vertices.append(centroid)
        
        # Remove duplicates
        unique_points = []
        for point in vertices:
            is_duplicate = False
            for existing in unique_points:
                if np.allclose(point, existing, atol=1e-6):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points


class GridCandidateGenerator(CandidateGenerator):
    """Generate grid-based candidates for even space coverage with denser intermediate levels supporting quartic leverage"""
    
    def generate_candidates(self, n_candidates: int, **kwargs) -> List[np.ndarray]:
        """
        Generate grid-based candidates.

        Enhancement: In addition to evenly spaced rational fractions, include custom dense intermediate levels
        such as 0.15, 0.2, 0.3 to support high quartic leverage points where all major factors are ≥ 0.15.
        """
        if self.n_components < 2:
            # Fall back to random generation for 1D
            random_gen = RandomCandidateGenerator(self.n_components, self.component_names, self.component_bounds)
            return random_gen.generate_candidates(n_candidates)
        
        # Base uniform grid for first two dimensions
        grid_size = int(np.ceil(np.sqrt(n_candidates)))
        base_levels_0 = np.linspace(self.component_bounds[0][0], self.component_bounds[0][1], grid_size)
        base_levels_1 = np.linspace(self.component_bounds[1][0], self.component_bounds[1][1], grid_size)

        # Insert extra intermediate levels specifically to catch JMP-style quartic leverage points
        extra_levels = np.array([0.15, 0.2, 0.3])
        # Clamp extras to within bounds
        extra_levels_0 = extra_levels[(extra_levels >= self.component_bounds[0][0]) & (extra_levels <= self.component_bounds[0][1])]
        extra_levels_1 = extra_levels[(extra_levels >= self.component_bounds[1][0]) & (extra_levels <= self.component_bounds[1][1])]

        levels_0 = np.unique(np.concatenate([base_levels_0, extra_levels_0]))
        levels_1 = np.unique(np.concatenate([base_levels_1, extra_levels_1]))

        candidates = []
        for val_0 in levels_0:
            for val_1 in levels_1:
                for val_2 in np.unique(np.concatenate([
                    np.linspace(self.component_bounds[2][0], self.component_bounds[2][1], grid_size) if self.n_components > 2 else [0],
                    extra_levels[(extra_levels >= self.component_bounds[2][0]) & (extra_levels <= self.component_bounds[2][1])] if self.n_components > 2 else []
                ])) if self.n_components > 2 else [0]:
                    for val_3 in np.unique(np.concatenate([
                        np.linspace(self.component_bounds[3][0], self.component_bounds[3][1], grid_size) if self.n_components > 3 else [0],
                        extra_levels[(extra_levels >= self.component_bounds[3][0]) & (extra_levels <= self.component_bounds[3][1])] if self.n_components > 3 else []
                    ])) if self.n_components > 3 else [0]:
                        point = np.zeros(self.n_components)
                        point[0] = val_0
                        point[1] = val_1
                        if self.n_components > 2:
                            point[2] = val_2
                        if self.n_components > 3:
                            point[3] = val_3
                        # For remaining components, bias towards quartic leverage by ensuring ≥ 0.15 when possible
                        for k in range(4, self.n_components):
                            min_val, max_val = self.component_bounds[k]
                            biased_min = max(min_val, 0.15 * (max_val - min_val) + min_val)
                            point[k] = np.random.uniform(biased_min, max_val)
                        candidates.append(point)

        # Add dedicated quartic-favoring candidates: ensure first 4 components >= 0.15
        quartic_candidates = []
        target_levels = [0.15, 0.22, 0.25, 0.28, 0.3]
        target_levels = [lvl for lvl in target_levels if lvl >= self.component_bounds[0][0] and lvl <= self.component_bounds[0][1]]
        from itertools import product
        for vals in product(target_levels, repeat=min(4, self.n_components)):
            if sum(vals) < 1.0001:  # ensure feasible mixture interpretation if applicable
                point = np.zeros(self.n_components)
                for k in range(min(4, self.n_components)):
                    point[k] = vals[k]
                # fill remaining comps with random feasible values
                for k in range(4, self.n_components):
                    min_val, max_val = self.component_bounds[k]
                    point[k] = np.random.uniform(min_val, max_val)
                quartic_candidates.append(point)
        # merge and deduplicate
        merged = np.vstack([candidates, quartic_candidates])
        unique_candidates = []
        seen = set()
        for pt in merged:
            key = tuple(np.round(pt, 8))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(pt)
        candidates = unique_candidates

        # If too many candidates, randomly subsample to requested number
        if len(candidates) > n_candidates:
            candidates = random.sample(candidates, n_candidates)
        
        return candidates


class ProportionalCandidateGenerator(CandidateGenerator):
    """Generate proportional candidates using constraint-aware sampling"""
    
    def generate_candidates(self, n_candidates: int, **kwargs) -> List[np.ndarray]:
        """Generate proportional candidates"""
        candidates = []
        
        # Calculate total bounds
        total_min = sum(bounds[0] for bounds in self.component_bounds)
        total_max = sum(bounds[1] for bounds in self.component_bounds)
        
        for _ in range(n_candidates):
            # Generate random proportions for components
            proportions = np.random.dirichlet(np.ones(self.n_components))
            
            # Scale to fit within component space
            total_parts = total_min + np.random.random() * (total_max - total_min)
            
            point = np.zeros(self.n_components)
            
            # Set components proportionally within bounds
            for j in range(self.n_components):
                min_val, max_val = self.component_bounds[j]
                # Use proportional allocation but respect bounds
                proposed_val = proportions[j] * total_parts
                point[j] = max(min_val, min(max_val, proposed_val))
            
            candidates.append(point)
        
        return candidates


class MixtureCandidateGenerator(CandidateGenerator):
    """
    Specialized candidate generator for mixture designs with fixed components
    """
    
    def __init__(self, component_names: List[str], 
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None):
        """
        Initialize mixture candidate generator
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components
        fixed_parts : Dict[str, float], optional
            Fixed component amounts in parts
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable components in parts
        """
        self.component_names = component_names
        self.fixed_parts = fixed_parts or {}
        self.variable_bounds = variable_bounds or {}
        
        # Setup derived attributes
        self.variable_names = [name for name in component_names if name not in self.fixed_parts]
        self.fixed_names = list(self.fixed_parts.keys())
        self.n_components = len(component_names)
        self.n_variable = len(self.variable_names)
        
        # Set default bounds for variable components
        for name in self.variable_names:
            if name not in self.variable_bounds:
                self.variable_bounds[name] = (0.0, 100.0)
        
        # Calculate design space properties
        self.total_fixed_parts = sum(self.fixed_parts.values())
        min_total = sum(self.variable_bounds[name][0] for name in self.variable_names)
        max_total = sum(self.variable_bounds[name][1] for name in self.variable_names)
        self.variable_total_range = (min_total, max_total)
    
    def generate_improved_candidate_set(self, n_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate improved candidate set using multiple strategies for better space-filling
        ENHANCED to include more aggressive extreme points to match old implementation
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (parts_candidates, prop_candidates, batch_candidates)
        """
        # Calculate candidate distribution - allocate more to structured/extreme points
        n_structured = min(100, max(2**self.n_variable + 20, 15))  # Increased structured points
        n_lhs = int(0.25 * n_candidates)  # Reduced LHS
        n_proportional = int(0.35 * n_candidates)  # Reduced proportional
        n_extreme = int(0.15 * n_candidates)  # NEW: dedicated extreme candidates
        n_random = n_candidates - n_structured - n_lhs - n_proportional - n_extreme
        
        all_parts = []
        all_props = []
        all_batches = []
        
        # 1. Enhanced Structured Points
        structured_parts = self._generate_enhanced_structured_points()
        for parts in structured_parts:
            props = self._parts_to_proportions(parts)
            batch_size = np.sum(parts)
            
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(batch_size)
        
        # 2. Latin Hypercube Sampling
        if n_lhs > 0:
            lhs_parts = self._generate_lhs_candidates(n_lhs)
            for parts in lhs_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # 3. Proportional candidates
        if n_proportional > 0:
            prop_parts = self._generate_proportional_candidates(n_proportional)
            for parts in prop_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # 4. EXTREME candidates - NEW aggressive extreme point generation
        if n_extreme > 0:
            extreme_parts = self._generate_extreme_candidates(n_extreme)
            for parts in extreme_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # 5. Random candidates
        if n_random > 0:
            random_parts = self._generate_random_candidates(n_random)
            for parts in random_parts:
                props = self._parts_to_proportions(parts)
                batch_size = np.sum(parts)
                
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(batch_size)
        
        # Convert to arrays
        candidate_parts = np.array(all_parts)
        candidate_props = np.array(all_props)
        candidate_batches = np.array(all_batches)
        
        return candidate_parts, candidate_props, candidate_batches
    
    def _generate_enhanced_structured_points(self) -> List[np.ndarray]:
        """Generate enhanced structured points including TRUE extreme corners and aggressive vertex points"""
        structured_points = []
        
        # TRUE EXTREME CORNER POINTS - be very aggressive to match old implementation
        if self.n_variable <= 4:
            for i in range(2**self.n_variable):
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                
                # Set variable components to EXTREME corners
                var_idx = 0
                for j, name in enumerate(self.component_names):
                    if name not in self.fixed_parts:
                        min_val, max_val = self.variable_bounds[name]
                        if (i >> var_idx) & 1:
                            parts[j] = max_val  # Use TRUE maximum
                        else:
                            parts[j] = min_val  # Use TRUE minimum
                        var_idx += 1
                
                structured_points.append(parts)
        
        # AGGRESSIVE SINGLE-VARIABLE EXTREME POINTS - push each variable to its absolute maximum
        for var_name in self.variable_names:
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
            
            # Set target variable to MAXIMUM, others to minimum
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts:
                    min_val, max_val = self.variable_bounds[name]
                    if name == var_name:
                        parts[j] = max_val  # Push THIS variable to absolute maximum
                    else:
                        parts[j] = min_val  # Set others to minimum
            
            structured_points.append(parts)
        
        # AGGRESSIVE DUAL-VARIABLE EXTREME POINTS - for better space coverage
        if self.n_variable >= 2:
            var_pairs = [(self.variable_names[i], self.variable_names[j]) 
                        for i in range(len(self.variable_names)) 
                        for j in range(i+1, len(self.variable_names))]
            
            for var1, var2 in var_pairs[:6]:  # Limit to first 6 pairs to avoid explosion
                # Both at maximum, others at minimum
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                
                # Set variable components
                for j, name in enumerate(self.component_names):
                    if name not in self.fixed_parts:
                        min_val, max_val = self.variable_bounds[name]
                        if name in [var1, var2]:
                            parts[j] = max_val  # Push BOTH to maximum
                        else:
                            parts[j] = min_val  # Others to minimum
                
                structured_points.append(parts)
        
        # Edge midpoints for 2D variable space
        if self.n_variable == 2:
            var_names = self.variable_names
            bounds_0 = self.variable_bounds[var_names[0]]
            bounds_1 = self.variable_bounds[var_names[1]]
            
            edge_configs = [
                (bounds_0[0], (bounds_1[0] + bounds_1[1]) / 2),
                (bounds_0[1], (bounds_1[0] + bounds_1[1]) / 2),
                ((bounds_0[0] + bounds_0[1]) / 2, bounds_1[0]),
                ((bounds_0[0] + bounds_0[1]) / 2, bounds_1[1]),
            ]
            
            for var_0, var_1 in edge_configs:
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                
                # Set variable components
                parts[self.component_names.index(var_names[0])] = var_0
                parts[self.component_names.index(var_names[1])] = var_1
                
                structured_points.append(parts)
        
        # Overall centroid
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
        """Generate Latin Hypercube Sampling candidates for variable components"""
        if self.n_variable == 0:
            return []
        
        # Generate LHS in [0,1] space
        lhs_samples = latin_hypercube_sampling(n_samples, self.n_variable)
        
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
    
    def _generate_proportional_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate proportional candidates using constraint-aware sampling"""
        candidates = []
        
        for _ in range(n_samples):
            # Generate random proportions for variable components
            var_props = np.random.dirichlet(np.ones(self.n_variable))
            
            # Scale to fit within variable space
            min_total, max_total = self.variable_total_range
            total_var_parts = min_total + np.random.random() * (max_total - min_total)
            
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
            
            # Set variable components proportionally
            var_idx = 0
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts:
                    min_val, max_val = self.variable_bounds[name]
                    proposed_val = var_props[var_idx] * total_var_parts
                    parts[j] = max(min_val, min(max_val, proposed_val))
                    var_idx += 1
            
            candidates.append(parts)
        
        return candidates
    
    def _generate_random_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate random candidates using uniform sampling"""
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
    
    def _generate_extreme_candidates(self, n_samples: int) -> List[np.ndarray]:
        """Generate EXTREMELY aggressive candidates using sophisticated vertex optimization from old implementation"""
        candidates = []
        
        # Generate one sophisticated vertex for each variable component
        for dominant_var in self.variable_names:
            for attempt in range(3):  # Multiple attempts per variable
                vertex = self._generate_sophisticated_vertex(dominant_var)
                if vertex is not None:
                    candidates.append(vertex)
        
        # Fill remaining with random extreme candidates
        while len(candidates) < n_samples:
            # Use original approach for remaining candidates
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
            
            # Randomly select 1-2 variables to push to VERY HIGH values
            selected_vars = np.random.choice(self.variable_names, 
                                           size=min(2, len(self.variable_names)), 
                                           replace=False)
            
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts:
                    min_val, max_val = self.variable_bounds[name]
                    
                    if name in selected_vars:
                        # Push selected variables to 85-100% of their maximum
                        aggressive_factor = 0.85 + 0.15 * np.random.random()
                        parts[j] = min_val + aggressive_factor * (max_val - min_val)
                    else:
                        # Set others to minimum + small buffer
                        buffer_factor = 0.0 + 0.1 * np.random.random()  # 0-10%
                        parts[j] = min_val + buffer_factor * (max_val - min_val)
            
            candidates.append(parts)
        
        return candidates[:n_samples]
    
    def _generate_sophisticated_vertex(self, dominant_variable: str) -> np.ndarray:
        """
        Generate sophisticated vertex point that maximizes one variable (like old implementation)
        Based on _generate_feasible_vertex from old implementation
        """
        best_vertex = None
        max_dominant_proportion = 0.0
        
        # Try many systematic approaches to maximize the dominant component (like old implementation)
        max_attempts = 500  # Same as old implementation
        
        for attempt in range(max_attempts):
            parts = np.zeros(self.n_components)
            
            # Set fixed components
            total_fixed_parts = 0.0
            for j, name in enumerate(self.component_names):
                if name in self.fixed_parts:
                    parts[j] = self.fixed_parts[name]
                    total_fixed_parts += self.fixed_parts[name]
            
            # Strategy: Set other variable components close to their minimums to maximize dominant
            total_used = total_fixed_parts
            for j, name in enumerate(self.component_names):
                if name not in self.fixed_parts and name != dominant_variable:
                    min_val, max_val = self.variable_bounds[name]
                    
                    # Use a range from minimum to slightly above minimum
                    # Earlier attempts use values closer to minimum (like old implementation)
                    progress = attempt / max_attempts
                    max_allowable = min_val + (max_val - min_val) * 0.1 * (1 - progress)
                    
                    parts[j] = min_val + np.random.uniform(0.0, max_allowable - min_val)
                    total_used += parts[j]
            
            # Try to maximize the dominant variable
            dominant_idx = self.component_names.index(dominant_variable)
            min_val_dom, max_val_dom = self.variable_bounds[dominant_variable]
            
            # Calculate maximum feasible value for dominant component
            # Try to use as much as possible for dominant component
            remaining_capacity = max_val_dom - min_val_dom
            if remaining_capacity > 0:
                # Use maximum feasible amount for dominant component
                parts[dominant_idx] = min_val_dom + remaining_capacity * (0.9 + 0.1 * np.random.random())
                total_used += parts[dominant_idx]
            
            # Calculate what proportion this would be if we convert to proportions
            if total_used > 0:
                dominant_proportion = parts[dominant_idx] / total_used
                
                # Check if this point is valid and has a higher dominant proportion
                if dominant_proportion > max_dominant_proportion:
                    max_dominant_proportion = dominant_proportion
                    best_vertex = parts.copy()
        
        # Additional systematic search for optimal vertex (like old implementation)
        if best_vertex is None or max_dominant_proportion < 0.7:
            # Try systematic combinations (like old implementation)
            for strategy in range(10):
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                total_fixed_parts = 0.0
                for j, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[j] = self.fixed_parts[name]
                        total_fixed_parts += self.fixed_parts[name]
                
                total_used = total_fixed_parts
                
                # Set other variable components to strategic values
                for j, name in enumerate(self.component_names):
                    if name not in self.fixed_parts and name != dominant_variable:
                        min_val, max_val = self.variable_bounds[name]
                        
                        if strategy < 5:
                            # Strategy 1-5: Use minimum + small increments
                            increment = (strategy / 4) * 0.05  # 0% to 5% above minimum
                            parts[j] = min_val + increment * (max_val - min_val)
                        else:
                            # Strategy 6-10: Use minimum + random small amount
                            parts[j] = min_val + np.random.uniform(0.0, 0.03 * (max_val - min_val))
                        
                        total_used += parts[j]
                
                # Maximize dominant component
                dominant_idx = self.component_names.index(dominant_variable)
                min_val_dom, max_val_dom = self.variable_bounds[dominant_variable]
                
                # Use maximum allowable for dominant
                parts[dominant_idx] = max_val_dom
                total_used += parts[dominant_idx]
                
                # Calculate proportion
                if total_used > 0:
                    dominant_proportion = parts[dominant_idx] / total_used
                    
                    # Check if this is better
                    if dominant_proportion > max_dominant_proportion:
                        max_dominant_proportion = dominant_proportion
                        best_vertex = parts.copy()
        
        if best_vertex is not None:
            return best_vertex
        
        # Final fallback: Generate a simple extreme point
        parts = np.zeros(self.n_components)
        
        # Set fixed components
        for j, name in enumerate(self.component_names):
            if name in self.fixed_parts:
                parts[j] = self.fixed_parts[name]
        
        # Set dominant variable to maximum
        dominant_idx = self.component_names.index(dominant_variable)
        min_val_dom, max_val_dom = self.variable_bounds[dominant_variable]
        parts[dominant_idx] = max_val_dom
        
        # Set others to minimum
        for j, name in enumerate(self.component_names):
            if name not in self.fixed_parts and name != dominant_variable:
                min_val, max_val = self.variable_bounds[name]
                parts[j] = min_val
        
        return parts
    
    def _parts_to_proportions(self, parts: np.ndarray) -> np.ndarray:
        """Convert parts to proportions (normalized to sum to 1)"""
        total = np.sum(parts)
        if total <= 0:
            raise ValueError("Total parts must be positive")
        return parts / total


class AntiClusteringCandidateGenerator(CandidateGenerator):
    """
    Specialized candidate generator for anti-clustering designs
    """
    
    def __init__(self, component_names: List[str], 
                 fixed_parts: Dict[str, float] = None,
                 variable_bounds: Dict[str, Tuple[float, float]] = None,
                 min_distance_factor: float = 0.15,
                 n_components: int = None,
                 component_bounds: List[Tuple[float, float]] = None):
        """
        Initialize anti-clustering candidate generator
        
        Parameters:
        -----------
        component_names : List[str]
            Names of all components
        fixed_parts : Dict[str, float], optional
            Fixed component amounts in parts (for parts mode)
        variable_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for variable components in parts (for parts mode)
        min_distance_factor : float
            Minimum distance factor for anti-clustering
        n_components : int, optional
            Number of components (for standard mixture mode)
        component_bounds : List[Tuple[float, float]], optional
            Component bounds (for standard mixture mode)
        """
        # Set min_distance_factor first so it can be used in initialization
        self.min_distance_factor = min_distance_factor
        
        # Determine if this is parts mode or standard mixture mode
        self.is_parts_mode = bool(fixed_parts)
        
        if self.is_parts_mode:
            # Parts mode - use mixture candidate generator logic
            self.component_names = component_names
            self.fixed_parts = fixed_parts or {}
            self.variable_bounds = variable_bounds or {}
            
            # Setup derived attributes
            self.variable_names = [name for name in component_names if name not in self.fixed_parts]
            self.fixed_names = list(self.fixed_parts.keys())
            self.n_components = len(component_names)
            self.n_variable = len(self.variable_names)
            
            # Set default bounds for variable components
            for name in self.variable_names:
                if name not in self.variable_bounds:
                    self.variable_bounds[name] = (0.0, 100.0)
            
            # Calculate design space properties
            self.total_fixed_parts = sum(self.fixed_parts.values())
            min_total = sum(self.variable_bounds[name][0] for name in self.variable_names)
            max_total = sum(self.variable_bounds[name][1] for name in self.variable_names)
            self.variable_total_range = (min_total, max_total)
            
            # Setup anti-clustering metrics for parts mode
            if self.n_variable >= 2:
                var_ranges = []
                for name in self.variable_names[:2]:
                    min_val, max_val = self.variable_bounds[name]
                    var_ranges.append(max_val - min_val)
                
                self.space_diagonal = np.sqrt(sum(r**2 for r in var_ranges))
                self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
            else:
                max_range = max(self.variable_bounds[self.variable_names[0]]) if self.variable_names else 1.0
                self.space_diagonal = max_range
                self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
        
        else:
            # Standard mixture mode - use base class
            n_comp = n_components or len(component_names)
            comp_bounds = component_bounds or [(0.0, 1.0)] * n_comp
            super().__init__(n_comp, component_names, comp_bounds)
            self.space_diagonal = np.sqrt(n_comp)
            self.min_distance_threshold = self.min_distance_factor * self.space_diagonal
    
    def generate_candidates(self, n_candidates: int, **kwargs) -> List[np.ndarray]:
        """Generate anti-clustering candidates"""
        if self.is_parts_mode:
            # Parts mode - use the specialized method
            candidate_parts, candidate_props, candidate_batches = self.generate_anti_clustering_candidates(n_candidates)
            return candidate_props.tolist()
        else:
            # Standard mixture mode - generate well-spaced points in simplex
            return self._generate_standard_anti_clustering(n_candidates)
    
    def _generate_standard_anti_clustering(self, n_candidates: int) -> List[np.ndarray]:
        """Generate anti-clustering candidates for standard mixture designs"""
        # Generate a larger pool of candidates using LHS
        pool_size = max(n_candidates * 3, 100)
        
        # Generate LHS candidates in simplex space
        lhs_samples = latin_hypercube_sampling(pool_size, self.n_components)
        candidates_pool = []
        
        for sample in lhs_samples:
            # Map to component bounds and normalize to simplex
            point = np.zeros(self.n_components)
            for j in range(self.n_components):
                min_val, max_val = self.component_bounds[j]
                point[j] = min_val + sample[j] * (max_val - min_val)
            
            # Normalize to simplex (sum = 1)
            point = normalize_to_simplex(point)
            candidates_pool.append(point)
        
        candidates_pool = np.array(candidates_pool)
        
        # Apply anti-clustering selection
        selected_indices = []
        selected_indices.append(0)  # Start with first point
        
        for _ in range(n_candidates - 1):
            best_idx = -1
            best_min_distance = -1
            
            for i, candidate in enumerate(candidates_pool):
                if i in selected_indices:
                    continue
                
                # Calculate minimum distance to already selected points
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    selected_point = candidates_pool[selected_idx]
                    distance = np.linalg.norm(candidate - selected_point)
                    min_distance = min(min_distance, distance)
                
                # Keep point with maximum minimum distance (anti-clustering)
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
            else:
                # Fallback: add random point if no improvement found
                remaining_indices = [i for i in range(len(candidates_pool)) if i not in selected_indices]
                if remaining_indices:
                    selected_indices.append(remaining_indices[0])
        
        # Return selected points
        return [candidates_pool[i] for i in selected_indices[:n_candidates]]
    
    def generate_anti_clustering_candidates(self, n_candidates: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate anti-clustering candidate set"""
        # Calculate candidate distribution
        n_structured = min(20, max(5, 2**self.n_variable))
        n_grid = int(0.35 * n_candidates)
        n_lhs = int(0.25 * n_candidates)
        n_random = n_candidates - n_structured - n_grid - n_lhs
        
        all_parts, all_props, all_batches = [], [], []
        
        # 1. Enhanced structured points
        structured_parts = self._generate_enhanced_structured_points()[:n_structured]
        for parts in structured_parts:
            props = self._parts_to_proportions(parts)
            all_parts.append(parts)
            all_props.append(props)
            all_batches.append(np.sum(parts))
        
        # 2. Grid-based candidates
        if n_grid > 0:
            grid_parts = self._generate_grid_candidates(n_grid)
            for parts in grid_parts:
                props = self._parts_to_proportions(parts)
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(np.sum(parts))
        
        # 3. LHS candidates
        if n_lhs > 0:
            lhs_parts = self._generate_lhs_candidates(n_lhs)
            for parts in lhs_parts:
                props = self._parts_to_proportions(parts)
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(np.sum(parts))
        
        # 4. Random candidates
        if n_random > 0:
            random_parts = self._generate_random_candidates(n_random)
            for parts in random_parts:
                props = self._parts_to_proportions(parts)
                all_parts.append(parts)
                all_props.append(props)
                all_batches.append(np.sum(parts))
        
        candidate_parts = np.array(all_parts)
        candidate_props = np.array(all_props)
        candidate_batches = np.array(all_batches)
        
        return candidate_parts, candidate_props, candidate_batches
    
    def _generate_grid_candidates(self, n_candidates: int) -> List[np.ndarray]:
        """Generate grid-based candidates for even space coverage"""
        if self.n_variable < 2:
            return self._generate_random_candidates(n_candidates)
        
        # Create grid in first two variable dimensions
        grid_size = int(np.ceil(np.sqrt(n_candidates)))
        candidates = []
        
        var_names = self.variable_names[:2]
        bounds_0 = self.variable_bounds[var_names[0]]
        bounds_1 = self.variable_bounds[var_names[1]]
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(candidates) >= n_candidates:
                    break
                
                parts = np.zeros(self.n_components)
                
                # Set fixed components
                for k, name in enumerate(self.component_names):
                    if name in self.fixed_parts:
                        parts[k] = self.fixed_parts[name]
                
                # Set first two variable components on grid
                val_0 = bounds_0[0] + (i / (grid_size - 1)) * (bounds_0[1] - bounds_0[0]) if grid_size > 1 else (bounds_0[0] + bounds_0[1]) / 2
                val_1 = bounds_1[0] + (j / (grid_size - 1)) * (bounds_1[1] - bounds_1[0]) if grid_size > 1 else (bounds_1[0] + bounds_1[1]) / 2
                
                parts[self.component_names.index(var_names[0])] = val_0
                parts[self.component_names.index(var_names[1])] = val_1
                
                # Set remaining variable components randomly
                for name in self.variable_names[2:]:
                    min_val, max_val = self.variable_bounds[name]
                    parts[self.component_names.index(name)] = np.random.uniform(min_val, max_val)
                
                candidates.append(parts)
        
        return candidates[:n_candidates]


# Factory function for easy access
def create_candidate_generator(generator_type: str, **kwargs) -> CandidateGenerator:
    """
    Factory function to create candidate generators
    
    Parameters:
    -----------
    generator_type : str
        Type of generator: 'random', 'lhs', 'structured', 'grid', 'proportional', 
        'mixture', 'anti-clustering', 'simplex-lattice', 'simplex-centroid', 'extreme-vertices'
    **kwargs : dict
        Parameters for the specific generator
        
    Returns:
    --------
    CandidateGenerator
        Configured candidate generator
    """
    generators = {
        'random': RandomCandidateGenerator,
        'lhs': LHSCandidateGenerator,
        'structured': StructuredPointGenerator,
        'grid': GridCandidateGenerator,
        'proportional': ProportionalCandidateGenerator,
        'mixture': MixtureCandidateGenerator,
        'anti-clustering': AntiClusteringCandidateGenerator,
        'simplex-lattice': SimplexLatticeGenerator,
        'simplex-centroid': SimplexCentroidGenerator,
        'extreme-vertices': ExtremeVerticesGenerator
    }
    
    if generator_type not in generators:
        raise ValueError(f"Unknown generator type: {generator_type}. Choose from {list(generators.keys())}")
    
    generator_class = generators[generator_type]
    return generator_class(**kwargs)
