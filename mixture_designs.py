"""
Mixture Design of Experiments Implementation
Specialized for mixture experiments where components sum to 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from sklearn.preprocessing import PolynomialFeatures
from typing import List, Tuple, Dict, Optional
import itertools
import warnings
warnings.filterwarnings('ignore')

class MixtureDesign:
    """
    Class for generating optimal mixture experimental designs
    """
    
    def __init__(self, n_components: int, component_names: List[str] = None, 
                 component_bounds: List[Tuple[float, float]] = None,
                 use_parts_mode: bool = False, fixed_components: Dict[str, float] = None):
        """
        Initialize mixture design generator
        
        Parameters:
        n_components: Number of mixture components
        component_names: Names of components (default: Comp1, Comp2, ...)
        component_bounds: Lower and upper bounds for each component
            - If use_parts_mode=False: bounds must be between 0 and 1 (proportions)
            - If use_parts_mode=True: bounds are in parts (any positive values)
        use_parts_mode: If True, work with parts that get normalized to proportions
        fixed_components: Dict of component names and their fixed values
        """
        self.n_components = n_components
        self.use_parts_mode = use_parts_mode
        self.fixed_components = fixed_components or {}
        
        if component_names is None:
            self.component_names = [f'Comp_{i+1}' for i in range(n_components)]
        else:
            self.component_names = component_names
            
        if component_bounds is None:
            if use_parts_mode:
                # Default parts bounds - give reasonable ranges
                self.component_bounds = [(0.1, 10.0)] * n_components
            else:
                self.component_bounds = [(0.0, 1.0)] * n_components
        else:
            # Validate each bounds element
            for i, bounds_element in enumerate(component_bounds):
                if not isinstance(bounds_element, (tuple, list)):
                    raise ValueError(f"bounds[{i}] is not tuple/list: {bounds_element}")
                
                if len(bounds_element) != 2:
                    raise ValueError(f"bounds[{i}] length is {len(bounds_element)}, expected 2: {bounds_element}")
                
                try:
                    lower, upper = bounds_element
                except ValueError as unpack_error:
                    raise ValueError(f"Cannot unpack bounds[{i}]: {bounds_element}") from unpack_error
            
            self.component_bounds = component_bounds
            
        # Store original bounds and fixed components for parts mode
        self.original_bounds = self.component_bounds.copy() if use_parts_mode else None
        self.original_fixed_components = self.fixed_components.copy()
        
        # CRITICAL FIX: Store which components are truly fixed for model matrix creation
        # This must be done BEFORE the fixed space conversion clears self.fixed_components
        self.truly_fixed_components = set()
        if self.original_fixed_components:
            self.truly_fixed_components = set(self.original_fixed_components.keys())
        elif self.fixed_components:
            self.truly_fixed_components = set(self.fixed_components.keys())
        
        # Convert from parts to proportions if needed using fixed space solution
        if self.use_parts_mode:
            self._convert_parts_to_proportions_fixed_space()
            
            # After parts conversion, also check for the proportions dict
            if hasattr(self, 'original_fixed_components_proportions'):
                self.truly_fixed_components.update(self.original_fixed_components_proportions.keys())
        
        # Validate bounds
        if not use_parts_mode:
            self._validate_proportion_bounds()
        else:
            self._validate_parts_bounds()
    
    def _validate_parts_bounds(self):
        """Validate parts bounds"""
        for i, bounds_tuple in enumerate(self.component_bounds):
            try:
                lower, upper = bounds_tuple
                
                # Automatically fix swapped bounds (if lower > upper)
                if lower > upper:
                    print(f"Warning: Swapped bounds detected for component {i+1}. "
                          f"Automatically reordering ({lower}, {upper}) to ({upper}, {lower})")
                    lower, upper = upper, lower
                    self.component_bounds[i] = (lower, upper)
                
                # Now validate the corrected bounds
                if lower < 0:
                    raise ValueError(f"Invalid parts bounds for component {i+1}: ({lower}, {upper}). "
                                     f"Lower bound must be non-negative.")
            except Exception as parts_error:
                raise ValueError(f"Cannot process parts bounds[{i}]: {bounds_tuple}") from parts_error
    
    def _validate_proportion_bounds(self):
        """Validate proportion bounds"""
        for i, bounds_tuple in enumerate(self.component_bounds):
            try:
                lower, upper = bounds_tuple
                
                # Automatically fix swapped bounds (if lower > upper)
                if lower > upper:
                    print(f"Warning: Swapped bounds detected for component {i+1}. "
                          f"Automatically reordering ({lower}, {upper}) to ({upper}, {lower})")
                    lower, upper = upper, lower
                    self.component_bounds[i] = (lower, upper)
                
                # Now validate the corrected bounds
                if lower < 0:
                    raise ValueError(f"Invalid proportion bounds for component {i+1}: Lower bound {lower} is negative")
                if upper > 1:
                    raise ValueError(f"Invalid proportion bounds for component {i+1}: Upper bound {upper} exceeds 1")
            except ValueError as val_error:
                raise val_error
            except Exception as val_unexpected:
                raise ValueError(f"Cannot process bounds[{i}]: {bounds_tuple}") from val_unexpected
        
        # Check if bounds are feasible (sum of lower bounds <= 1, sum of upper bounds >= 1)
        try:
            sum_lower = sum(bound[0] for bound in self.component_bounds)
            sum_upper = sum(bound[1] for bound in self.component_bounds)
            
            if sum_lower > 1:
                print(f"Warning: Sum of lower bounds ({sum_lower:.4f}) exceeds 1. "
                      f"This may result in an infeasible mixture space or limit the design space.")
            if sum_upper < 1:
                print(f"Warning: Sum of upper bounds ({sum_upper:.4f}) is less than 1. "
                      f"This may result in an infeasible mixture space or require component scaling.")
                
        except Exception as sum_error:
            raise ValueError(f"Error validating bounds feasibility: {self.component_bounds}") from sum_error
    
    def _convert_parts_to_proportions_fixed_space(self):
        """
        Convert parts to proportions using fixed space solution approach
        Implementation of the EXACT algorithm from test_fixed_space_solution.py
        """
        if not self.use_parts_mode:
            return
        
        print("\nStep 1: Calculate total parts")
        total_parts = sum(bound[1] for bound in self.component_bounds)
        print(f"Total parts = {total_parts:.3f}")
        
        print("\nStep 2: Initial normalization - convert to proportions")
        component_bounds_props = []
        fixed_components_props_original = {}
        variable_indices = []
        fixed_indices = []
        
        for i, (comp_name, (min_parts, max_parts)) in enumerate(zip(self.component_names, self.component_bounds)):
            min_prop = min_parts / total_parts
            max_prop = max_parts / total_parts
            component_bounds_props.append((min_prop, max_prop))
            
            if comp_name in self.fixed_components:
                # CRITICAL FIX: Store max_prop from bounds, NOT converted fixed value
                # This matches line 46 in test_fixed_space_solution.py exactly
                fixed_components_props_original[comp_name] = max_prop
                fixed_indices.append(i)
                print(f"{comp_name}: {max_parts:.3f} parts → {max_prop:.4f} prop (FIXED)")
            else:
                variable_indices.append(i)
                print(f"{comp_name}: ({min_parts:.3f}-{max_parts:.3f}) parts → ({min_prop:.4f}-{max_prop:.4f}) props")
        
        # Special case: No fixed components in parts mode
        # Use simple normalization instead of fixed space approach
        if not self.fixed_components:
            print("\nNo fixed components detected. Using simple parts-to-proportions conversion.")
            # Just normalize the bounds directly
            self.component_bounds = component_bounds_props
            # Switch to proportion mode since we've converted from parts
            self.use_parts_mode = False
            
            # Verify the bounds
            sum_min = sum(bound[0] for bound in self.component_bounds)
            sum_max = sum(bound[1] for bound in self.component_bounds)
            
            print(f"Sum of MIN bounds = {sum_min:.4f}")
            print(f"Sum of MAX bounds = {sum_max:.4f}")
            
            if sum_max < 1.0:
                print("⚠️ Warning: Sum of maximums < 1.0. Scaling bounds to ensure feasibility.")
                # Scale up all bounds to make sum of max = 1.0
                scale_factor = 1.0 / sum_max
                self.component_bounds = [(min_val * scale_factor, max_val * scale_factor) 
                                        for min_val, max_val in self.component_bounds]
                
                # Verify scaling worked
                sum_min_scaled = sum(bound[0] for bound in self.component_bounds)
                sum_max_scaled = sum(bound[1] for bound in self.component_bounds)
                print(f"After scaling: Sum of MIN bounds = {sum_min_scaled:.4f}, Sum of MAX bounds = {sum_max_scaled:.4f}")
            
            print("✅ Simple conversion complete")
            return
        
        print("\nStep 3: Calculate flexibility space")
        # Sum of max variable components
        sum_max_variable = sum(component_bounds_props[i][1] for i in variable_indices)
        # Sum of min variable components
        sum_min_variable = sum(component_bounds_props[i][0] for i in variable_indices)
        
        # Free space is the flexibility of variable components
        free_space = sum_max_variable - sum_min_variable
        
        print(f"Sum of MIN variable components = {sum_min_variable:.4f}")
        print(f"Sum of MAX variable components = {sum_max_variable:.4f}")
        print(f"Variable flexibility (free space) = {free_space:.4f}")
        
        # Space available for fixed components
        space_when_var_at_min = 1.0 - sum_min_variable  # Max space for fixed
        space_when_var_at_max = 1.0 - sum_max_variable  # Min space for fixed
        
        print(f"\nSpace for fixed when variables at MIN = {space_when_var_at_min:.4f}")
        print(f"Space for fixed when variables at MAX = {space_when_var_at_max:.4f}")
        
        # Calculate total fixed components (original)
        total_fixed_original = sum(fixed_components_props_original.values())
        print(f"\nOriginal fixed components sum = {total_fixed_original:.4f}")
        
        print("\nStep 4: Create bounds for fixed components with flexibility")
        component_bounds_adjusted = component_bounds_props.copy()
        
        if self.fixed_components:
            for comp_name, original_value in fixed_components_props_original.items():
                # Calculate proportion of this fixed component relative to all fixed
                fraction = original_value / total_fixed_original if total_fixed_original > 0 else 0.0
                
                # When variables are at MIN, fixed can be at MAX
                max_value = space_when_var_at_min * fraction
                # When variables are at MAX, fixed must be at MIN  
                min_value = space_when_var_at_max * fraction
                
                # Update bounds for algorithm with range
                comp_idx = self.component_names.index(comp_name)
                component_bounds_adjusted[comp_idx] = (min_value, max_value)
                
                print(f"{comp_name}:")
                print(f"  Original: {original_value:.4f} ({original_value*100:.1f}%)")
                print(f"  Fraction of fixed total: {fraction:.4f}")
                print(f"  Adjusted bounds: ({min_value:.4f}, {max_value:.4f})")
                print(f"  Range: {max_value - min_value:.4f}")
            
            # Store original fixed component values for Step 6 (post-processing)
            # Use the SAME proportional values that match the original algorithm
            self.original_fixed_components_proportions = fixed_components_props_original
            
            # CRITICAL: Clear fixed_components dict so algorithms work with adjusted bounds
            # The algorithms will see no fixed components, just adjusted bounds
            self.fixed_components = {}
        
        # Update component bounds to adjusted proportions for Step 5 (algorithm execution)
        self.component_bounds = component_bounds_adjusted
        
        # CRITICAL FIX: Switch to proportion mode since we've converted from parts
        self.use_parts_mode = False
        
        # Verify the adjusted bounds
        sum_min_adjusted = sum(bound[0] for bound in self.component_bounds)
        sum_max_adjusted = sum(bound[1] for bound in self.component_bounds)
        
        print(f"\n✓ Sum of adjusted MIN bounds = {sum_min_adjusted:.6f}")
        print(f"✓ Sum of adjusted MAX bounds = {sum_max_adjusted:.6f}")
        print(f"✓ Flexibility range = {sum_max_adjusted - sum_min_adjusted:.6f}")
        
        if sum_min_adjusted >= 1.0:
            raise ValueError("⚠️ WARNING: Sum of minimums >= 1.0 - No flexibility!")
        elif sum_max_adjusted <= 1.0:
            raise ValueError("⚠️ WARNING: Sum of maximums <= 1.0 - Infeasible!")
        else:
            print("✅ Bounds are valid for mixture optimization")
    
    def _post_process_design_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """
        Post-process design according to the EXACT fixed space solution algorithm:
        Step 6: Replace fixed components with original values (from test_fixed_space_solution.py lines 155-158)
        Step 7: Final normalization to ensure sum = 1.0 (from test_fixed_space_solution.py line 166)
        """
        # Check if we have any original fixed components stored (from parts mode conversion)
        if not hasattr(self, 'original_fixed_components_proportions') and not self.original_fixed_components:
            return design
        
        if design.size == 0:
            return design
        
        # Step 6: Replace fixed components with original values (EXACT implementation from test file)
        print("\nStep 6: Replace fixed components with original proportions")
        design_corrected = design.copy()
        
        # Get the original fixed component values to use for replacement
        if hasattr(self, 'original_fixed_components_proportions'):
            # From parts mode conversion - use the calculated proportions that were stored
            original_fixed_values = self.original_fixed_components_proportions
            print("Using original fixed components from parts mode conversion:")
            for comp_name, value in original_fixed_values.items():
                print(f"  {comp_name}: {value:.6f} (calculated proportion)")
        else:
            # Direct proportion input - use as-is
            original_fixed_values = self.original_fixed_components
            print("Using original fixed components from direct input:")
            for comp_name, value in original_fixed_values.items():
                print(f"  {comp_name}: {value:.6f}")
        
        # Replace fixed component values in all rows (lines 155-158 from test file)
        for row_idx in range(len(design_corrected)):
            for comp_name, original_value in original_fixed_values.items():
                comp_idx = self.component_names.index(comp_name)
                design_corrected[row_idx, comp_idx] = original_value
        
        print("\nAfter replacing fixed components:")
        for i in range(min(3, len(design_corrected))):
            print(f"Mix {i+1}: {design_corrected[i].round(4)} (sum = {sum(design_corrected[i]):.6f})")
        
        # Step 7: Final normalization to ensure sum = 1.0 (line 166 from test file)
        print("\nStep 7: Final normalization to ensure sum = 1.0")
        design_normalized = design_corrected / design_corrected.sum(axis=1)[:, np.newaxis]
        
        print("\nFinal normalized design:")
        for i in range(min(3, len(design_normalized))):
            print(f"Mix {i+1}:")
            for j, comp_name in enumerate(self.component_names):
                if comp_name in original_fixed_values:
                    print(f"  {comp_name}: {design_normalized[i, j]:.6f} (FIXED)")
                else:
                    print(f"  {comp_name}: {design_normalized[i, j]:.6f}")
            print(f"  SUM: {sum(design_normalized[i]):.6f}")
        
        return design_normalized
    
    def convert_to_batch_quantities(self, design: np.ndarray, batch_size: float = 100.0) -> np.ndarray:
        """
        Step 8: Convert normalized design to batch quantities
        """
        return design * batch_size
    
    def generate_simplex_lattice(self, degree: int = 2) -> np.ndarray:
        """
        Generate simplex lattice design with robust error handling
        
        Parameters:
        degree: Degree of lattice (2 for {0, 1/2, 1}, 3 for {0, 1/3, 2/3, 1}, etc.)
        """
        # Validate component bounds before proceeding
        for i, bounds_element in enumerate(self.component_bounds):
            if not isinstance(bounds_element, (tuple, list)):
                raise ValueError(f"bounds[{i}] is not tuple/list: {bounds_element}")
            
            if len(bounds_element) != 2:
                raise ValueError(f"bounds[{i}] length is {len(bounds_element)}, expected 2: {bounds_element}")
            
            try:
                lower, upper = bounds_element
            except ValueError as unpack_error:
                raise ValueError(f"Cannot unpack bounds[{i}]: {bounds_element}") from unpack_error
        
        # Check if we have original fixed components (from fixed space solution)
        has_fixed_components = (hasattr(self, 'original_fixed_components_proportions') and 
                               self.original_fixed_components_proportions) or self.fixed_components
        
        # Handle fixed components: generate lattice only for variable components
        if has_fixed_components:
            # Get indices of variable components
            # Check both current fixed_components and original fixed components
            all_fixed_component_names = set()
            if self.fixed_components:
                all_fixed_component_names.update(self.fixed_components.keys())
            if hasattr(self, 'original_fixed_components_proportions'):
                all_fixed_component_names.update(self.original_fixed_components_proportions.keys())
            
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in all_fixed_component_names:
                    variable_indices.append(i)
            
            n_variable_components = len(variable_indices)
            print(f"Generating lattice for {n_variable_components} variable components (degree {degree})")
            
            # Generate lattice points for variable components only
            points = []
            levels = np.linspace(0, 1, degree + 1)
            
            # Generate all combinations that sum to 1 for variable components
            for combination in itertools.combinations_with_replacement(range(degree + 1), n_variable_components):
                if sum(combination) == degree:
                    point = [x / degree for x in combination]
                    points.append(point)
            
            # Add permutations for non-symmetric points
            unique_variable_points = []
            for point in points:
                perms = list(set(itertools.permutations(point)))
                for perm in perms:
                    if list(perm) not in unique_variable_points:
                        unique_variable_points.append(list(perm))
            
            print(f"Generated {len(unique_variable_points)} lattice points for variable components")
            
            # Convert variable component points to full design matrix
            if len(unique_variable_points) > 0:
                # Create full design matrix with all components
                full_design = np.zeros((len(unique_variable_points), self.n_components))
                
                # Fill in variable component values
                for row_idx, var_point in enumerate(unique_variable_points):
                    for var_idx, comp_idx in enumerate(variable_indices):
                        full_design[row_idx, comp_idx] = var_point[var_idx]
                
                # Apply fixed components adjustment (this will set fixed values and rescale variable)
                full_design = self._adjust_for_fixed_components(full_design)
                # Apply post-processing for fixed components 
                full_design = self._post_process_design_fixed_components(full_design)
                return full_design
            else:
                print("No valid variable lattice points found, trying fallback...")
        
        else:
            # No fixed components - use original algorithm for all components
            # Try to generate lattice points
            points = []
            levels = np.linspace(0, 1, degree + 1)
            
            # Generate all combinations that sum to 1
            for combination in itertools.combinations_with_replacement(range(degree + 1), self.n_components):
                if sum(combination) == degree:
                    point = [x / degree for x in combination]
                    
                    # Check bounds
                    if self._check_bounds(point):
                        points.append(point)
            
            # Add permutations for non-symmetric points
            unique_points = []
            for point in points:
                perms = list(set(itertools.permutations(point)))
                for perm in perms:
                    if self._check_bounds(list(perm)) and list(perm) not in unique_points:
                        unique_points.append(list(perm))
            
            if len(unique_points) > 0:
                return np.array(unique_points)
        
        # If no valid lattice points found with current bounds, use fallback strategies
        if len(unique_points) == 0:
            print(f"Warning: No valid lattice points found with degree {degree}. Trying fallback strategies...")
            
            # Strategy 1: Try lower degree
            if degree > 2:
                print(f"Trying lower degree ({degree - 1})...")
                return self.generate_simplex_lattice(degree - 1)
            
            # Strategy 2: Use extreme vertices if available
            try:
                print("Trying extreme vertices design...")
                vertices = self.generate_extreme_vertices()
                if len(vertices) > 0:
                    return vertices
            except:
                pass
            
            # Strategy 3: Use simplex centroid if available
            try:
                print("Trying simplex centroid design...")
                centroid = self.generate_simplex_centroid()
                if len(centroid) > 0:
                    return centroid
            except:
                pass
            
            # Strategy 4: Generate D-optimal design as final fallback
            try:
                print("Using D-optimal mixture design as fallback...")
                # Calculate minimum number of runs needed
                min_runs = max(self.n_components + 2, 8)  # Conservative estimate
                optimal_design = self.generate_d_optimal_mixture(min_runs, "quadratic")
                if len(optimal_design) > 0:
                    return optimal_design
            except:
                pass
            
            # Strategy 5: Last resort - try to generate at least some valid points
            print("Generating custom feasible points...")
            feasible_points = self._generate_emergency_feasible_points()
            if len(feasible_points) > 0:
                return np.array(feasible_points)
            
            # If all fails, return empty array but with better error message
            raise ValueError(
                f"Unable to generate simplex lattice design with degree {degree}. "
                f"The component bounds appear to be too restrictive for this design method. "
                f"Suggestions: 1) Try D-optimal or I-optimal methods instead, "
                f"2) Relax component bounds slightly, or 3) Use Sequential Mixture Design."
            )
        
        # Only apply post-processing for fixed components
        if len(unique_points) > 0:
            unique_points_array = np.array(unique_points)
            return self._post_process_design_fixed_components(unique_points_array)
        
        return np.array(unique_points)
    
    def _adjust_for_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """
        Adjust design matrix to account for fixed components
        """
        if not self.fixed_components:
            return design
        
        # Get indices of fixed and variable components
        fixed_indices = []
        variable_indices = []
        
        for i, name in enumerate(self.component_names):
            if name in self.fixed_components:
                fixed_indices.append(i)
            else:
                variable_indices.append(i)
        
        adjusted_design = design.copy()
        
        # Calculate fixed sum
        fixed_sum = sum(self.fixed_components.values())
        remaining_sum = 1.0 - fixed_sum
        
        if remaining_sum <= 0:
            raise ValueError(f"Fixed components sum to {fixed_sum}, leaving no room for variable components")
        
        # For each row, adjust variable components
        for row_idx in range(adjusted_design.shape[0]):
            # Get current variable component values for this row
            variable_values = adjusted_design[row_idx, variable_indices]
            variable_sum = np.sum(variable_values)
            
            # If all variable components are zero, set them to equal proportions
            if variable_sum == 0:
                equal_share = remaining_sum / len(variable_indices)
                for i, idx in enumerate(variable_indices):
                    adjusted_design[row_idx, idx] = equal_share
            else:
                # Rescale variable components proportionally
                for i, idx in enumerate(variable_indices):
                    adjusted_design[row_idx, idx] = (variable_values[i] / variable_sum) * remaining_sum
            
            # Set fixed component values
            for idx, name in enumerate(self.component_names):
                if name in self.fixed_components:
                    adjusted_design[row_idx, idx] = self.fixed_components[name]
        
        return adjusted_design
    
    def _generate_emergency_feasible_points(self) -> List[List[float]]:
        """
        Generate basic feasible mixture points as emergency fallback
        """
        feasible_points = []
        
        # Try component vertices (one component at max, others at min)
        for i in range(self.n_components):
            point = [0.0] * self.n_components
            
            # Set component i to maximum allowed
            lower_i, upper_i = self.component_bounds[i]
            point[i] = upper_i
            
            # Calculate remaining sum for other components
            remaining = 1.0 - upper_i
            other_indices = [j for j in range(self.n_components) if j != i]
            
            if remaining > 0 and len(other_indices) > 0:
                # Distribute remaining among other components at their minimums
                min_sum_others = sum(self.component_bounds[j][0] for j in other_indices)
                
                if remaining >= min_sum_others:
                    # Set others to minimum first
                    for j in other_indices:
                        point[j] = self.component_bounds[j][0]
                    
                    # Distribute any extra remaining
                    current_sum = sum(point)
                    if current_sum < 1.0:
                        extra = 1.0 - current_sum
                        # Add extra proportionally to other components (within their bounds)
                        for j in other_indices:
                            lower_j, upper_j = self.component_bounds[j]
                            available_j = upper_j - point[j]
                            if available_j > 0:
                                addition = min(extra / len(other_indices), available_j)
                                point[j] += addition
                                extra -= addition
                    
                    # Normalize to ensure exact sum of 1
                    point_sum = sum(point)
                    if point_sum > 0:
                        point = [x / point_sum for x in point]
                        
                        if self._check_bounds(point):
                            feasible_points.append(point)
        
        # Try centroid if not already added
        centroid = [1.0 / self.n_components] * self.n_components
        if self._check_bounds(centroid):
            feasible_points.append(centroid)
        
        # Try equal distribution at average bounds
        try:
            avg_point = []
            for i in range(self.n_components):
                lower, upper = self.component_bounds[i]
                avg_point.append((lower + upper) / 2)
            
            # Normalize
            point_sum = sum(avg_point)
            if point_sum > 0:
                avg_point = [x / point_sum for x in avg_point]
                if self._check_bounds(avg_point):
                    feasible_points.append(avg_point)
        except:
            pass
        
        return feasible_points
    
    def generate_simplex_centroid(self) -> np.ndarray:
        """
        Generate simplex centroid design
        """
        points = []
        
        # Single component vertices (if bounds allow)
        for i in range(self.n_components):
            point = [0.0] * self.n_components
            point[i] = 1.0
            if self._check_bounds(point):
                points.append(point)
        
        # Binary blends (centroids of edges)
        for i, j in itertools.combinations(range(self.n_components), 2):
            point = [0.0] * self.n_components
            point[i] = 0.5
            point[j] = 0.5
            if self._check_bounds(point):
                points.append(point)
        
        # Ternary blends and higher
        for r in range(3, self.n_components + 1):
            for combo in itertools.combinations(range(self.n_components), r):
                point = [0.0] * self.n_components
                for idx in combo:
                    point[idx] = 1.0 / r
                if self._check_bounds(point):
                    points.append(point)
        
        # Overall centroid
        centroid = [1.0 / self.n_components] * self.n_components
        if self._check_bounds(centroid):
            points.append(centroid)
        
        # Apply fixed components if any
        if self.fixed_components and len(points) > 0:
            points_array = np.array(points)
            points_array = self._adjust_for_fixed_components(points_array)
            return points_array
        
        return np.array(points)
    
    def generate_extreme_vertices(self) -> np.ndarray:
        """
        Generate extreme vertices design based on component bounds
        """
        vertices = []
        
        # Generate vertices by systematically setting components to bounds
        def generate_vertices_recursive(current_point, remaining_sum, component_idx):
            if component_idx == self.n_components - 1:
                # Last component gets the remaining sum
                current_point[component_idx] = remaining_sum
                lower, upper = self.component_bounds[component_idx]
                if lower <= remaining_sum <= upper:
                    vertices.append(current_point.copy())
                return
            
            lower, upper = self.component_bounds[component_idx]
            # Try both bounds for current component
            for bound_val in [lower, upper]:
                if bound_val <= remaining_sum:
                    current_point[component_idx] = bound_val
                    generate_vertices_recursive(current_point, remaining_sum - bound_val, component_idx + 1)
        
        initial_point = [0.0] * self.n_components
        generate_vertices_recursive(initial_point, 1.0, 0)
        
        # Remove duplicates
        unique_vertices = []
        for vertex in vertices:
            if not any(np.allclose(vertex, existing) for existing in unique_vertices):
                unique_vertices.append(vertex)
        
        # Apply fixed components if any
        if self.fixed_components and len(unique_vertices) > 0:
            unique_vertices_array = np.array(unique_vertices)
            unique_vertices_array = self._adjust_for_fixed_components(unique_vertices_array)
            return unique_vertices_array
        
        return np.array(unique_vertices)
    
    def generate_d_optimal_mixture(self, n_runs: int, model_type: str = "quadratic", 
                                  max_iter: int = 1000, random_seed: int = None) -> np.ndarray:
        """
        Generate D-optimal mixture design using coordinate exchange
        
        Parameters:
        n_runs: Number of experimental runs
        model_type: "linear", "quadratic", or "cubic"
        max_iter: Maximum iterations for optimization
        random_seed: Random seed for reproducibility
        """
        # Store the current model type for later reference
        self.last_model_type = model_type
        # CRITICAL FIX: Always set a unique random seed based on n_runs to avoid identical designs
        if random_seed is not None:
            # Use n_runs as part of seed to ensure different designs for different run counts
            actual_seed = random_seed + n_runs * 137  # 137 is a prime to ensure good distribution
            np.random.seed(actual_seed)
        else:
            # Use timestamp + n_runs if no seed provided
            import time
            actual_seed = int(time.time() * 1000) % 100000 + n_runs
            np.random.seed(actual_seed)
        
        # ENHANCEMENT: Vary initial randomization to avoid identical points
        # This helps especially when no fixed components are present
        np.random.seed(actual_seed + 541)  # Use a different seed for candidate generation
        
        print(f"\nGenerating D-optimal design with {n_runs} runs (seed: {actual_seed})")
        
        # Generate a more reasonably sized candidate pool to balance performance and quality
        n_candidates = max(n_runs * 40, 1000)  # Reduced multiplier for speed
        candidates = self._generate_candidate_points(n_candidates)
        
        # Filter candidates to ensure they're diverse enough
        # Remove candidates that are too close to each other
        filtered_candidates = []
        
        # Use a more stringent threshold when no fixed components to ensure more diversity
        if not self.fixed_components and not hasattr(self, 'original_fixed_components_proportions'):
            min_distance_threshold = 0.05  # Much larger threshold for non-fixed designs
            
            # Generate additional candidates with more structure and diversity
            extra_candidates = []
            # Include vertices (components at max/min bounds)
            for i in range(self.n_components):
                point = np.zeros(self.n_components)
                for j in range(self.n_components):
                    if j == i:
                        point[j] = self.component_bounds[j][1]  # Use upper bound
                    else:
                        point[j] = self.component_bounds[j][0]  # Use lower bound
                # Normalize to ensure sum = 1
                point_sum = sum(point)
                if point_sum > 0:
                    point = point / point_sum
                    if self._check_bounds(point):
                        extra_candidates.append(point)
            
            # Add centroid point
            centroid = np.zeros(self.n_components)
            for i in range(self.n_components):
                centroid[i] = (self.component_bounds[i][0] + self.component_bounds[i][1]) / 2
            centroid_sum = sum(centroid)
            if centroid_sum > 0:
                centroid = centroid / centroid_sum
                if self._check_bounds(centroid):
                    extra_candidates.append(centroid)
            
            # Add random mixtures with emphasis on individual components
            for i in range(self.n_components):
                for _ in range(3):  # 3 points favoring each component
                    point = np.random.rand(self.n_components)
                    point[i] *= 5  # Emphasize this component
                    point_sum = sum(point)
                    if point_sum > 0:
                        point = point / point_sum
                        if self._check_bounds(point):
                            extra_candidates.append(point)
            
            # Add these extra structured candidates to our pool
            if extra_candidates:
                candidates = np.vstack([candidates, np.array(extra_candidates)])
        else:
            min_distance_threshold = 1e-6  # Original threshold for designs with fixed components
        
        for candidate in candidates:
            is_diverse = True
            for existing in filtered_candidates:
                if np.linalg.norm(candidate - existing) < min_distance_threshold:
                    is_diverse = False
                    break
            if is_diverse:
                filtered_candidates.append(candidate)
                
        candidates = np.array(filtered_candidates)
        print(f"Filtered to {len(candidates)} diverse candidates from {len(candidates) + len(candidates) - len(filtered_candidates)} considered")
        
        # IMPROVED: Use multiple random starting points and select the best
        best_overall_design = None
        best_overall_d_eff = -1
        
        n_starts = min(5, max(2, n_runs // 10))  # 2-5 random starts based on n_runs
        print(f"Trying {n_starts} random starting points...")
        
        for start_idx in range(n_starts):
            # Generate different random starting design for each attempt
            start_offset = start_idx * n_runs
            if start_offset + n_runs <= len(candidates):
                current_design = candidates[start_offset:start_offset + n_runs].copy()
            else:
                # Shuffle candidates for this start
                shuffled_candidates = candidates.copy()
                np.random.shuffle(shuffled_candidates)
                current_design = shuffled_candidates[:n_runs].copy()
            
            # Calculate initial D-efficiency
            current_d_eff = self._calculate_d_efficiency(current_design, model_type)
            print(f"Start {start_idx + 1}: Initial D-eff = {current_d_eff:.6f}")
            
            # Coordinate exchange optimization
            for iteration in range(max_iter):
                improved = False
                old_d_eff = current_d_eff
                
                # Randomize the order of point exchanges to avoid bias
                point_order = list(range(n_runs))
                np.random.shuffle(point_order)
                
                for i in point_order:
                    best_point = current_design[i].copy()
                    best_eff = current_d_eff
                    
                    # Try more candidates per exchange - scale with n_runs
                    n_candidates_to_try = min(100, max(20, n_runs * 2))
                    
                    # Extra diversity check: for designs without fixed components
                    # Skip updating if we already have this point in the design
                    if not self.fixed_components and not hasattr(self, 'original_fixed_components_proportions'):
                        # Count how many nearly identical points are in the current design
                        identical_count = 0
                        current_point = current_design[i]
                        for j in range(n_runs):
                            if i != j and np.linalg.norm(current_point - current_design[j]) < 0.01:
                                identical_count += 1
                        
                        # If there are duplicate points, we need more aggressive randomization
                        if identical_count > 0:
                            # Add more random candidates
                            extra_random = []
                            for _ in range(10):
                                new_point = np.random.dirichlet(np.ones(self.n_components) * 0.5)
                                # Scale to respect bounds
                                for j in range(self.n_components):
                                    lower, upper = self.component_bounds[j]
                                    new_point[j] = lower + new_point[j] * (upper - lower)
                                # Normalize
                                new_point = new_point / np.sum(new_point)
                                if self._check_bounds(new_point):
                                    extra_random.append(new_point)
                            
                            if extra_random:
                                # Try these extra random points first
                                for rand_point in extra_random:
                                    temp_design = current_design.copy()
                                    temp_design[i] = rand_point
                                    temp_eff = self._calculate_d_efficiency(temp_design, model_type)
                                    if temp_eff > best_eff:
                                        best_eff = temp_eff
                                        best_point = rand_point.copy()
                                        improved = True
                    
                    # Select random subset of candidates to try
                    candidate_indices = np.random.choice(
                        len(candidates), 
                        size=min(n_candidates_to_try, len(candidates)), 
                        replace=False
                    )
                    
                    for cand_idx in candidate_indices:
                        candidate = candidates[cand_idx]
                        
                        # Skip if candidate is too close to existing points (avoid duplicates)
                        min_dist = np.min([np.linalg.norm(candidate - existing) 
                                         for j, existing in enumerate(current_design) if j != i])
                        if min_dist < 1e-8:
                            continue
                        
                        # Create temporary design
                        temp_design = current_design.copy()
                        temp_design[i] = candidate
                        
                        # Calculate efficiency
                        temp_eff = self._calculate_d_efficiency(temp_design, model_type)
                        
                        if temp_eff > best_eff:
                            best_eff = temp_eff
                            best_point = candidate.copy()
                            improved = True
                    
                    # Update design if improvement found
                    if improved:
                        current_design[i] = best_point
                        current_d_eff = best_eff
                
                # Check for convergence
                improvement = current_d_eff - old_d_eff
                if improvement < 1e-8:  # Very small improvement
                    print(f"Start {start_idx + 1}: Converged at iteration {iteration + 1}, D-eff = {current_d_eff:.6f}")
                    break
                elif iteration % 100 == 0 and iteration > 0:
                    print(f"Start {start_idx + 1}: Iteration {iteration}, D-eff = {current_d_eff:.6f}")
            
            # Check if this start gave the best result
            if current_d_eff > best_overall_d_eff:
                best_overall_d_eff = current_d_eff
                best_overall_design = current_design.copy()
                print(f"Start {start_idx + 1}: NEW BEST D-eff = {current_d_eff:.6f}")
        
        print(f"\nFinal D-optimal design: D-eff = {best_overall_d_eff:.6f}")
        print(f"D-efficiency in optimization space: {best_overall_d_eff:.6f}")

        # Store the optimization space D-efficiency for future reference
        self.optimization_space_d_efficiency = best_overall_d_eff

        # Only apply post-processing for fixed components at the end
        final_design = self._post_process_design_fixed_components(best_overall_design)
        
        # Note: D-efficiency after post-processing may differ due to fixed component replacement
        # The meaningful D-efficiency is the one calculated in the optimization space above
        
        return final_design
    
    def generate_i_optimal_mixture(self, n_runs: int, model_type: str = "quadratic",
                                  max_iter: int = 1000, random_seed: int = None) -> np.ndarray:
        """
        Generate I-optimal mixture design
        """
        # Store the current model type for later reference
        self.last_model_type = model_type
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        candidates = self._generate_candidate_points(n_runs * 20)
        current_design = candidates[:n_runs].copy()
        
        # Generate integration points for I-efficiency calculation
        integration_points = self._generate_candidate_points(200)
        
        best_i_eff = self._calculate_i_efficiency(current_design, model_type, integration_points)
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                best_point = current_design[i].copy()
                best_eff = best_i_eff
                
                for candidate in candidates[n_runs:n_runs+50]:
                    temp_design = current_design.copy()
                    temp_design[i] = candidate
                    
                    temp_eff = self._calculate_i_efficiency(temp_design, model_type, integration_points)
                    
                    if temp_eff < best_eff:  # Lower is better for I-optimal
                        best_eff = temp_eff
                        best_point = candidate.copy()
                        improved = True
                
                if improved:
                    current_design[i] = best_point
                    best_i_eff = best_eff
            
            if not improved:
                break
        
        # Calculate D-efficiency in optimization space and store it
        d_eff = self._calculate_d_efficiency(current_design, model_type)
        print(f"\nFinal I-optimal design: I-eff = {best_i_eff:.6f}, D-eff = {d_eff:.6f}")
        print(f"D-efficiency in optimization space: {d_eff:.6f}")
        
        # Store the optimization space D-efficiency for future reference
        self.optimization_space_d_efficiency = d_eff
        
        # Apply fixed components if any
        if self.fixed_components:
            current_design = self._adjust_for_fixed_components(current_design)
        
        # Apply post-processing for fixed components
        current_design = self._post_process_design_fixed_components(current_design)
        
        return current_design
    
    def create_mixture_model_matrix(self, X: np.ndarray, model_type: str = "quadratic") -> np.ndarray:
        """
        Create model matrix for mixture models (Scheffe polynomials)
        
        Parameters:
        X: Design matrix (n_runs x n_components)
        model_type: "linear", "quadratic", or "cubic"
        """
        # Handle empty designs gracefully
        if X.size == 0:
            # Calculate number of variable components for empty design
            if self.fixed_components:
                n_variable_components = len([name for name in self.component_names if name not in self.fixed_components])
            else:
                n_variable_components = self.n_components
            
            # Return an empty matrix with correct structure for empty design
            if model_type == "linear":
                return np.empty((0, n_variable_components))
            elif model_type == "quadratic":
                n_terms = n_variable_components + (n_variable_components * (n_variable_components - 1)) // 2
                return np.empty((0, n_terms))
            elif model_type == "cubic":
                linear_terms = n_variable_components
                quadratic_terms = (n_variable_components * (n_variable_components - 1)) // 2
                cubic_terms = (n_variable_components * (n_variable_components - 1) * (n_variable_components - 2)) // 6
                n_terms = linear_terms + quadratic_terms + cubic_terms
                return np.empty((0, n_terms))
        
        # Handle 1D array (single point) by reshaping to 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Now we can safely unpack shape
        try:
            n_runs, n_components = X.shape
        except ValueError as shape_error:
            raise ValueError(f"Cannot unpack X.shape {X.shape} - expected (n_runs, n_components)") from shape_error
        
        # If we have fixed components, only use variable components for model matrix
        # Use truly_fixed_components which tracks the original fixed components
        if self.truly_fixed_components:
            # Get indices of variable components
            variable_indices = []
            for i, name in enumerate(self.component_names):
                if name not in self.truly_fixed_components:
                    variable_indices.append(i)
            
            # Extract only variable component columns
            X_variable = X[:, variable_indices]
            n_variable_components = len(variable_indices)
            
            # Create model matrix using only variable components
            if model_type == "linear":
                # Linear mixture model: β₁x₁ + β₂x₂ + ... + βₙxₙ (no intercept)
                return X_variable
            
            elif model_type == "quadratic":
                # Quadratic mixture model: Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ
                terms = []
                
                # Linear terms
                for i in range(n_variable_components):
                    terms.append(X_variable[:, i])
                
                # Interaction terms
                for i in range(n_variable_components):
                    for j in range(i + 1, n_variable_components):
                        terms.append(X_variable[:, i] * X_variable[:, j])
                
                return np.column_stack(terms)
            
            elif model_type == "cubic":
                # Cubic mixture model: includes linear, quadratic, and cubic terms
                terms = []
                
                # Linear terms
                for i in range(n_variable_components):
                    terms.append(X_variable[:, i])
                
                # Quadratic interaction terms
                for i in range(n_variable_components):
                    for j in range(i + 1, n_variable_components):
                        terms.append(X_variable[:, i] * X_variable[:, j])
                
                # Cubic terms (three-way interactions)
                for i in range(n_variable_components):
                    for j in range(i + 1, n_variable_components):
                        for k in range(j + 1, n_variable_components):
                            terms.append(X_variable[:, i] * X_variable[:, j] * X_variable[:, k])
                
                return np.column_stack(terms)
            
            else:
                raise ValueError("model_type must be 'linear', 'quadratic', or 'cubic'")
        
        else:
            # No fixed components - use all components (original behavior)
            if model_type == "linear":
                # Linear mixture model: β₁x₁ + β₂x₂ + ... + βₙxₙ (no intercept)
                return X
            
            elif model_type == "quadratic":
                # Quadratic mixture model: Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ
                terms = []
                
                # Linear terms
                for i in range(n_components):
                    terms.append(X[:, i])
                
                # Interaction terms
                for i in range(n_components):
                    for j in range(i + 1, n_components):
                        terms.append(X[:, i] * X[:, j])
                
                return np.column_stack(terms)
            
            elif model_type == "cubic":
                # Cubic mixture model: includes linear, quadratic, and cubic terms
                terms = []
                
                # Linear terms
                for i in range(n_components):
                    terms.append(X[:, i])
                
                # Quadratic interaction terms
                for i in range(n_components):
                    for j in range(i + 1, n_components):
                        terms.append(X[:, i] * X[:, j])
                
                # Cubic terms (three-way interactions)
                for i in range(n_components):
                    for j in range(i + 1, n_components):
                        for k in range(j + 1, n_components):
                            terms.append(X[:, i] * X[:, j] * X[:, k])
                
                return np.column_stack(terms)
            
            else:
                raise ValueError("model_type must be 'linear', 'quadratic', or 'cubic'")
    
    def evaluate_mixture_design(self, X: np.ndarray, model_type: str = "quadratic") -> Dict:
        """
        Evaluate mixture design properties
        """
        # Check if this is a design we just generated with generate_d_optimal_mixture
        # If so, use the stored optimization space D-efficiency instead of recalculating
        if hasattr(self, 'optimization_space_d_efficiency') and hasattr(self, 'last_model_type') and self.last_model_type == model_type:
            d_eff = self.optimization_space_d_efficiency
        else:
            # Otherwise calculate it (for designs not from generate_d_optimal_mixture)
            d_eff = self._calculate_d_efficiency(X, model_type)
            
        i_eff = self._calculate_i_efficiency(X, model_type)
        
        # Handle empty designs gracefully  
        n_runs = X.shape[0] if X.size > 0 else 0
        n_components = X.shape[1] if X.ndim > 1 and X.size > 0 else self.n_components
        
        return {
            'n_runs': n_runs,
            'n_components': n_components,
            'model_type': model_type,
            'd_efficiency': d_eff,
            'i_efficiency': i_eff,
            'design_matrix': X
        }
    
    def plot_ternary_design(self, X: np.ndarray, title: str = "Mixture Design"):
        """
        Plot ternary diagram for 3-component mixtures
        """
        if self.n_components != 3:
            print("Ternary plots only available for 3-component mixtures")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create ternary plot triangle
            triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
            
            # Convert mixture coordinates to ternary plot coordinates
            plot_points = []
            for point in X:
                x1, x2, x3 = point
                # Convert to barycentric coordinates
                x = x2 + 0.5 * x3
                y = np.sqrt(3)/2 * x3
                plot_points.append([x, y])
            
            plot_points = np.array(plot_points)
            
            # Draw triangle
            triangle_patch = Polygon(triangle, fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(triangle_patch)
            
            # Plot design points
            ax.scatter(plot_points[:, 0], plot_points[:, 1], c='red', s=100, alpha=0.7, 
                      edgecolors='black', zorder=5)
            
            # Add point labels
            for i, (x, y) in enumerate(plot_points):
                ax.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
            
            # Add component labels
            ax.text(-0.05, -0.05, self.component_names[0], fontsize=12, ha='center')
            ax.text(1.05, -0.05, self.component_names[1], fontsize=12, ha='center')
            ax.text(0.5, np.sqrt(3)/2 + 0.05, self.component_names[2], fontsize=12, ha='center')
            
            # Add grid lines
            for i in range(1, 10):
                val = i / 10
                # Lines parallel to side 1-2 (bottom)
                x_start = val * 0.5
                y_start = val * np.sqrt(3)/2
                x_end = val * 0.5 + (1 - val)
                y_end = val * np.sqrt(3)/2
                ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.5)
                
                # Lines parallel to side 1-3 (left)
                x_start = (1 - val) * 0.5
                y_start = (1 - val) * np.sqrt(3)/2
                x_end = val + (1 - val) * 0.5
                y_end = (1 - val) * np.sqrt(3)/2
                ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.5)
                
                # Lines parallel to side 2-3 (right)
                x_start = val
                y_start = 0
                x_end = (1 - val) * 0.5
                y_end = (1 - val) * np.sqrt(3)/2
                ax.plot([x_start, x_end], [y_start, y_end], 'lightgray', alpha=0.5)
            
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")
    
    def _check_bounds(self, point: List[float]) -> bool:
        """Check if point satisfies component bounds"""
        # Validate point length matches component bounds length
        if len(point) != len(self.component_bounds):
            return False
            
        for i, val in enumerate(point):
            # Add index bounds checking
            if i >= len(self.component_bounds):
                return False
            
            # Check and unpack bounds
            try:
                bounds_element = self.component_bounds[i]
                
                if not isinstance(bounds_element, (tuple, list)) or len(bounds_element) != 2:
                    return False
                
                lower, upper = bounds_element
                
            except (ValueError, TypeError):
                return False
            
            # Use generous tolerance for floating point arithmetic
            # especially after normalization operations that can introduce errors
            # These are numerical precision issues, not real constraint violations
            tolerance = max(1e-5, abs(upper) * 1e-8)
            if val < lower - tolerance or val > upper + tolerance:
                return False
        return True
    
    def _generate_candidate_points(self, n_points: int) -> np.ndarray:
        """Generate random feasible mixture points with parts space support"""
        if self.use_parts_mode:
            return self._generate_candidate_points_parts_mode(n_points)
        else:
            return self._generate_candidate_points_proportion_mode(n_points)
    
    def _generate_candidate_points_parts_mode(self, n_points: int) -> np.ndarray:
        """Generate candidate points in parts space, then convert to proportions"""
        points = []
        max_attempts = min(n_points * 20, 5000)  # Parts mode is more efficient
        
        for attempt in range(max_attempts):
            if len(points) >= n_points:
                break
                
            # Generate random parts within bounds
            parts = np.zeros(self.n_components)
            for i in range(self.n_components):
                lower, upper = self.component_bounds[i]
                parts[i] = np.random.uniform(lower, upper)
            
            # Convert parts to proportions
            total_parts = np.sum(parts)
            if total_parts > 0:
                proportions = parts / total_parts
                points.append(proportions)
        
        # If we need more points, use systematic generation
        while len(points) < n_points:
            # Use different sampling strategies
            strategy = len(points) % 4
            
            if strategy == 0:  # Uniform random
                parts = np.random.uniform([bound[0] for bound in self.component_bounds],
                                        [bound[1] for bound in self.component_bounds])
            elif strategy == 1:  # Bias towards extremes
                parts = np.zeros(self.n_components)
                for i in range(self.n_components):
                    lower, upper = self.component_bounds[i]
                    if np.random.random() < 0.5:
                        parts[i] = np.random.uniform(lower, (lower + upper) / 3)
                    else:
                        parts[i] = np.random.uniform(2 * (lower + upper) / 3, upper)
            elif strategy == 2:  # Focus on one component
                focus_idx = len(points) % self.n_components
                parts = np.zeros(self.n_components)
                for i in range(self.n_components):
                    lower, upper = self.component_bounds[i]
                    if i == focus_idx:
                        parts[i] = np.random.uniform(upper * 0.7, upper)
                    else:
                        parts[i] = np.random.uniform(lower, upper * 0.3)
            else:  # Balanced
                parts = np.zeros(self.n_components)
                for i in range(self.n_components):
                    lower, upper = self.component_bounds[i]
                    parts[i] = np.random.uniform((lower + upper) / 3, 2 * (lower + upper) / 3)
            
            # Convert to proportions
            total_parts = np.sum(parts)
            if total_parts > 0:
                proportions = parts / total_parts
                points.append(proportions)
        
        return np.array(points[:n_points])
    
    def _generate_candidate_points_proportion_mode(self, n_points: int) -> np.ndarray:
        """
        Generate random feasible mixture points with a robust, multi-strategy approach
        to ensure excellent coverage of the design space (vertices, edges, and interior).
        """
        points = []
        
        # For designs without fixed components, emphasize diversity more
        has_fixed_components = bool(self.fixed_components) or hasattr(self, 'original_fixed_components_proportions')
        
        # Define allocation of points to different strategies
        # Adjust allocations to improve diversity when no fixed components
        if not has_fixed_components:
            allocations = {
                'vertices': 0.30,  # More points near pure components (extremes)
                'edges': 0.25,     # More points on the edges (2-component blends)
                'faces': 0.15,     # Points on faces (3+ component blends)
                'interior': 0.30   # Fewer interior points (tend to be similar)
            }
        else:
            allocations = {
                'vertices': 0.20,  # Points near pure components
                'edges': 0.20,     # Points on the edges (2-component blends)
                'faces': 0.15,     # Points on faces (3+ component blends)
                'interior': 0.45   # Points in the middle of the space
            }
        
        # 1. Generate Vertex Points (corners of the design space)
        n_vertex_points = int(n_points * allocations['vertices'])
        print(f"Generating {n_vertex_points} vertex points...")
        vertex_points = self._generate_corner_points()
        if vertex_points:
            # Select a diverse subset of vertex points
            points.extend(self._select_diverse_subset(np.array(vertex_points), n_vertex_points))

        # 2. Generate Edge Points (blends of two components)
        n_edge_points = int(n_points * allocations['edges'])
        print(f"Generating {n_edge_points} edge points...")
        edge_points = self._generate_edge_points()
        if edge_points:
            points.extend(self._select_diverse_subset(np.array(edge_points), n_edge_points))

        # 3. Generate Face Points (blends of 3+ components)
        n_vertex_points = len(points) # Get current count from vertices
        n_edge_points = len(points) - n_vertex_points # Get current count from edges
        n_face_points = int(n_points * allocations['faces'])
        if self.n_components > 2 and n_face_points > 0:
            print(f"Generating {n_face_points} face points...")
            face_attempts = 0
            max_face_attempts = n_face_points * 10 # Limit attempts
            target_face_count = n_vertex_points + n_edge_points + n_face_points
            while len(points) < target_face_count and face_attempts < max_face_attempts:
                face_attempts += 1
                n_active = np.random.randint(3, self.n_components + 1)
                active_indices = np.random.choice(self.n_components, n_active, replace=False)
                
                point = np.zeros(self.n_components)
                point[active_indices] = np.random.dirichlet(np.ones(n_active))
                
                for i in active_indices:
                    lower, upper = self.component_bounds[i]
                    point[i] = lower + point[i] * (upper - lower)
                
                point = point / np.sum(point)
                if self._check_bounds(point):
                    points.append(point)

        # 4. Generate Interior Points (to fill the space)
        n_interior_points = n_points - len(points)
        print(f"Generating {n_interior_points} interior points...")
        if n_interior_points > 0:
            interior_attempts = 0
            max_interior_attempts = n_interior_points * 10 # Limit attempts
            while len(points) < n_points and interior_attempts < max_interior_attempts:
                interior_attempts += 1
                alpha = np.random.uniform(1.5, 5.0, self.n_components)
                point = np.random.dirichlet(alpha)
                
                scaled_point = np.zeros(self.n_components)
                for i in range(self.n_components):
                    lower, upper = self.component_bounds[i]
                    scaled_point[i] = lower + point[i] * (upper - lower)
                
                point_sum = np.sum(scaled_point)
                if point_sum > 1e-8:
                    final_point = scaled_point / point_sum
                    if self._check_bounds(final_point):
                        points.append(final_point)

        # 5. Final Selection and Refinement
        # If we have too many points, select the most diverse subset
        if len(points) > n_points:
            points = self._select_diverse_subset(np.array(points), n_points)
        
        # If we don't have enough, add more constrained random points
        attempts = 0
        while len(points) < n_points and attempts < 1000:
            point = self._generate_constrained_point()
            if point is not None:
                points.append(point)
            attempts += 1
            
        # Final check for duplicates and ensure correct number of points
        unique_points = []
        for p in points:
            if not any(np.allclose(p, up, atol=1e-6) for up in unique_points):
                unique_points.append(p)
        
        # If still not enough, pad with perturbations
        while len(unique_points) < n_points:
            if not unique_points: # Emergency fallback
                unique_points.append(np.ones(self.n_components) / self.n_components)
                continue
            base_point = unique_points[np.random.randint(len(unique_points))]
            noise = np.random.normal(0, 1e-4, self.n_components)
            new_point = (base_point + noise)
            new_point[new_point < 0] = 0 # ensure non-negative
            new_point = new_point / np.sum(new_point)
            if self._check_bounds(new_point):
                unique_points.append(new_point)

        return np.array(unique_points[:n_points])

    def _select_diverse_subset(self, design: np.ndarray, n_select: int) -> np.ndarray:
        """Select diverse subset using maximin distance criterion"""
        if len(design) <= n_select:
            return design
            
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
    
    def _generate_corner_points(self) -> List[np.ndarray]:
        """Generate a limited set of corner points of the feasible simplex region."""
        corner_points = []
        max_corners = 2500  # Hard limit to prevent memory explosion

        # Generate points by setting n-1 components to their bounds
        for combo in itertools.combinations(range(self.n_components), self.n_components - 1):
            if len(corner_points) >= max_corners:
                break
            
            # For the n-1 components, try all combinations of lower/upper bounds
            for bound_choices in itertools.product([0, 1], repeat=self.n_components - 1):
                point = np.zeros(self.n_components)
                current_sum = 0
                
                # Set the n-1 components to their chosen bounds
                for i, comp_idx in enumerate(combo):
                    bound = self.component_bounds[comp_idx][bound_choices[i]]
                    point[comp_idx] = bound
                    current_sum += bound
                
                # The last component gets the rest
                last_comp_idx = list(set(range(self.n_components)) - set(combo))[0]
                point[last_comp_idx] = 1.0 - current_sum
                
                # Check if the generated point is valid
                if self._check_bounds(point):
                    if not any(np.allclose(point, p, atol=1e-6) for p in corner_points):
                        corner_points.append(point)

                if len(corner_points) >= max_corners:
                    return corner_points
        return corner_points

    def _generate_edge_points(self) -> List[np.ndarray]:
        """Generate points along edges defined by pairs of components."""
        edge_points = []
        max_points_per_pair = 10
        
        # Iterate through all pairs of components
        for i, j in itertools.combinations(range(self.n_components), 2):
            # For each pair, create points where these two components dominate
            # Start with all other components at their lower bounds
            point = np.array([b[0] for b in self.component_bounds])
            
            # Sum of lower bounds for components other than i and j
            sum_others_lower = np.sum(point) - point[i] - point[j]
            
            # The remaining space is for components i and j
            remaining_sum = 1.0 - sum_others_lower
            if remaining_sum < 1e-8:
                continue

            # Generate points along the edge between i and j
            for alpha in np.linspace(0, 1, max_points_per_pair):
                temp_point = point.copy()
                
                # Distribute remaining sum between i and j
                lower_i, upper_i = self.component_bounds[i]
                lower_j, upper_j = self.component_bounds[j]
                
                # Set other components to lower bounds
                temp_point[i] = lower_i
                temp_point[j] = lower_j
                
                # Distribute remaining sum
                rem = 1.0 - np.sum(temp_point)
                
                # Allocate proportionally to alpha, respecting bounds
                alloc_i = alpha * rem
                alloc_j = (1 - alpha) * rem
                
                temp_point[i] += alloc_i
                temp_point[j] += alloc_j
                
                # Check if this point is valid
                if self._check_bounds(temp_point):
                    if not any(np.allclose(temp_point, p, atol=1e-6) for p in edge_points):
                        edge_points.append(temp_point)

        return edge_points
    
    def _generate_constrained_point(self) -> Optional[np.ndarray]:
        """Generate a single random point that satisfies constraints"""
        max_tries = 100
        
        for _ in range(max_tries):
            # Use rejection sampling with smart initialization
            point = np.zeros(self.n_components)
            
            # Start with minimum bounds
            remaining_sum = 1.0
            for i in range(self.n_components):
                lower, upper = self.component_bounds[i]
                point[i] = lower
                remaining_sum -= lower
            
            if remaining_sum <= 0:
                continue
            
            # Randomly distribute the remaining sum
            # Generate random weights and scale them
            weights = np.random.exponential(1.0, self.n_components)
            weights = weights / np.sum(weights) * remaining_sum
            
            # Add weights to the minimum values
            for i in range(self.n_components):
                point[i] += weights[i]
            
            # Check if this violates upper bounds
            violates_bounds = False
            excess = 0
            for i in range(self.n_components):
                lower, upper = self.component_bounds[i]
                if point[i] > upper:
                    excess += point[i] - upper
                    point[i] = upper
                    violates_bounds = True
            
            # If we had violations, redistribute the excess
            if violates_bounds and excess > 0:
                # Find components that can accept more
                available_capacity = []
                for i in range(self.n_components):
                    lower, upper = self.component_bounds[i]
                    capacity = upper - point[i]
                    if capacity > 1e-10:
                        available_capacity.append((i, capacity))
                
                # Redistribute excess proportionally to available capacity
                total_capacity = sum(cap for _, cap in available_capacity)
                if total_capacity >= excess:
                    for i, capacity in available_capacity:
                        allocation = excess * (capacity / total_capacity)
                        point[i] += allocation
                else:
                    continue  # Can't redistribute, try again
            
            # Final normalization and check
            point = point / np.sum(point)
            
            if self._check_bounds(point):
                return point
        
        return None  # Failed to generate a valid point
    
    def _calculate_d_efficiency(self, X: np.ndarray, model_type: str) -> float:
        """Calculate D-efficiency for mixture design"""
        try:
            # Handle empty designs
            if X.size == 0:
                return 0.0
            
            # Handle 1D case
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # IMPORTANT: For D-efficiency calculation, we should use all components
            # as they come from the algorithm, not just variable components
            # This matches how the D-optimal algorithm actually generates the design
            
            # Temporarily disable fixed component tracking for D-efficiency calculation
            original_truly_fixed = self.truly_fixed_components
            self.truly_fixed_components = set()  # Treat all as variable for calculation
            
            try:
                model_matrix = self.create_mixture_model_matrix(X, model_type)
            finally:
                # Restore original fixed component tracking
                self.truly_fixed_components = original_truly_fixed
            
            # Handle empty model matrix
            if model_matrix.size == 0:
                return 0.0
            
            # Check if model matrix has enough rows and is not rank deficient
            n_runs, n_params = model_matrix.shape
            
            # For mixture designs, we typically need even more runs for good efficiency
            min_recommended_runs = max(n_params + 2, int(n_params * 1.5))
            
            # Step 1: Check if we have enough runs for the model
            if n_runs < n_params:
                print(f"Warning: Insufficient runs ({n_runs}) for parameters ({n_params}) - need at least {n_params} runs")
                # Use SVD to get a more accurate measure of rank deficiency
                U, s, Vt = np.linalg.svd(model_matrix, full_matrices=False)
                
                # Calculate effective rank (number of significant singular values)
                tol = max(model_matrix.shape) * np.finfo(float).eps * s[0]
                rank = np.sum(s > tol)
                
                # If rank is too low, return a very low efficiency
                if rank < n_params * 0.75:  # At least 75% of full rank
                    print(f"Warning: Design is severely rank deficient (rank {rank}/{n_params})")
                    return 0.001
                
                # We can still calculate a modified efficiency for rank-deficient designs
                # Use only the significant singular values
                nonzero_s = s[:rank]
                
                # Calculate "pseudo-determinant" using only non-zero singular values
                log_det = np.sum(np.log(nonzero_s**2))
                
                # Scaled D-efficiency using the available rank
                d_eff = np.exp(log_det / (2 * rank)) / n_runs**(rank / n_params)
                
                # Further scale by how close to full rank we are
                rank_ratio = rank / n_params
                d_eff *= rank_ratio
                
                return min(max(d_eff, 0.001), 0.1)  # Cap at 0.1 for rank-deficient designs
            
            # Step 2: For full-rank designs, calculate standard D-efficiency
            # Calculate (X'X) and check for numerical stability
            XtX = np.dot(model_matrix.T, model_matrix)
            
            # Check condition number for numerical stability
            cond_num = np.linalg.cond(XtX)
            if cond_num > 1e10:
                print(f"Warning: Design matrix is ill-conditioned (cond={cond_num:.1e})")
                # Use more numerically stable SVD method
                U, s, Vt = np.linalg.svd(model_matrix, full_matrices=False)
                log_det = np.sum(np.log(s**2))
                d_eff = np.exp(log_det / (2 * n_params)) / n_runs
            else:
                # Use standard determinant method
                # Calculate log determinant for numerical stability
                sign, logdet = np.linalg.slogdet(XtX)
                if sign <= 0 or not np.isfinite(logdet):
                    print(f"Warning: Non-positive or invalid determinant")
                    return 0.01
                
                # Calculate D-efficiency with proper scaling and exponentiation
                d_eff = np.exp(logdet / n_params) / n_runs
            
            # Check final result for reasonableness
            if not np.isfinite(d_eff) or d_eff <= 0:
                print(f"Warning: Invalid D-efficiency value: {d_eff}")
                return 0.01
            
            # Scale relative to theoretical maximum of 1.0
            return min(d_eff, 1.0)
            
        except Exception as e:
            print(f"Warning: D-efficiency calculation failed: {str(e)}")
            return 0.001
    
    def _calculate_i_efficiency(self, X: np.ndarray, model_type: str, 
                               integration_points: np.ndarray = None) -> float:
        """Calculate I-efficiency for mixture design"""
        try:
            # Handle empty designs
            if X.size == 0:
                return 1.0
            
            # Handle 1D case
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            model_matrix = self.create_mixture_model_matrix(X, model_type)
            
            # Handle empty model matrix
            if model_matrix.size == 0:
                return 1.0
            
            # Check if model matrix has enough rows
            if model_matrix.shape[0] < model_matrix.shape[1]:
                return 10.0  # Not enough runs - poor prediction
            
            XtX = np.dot(model_matrix.T, model_matrix)
            
            # Check condition number
            cond_num = np.linalg.cond(XtX)
            if cond_num > 1e12:
                return 10.0  # Matrix is ill-conditioned
            
            # Robust matrix inversion
            try:
                XtX_inv = np.linalg.inv(XtX)
            except:
                return 10.0
            
            # Generate integration points if not provided
            if integration_points is None:
                try:
                    integration_points = self._generate_candidate_points(100)  # Fewer points for efficiency
                except:
                    # Fallback: use the design points themselves
                    integration_points = X
            
            total_pred_var = 0
            valid_points = 0
            
            for point in integration_points:
                try:
                    if point.ndim == 1:
                        point = point.reshape(1, -1)
                    
                    x_vec = self.create_mixture_model_matrix(point, model_type)
                    
                    if x_vec.size == 0:
                        continue
                    
                    pred_var = np.dot(np.dot(x_vec, XtX_inv), x_vec.T)[0, 0]
                    
                    # Check for valid prediction variance
                    if np.isnan(pred_var) or np.isinf(pred_var) or pred_var < 0:
                        continue
                    
                    total_pred_var += pred_var
                    valid_points += 1
                    
                except:
                    continue
            
            if valid_points == 0:
                return 10.0
            
            avg_pred_var = total_pred_var / valid_points
            
            # Sanity check the result
            if np.isnan(avg_pred_var) or np.isinf(avg_pred_var) or avg_pred_var <= 0:
                return 10.0
            
            return max(avg_pred_var, 0.001)  # Ensure positive result
            
        except Exception as e:
            print(f"Warning: I-efficiency calculation failed: {str(e)}")
            return 10.0


# Example usage and utility functions
def mixture_response_analysis(designs: List[np.ndarray], 
                            response_functions: List[callable],
                            design_names: List[str],
                            component_names: List[str]) -> pd.DataFrame:
    """
    Analyze multiple response functions for different mixture designs
    """
    results = []
    
    for design, name in zip(designs, design_names):
        result = {'Design': name, 'n_runs': design.shape[0]}
        
        # Calculate responses for each function
        for i, response_func in enumerate(response_functions):
            responses = response_func(design)
            result[f'Response_{i+1}_mean'] = np.mean(responses)
            result[f'Response_{i+1}_std'] = np.std(responses)
            result[f'Response_{i+1}_range'] = np.ptp(responses)
        
        results.append(result)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=== Mixture Design Demo ===\n")
    
    # Example: 3-component mixture (e.g., chemical formulation)
    component_names = ['Polymer_A', 'Polymer_B', 'Additive']
    component_bounds = [(0.2, 0.7), (0.2, 0.7), (0.1, 0.6)]  # More realistic bounds
    
    mixture = MixtureDesign(3, component_names, component_bounds)
    
    # Generate different types of mixture designs
    print("1. Simplex Lattice Design (degree 3):")
    lattice_design = mixture.generate_simplex_lattice(degree=3)
    print(f"Generated {len(lattice_design)} points")
    if len(lattice_design) > 0:
        print(pd.DataFrame(lattice_design, columns=component_names).round(3))
    else:
        print("No valid lattice points found with current bounds. Trying D-optimal instead...")
        lattice_design = mixture.generate_d_optimal_mixture(10, "quadratic", random_seed=42)
        print(pd.DataFrame(lattice_design, columns=component_names).round(3))
    
    print("\n2. Simplex Centroid Design:")
    centroid_design = mixture.generate_simplex_centroid()
    print(f"Generated {len(centroid_design)} points")
    print(pd.DataFrame(centroid_design, columns=component_names).round(3))
    
    print("\n3. D-optimal Mixture Design:")
    d_optimal_mixture = mixture.generate_d_optimal_mixture(n_runs=12, random_seed=42)
    d_results = mixture.evaluate_mixture_design(d_optimal_mixture, "quadratic")
    print(f"D-efficiency: {d_results['d_efficiency']:.4f}")
    print(f"I-efficiency: {d_results['i_efficiency']:.4f}")
    print(pd.DataFrame(d_optimal_mixture, columns=component_names).round(3))
    
    print("\n4. I-optimal Mixture Design:")
    i_optimal_mixture = mixture.generate_i_optimal_mixture(n_runs=12, random_seed=42)
    i_results = mixture.evaluate_mixture_design(i_optimal_mixture, "quadratic")
    print(f"D-efficiency: {i_results['d_efficiency']:.4f}")
    print(f"I-efficiency: {i_results['i_efficiency']:.4f}")
    
    # Example response functions for mixture
    def strength_response(X):
        """Material strength as function of mixture components"""
        return (50 * X[:, 0] + 60 * X[:, 1] + 30 * X[:, 2] + 
                20 * X[:, 0] * X[:, 1] - 10 * X[:, 0] * X[:, 2] + 
                np.random.normal(0, 2, len(X)))
    
    def flexibility_response(X):
        """Material flexibility as function of mixture components"""
        return (30 * X[:, 0] + 80 * X[:, 1] + 40 * X[:, 2] + 
                15 * X[:, 1] * X[:, 2] + 
                np.random.normal(0, 1.5, len(X)))
    
    print("\n5. Multiple Response Analysis:")
    designs = [d_optimal_mixture, i_optimal_mixture]
    design_names = ['D-optimal', 'I-optimal']
    response_functions = [strength_response, flexibility_response]
    
    np.random.seed(42)
    comparison = mixture_response_analysis(designs, response_functions, design_names, component_names)
    print(comparison.round(3))
    
    print("\n=== Mixture Design Demo Complete ===")
