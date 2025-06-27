"""
Mixture Design Algorithms Implementation
Contains implementations of various mixture design algorithms:
- Simplex Lattice
- Simplex Centroid
- Extreme Vertices
"""

import numpy as np
import itertools
from typing import List, Tuple, Dict, Optional, Union

from mixture_utils import (
    check_bounds,
    generate_emphasis_point,
    generate_boundary_biased_point,
    generate_binary_mixture
)

def generate_simplex_lattice(n_components: int, degree: int, 
                           component_bounds: List[Tuple[float, float]],
                           fixed_components: Dict[str, float] = None,
                           component_names: List[str] = None) -> np.ndarray:
    """
    Generate simplex lattice design with robust error handling
    
    Parameters:
    -----------
    n_components : int
        Number of components
    degree : int
        Degree of lattice (2 for {0, 1/2, 1}, 3 for {0, 1/3, 2/3, 1}, etc.)
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
    fixed_components : Dict[str, float], optional
        Dictionary of fixed component names and values
    component_names : List[str], optional
        List of component names
        
    Returns:
    --------
    np.ndarray : Design matrix
    """
    # Handle fixed components: generate lattice only for variable components
    if fixed_components:
        if component_names is None:
            component_names = [f'Comp_{i+1}' for i in range(n_components)]
            
        # Get indices of variable components
        all_fixed_component_names = set(fixed_components.keys())
        
        variable_indices = []
        for i, name in enumerate(component_names):
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
            full_design = np.zeros((len(unique_variable_points), n_components))
            
            # Fill in variable component values
            for row_idx, var_point in enumerate(unique_variable_points):
                for var_idx, comp_idx in enumerate(variable_indices):
                    full_design[row_idx, comp_idx] = var_point[var_idx]
            
            # Adjust for fixed components
            full_design = _adjust_for_fixed_components(full_design, fixed_components, component_names)
            return full_design
        else:
            print("No valid variable lattice points found, trying fallback...")
    
    # No fixed components - use improved algorithm for all components
    # Try to generate lattice points with better distribution
    points = []
    
    # Generate standard lattice points
    standard_points = _generate_standard_lattice_points(n_components, degree)
    
    # Check which standard points satisfy bounds
    for point in standard_points:
        if check_bounds(point, component_bounds):
            points.append(point)
    
    # If we have enough points, return them
    if len(points) >= n_components + 1:  # Minimum needed for a simplex
        return np.array(points)
    
    # If not enough standard points, try adaptive approach
    print(f"Only {len(points)} valid standard lattice points found. Trying adaptive approach...")
    
    # Try to generate adaptive lattice points based on bounds
    adaptive_points = _generate_adaptive_lattice_points(n_components, degree, component_bounds)
    
    # Combine with any valid standard points
    all_points = points + adaptive_points
    
    # Remove duplicates
    unique_points = []
    for point in all_points:
        if not any(np.allclose(point, p, atol=1e-6) for p in unique_points):
            unique_points.append(point)
    
    if len(unique_points) > 0:
        return np.array(unique_points)
    
    # If no valid lattice points found with current bounds, use fallback strategies
    print(f"Warning: No valid lattice points found with degree {degree}. Trying fallback strategies...")
    
    # Strategy 1: Try lower degree
    if degree > 2:
        print(f"Trying lower degree ({degree - 1})...")
        return generate_simplex_lattice(n_components, degree - 1, component_bounds, fixed_components, component_names)
    
    # Strategy 2: Use extreme vertices if available
    try:
        print("Trying extreme vertices design...")
        vertices = generate_extreme_vertices(n_components, component_bounds, fixed_components, component_names)
        if len(vertices) > 0:
            return vertices
    except Exception as e:
        print(f"Error generating extreme vertices: {e}")
    
    # Strategy 3: Use simplex centroid if available
    try:
        print("Trying simplex centroid design...")
        centroid = generate_simplex_centroid(n_components, component_bounds, fixed_components, component_names)
        if len(centroid) > 0:
            return centroid
    except Exception as e:
        print(f"Error generating simplex centroid: {e}")
    
    # Strategy 4: Generate emergency feasible points
    print("Generating custom feasible points...")
    feasible_points = _generate_emergency_feasible_points(n_components, component_bounds)
    if len(feasible_points) > 0:
        return np.array(feasible_points)
    
    # If all fails, return empty array but with better error message
    raise ValueError(
        f"Unable to generate simplex lattice design with degree {degree}. "
        f"The component bounds appear to be too restrictive for this design method. "
        f"Suggestions: 1) Try D-optimal or I-optimal methods instead, "
        f"2) Relax component bounds slightly, or 3) Use Sequential Mixture Design."
    )

def _generate_standard_lattice_points(n_components: int, degree: int) -> List[List[float]]:
    """
    Generate standard lattice points for the given degree
    
    Parameters:
    -----------
    n_components : int
        Number of components
    degree : int
        Degree of lattice
        
    Returns:
    --------
    List[List[float]] : List of lattice points
    """
    points = []
    
    # Generate all combinations that sum to degree
    for combination in itertools.combinations_with_replacement(range(degree + 1), n_components):
        if sum(combination) == degree:
            point = [x / degree for x in combination]
            points.append(point)
    
    # Add permutations for non-symmetric points
    unique_points = []
    for point in points:
        perms = list(set(itertools.permutations(point)))
        for perm in perms:
            if list(perm) not in unique_points:
                unique_points.append(list(perm))
    
    # Print debug information
    print(f"Generated {len(unique_points)} standard lattice points for degree {degree}")
    print(f"First few points: {[np.round(p, 4) for p in unique_points[:3]]}")
    
    return unique_points

def _generate_adaptive_lattice_points(n_components: int, degree: int, 
                                    component_bounds: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Generate lattice points adapted to the component bounds
    
    Parameters:
    -----------
    n_components : int
        Number of components
    degree : int
        Degree of lattice
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
        
    Returns:
    --------
    List[List[float]] : List of adaptive lattice points
    """
    adaptive_points = []
    
    # Calculate midpoints and ranges for each component
    midpoints = [(lb + ub) / 2 for lb, ub in component_bounds]
    ranges = [ub - lb for lb, ub in component_bounds]
    
    # Add centroid first (always valid for mixture designs)
    centroid = [1.0 / n_components] * n_components
    if check_bounds(centroid, component_bounds):
        adaptive_points.append(centroid)
        print(f"Added centroid point: {np.round(centroid, 4)}")
    else:
        # If centroid doesn't satisfy bounds, create a valid centroid-like point
        # by using the midpoint of each component's bounds
        adjusted_centroid = midpoints.copy()
        # Normalize to sum to 1
        centroid_sum = sum(adjusted_centroid)
        if centroid_sum > 0:
            adjusted_centroid = [m / centroid_sum for m in adjusted_centroid]
            if check_bounds(adjusted_centroid, component_bounds):
                adaptive_points.append(adjusted_centroid)
                print(f"Added adjusted centroid: {np.round(adjusted_centroid, 4)}")
        
        # If adjusted centroid still doesn't work, try a weighted centroid
        if len(adaptive_points) == 0:
            # Weight by upper bounds to favor components with higher limits
            weighted_centroid = [component_bounds[i][1] for i in range(n_components)]
            centroid_sum = sum(weighted_centroid)
            if centroid_sum > 0:
                weighted_centroid = [w / centroid_sum for w in weighted_centroid]
                if check_bounds(weighted_centroid, component_bounds):
                    adaptive_points.append(weighted_centroid)
                    print(f"Added weighted centroid: {np.round(weighted_centroid, 4)}")
    
    # Check if we have restrictive bounds (all max bounds < 1.0)
    restrictive_bounds = all(bound[1] < 1.0 for bound in component_bounds)
    
    if restrictive_bounds:
        print("Detected restrictive bounds. Using specialized point generation...")
        
        # EMERGENCY APPROACH: Generate many random points and keep valid ones
        print("Using emergency random point generation for restrictive bounds...")
        
        # First, try to add the centroid with small perturbations
        for _ in range(10):
            # Create small random perturbations around centroid
            perturbation = np.random.uniform(-0.05, 0.05, n_components)
            point = np.array(centroid) + perturbation
            
            # Ensure non-negative
            point = np.maximum(point, 0)
            
            # Normalize
            point_sum = np.sum(point)
            if point_sum > 0:
                normalized_point = point / point_sum
                
                # Check if valid
                if check_bounds(normalized_point.tolist(), component_bounds):
                    adaptive_points.append(normalized_point.tolist())
                    print(f"Added perturbed centroid: {np.round(normalized_point, 4)}")
        
        # Generate points by perturbing the centroid in valid directions
        for i in range(n_components):
            for j in range(n_components):
                if i != j:
                    # Try different perturbation levels
                    for perturb in [0.02, 0.05, 0.08, 0.1, 0.15]:
                        # Start with centroid
                        point = centroid.copy()
                        
                        # Increase component i and decrease component j
                        point[i] = min(point[i] + perturb, component_bounds[i][1])
                        point[j] = max(point[j] - perturb, component_bounds[j][0])
                        
                        # Normalize to ensure sum = 1
                        point_sum = sum(point)
                        if point_sum > 0:
                            normalized_point = [p / point_sum for p in point]
                            
                            # Check if point satisfies bounds
                            if check_bounds(normalized_point, component_bounds):
                                adaptive_points.append(normalized_point)
                                print(f"Added perturbed point: {np.round(normalized_point, 4)}")
        
        # Generate points at corners of the feasible region
        for i in range(n_components):
            # Try to maximize each component within bounds
            point = [component_bounds[j][0] for j in range(n_components)]  # Start with all minimums
            point[i] = component_bounds[i][1]  # Set component i to maximum
            
            # Normalize to ensure sum = 1
            point_sum = sum(point)
            if point_sum > 0:
                normalized_point = [p / point_sum for p in point]
                
                # Check if point satisfies bounds
                if check_bounds(normalized_point, component_bounds):
                    adaptive_points.append(normalized_point)
                    print(f"Added corner point for component {i+1}: {np.round(normalized_point, 4)}")
        
        # Try binary mixtures with different ratios
        for i in range(n_components):
            for j in range(i+1, n_components):
                for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    point = [0.0] * n_components
                    point[i] = ratio
                    point[j] = 1.0 - ratio
                    
                    if check_bounds(point, component_bounds):
                        adaptive_points.append(point)
                        print(f"Added binary mixture {i+1}-{j+1} ({ratio:.1f}): {np.round(point, 4)}")
                    
                    # Also try points with minimum values for other components
                    point_with_mins = np.array([component_bounds[k][0] for k in range(n_components)])
                    remaining = 1.0 - sum(point_with_mins)
                    if remaining > 0:
                        point_with_mins[i] += remaining * ratio
                        point_with_mins[j] += remaining * (1.0 - ratio)
                        
                        # Normalize
                        point_with_mins = point_with_mins / np.sum(point_with_mins)
                        
                        if check_bounds(point_with_mins, component_bounds):
                            adaptive_points.append(point_with_mins)
                            print(f"Added constrained binary mixture {i+1}-{j+1} ({ratio:.1f}): {np.round(point_with_mins, 4)}")
    else:
        # Standard approach for non-restrictive bounds
        # Generate points at different levels within the feasible region
        for level in range(1, degree + 1):
            level_fraction = level / degree
            
            # Try different combinations of components at this level
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    # Create a point with components i and j at this level
                    point = midpoints.copy()  # Start with midpoints
                    
                    # Adjust components i and j
                    point[i] = component_bounds[i][0] + level_fraction * ranges[i]
                    point[j] = component_bounds[j][0] + (1 - level_fraction) * ranges[j]
                    
                    # Normalize to ensure sum = 1
                    point_sum = sum(point)
                    if point_sum > 0:
                        normalized_point = [p / point_sum for p in point]
                        
                        # Check if point satisfies bounds
                        if check_bounds(normalized_point, component_bounds):
                            adaptive_points.append(normalized_point)
        
        # Add pure components (vertices) if they satisfy bounds
        for i in range(n_components):
            vertex = [0.0] * n_components
            vertex[i] = 1.0
            if check_bounds(vertex, component_bounds):
                adaptive_points.append(vertex)
                print(f"Added vertex {i+1}: {vertex}")
        
        # Add binary blends at different ratios
        for i in range(n_components):
            for j in range(i + 1, n_components):
                for ratio in [0.25, 0.5, 0.75]:
                    binary = [0.0] * n_components
                    binary[i] = ratio
                    binary[j] = 1.0 - ratio
                    if check_bounds(binary, component_bounds):
                        adaptive_points.append(binary)
    
    # Generate additional random points within bounds
    n_random = max(10, degree * n_components)
    for _ in range(n_random):
        # Generate random weights within bounds
        weights = np.zeros(n_components)
        for i in range(n_components):
            lb, ub = component_bounds[i]
            weights[i] = np.random.uniform(lb, ub)
        
        # Normalize to sum to 1
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            normalized_weights = weights / weights_sum
            
            # Check if point satisfies bounds
            if check_bounds(normalized_weights, component_bounds):
                adaptive_points.append(normalized_weights.tolist())
    
    # Remove duplicates
    unique_adaptive_points = []
    for point in adaptive_points:
        if not any(np.allclose(point, p, atol=1e-6) for p in unique_adaptive_points):
            unique_adaptive_points.append(point)
    
    print(f"Generated {len(unique_adaptive_points)} unique adaptive points")
    return unique_adaptive_points

def _generate_emergency_feasible_points(n_components: int, 
                                      component_bounds: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Generate basic feasible mixture points as emergency fallback
    
    Parameters:
    -----------
    n_components : int
        Number of components
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
        
    Returns:
    --------
    List[List[float]] : List of feasible points
    """
    feasible_points = []
    
    # Try component vertices (one component at max, others at min)
    for i in range(n_components):
        point = [0.0] * n_components
        
        # Set component i to maximum allowed
        lower_i, upper_i = component_bounds[i]
        point[i] = upper_i
        
        # Calculate remaining sum for other components
        remaining = 1.0 - upper_i
        other_indices = [j for j in range(n_components) if j != i]
        
        if remaining > 0 and len(other_indices) > 0:
            # Distribute remaining among other components at their minimums
            min_sum_others = sum(component_bounds[j][0] for j in other_indices)
            
            if remaining >= min_sum_others:
                # Set others to minimum first
                for j in other_indices:
                    point[j] = component_bounds[j][0]
                
                # Distribute any extra remaining
                current_sum = sum(point)
                if current_sum < 1.0:
                    extra = 1.0 - current_sum
                    # Add extra proportionally to other components (within their bounds)
                    for j in other_indices:
                        lower_j, upper_j = component_bounds[j]
                        available_j = upper_j - point[j]
                        if available_j > 0:
                            addition = min(extra / len(other_indices), available_j)
                            point[j] += addition
                            extra -= addition
                
                # Normalize to ensure exact sum of 1
                point_sum = sum(point)
                if point_sum > 0:
                    point = [x / point_sum for x in point]
                    
                    if check_bounds(point, component_bounds):
                        feasible_points.append(point)
    
    # Try centroid if not already added
    centroid = [1.0 / n_components] * n_components
    if check_bounds(centroid, component_bounds):
        feasible_points.append(centroid)
    
    # Try equal distribution at average bounds
    try:
        avg_point = []
        for i in range(n_components):
            lower, upper = component_bounds[i]
            avg_point.append((lower + upper) / 2)
        
        # Normalize
        point_sum = sum(avg_point)
        if point_sum > 0:
            avg_point = [x / point_sum for x in avg_point]
            if check_bounds(avg_point, component_bounds):
                feasible_points.append(avg_point)
    except:
        pass
    
    return feasible_points

def generate_simplex_centroid(n_components: int, 
                            component_bounds: List[Tuple[float, float]],
                            fixed_components: Dict[str, float] = None,
                            component_names: List[str] = None) -> np.ndarray:
    """
    Generate simplex centroid design
    
    Parameters:
    -----------
    n_components : int
        Number of components
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
    fixed_components : Dict[str, float], optional
        Dictionary of fixed component names and values
    component_names : List[str], optional
        List of component names
        
    Returns:
    --------
    np.ndarray : Design matrix
    """
    # Handle fixed components
    if fixed_components:
        if component_names is None:
            component_names = [f'Comp_{i+1}' for i in range(n_components)]
            
        # Get indices of variable components
        all_fixed_component_names = set(fixed_components.keys())
        
        variable_indices = []
        for i, name in enumerate(component_names):
            if name not in all_fixed_component_names:
                variable_indices.append(i)
        
        n_variable_components = len(variable_indices)
        print(f"Generating centroid for {n_variable_components} variable components")
        
        # Generate centroid points for variable components only
        variable_points = generate_simplex_centroid(
            n_variable_components, 
            [component_bounds[i] for i in variable_indices],
            None,
            [component_names[i] for i in variable_indices]
        )
        
        # Convert variable component points to full design matrix
        if len(variable_points) > 0:
            # Create full design matrix with all components
            full_design = np.zeros((len(variable_points), n_components))
            
            # Fill in variable component values
            for row_idx, var_point in enumerate(variable_points):
                for var_idx, comp_idx in enumerate(variable_indices):
                    full_design[row_idx, comp_idx] = var_point[var_idx]
            
            # Adjust for fixed components
            full_design = _adjust_for_fixed_components(full_design, fixed_components, component_names)
            return full_design
        else:
            print("No valid variable centroid points found, trying fallback...")
    
    points = []
    
    # Single component vertices (if bounds allow)
    for i in range(n_components):
        point = [0.0] * n_components
        point[i] = 1.0
        if check_bounds(point, component_bounds):
            points.append(point)
            print(f"Added vertex {i+1} to simplex centroid design")
    
    # Overall centroid
    centroid = [1.0 / n_components] * n_components
    if check_bounds(centroid, component_bounds):
        points.append(centroid)
        print(f"Added centroid to simplex centroid design: {np.round(centroid, 4)}")
    
    # Centroids of faces (if n_components > 2)
    if n_components > 2:
        for r in range(2, n_components):
            for indices in itertools.combinations(range(n_components), r):
                point = [0.0] * n_components
                for idx in indices:
                    point[idx] = 1.0 / r
                if check_bounds(point, component_bounds):
                    points.append(point)
                    print(f"Added {r}-component centroid to simplex centroid design")
    
    # Check if we have restrictive bounds (all max bounds < 1.0)
    restrictive_bounds = all(bound[1] < 1.0 for bound in component_bounds)
    
    # If no valid points found or restrictive bounds, use emergency approach
    if len(points) < 3 or restrictive_bounds:
        print("Using emergency point generation for simplex centroid design...")
        
        # EMERGENCY APPROACH: Generate many random points and keep valid ones
        # First, try to add the centroid with small perturbations
        if centroid in points:
            for _ in range(10):
                # Create small random perturbations around centroid
                perturbation = np.random.uniform(-0.05, 0.05, n_components)
                point = np.array(centroid) + perturbation
                
                # Ensure non-negative
                point = np.maximum(point, 0)
                
                # Normalize
                point_sum = np.sum(point)
                if point_sum > 0:
                    normalized_point = point / point_sum
                    
                    # Check if valid
                    if check_bounds(normalized_point.tolist(), component_bounds):
                        points.append(normalized_point.tolist())
                        print(f"Added perturbed centroid: {np.round(normalized_point, 4)}")
        
        # Try binary mixtures with different ratios
        for i in range(n_components):
            for j in range(i+1, n_components):
                for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    point = [0.0] * n_components
                    point[i] = ratio
                    point[j] = 1.0 - ratio
                    
                    if check_bounds(point, component_bounds):
                        points.append(point)
                        print(f"Added binary mixture {i+1}-{j+1} ({ratio:.1f}): {np.round(point, 4)}")
        
        # Try to maximize each component within bounds
        for i in range(n_components):
            point = [component_bounds[j][0] for j in range(n_components)]  # Start with all minimums
            point[i] = component_bounds[i][1]  # Set component i to maximum
            
            # Normalize to ensure sum = 1
            point_sum = sum(point)
            if point_sum > 0:
                normalized_point = [p / point_sum for p in point]
                
                # Check if point satisfies bounds
                if check_bounds(normalized_point, component_bounds):
                    points.append(normalized_point)
                    print(f"Added max-component point for {i+1}: {np.round(normalized_point, 4)}")
        
        # Generate completely random points (last resort)
        if len(points) < 3:
            n_random = 1000  # Try many random points
            print(f"Generating {n_random} random points for simplex centroid design...")
            
            for _ in range(n_random):
                # Generate random weights
                weights = np.random.random(n_components)
                
                # Normalize to sum to 1
                point = weights / np.sum(weights)
                
                # Check if point satisfies bounds
                if check_bounds(point.tolist(), component_bounds):
                    points.append(point.tolist())
            
            # If we still don't have enough points, try relaxing bounds slightly
            if len(points) < 3:
                print("Still not enough points. Trying with slightly relaxed bounds...")
                
                # Temporarily relax bounds by 10%
                original_bounds = component_bounds.copy()
                relaxed_bounds = []
                
                for lower, upper in component_bounds:
                    range_val = upper - lower
                    new_lower = max(0, lower - range_val * 0.1)
                    new_upper = min(1, upper + range_val * 0.1)
                    relaxed_bounds.append((new_lower, new_upper))
                
                # Generate random points with relaxed bounds
                for _ in range(n_random):
                    weights = np.random.random(n_components)
                    point = weights / np.sum(weights)
                    
                    if check_bounds(point.tolist(), relaxed_bounds):
                        # Check if point is close to satisfying original bounds
                        close_enough = True
                        for i, (lower, upper) in enumerate(original_bounds):
                            if point[i] < lower - 0.05 or point[i] > upper + 0.05:
                                close_enough = False
                                break
                        
                        if close_enough:
                            points.append(point.tolist())
    
    # If still no points, use feasible points generator
    if len(points) == 0:
        print("Warning: No valid simplex centroid points found. Trying fallback strategies...")
        
        # Try to generate at least some valid points
        feasible_points = _generate_emergency_feasible_points(n_components, component_bounds)
        if len(feasible_points) > 0:
            points.extend(feasible_points)
        
        # If still no points, raise error
        if len(points) == 0:
            raise ValueError(
                "Unable to generate simplex centroid design. "
                "The component bounds appear to be too restrictive for this design method. "
                "Suggestions: 1) Try D-optimal or I-optimal methods instead, "
                "2) Relax component bounds slightly, or 3) Use Sequential Mixture Design."
            )
    
    print(f"Generated {len(points)} points for simplex centroid design")
    
    return np.array(points)

def generate_extreme_vertices(n_components: int, 
                            component_bounds: List[Tuple[float, float]],
                            fixed_components: Dict[str, float] = None,
                            component_names: List[str] = None) -> np.ndarray:
    """
    Generate extreme vertices design
    
    Parameters:
    -----------
    n_components : int
        Number of components
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
    fixed_components : Dict[str, float], optional
        Dictionary of fixed component names and values
    component_names : List[str], optional
        List of component names
        
    Returns:
    --------
    np.ndarray : Design matrix
    """
    # Handle fixed components
    if fixed_components:
        if component_names is None:
            component_names = [f'Comp_{i+1}' for i in range(n_components)]
            
        # Get indices of variable components
        all_fixed_component_names = set(fixed_components.keys())
        
        variable_indices = []
        for i, name in enumerate(component_names):
            if name not in all_fixed_component_names:
                variable_indices.append(i)
        
        n_variable_components = len(variable_indices)
        print(f"Generating extreme vertices for {n_variable_components} variable components")
        
        # Generate extreme vertices for variable components only
        variable_points = generate_extreme_vertices(
            n_variable_components, 
            [component_bounds[i] for i in variable_indices],
            None,
            [component_names[i] for i in variable_indices]
        )
        
        # Convert variable component points to full design matrix
        if len(variable_points) > 0:
            # Create full design matrix with all components
            full_design = np.zeros((len(variable_points), n_components))
            
            # Fill in variable component values
            for row_idx, var_point in enumerate(variable_points):
                for var_idx, comp_idx in enumerate(variable_indices):
                    full_design[row_idx, comp_idx] = var_point[var_idx]
            
            # Adjust for fixed components
            full_design = _adjust_for_fixed_components(full_design, fixed_components, component_names)
            return full_design
        else:
            print("No valid variable extreme vertices found, trying fallback...")
    
    # Find extreme vertices
    vertices = []
    
    # Check if we have restrictive bounds (all max bounds < 1.0)
    restrictive_bounds = all(bound[1] < 1.0 for bound in component_bounds)
    
    if restrictive_bounds:
        print("Detected restrictive bounds. Using specialized vertex generation...")
        
        # For restrictive bounds, try to find vertices at the corners of the feasible region
        # Start with all components at their lower bounds
        for i in range(n_components):
            # Try to maximize each component within bounds
            point = [component_bounds[j][0] for j in range(n_components)]  # Start with all minimums
            point[i] = component_bounds[i][1]  # Set component i to maximum
            
            # Normalize to ensure sum = 1
            point_sum = sum(point)
            if point_sum > 0:
                normalized_point = [p / point_sum for p in point]
                
                # Check if point satisfies bounds
                if check_bounds(normalized_point, component_bounds):
                    vertices.append(normalized_point)
                    print(f"Added extreme vertex for component {i+1}: {np.round(normalized_point, 4)}")
        
        # Try binary mixtures at extremes
        for i in range(n_components):
            for j in range(i+1, n_components):
                # Try different ratios
                for ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    # Create binary mixture with all other components at minimum
                    point = [component_bounds[k][0] for k in range(n_components)]
                    
                    # Calculate remaining after setting all to minimum
                    remaining = 1.0 - sum(point)
                    
                    # Distribute remaining between components i and j
                    if remaining > 0:
                        point[i] += remaining * ratio
                        point[j] += remaining * (1.0 - ratio)
                        
                        # Normalize
                        point_sum = sum(point)
                        normalized_point = [p / point_sum for p in point]
                        
                        if check_bounds(normalized_point, component_bounds):
                            vertices.append(normalized_point)
                            print(f"Added binary extreme vertex {i+1}-{j+1} ({ratio:.2f}): {np.round(normalized_point, 4)}")
    else:
        # Standard approach for non-restrictive bounds
        # Try all combinations of components at their bounds
        for combination in itertools.product([0, 1], repeat=n_components):
            # Skip all zeros or all ones
            if sum(combination) == 0 or sum(combination) == n_components:
                continue
            
            # Create point with components at their bounds based on combination
            point = [0.0] * n_components
            for i, use_upper in enumerate(combination):
                if use_upper:
                    point[i] = component_bounds[i][1]  # Upper bound
                else:
                    point[i] = component_bounds[i][0]  # Lower bound
            
            # Normalize to ensure sum = 1
            point_sum = sum(point)
            if point_sum > 0:
                normalized_point = [p / point_sum for p in point]
                
                # Check if point satisfies bounds
                if check_bounds(normalized_point, component_bounds):
                    vertices.append(normalized_point)
    
    # Add centroid if not already included
    centroid = [1.0 / n_components] * n_components
    if check_bounds(centroid, component_bounds):
        if not any(np.allclose(centroid, v, atol=1e-6) for v in vertices):
            vertices.append(centroid)
            print(f"Added centroid to extreme vertices design: {np.round(centroid, 4)}")
    
    # If not enough vertices found, try emergency approach
    if len(vertices) < n_components + 1:
        print(f"Only {len(vertices)} extreme vertices found. Trying emergency approach...")
        
        # Try to generate diverse points
        for _ in range(20):
            # Generate emphasis point
            point = generate_emphasis_point(component_bounds)
            
            # Normalize
            point_sum = np.sum(point)
            if point_sum > 0:
                normalized_point = point / point_sum
                
                # Check if point satisfies bounds
                if check_bounds(normalized_point, component_bounds):
                    vertices.append(normalized_point)
        
        # Try boundary-biased points
        for _ in range(20):
            # Generate boundary-biased point
            point = generate_boundary_biased_point(component_bounds)
            
            # Normalize
            point_sum = np.sum(point)
            if point_sum > 0:
                normalized_point = point / point_sum
                
                # Check if point satisfies bounds
                if check_bounds(normalized_point, component_bounds):
                    vertices.append(normalized_point)
        
        # Try binary mixtures
        for _ in range(10):
            # Generate binary mixture
            point = generate_binary_mixture(n_components)
            
            # Check if point satisfies bounds
            if check_bounds(point, component_bounds):
                vertices.append(point)
    
    # If still not enough vertices, use feasible points generator
    if len(vertices) < n_components + 1:
        print("Still not enough extreme vertices. Using feasible points generator...")
        
        feasible_points = _generate_emergency_feasible_points(n_components, component_bounds)
        if len(feasible_points) > 0:
            vertices.extend(feasible_points)
    
    # Remove duplicates
    unique_vertices = []
    for vertex in vertices:
        if not any(np.allclose(vertex, v, atol=1e-6) for v in unique_vertices):
            unique_vertices.append(vertex)
    
    print(f"Generated {len(unique_vertices)} unique extreme vertices")
    
    # If still no vertices, raise error
    if len(unique_vertices) == 0:
        raise ValueError(
            "Unable to generate extreme vertices design. "
            "The component bounds appear to be too restrictive for this design method. "
            "Suggestions: 1) Try D-optimal or I-optimal methods instead, "
            "2) Relax component bounds slightly, or 3) Use Sequential Mixture Design."
        )
    
    return np.array(unique_vertices)

def _adjust_for_fixed_components(design: np.ndarray, 
                               fixed_components: Dict[str, float],
                               component_names: List[str]) -> np.ndarray:
    """
    Adjust design matrix to account for fixed components
    
    Parameters:
    -----------
    design : np.ndarray
        Design matrix
    fixed_components : Dict[str, float]
        Dictionary of fixed component names and values
    component_names : List[str]
        List of component names
        
    Returns:
    --------
    np.ndarray : Adjusted design matrix
    """
    if not fixed_components:
        return design
    
    # Get indices of fixed and variable components
    fixed_indices = []
    variable_indices = []
    
    for i, name in enumerate(component_names):
        if name in fixed_components:
            fixed_indices.append(i)
        else:
            variable_indices.append(i)
    
    adjusted_design = design.copy()
    
    # Calculate fixed sum
    fixed_sum = sum(fixed_components.values())
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
        for i, name in enumerate(component_names):
            if name in fixed_components:
                adjusted_design[row_idx, i] = fixed_components[name]
    
    return adjusted_design
