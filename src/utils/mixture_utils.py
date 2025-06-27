"""
Utility functions for mixture designs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import itertools

def validate_proportion_bounds(component_bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Validate proportion bounds (0-1 range)
    
    Parameters:
    -----------
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
        
    Returns:
    --------
    List[Tuple[float, float]] : Validated bounds
    
    Raises:
    -------
    ValueError : If bounds are invalid
    """
    validated_bounds = []
    
    for i, bounds_tuple in enumerate(component_bounds):
        try:
            lower, upper = bounds_tuple
            
            # Automatically fix swapped bounds (if lower > upper)
            if lower > upper:
                print(f"Warning: Swapped bounds detected for component {i+1}. "
                      f"Automatically reordering ({lower}, {upper}) to ({upper}, {lower})")
                lower, upper = upper, lower
            
            # Now validate the corrected bounds
            if lower < 0:
                raise ValueError(f"Invalid proportion bounds for component {i+1}: Lower bound {lower} is negative")
            if upper > 1:
                raise ValueError(f"Invalid proportion bounds for component {i+1}: Upper bound {upper} exceeds 1")
                
            validated_bounds.append((lower, upper))
        except ValueError as val_error:
            raise val_error
        except Exception as val_unexpected:
            raise ValueError(f"Cannot process bounds[{i}]: {bounds_tuple}") from val_unexpected
    
    # Check if bounds are feasible (sum of lower bounds <= 1, sum of upper bounds >= 1)
    try:
        sum_lower = sum(bound[0] for bound in validated_bounds)
        sum_upper = sum(bound[1] for bound in validated_bounds)
        
        if sum_lower > 1:
            print(f"Warning: Sum of lower bounds ({sum_lower:.4f}) exceeds 1. "
                  f"This may result in an infeasible mixture space or limit the design space.")
        if sum_upper < 1:
            print(f"Warning: Sum of upper bounds ({sum_upper:.4f}) is less than 1. "
                  f"This may result in an infeasible mixture space or require component scaling.")
            
    except Exception as sum_error:
        raise ValueError(f"Error validating bounds feasibility: {validated_bounds}") from sum_error
    
    return validated_bounds

def validate_parts_bounds(component_bounds: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Validate parts bounds (non-negative values)
    
    Parameters:
    -----------
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component in parts
        
    Returns:
    --------
    List[Tuple[float, float]] : Validated bounds
    
    Raises:
    -------
    ValueError : If bounds are invalid
    """
    validated_bounds = []
    
    for i, bounds_tuple in enumerate(component_bounds):
        try:
            lower, upper = bounds_tuple
            
            # Automatically fix swapped bounds (if lower > upper)
            if lower > upper:
                print(f"Warning: Swapped bounds detected for component {i+1}. "
                      f"Automatically reordering ({lower}, {upper}) to ({upper}, {lower})")
                lower, upper = upper, lower
            
            # Now validate the corrected bounds
            if lower < 0:
                raise ValueError(f"Invalid parts bounds for component {i+1}: ({lower}, {upper}). "
                                 f"Lower bound must be non-negative.")
                
            validated_bounds.append((lower, upper))
        except Exception as parts_error:
            raise ValueError(f"Cannot process parts bounds[{i}]: {bounds_tuple}") from parts_error
    
    return validated_bounds

def convert_parts_to_proportions(component_bounds_parts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Convert bounds from parts to proportions
    
    Parameters:
    -----------
    component_bounds_parts : List[Tuple[float, float]]
        List of (min, max) tuples for each component in parts
        
    Returns:
    --------
    List[Tuple[float, float]] : Bounds converted to proportions
    """
    # Calculate total parts
    total_parts = sum(bound[1] for bound in component_bounds_parts)
    
    # Convert to proportions
    component_bounds_props = []
    for min_parts, max_parts in component_bounds_parts:
        min_prop = min_parts / total_parts
        max_prop = max_parts / total_parts
        component_bounds_props.append((min_prop, max_prop))
    
    return component_bounds_props

def check_bounds(point: List[float], component_bounds: List[Tuple[float, float]]) -> bool:
    """
    Check if a point satisfies component bounds
    
    Parameters:
    -----------
    point : List[float]
        Point to check
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
        
    Returns:
    --------
    bool : True if point satisfies bounds, False otherwise
    """
    if len(point) != len(component_bounds):
        return False
    
    # Check if point sums to 1 (within tolerance)
    # Use a more lenient tolerance for sum check
    if not np.isclose(sum(point), 1.0, atol=1e-4):
        # Try to normalize the point if it's close enough
        point_sum = sum(point)
        if 0.9 < point_sum < 1.1:  # If within 10% of 1.0
            # Point can be normalized
            return True
        return False
    
    # Check if point satisfies bounds with a more lenient tolerance
    for i, (lower, upper) in enumerate(component_bounds):
        # Use a more lenient tolerance for bounds checking
        if point[i] < lower - 1e-4 or point[i] > upper + 1e-4:
            # Special case: if the bound is very close to 0 or 1
            if (lower <= 1e-4 and point[i] >= 0) or (upper >= 1.0 - 1e-4 and point[i] <= 1.0):
                continue
            return False
    
    return True

def generate_emphasis_point(bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generate a point that emphasizes one component at a time
    Used by extreme vertices design to create diverse points
    
    Parameters:
    -----------
    bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
        
    Returns:
    --------
    np.ndarray : Point with emphasis on one component
    """
    n_components = len(bounds)
    
    # Randomly select one component to emphasize
    emphasis_idx = np.random.randint(0, n_components)
    
    # Create point with emphasis on selected component
    point = np.zeros(n_components)
    
    # Set all components to their lower bounds
    for i in range(n_components):
        if i == emphasis_idx:
            # Emphasized component gets a value closer to its upper bound
            lb, ub = bounds[i]
            # Use a value between 60-90% of the range
            emphasis_factor = np.random.uniform(0.6, 0.9)
            point[i] = lb + emphasis_factor * (ub - lb)
        else:
            # Other components get values closer to their lower bounds
            lb, ub = bounds[i]
            # Use a value between 10-40% of the range
            background_factor = np.random.uniform(0.1, 0.4)
            point[i] = lb + background_factor * (ub - lb)
    
    return point

def generate_boundary_biased_point(component_bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Generate a point that is biased toward the boundaries of the feasible region
    Used to improve diversity in candidate point generation
    
    Parameters:
    -----------
    component_bounds : List[Tuple[float, float]]
        List of (min, max) tuples for each component
        
    Returns:
    --------
    np.ndarray : Point biased toward boundaries
    """
    n_components = len(component_bounds)
    point = np.zeros(n_components)
    
    # For each component, decide whether to use a value near the boundary
    for i in range(n_components):
        lb, ub = component_bounds[i]
        
        # 60% chance of using a boundary value
        if np.random.random() < 0.6:
            # 50% chance of using lower bound, 50% chance of using upper bound
            if np.random.random() < 0.5:
                # Use value near lower bound (within 20% of range)
                range_val = ub - lb
                point[i] = lb + np.random.uniform(0, 0.2) * range_val
            else:
                # Use value near upper bound (within 20% of range)
                range_val = ub - lb
                point[i] = ub - np.random.uniform(0, 0.2) * range_val
        else:
            # Use random value in middle range
            range_val = ub - lb
            point[i] = lb + np.random.uniform(0.2, 0.8) * range_val
    
    return point

def generate_binary_mixture(n_components: int) -> np.ndarray:
    """
    Generate a binary mixture (only two components have non-zero values)
    Used to improve diversity in candidate point generation
    
    Parameters:
    -----------
    n_components : int
        Number of components
        
    Returns:
    --------
    np.ndarray : Binary mixture point
    """
    point = np.zeros(n_components)
    
    # Select two components randomly
    if n_components >= 2:
        i, j = np.random.choice(range(n_components), 2, replace=False)
        
        # Random ratio between the two components
        ratio = np.random.random()
        
        # Set values for the two components
        point[i] = ratio
        point[j] = 1.0 - ratio
    else:
        # Fallback for single component case
        point[0] = 1.0
    
    return point

def adjust_largest_more(point: np.ndarray, indices: List[int], delta: float) -> List[float]:
    """
    Adjustment strategy that reduces the largest component more when increasing another component
    This helps explore more diverse designs by creating more extreme points
    
    Parameters:
    -----------
    point : np.ndarray
        Current point
    indices : List[int]
        Indices of components to adjust
    delta : float
        Amount to adjust by (positive means increasing a component, negative means decreasing)
        
    Returns:
    --------
    List[float] : Adjusted values for the components in indices
    """
    if not indices:
        return []
        
    # Find the largest component
    values = [point[i] for i in indices]
    largest_idx = indices[np.argmax(values)]
    largest_val = point[largest_idx]
    
    # Calculate sum of all components to adjust
    total = sum(values)
    
    # If delta is positive, we're increasing another component, so reduce others
    if delta > 0:
        # If largest component is big enough, take most from it
        if largest_val > 0.5 * total:
            # Take 70% from largest, 30% proportionally from others
            largest_reduction = delta * 0.7
            other_reduction = delta * 0.3
            
            # Adjust values
            adjusted = []
            for i in indices:
                if i == largest_idx:
                    # Reduce largest component more
                    new_val = point[i] - largest_reduction
                    adjusted.append(max(0, new_val))
                else:
                    # Reduce other components proportionally
                    other_sum = total - largest_val
                    if other_sum > 0:
                        reduction_factor = point[i] / other_sum
                        new_val = point[i] - (other_reduction * reduction_factor)
                        adjusted.append(max(0, new_val))
                    else:
                        adjusted.append(point[i])
        else:
            # Reduce all proportionally
            adjusted = []
            for i in indices:
                reduction_factor = point[i] / total
                new_val = point[i] - (delta * reduction_factor)
                adjusted.append(max(0, new_val))
    else:
        # If delta is negative, we're decreasing another component, so increase others
        # Increase all proportionally
        adjusted = []
        for i in indices:
            increase_factor = point[i] / total
            new_val = point[i] - (delta * increase_factor)  # delta is negative, so this increases
            adjusted.append(new_val)
    
    return adjusted

def select_diverse_subset(design: np.ndarray, n_select: int) -> np.ndarray:
    """
    Select diverse subset using maximin distance criterion
    
    Parameters:
    -----------
    design : np.ndarray
        Design matrix
    n_select : int
        Number of points to select
        
    Returns:
    --------
    np.ndarray : Selected subset of design points
    """
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
