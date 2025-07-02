"""
Proportional Parts Mixture Design Implementation
Handles parts mode with proper proportion maintenance for component boundaries.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random


class ProportionalPartsMixture:
    """
    Mixture design class that handles parts mode with proportional boundaries.
    
    In parts mode, when boundaries are set for components, this class ensures
    that candidate points maintain proper proportional relationships between
    components while respecting the specified boundaries.
    """
    
    def __init__(self, 
                 n_components: int,
                 component_ranges: List[Tuple[float, float]],
                 fixed_components: Optional[Dict[int, float]] = None):
        """
        Initialize proportional parts mixture design.
        
        Parameters:
        -----------
        n_components : int
            Number of components in the mixture
        component_ranges : List[Tuple[float, float]]
            Component boundaries as [(min1, max1), (min2, max2), ...]
        fixed_components : dict, optional
            Fixed component values as {component_index: value}
        """
        self.n_components = n_components
        self.component_ranges = component_ranges
        self.fixed_components = fixed_components or {}
        
        # Calculate proportional ranges
        self.proportional_ranges = self._calculate_proportional_ranges()
        
        print(f"Proportional Parts Mixture initialized:")
        print(f"  Components: {n_components}")
        print(f"  Original ranges: {component_ranges}")
        print(f"  Proportional ranges: {self.proportional_ranges}")
        
    def _calculate_proportional_ranges(self) -> List[Tuple[float, float]]:
        """
        Calculate proportional ranges that maintain proper relationships.
        
        This method converts parts ranges to proportional ranges while ensuring
        that the sum constraint is satisfied and proportional relationships
        are maintained.
        """
        proportional_ranges = []
        
        # Method 1: Use minimum feasible total as denominator
        min_total = sum(min_val for min_val, max_val in self.component_ranges)
        max_total = sum(max_val for min_val, max_val in self.component_ranges)
        
        print(f"  Feasible total range: [{min_total:.3f}, {max_total:.3f}]")
        
        # For each component, calculate its feasible proportion range
        for i, (min_parts, max_parts) in enumerate(self.component_ranges):
            # Minimum proportion: when this component is at minimum and others can be at maximum
            other_max_total = sum(max_val for j, (min_val, max_val) in enumerate(self.component_ranges) if j != i)
            min_feasible_total = min_parts + other_max_total
            min_proportion = min_parts / min_feasible_total if min_feasible_total > 0 else 0.0
            
            # Maximum proportion: when this component is at maximum and others are at minimum
            other_min_total = sum(min_val for j, (min_val, max_val) in enumerate(self.component_ranges) if j != i)
            max_feasible_total = max_parts + other_min_total
            max_proportion = max_parts / max_feasible_total if max_feasible_total > 0 else 1.0
            
            # Ensure proportions are valid (between 0 and 1)
            min_proportion = max(0.0, min(1.0, min_proportion))
            max_proportion = max(0.0, min(1.0, max_proportion))
            
            # Ensure min <= max
            if min_proportion > max_proportion:
                min_proportion, max_proportion = max_proportion, min_proportion
            
            proportional_ranges.append((min_proportion, max_proportion))
            
            print(f"    Component {i+1}: parts [{min_parts:.3f}, {max_parts:.3f}] -> proportions [{min_proportion:.6f}, {max_proportion:.6f}]")
        
        return proportional_ranges
    
    def generate_proportional_candidate(self) -> List[float]:
        """
        Generate a candidate point that maintains proportional relationships.
        
        This method generates candidates in proportion space while respecting
        the original parts boundaries through proper scaling.
        """
        max_attempts = 100
        
        for attempt in range(max_attempts):
            # Generate random proportions within calculated ranges
            candidate = []
            for i, (min_prop, max_prop) in enumerate(self.proportional_ranges):
                if i in self.fixed_components:
                    # Handle fixed components later
                    candidate.append(0.0)  # Placeholder
                else:
                    # Generate random proportion within feasible range
                    if max_prop > min_prop:
                        prop = random.uniform(min_prop, max_prop)
                    else:
                        prop = min_prop
                    candidate.append(prop)
            
            # Normalize to sum = 1
            current_sum = sum(candidate)
            if current_sum > 1e-10:
                candidate = [x / current_sum for x in candidate]
            else:
                continue  # Try again if sum is too small
            
            # Check if this candidate can be converted to valid parts
            if self._is_valid_proportional_candidate(candidate):
                return candidate
        
        # Fallback: Generate equal proportions
        equal_prop = 1.0 / self.n_components
        return [equal_prop] * self.n_components
    
    def _is_valid_proportional_candidate(self, proportions: List[float]) -> bool:
        """
        Check if proportional candidate can be converted to valid parts.
        
        This method verifies that when proportions are converted back to parts,
        they respect the original parts boundaries.
        """
        # Try different total parts values to see if any satisfy all constraints
        candidate_totals = []
        
        # Generate candidate total parts based on the proportions
        for i, prop in enumerate(proportions):
            if prop > 1e-10:  # Avoid division by very small numbers
                min_parts, max_parts = self.component_ranges[i]
                # Calculate what total would make this proportion give min/max parts
                total_for_min = min_parts / prop
                total_for_max = max_parts / prop
                candidate_totals.extend([total_for_min, total_for_max])
        
        if not candidate_totals:
            return False
        
        # Try each candidate total
        for total_parts in candidate_totals:
            if total_parts <= 0:
                continue
                
            # Check if this total makes all components satisfy their bounds
            valid = True
            for i, prop in enumerate(proportions):
                parts_value = prop * total_parts
                min_parts, max_parts = self.component_ranges[i]
                
                if parts_value < min_parts - 1e-10 or parts_value > max_parts + 1e-10:
                    valid = False
                    break
            
            if valid:
                return True
        
        return False
    
    def convert_proportions_to_parts(self, proportions: List[float]) -> Tuple[List[float], float]:
        """
        Convert proportional candidate to parts while respecting boundaries.
        
        Parameters:
        -----------
        proportions : List[float]
            Proportional values that sum to 1
            
        Returns:
        --------
        Tuple[List[float], float]
            (parts_values, total_parts)
        """
        best_parts = None
        best_total = None
        min_violation = float('inf')
        
        # Strategy 1: Try to find exact solution
        # Generate candidate totals based on each component at its bounds
        candidate_totals = []
        
        for i, prop in enumerate(proportions):
            if prop > 1e-10:
                min_parts, max_parts = self.component_ranges[i]
                candidate_totals.extend([
                    min_parts / prop,  # Total that makes component i at minimum
                    max_parts / prop   # Total that makes component i at maximum
                ])
        
        # Also try some intermediate values
        if candidate_totals:
            min_total = min(candidate_totals)
            max_total = max(candidate_totals)
            for factor in [0.25, 0.5, 0.75]:
                candidate_totals.append(min_total + factor * (max_total - min_total))
        
        # Test each candidate total
        for total_parts in candidate_totals:
            if total_parts <= 0:
                continue
                
            parts_values = [prop * total_parts for prop in proportions]
            
            # Calculate violation (how much we exceed bounds)
            violation = 0.0
            valid = True
            
            for i, parts_val in enumerate(parts_values):
                min_parts, max_parts = self.component_ranges[i]
                
                if parts_val < min_parts:
                    violation += (min_parts - parts_val) ** 2
                elif parts_val > max_parts:
                    violation += (parts_val - max_parts) ** 2
            
            if violation < min_violation:
                min_violation = violation
                best_parts = parts_values[:]
                best_total = total_parts
                
                if violation < 1e-10:  # Exact solution found
                    break
        
        # If no good solution found, use fallback
        if best_parts is None:
            # Use average of all range midpoints as total
            avg_total = sum((min_val + max_val) / 2 for min_val, max_val in self.component_ranges)
            best_parts = [prop * avg_total for prop in proportions]
            best_total = avg_total
        
        return best_parts, best_total
    
    def normalize_parts_to_proportions(self, parts_values: List[float]) -> List[float]:
        """
        Normalize parts values back to proportions (sum = 1).
        
        Parameters:
        -----------
        parts_values : List[float]
            Parts values
            
        Returns:
        --------
        List[float]
            Normalized proportions that sum to 1
        """
        total_parts = sum(parts_values)
        if total_parts > 1e-10:
            return [part / total_parts for part in parts_values]
        else:
            # Equal proportions fallback
            return [1.0 / self.n_components] * self.n_components
    
    def validate_parts_candidate(self, parts_values: List[float]) -> bool:
        """
        Validate that parts values respect all boundaries.
        
        Parameters:
        -----------
        parts_values : List[float]
            Parts values to validate
            
        Returns:
        --------
        bool
            True if all boundaries are respected
        """
        for i, parts_val in enumerate(parts_values):
            min_parts, max_parts = self.component_ranges[i]
            if parts_val < min_parts - 1e-10 or parts_val > max_parts + 1e-10:
                return False
        return True
    
    def generate_feasible_parts_candidate(self) -> Tuple[List[float], List[float]]:
        """
        Generate a feasible candidate in both parts and proportions.
        
        Returns:
        --------
        Tuple[List[float], List[float]]
            (parts_values, proportions)
        """
        # Generate proportional candidate
        proportions = self.generate_proportional_candidate()
        
        # Convert to parts
        parts_values, total_parts = self.convert_proportions_to_parts(proportions)
        
        # Verify and adjust if needed
        for i, parts_val in enumerate(parts_values):
            min_parts, max_parts = self.component_ranges[i]
            parts_values[i] = max(min_parts, min(max_parts, parts_val))
        
        # Renormalize proportions
        final_proportions = self.normalize_parts_to_proportions(parts_values)
        
        return parts_values, final_proportions


def test_proportional_parts_mixture():
    """Test the proportional parts mixture implementation"""
    print("="*80)
    print("TESTING PROPORTIONAL PARTS MIXTURE")
    print("="*80)
    
    # Test case: 3 components with different ranges
    component_ranges = [
        (0.1, 5.0),   # Component A: 0.1 to 5.0 parts
        (0.2, 3.0),   # Component B: 0.2 to 3.0 parts  
        (0.1, 2.0)    # Component C: 0.1 to 2.0 parts
    ]
    
    ppm = ProportionalPartsMixture(
        n_components=3,
        component_ranges=component_ranges
    )
    
    print(f"\nTesting candidate generation:")
    print(f"{'Run':<4} {'Parts':<30} {'Proportions':<30} {'Valid':<6}")
    print("-" * 80)
    
    for i in range(10):
        parts_values, proportions = ppm.generate_feasible_parts_candidate()
        
        # Validate
        parts_valid = ppm.validate_parts_candidate(parts_values)
        prop_sum = sum(proportions)
        
        parts_str = f"[{', '.join(f'{x:.3f}' for x in parts_values)}]"
        prop_str = f"[{', '.join(f'{x:.3f}' for x in proportions)}]"
        
        print(f"{i+1:<4} {parts_str:<30} {prop_str:<30} {parts_valid and abs(prop_sum - 1.0) < 1e-6}")
    
    print(f"\nTesting boundary cases:")
    
    # Test with extreme proportions
    test_proportions = [
        [0.9, 0.05, 0.05],   # Dominant first component
        [0.33, 0.33, 0.34],  # Nearly equal
        [0.1, 0.8, 0.1],     # Dominant second component
    ]
    
    for i, props in enumerate(test_proportions):
        print(f"\nTest {i+1}: Input proportions = {props}")
        parts_values, total = ppm.convert_proportions_to_parts(props)
        final_props = ppm.normalize_parts_to_proportions(parts_values)
        
        print(f"  Parts: [{', '.join(f'{x:.3f}' for x in parts_values)}] (total = {total:.3f})")
        print(f"  Final proportions: [{', '.join(f'{x:.3f}' for x in final_props)}]")
        print(f"  Valid parts: {ppm.validate_parts_candidate(parts_values)}")
        print(f"  Sum check: {sum(final_props):.6f}")


if __name__ == "__main__":
    test_proportional_parts_mixture()
