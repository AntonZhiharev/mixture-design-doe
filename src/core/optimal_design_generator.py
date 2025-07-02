"""
Optimal Design Generator - Clean Implementation
==============================================

This module provides optimal experimental design generation with clear separation between:
1. MIXTURE DESIGNS: Components sum to 1, reduced quadratic model (15 parameters)
2. STANDARD DOE: Independent variables, full quadratic model (20 parameters)

Key principles:
- Mixture designs use Scheffé canonical polynomials (no pure quadratic terms)
- Standard DOE uses full polynomial models (includes pure quadratic terms)
- Each design type operates in its appropriate space (simplex vs. hypercube)
"""

import math
import random

def calculate_determinant(matrix):
    """Calculate determinant using Gaussian elimination"""
    n = len(matrix)
    if n == 0:
        return 1.0
    
    # Ensure matrix is square
    for row in matrix:
        if len(row) != n:
            raise ValueError("Matrix must be square for determinant calculation")
    
    # Create a copy of the matrix
    temp = [row[:] for row in matrix]
    
    det = 1.0
    
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(temp[k][i]) > abs(temp[max_row][i]):
                max_row = k
        
        # Swap rows if needed
        if max_row != i:
            temp[i], temp[max_row] = temp[max_row], temp[i]
            det *= -1  # Determinant changes sign when rows are swapped
        
        # Check for zero pivot
        if abs(temp[i][i]) < 1e-10:
            return 0.0  # Matrix is singular
        
        det *= temp[i][i]
        
        # Eliminate below pivot
        for k in range(i + 1, n):
            if abs(temp[i][i]) > 1e-10:
                factor = temp[k][i] / temp[i][i]
                for j in range(i, n):
                    temp[k][j] -= factor * temp[i][j]
    
    return det

def matrix_multiply(A, B):
    """Multiply two matrices A and B"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Cannot multiply matrices: incompatible dimensions")
    
    result = []
    for i in range(rows_A):
        row = []
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)
    return result

def transpose_matrix(matrix):
    """Transpose a matrix"""
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def gram_matrix(A):
    """Calculate Gram matrix A^T * A"""
    A_T = transpose_matrix(A)
    return matrix_multiply(A_T, A)

def matrix_inverse(matrix):
    """Calculate matrix inverse using Gauss-Jordan elimination"""
    n = len(matrix)
    if n == 0:
        return []
    
    # Create augmented matrix [A|I]
    augmented = []
    for i in range(n):
        row = matrix[i][:] + [0.0] * n
        row[n + i] = 1.0
        augmented.append(row)
    
    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k
        
        # Swap rows
        if max_row != i:
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
        
        # Check for zero pivot
        if abs(augmented[i][i]) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        # Scale pivot row
        pivot = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= pivot
        
        # Eliminate column
        for k in range(n):
            if k != i:
                factor = augmented[k][i]
                for j in range(2 * n):
                    augmented[k][j] -= factor * augmented[i][j]
    
    # Extract inverse matrix
    inverse = []
    for i in range(n):
        inverse.append(augmented[i][n:])
    
    return inverse

def matrix_trace(matrix):
    """Calculate trace (sum of diagonal elements) of a matrix"""
    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square for trace calculation")
    
    trace = 0.0
    for i in range(len(matrix)):
        trace += matrix[i][i]
    
    return trace

def evaluate_mixture_model_terms(x_values, model_type):
    """
    Evaluate mixture model terms (Scheffé canonical polynomials)
    
    For mixture designs where sum(xi) = 1:
    - Linear: x1, x2, ..., xk
    - Quadratic: x1, x2, ..., xk, x1*x2, x1*x3, ..., x(k-1)*xk (NO pure quadratic terms)
    - Cubic: adds three-way interactions xi*xj*xk
    """
    num_variables = len(x_values)
    row = []
    
    # Linear terms: component proportions
    for i in range(num_variables):
        row.append(x_values[i])
    
    if model_type in ["quadratic", "cubic"]:
        # Two-way interaction terms: xi*xj for all i < j
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                row.append(x_values[i] * x_values[j])
    
    if model_type == "cubic":
        # Three-way interaction terms: xi*xj*xk for all i < j < k
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                for k in range(j+1, num_variables):
                    row.append(x_values[i] * x_values[j] * x_values[k])
    
    return row

def evaluate_standard_model_terms(x_values, model_type):
    """
    Evaluate standard model terms
    
    For standard DOE with independent variables:
    - Linear: x1, x2, ..., xk
    - Quadratic: adds x1², x2², ..., xk², x1*x2, x1*x3, ..., x(k-1)*xk
    - Cubic: adds x1³, x2³, ..., xk³, x1²*x2, x1*x2², ..., x1*x2*x3, ...
    """
    num_variables = len(x_values)
    row = []
    
    # Linear terms
    for i in range(num_variables):
        row.append(x_values[i])
    
    if model_type in ["quadratic", "cubic"]:
        # Pure quadratic terms
        for i in range(num_variables):
            row.append(x_values[i]**2)
        
        # Two-way interaction terms
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                row.append(x_values[i] * x_values[j])
    
    if model_type == "cubic":
        # Cubic terms
        for i in range(num_variables):
            row.append(x_values[i]**3)
        
        # Quadratic-linear interaction terms: xi²*xj for all i≠j
        for i in range(num_variables):
            for j in range(num_variables):
                if i != j:
                    row.append(x_values[i]**2 * x_values[j])
        
        # Three-way interaction terms: xi*xj*xk for all i < j < k
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                for k in range(j+1, num_variables):
                    row.append(x_values[i] * x_values[j] * x_values[k])
    
    return row

# Backward compatibility functions
def evaluate_mixture_quadratic_terms(x_values):
    """Backward compatibility function"""
    return evaluate_mixture_model_terms(x_values, "quadratic")

def evaluate_standard_quadratic_terms(x_values):
    """Backward compatibility function"""
    return evaluate_standard_model_terms(x_values, "quadratic")

class OptimalDesignGenerator:
    """
    Optimal Design Generator with clear separation between design types
    """
    
    def __init__(self, num_variables, num_runs, design_type="mixture", model_type="quadratic", component_ranges=None):
        """
        Initialize the optimal design generator
        
        Parameters:
        - num_variables: Number of components/variables
        - num_runs: Number of experimental runs
        - design_type: "mixture" or "standard"
        - model_type: "linear", "quadratic", or "cubic"
        - component_ranges: List of tuples [(min1, max1), (min2, max2), ...] for each component
                           If provided, enables parts conversion after optimization
        """
        self.num_variables = num_variables
        self.num_runs = num_runs
        self.design_type = design_type.lower()
        self.model_type = model_type.lower()
        self.component_ranges = component_ranges
        
        # Initialize proportional parts mixture functionality if component ranges provided
        self.proportional_ranges = None
        if self.component_ranges and self.design_type == "mixture":
            self.proportional_ranges = self._calculate_proportional_ranges()
            print(f"  Using proportional parts mixture approach for bounded components")
        
        # Calculate number of parameters based on design and model type
        self.num_parameters = self._calculate_num_parameters()
        
        print(f"{self.design_type.title()} Design Generator:")
        print(f"  Variables: {num_variables} ({'mixture components' if design_type == 'mixture' else 'independent factors'})")
        if design_type == "mixture":
            print(f"  Constraint: sum(xi) = 1")
            if component_ranges:
                print(f"  Parts bounds: {component_ranges}")
        else:
            print(f"  Range: [-1, 1] for each variable")
        print(f"  Model: {self.model_type.title()} ({self.num_parameters} parameters)")
        print(f"  Target runs: {num_runs}")
        
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
    
    def _calculate_num_parameters(self):
        """Calculate number of parameters based on design type and model type"""
        n = self.num_variables
        
        if self.design_type == "mixture":
            if self.model_type == "linear":
                # Mixture linear: x1, x2, ..., xn
                return n
            elif self.model_type == "quadratic":
                # Mixture quadratic: x1, x2, ..., xn, x1*x2, x1*x3, ..., x(n-1)*xn
                # NO pure quadratic terms (xi²) due to constraint sum(xi) = 1
                linear_terms = n
                interaction_terms = n * (n - 1) // 2
                return linear_terms + interaction_terms
            elif self.model_type == "cubic":
                # Mixture cubic: includes all mixture terms up to degree 3
                linear_terms = n
                interaction_terms = n * (n - 1) // 2
                # Three-way interactions: xi*xj*xk for i < j < k
                three_way_terms = n * (n - 1) * (n - 2) // 6
                return linear_terms + interaction_terms + three_way_terms
            else:
                raise ValueError("model_type must be 'linear', 'quadratic', or 'cubic'")
                
        else:  # standard
            if self.model_type == "linear":
                # Standard linear: x1, x2, ..., xn
                return n
            elif self.model_type == "quadratic":
                # Standard quadratic: x1, x2, ..., xn, x1², x2², ..., xn², x1*x2, x1*x3, ..., x(n-1)*xn
                linear_terms = n
                quadratic_terms = n
                interaction_terms = n * (n - 1) // 2
                return linear_terms + quadratic_terms + interaction_terms
            elif self.model_type == "cubic":
                # Standard cubic: includes all terms up to degree 3
                linear_terms = n
                quadratic_terms = n
                interaction_terms = n * (n - 1) // 2
                cubic_terms = n
                quadratic_linear_terms = n * (n - 1)  # xi²*xj for i≠j
                three_way_terms = n * (n - 1) * (n - 2) // 6
                return linear_terms + quadratic_terms + interaction_terms + cubic_terms + quadratic_linear_terms + three_way_terms
            else:
                raise ValueError("model_type must be 'linear', 'quadratic', or 'cubic'")
    
    def _calculate_proportional_ranges(self):
        """
        Calculate proportional ranges that maintain proper relationships.
        
        This method converts parts ranges to proportional ranges while ensuring
        that the sum constraint is satisfied and proportional relationships
        are maintained.
        """
        proportional_ranges = []
        
        # Calculate feasible total range
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

    def _generate_feasible_vertex(self, dominant_component_index):
        """
        Generate a feasible vertex-like point that respects proportional bounds.
        
        This method tries systematically to make the dominant component as large as possible
        while ensuring all other components respect their minimum bounds.
        
        Parameters:
        - dominant_component_index: Index of component that should be dominant
        
        Returns:
        - Feasible vertex-like point (list of proportions summing to 1)
        """
        if self.proportional_ranges is None:
            # Fallback to pure vertex if no bounds
            point = [0.0] * self.num_variables
            point[dominant_component_index] = 1.0
            return point
        
        best_vertex = None
        max_dominant_proportion = 0.0
        
        # Try many systematic approaches to maximize the dominant component
        max_attempts = 500  # Increased attempts for better vertex search
        
        for attempt in range(max_attempts):
            point = [0.0] * self.num_variables
            
            # Strategy: Set other components close to their minimums to maximize dominant
            total_used = 0.0
            for i in range(self.num_variables):
                if i != dominant_component_index:
                    min_prop, max_prop = self.proportional_ranges[i]
                    
                    # Use a range from minimum to slightly above minimum
                    # Earlier attempts use values closer to minimum
                    progress = attempt / max_attempts
                    max_allowable = min_prop + (max_prop - min_prop) * 0.1 * (1 - progress)
                    
                    point[i] = random.uniform(min_prop, max_allowable)
                    total_used += point[i]
            
            # Assign remaining to dominant component
            remaining = 1.0 - total_used
            if remaining > 0:
                min_prop_dom, max_prop_dom = self.proportional_ranges[dominant_component_index]
                
                # Try to use as much as possible for dominant component
                if remaining >= min_prop_dom:
                    # Use the maximum feasible amount for dominant component
                    point[dominant_component_index] = min(remaining, max_prop_dom)
                    
                    # Adjust if we couldn't use all remaining
                    if point[dominant_component_index] < remaining:
                        # Redistribute excess among other components proportionally
                        excess = remaining - point[dominant_component_index]
                        other_indices = [i for i in range(self.num_variables) if i != dominant_component_index]
                        
                        # Add excess proportionally to other components within their bounds
                        for i in other_indices:
                            min_prop, max_prop = self.proportional_ranges[i]
                            max_additional = max_prop - point[i]
                            additional = min(excess / len(other_indices), max_additional)
                            point[i] += additional
                            excess -= additional
                            if excess <= 1e-10:
                                break
                    
                    # Normalize to ensure exact sum = 1
                    total = sum(point)
                    if total > 1e-10:
                        point = [x / total for x in point]
                    
                    # Check if this point is valid and has a higher dominant proportion
                    if (self._is_valid_proportional_candidate(point) and 
                        point[dominant_component_index] > max_dominant_proportion):
                        max_dominant_proportion = point[dominant_component_index]
                        best_vertex = point[:]
        
        # Additional systematic search for optimal vertex
        if best_vertex is None or max_dominant_proportion < 0.7:  # If no good vertex found
            print(f"    Performing systematic search for component {dominant_component_index+1}...")
            
            # Try systematic combinations
            min_prop_dom, max_prop_dom = self.proportional_ranges[dominant_component_index]
            
            # Try different strategies for other components
            for strategy in range(10):
                point = [0.0] * self.num_variables
                total_used = 0.0
                
                for i in range(self.num_variables):
                    if i != dominant_component_index:
                        min_prop, max_prop = self.proportional_ranges[i]
                        
                        if strategy < 5:
                            # Strategy 1-5: Use minimum + small increments
                            increment = (strategy / 4) * 0.05  # 0% to 5% above minimum
                            point[i] = min_prop + increment * (max_prop - min_prop)
                        else:
                            # Strategy 6-10: Use minimum + random small amount
                            point[i] = min_prop + random.uniform(0.0, 0.03)
                        
                        total_used += point[i]
                
                # Assign maximum feasible to dominant component
                remaining = 1.0 - total_used
                if remaining >= min_prop_dom:
                    point[dominant_component_index] = min(remaining, max_prop_dom)
                    
                    # Normalize
                    total = sum(point)
                    if total > 1e-10:
                        point = [x / total for x in point]
                    
                    # Check if this is better
                    if (self._is_valid_proportional_candidate(point) and 
                        point[dominant_component_index] > max_dominant_proportion):
                        max_dominant_proportion = point[dominant_component_index]
                        best_vertex = point[:]
        
        if best_vertex is not None:
            print(f"    Found vertex with {max_dominant_proportion:.1%} dominant component")
            return best_vertex
        
        # Final fallback: Generate a balanced feasible point
        print(f"    Warning: Could not find good vertex for component {dominant_component_index+1}, using fallback")
        return self._generate_proportional_candidate()

    def _generate_proportional_candidate(self):
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
        equal_prop = 1.0 / self.num_variables
        return [equal_prop] * self.num_variables

    def _is_valid_proportional_candidate(self, proportions):
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

    def _generate_candidate_point(self):
        """Generate a random candidate point appropriate for design type"""
        if self.design_type == "mixture":
            # Use proportional parts mixture approach if proportional ranges are available
            if self.proportional_ranges is not None:
                # Generate candidate using proportional parts approach
                try:
                    return self._generate_proportional_candidate()
                except Exception as e:
                    # Fallback to standard method if proportional approach fails
                    pass
            
            # Standard simplex point generation (sum=1, all >=0)
            point = [random.random() for _ in range(self.num_variables)]
            total = sum(point)
            return [x/total for x in point]  # Normalize to sum=1
            
        else:  # standard
            # Generate random point in [-1,1] hypercube
            return [random.uniform(-1.0, 1.0) for _ in range(self.num_variables)]
    
    def _evaluate_candidate_determinant(self, candidate_point):
        """Evaluate determinant if candidate point is added"""
        try:
            # Create test design matrix
            test_design_matrix = self.design_matrix[:]
            if self.design_type == "mixture":
                test_design_matrix.append(evaluate_mixture_model_terms(candidate_point, self.model_type))
            else:
                test_design_matrix.append(evaluate_standard_model_terms(candidate_point, self.model_type))
            
            # Calculate information matrix
            test_info_matrix = gram_matrix(test_design_matrix)
            
            # Calculate determinant
            return calculate_determinant(test_info_matrix)
        except:
            return 0.0
    
    def add_design_point(self, x_values):
        """Add a design point to the design"""
        # Clean up numerical precision issues for standard mixture designs
        if self.design_type == "mixture" and self.proportional_ranges is None:
            # For standard mixture designs (without parts mode), clean up near-zero and near-one values
            cleaned_values = []
            for x in x_values:
                if x < 1e-6:  # Very close to zero
                    cleaned_values.append(0.0)
                elif x > 1.0 - 1e-6:  # Very close to one
                    cleaned_values.append(1.0)
                else:
                    cleaned_values.append(x)
            
            # Renormalize to ensure exact sum = 1.0
            total = sum(cleaned_values)
            if total > 1e-10:
                cleaned_values = [x / total for x in cleaned_values]
            
            self.design_points.append(cleaned_values)
            x_values = cleaned_values
        else:
            self.design_points.append(x_values[:])
        
        if self.design_type == "mixture":
            design_row = evaluate_mixture_model_terms(x_values, self.model_type)
        else:
            design_row = evaluate_standard_model_terms(x_values, self.model_type)
            
        self.design_matrix.append(design_row)
        
        # Update information matrix and determinant
        info_matrix = gram_matrix(self.design_matrix)
        current_det = calculate_determinant(info_matrix)
        self.determinant_history.append(current_det)
        
        return current_det
    
    def _evaluate_candidate_i_optimality(self, candidate_point):
        """Evaluate I-optimality criterion if candidate point is added"""
        try:
            # Create test design matrix
            test_design_matrix = self.design_matrix[:]
            if self.design_type == "mixture":
                test_design_matrix.append(evaluate_mixture_model_terms(candidate_point, self.model_type))
            else:
                test_design_matrix.append(evaluate_standard_model_terms(candidate_point, self.model_type))
            
            # Calculate information matrix
            test_info_matrix = gram_matrix(test_design_matrix)
            
            # Calculate inverse and trace for I-optimality
            try:
                # Check if matrix is well-conditioned first
                det_value = calculate_determinant(test_info_matrix)
                if det_value < 1e-10:
                    return 0.0  # Singular matrix
                
                inverse_matrix = matrix_inverse(test_info_matrix)
                trace_value = matrix_trace(inverse_matrix)
                
                # I-optimality: minimize trace(X'X)^(-1)
                # Return negative trace so algorithm maximizes -trace (i.e., minimizes trace)
                # Scale by 1000 to avoid numerical issues with small values
                return -trace_value * 1000.0 if trace_value > 1e-10 else -1e6
            except:
                return -1e6  # Heavy penalty for failed inversion
        except:
            return -1e6

    def generate_simplex_lattice_design(self, levels=3):
        """Generate simplex lattice design for mixture experiments"""
        if self.design_type != "mixture":
            raise ValueError("Simplex lattice design is only for mixture experiments")
        
        print(f"\nGenerating {levels}-level simplex lattice design...")
        
        # Clear existing design
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
        
        # Generate lattice points
        from itertools import combinations_with_replacement
        
        # Create lattice coordinates
        lattice_coords = []
        for i in range(levels + 1):
            lattice_coords.append(i / levels)
        
        print(f"Lattice coordinates: {lattice_coords}")
        
        # Generate all combinations that sum to 1
        points = []
        for combo in combinations_with_replacement(range(levels + 1), self.num_variables):
            if sum(combo) == levels:  # Ensures sum = 1 when normalized
                point = [x / levels for x in combo]
                # Generate all permutations of this combination
                from itertools import permutations
                for perm in set(permutations(point)):
                    if list(perm) not in points:
                        points.append(list(perm))
        
        print(f"Generated {len(points)} simplex lattice points")
        
        # Add points to design (up to num_runs)
        for i, point in enumerate(points[:self.num_runs]):
            det = self.add_design_point(point)
            print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in point]} (det = {det:.3e})")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"\nSimplex lattice design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det

    def generate_extreme_vertices_design(self):
        """Generate extreme vertices design for mixture experiments"""
        if self.design_type != "mixture":
            raise ValueError("Extreme vertices design is only for mixture experiments")
        
        print(f"\nGenerating extreme vertices design...")
        
        # Clear existing design
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
        
        # Generate extreme vertices (pure components and binary mixtures)
        vertices = []
        
        # Pure components (vertices of simplex)
        for i in range(self.num_variables):
            vertex = [0.0] * self.num_variables
            vertex[i] = 1.0
            vertices.append(vertex)
        
        # Binary mixtures (edge midpoints)
        for i in range(self.num_variables):
            for j in range(i + 1, self.num_variables):
                vertex = [0.0] * self.num_variables
                vertex[i] = 0.5
                vertex[j] = 0.5
                vertices.append(vertex)
        
        # Add centroid if needed
        if len(vertices) < self.num_runs:
            centroid = [1.0 / self.num_variables] * self.num_variables
            vertices.append(centroid)
        
        print(f"Generated {len(vertices)} extreme vertices")
        
        # Add vertices to design (up to num_runs)
        for i, vertex in enumerate(vertices[:self.num_runs]):
            det = self.add_design_point(vertex)
            print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in vertex]} (det = {det:.3e})")
        
        # Fill remaining runs with optimized points if needed
        while len(self.design_points) < self.num_runs:
            best_point = None
            best_det = self.determinant_history[-1] if self.determinant_history else 0
            
            for attempt in range(500):
                candidate = self._generate_candidate_point()
                test_det = self._evaluate_candidate_determinant(candidate)
                
                if test_det > best_det:
                    best_det = test_det
                    best_point = candidate
            
            if best_point:
                det = self.add_design_point(best_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in best_point]} (det = {det:.3e}, optimized)")
            else:
                random_point = self._generate_candidate_point()
                det = self.add_design_point(random_point)
                print(f"  Point {len(self.design_points)}: Random point added (det = {det:.3e})")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"\nExtreme vertices design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det

    def generate_centroid_design(self):
        """Generate centroid design for mixture experiments"""
        if self.design_type != "mixture":
            raise ValueError("Centroid design is only for mixture experiments")
        
        print(f"\nGenerating centroid design...")
        
        # Clear existing design
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
        
        centroids = []
        
        # Overall centroid (all components equal)
        overall_centroid = [1.0 / self.num_variables] * self.num_variables
        centroids.append(overall_centroid)
        
        # Binary centroids (two components equal, others zero)
        for i in range(self.num_variables):
            for j in range(i + 1, self.num_variables):
                centroid = [0.0] * self.num_variables
                centroid[i] = 0.5
                centroid[j] = 0.5
                centroids.append(centroid)
        
        # Ternary centroids (three components equal, others zero) if applicable
        if self.num_variables >= 3:
            for i in range(self.num_variables):
                for j in range(i + 1, self.num_variables):
                    for k in range(j + 1, self.num_variables):
                        centroid = [0.0] * self.num_variables
                        centroid[i] = 1.0 / 3.0
                        centroid[j] = 1.0 / 3.0
                        centroid[k] = 1.0 / 3.0
                        centroids.append(centroid)
        
        # Pure components
        for i in range(self.num_variables):
            centroid = [0.0] * self.num_variables
            centroid[i] = 1.0
            centroids.append(centroid)
        
        print(f"Generated {len(centroids)} centroid points")
        
        # Add centroids to design (up to num_runs)
        for i, centroid in enumerate(centroids[:self.num_runs]):
            det = self.add_design_point(centroid)
            print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in centroid]} (det = {det:.3e})")
        
        # Fill remaining runs with optimized points if needed
        while len(self.design_points) < self.num_runs:
            best_point = None
            best_det = self.determinant_history[-1] if self.determinant_history else 0
            
            for attempt in range(500):
                candidate = self._generate_candidate_point()
                test_det = self._evaluate_candidate_determinant(candidate)
                
                if test_det > best_det:
                    best_det = test_det
                    best_point = candidate
            
            if best_point:
                det = self.add_design_point(best_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in best_point]} (det = {det:.3e}, optimized)")
            else:
                random_point = self._generate_candidate_point()
                det = self.add_design_point(random_point)
                print(f"  Point {len(self.design_points)}: Random point added (det = {det:.3e})")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"\nCentroid design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det

    def generate_i_optimal_design(self):
        """Generate I-optimal design with balanced vertex, edge, and center coverage"""
        print(f"\nGenerating I-optimal {self.design_type} design...")
        
        # Clear existing design
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
        
        if self.design_type == "mixture":
            return self._generate_mixture_i_optimal()
        else:  # standard
            return self._generate_standard_i_optimal()
    
    def _generate_mixture_i_optimal(self):
        """Generate I-optimal design for mixture experiments - PURE I-OPTIMALITY APPROACH"""
        print("Mixture I-optimal design optimized for prediction variance (I-optimality)...")
        
        # I-optimal strategy: Start with minimal base and then purely optimize for I-criterion
        # Unlike D-optimal, we don't force specific structure - let I-optimality drive the selection
        
        # Phase 1: Add minimal base set (just enough to avoid singularity)
        print("Phase 1: Adding minimal base set...")
        
        # Add pure components (vertices) - minimum required for identifiability
        for i in range(min(self.num_variables, self.num_runs)):
            point = [0.0] * self.num_variables
            point[i] = 1.0  # Pure component
            det = self.add_design_point(point)
            print(f"  Base vertex {i+1}: {[f'{x:7.3f}' for x in point]} (det = {det:.3e})")
        
        if len(self.design_points) >= self.num_runs:
            return self.determinant_history[-1]
        
        # Phase 2: Pure I-optimal point-by-point optimization
        print("Phase 2: Pure I-optimal optimization (minimizing prediction variance)...")
        
        while len(self.design_points) < self.num_runs:
            best_point = None
            best_i_value = -1e10  # Start with very negative value (since we return negative trace)
            
            # Generate many diverse candidates specifically for I-optimality
            # BALANCED APPROACH: Interior points have best I-values, but need edges to avoid singularity
            candidates = []
            
            # Strategy 1: Interior-focused candidates (50% of candidates) - these have BEST I-values!
            for _ in range(1500):
                # Generate well-balanced interior points - these have the best I-values for prediction
                if self.num_variables == 3:
                    # Use systematic interior patterns for 3-component mixtures
                    # Generate around centroid with variations
                    base_props = [1.0/3] * 3  # Start with equal proportions
                    
                    # Add random variations around the center
                    variations = [random.uniform(-0.2, 0.2) for _ in range(3)]
                    # Ensure they sum to 0 (to maintain sum=1 constraint)
                    variations[-1] = -sum(variations[:-1])
                    
                    point = []
                    for i in range(3):
                        prop = base_props[i] + variations[i]
                        point.append(max(0.1, min(0.8, prop)))  # Keep reasonable bounds
                    
                    # Normalize to ensure sum=1
                    total = sum(point)
                    point = [p/total for p in point]
                else:
                    # For other dimensions, use Dirichlet-like distribution favoring interior
                    point = [random.uniform(0.15, 0.7) for _ in range(self.num_variables)]
                    total = sum(point)
                    point = [p/total for p in point]
                
                candidates.append(point)
            
            # Strategy 2: Edge-focused candidates (30% of candidates) - needed for structural diversity
            for _ in range(900):
                # Generate binary mixtures (edge points) - these provide necessary diversity
                point = [0.0] * self.num_variables
                
                # Select 2 random components for binary mixture
                active_indices = random.sample(range(self.num_variables), 2)
                
                # Use diverse ratios for binary mixtures
                ratio_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                ratio = random.choice(ratio_options)
                
                point[active_indices[0]] = ratio
                point[active_indices[1]] = 1.0 - ratio
                
                candidates.append(point)
            
            # Strategy 3: Near-edge candidates (20% of candidates) - bridge between edge and interior
            for _ in range(600):
                # Generate points near edges but with small amount in third component
                point = [0.0] * self.num_variables
                
                # Select 2 dominant components
                active_indices = random.sample(range(self.num_variables), 2)
                
                # Allocate small amount to a third component (makes design more robust)
                if self.num_variables >= 3:
                    third_component = random.choice([i for i in range(self.num_variables) if i not in active_indices])
                    small_allocation = random.uniform(0.05, 0.2)  # 5-20% to third component
                    point[third_component] = small_allocation
                    
                    # Distribute remaining between the two main components
                    remaining = 1.0 - small_allocation
                    ratio = random.uniform(0.3, 0.7)
                    point[active_indices[0]] = remaining * ratio
                    point[active_indices[1]] = remaining * (1.0 - ratio)
                else:
                    # For 2 components, just use edge points
                    ratio = random.uniform(0.3, 0.7)
                    point[active_indices[0]] = ratio
                    point[active_indices[1]] = 1.0 - ratio
                
                candidates.append(point)
            
            # Evaluate all candidates using I-optimality criterion
            for candidate in candidates:
                # Apply distance filtering to avoid clustering
                if self._is_too_close_to_existing_vertices(candidate) or self._is_too_close_to_existing_points(candidate, min_distance=0.05):
                    continue
                    
                test_i_value = self._evaluate_candidate_i_optimality(candidate)
                if test_i_value > best_i_value:
                    best_i_value = test_i_value
                    best_point = candidate
            
            if best_point:
                det = self.add_design_point(best_point)
                
                # Calculate actual I-optimality metrics for reporting
                try:
                    info_matrix = gram_matrix(self.design_matrix)
                    inverse_matrix = matrix_inverse(info_matrix)
                    trace_value = matrix_trace(inverse_matrix)
                    i_efficiency = 1.0 / trace_value if trace_value > 1e-10 else 0.0
                    trace_improvement = -best_i_value / 1000.0  # Convert back from scaled value
                except:
                    i_efficiency = 0.0
                    trace_improvement = 0.0
                
                # Classify point type for reporting
                point_type = self._classify_mixture_point(best_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in best_point]} (det = {det:.3e}, {point_type}, trace = {trace_improvement:.3f})")
            else:
                # Emergency fallback - add random point if no improvement found
                random_point = self._generate_candidate_point()
                det = self.add_design_point(random_point)
                point_type = self._classify_mixture_point(random_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in random_point]} (det = {det:.3e}, {point_type}, fallback)")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        
        # Calculate final I-efficiency
        try:
            info_matrix = gram_matrix(self.design_matrix)
            inverse_matrix = matrix_inverse(info_matrix)
            final_trace = matrix_trace(inverse_matrix)
            final_i_efficiency = 1.0 / final_trace if final_trace > 1e-10 else 0.0
            print(f"\nMixture I-optimal design complete: {len(self.design_points)} points")
            print(f"Final trace(X'X)^(-1) = {final_trace:.6f}, I-efficiency = {final_i_efficiency:.6f}")
            print(f"Determinant = {final_det:.6e}")
        except:
            print(f"\nMixture I-optimal design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det
    
    def _generate_standard_i_optimal(self):
        """Generate I-optimal design for standard experiments"""
        print("Phase 1: Adding factorial corner points...")
        for i in range(min(2**self.num_variables, self.num_runs)):
            point = []
            for j in range(self.num_variables):
                if (i >> j) & 1:
                    point.append(1.0)
                else:
                    point.append(-1.0)
            
            det = self.add_design_point(point)
            print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in point]} (det = {det:.3e})")
        
        # Phase 2: Optimize remaining points using I-optimality criterion
        print("Phase 2: Optimizing remaining points (I-optimality)...")
        while len(self.design_points) < self.num_runs:
            best_point = None
            best_i_value = 0
            
            # Try many random candidates
            for attempt in range(1000):
                candidate = self._generate_candidate_point()
                test_i_value = self._evaluate_candidate_i_optimality(candidate)
                
                if test_i_value > best_i_value:
                    best_i_value = test_i_value
                    best_point = candidate
            
            if best_point:
                det = self.add_design_point(best_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in best_point]} (det = {det:.3e}, I-optimal)")
            else:
                # Add random point if no improvement found
                random_point = self._generate_candidate_point()
                det = self.add_design_point(random_point)
                print(f"  Point {len(self.design_points)}: Random point added (det = {det:.3e})")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"\nStandard I-optimal design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det

    def generate_optimal_design(self, method="d_optimal"):
        """Generate optimal design using specified method"""
        method = method.lower()
        
        if method == "d_optimal":
            return self._generate_d_optimal_design()
        elif method == "i_optimal":
            return self.generate_i_optimal_design()
        elif method == "simplex_lattice":
            return self.generate_simplex_lattice_design()
        elif method == "extreme_vertices":
            return self.generate_extreme_vertices_design()
        elif method == "centroid":
            return self.generate_centroid_design()
        else:
            raise ValueError(f"Unknown method: {method}. Available: d_optimal, i_optimal, simplex_lattice, extreme_vertices, centroid")

    def _generate_d_optimal_design(self):
        """Generate D-optimal design with balanced vertex, edge, and center coverage"""
        print(f"\nGenerating D-optimal {self.design_type} design...")
        
        # Clear existing design
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
        
        if self.design_type == "mixture":
            return self._generate_mixture_d_optimal()
        else:  # standard
            return self._generate_standard_d_optimal()
    
    def _generate_mixture_d_optimal(self):
        """Generate D-optimal design for mixture experiments with balanced coverage"""
        print("Mixture D-optimal design with balanced vertex/edge/center coverage...")
        
        # Phase 1: Add vertices - check if proportional bounds allow pure components
        print("Phase 1: Adding vertices (pure components)...")
        vertices_added = 0
        
        if self.proportional_ranges is not None:
            # With proportional parts constraints, generate feasible vertices instead of pure components
            print("  Proportional bounds detected - generating feasible vertices...")
            for i in range(min(self.num_variables, self.num_runs)):
                # Generate a vertex-like point that respects proportional bounds
                point = self._generate_feasible_vertex(i)
                det = self.add_design_point(point)
                vertices_added += 1
                print(f"  Feasible vertex {vertices_added}: {[f'{x:7.3f}' for x in point]} (det = {det:.3e}) - FEASIBLE VERTEX")
        else:
            # Standard mixture design - use pure components
            for i in range(min(self.num_variables, self.num_runs)):
                point = [0.0] * self.num_variables
                point[i] = 1.0  # Pure component - EXACT vertex
                det = self.add_design_point(point)
                vertices_added += 1
                print(f"  Vertex {vertices_added}: {[f'{x:7.3f}' for x in point]} (det = {det:.3e}) - EXACT VERTEX")
        
        if len(self.design_points) >= self.num_runs:
            return self.determinant_history[-1]
        
        # Phase 2: Add edge points (binary mixtures) - essential for interaction terms
        print("Phase 2: Adding edge points (binary mixtures)...")
        edge_points = []
        
        if self.proportional_ranges is not None:
            # With proportional bounds, generate feasible edge points
            print("  Generating feasible edge points respecting proportional bounds...")
            binary_proportions = [0.5, 0.3, 0.7, 0.4, 0.6]  # Different edge ratios
            
            for i in range(self.num_variables):
                for j in range(i + 1, self.num_variables):
                    for dominant_ratio in binary_proportions:
                        if len(edge_points) >= 20:  # Limit number of candidates
                            break
                        
                        # Generate feasible binary mixture
                        point = [0.0] * self.num_variables
                        
                        # Set other components to minimum proportions
                        total_used = 0.0
                        for k in range(self.num_variables):
                            if k != i and k != j:
                                min_prop, max_prop = self.proportional_ranges[k]
                                point[k] = min_prop + random.uniform(0.0, 0.01)  # Small buffer
                                total_used += point[k]
                        
                        # Distribute remaining between the two dominant components
                        remaining = 1.0 - total_used
                        if remaining > 0:
                            point[i] = remaining * dominant_ratio
                            point[j] = remaining * (1.0 - dominant_ratio)
                            
                            # Normalize to ensure sum=1
                            total = sum(point)
                            if total > 1e-10:
                                point = [x / total for x in point]
                            
                            # Only add if it's a valid proportional candidate
                            if self._is_valid_proportional_candidate(point):
                                edge_points.append(point)
        else:
            # Standard mixture design - use binary mixtures with zeros
            binary_proportions = [0.5, 0.3, 0.7]  # Different edge ratios
            
            for i in range(self.num_variables):
                for j in range(i + 1, self.num_variables):
                    for prop in binary_proportions:
                        if len(self.design_points) >= self.num_runs:
                            break
                        
                        point = [0.0] * self.num_variables
                        point[i] = prop
                        point[j] = 1.0 - prop
                        edge_points.append(point)
        
        # Add edge points with D-optimal selection
        edge_added = 0
        max_edges = min(len(edge_points), max(1, self.num_runs - len(self.design_points) - 2))  # Reserve space for center points
        
        for _ in range(max_edges):
            if len(self.design_points) >= self.num_runs or not edge_points:
                break
                
            best_edge = None
            best_det = self.determinant_history[-1] if self.determinant_history else 0
            best_idx = -1
            
            for idx, edge_point in enumerate(edge_points):
                test_det = self._evaluate_candidate_determinant(edge_point)
                if test_det > best_det:
                    best_det = test_det
                    best_edge = edge_point
                    best_idx = idx
            
            if best_edge is not None:
                det = self.add_design_point(best_edge)
                edge_points.pop(best_idx)
                edge_added += 1
                print(f"  Edge {edge_added}: {[f'{x:7.3f}' for x in best_edge]} (det = {det:.3e})")
            else:
                break
        
        if len(self.design_points) >= self.num_runs:
            return self.determinant_history[-1]
        
        # Phase 3: Add center point (overall centroid)
        if len(self.design_points) < self.num_runs:
            print("Phase 3: Adding center point...")
            centroid = [1.0 / self.num_variables] * self.num_variables
            det = self.add_design_point(centroid)
            print(f"  Centroid: {[f'{x:7.3f}' for x in centroid]} (det = {det:.3e})")
        
        # Phase 4: Optimize remaining points with strategic candidate generation
        print("Phase 4: Optimizing remaining points...")
        while len(self.design_points) < self.num_runs:
            best_point = None
            best_det = self.determinant_history[-1] if self.determinant_history else 0
            
            # Generate candidates respecting proportional bounds if present
            candidates = []
            
            if self.proportional_ranges is not None:
                # With proportional bounds, all candidates must respect bounds
                print("    Generating feasible candidates respecting proportional bounds...")
                
                # Generate many feasible candidates using proportional approach
                for _ in range(1000):
                    candidate = self._generate_proportional_candidate()
                    candidates.append(candidate)
                    
            else:
                # Standard approach without bounds - can use points with zeros
                # 30% vertex-biased (near pure components, but not too close to exact vertices)
                for _ in range(300):
                    point = [0.0] * self.num_variables
                    dominant_comp = random.randint(0, self.num_variables - 1)
                    # Avoid points too close to exact vertices (already added in Phase 1)
                    point[dominant_comp] = random.uniform(0.7, 0.95)  # Stay away from 1.0
                    
                    remaining = 1.0 - point[dominant_comp]
                    if remaining > 0:
                        # Distribute remaining among other components
                        other_comps = [i for i in range(self.num_variables) if i != dominant_comp]
                        random.shuffle(other_comps)
                        
                        for i, comp_idx in enumerate(other_comps[:-1]):
                            if remaining > 0:
                                allocation = random.uniform(0, remaining)
                                point[comp_idx] = allocation
                                remaining -= allocation
                        
                        if other_comps:
                            point[other_comps[-1]] = remaining
                    
                    candidates.append(point)
                
                # 40% edge-biased (binary and ternary mixtures)
                for _ in range(400):
                    point = [0.0] * self.num_variables
                    num_active = random.choice([2, 3]) if self.num_variables >= 3 else 2
                    num_active = min(num_active, self.num_variables)
                    
                    active_indices = random.sample(range(self.num_variables), num_active)
                    
                    # Generate random proportions for active components
                    proportions = [random.random() for _ in range(num_active)]
                    total = sum(proportions)
                    proportions = [p/total for p in proportions]
                    
                    for i, idx in enumerate(active_indices):
                        point[idx] = proportions[i]
                    
                    candidates.append(point)
                
                # 30% center-biased (interior points)
                for _ in range(300):
                    point = self._generate_candidate_point()
                    candidates.append(point)
            
            # Evaluate all candidates
            for candidate in candidates:
                # Skip candidates too close to existing points or vertices
                if self._is_too_close_to_existing_vertices(candidate) or self._is_too_close_to_existing_points(candidate, min_distance=0.08):
                    continue
                    
                test_det = self._evaluate_candidate_determinant(candidate)
                if test_det > best_det:
                    best_det = test_det
                    best_point = candidate
            
            if best_point:
                det = self.add_design_point(best_point)
                improvement = det / self.determinant_history[-2] if len(self.determinant_history) > 1 and self.determinant_history[-2] > 1e-10 else det
                
                # Classify point type for reporting
                point_type = self._classify_mixture_point(best_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in best_point]} (det = {det:.3e}, {point_type}, improvement = {improvement:.3f}x)")
            else:
                # Add random point if no improvement found
                random_point = self._generate_candidate_point()
                det = self.add_design_point(random_point)
                point_type = self._classify_mixture_point(random_point)
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in random_point]} (det = {det:.3e}, {point_type}, random)")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"\nMixture D-optimal design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det
    
    def _generate_standard_d_optimal(self):
        """Generate D-optimal design for standard experiments"""
        print("Phase 1: Adding factorial corner points...")
        for i in range(min(2**self.num_variables, self.num_runs)):
            point = []
            for j in range(self.num_variables):
                if (i >> j) & 1:
                    point.append(1.0)
                else:
                    point.append(-1.0)
            
            det = self.add_design_point(point)
            print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in point]} (det = {det:.3e})")
        
        # Phase 2: Optimize remaining points
        print("Phase 2: Optimizing remaining points...")
        while len(self.design_points) < self.num_runs:
            best_point = None
            best_det = self.determinant_history[-1] if self.determinant_history else 0
            
            # Try many random candidates
            for attempt in range(1000):
                candidate = self._generate_candidate_point()
                test_det = self._evaluate_candidate_determinant(candidate)
                
                if test_det > best_det:
                    best_det = test_det
                    best_point = candidate
            
            if best_point:
                det = self.add_design_point(best_point)
                improvement = det / self.determinant_history[-2] if len(self.determinant_history) > 1 and self.determinant_history[-2] > 1e-10 else det
                print(f"  Point {len(self.design_points)}: {[f'{x:7.3f}' for x in best_point]} (det = {det:.3e}, improvement = {improvement:.3f}x)")
            else:
                # Add random point if no improvement found
                random_point = self._generate_candidate_point()
                det = self.add_design_point(random_point)
                print(f"  Point {len(self.design_points)}: Random point added (det = {det:.3e})")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"\nStandard D-optimal design complete: {len(self.design_points)} points, determinant = {final_det:.6e}")
        
        return final_det
    
    def _classify_mixture_point(self, point, tolerance=0.05):
        """Classify a mixture point as vertex, edge, or interior"""
        # Count components with significant values
        significant_components = sum(1 for x in point if x > tolerance)
        
        if significant_components == 1:
            return "vertex"
        elif significant_components == 2:
            return "edge"
        elif significant_components == 3 and len(point) >= 3:
            return "ternary"
        else:
            return "interior"
    
    def _is_vertex_point(self, point, tolerance=0.02):
        """Check if a point is a vertex (one component ≈ 1.0, others ≈ 0.0)"""
        if self.design_type != "mixture":
            return False
        
        # Check if any component is very close to 1.0
        max_component = max(point)
        if max_component >= (1.0 - tolerance):
            # Check if other components are close to 0
            other_components_sum = sum(x for x in point if x != max_component)
            return other_components_sum <= tolerance
        
        return False
    
    def _is_too_close_to_existing_points(self, candidate_point, min_distance=0.1):
        """Check if candidate point is too close to any existing point"""
        if not self.design_points:
            return False
        
        for existing_point in self.design_points:
            # Calculate Euclidean distance
            distance = sum((a - b)**2 for a, b in zip(candidate_point, existing_point))**0.5
            if distance < min_distance:
                return True
        
        return False
    
    def _is_too_close_to_existing_vertices(self, candidate_point, min_distance=0.15):
        """Check if candidate point is too close to existing vertex points"""
        if self.design_type != "mixture":
            return False
        
        # Check distance to all existing vertex points
        for existing_point in self.design_points:
            if self._is_vertex_point(existing_point):
                # Calculate Euclidean distance
                distance = sum((a - b)**2 for a, b in zip(candidate_point, existing_point))**0.5
                if distance < min_distance:
                    return True
        
        return False
    
    def _remove_near_vertex_duplicates(self, min_distance=0.1):
        """Remove points that are too close to vertex points"""
        if self.design_type != "mixture" or len(self.design_points) <= self.num_variables:
            return
        
        print(f"Checking for near-vertex duplicates (min_distance = {min_distance})...")
        
        # Identify vertex points first
        vertex_indices = []
        for i, point in enumerate(self.design_points):
            if self._is_vertex_point(point):
                vertex_indices.append(i)
        
        print(f"Found {len(vertex_indices)} vertex points: {vertex_indices}")
        
        # Find points to remove (too close to vertices)
        points_to_remove = []
        for i, point in enumerate(self.design_points):
            if i in vertex_indices:
                continue  # Don't remove vertex points themselves
            
            # Check if this point is too close to any vertex
            for vertex_idx in vertex_indices:
                vertex_point = self.design_points[vertex_idx]
                distance = sum((a - b)**2 for a, b in zip(point, vertex_point))**0.5
                
                if distance < min_distance:
                    points_to_remove.append(i)
                    print(f"  Point {i+1} too close to vertex {vertex_idx+1} (distance = {distance:.4f})")
                    break
        
        # Remove duplicates from points_to_remove and sort in reverse order
        points_to_remove = sorted(list(set(points_to_remove)), reverse=True)
        
        if points_to_remove:
            print(f"Removing {len(points_to_remove)} near-vertex points: {[i+1 for i in points_to_remove]}")
            
            # Remove points in reverse order to maintain indices
            for idx in points_to_remove:
                self.design_points.pop(idx)
                self.design_matrix.pop(idx)
            
            # Recalculate determinant history
            self.determinant_history = []
            for i in range(len(self.design_points)):
                current_matrix = self.design_matrix[:i+1]
                info_matrix = gram_matrix(current_matrix)
                det = calculate_determinant(info_matrix)
                self.determinant_history.append(det)
            
            print(f"After removal: {len(self.design_points)} points remaining")
        else:
            print("No near-vertex duplicates found")
    
    def convert_to_parts(self, component_ranges):
        """
        Convert the optimized design to component parts ranges
        
        For MIXTURE designs: Convert from [0,1] simplex space to parts, then normalize
        For STANDARD DOE: Convert from [-1,1] range to parts ranges
        
        Parameters:
        - component_ranges: List of tuples [(min1, max1), (min2, max2), ...] for each component
        
        Returns:
        - design_points_parts: Design points converted to parts ranges
        - design_points_normalized: Normalized design points in parts ranges
        """
        if len(component_ranges) != self.num_variables:
            raise ValueError(f"Component ranges must be provided for all {self.num_variables} variables")
        
        if not self.design_points:
            raise ValueError("No design points available. Generate design first.")
        
        print(f"\n{'='*80}")
        print(f"CONVERTING {self.design_type.upper()} DESIGN TO PARTS SPACE")
        print(f"{'='*80}")
        
        print(f"\nComponent Ranges (Parts):")
        for i, (min_val, max_val) in enumerate(component_ranges):
            print(f"  x{i+1}: [{min_val:.3f}, {max_val:.3f}]")
        
        print(f"\nDesign Type: {self.design_type}")
        
        if self.design_type == "mixture":
            # MIXTURE DESIGN: Convert from [0,1] simplex space to parts
            print("Converting from [0,1] simplex space (sum=1) to parts...")
            design_points_parts = self._convert_mixture_to_parts(component_ranges)
        else:
            # STANDARD DOE: Convert from [-1,1] range to parts
            print("Converting from [-1,1] range to parts...")
            design_points_parts = self._convert_standard_to_parts(component_ranges)
        
        # Calculate normalization factors for parts
        parts_sums = [sum(point) for point in design_points_parts]
        
        # Normalize each point so components sum to 1
        design_points_normalized = []
        for i, (point, total) in enumerate(zip(design_points_parts, parts_sums)):
            if total > 1e-10:  # Avoid division by zero
                normalized_point = [x / total for x in point]
            else:
                # If sum is zero or negative, use equal proportions
                normalized_point = [1.0 / self.num_variables] * self.num_variables
            design_points_normalized.append(normalized_point)
        
        return design_points_parts, design_points_normalized
    
    def _convert_mixture_to_parts(self, component_ranges):
        """Convert mixture design from [0,1] simplex space to parts"""
        # For mixture designs, we need to scale proportions to parts
        # Method: Use the sum of max bounds as total parts budget
        total_parts_budget = sum(max_val for _, max_val in component_ranges)
        
        design_points_parts = []
        for point in self.design_points:
            parts_point = []
            for i, proportion in enumerate(point):
                min_val, max_val = component_ranges[i]
                # Scale proportion to parts based on total budget
                parts_val = proportion * total_parts_budget
                # Ensure it's within bounds
                parts_val = max(min_val, min(max_val, parts_val))
                parts_point.append(parts_val)
            design_points_parts.append(parts_point)
        
        return design_points_parts
    
    def _convert_standard_to_parts(self, component_ranges):
        """Convert standard DOE from [-1,1] range to parts"""
        design_points_parts = []
        for point in self.design_points:
            parts_point = []
            for i, x_std in enumerate(point):
                min_val, max_val = component_ranges[i]
                # Convert from [-1,1] to [min_val, max_val]
                x_parts = min_val + (x_std + 1) * (max_val - min_val) / 2
                parts_point.append(x_parts)
            design_points_parts.append(parts_point)
        
        return design_points_parts
    
    def print_design_summary(self, component_ranges=None):
        """Print comprehensive design summary with optional parts conversion"""
        print(f"\n{'='*80}")
        print(f"OPTIMAL DESIGN SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nModel Specification:")
        print(f"  Variables: {self.num_variables} (x1, x2, ..., x{self.num_variables})")
        print(f"  Design type: {self.design_type}")
        print(f"  Parameters: {self.num_parameters}")
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        d_efficiency = final_det**(1/self.num_parameters) if final_det > 1e-10 else 0.0
        
        print(f"\nDesign Properties:")
        print(f"  Number of runs: {len(self.design_points)}")
        print(f"  Information matrix determinant: {final_det:.6e}")
        print(f"  D-efficiency: {d_efficiency:.6f}")
        print(f"  Singularity: {'Singular (det≈0)' if final_det < 1e-10 else 'Non-singular'}")
        
        if self.design_type == "mixture":
            print(f"\nDesign Points (Simplex Space - sum=1):")
            for i, point in enumerate(self.design_points):
                total = sum(point)
                print(f"  Run {i+1:3d}: [{', '.join(f'{x:7.3f}' for x in point)}] (sum = {total:.6f})")
        else:
            print(f"\nDesign Points (Standard [-1,1] Range):")
            for i, point in enumerate(self.design_points):
                print(f"  Run {i+1:3d}: [{', '.join(f'{x:7.3f}' for x in point)}]")
        
        # Convert to parts if ranges provided
        if component_ranges is not None:
            try:
                design_points_parts, design_points_normalized = self.convert_to_parts(component_ranges)
                
                print(f"\nDesign Points (Parts Ranges):")
                for i, point in enumerate(design_points_parts):
                    print(f"  Run {i+1:3d}: [{', '.join(f'{x:7.3f}' for x in point)}]")
                
                print(f"\nDesign Points (Normalized Parts - Sum = 1):")
                for i, point in enumerate(design_points_normalized):
                    total = sum(point)
                    print(f"  Run {i+1:3d}: [{', '.join(f'{x:7.3f}' for x in point)}] (sum = {total:.3f})")
                
                # Verify normalization
                print(f"\nNormalization Verification:")
                for i, point in enumerate(design_points_normalized):
                    total = sum(point)
                    if abs(total - 1.0) < 1e-6:
                        status = "✓"
                    else:
                        status = "✗"
                    print(f"  Run {i+1:3d}: Sum = {total:.6f} {status}")
                
            except Exception as e:
                print(f"\nError in parts conversion: {e}")

def demonstrate_parts_conversion():
    """Demonstrate the parts conversion functionality"""
    
    print("\n" + "="*80)
    print("OPTIMAL DESIGN WITH PARTS CONVERSION DEMONSTRATION")
    print("="*80)
    
    # Example 1: 3-component mixture with different ranges
    print("\nEXAMPLE 1: 3-Component Mixture Design with Parts Conversion")
    print("="*60)
    
    # Define component ranges (parts constraints)
    component_ranges = [
        (0.0, 1.0),    # Component 1: 0% to 100%
        (0.0, 0.8),    # Component 2: 0% to 80% 
        (0.1, 0.6)     # Component 3: 10% to 60%
    ]
    
    # Create generator for 3 variables, mixture model with parts
    generator = OptimalDesignGenerator(
        num_variables=3,
        design_type="mixture", 
        num_runs=15,
        component_ranges=component_ranges
    )
    
    # Generate optimal design in simplex space
    print("\nStep 1: Generate optimal design in simplex space...")
    final_det = generator.generate_optimal_design()
    
    print(f"\nStep 2: Convert to parts space and normalize...")
    
    # Print summary with parts conversion (uses component_ranges from constructor)
    generator.print_design_summary(component_ranges=component_ranges)

def demonstrate_design_types():
    """Demonstrate the difference between mixture and standard designs"""
    print("DEMONSTRATION: MIXTURE vs STANDARD DESIGNS")
    print("=" * 60)
    
    num_variables = 3
    num_runs = 15
    
    # Test mixture design
    print(f"\nTEST 1: MIXTURE DESIGN")
    print("-" * 40)
    mixture_gen = OptimalDesignGenerator(num_variables, num_runs, "mixture")
    mixture_det = mixture_gen.generate_optimal_design()
    mixture_gen.print_design_summary()
    
    # Test standard design
    print(f"\n\nTEST 2: STANDARD DOE")
    print("-" * 40)
    standard_gen = OptimalDesignGenerator(num_variables, num_runs, "standard")
    standard_det = standard_gen.generate_optimal_design()
    standard_gen.print_design_summary()
    
    # Compare
    print(f"\n\nCOMPARISON:")
    print("-" * 40)
    print(f"Mixture design: {mixture_gen.num_parameters} parameters, det = {mixture_det:.3e}")
    print(f"Standard design: {standard_gen.num_parameters} parameters, det = {standard_det:.3e}")
    print(f"\nNote: These designs serve different purposes and operate in different spaces.")
    print(f"      Direct efficiency comparison is not meaningful.")

if __name__ == "__main__":
    # First demonstrate basic design types
    demonstrate_design_types()
    
    # Then demonstrate parts conversion
    demonstrate_parts_conversion()
