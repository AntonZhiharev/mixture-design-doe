import math
import random
from itertools import combinations_with_replacement, combinations

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

def print_matrix(matrix, title="Matrix"):
    """Print matrix in a formatted way"""
    print(f"\n{title}:")
    for row in matrix:
        print("[" + ", ".join(f"{val:8.4f}" for val in row) + "]")

class OptimalDesignGenerator:
    """
    Class for generating optimal experimental designs for polynomial models
    with maximum Fisher Information Matrix determinant
    """
    
    def __init__(self, num_variables, model_type="linear", num_runs=None, component_ranges=None):
        """
        Initialize the optimal design generator
        
        Parameters:
        - num_variables: Number of input variables (x1, x2, ..., xn)
        - model_type: Type of polynomial model ("linear", "quadratic", "cubic")
        - num_runs: Number of experimental runs (if None, uses minimum required)
        - component_ranges: List of tuples [(min1, max1), (min2, max2), ...] for each component
                           If provided, will be used for parts conversion after optimization
        """
        self.num_variables = num_variables
        self.model_type = model_type.lower()
        self.component_ranges = component_ranges
        
        # Generate polynomial terms for the model
        self.terms = self._generate_polynomial_terms()
        self.num_parameters = len(self.terms)
        
        # Set number of runs
        if num_runs is None:
            self.num_runs = max(self.num_parameters + 2, 2 * self.num_parameters)
        else:
            self.num_runs = max(num_runs, self.num_parameters)
        
        # Initialize design matrix and points
        self.design_points = []
        self.design_matrix = []
        self.information_matrix = []
        self.determinant_history = []
        
        print(f"Optimal Design Generator Initialized:")
        print(f"  Variables: {self.num_variables}")
        print(f"  Model type: {self.model_type}")
        print(f"  Number of parameters: {self.num_parameters}")
        print(f"  Number of runs: {self.num_runs}")
        print(f"  Model terms: {self.terms}")
        if component_ranges:
            print(f"  Component ranges (parts): {component_ranges}")
        else:
            print(f"  Using standard [-1,1] range (parts conversion available later)")
    
    def _generate_polynomial_terms(self):
        """Generate polynomial terms based on model type and number of variables"""
        terms = []
        
        if self.model_type == "linear":
            # Linear model: x1, x2, ..., xn
            for i in range(self.num_variables):
                terms.append(f"x{i+1}")
        
        elif self.model_type == "quadratic":
            # Quadratic model: x1, x2, ..., xn, x1^2, x2^2, ..., xn^2, x1*x2, x1*x3, ..., x(n-1)*xn
            
            # Linear terms
            for i in range(self.num_variables):
                terms.append(f"x{i+1}")
            
            # Quadratic terms
            for i in range(self.num_variables):
                terms.append(f"x{i+1}^2")
            
            # Interaction terms
            for i in range(self.num_variables):
                for j in range(i+1, self.num_variables):
                    terms.append(f"x{i+1}*x{j+1}")
        
        elif self.model_type == "cubic":
            # Cubic model: includes all terms up to degree 3
            
            # Linear terms
            for i in range(self.num_variables):
                terms.append(f"x{i+1}")
            
            # Quadratic terms
            for i in range(self.num_variables):
                terms.append(f"x{i+1}^2")
            
            # Two-way interactions
            for i in range(self.num_variables):
                for j in range(i+1, self.num_variables):
                    terms.append(f"x{i+1}*x{j+1}")
            
            # Cubic terms
            for i in range(self.num_variables):
                terms.append(f"x{i+1}^3")
            
            # Quadratic-linear interactions
            for i in range(self.num_variables):
                for j in range(self.num_variables):
                    if i != j:
                        terms.append(f"x{i+1}^2*x{j+1}")
            
            # Three-way interactions
            for i in range(self.num_variables):
                for j in range(i+1, self.num_variables):
                    for k in range(j+1, self.num_variables):
                        terms.append(f"x{i+1}*x{j+1}*x{k+1}")
        
        else:
            raise ValueError("Model type must be 'linear', 'quadratic', or 'cubic'")
        
        return terms
    
    def _evaluate_polynomial_terms(self, x_values):
        """Evaluate all polynomial terms for given x values"""
        if len(x_values) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {len(x_values)}")
        
        row = []
        
        if self.model_type == "linear":
            # Linear terms
            for i in range(self.num_variables):
                row.append(x_values[i])
        
        elif self.model_type == "quadratic":
            # Linear terms
            for i in range(self.num_variables):
                row.append(x_values[i])
            
            # Quadratic terms
            for i in range(self.num_variables):
                row.append(x_values[i]**2)
            
            # Interaction terms
            for i in range(self.num_variables):
                for j in range(i+1, self.num_variables):
                    row.append(x_values[i] * x_values[j])
        
        elif self.model_type == "cubic":
            # Linear terms
            for i in range(self.num_variables):
                row.append(x_values[i])
            
            # Quadratic terms
            for i in range(self.num_variables):
                row.append(x_values[i]**2)
            
            # Two-way interactions
            for i in range(self.num_variables):
                for j in range(i+1, self.num_variables):
                    row.append(x_values[i] * x_values[j])
            
            # Cubic terms
            for i in range(self.num_variables):
                row.append(x_values[i]**3)
            
            # Quadratic-linear interactions
            for i in range(self.num_variables):
                for j in range(self.num_variables):
                    if i != j:
                        row.append(x_values[i]**2 * x_values[j])
            
            # Three-way interactions
            for i in range(self.num_variables):
                for j in range(i+1, self.num_variables):
                    for k in range(j+1, self.num_variables):
                        row.append(x_values[i] * x_values[j] * x_values[k])
        
        return row
    
    def _generate_candidate_point(self, strategy="random"):
        """Generate a candidate design point"""
        if strategy == "random":
            return [random.uniform(-1.0, 1.0) for _ in range(self.num_variables)]
        
        elif strategy == "factorial":
            # Generate factorial-like points
            if len(self.design_points) < 2**self.num_variables:
                # Generate corner points of hypercube
                point_index = len(self.design_points)
                point = []
                for i in range(self.num_variables):
                    if (point_index >> i) & 1:
                        point.append(1.0)
                    else:
                        point.append(-1.0)
                return point
            else:
                # Fall back to random
                return [random.uniform(-1.0, 1.0) for _ in range(self.num_variables)]
        
        elif strategy == "center_points":
            # Generate points around center
            if len(self.design_points) % 3 == 0:
                return [0.0] * self.num_variables
            else:
                return [random.uniform(-0.5, 0.5) for _ in range(self.num_variables)]
        
        elif strategy == "optimal":
            # Try multiple random points and pick best
            best_point = None
            best_det = 0
            
            for _ in range(50):
                candidate = [random.uniform(-1.0, 1.0) for _ in range(self.num_variables)]
                test_det = self._evaluate_candidate_determinant(candidate)
                if test_det > best_det:
                    best_det = test_det
                    best_point = candidate
            
            return best_point if best_point else [random.uniform(-1.0, 1.0) for _ in range(self.num_variables)]
    
    def _evaluate_candidate_determinant(self, candidate_point):
        """Evaluate determinant if candidate point is added"""
        try:
            # Create test design matrix
            test_design_matrix = self.design_matrix[:]
            test_design_matrix.append(self._evaluate_polynomial_terms(candidate_point))
            
            # Calculate information matrix
            test_info_matrix = gram_matrix(test_design_matrix)
            
            # Calculate determinant
            return calculate_determinant(test_info_matrix)
        except:
            return 0.0
    
    def add_design_point(self, x_values=None, strategy="optimal"):
        """Add a design point to maximize information matrix determinant"""
        if x_values is None:
            # Generate candidate point
            x_values = self._generate_candidate_point(strategy)
        
        # Add to design
        self.design_points.append(x_values[:])
        design_row = self._evaluate_polynomial_terms(x_values)
        self.design_matrix.append(design_row)
        
        # Update information matrix
        self.information_matrix = gram_matrix(self.design_matrix)
        current_det = calculate_determinant(self.information_matrix)
        self.determinant_history.append(current_det)
        
        return current_det
    
    def generate_initial_design(self, strategy="factorial"):
        """Generate initial design to ensure non-singular information matrix"""
        print(f"\nGenerating initial design using '{strategy}' strategy...")
        
        # Clear existing design
        self.design_points = []
        self.design_matrix = []
        self.determinant_history = []
        
        # Add minimum required points
        attempts = 0
        max_attempts = 1000
        
        while len(self.design_points) < self.num_parameters and attempts < max_attempts:
            candidate = self._generate_candidate_point(strategy)
            test_det = self._evaluate_candidate_determinant(candidate)
            
            # Add point if it improves determinant or if we need more points
            if test_det > (self.determinant_history[-1] if self.determinant_history else 0) or len(self.design_points) < self.num_parameters:
                det = self.add_design_point(candidate)
                print(f"  Added point {len(self.design_points)}: {[f'{x:.3f}' for x in candidate]} (det = {det:.6f})")
            
            attempts += 1
        
        if len(self.design_points) < self.num_parameters:
            print(f"Warning: Could not generate full initial design. Only {len(self.design_points)} points added.")
        
        return self.determinant_history[-1] if self.determinant_history else 0
    
    def optimize_design(self, strategy="optimal", max_iterations=None):
        """Optimize design by adding points to maximize determinant"""
        if max_iterations is None:
            max_iterations = self.num_runs - len(self.design_points)
        
        print(f"\nOptimizing design with '{strategy}' strategy...")
        print(f"Target: {self.num_runs} total runs, Current: {len(self.design_points)} runs")
        
        iteration = 0
        while len(self.design_points) < self.num_runs and iteration < max_iterations:
            initial_det = self.determinant_history[-1] if self.determinant_history else 0
            
            # Try multiple candidates and pick the best
            best_point = None
            best_det = initial_det
            
            for attempt in range(100):
                candidate = self._generate_candidate_point(strategy)
                test_det = self._evaluate_candidate_determinant(candidate)
                
                if test_det > best_det:
                    best_det = test_det
                    best_point = candidate
            
            if best_point and best_det > initial_det:
                det = self.add_design_point(best_point)
                improvement = det / initial_det if initial_det > 1e-10 else det
                print(f"  Run {len(self.design_points)}: {[f'{x:.3f}' for x in best_point]} (det = {det:.6f}, improvement = {improvement:.3f}x)")
            else:
                # Add random point if no improvement found
                det = self.add_design_point(strategy="random")
                print(f"  Run {len(self.design_points)}: Random point added (det = {det:.6f})")
            
            iteration += 1
        
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        initial_det = self.determinant_history[0] if self.determinant_history else 1e-10
        total_improvement = final_det / initial_det if initial_det > 1e-10 else final_det
        
        print(f"\nOptimization complete!")
        print(f"  Final determinant: {final_det:.6f}")
        print(f"  Total improvement: {total_improvement:.3f}x")
        
        return final_det
    
    def generate_optimal_design(self, initial_strategy="factorial", optimization_strategy="optimal"):
        """Generate complete optimal design"""
        print(f"="*80)
        print(f"GENERATING OPTIMAL DESIGN")
        print(f"="*80)
        
        # Generate initial design
        initial_det = self.generate_initial_design(initial_strategy)
        
        # Optimize design
        final_det = self.optimize_design(optimization_strategy)
        
        return final_det
    
    def convert_to_parts(self, component_ranges):
        """
        Convert the optimized design from standard [-1,1] range to component parts ranges
        
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
        print(f"CONVERTING DESIGN TO PARTS SPACE")
        print(f"{'='*80}")
        
        print(f"\nComponent Ranges (Parts):")
        for i, (min_val, max_val) in enumerate(component_ranges):
            print(f"  x{i+1}: [{min_val:.3f}, {max_val:.3f}]")
        
        # Convert from [-1,1] to parts ranges
        design_points_parts = []
        for point in self.design_points:
            parts_point = []
            for i, x_std in enumerate(point):
                min_val, max_val = component_ranges[i]
                # Convert from [-1,1] to [min_val, max_val]
                x_parts = min_val + (x_std + 1) * (max_val - min_val) / 2
                parts_point.append(x_parts)
            design_points_parts.append(parts_point)
        
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
    
    def print_design_summary(self, component_ranges=None):
        """Print comprehensive design summary with optional parts conversion"""
        print(f"\n{'='*80}")
        print(f"OPTIMAL DESIGN SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nModel Specification:")
        print(f"  Variables: {self.num_variables} (x1, x2, ..., x{self.num_variables})")
        print(f"  Model type: {self.model_type}")
        print(f"  Parameters: {self.num_parameters}")
        print(f"  Model terms: {', '.join(self.terms)}")
        
        print(f"\nDesign Properties (Standard [-1,1] Range):")
        print(f"  Number of runs: {len(self.design_points)}")
        final_det = self.determinant_history[-1] if self.determinant_history else 0
        print(f"  Information Matrix determinant: {final_det:.6f}")
        d_efficiency = final_det**(1/self.num_parameters)
        print(f"  D-efficiency: {d_efficiency:.6f}")
        print(f"  * D-efficiency calculated for standard [-1,1] component values")
        
        print(f"\nDesign Points (Standard [-1,1] Range):")
        for i, point in enumerate(self.design_points):
            print(f"  Run {i+1:2d}: [{', '.join(f'{x:7.3f}' for x in point)}]")
        
        # Convert to parts if ranges provided
        if component_ranges is not None:
            try:
                design_points_parts, design_points_normalized = self.convert_to_parts(component_ranges)
                
                print(f"\nDesign Points (Parts Ranges):")
                for i, point in enumerate(design_points_parts):
                    print(f"  Run {i+1:2d}: [{', '.join(f'{x:7.3f}' for x in point)}]")
                
                print(f"\nDesign Points (Normalized Parts - Sum = 1):")
                for i, point in enumerate(design_points_normalized):
                    total = sum(point)
                    print(f"  Run {i+1:2d}: [{', '.join(f'{x:7.3f}' for x in point)}] (sum = {total:.3f})")
                
                # Verify normalization
                print(f"\nNormalization Verification:")
                for i, point in enumerate(design_points_normalized):
                    total = sum(point)
                    if abs(total - 1.0) < 1e-6:
                        status = "✓"
                    else:
                        status = "✗"
                    print(f"  Run {i+1:2d}: Sum = {total:.6f} {status}")
                
            except Exception as e:
                print(f"\nError in parts conversion: {e}")
        
        if len(self.design_matrix) > 0:
            print_matrix(self.design_matrix, "Design Matrix X (Standard Range)")
        
        if len(self.information_matrix) > 0:
            print_matrix(self.information_matrix, "Information Matrix (X^T * X)")
        
        print(f"\nDeterminant History:")
        for i, det in enumerate(self.determinant_history):
            if i == 0:
                print(f"  Initial: {det:.6f}")
            else:
                improvement = det / self.determinant_history[i-1] if self.determinant_history[i-1] > 1e-10 else det
                print(f"  Run {i+1}: {det:.6f} (×{improvement:.3f})")

def demonstrate_optimal_design_generator():
    """Demonstrate the optimal design generator for different scenarios"""
    
    print("OPTIMAL EXPERIMENTAL DESIGN GENERATOR DEMONSTRATION")
    print("="*80)
    
    # Test cases
    test_cases = [
        {"variables": 2, "model": "linear", "runs": 6},
        {"variables": 2, "model": "quadratic", "runs": 10},
        {"variables": 3, "model": "linear", "runs": 8},
        {"variables": 3, "model": "quadratic", "runs": 15},
        {"variables": 2, "model": "cubic", "runs": 12}
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i+1}: {case['variables']} variables, {case['model']} model, {case['runs']} runs")
        print(f"{'='*80}")
        
        # Create and optimize design
        generator = OptimalDesignGenerator(
            num_variables=case['variables'],
            model_type=case['model'],
            num_runs=case['runs']
        )
        
        # Generate optimal design
        final_det = generator.generate_optimal_design()
        
        # Print summary
        generator.print_design_summary()

def demonstrate_parts_conversion():
    """Demonstrate the parts conversion functionality"""
    
    print("\n" + "="*80)
    print("OPTIMAL DESIGN WITH PARTS CONVERSION DEMONSTRATION")
    print("="*80)
    
    # Example 1: 3-component mixture with different ranges
    print("\nEXAMPLE 1: 3-Component Mixture Design")
    print("="*50)
    
    # Define component ranges (parts constraints)
    component_ranges = [
        (0.0, 1.0),    # Component 1: 0% to 100%
        (0.0, 1.0),    # Component 2: 0% to 80% 
        (0.0, 0.1)     # Component 3: 10% to 60%
    ]
    
    # Create generator for 3 variables, quadratic model with parts
    generator = OptimalDesignGenerator(
        num_variables=3,
        model_type="quadratic", 
        num_runs=30,
        component_ranges=component_ranges
    )
    
    # Generate optimal design in standard [-1,1] range
    print("\nStep 1: Generate optimal design in standard [-1,1] range...")
    final_det = generator.generate_optimal_design()
    
    print(f"\nStep 2: Convert to parts space and normalize...")
    
    # Print summary with parts conversion (uses component_ranges from constructor)
    generator.print_design_summary(component_ranges=component_ranges)
    
    # Example 2: 2-component system with specific ranges
    print("\n" + "="*80)
    print("EXAMPLE 2: 2-Component System with Specific Ranges")
    print("="*80)
    
    # Define different component ranges
    component_ranges2 = [
        (0.0, 1.0),    # Component 1: 20% to 80%
        (0.0, 1.0),     # Component 2: 10% to 90%
        (0.0, 0.1)
    ]
    
    # Create generator for 2 variables, linear model
    generator2 = OptimalDesignGenerator(
        num_variables=3,
        model_type="linear",
        num_runs=12,
        component_ranges=component_ranges2
    )
    
    # Generate optimal design
    print("\nStep 1: Generate optimal design in standard [-1,1] range...")
    final_det2 = generator2.generate_optimal_design()
    
    print(f"\nStep 2: Convert to parts space and normalize...")
    
    # Print summary with parts conversion
    generator2.print_design_summary(component_ranges=component_ranges2)

def demonstrate_conversion_math():
    """Demonstrate the mathematical conversion process step by step"""
    
    print("\n" + "="*80)
    print("MATHEMATICAL CONVERSION PROCESS DEMONSTRATION")
    print("="*80)
    
    # Example conversion
    print("\nExample: Converting point from [-1,1] to parts range")
    print("-" * 50)
    
    # Standard point in [-1,1] range
    std_point = [-0.487, 0.234]
    component_ranges = [(0.0, 1.0), (0.2, 0.6)]
    
    print(f"Standard point ([-1,1] range): {std_point}")
    print(f"Component ranges: {component_ranges}")
    
    # Convert each component
    parts_point = []
    for i, (x_std, (min_val, max_val)) in enumerate(zip(std_point, component_ranges)):
        # Formula: x_parts = min_val + (x_std + 1) * (max_val - min_val) / 2
        x_parts = min_val + (x_std + 1) * (max_val - min_val) / 2
        parts_point.append(x_parts)
        
        print(f"\nComponent {i+1}:")
        print(f"  Standard value: {x_std:.3f}")
        print(f"  Range: [{min_val:.3f}, {max_val:.3f}]")
        print(f"  Conversion: {min_val:.3f} + ({x_std:.3f} + 1) * ({max_val:.3f} - {min_val:.3f}) / 2")
        print(f"  Parts value: {x_parts:.3f}")
    
    print(f"\nParts point: {[f'{x:.3f}' for x in parts_point]}")
    
    # Normalize
    total = sum(parts_point)
    normalized_point = [x / total for x in parts_point]
    
    print(f"Sum of parts: {total:.3f}")
    print(f"Normalized point: {[f'{x:.3f}' for x in normalized_point]}")
    print(f"Sum of normalized: {sum(normalized_point):.3f}")
    
    # Verify your example
    print("\n" + "-" * 50)
    print("Verifying your example: x1 = -0.487 → range [0,1]")
    x_std = -0.487
    min_val, max_val = 0.0, 1.0
    x_parts = min_val + (x_std + 1) * (max_val - min_val) / 2
    print(f"x_parts = 0.0 + (-0.487 + 1) * (1.0 - 0.0) / 2")
    print(f"x_parts = 0.0 + 0.513 * 1.0 / 2")
    print(f"x_parts = 0.0 + 0.2565")
    print(f"x_parts = {x_parts:.4f}")
    print(f"✓ Matches your expected result: 0.2565")

if __name__ == "__main__":
    demonstrate_optimal_design_generator()
