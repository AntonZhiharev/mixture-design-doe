"""
Mathematical Utilities for DOE
==============================

This module contains all shared mathematical functions used across different
design of experiments classes. Extracted from various classes to eliminate
code duplication and improve maintainability.

Functions include:
- Matrix operations (determinant, inverse, transpose, etc.)
- Mathematical calculations for optimal design
- Efficiency calculations
- Model term evaluation
"""

import math
import random
import numpy as np
from typing import List, Tuple, Union


def calculate_determinant(matrix: List[List[float]]) -> float:
    """
    Calculate determinant using Gaussian elimination
    
    Parameters:
    -----------
    matrix : List[List[float]]
        Square matrix as list of lists
        
    Returns:
    --------
    float
        Determinant value
    """
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


def matrix_multiply(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """
    Multiply two matrices A and B
    
    Parameters:
    -----------
    A : List[List[float]]
        First matrix
    B : List[List[float]]
        Second matrix
        
    Returns:
    --------
    List[List[float]]
        Product matrix A * B
    """
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


def transpose_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """
    Transpose a matrix
    
    Parameters:
    -----------
    matrix : List[List[float]]
        Input matrix
        
    Returns:
    --------
    List[List[float]]
        Transposed matrix
    """
    if not matrix:
        return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def gram_matrix(A: List[List[float]]) -> List[List[float]]:
    """
    Calculate Gram matrix A^T * A
    
    Parameters:
    -----------
    A : List[List[float]]
        Input matrix
        
    Returns:
    --------
    List[List[float]]
        Gram matrix A^T * A
    """
    A_T = transpose_matrix(A)
    return matrix_multiply(A_T, A)


def matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
    """
    Calculate matrix inverse using Gauss-Jordan elimination
    
    Parameters:
    -----------
    matrix : List[List[float]]
        Square matrix to invert
        
    Returns:
    --------
    List[List[float]]
        Inverse matrix
        
    Raises:
    -------
    ValueError
        If matrix is singular
    """
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


def matrix_trace(matrix: List[List[float]]) -> float:
    """
    Calculate trace (sum of diagonal elements) of a matrix
    
    Parameters:
    -----------
    matrix : List[List[float]]
        Square matrix
        
    Returns:
    --------
    float
        Trace of the matrix
    """
    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("Matrix must be square for trace calculation")
    
    trace = 0.0
    for i in range(len(matrix)):
        trace += matrix[i][i]
    
    return trace


def evaluate_mixture_model_terms(x_values: List[float], model_type: str) -> List[float]:
    """
    Evaluate mixture model terms (Scheffé canonical polynomials)
    
    For mixture designs where sum(xi) = 1:
    - Linear: x1, x2, ..., xk
    - Quadratic: x1, x2, ..., xk, x1*x2, x1*x3, ..., x(k-1)*xk (NO pure quadratic terms)
    - Cubic: adds three-way interactions xi*xj*xk
    - Quartic: adds four-way interactions xi*xj*xk*xl
    
    Parameters:
    -----------
    x_values : List[float]
        Component values (proportions summing to 1)
    model_type : str
        Model type ("linear", "quadratic", "cubic", "quartic")
        
    Returns:
    --------
    List[float]
        Model terms for the point
    """
    num_variables = len(x_values)
    row = []
    
    # Linear terms: component proportions
    for i in range(num_variables):
        row.append(x_values[i])
    
    if model_type in ["quadratic", "cubic", "quartic"]:
        # Two-way interaction terms: xi*xj for all i < j
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                row.append(x_values[i] * x_values[j])
    
    if model_type in ["cubic", "quartic"]:
        # Three-way interaction terms: xi*xj*xk for all i < j < k
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                for k in range(j+1, num_variables):
                    row.append(x_values[i] * x_values[j] * x_values[k])
    
    if model_type == "quartic":
        # Four-way interaction terms: xi*xj*xk*xl for all i < j < k < l
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                for k in range(j+1, num_variables):
                    for l in range(k+1, num_variables):
                        row.append(x_values[i] * x_values[j] * x_values[k] * x_values[l])
    
    return row


def evaluate_standard_model_terms(x_values: List[float], model_type: str) -> List[float]:
    """
    Evaluate standard model terms
    
    For standard DOE with independent variables:
    - Linear: x1, x2, ..., xk
    - Quadratic: adds x1², x2², ..., xk², x1*x2, x1*x3, ..., x(k-1)*xk
    - Cubic: adds x1³, x2³, ..., xk³, x1²*x2, x1*x2², ..., x1*x2*x3, ...
    - Quartic: adds x1⁴, x2⁴, ..., xk⁴, x1³*x2, x1²*x2², ..., x1*x2*x3*x4, ...
    
    Parameters:
    -----------
    x_values : List[float]
        Variable values
    model_type : str
        Model type ("linear", "quadratic", "cubic", "quartic")
        
    Returns:
    --------
    List[float]
        Model terms for the point
    """
    num_variables = len(x_values)
    row = []
    
    # Linear terms
    for i in range(num_variables):
        row.append(x_values[i])
    
    if model_type in ["quadratic", "cubic", "quartic"]:
        # Pure quadratic terms
        for i in range(num_variables):
            row.append(x_values[i]**2)
        
        # Two-way interaction terms
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                row.append(x_values[i] * x_values[j])
    
    if model_type in ["cubic", "quartic"]:
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
    
    if model_type == "quartic":
        # Quartic terms: xi⁴
        for i in range(num_variables):
            row.append(x_values[i]**4)
        
        # Cubic-linear interaction terms: xi³*xj for all i≠j
        for i in range(num_variables):
            for j in range(num_variables):
                if i != j:
                    row.append(x_values[i]**3 * x_values[j])
        
        # Quadratic-quadratic interaction terms: xi²*xj² for all i<j
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                row.append(x_values[i]**2 * x_values[j]**2)
        
        # Quadratic-linear-linear interaction terms: xi²*xj*xk for all i<j<k
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                for k in range(j+1, num_variables):
                    row.append(x_values[i]**2 * x_values[j] * x_values[k])
                    row.append(x_values[i] * x_values[j]**2 * x_values[k])
                    row.append(x_values[i] * x_values[j] * x_values[k]**2)
        
        # Four-way interaction terms: xi*xj*xk*xl for all i < j < k < l
        for i in range(num_variables):
            for j in range(i+1, num_variables):
                for k in range(j+1, num_variables):
                    for l in range(k+1, num_variables):
                        row.append(x_values[i] * x_values[j] * x_values[k] * x_values[l])
    
    return row


def calculate_d_efficiency(design: Union[List[List[float]], np.ndarray], 
                          model_type: str = "quadratic",
                          design_type: str = "mixture") -> float:
    """
    Calculate D-efficiency of a design
    
    Parameters:
    -----------
    design : Union[List[List[float]], np.ndarray]
        Design matrix (n_runs x n_factors)
    model_type : str
        Model type ("linear", "quadratic", "cubic")
    design_type : str
        Design type ("mixture" or "standard")
        
    Returns:
    --------
    float
        D-efficiency value
    """
    try:
        # Convert to list if numpy array
        if isinstance(design, np.ndarray):
            design = design.tolist()
        
        # Build model matrix
        model_matrix = []
        for point in design:
            if design_type == "mixture":
                terms = evaluate_mixture_model_terms(point, model_type)
            else:
                terms = evaluate_standard_model_terms(point, model_type)
            model_matrix.append(terms)
        
        # Calculate information matrix
        info_matrix = gram_matrix(model_matrix)
        
        # Calculate determinant
        det_value = calculate_determinant(info_matrix)
        
        # Calculate D-efficiency
        n_runs = len(design)
        n_params = len(model_matrix[0]) if model_matrix else 1
        
        if det_value > 0 and n_params > 0:
            d_efficiency = (det_value / n_runs) ** (1 / n_params)
        else:
            d_efficiency = 0.0
        
        return d_efficiency
    except Exception:
        return 0.0


def calculate_i_efficiency(design: Union[List[List[float]], np.ndarray],
                          candidates: Union[List[List[float]], np.ndarray],
                          model_type: str = "quadratic",
                          design_type: str = "mixture") -> float:
    """
    Calculate I-efficiency of a design
    
    I-efficiency is related to the average prediction variance over the design region.
    
    Parameters:
    -----------
    design : Union[List[List[float]], np.ndarray]
        Design matrix (n_runs x n_factors)
    candidates : Union[List[List[float]], np.ndarray]
        Candidate points for prediction variance calculation
    model_type : str
        Model type ("linear", "quadratic", "cubic")
    design_type : str
        Design type ("mixture" or "standard")
        
    Returns:
    --------
    float
        I-efficiency value
    """
    try:
        # Convert to lists if numpy arrays
        if isinstance(design, np.ndarray):
            design = design.tolist()
        if isinstance(candidates, np.ndarray):
            candidates = candidates.tolist()
        
        # Build model matrix for design
        design_matrix = []
        for point in design:
            if design_type == "mixture":
                terms = evaluate_mixture_model_terms(point, model_type)
            else:
                terms = evaluate_standard_model_terms(point, model_type)
            design_matrix.append(terms)
        
        # Calculate information matrix
        info_matrix = gram_matrix(design_matrix)
        
        # Calculate inverse
        try:
            info_inverse = matrix_inverse(info_matrix)
        except ValueError:
            return 0.0  # Singular matrix
        
        # Calculate average prediction variance over candidates
        total_pred_var = 0.0
        n_candidates = len(candidates)
        
        # Use subset of candidates for computational efficiency
        step = max(1, n_candidates // 100)  # Use at most 100 points
        
        for i in range(0, n_candidates, step):
            point = candidates[i]
            
            if design_type == "mixture":
                terms = evaluate_mixture_model_terms(point, model_type)
            else:
                terms = evaluate_standard_model_terms(point, model_type)
            
            # Calculate prediction variance: x^T * (X^T X)^-1 * x
            pred_var = 0.0
            for j in range(len(terms)):
                for k in range(len(terms)):
                    pred_var += terms[j] * info_inverse[j][k] * terms[k]
            
            total_pred_var += pred_var
        
        avg_pred_var = total_pred_var / (n_candidates // step)
        
        # I-efficiency is inverse of average prediction variance
        i_efficiency = 1.0 / avg_pred_var if avg_pred_var > 0 else 0.0
        
        return i_efficiency
    except Exception:
        return 0.0


def latin_hypercube_sampling(n_samples: int, n_dims: int) -> List[List[float]]:
    """
    Generate Latin Hypercube Sampling points in [0,1]^n_dims
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_dims : int
        Number of dimensions
        
    Returns:
    --------
    List[List[float]]
        LHS samples in [0,1] space
    """
    samples = [[0.0] * n_dims for _ in range(n_samples)]
    
    for i in range(n_dims):
        # Create stratified intervals
        intervals = [j / n_samples for j in range(n_samples)]
        # Add random jitter within each interval
        jitter = [random.uniform(0, 1/n_samples) for _ in range(n_samples)]
        stratified_samples = [intervals[j] + jitter[j] for j in range(n_samples)]
        # Random permutation to break correlation between dimensions
        random.shuffle(stratified_samples)
        
        for j in range(n_samples):
            samples[j][i] = stratified_samples[j]
    
    return samples


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Parameters:
    -----------
    point1 : List[float]
        First point
    point2 : List[float]
        Second point
        
    Returns:
    --------
    float
        Euclidean distance
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimensions")
    
    sum_sq = sum((a - b)**2 for a, b in zip(point1, point2))
    return math.sqrt(sum_sq)


def normalize_to_simplex(point: List[float]) -> List[float]:
    """
    Normalize a point to lie on the unit simplex (sum = 1)
    
    Parameters:
    -----------
    point : List[float]
        Point to normalize
        
    Returns:
    --------
    List[float]
        Normalized point summing to 1
    """
    total = sum(point)
    if total <= 1e-10:
        # Return equal proportions if sum is zero
        n = len(point)
        return [1.0 / n] * n
    
    return [x / total for x in point]


# Backward compatibility functions
def evaluate_mixture_quadratic_terms(x_values: List[float]) -> List[float]:
    """Backward compatibility function for quadratic mixture terms"""
    return evaluate_mixture_model_terms(x_values, "quadratic")


def evaluate_standard_quadratic_terms(x_values: List[float]) -> List[float]:
    """Backward compatibility function for quadratic standard terms"""
    return evaluate_standard_model_terms(x_values, "quadratic")
