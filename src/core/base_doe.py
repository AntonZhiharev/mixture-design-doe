"""
Base Design of Experiments (DOE) functionality
Includes D-optimal and I-optimal design generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import det, inv
from sklearn.preprocessing import PolynomialFeatures
import itertools
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class OptimalDOE:
    """
    Class for generating optimal experimental designs
    """
    
    def __init__(self, n_factors: int, factor_ranges: List[Tuple[float, float]] = None):
        """
        Initialize the DOE generator
        
        Parameters:
        n_factors: Number of factors
        factor_ranges: List of (min, max) tuples for each factor. If None, uses [-1, 1] for all
        """
        self.n_factors = n_factors
        if factor_ranges is None:
            self.factor_ranges = [(-1, 1)] * n_factors
        else:
            self.factor_ranges = factor_ranges
            
        # Validate factor ranges
        if len(self.factor_ranges) != n_factors:
            raise ValueError("Number of factor ranges must match number of factors")
    
    def generate_candidate_points(self, n_candidates: int = 1000) -> np.ndarray:
        """
        Generate candidate points for optimal design selection
        """
        candidates = np.random.uniform(0, 1, (n_candidates, self.n_factors))
        
        # Scale to actual factor ranges
        for i in range(self.n_factors):
            min_val, max_val = self.factor_ranges[i]
            candidates[:, i] = candidates[:, i] * (max_val - min_val) + min_val
            
        return candidates
    
    def create_model_matrix(self, X: np.ndarray, model_order: int = 2) -> np.ndarray:
        """
        Create model matrix for polynomial regression
        
        Parameters:
        X: Design matrix (n_runs x n_factors)
        model_order: Order of polynomial model (1=linear, 2=quadratic)
        """
        if model_order == 1:
            # Linear model: 1 + x1 + x2 + ... + xn
            poly = PolynomialFeatures(degree=1, include_bias=True)
        elif model_order == 2:
            # Quadratic model: 1 + x1 + x2 + ... + x1^2 + x2^2 + ... + x1*x2 + ...
            poly = PolynomialFeatures(degree=2, include_bias=True)
        else:
            raise ValueError("Model order must be 1 or 2")
            
        return poly.fit_transform(X)
    
    def d_efficiency(self, X: np.ndarray, model_order: int = 2) -> float:
        """
        Calculate D-efficiency of design
        """
        try:
            model_matrix = self.create_model_matrix(X, model_order)
            XtX = np.dot(model_matrix.T, model_matrix)
            
            # Check for singularity
            if np.linalg.cond(XtX) > 1e12:
                return -1e6
                
            det_XtX = det(XtX)
            n_runs = X.shape[0]
            n_params = model_matrix.shape[1]
            
            # D-efficiency
            d_eff = (det_XtX / n_runs) ** (1/n_params)
            return d_eff
        except:
            return -1e6
    
    def i_efficiency(self, X: np.ndarray, model_order: int = 2, 
                    integration_points: np.ndarray = None) -> float:
        """
        Calculate I-efficiency (average prediction variance)
        """
        try:
            model_matrix = self.create_model_matrix(X, model_order)
            XtX = np.dot(model_matrix.T, model_matrix)
            
            # Check for singularity
            if np.linalg.cond(XtX) > 1e12:
                return 1e6
                
            XtX_inv = inv(XtX)
            
            # Generate integration points if not provided
            if integration_points is None:
                integration_points = self.generate_candidate_points(500)
            
            # Calculate average prediction variance
            total_pred_var = 0
            for point in integration_points:
                x_vec = self.create_model_matrix(point.reshape(1, -1), model_order)
                pred_var = np.dot(np.dot(x_vec, XtX_inv), x_vec.T)[0, 0]
                total_pred_var += pred_var
                
            avg_pred_var = total_pred_var / len(integration_points)
            return avg_pred_var
        except:
            return 1e6
    
    def generate_d_optimal(self, n_runs: int, model_order: int = 2, 
                          max_iter: int = 1000, random_seed: int = None) -> np.ndarray:
        """
        Generate D-optimal design using coordinate exchange algorithm
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Initialize with random design
        candidates = self.generate_candidate_points(n_runs * 10)
        current_design = candidates[:n_runs].copy()
        
        best_d_eff = self.d_efficiency(current_design, model_order)
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                best_point = current_design[i].copy()
                best_eff = best_d_eff
                
                # Try replacing point i with candidates
                for candidate in candidates[n_runs:n_runs+50]:  # Try subset of candidates
                    temp_design = current_design.copy()
                    temp_design[i] = candidate
                    
                    temp_eff = self.d_efficiency(temp_design, model_order)
                    
                    if temp_eff > best_eff:
                        best_eff = temp_eff
                        best_point = candidate.copy()
                        improved = True
                
                if improved:
                    current_design[i] = best_point
                    best_d_eff = best_eff
            
            if not improved:
                break
                
        return current_design
    
    def generate_i_optimal(self, n_runs: int, model_order: int = 2, 
                          max_iter: int = 1000, random_seed: int = None) -> np.ndarray:
        """
        Generate I-optimal design using coordinate exchange algorithm
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Initialize with random design
        candidates = self.generate_candidate_points(n_runs * 10)
        current_design = candidates[:n_runs].copy()
        
        # Generate integration points once
        integration_points = self.generate_candidate_points(200)
        
        best_i_eff = self.i_efficiency(current_design, model_order, integration_points)
        
        for iteration in range(max_iter):
            improved = False
            
            for i in range(n_runs):
                best_point = current_design[i].copy()
                best_eff = best_i_eff
                
                # Try replacing point i with candidates
                for candidate in candidates[n_runs:n_runs+50]:  # Try subset of candidates
                    temp_design = current_design.copy()
                    temp_design[i] = candidate
                    
                    temp_eff = self.i_efficiency(temp_design, model_order, integration_points)
                    
                    if temp_eff < best_eff:  # Lower is better for I-optimal
                        best_eff = temp_eff
                        best_point = candidate.copy()
                        improved = True
                
                if improved:
                    current_design[i] = best_point
                    best_i_eff = best_eff
            
            if not improved:
                break
                
        return current_design
    
    def evaluate_design(self, X: np.ndarray, model_order: int = 2) -> dict:
        """
        Evaluate design properties
        """
        d_eff = self.d_efficiency(X, model_order)
        i_eff = self.i_efficiency(X, model_order)
        
        return {
            'n_runs': X.shape[0],
            'n_factors': X.shape[1],
            'model_order': model_order,
            'd_efficiency': d_eff,
            'i_efficiency': i_eff,
            'design_matrix': X
        }
    
    def plot_design_2d(self, X: np.ndarray, title: str = "Experimental Design"):
        """
        Plot 2D design (for 2 factors only)
        """
        if self.n_factors != 2:
            print("2D plotting only available for 2 factors")
            return
            
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c='red', s=100, alpha=0.7, edgecolors='black')
        plt.xlabel(f'Factor 1 (Range: {self.factor_ranges[0]})')
        plt.ylabel(f'Factor 2 (Range: {self.factor_ranges[1]})')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add point labels
        for i, (x, y) in enumerate(X):
            plt.annotate(f'{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.show()


def multiple_response_analysis(designs: List[np.ndarray], 
                             response_functions: List[callable],
                             design_names: List[str]) -> pd.DataFrame:
    """
    Analyze multiple response functions for different designs
    
    Parameters:
    designs: List of design matrices
    response_functions: List of functions that take X and return response values
    design_names: Names of the designs
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
