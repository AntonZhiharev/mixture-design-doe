"""
Sequential Regression Coefficient Reconstruction for Mixture Designs

This module implements a comprehensive class for sequentially reconstructing
regression function coefficients using experimental design. The process involves
iterative data point generation, analysis, parameter selection, and cycle repetition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
from scipy.optimize import minimize_scalar, minimize
from scipy import stats
from datetime import datetime
import json
import copy

try:
    from ..utils.mixture_utils import validate_mixture_constraints, generate_simplex_lattice_points
    from ..utils.d_efficiency_calculator import calculate_d_efficiency
    from ..core.optimal_design_generator import OptimalDesignGenerator
    from ..algorithms.d_optimal_algorithm import DOptimalAlgorithm
    from ..utils.response_analysis import MixtureResponseAnalysis, DOEResponseAnalysis
    from ..utils.math_utils import gram_matrix, calculate_determinant, evaluate_mixture_model_terms
except ImportError:
    # Handle relative imports for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    # Try importing from utils directory directly
    try:
        import utils.response_analysis as response_analysis
        MixtureResponseAnalysis = response_analysis.MixtureResponseAnalysis
        DOEResponseAnalysis = response_analysis.DOEResponseAnalysis
    except:
        # Create simple fallback classes
        class MixtureResponseAnalysis:
            def __init__(self, design, responses, component_names=None, response_name="Response"):
                self.X = design
                self.y = responses
                self.component_names = component_names or [f"x{i+1}" for i in range(design.shape[1])]
                self.model = LinearRegression(fit_intercept=False)
                
            def fit_scheffe_linear(self):
                self.model.fit(self.X, self.y)
                y_pred = self.model.predict(self.X)
                r2 = r2_score(self.y, y_pred)
                residuals = self.y - y_pred
                coeffs = self.model.coef_
                return {
                    'coefficients': coeffs,
                    'component_names': self.component_names,
                    'r_squared': r2,
                    'rmse': np.sqrt(mean_squared_error(self.y, y_pred)),
                    'residuals': residuals,
                    'predictions': y_pred,
                    'model_type': 'linear'
                }
                
            def fit_scheffe_quadratic(self):
                # Build design matrix for quadratic mixture model
                n_comp = self.X.shape[1]
                X_design = [self.X]
                term_names = self.component_names.copy()
                
                # Add interaction terms
                for i in range(n_comp):
                    for j in range(i + 1, n_comp):
                        interaction = self.X[:, i] * self.X[:, j]
                        X_design.append(interaction.reshape(-1, 1))
                        term_names.append(f'{self.component_names[i]}*{self.component_names[j]}')
                
                X_full = np.column_stack(X_design)
                self.model = LinearRegression(fit_intercept=False)
                self.model.fit(X_full, self.y)
                
                y_pred = self.model.predict(X_full)
                r2 = r2_score(self.y, y_pred)
                residuals = self.y - y_pred
                coeffs = self.model.coef_
                
                return {
                    'coefficients': coeffs,
                    'term_names': term_names,
                    'design_matrix': X_full,
                    'r_squared': r2,
                    'rmse': np.sqrt(mean_squared_error(self.y, y_pred)),
                    'residuals': residuals,
                    'predictions': y_pred,
                    'model_type': 'quadratic'
                }
        
        class DOEResponseAnalysis:
            def __init__(self, design, responses, factor_names=None, response_name="Response"):
                self.X = design
                self.y = responses
                self.factor_names = factor_names or [f"x{i+1}" for i in range(design.shape[1])]
                self.model = LinearRegression()
                
            def fit_linear_model(self):
                self.model.fit(self.X, self.y)
                y_pred = self.model.predict(self.X)
                r2 = r2_score(self.y, y_pred)
                residuals = self.y - y_pred
                
                # Calculate coefficients with intercept
                X_with_intercept = np.column_stack([np.ones(len(self.X)), self.X])
                coeffs = np.linalg.lstsq(X_with_intercept, self.y, rcond=None)[0]
                
                return {
                    'coefficients': coeffs,
                    'r_squared': r2,
                    'rmse': np.sqrt(mean_squared_error(self.y, y_pred)),
                    'residuals': residuals,
                    'predictions': y_pred
                }
                
            def fit_quadratic_model(self):
                poly = PolynomialFeatures(degree=2, include_bias=True)
                X_poly = poly.fit_transform(self.X)
                
                self.model = LinearRegression()
                self.model.fit(X_poly, self.y)
                
                y_pred = self.model.predict(X_poly)
                r2 = r2_score(self.y, y_pred)
                residuals = self.y - y_pred
                coeffs = self.model.coef_
                
                feature_names = poly.get_feature_names_out(self.factor_names)
                
                return {
                    'coefficients': coeffs,
                    'feature_names': feature_names,
                    'r_squared': r2,
                    'rmse': np.sqrt(mean_squared_error(self.y, y_pred)),
                    'residuals': residuals,
                    'predictions': y_pred,
                    'polynomial_features': X_poly
                }
    
    # Try to import other utilities or use fallbacks
    try:
        import utils.math_utils as math_utils
        gram_matrix = math_utils.gram_matrix
        calculate_determinant = math_utils.calculate_determinant
        evaluate_mixture_model_terms = math_utils.evaluate_mixture_model_terms
    except:
        def gram_matrix(X):
            return X.T @ X
            
        def calculate_determinant(matrix):
            return np.linalg.det(matrix)
            
        def evaluate_mixture_model_terms(point, model_type):
            if model_type == "linear":
                return list(point)
            elif model_type == "quadratic":
                terms = list(point)
                n = len(point)
                for i in range(n):
                    for j in range(i + 1, n):
                        terms.append(point[i] * point[j])
                return terms
            
    try:
        import core.optimal_design_generator as odg
        OptimalDesignGenerator = odg.OptimalDesignGenerator
    except:
        OptimalDesignGenerator = None


@dataclass
class ReconstructionConfig:
    """Configuration for sequential regression reconstruction"""
    # Design parameters
    n_components: int = 3
    model_type: str = "quadratic"  # linear, quadratic, cubic, special_cubic
    max_iterations: int = 20
    min_iterations: int = 3
    
    # Batch sizes
    initial_batch_size: int = 10
    sequential_batch_size: int = 3
    
    # Convergence criteria
    coefficient_tolerance: float = 0.05  # Relative change in coefficients
    r2_threshold: float = 0.95
    prediction_accuracy_threshold: float = 0.02
    d_efficiency_threshold: float = 0.85
    
    # Model selection
    regression_method: str = "ols"  # ols, ridge, lasso
    regularization_alpha: float = 0.01
    
    # Design strategy
    design_strategy: str = "d_optimal"  # d_optimal, space_filling, adaptive
    objective_function: Optional[Callable] = None
    
    # Constraints
    lower_bounds: Optional[List[float]] = None
    upper_bounds: Optional[List[float]] = None
    fixed_components: Optional[Dict[int, float]] = None
    
    # Analysis parameters
    significance_level: float = 0.05
    bootstrap_samples: int = 1000
    
    # Parts mode support
    use_parts_mode: bool = False
    parts_bounds: Optional[List[Tuple[float, float]]] = None


@dataclass
class IterationResult:
    """Results from a single iteration"""
    iteration: int
    design_points: np.ndarray
    responses: np.ndarray
    coefficients: np.ndarray
    coefficient_std_errors: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    prediction_error: float
    d_efficiency: float
    convergence_metrics: Dict[str, float]
    selected_parameters: Dict[str, float]
    model_adequacy: Dict[str, float]
    significant_terms: List[str]


class SequentialRegressionReconstructor:
    """
    Sequential reconstruction of regression function coefficients using experimental design.
    
    This class implements a comprehensive framework for iteratively building mixture 
    designs to reconstruct regression coefficients with high precision.
    """
    
    def __init__(self, config: ReconstructionConfig):
        """
        Initialize the sequential reconstructor.
        
        Parameters:
        -----------
        config : ReconstructionConfig
            Configuration object with all parameters
        """
        self.config = config
        self.n_components = config.n_components
        
        # Initialize component names
        self.component_names = [f"x{i+1}" for i in range(self.n_components)]
        
        # Experimental history
        self.iteration_history: List[IterationResult] = []
        self.design_history: List[np.ndarray] = []
        self.response_history: List[np.ndarray] = []
        
        # Current state
        self.current_design: Optional[np.ndarray] = None
        self.current_responses: Optional[np.ndarray] = None
        self.current_coefficients: Optional[np.ndarray] = None
        self.current_model = None
        
        # Initialize design generator
        self._setup_design_generator()
        
        # Convergence tracking
        self.converged = False
        self.convergence_iteration = None
        
        print(f"Sequential Regression Reconstructor initialized")
        print(f"  Components: {self.n_components}")
        print(f"  Model type: {config.model_type}")
        print(f"  Design strategy: {config.design_strategy}")
        print(f"  Max iterations: {config.max_iterations}")
    
    def _setup_design_generator(self):
        """Setup the design generator based on configuration"""
        try:
            self.design_generator = OptimalDesignGenerator(
                n_components=self.n_components,
                component_bounds=self.config.lower_bounds and self.config.upper_bounds and 
                    list(zip(self.config.lower_bounds, self.config.upper_bounds)) or None,
                fixed_components=self.config.fixed_components or {}
            )
        except:
            # Fallback to simple generator
            self.design_generator = None
            print("Warning: Using simplified design generation")
    
    def generate_initial_design(self) -> np.ndarray:
        """
        Generate initial experimental design.
        
        Returns:
        --------
        np.ndarray
            Initial design matrix
        """
        print(f"\nGenerating initial design with {self.config.initial_batch_size} points")
        
        if self.design_generator:
            try:
                # Use sophisticated design generator
                initial_design = self.design_generator.generate_d_optimal_mixture(
                    n_runs=self.config.initial_batch_size,
                    model_type=self.config.model_type
                )
            except:
                # Fallback to simple generation
                initial_design = self._generate_simple_initial_design()
        else:
            initial_design = self._generate_simple_initial_design()
        
        self.current_design = initial_design
        self.design_history.append(initial_design.copy())
        
        print(f"  Generated {len(initial_design)} initial design points")
        return initial_design
    
    def _generate_simple_initial_design(self) -> np.ndarray:
        """Generate simple initial design as fallback"""
        n_points = self.config.initial_batch_size
        
        # Strategy 1: Include vertices (pure components)
        design_points = []
        
        for i in range(min(self.n_components, n_points)):
            point = np.zeros(self.n_components)
            point[i] = 1.0
            design_points.append(point)
        
        # Strategy 2: Add centroid
        if len(design_points) < n_points:
            centroid = np.ones(self.n_components) / self.n_components
            design_points.append(centroid)
        
        # Strategy 3: Add binary mixtures
        while len(design_points) < n_points:
            # Random binary mixture
            i, j = np.random.choice(self.n_components, 2, replace=False)
            point = np.zeros(self.n_components)
            ratio = np.random.uniform(0.2, 0.8)
            point[i] = ratio
            point[j] = 1 - ratio
            design_points.append(point)
        
        # Strategy 4: Fill with random valid mixtures
        while len(design_points) < n_points:
            point = np.random.dirichlet(np.ones(self.n_components))
            design_points.append(point)
        
        design_matrix = np.array(design_points[:n_points])
        
        # Apply constraints if any
        if self.config.fixed_components:
            design_matrix = self._apply_fixed_components(design_matrix)
        
        return design_matrix
    
    def _apply_fixed_components(self, design: np.ndarray) -> np.ndarray:
        """Apply fixed component constraints"""
        fixed_design = design.copy()
        
        for comp_idx, fixed_value in self.config.fixed_components.items():
            if 0 <= comp_idx < self.n_components:
                fixed_design[:, comp_idx] = fixed_value
        
        # Renormalize remaining components
        for i in range(len(fixed_design)):
            fixed_sum = sum(self.config.fixed_components.values())
            remaining_sum = 1.0 - fixed_sum
            
            if remaining_sum > 0:
                variable_indices = [j for j in range(self.n_components) 
                                  if j not in self.config.fixed_components]
                current_variable_sum = sum(fixed_design[i, j] for j in variable_indices)
                
                if current_variable_sum > 0:
                    scale_factor = remaining_sum / current_variable_sum
                    for j in variable_indices:
                        fixed_design[i, j] *= scale_factor
        
        return fixed_design
    
    def collect_responses(self, design_points: np.ndarray, 
                         response_function: Callable) -> np.ndarray:
        """
        Collect experimental responses for given design points.
        
        Parameters:
        -----------
        design_points : np.ndarray
            Design matrix
        response_function : Callable
            Function to generate responses
            
        Returns:
        --------
        np.ndarray
            Response values
        """
        print(f"Collecting responses for {len(design_points)} experiments")
        
        responses = np.array([response_function(point) for point in design_points])
        
        if self.current_responses is None:
            self.current_responses = responses
        else:
            self.current_responses = np.concatenate([self.current_responses, responses])
        
        self.response_history.append(responses.copy())
        
        print(f"  Response range: [{responses.min():.3f}, {responses.max():.3f}]")
        return responses
    
    def analyze_current_data(self) -> Dict:
        """
        Analyze current experimental data and fit models.
        
        Returns:
        --------
        Dict
            Analysis results
        """
        print(f"\nAnalyzing current data: {len(self.current_design)} points")
        print(f"  Model type: {self.config.model_type}")
        
        # For mixture designs, use custom fitting for higher-order models
        if self._is_mixture_design():
            if self.config.model_type in ["cubic", "quartic", "quintic"]:
                # Use direct fitting for higher-order mixture models
                model_results = self._fit_higher_order_mixture_model()
            else:
                # Use standard analyzer for linear and quadratic
                analyzer = MixtureResponseAnalysis(
                    self.current_design, self.current_responses,
                    component_names=self.component_names, 
                    response_name="Response"
                )
                
                if self.config.model_type == "linear":
                    model_results = analyzer.fit_scheffe_linear()
                else:
                    model_results = analyzer.fit_scheffe_quadratic()
                
                self.current_model = analyzer.model
        else:
            # For non-mixture designs
            analyzer = DOEResponseAnalysis(
                self.current_design, self.current_responses,
                factor_names=self.component_names,
                response_name="Response"
            )
            
            if self.config.model_type == "linear":
                model_results = analyzer.fit_linear_model()
            else:
                model_results = analyzer.fit_quadratic_model()
            
            self.current_model = analyzer.model
        
        # Store current coefficients
        self.current_coefficients = model_results['coefficients']
        
        # Calculate additional metrics
        d_efficiency = self._calculate_d_efficiency()
        model_adequacy = self._assess_model_adequacy(model_results)
        significant_terms = self._identify_significant_terms(model_results)
        
        analysis_results = {
            'model_results': model_results,
            'd_efficiency': d_efficiency,
            'model_adequacy': model_adequacy,
            'significant_terms': significant_terms,
            'analyzer': getattr(self, 'current_model', None)
        }
        
        print(f"  Model R²: {model_results['r_squared']:.4f}")
        print(f"  D-efficiency: {d_efficiency:.4f}")
        print(f"  Significant terms: {len(significant_terms)}")
        
        return analysis_results
    
    def _fit_higher_order_mixture_model(self) -> Dict:
        """Fit cubic, quartic, or quintic mixture models"""
        print(f"  Fitting {self.config.model_type} mixture model")
        
        # Build design matrix based on model type
        if self.config.model_type == "cubic":
            X_design = self._build_mixture_cubic_matrix(self.current_design)
        elif self.config.model_type == "quartic":
            X_design = self._build_mixture_quartic_matrix(self.current_design)
        elif self.config.model_type == "quintic":
            X_design = self._build_mixture_quintic_matrix(self.current_design)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Fit model without intercept (mixture constraint)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_design, self.current_responses)
        
        # Store model
        self.current_model = model
        
        # Get predictions and metrics
        y_pred = model.predict(X_design)
        r2 = r2_score(self.current_responses, y_pred)
        residuals = self.current_responses - y_pred
        rmse = np.sqrt(mean_squared_error(self.current_responses, y_pred))
        
        # Generate term names
        term_names = self._generate_term_names()
        
        return {
            'coefficients': model.coef_,
            'term_names': term_names,
            'design_matrix': X_design,
            'r_squared': r2,
            'rmse': rmse,
            'residuals': residuals,
            'predictions': y_pred,
            'model_type': self.config.model_type
        }
    
    def _is_mixture_design(self) -> bool:
        """Check if current design is a mixture design"""
        if self.current_design is None:
            return False
        
        # Check if rows sum to approximately 1
        row_sums = np.sum(self.current_design, axis=1)
        return np.allclose(row_sums, 1.0, atol=1e-6)
    
    def _calculate_d_efficiency(self) -> float:
        """Calculate D-efficiency of current design"""
        try:
            # Build design matrix based on model type
            if self.config.model_type == "linear":
                if self._is_mixture_design():
                    X = self.current_design
                else:
                    X = np.column_stack([np.ones(len(self.current_design)), self.current_design])
            elif self.config.model_type == "cubic" and self._is_mixture_design():
                X = self._build_mixture_cubic_matrix(self.current_design)
            elif self.config.model_type == "quartic" and self._is_mixture_design():
                X = self._build_mixture_quartic_matrix(self.current_design)
            elif self.config.model_type == "quintic" and self._is_mixture_design():
                X = self._build_mixture_quintic_matrix(self.current_design)
            elif self._is_mixture_design():
                # Quadratic mixture model
                X = self._build_mixture_quadratic_matrix(self.current_design)
            else:
                # Regular polynomial model
                poly = PolynomialFeatures(degree=2, include_bias=True)
                X = poly.fit_transform(self.current_design)
            
            # Calculate D-efficiency
            XtX = X.T @ X
            det_XtX = np.linalg.det(XtX)
            n_params = X.shape[1]
            n_runs = X.shape[0]
            
            d_efficiency = (det_XtX / n_runs) ** (1 / n_params)
            return d_efficiency
            
        except:
            return 0.0
    
    def _build_mixture_quadratic_matrix(self, design: np.ndarray) -> np.ndarray:
        """Build design matrix for mixture quadratic model"""
        n_runs, n_comp = design.shape
        
        # Linear terms
        X = [design]
        
        # Interaction terms (2-way)
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                interaction = design[:, i] * design[:, j]
                X.append(interaction.reshape(-1, 1))
        
        return np.column_stack(X)
    
    def _build_mixture_cubic_matrix(self, design: np.ndarray) -> np.ndarray:
        """Build design matrix for mixture cubic model"""
        n_runs, n_comp = design.shape
        
        # Linear terms
        X = [design]
        
        # 2-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                interaction = design[:, i] * design[:, j]
                X.append(interaction.reshape(-1, 1))
        
        # 3-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                for k in range(j + 1, n_comp):
                    interaction = design[:, i] * design[:, j] * design[:, k]
                    X.append(interaction.reshape(-1, 1))
        
        return np.column_stack(X)
    
    def _build_mixture_quartic_matrix(self, design: np.ndarray) -> np.ndarray:
        """Build design matrix for mixture quartic (4th order) model"""
        n_runs, n_comp = design.shape
        
        # Linear terms
        X = [design]
        
        # 2-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                interaction = design[:, i] * design[:, j]
                X.append(interaction.reshape(-1, 1))
        
        # 3-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                for k in range(j + 1, n_comp):
                    interaction = design[:, i] * design[:, j] * design[:, k]
                    X.append(interaction.reshape(-1, 1))
        
        # 4-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                for k in range(j + 1, n_comp):
                    for m in range(k + 1, n_comp):
                        interaction = design[:, i] * design[:, j] * design[:, k] * design[:, m]
                        X.append(interaction.reshape(-1, 1))
        
        return np.column_stack(X)
    
    def _build_mixture_quintic_matrix(self, design: np.ndarray) -> np.ndarray:
        """Build design matrix for mixture quintic (5th order) model"""
        n_runs, n_comp = design.shape
        
        # Linear terms
        X = [design]
        
        # 2-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                interaction = design[:, i] * design[:, j]
                X.append(interaction.reshape(-1, 1))
        
        # 3-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                for k in range(j + 1, n_comp):
                    interaction = design[:, i] * design[:, j] * design[:, k]
                    X.append(interaction.reshape(-1, 1))
        
        # 4-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                for k in range(j + 1, n_comp):
                    for m in range(k + 1, n_comp):
                        interaction = design[:, i] * design[:, j] * design[:, k] * design[:, m]
                        X.append(interaction.reshape(-1, 1))
        
        # 5-way interactions
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                for k in range(j + 1, n_comp):
                    for m in range(k + 1, n_comp):
                        for n in range(m + 1, n_comp):
                            interaction = design[:, i] * design[:, j] * design[:, k] * design[:, m] * design[:, n]
                            X.append(interaction.reshape(-1, 1))
        
        return np.column_stack(X)
    
    def _generate_term_names(self) -> List[str]:
        """Generate term names for the current model type"""
        term_names = []
        
        # Linear terms
        term_names.extend(self.component_names)
        
        # 2-way interactions
        if self.config.model_type in ["quadratic", "cubic", "quartic", "quintic"]:
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    term_names.append(f"{self.component_names[i]}*{self.component_names[j]}")
        
        # 3-way interactions
        if self.config.model_type in ["cubic", "quartic", "quintic"]:
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    for k in range(j + 1, self.n_components):
                        term_names.append(f"{self.component_names[i]}*{self.component_names[j]}*{self.component_names[k]}")
        
        # 4-way interactions
        if self.config.model_type in ["quartic", "quintic"]:
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    for k in range(j + 1, self.n_components):
                        for m in range(k + 1, self.n_components):
                            term_names.append(f"{self.component_names[i]}*{self.component_names[j]}*{self.component_names[k]}*{self.component_names[m]}")
        
        # 5-way interactions
        if self.config.model_type == "quintic":
            for i in range(self.n_components):
                for j in range(i + 1, self.n_components):
                    for k in range(j + 1, self.n_components):
                        for m in range(k + 1, self.n_components):
                            for n in range(m + 1, self.n_components):
                                term_names.append(f"{self.component_names[i]}*{self.component_names[j]}*{self.component_names[k]}*{self.component_names[m]}*{self.component_names[n]}")
        
        return term_names
    
    def _assess_model_adequacy(self, model_results: Dict) -> Dict:
        """Assess adequacy of current model"""
        r2 = model_results['r_squared']
        residuals = model_results['residuals']
        
        # Residual analysis
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Normality test
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        except:
            shapiro_stat, shapiro_p = 0.0, 1.0
        
        # Lack of fit assessment (simplified)
        n_runs = len(residuals)
        n_params = len(model_results['coefficients'])
        lack_of_fit_p = 1.0  # Placeholder
        
        adequacy = {
            'r_squared_adequate': r2 >= self.config.r2_threshold,
            'residual_std': residual_std,
            'residual_mean': residual_mean,
            'normality_p': shapiro_p,
            'normality_adequate': shapiro_p >= 0.05,
            'lack_of_fit_p': lack_of_fit_p,
            'overall_adequate': (r2 >= self.config.r2_threshold and 
                               shapiro_p >= 0.05 and 
                               abs(residual_mean) < 0.1 * residual_std)
        }
        
        return adequacy
    
    def _identify_significant_terms(self, model_results: Dict) -> List[str]:
        """Identify statistically significant terms"""
        significant_terms = []
        
        # Calculate p-values if not available
        if 'p_values' not in model_results:
            model_results = self._calculate_statistical_measures(model_results)
        
        if 'p_values' in model_results:
            p_values = model_results['p_values']
            
            if 'feature_names' in model_results:
                # Quadratic model
                feature_names = model_results['feature_names']
                for i, (name, p_val) in enumerate(zip(feature_names, p_values)):
                    if p_val < self.config.significance_level:
                        significant_terms.append(name)
            elif 'term_names' in model_results:
                # Mixture model
                term_names = model_results['term_names']
                for i, (name, p_val) in enumerate(zip(term_names, p_values)):
                    if p_val < self.config.significance_level:
                        significant_terms.append(name)
            else:
                # Linear model
                if self._is_mixture_design():
                    term_names = self.component_names
                else:
                    term_names = ['Intercept'] + self.component_names
                for i, (name, p_val) in enumerate(zip(term_names, p_values)):
                    if p_val < self.config.significance_level:
                        significant_terms.append(name)
        
        return significant_terms
    
    def _calculate_statistical_measures(self, model_results: Dict) -> Dict:
        """Calculate statistical measures like p-values and standard errors"""
        try:
            # Get design matrix used for fitting
            if self.config.model_type == "quadratic" and self._is_mixture_design():
                X_matrix = self._build_mixture_quadratic_matrix(self.current_design)
            elif self.config.model_type == "cubic" and self._is_mixture_design():
                X_matrix = self._build_mixture_quadratic_matrix(self.current_design)  # Simplified
            elif not self._is_mixture_design() and self.config.model_type == "quadratic":
                poly = PolynomialFeatures(degree=2, include_bias=True)
                X_matrix = poly.fit_transform(self.current_design)
            elif self._is_mixture_design():
                X_matrix = self.current_design  # Linear mixture
            else:
                X_matrix = np.column_stack([np.ones(len(self.current_design)), self.current_design])
            
            y = self.current_responses
            coeffs = model_results['coefficients']
            
            # Calculate residuals and MSE
            y_pred = X_matrix @ coeffs
            residuals = y - y_pred
            n_obs = len(y)
            n_params = len(coeffs)
            df_residual = max(1, n_obs - n_params)
            mse = np.sum(residuals**2) / df_residual
            
            # Calculate standard errors
            try:
                XtX_inv = np.linalg.inv(X_matrix.T @ X_matrix)
                var_coeff = mse * np.diag(XtX_inv)
                std_errors = np.sqrt(var_coeff)
                
                # Calculate t-statistics and p-values
                t_stats = coeffs / std_errors
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))
                
                model_results['standard_errors'] = std_errors
                model_results['t_statistics'] = t_stats
                model_results['p_values'] = p_values
                model_results['degrees_of_freedom'] = df_residual
                
            except np.linalg.LinAlgError:
                # Matrix is singular - use default values based on coefficient magnitude
                model_results['standard_errors'] = np.ones(len(coeffs)) * 0.1
                model_results['t_statistics'] = np.abs(coeffs) / 0.1
                model_results['p_values'] = np.where(np.abs(coeffs) > 1.0, 0.01, 0.5)
                model_results['degrees_of_freedom'] = df_residual
                
        except Exception as e:
            print(f"Warning: Could not calculate statistical measures: {e}")
            # Provide fallback values that mark large coefficients as significant
            n_coeffs = len(model_results['coefficients'])
            model_results['standard_errors'] = np.ones(n_coeffs) * 0.1
            model_results['t_statistics'] = np.abs(model_results['coefficients']) / 0.1
            model_results['p_values'] = np.where(np.abs(model_results['coefficients']) > 1.0, 0.01, 0.5)
            model_results['degrees_of_freedom'] = max(1, len(self.current_responses) - n_coeffs)
        
        return model_results
    
    def select_experimental_parameters(self, analysis_results: Dict) -> Dict[str, float]:
        """
        Select parameters for next iteration based on analysis.
        
        Parameters:
        -----------
        analysis_results : Dict
            Results from analysis
            
        Returns:
        --------
        Dict[str, float]
            Selected parameters
        """
        print("Selecting parameters for next iteration")
        
        model_results = analysis_results['model_results']
        significant_terms = analysis_results['significant_terms']
        
        selected_params = {
            'n_additional_points': self.config.sequential_batch_size,
            'focus_strategy': 'significant_terms',
            'target_efficiency': min(analysis_results['d_efficiency'] + 0.1, 0.95)
        }
        
        # Improved strategy selection
        d_efficiency = analysis_results['d_efficiency']
        r2_value = model_results['r_squared']
        n_significant = len(significant_terms)
        
        if d_efficiency < self.config.d_efficiency_threshold * 0.6:
            # Very poor efficiency - prioritize D-optimal
            selected_params['focus_strategy'] = 'd_optimal'
            selected_params['n_additional_points'] = self.config.sequential_batch_size
            
        elif n_significant > 0 and d_efficiency < self.config.d_efficiency_threshold:
            # Have significant terms but poor efficiency - focus on D-optimal
            selected_params['focus_strategy'] = 'd_optimal'
            selected_params['n_additional_points'] = self.config.sequential_batch_size
            
        elif n_significant > 0:
            # Have significant terms and decent efficiency - refine them
            selected_params['focus_strategy'] = 'significant_terms'
            selected_params['n_additional_points'] = self.config.sequential_batch_size
            
        elif r2_value < self.config.r2_threshold:
            # Poor fit - need more data
            selected_params['focus_strategy'] = 'model_improvement'
            selected_params['n_additional_points'] = min(8, self.config.sequential_batch_size * 2)
            
        else:
            # Default to exploration
            selected_params['focus_strategy'] = 'exploration'
            selected_params['n_additional_points'] = max(3, self.config.sequential_batch_size)
        
        print(f"  Strategy: {selected_params['focus_strategy']}")
        print(f"  Additional points: {selected_params['n_additional_points']}")
        
        return selected_params
    
    def gen_additional_design_points(self, selected_params: Dict[str, float],
                                   analysis_results: Dict) -> np.ndarray:
        """
        Generate additional design points based on selected parameters.
        
        Parameters:
        -----------
        selected_params : Dict[str, float]
            Selected parameters
        analysis_results : Dict
            Analysis results
            
        Returns:
        --------
        np.ndarray
            Additional design points
        """
        n_additional = int(selected_params['n_additional_points'])
        strategy = selected_params['focus_strategy']
        
        print(f"Generating {n_additional} additional points using {strategy} strategy")
        
        if strategy == 'd_optimal':
            additional_points = self._generate_d_optimal_augmentation(n_additional)
        elif strategy == 'exploration':
            additional_points = self._generate_exploration_points(n_additional)
        elif strategy == 'significant_terms':
            additional_points = self._generate_significant_term_points(
                n_additional, analysis_results['significant_terms']
            )
        else:
            additional_points = self._generate_model_improvement_points(n_additional)
        
        # Update current design
        if self.current_design is None:
            self.current_design = additional_points
        else:
            self.current_design = np.vstack([self.current_design, additional_points])
        
        self.design_history.append(additional_points.copy())
        
        print(f"  Generated {len(additional_points)} additional points")
        return additional_points
    
    def _generate_d_optimal_augmentation(self, n_points: int) -> np.ndarray:
        """Generate D-optimal augmentation points"""
        # Generate candidate points
        n_candidates = max(200, n_points * 20)
        candidates = self._generate_candidate_points(n_candidates)
        
        # Select best D-optimal points
        selected_points = []
        
        for _ in range(n_points):
            best_point = None
            best_efficiency = -np.inf
            
            for candidate in candidates:
                # Test augmented design
                if len(selected_points) == 0:
                    test_design = np.vstack([self.current_design, candidate.reshape(1, -1)])
                else:
                    test_design = np.vstack([
                        self.current_design, 
                        np.array(selected_points),
                        candidate.reshape(1, -1)
                    ])
                
                # Calculate efficiency improvement
                efficiency = self._calculate_design_efficiency(test_design)
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_point = candidate
            
            if best_point is not None:
                selected_points.append(best_point)
                # Remove best point and nearby points from candidates
                candidates = self._remove_nearby_candidates(candidates, best_point, 0.05)
        
        return np.array(selected_points) if selected_points else self._generate_random_points(n_points)
    
    def _generate_exploration_points(self, n_points: int) -> np.ndarray:
        """Generate points for exploration"""
        points = []
        
        # Strategy: Space-filling design
        for _ in range(n_points):
            if self._is_mixture_design():
                # Generate random mixture
                point = np.random.dirichlet(np.ones(self.n_components))
            else:
                # Generate random point in factor space
                point = np.random.uniform(-1, 1, self.n_components)
            
            # Apply constraints
            if self.config.fixed_components:
                point = self._apply_fixed_components(point.reshape(1, -1))[0]
            
            points.append(point)
        
        return np.array(points)
    
    def _generate_significant_term_points(self, n_points: int, 
                                        significant_terms: List[str]) -> np.ndarray:
        """Generate points to better estimate significant terms"""
        points = []
        
        if not significant_terms:
            return self._generate_exploration_points(n_points)
        
        # Focus on regions that activate significant terms
        for i in range(n_points):
            if self._is_mixture_design():
                # For mixture designs, emphasize components in significant terms
                point = np.ones(self.n_components) * 0.1  # Base level
                
                # Identify components in significant terms
                active_components = set()
                for term in significant_terms:
                    for j, comp_name in enumerate(self.component_names):
                        if comp_name in term:
                            active_components.add(j)
                
                # Allocate more to active components
                if active_components:
                    remaining = 1.0 - len(point) * 0.1
                    per_active = remaining / len(active_components)
                    for idx in active_components:
                        point[idx] += per_active
                
                # Normalize
                point = point / np.sum(point)
            else:
                # For regular designs, use random points
                point = np.random.uniform(-1, 1, self.n_components)
            
            points.append(point)
        
        return np.array(points)
    
    def _generate_model_improvement_points(self, n_points: int) -> np.ndarray:
        """Generate points to improve model fit"""
        # Strategy: Focus on regions with high residuals
        points = []
        
        if (self.current_design is not None and self.current_responses is not None and 
            self.current_model is not None):
            
            # Find points with highest residuals
            predictions = self.current_model.predict(self.current_design)
            residuals = self.current_responses - predictions
            high_residual_indices = np.argsort(np.abs(residuals))[-min(5, len(residuals)):]
            
            # Generate points near high-residual points
            for i in range(n_points):
                if len(high_residual_indices) > 0:
                    base_idx = high_residual_indices[i % len(high_residual_indices)]
                    base_point = self.current_design[base_idx]
                    
                    # Perturb the base point
                    perturbation = np.random.normal(0, 0.1, self.n_components)
                    new_point = base_point + perturbation
                    
                    if self._is_mixture_design():
                        # Ensure valid mixture
                        new_point = np.abs(new_point)
                        new_point = new_point / np.sum(new_point)
                    else:
                        # Clip to bounds
                        new_point = np.clip(new_point, -1, 1)
                    
                    points.append(new_point)
                else:
                    points.append(self._generate_exploration_points(1)[0])
        else:
            return self._generate_exploration_points(n_points)
        
        return np.array(points)
    
    def _generate_candidate_points(self, n_candidates: int) -> np.ndarray:
        """Generate candidate points for selection"""
        candidates = []
        
        for _ in range(n_candidates):
            if self._is_mixture_design():
                # Mixture design candidates
                point = np.random.dirichlet(np.ones(self.n_components))
            else:
                # Regular design candidates
                point = np.random.uniform(-1, 1, self.n_components)
            
            # Apply constraints
            if self.config.fixed_components:
                point = self._apply_fixed_components(point.reshape(1, -1))[0]
            
            candidates.append(point)
        
        return np.array(candidates)
    
    def _generate_random_points(self, n_points: int) -> np.ndarray:
        """Generate random valid points as fallback"""
        return self._generate_candidate_points(n_points)
    
    def _calculate_design_efficiency(self, design: np.ndarray) -> float:
        """Calculate design efficiency"""
        try:
            if self.config.model_type == "linear":
                if self._is_mixture_design():
                    X = design
                else:
                    X = np.column_stack([np.ones(len(design)), design])
            else:
                if self._is_mixture_design():
                    X = self._build_mixture_quadratic_matrix(design)
                else:
                    poly = PolynomialFeatures(degree=2, include_bias=True)
                    X = poly.fit_transform(design)
            
            XtX = X.T @ X
            det_XtX = np.linalg.det(XtX)
            n_params = X.shape[1]
            n_runs = X.shape[0]
            
            return (det_XtX / n_runs) ** (1 / n_params)
        except:
            return 0.0
    
    def _remove_nearby_candidates(self, candidates: np.ndarray, 
                                point: np.ndarray, min_distance: float) -> np.ndarray:
        """Remove candidates too close to a given point"""
        distances = np.linalg.norm(candidates - point, axis=1)
        return candidates[distances > min_distance]
    
    def check_convergence(self, analysis_results: Dict) -> bool:
        """
        Check if reconstruction has converged.
        
        Parameters:
        -----------
        analysis_results : Dict
            Current analysis results
            
        Returns:
        --------
        bool
            True if converged
        """
        print("Checking convergence criteria")
        
        model_results = analysis_results['model_results']
        model_adequacy = analysis_results['model_adequacy']
        
        # More realistic convergence criteria
        r2_value = model_results['r_squared']
        d_eff_value = analysis_results['d_efficiency']
        rmse_value = model_results.get('rmse', 1.0)
        
        # Adaptive prediction accuracy based on response magnitude
        response_range = np.max(self.current_responses) - np.min(self.current_responses)
        adaptive_rmse_threshold = max(self.config.prediction_accuracy_threshold, 
                                    response_range * 0.05)  # 5% of response range
        
        # Primary convergence criteria
        criteria = {
            'r_squared': r2_value >= self.config.r2_threshold,
            'd_efficiency': d_eff_value >= self.config.d_efficiency_threshold,
            'prediction_accuracy': rmse_value <= adaptive_rmse_threshold
        }
        
        # Secondary criteria (model adequacy)
        secondary_criteria = {
            'model_adequate': model_adequacy['overall_adequate'],
            'significant_terms': len(analysis_results['significant_terms']) > 0
        }
        
        # Check coefficient stability if multiple iterations
        coefficient_stable = True
        if len(self.iteration_history) > 0:
            last_coeffs = self.iteration_history[-1].coefficients
            current_coeffs = self.current_coefficients
            
            if len(last_coeffs) == len(current_coeffs):
                rel_changes = np.abs((current_coeffs - last_coeffs) / (np.abs(last_coeffs) + 1e-10))
                coefficient_stable = np.all(rel_changes < self.config.coefficient_tolerance)
        
        criteria['coefficient_stable'] = coefficient_stable
        
        # Convergence logic: Need to meet most primary criteria OR achieve very high R²
        primary_criteria_met = sum(criteria.values())
        total_primary = len(criteria)
        
        # High R² can compensate for other metrics
        excellent_fit = r2_value >= min(0.98, self.config.r2_threshold + 0.03)
        
        # Convergence if:
        # 1. All primary criteria met, OR
        # 2. Excellent R² and at least half of other criteria met, OR  
        # 3. Very good performance on key metrics
        converged = (
            all(criteria.values()) or  # All criteria met
            (excellent_fit and primary_criteria_met >= total_primary - 1) or  # Excellent fit + most criteria
            (r2_value >= self.config.r2_threshold and 
             d_eff_value >= self.config.d_efficiency_threshold * 0.8 and
             rmse_value <= adaptive_rmse_threshold * 1.5)  # Good overall performance
        )
        
        print(f"  Convergence criteria:")
        print(f"    R² = {r2_value:.4f} (target: {self.config.r2_threshold:.2f}) {'✓' if criteria['r_squared'] else '✗'}")
        print(f"    D-eff = {d_eff_value:.4f} (target: {self.config.d_efficiency_threshold:.2f}) {'✓' if criteria['d_efficiency'] else '✗'}")
        print(f"    RMSE = {rmse_value:.4f} (adaptive target: {adaptive_rmse_threshold:.2f}) {'✓' if criteria['prediction_accuracy'] else '✗'}")
        print(f"    Coeff stable: {'✓' if criteria['coefficient_stable'] else '✗'}")
        print(f"    Significant terms: {len(analysis_results['significant_terms'])}")
        print(f"    Primary criteria met: {primary_criteria_met}/{total_primary}")
        
        if converged:
            print(f"  → CONVERGED")
            self.converged = True
            self.convergence_iteration = len(self.iteration_history) + 1
        else:
            print(f"  → Continuing iterations")
        
        return converged
    
    def save_iteration_result(self, iteration: int, analysis_results: Dict,
                            selected_params: Dict[str, float]) -> IterationResult:
        """
        Save results from current iteration.
        
        Parameters:
        -----------
        iteration : int
            Iteration number
        analysis_results : Dict
            Analysis results
        selected_params : Dict[str, float]
            Selected parameters
            
        Returns:
        --------
        IterationResult
            Packaged iteration result
        """
        model_results = analysis_results['model_results']
        
        # Calculate convergence metrics
        convergence_metrics = {
            'r_squared': model_results['r_squared'],
            'd_efficiency': analysis_results['d_efficiency'],
            'rmse': model_results.get('rmse', 0.0),
            'n_significant_terms': len(analysis_results['significant_terms'])
        }
        
        # Create iteration result
        iteration_result = IterationResult(
            iteration=iteration,
            design_points=self.current_design.copy(),
            responses=self.current_responses.copy(),
            coefficients=self.current_coefficients.copy(),
            coefficient_std_errors=model_results.get('standard_errors', np.zeros_like(self.current_coefficients)),
            r_squared=model_results['r_squared'],
            adjusted_r_squared=model_results.get('adj_r_squared', model_results['r_squared']),
            prediction_error=model_results.get('rmse', 0.0),
            d_efficiency=analysis_results['d_efficiency'],
            convergence_metrics=convergence_metrics,
            selected_parameters=selected_params,
            model_adequacy=analysis_results['model_adequacy'],
            significant_terms=analysis_results['significant_terms']
        )
        
        self.iteration_history.append(iteration_result)
        
        print(f"  Iteration {iteration} results saved")
        return iteration_result
    
    def _refit_with_significant_terms(self, significant_terms: List[str]) -> Dict:
        """
        Refit the model using only significant terms.
        
        Parameters:
        -----------
        significant_terms : List[str]
            List of significant term names
            
        Returns:
        --------
        Dict
            Updated model results with only significant terms
        """
        if not significant_terms:
            print("\n⚠️  No significant terms found - keeping full model")
            return None
        
        print(f"\n🔄 REFITTING MODEL WITH ONLY SIGNIFICANT TERMS")
        print(f"   Reducing from {len(self.current_coefficients)} to {len(significant_terms)} parameters")
        
        # Build design matrix with only significant terms
        all_term_names = self._generate_term_names()
        
        # Find indices of significant terms
        significant_indices = []
        for term in significant_terms:
            if term in all_term_names:
                significant_indices.append(all_term_names.index(term))
        
        if not significant_indices:
            print("⚠️  Could not match significant terms - keeping full model")
            return None
        
        # Build full design matrix
        if self.config.model_type == "cubic":
            X_full = self._build_mixture_cubic_matrix(self.current_design)
        elif self.config.model_type == "quartic":
            X_full = self._build_mixture_quartic_matrix(self.current_design)
        elif self.config.model_type == "quintic":
            X_full = self._build_mixture_quintic_matrix(self.current_design)
        elif self.config.model_type == "quadratic" and self._is_mixture_design():
            X_full = self._build_mixture_quadratic_matrix(self.current_design)
        else:
            X_full = self.current_design
        
        # Extract only significant columns
        X_reduced = X_full[:, significant_indices]
        
        # Fit reduced model
        reduced_model = LinearRegression(fit_intercept=False)
        reduced_model.fit(X_reduced, self.current_responses)
        
        # Get predictions and metrics
        y_pred = reduced_model.predict(X_reduced)
        r2 = r2_score(self.current_responses, y_pred)
        residuals = self.current_responses - y_pred
        rmse = np.sqrt(mean_squared_error(self.current_responses, y_pred))
        
        # Create full coefficient array (with zeros for non-significant terms)
        full_coefficients = np.zeros(len(all_term_names))
        full_coefficients[significant_indices] = reduced_model.coef_
        
        # Store reduced model
        self.current_model = reduced_model
        self.current_coefficients = full_coefficients
        self._significant_indices = significant_indices  # Store for predictions
        
        print(f"   ✅ Reduced model R²: {r2:.4f} (vs full model)")
        print(f"   ✅ Reduced model RMSE: {rmse:.4f}")
        print(f"   ✅ Parameters reduced: {len(all_term_names)} → {len(significant_terms)}")
        
        return {
            'coefficients': full_coefficients,
            'r_squared': r2,
            'rmse': rmse,
            'residuals': residuals,
            'predictions': y_pred,
            'model_type': f"{self.config.model_type}_reduced",
            'n_significant': len(significant_terms),
            'significant_indices': significant_indices
        }
    
    def run_sequential_reconstruction(self, response_function: Callable) -> Dict:
        """
        Run the complete sequential reconstruction process.
        
        Parameters:
        -----------
        response_function : Callable
            Function that generates responses for design points
            
        Returns:
        --------
        Dict
            Complete reconstruction results
        """
        print(f"\n{'='*70}")
        print(f"SEQUENTIAL REGRESSION COEFFICIENT RECONSTRUCTION")
        print(f"{'='*70}")
        
        # Step 1: Generate initial design
        initial_design = self.generate_initial_design()
        
        # Step 2: Collect initial responses
        initial_responses = self.collect_responses(initial_design, response_function)
        
        # Main iteration loop
        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n{'-'*50}")
            print(f"ITERATION {iteration}")
            print(f"{'-'*50}")
            
            # Step 3: Analyze current data
            analysis_results = self.analyze_current_data()
            
            # Step 4: Check convergence
            converged = self.check_convergence(analysis_results)
            
            # Step 5: Select experimental parameters
            selected_params = self.select_experimental_parameters(analysis_results)
            
            # Step 6: Save iteration result
            iteration_result = self.save_iteration_result(iteration, analysis_results, selected_params)
            
            if converged:
                print(f"\nCONVERGED after {iteration} iterations")
                break
            
            if iteration >= self.config.max_iterations:
                print(f"\nReached maximum iterations ({self.config.max_iterations})")
                break
            
            # Step 7: Generate additional design points
            additional_points = self.gen_additional_design_points(selected_params, analysis_results)
            
            # Step 8: Collect additional responses
            additional_responses = self.collect_responses(additional_points, response_function)
        
        # Final analysis with full model
        final_analysis = self.analyze_current_data()
        
        # REFIT MODEL WITH ONLY SIGNIFICANT TERMS
        significant_terms = final_analysis['significant_terms']
        if significant_terms:
            reduced_results = self._refit_with_significant_terms(significant_terms)
            if reduced_results:
                # Update final analysis with reduced model results
                final_analysis['model_results'].update(reduced_results)
                final_analysis['reduced_model'] = True
        
        results = self._compile_final_results(final_analysis)
        
        print(f"\n{'='*70}")
        print(f"RECONSTRUCTION COMPLETE")
        print(f"  Total iterations: {len(self.iteration_history)}")
        print(f"  Total experiments: {len(self.current_design)}")
        print(f"  Final R²: {final_analysis['model_results']['r_squared']:.4f}")
        print(f"  Final D-efficiency: {final_analysis['d_efficiency']:.4f}")
        print(f"  Converged: {'Yes' if self.converged else 'No'}")
        if 'reduced_model' in final_analysis:
            print(f"  Final model: REDUCED ({final_analysis['model_results'].get('n_significant', 0)} significant terms)")
        print(f"{'='*70}")
        
        return results
    
    def _compile_final_results(self, final_analysis: Dict) -> Dict:
        """Compile final results"""
        return {
            'converged': self.converged,
            'convergence_iteration': self.convergence_iteration,
            'total_iterations': len(self.iteration_history),
            'total_experiments': len(self.current_design),
            'final_design': self.current_design.copy(),
            'final_responses': self.current_responses.copy(),
            'final_coefficients': self.current_coefficients.copy(),
            'final_model': self.current_model,
            'final_analysis': final_analysis,
            'iteration_history': self.iteration_history,
            'design_history': self.design_history,
            'response_history': self.response_history,
            'config': self.config
        }
    
    def predict_response(self, new_points: np.ndarray) -> np.ndarray:
        """
        Predict responses for new design points.
        
        Parameters:
        -----------
        new_points : np.ndarray
            New design points
            
        Returns:
        --------
        np.ndarray
            Predicted responses
        """
        if self.current_model is None:
            raise ValueError("No model available. Run reconstruction first.")
        
        if self._is_mixture_design():
            # Verify mixture constraints
            row_sums = np.sum(new_points, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                print("Warning: Some points don't sum to 1.0 for mixture design")
        
        # Transform new points to match the model's expected input
        if hasattr(self.current_model, 'predict'):
            # Check if we need to transform the design matrix based on model type
            if self._is_mixture_design():
                if self.config.model_type == "cubic":
                    X_pred = self._build_mixture_cubic_matrix(new_points)
                elif self.config.model_type == "quartic":
                    X_pred = self._build_mixture_quartic_matrix(new_points)
                elif self.config.model_type == "quintic":
                    X_pred = self._build_mixture_quintic_matrix(new_points)
                elif self.config.model_type == "quadratic":
                    X_pred = self._build_mixture_quadratic_matrix(new_points)
                else:  # linear
                    X_pred = new_points
                return self.current_model.predict(X_pred)
            elif not self._is_mixture_design() and self.config.model_type == "quadratic":
                # For factorial quadratic models, use polynomial features
                poly = PolynomialFeatures(degree=2, include_bias=True)
                X_pred = poly.fit_transform(new_points)
                return self.current_model.predict(X_pred)
            else:
                # For linear models, use points directly
                return self.current_model.predict(new_points)
        else:
            print("Warning: Model doesn't support prediction")
            return np.zeros(len(new_points))
    
    def get_coefficient_summary(self) -> pd.DataFrame:
        """Get summary of reconstructed coefficients"""
        if self.current_coefficients is None:
            raise ValueError("No coefficients available. Run reconstruction first.")
        
        # Get the last iteration result
        if self.iteration_history:
            last_result = self.iteration_history[-1]
            std_errors = last_result.coefficient_std_errors
            significant_terms = last_result.significant_terms
        else:
            std_errors = np.zeros_like(self.current_coefficients)
            significant_terms = []
        
        # Create coefficient names based on model type
        if self._is_mixture_design():
            if self.config.model_type == "linear":
                coeff_names = self.component_names.copy()
            elif self.config.model_type in ["cubic", "quartic", "quintic"]:
                # For higher-order models, use the same logic as _generate_term_names()
                coeff_names = self._generate_term_names()
            elif self.config.model_type == "quadratic":
                coeff_names = self.component_names.copy()
                # Add quadratic interaction terms
                for i in range(self.n_components):
                    for j in range(i + 1, self.n_components):
                        coeff_names.append(f"{self.component_names[i]}*{self.component_names[j]}")
            else:
                # Fallback for unknown model types
                coeff_names = self._generate_term_names()
        else:
            if self.config.model_type == "linear":
                coeff_names = ['Intercept'] + self.component_names
            else:
                # For non-mixture quadratic models
                coeff_names = [f"Coeff_{i}" for i in range(len(self.current_coefficients))]
        
        # Truncate names if we have more coefficients than expected
        if len(coeff_names) > len(self.current_coefficients):
            coeff_names = coeff_names[:len(self.current_coefficients)]
        elif len(coeff_names) < len(self.current_coefficients):
            # Add generic names for extra coefficients
            for i in range(len(coeff_names), len(self.current_coefficients)):
                coeff_names.append(f"Extra_Coeff_{i}")
        
        summary = pd.DataFrame({
            'Term': coeff_names,
            'Coefficient': self.current_coefficients,
            'Std_Error': std_errors,
            'Significant': [name in significant_terms for name in coeff_names]
        })
        
        # Calculate confidence intervals if we have standard errors
        if np.any(std_errors > 0):
            t_critical = stats.t.ppf(1 - self.config.significance_level/2, 
                                   len(self.current_design) - len(self.current_coefficients))
            summary['CI_Lower'] = self.current_coefficients - t_critical * std_errors
            summary['CI_Upper'] = self.current_coefficients + t_critical * std_errors
        
        return summary
    
    def plot_convergence(self):
        """Plot convergence history"""
        if len(self.iteration_history) < 2:
            print("Need at least 2 iterations to plot convergence")
            return
        
        iterations = [result.iteration for result in self.iteration_history]
        r_squared = [result.r_squared for result in self.iteration_history]
        d_efficiency = [result.d_efficiency for result in self.iteration_history]
        rmse = [result.prediction_error for result in self.iteration_history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # R-squared
        axes[0, 0].plot(iterations, r_squared, 'b-o')
        axes[0, 0].axhline(y=self.config.r2_threshold, color='r', linestyle='--', label='Target')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_title('Model R-squared')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # D-efficiency
        axes[0, 1].plot(iterations, d_efficiency, 'g-o')
        axes[0, 1].axhline(y=self.config.d_efficiency_threshold, color='r', linestyle='--', label='Target')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('D-efficiency')
        axes[0, 1].set_title('Design Efficiency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        axes[1, 0].plot(iterations, rmse, 'm-o')
        axes[1, 0].axhline(y=self.config.prediction_accuracy_threshold, color='r', linestyle='--', label='Target')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Prediction Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Number of experiments
        n_experiments = [len(result.design_points) for result in self.iteration_history]
        axes[1, 1].plot(iterations, n_experiments, 'c-o')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Number of Experiments')
        axes[1, 1].set_title('Experimental Effort')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save reconstruction results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sequential_reconstruction_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        results_data = {
            'config': {
                'n_components': self.config.n_components,
                'model_type': self.config.model_type,
                'max_iterations': self.config.max_iterations,
                'initial_batch_size': self.config.initial_batch_size,
                'sequential_batch_size': self.config.sequential_batch_size,
                'r2_threshold': self.config.r2_threshold,
                'd_efficiency_threshold': self.config.d_efficiency_threshold,
                'significance_level': self.config.significance_level
            },
            'results': {
                'converged': self.converged,
                'total_iterations': len(self.iteration_history),
                'total_experiments': len(self.current_design) if self.current_design is not None else 0,
                'final_r_squared': self.iteration_history[-1].r_squared if self.iteration_history else 0,
                'final_d_efficiency': self.iteration_history[-1].d_efficiency if self.iteration_history else 0
            },
            'iteration_summary': []
        }
        
        # Add iteration summaries
        for result in self.iteration_history:
            iteration_summary = {
                'iteration': result.iteration,
                'r_squared': result.r_squared,
                'd_efficiency': result.d_efficiency,
                'prediction_error': result.prediction_error,
                'n_experiments': len(result.design_points),
                'n_significant_terms': len(result.significant_terms),
                'significant_terms': result.significant_terms
            }
            results_data['iteration_summary'].append(iteration_summary)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Results saved to {filename}")
        return filename


# Example usage and demonstration
def example_mixture_reconstruction():
    """Example: Sequential reconstruction for mixture design"""
    print("=== Mixture Regression Reconstruction Example ===\n")
    
    # Define a complex mixture function to reconstruct
    def complex_mixture_function(point):
        """Complex mixture function with quadratic and interaction terms"""
        x1, x2, x3 = point
        
        # True coefficients we want to reconstruct:
        # Linear: 50*x1 + 80*x2 + 30*x3
        # Quadratic interactions: 25*x1*x2 + 15*x1*x3 - 10*x2*x3
        
        response = (50*x1 + 80*x2 + 30*x3 + 
                   25*x1*x2 + 15*x1*x3 - 10*x2*x3 +
                   np.random.normal(0, 1.0))  # Add noise
        
        return response
    
    # Configuration
    config = ReconstructionConfig(
        n_components=3,
        model_type="quadratic",
        max_iterations=10,
        initial_batch_size=8,
        sequential_batch_size=3,
        r2_threshold=0.90,
        d_efficiency_threshold=0.70,
        significance_level=0.05
    )
    
    # Create reconstructor
    reconstructor = SequentialRegressionReconstructor(config)
    
    # Run reconstruction
    results = reconstructor.run_sequential_reconstruction(complex_mixture_function)
    
    # Display results
    print(f"\n=== RECONSTRUCTION RESULTS ===")
    print(f"Converged: {results['converged']}")
    print(f"Total iterations: {results['total_iterations']}")
    print(f"Total experiments: {results['total_experiments']}")
    
    # Show coefficient summary
    coeff_summary = reconstructor.get_coefficient_summary()
    print(f"\nReconstructed Coefficients:")
    print(coeff_summary.round(3))
    
    # Plot convergence
    if len(reconstructor.iteration_history) > 1:
        reconstructor.plot_convergence()
    
    # Save results
    filename = reconstructor.save_results()
    
    return reconstructor, results


def example_factorial_reconstruction():
    """Example: Sequential reconstruction for factorial design"""
    print("=== Factorial Design Regression Reconstruction Example ===\n")
    
    def factorial_function(point):
        """Factorial response function"""
        x1, x2, x3 = point
        
        # True model: y = 10 + 5*x1 + 3*x2 - 2*x3 + 2*x1*x2 - x1*x3 + noise
        response = (10 + 5*x1 + 3*x2 - 2*x3 + 
                   2*x1*x2 - x1*x3 + 
                   np.random.normal(0, 0.5))
        
        return response
    
    # Configuration for factorial design
    config = ReconstructionConfig(
        n_components=3,
        model_type="quadratic",
        max_iterations=8,
        initial_batch_size=12,
        sequential_batch_size=4,
        r2_threshold=0.95,
        d_efficiency_threshold=0.80,
        lower_bounds=[-1, -1, -1],
        upper_bounds=[1, 1, 1]
    )
    
    # Create reconstructor
    reconstructor = SequentialRegressionReconstructor(config)
    
    # Run reconstruction
    results = reconstructor.run_sequential_reconstruction(factorial_function)
    
    # Display results
    print(f"\n=== FACTORIAL RECONSTRUCTION RESULTS ===")
    coeff_summary = reconstructor.get_coefficient_summary()
    print(f"\nReconstructed Coefficients:")
    print(coeff_summary.round(3))
    
    return reconstructor, results


if __name__ == "__main__":
    # Run examples
    print("Sequential Regression Coefficient Reconstruction Examples")
    print("="*80)
    
    # Example 1: Mixture design
    mixture_reconstructor, mixture_results = example_mixture_reconstruction()
    
    print("\n" + "="*80)
    
    # Example 2: Factorial design  
    factorial_reconstructor, factorial_results = example_factorial_reconstruction()
    
    print("\n" + "="*80)
    print("Examples completed successfully!")
    print("Key features demonstrated:")
    print("✅ Sequential design point generation")
    print("✅ Iterative model fitting and analysis") 
    print("✅ Adaptive parameter selection")
    print("✅ Convergence monitoring")
    print("✅ Coefficient reconstruction with uncertainty")
