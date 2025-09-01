"""
Sequential Adaptive Mixture Design Generator
===========================================

This module implements a sequential adaptive approach to mixture design that:
1. Starts with efficient designs for main effects and 2-factor interactions
2. Analyzes experimental results to identify statistically significant interactions
3. Adaptively augments the design with points targeting significant higher-order terms
4. Iterates until sufficient model clarity and precision are achieved

This approach avoids the confounding and inefficiency of trying to optimize 
for all possible interactions simultaneously.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union, Callable
from itertools import combinations
from scipy import stats
import sys
import os

# Add parent directory to path for imports if needed
if 'src' not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.math_utils import (
    gram_matrix, calculate_determinant, evaluate_mixture_model_terms,
    euclidean_distance
)
from algorithms.candidate_generation import CandidateGenerator


class SequentialMixtureAnalyzer:
    """
    Analyzes experimental results to identify statistically significant interactions
    and guide adaptive design augmentation
    """
    
    def __init__(self, n_components: int, significance_level: float = 0.05):
        """
        Initialize sequential analyzer
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        significance_level : float
            Statistical significance threshold for interaction detection
        """
        self.n_components = n_components
        self.significance_level = significance_level
        
        # Define interaction hierarchy
        self.linear_terms = list(range(n_components))
        self.quadratic_terms = list(combinations(range(n_components), 2))
        self.tertiary_terms = list(combinations(range(n_components), 3))
        self.quartic_terms = list(combinations(range(n_components), 4))
        
        if n_components >= 5:
            self.quintic_terms = list(combinations(range(n_components), 5))
        else:
            self.quintic_terms = []
        
        print(f"Sequential Analyzer initialized for {n_components} components")
        print(f"  Significance level: {significance_level}")
    
    def analyze_experimental_results(self, design_points: List[List[float]], 
                                   responses: List[float]) -> Dict:
        """
        Analyze experimental results to identify significant interactions
        
        Parameters:
        -----------
        design_points : List[List[float]]
            Experimental design points
        responses : List[float]
            Observed responses
            
        Returns:
        --------
        Dict
            Analysis results with significant interactions identified
        """
        print(f"\nAnalyzing {len(design_points)} experimental results...")
        
        # Build design matrix for current model
        X = self._build_design_matrix(design_points, include_higher_order=False)
        
        # Fit linear + quadratic model
        base_model = self._fit_model(X, responses)
        
        # Test for higher-order interactions
        significant_interactions = self._test_higher_order_interactions(
            design_points, responses, base_model
        )
        
        # Calculate model quality metrics
        quality_metrics = self._calculate_model_quality(X, responses, base_model)
        
        analysis_results = {
            'base_model': base_model,
            'significant_interactions': significant_interactions,
            'quality_metrics': quality_metrics,
            'n_experiments': len(design_points),
            'recommended_augmentation': self._recommend_augmentation(significant_interactions)
        }
        
        print(f"  Found {len(significant_interactions)} significant higher-order interactions")
        print(f"  Model R²: {quality_metrics.get('r_squared', 0):.3f}")
        
        return analysis_results
    
    def _build_design_matrix(self, design_points: List[List[float]], 
                           include_higher_order: bool = False,
                           specific_interactions: List[Tuple] = None) -> np.ndarray:
        """Build design matrix for specified model terms"""
        X = []
        
        for point in design_points:
            # Start with standard mixture terms (linear + quadratic)
            terms = evaluate_mixture_model_terms(point, "quadratic")
            
            # Add specific higher-order terms if requested
            if include_higher_order and specific_interactions:
                for interaction in specific_interactions:
                    interaction_value = 1.0
                    for idx in interaction:
                        if idx < len(point):
                            interaction_value *= point[idx]
                    terms.append(interaction_value)
            
            X.append(terms)
        
        return np.array(X)
    
    def _fit_model(self, X: np.ndarray, y: List[float]) -> Dict:
        """Fit linear model and return coefficients with statistics"""
        y = np.array(y)
        
        try:
            # Use normal equations for fitting: β = (X'X)⁻¹X'y
            XtX = X.T @ X
            Xty = X.T @ y
            
            # Add regularization if needed for numerical stability
            if np.linalg.cond(XtX) > 1e12:
                XtX += np.eye(XtX.shape[0]) * 1e-6
            
            coefficients = np.linalg.solve(XtX, Xty)
            
            # Calculate fitted values and residuals
            y_pred = X @ coefficients
            residuals = y - y_pred
            
            # Calculate standard errors
            mse = np.sum(residuals**2) / (len(y) - X.shape[1])
            var_coeff = mse * np.linalg.inv(XtX)
            std_errors = np.sqrt(np.diag(var_coeff))
            
            # Calculate t-statistics and p-values
            t_stats = coefficients / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(y) - X.shape[1]))
            
            return {
                'coefficients': coefficients,
                'std_errors': std_errors,
                't_statistics': t_stats,
                'p_values': p_values,
                'fitted_values': y_pred,
                'residuals': residuals,
                'mse': mse
            }
            
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in model fitting")
            return {
                'coefficients': np.zeros(X.shape[1]),
                'std_errors': np.ones(X.shape[1]),
                't_statistics': np.zeros(X.shape[1]),
                'p_values': np.ones(X.shape[1]),
                'fitted_values': np.zeros(len(y)),
                'residuals': y,
                'mse': np.var(y)
            }
    
    def _test_higher_order_interactions(self, design_points: List[List[float]], 
                                      responses: List[float], base_model: Dict) -> List[Tuple]:
        """Test higher-order interactions for statistical significance"""
        significant_interactions = []
        
        # Get base model predictions
        base_residuals = base_model['residuals']
        
        # Test tertiary interactions if we have enough data points
        if len(design_points) > len(self.linear_terms) + len(self.quadratic_terms) + 5:
            significant_interactions.extend(
                self._test_interaction_level(design_points, base_residuals, self.tertiary_terms, "tertiary")
            )
        
        # Test quartic interactions if we have sufficient data and significant tertiary
        if len(design_points) > 25 and len(significant_interactions) > 0:
            significant_interactions.extend(
                self._test_interaction_level(design_points, base_residuals, self.quartic_terms, "quartic")
            )
        
        return significant_interactions
    
    def _test_interaction_level(self, design_points: List[List[float]], 
                              residuals: np.ndarray, interaction_terms: List[Tuple],
                              level_name: str) -> List[Tuple]:
        """Test interactions at a specific level for significance"""
        significant = []
        
        print(f"  Testing {level_name} interactions...")
        
        for interaction in interaction_terms:
            # Calculate interaction values
            interaction_values = []
            for point in design_points:
                value = 1.0
                for idx in interaction:
                    if idx < len(point):
                        value *= point[idx]
                interaction_values.append(value)
            
            interaction_values = np.array(interaction_values)
            
            # Skip if interaction has no variation
            if np.std(interaction_values) < 1e-10:
                continue
            
            # Test correlation with residuals
            try:
                correlation, p_value = stats.pearsonr(interaction_values, residuals)
                
                if p_value < self.significance_level and abs(correlation) > 0.3:
                    significant.append(interaction)
                    print(f"    Significant {level_name}: {interaction} (p={p_value:.4f}, r={correlation:.3f})")
                    
            except:
                continue
        
        return significant
    
    def _calculate_model_quality(self, X: np.ndarray, y: List[float], model: Dict) -> Dict:
        """Calculate model quality metrics"""
        y = np.array(y)
        y_pred = model['fitted_values']
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Adjusted R-squared
        n = len(y)
        p = X.shape[1]
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else 0
        
        # AIC and BIC
        mse = model['mse']
        aic = n * np.log(2 * np.pi * mse) + n + 2 * p
        bic = n * np.log(2 * np.pi * mse) + n + p * np.log(n)
        
        return {
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'rmse': np.sqrt(mse),
            'aic': aic,
            'bic': bic,
            'condition_number': np.linalg.cond(X.T @ X)
        }
    
    def _recommend_augmentation(self, significant_interactions: List[Tuple]) -> Dict:
        """Recommend design augmentation strategy based on significant interactions"""
        if not significant_interactions:
            return {
                'strategy': 'complete',
                'message': 'No significant higher-order interactions found. Current model appears adequate.',
                'additional_points': 0
            }
        
        # Categorize interactions by order
        tertiary = [i for i in significant_interactions if len(i) == 3]
        quartic = [i for i in significant_interactions if len(i) == 4]
        quintic = [i for i in significant_interactions if len(i) == 5]
        
        # Recommend points based on interaction complexity
        additional_points = len(tertiary) * 3 + len(quartic) * 5 + len(quintic) * 7
        
        strategy = 'augment_significant'
        message = f"Found {len(significant_interactions)} significant interactions. Recommend {additional_points} additional points."
        
        return {
            'strategy': strategy,
            'message': message,
            'additional_points': additional_points,
            'significant_tertiary': len(tertiary),
            'significant_quartic': len(quartic),
            'significant_quintic': len(quintic),
            'target_interactions': significant_interactions
        }


class SequentialMixtureDesignGenerator:
    """
    Sequential adaptive mixture design generator that builds designs iteratively
    based on statistical analysis of experimental results
    """
    
    def __init__(self, n_components: int, response_function: Optional[Callable] = None):
        """
        Initialize sequential design generator
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        response_function : Callable, optional
            Function to generate responses for simulation (for testing)
        """
        self.n_components = n_components
        self.response_function = response_function
        self.component_names = [f"x{i+1}" for i in range(n_components)]
        
        # Initialize analyzer
        self.analyzer = SequentialMixtureAnalyzer(n_components)
        
        # Initialize candidate generator for augmentation
        self.candidate_generator = CandidateGenerator(n_components, self.component_names)
        
        # Design history
        self.design_history = []
        self.analysis_history = []
        
        print(f"Sequential Mixture Design Generator initialized for {n_components} components")
    
    def generate_initial_design(self, n_initial_points: int = None) -> List[List[float]]:
        """
        Generate initial design optimized for linear + quadratic terms
        
        Parameters:
        -----------
        n_initial_points : int, optional
            Number of initial points. If None, uses efficient default based on model size
            
        Returns:
        --------
        List[List[float]]
            Initial design points
        """
        # Calculate efficient initial size
        n_linear_quad_params = self.n_components + len(list(combinations(range(self.n_components), 2)))
        
        if n_initial_points is None:
            # Use 1.5x the number of parameters as a good starting point
            n_initial_points = max(int(1.5 * n_linear_quad_params), 15)
        
        print(f"\nGenerating initial design with {n_initial_points} points")
        print(f"  Targeting {n_linear_quad_params} linear + quadratic parameters")
        
        # Generate candidates for D-optimal selection
        n_candidates = max(500, n_initial_points * 20)
        candidates = self._generate_initial_candidates(n_candidates)
        
        # Select D-optimal design for linear + quadratic model
        initial_design = self._select_d_optimal_initial(candidates, n_initial_points)
        
        # Store in history
        self.design_history.append({
            'phase': 'initial',
            'points': initial_design,
            'n_points': len(initial_design),
            'target_model': 'linear_quadratic'
        })
        
        print(f"  Generated {len(initial_design)} initial points")
        return initial_design
    
    def _generate_initial_candidates(self, n_candidates: int) -> List[List[float]]:
        """Generate candidate points for initial design selection"""
        candidates = []
        
        # Strategy 1: Vertices (pure components)
        for i in range(self.n_components):
            point = [0.0] * self.n_components
            point[i] = 1.0
            candidates.append(point)
        
        # Strategy 2: Binary mixtures with various ratios
        ratios = [0.5, 0.3, 0.7, 0.25, 0.75, 0.4, 0.6]
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                for ratio in ratios:
                    point = [0.0] * self.n_components
                    point[i] = ratio
                    point[j] = 1.0 - ratio
                    candidates.append(point)
        
        # Strategy 3: Ternary mixtures
        ternary_patterns = [
            [1/3, 1/3, 1/3],
            [0.5, 0.25, 0.25],
            [0.6, 0.2, 0.2],
            [0.45, 0.35, 0.2],
            [0.4, 0.4, 0.2]
        ]
        
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                for k in range(j + 1, self.n_components):
                    for pattern in ternary_patterns:
                        point = [0.0] * self.n_components
                        point[i] = pattern[0]
                        point[j] = pattern[1]
                        point[k] = pattern[2]
                        candidates.append(point)
        
        # Strategy 4: Centroid and near-centroid points
        centroid = [1.0 / self.n_components] * self.n_components
        candidates.append(centroid)
        
        # Add perturbed centroids
        for _ in range(10):
            point = [1.0 / self.n_components] * self.n_components
            # Add small random perturbations
            perturbations = np.random.normal(0, 0.05, self.n_components)
            point = np.array(point) + perturbations
            point = np.maximum(point, 0.01)  # Ensure positive
            point = point / np.sum(point)  # Renormalize
            candidates.append(point.tolist())
        
        # Strategy 5: Random space-filling points
        while len(candidates) < n_candidates:
            # Generate random simplex point using Dirichlet distribution
            point = np.random.dirichlet([1] * self.n_components)
            candidates.append(point.tolist())
        
        return candidates[:n_candidates]
    
    def _select_d_optimal_initial(self, candidates: List[List[float]], 
                                n_points: int) -> List[List[float]]:
        """Select D-optimal initial design for linear + quadratic model"""
        print(f"  Selecting D-optimal design from {len(candidates)} candidates...")
        
        # Convert candidates to numpy array
        candidates_array = [np.array(c) for c in candidates]
        
        # Initialize design with centroid or best single point
        best_start_det = 0
        best_start_idx = 0
        
        for i, candidate in enumerate(candidates):
            X = self._build_linear_quad_matrix([candidate])
            try:
                det = calculate_determinant(gram_matrix(X))
                if det > best_start_det:
                    best_start_det = det
                    best_start_idx = i
            except:
                continue
        
        design_indices = [best_start_idx]
        
        # Greedy D-optimal selection
        for _ in range(n_points - 1):
            best_candidate_idx = None
            best_det = 0
            
            for i, candidate in enumerate(candidates):
                if i in design_indices:
                    continue
                
                # Test adding this candidate
                test_design = [candidates[j] for j in design_indices] + [candidate]
                X = self._build_linear_quad_matrix(test_design)
                
                try:
                    det = calculate_determinant(gram_matrix(X))
                    if det > best_det:
                        best_det = det
                        best_candidate_idx = i
                except:
                    continue
            
            if best_candidate_idx is not None:
                design_indices.append(best_candidate_idx)
            else:
                break
        
        # Return selected design
        design = [candidates[i] for i in design_indices]
        print(f"  Selected {len(design)} points with determinant {best_det:.6e}")
        return design
    
    def _build_linear_quad_matrix(self, design_points: List[List[float]]) -> List[List[float]]:
        """Build design matrix for linear + quadratic mixture model"""
        X = []
        for point in design_points:
            terms = evaluate_mixture_model_terms(point, "quadratic")
            X.append(terms)
        return X
    
    def analyze_and_augment(self, current_design: List[List[float]], 
                          responses: List[float], max_additional_points: int = 20) -> Dict:
        """
        Analyze current results and augment design if needed
        
        Parameters:
        -----------
        current_design : List[List[float]]
            Current experimental design
        responses : List[float]
            Observed responses
        max_additional_points : int
            Maximum number of additional points to add
            
        Returns:
        --------
        Dict
            Analysis results and augmented design
        """
        print(f"\nAnalyzing current design with {len(current_design)} points...")
        
        # Analyze experimental results
        analysis_results = self.analyzer.analyze_experimental_results(current_design, responses)
        
        # Store analysis in history
        self.analysis_history.append(analysis_results)
        
        # Check if augmentation is needed
        recommendation = analysis_results['recommended_augmentation']
        
        if recommendation['strategy'] == 'complete':
            print("  No augmentation needed - model appears adequate")
            return {
                'current_design': current_design,
                'augmented_design': current_design,
                'analysis': analysis_results,
                'augmentation_added': [],
                'status': 'complete'
            }
        
        # Generate augmentation points
        n_additional = min(recommendation['additional_points'], max_additional_points)
        significant_interactions = recommendation['target_interactions']
        
        augmentation_points = self._generate_augmentation_points(
            current_design, significant_interactions, n_additional
        )
        
        # Combine designs
        augmented_design = current_design + augmentation_points
        
        # Store augmented design in history
        self.design_history.append({
            'phase': 'augmentation',
            'points': augmentation_points,
            'n_points': len(augmentation_points),
            'target_interactions': significant_interactions,
            'previous_design_size': len(current_design)
        })
        
        print(f"  Added {len(augmentation_points)} augmentation points")
        print(f"  Total design size: {len(augmented_design)} points")
        
        return {
            'current_design': current_design,
            'augmented_design': augmented_design,
            'analysis': analysis_results,
            'augmentation_added': augmentation_points,
            'status': 'augmented'
        }
    
    def _generate_augmentation_points(self, existing_design: List[List[float]], 
                                    target_interactions: List[Tuple], 
                                    n_points: int) -> List[List[float]]:
        """Generate points specifically targeting significant interactions"""
        print(f"  Generating {n_points} augmentation points for {len(target_interactions)} interactions")
        
        augmentation_points = []
        
        # Strategy 1: Points that maximize leverage for specific interactions
        for interaction in target_interactions:
            if len(augmentation_points) >= n_points:
                break
                
            # Generate several candidates for this interaction
            interaction_candidates = self._generate_interaction_candidates(interaction, 5)
            
            # Select best candidate that's not too close to existing design
            for candidate in interaction_candidates:
                if self._is_sufficiently_distant(candidate, existing_design + augmentation_points):
                    augmentation_points.append(candidate)
                    break
        
        # Strategy 2: Fill remaining slots with general space-filling points
        while len(augmentation_points) < n_points:
            # Generate random candidate
            candidate = np.random.dirichlet([1] * self.n_components).tolist()
            
            if self._is_sufficiently_distant(candidate, existing_design + augmentation_points):
                augmentation_points.append(candidate)
        
        return augmentation_points
    
    def _generate_interaction_candidates(self, interaction: Tuple, n_candidates: int) -> List[List[float]]:
        """Generate candidate points that maximize leverage for a specific interaction"""
        candidates = []
        interaction_order = len(interaction)
        
        for _ in range(n_candidates):
            point = [0.0] * self.n_components
            
            if interaction_order == 3:
                # Ternary interaction - allocate roughly equally among the three components
                base_allocation = 0.7 / interaction_order
                for idx in interaction:
                    point[idx] = base_allocation + np.random.uniform(-0.05, 0.05)
                
            elif interaction_order == 4:
                # Quartic interaction - allocate among four components
                base_allocation = 0.8 / interaction_order
                for idx in interaction:
                    point[idx] = base_allocation + np.random.uniform(-0.03, 0.03)
                    
            elif interaction_order == 5:
                # Quintic interaction - allocate among all components
                base_allocation = 0.9 / interaction_order
                for idx in interaction:
                    point[idx] = base_allocation + np.random.uniform(-0.02, 0.02)
            
            # Ensure non-negative values
            point = [max(0.01, x) for x in point]
            
            # Distribute remaining among other components
            allocated = sum(point[i] for i in interaction)
            remaining = 1.0 - allocated
            other_components = [i for i in range(self.n_components) if i not in interaction]
            
            if other_components and remaining > 0:
                per_other = remaining / len(other_components)
                for i in other_components:
                    point[i] = per_other
            
            # Final normalization
            total = sum(point)
            if total > 0:
                point = [x / total for x in point]
                candidates.append(point)
        
        return candidates
    
    def _is_sufficiently_distant(self, candidate: List[float], existing_design: List[List[float]], 
                               min_distance: float = 0.1) -> bool:
        """Check if candidate is sufficiently distant from existing points"""
        for existing_point in existing_design:
            distance = euclidean_distance(candidate, existing_point)
            if distance < min_distance:
                return False
        return True
    
    def run_sequential_experiment(self, true_function: Callable, 
                                max_total_points: int = 50, 
                                initial_fraction: float = 0.6) -> Dict:
        """
        Run complete sequential experiment with a known function (for testing)
        
        Parameters:
        -----------
        true_function : Callable
            Function that generates responses given design points
        max_total_points : int
            Maximum total experimental points
        initial_fraction : float
            Fraction of points to use in initial design
            
        Returns:
        --------
        Dict
            Complete experimental results
        """
        print(f"\n{'='*60}")
        print(f"SEQUENTIAL ADAPTIVE MIXTURE EXPERIMENT")
        print(f"{'='*60}")
        
        # Phase 1: Initial design
        n_initial = int(max_total_points * initial_fraction)
        initial_design = self.generate_initial_design(n_initial)
        
        # Get initial responses
        initial_responses = [true_function(point) for point in initial_design]
        
        print(f"\nPhase 1 Complete: {len(initial_design)} initial points")
        
        current_design = initial_design
        current_responses = initial_responses
        iteration = 1
        
        # Phase 2: Iterative augmentation
        while len(current_design) < max_total_points:
            print(f"\n--- Iteration {iteration} ---")
            
            max_additional = max_total_points - len(current_design)
            
            # Analyze and augment
            augmentation_result = self.analyze_and_augment(
                current_design, current_responses, max_additional
            )
            
            if augmentation_result['status'] == 'complete':
                print("Experiment complete - no further augmentation needed")
                break
            
            # Get responses for new points
            new_points = augmentation_result['augmentation_added']
            new_responses = [true_function(point) for point in new_points]
            
            # Update current design
            current_design = augmentation_result['augmented_design']
            current_responses = current_responses + new_responses
            
            iteration += 1
            
            # Safety check
            if iteration > 10:
                print("Maximum iterations reached")
                break
        
        # Final analysis
        final_analysis = self.analyzer.analyze_experimental_results(current_design, current_responses)
        
        results = {
            'final_design': current_design,
            'final_responses': current_responses,
            'n_total_points': len(current_design),
            'n_iterations': iteration,
            'design_history': self.design_history,
            'analysis_history': self.analysis_history,
            'final_analysis': final_analysis
        }
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE")
        print(f"  Total points: {len(current_design)}")
        print(f"  Iterations: {iteration}")
        print(f"  Final R²: {final_analysis['quality_metrics']['r_squared']:.3f}")
        print(f"{'='*60}")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    
    def example_mixture_function(point):
        """Example mixture function with complex interactions"""
        x1, x2, x3, x4, x5 = point
        
        # Linear terms
        linear = 10*x1 + 15*x2 + 8*x3 + 12*x4 + 6*x5
        
        # Quadratic terms
        quadratic = 20*x1*x2 + 15*x1*x3 + 10*x2*x3 + 8*x1*x4 + 12*x2*x4
        
        # Higher-order terms (only some are significant)
        higher_order = 35*x3*x4*x5 + 40*x1*x2*x3*x4
        
        # Add some noise
        noise = np.random.normal(0, 1.0)
        
        return linear + quadratic + higher_order + noise
    
    # Run example sequential experiment
    generator = SequentialMixtureDesignGenerator(n_components=5)
    results = generator.run_sequential_experiment(
        true_function=example_mixture_function,
        max_total_points=40,
        initial_fraction=0.6
    )
    
    print("\n" + "="*60)
    print("SEQUENTIAL EXPERIMENT RESULTS")
    print("="*60)
    print(f"Total design points: {results['n_total_points']}")
    print(f"Number of iterations: {results['n_iterations']}")
    print(f"Final R²: {results['final_analysis']['quality_metrics']['r_squared']:.3f}")
    
    # Print significant interactions found
    significant = results['final_analysis']['significant_interactions']
    if significant:
        print(f"\nSignificant interactions found:")
        for interaction in significant:
            print(f"  {interaction}")
    else:
        print("\nNo significant higher-order interactions detected")
