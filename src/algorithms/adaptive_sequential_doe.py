"""
Adaptive Sequential DOE Algorithm

This module implements a sophisticated sequential DOE approach that:
1. Screens the design space to identify main effects
2. Refines main effects accuracy with targeted points
3. Detects interaction effects through strategic point placement
4. Uses statistical analysis at each stage to guide decisions
5. Iteratively improves model accuracy

Author: DOE Development Team
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Import our existing modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import evaluate_mixture_model_terms
from utils.d_efficiency_calculator import calculate_d_efficiency


class AdaptiveSequentialDOE:
    """
    Adaptive Sequential DOE Algorithm that builds optimal experimental designs
    through intelligent multi-stage screening and refinement.
    """
    
    def __init__(self, n_components: int, component_ranges: List[Tuple[float, float]], 
                 response_function: Callable, max_total_runs: int = 50,
                 alpha_significance: float = 0.05, improvement_threshold: float = 0.05):
        """
        Initialize the Adaptive Sequential DOE
        
        Args:
            n_components: Number of mixture components
            component_ranges: List of (min, max) ranges for each component
            response_function: Function to evaluate true responses
            max_total_runs: Maximum total experimental runs allowed
            alpha_significance: Statistical significance level
            improvement_threshold: Minimum R² improvement to continue
        """
        self.n_components = n_components
        self.component_ranges = component_ranges
        self.response_function = response_function
        self.max_total_runs = max_total_runs
        self.alpha_significance = alpha_significance
        self.improvement_threshold = improvement_threshold
        
        # Experimental data storage
        self.design_points = []
        self.responses = []
        self.stage_info = []
        
        # Model progression tracking
        self.model_history = []
        self.r_squared_history = []
        self.significant_effects = {'main': [], 'interactions': []}
        
        # Current stage information
        self.current_stage = 0
        self.stage_names = ['Screening', 'Main Effects Refinement', 'Interaction Detection', 'Model Optimization']
        
        print(f"🧪 Adaptive Sequential DOE Initialized")
        print(f"   Components: {n_components}")
        print(f"   Max runs: {max_total_runs}")
        print(f"   Significance level: {alpha_significance}")
        print(f"   Improvement threshold: {improvement_threshold}")
    
    def run_complete_doe(self) -> Dict:
        """
        Execute the complete adaptive sequential DOE process
        
        Returns:
            Dictionary containing all results and analysis
        """
        print(f"\n{'='*80}")
        print(f"🚀 STARTING ADAPTIVE SEQUENTIAL DOE")
        print(f"{'='*80}")
        
        results = {
            'stage_results': [],
            'final_model': None,
            'model_progression': [],
            'efficiency_metrics': {},
            'design_points': [],
            'responses': [],
            'total_runs': 0
        }
        
        # Stage 1: Screening
        stage1_results = self._stage1_screening()
        results['stage_results'].append(stage1_results)
        
        # Stage 2: Main Effects Refinement
        if self._should_continue_to_next_stage(stage1_results):
            stage2_results = self._stage2_main_effects_refinement()
            results['stage_results'].append(stage2_results)
            
            # Stage 3: Interaction Detection
            if self._should_continue_to_next_stage(stage2_results):
                stage3_results = self._stage3_interaction_detection()
                results['stage_results'].append(stage3_results)
                
                # Stage 4: Model Optimization (if needed)
                if self._should_continue_to_next_stage(stage3_results):
                    stage4_results = self._stage4_model_optimization()
                    results['stage_results'].append(stage4_results)
        
        # Compile final results
        results['final_model'] = self._fit_final_model()
        results['model_progression'] = self.model_history
        results['efficiency_metrics'] = self._calculate_efficiency_metrics()
        results['design_points'] = np.array(self.design_points)
        results['responses'] = np.array(self.responses)
        results['total_runs'] = len(self.design_points)
        
        self._print_final_summary(results)
        
        return results
    
    def _stage1_screening(self) -> Dict:
        """
        Stage 1: Screening the design space to identify main effects
        
        Returns:
            Dictionary with stage 1 results
        """
        print(f"\n{'='*60}")
        print(f"📊 STAGE 1: SCREENING DESIGN SPACE")
        print(f"{'='*60}")
        
        self.current_stage = 1
        
        # Generate screening design (space-filling)
        n_screening_runs = min(15, self.max_total_runs // 3)
        screening_points = self._generate_space_filling_design(n_screening_runs)
        
        print(f"Generated {len(screening_points)} screening points using space-filling design")
        
        # Evaluate responses
        screening_responses = []
        for i, point in enumerate(screening_points):
            response = self.response_function(point)
            screening_responses.append(response)
            self.design_points.append(point)
            self.responses.append(response)
            self.stage_info.append(f"Stage1_Screening_{i+1}")
            
            print(f"  Run {len(self.design_points):2d}: {[f'{x:.3f}' for x in point]} → Response: {response:.3f}")
        
        # Fit initial linear model and analyze main effects
        model_results = self._fit_and_analyze_main_effects()
        
        print(f"\nStage 1 Results:")
        print(f"  R²: {model_results['r_squared']:.3f}")
        print(f"  Significant main effects: {len(model_results['significant_main_effects'])}")
        print(f"  Effects: {model_results['significant_main_effects']}")
        
        stage1_results = {
            'stage': 1,
            'stage_name': 'Screening',
            'runs_added': len(screening_points),
            'total_runs': len(self.design_points),
            'r_squared': model_results['r_squared'],
            'significant_effects': model_results['significant_main_effects'],
            'model': model_results,
            'improvement': model_results['r_squared']  # First stage, so improvement = R²
        }
        
        self.model_history.append(model_results)
        self.r_squared_history.append(model_results['r_squared'])
        
        return stage1_results
    
    def _stage2_main_effects_refinement(self) -> Dict:
        """
        Stage 2: Refine main effects by adding targeted points
        
        Returns:
            Dictionary with stage 2 results
        """
        print(f"\n{'='*60}")
        print(f"🎯 STAGE 2: MAIN EFFECTS REFINEMENT")
        print(f"{'='*60}")
        
        self.current_stage = 2
        previous_r_squared = self.r_squared_history[-1]
        
        # Identify regions for refinement based on significant main effects
        n_refinement_runs = min(10, (self.max_total_runs - len(self.design_points)) // 2)
        refinement_points = self._generate_main_effects_refinement_points(n_refinement_runs)
        
        print(f"Adding {len(refinement_points)} refinement points targeting significant main effects")
        
        # Evaluate responses
        for i, point in enumerate(refinement_points):
            response = self.response_function(point)
            self.design_points.append(point)
            self.responses.append(response)
            self.stage_info.append(f"Stage2_Refinement_{i+1}")
            
            print(f"  Run {len(self.design_points):2d}: {[f'{x:.3f}' for x in point]} → Response: {response:.3f}")
        
        # Refit model with additional data
        model_results = self._fit_and_analyze_main_effects()
        
        improvement = model_results['r_squared'] - previous_r_squared
        
        print(f"\nStage 2 Results:")
        print(f"  R²: {model_results['r_squared']:.3f} (improvement: +{improvement:.3f})")
        print(f"  Significant main effects: {len(model_results['significant_main_effects'])}")
        print(f"  Effects: {model_results['significant_main_effects']}")
        
        stage2_results = {
            'stage': 2,
            'stage_name': 'Main Effects Refinement',
            'runs_added': len(refinement_points),
            'total_runs': len(self.design_points),
            'r_squared': model_results['r_squared'],
            'significant_effects': model_results['significant_main_effects'],
            'model': model_results,
            'improvement': improvement
        }
        
        self.model_history.append(model_results)
        self.r_squared_history.append(model_results['r_squared'])
        
        return stage2_results
    
    def _stage3_interaction_detection(self) -> Dict:
        """
        Stage 3: Detect interaction effects through strategic point placement
        
        Returns:
            Dictionary with stage 3 results
        """
        print(f"\n{'='*60}")
        print(f"🔍 STAGE 3: INTERACTION DETECTION")
        print(f"{'='*60}")
        
        self.current_stage = 3
        previous_r_squared = self.r_squared_history[-1]
        
        # Generate points specifically designed to detect interactions
        remaining_runs = self.max_total_runs - len(self.design_points)
        n_interaction_runs = min(15, remaining_runs)
        
        interaction_points = self._generate_interaction_detection_points(n_interaction_runs)
        
        print(f"Adding {len(interaction_points)} points designed to detect interactions")
        
        # Evaluate responses
        for i, point in enumerate(interaction_points):
            response = self.response_function(point)
            self.design_points.append(point)
            self.responses.append(response)
            self.stage_info.append(f"Stage3_Interaction_{i+1}")
            
            print(f"  Run {len(self.design_points):2d}: {[f'{x:.3f}' for x in point]} → Response: {response:.3f}")
        
        # Fit model with interactions and analyze
        model_results = self._fit_and_analyze_interactions()
        
        improvement = model_results['r_squared'] - previous_r_squared
        
        print(f"\nStage 3 Results:")
        print(f"  R²: {model_results['r_squared']:.3f} (improvement: +{improvement:.3f})")
        print(f"  Significant interactions: {len(model_results['significant_interactions'])}")
        print(f"  Interactions: {model_results['significant_interactions']}")
        
        stage3_results = {
            'stage': 3,
            'stage_name': 'Interaction Detection',
            'runs_added': len(interaction_points),
            'total_runs': len(self.design_points),
            'r_squared': model_results['r_squared'],
            'significant_effects': model_results['significant_interactions'],
            'model': model_results,
            'improvement': improvement
        }
        
        self.model_history.append(model_results)
        self.r_squared_history.append(model_results['r_squared'])
        
        return stage3_results
    
    def _stage4_model_optimization(self) -> Dict:
        """
        Stage 4: Final model optimization with remaining runs
        
        Returns:
            Dictionary with stage 4 results
        """
        print(f"\n{'='*60}")
        print(f"⚡ STAGE 4: MODEL OPTIMIZATION")
        print(f"{'='*60}")
        
        self.current_stage = 4
        previous_r_squared = self.r_squared_history[-1]
        
        # Use remaining runs for model optimization
        remaining_runs = self.max_total_runs - len(self.design_points)
        
        if remaining_runs > 0:
            optimization_points = self._generate_optimization_points(remaining_runs)
            
            print(f"Adding {len(optimization_points)} optimization points")
            
            # Evaluate responses
            for i, point in enumerate(optimization_points):
                response = self.response_function(point)
                self.design_points.append(point)
                self.responses.append(response)
                self.stage_info.append(f"Stage4_Optimization_{i+1}")
                
                print(f"  Run {len(self.design_points):2d}: {[f'{x:.3f}' for x in point]} → Response: {response:.3f}")
        
        # Final model fitting
        model_results = self._fit_final_model()
        
        improvement = model_results['r_squared'] - previous_r_squared
        
        print(f"\nStage 4 Results:")
        print(f"  Final R²: {model_results['r_squared']:.3f} (improvement: +{improvement:.3f})")
        print(f"  Total runs used: {len(self.design_points)}")
        
        stage4_results = {
            'stage': 4,
            'stage_name': 'Model Optimization',
            'runs_added': remaining_runs,
            'total_runs': len(self.design_points),
            'r_squared': model_results['r_squared'],
            'model': model_results,
            'improvement': improvement
        }
        
        self.model_history.append(model_results)
        self.r_squared_history.append(model_results['r_squared'])
        
        return stage4_results
    
    def _generate_space_filling_design(self, n_points: int) -> List[np.ndarray]:
        """Generate space-filling design for screening"""
        points = []
        
        # Use Latin Hypercube-like sampling for mixture constraints
        for _ in range(n_points):
            # Generate random point that satisfies mixture constraints
            point = self._generate_feasible_mixture_point()
            points.append(point)
        
        # Ensure good space coverage by adding extreme points
        if n_points >= self.n_components:
            # Add some vertex-like points
            for i in range(min(self.n_components, n_points - len(points))):
                vertex_point = self._generate_vertex_like_point(i)
                if vertex_point is not None:
                    points.append(vertex_point)
        
        return points[:n_points]
    
    def _generate_main_effects_refinement_points(self, n_points: int) -> List[np.ndarray]:
        """Generate points to refine main effects estimation"""
        points = []
        
        # Focus on regions where significant main effects are most influential
        significant_components = [i for i, effect in enumerate(self.significant_effects['main']) 
                                if f'X{i+1}' in effect]
        
        for _ in range(n_points):
            if significant_components:
                # Generate points that emphasize significant components
                point = self._generate_component_emphasized_point(significant_components)
            else:
                # Fallback to random feasible point
                point = self._generate_feasible_mixture_point()
            points.append(point)
        
        return points
    
    def _generate_interaction_detection_points(self, n_points: int) -> List[np.ndarray]:
        """Generate points specifically designed to detect interactions"""
        points = []
        
        # Generate points at extreme combinations to maximize interaction effects
        for _ in range(n_points):
            point = self._generate_interaction_revealing_point()
            points.append(point)
        
        return points
    
    def _generate_optimization_points(self, n_points: int) -> List[np.ndarray]:
        """Generate points for final model optimization"""
        points = []
        
        # Use D-optimal or I-optimal criteria for remaining points
        existing_points = np.array(self.design_points)
        
        for _ in range(n_points):
            # Generate candidate points and select best for information gain
            candidates = [self._generate_feasible_mixture_point() for _ in range(20)]
            best_point = self._select_best_candidate_point(candidates, existing_points)
            points.append(best_point)
            existing_points = np.vstack([existing_points, best_point])
        
        return points
    
    def _generate_feasible_mixture_point(self) -> np.ndarray:
        """Generate a random feasible mixture point"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Generate random proportions
            point = np.random.random(self.n_components)
            point = point / np.sum(point)  # Normalize to sum to 1
            
            # Check if point satisfies component ranges when converted to parts
            if self._is_point_feasible(point):
                return point
        
        # Fallback: generate centroid
        ranges = np.array(self.component_ranges)
        min_vals, max_vals = ranges[:, 0], ranges[:, 1]
        centroid_parts = (min_vals + max_vals) / 2
        centroid_proportions = centroid_parts / np.sum(centroid_parts)
        
        return centroid_proportions
    
    def _generate_vertex_like_point(self, component_index: int) -> Optional[np.ndarray]:
        """Generate a vertex-like point emphasizing one component"""
        # Try to maximize one component while satisfying constraints
        point = np.array([0.0] * self.n_components)
        
        # Set the emphasized component to its maximum feasible value
        ranges = np.array(self.component_ranges)
        min_vals, max_vals = ranges[:, 0], ranges[:, 1]
        
        # Start with minimum values for all components
        point_parts = min_vals.copy()
        
        # Try to maximize the target component
        remaining_budget = 1.0 - np.sum(point_parts)
        available_increase = max_vals[component_index] - min_vals[component_index]
        
        if remaining_budget > 0 and available_increase > 0:
            increase = min(remaining_budget, available_increase)
            point_parts[component_index] += increase
            
            # Convert to proportions
            point = point_parts / np.sum(point_parts)
            
            if self._is_point_feasible(point):
                return point
        
        return None
    
    def _generate_component_emphasized_point(self, emphasized_components: List[int]) -> np.ndarray:
        """Generate point that emphasizes specific components"""
        point = self._generate_feasible_mixture_point()
        
        # Adjust point to emphasize specified components
        ranges = np.array(self.component_ranges)
        min_vals, max_vals = ranges[:, 0], ranges[:, 1]
        
        # Convert proportions to parts for manipulation
        point_parts = point * np.sum(min_vals + max_vals) / 2  # Rough scaling
        
        # Try to increase emphasized components
        for comp_idx in emphasized_components:
            if comp_idx < len(point_parts):
                target_increase = (max_vals[comp_idx] - min_vals[comp_idx]) * 0.3
                point_parts[comp_idx] += target_increase
        
        # Renormalize
        point = point_parts / np.sum(point_parts)
        
        if self._is_point_feasible(point):
            return point
        else:
            return self._generate_feasible_mixture_point()
    
    def _generate_interaction_revealing_point(self) -> np.ndarray:
        """Generate point designed to reveal interaction effects"""
        # Use extreme combinations that are most likely to show interactions
        point = self._generate_feasible_mixture_point()
        
        # Modify to create more extreme combinations
        ranges = np.array(self.component_ranges)
        min_vals, max_vals = ranges[:, 0], ranges[:, 1]
        
        # Randomly select components to push to extremes
        n_extreme = np.random.randint(2, min(4, self.n_components))
        extreme_components = np.random.choice(self.n_components, size=n_extreme, replace=False)
        
        point_parts = point * np.sum(min_vals + max_vals) / 2
        
        for i, comp_idx in enumerate(extreme_components):
            if i % 2 == 0:
                # Push to maximum
                point_parts[comp_idx] = max_vals[comp_idx] * 0.8
            else:
                # Push to minimum
                point_parts[comp_idx] = min_vals[comp_idx] * 1.2
        
        # Renormalize
        point = point_parts / np.sum(point_parts)
        
        if self._is_point_feasible(point):
            return point
        else:
            return self._generate_feasible_mixture_point()
    
    def _select_best_candidate_point(self, candidates: List[np.ndarray], 
                                   existing_points: np.ndarray) -> np.ndarray:
        """Select best candidate point based on information criteria"""
        best_point = candidates[0]
        best_score = -np.inf
        
        for candidate in candidates:
            # Calculate information gain (simplified D-optimality)
            score = self._calculate_information_score(candidate, existing_points)
            if score > best_score:
                best_score = score
                best_point = candidate
        
        return best_point
    
    def _calculate_information_score(self, candidate: np.ndarray, 
                                   existing_points: np.ndarray) -> float:
        """Calculate information score for candidate point"""
        # Simple distance-based criterion (can be enhanced)
        if len(existing_points) == 0:
            return 1.0
        
        # Calculate minimum distance to existing points
        distances = np.linalg.norm(existing_points - candidate, axis=1)
        min_distance = np.min(distances)
        
        return min_distance
    
    def _is_point_feasible(self, point: np.ndarray) -> bool:
        """Check if a proportion point is feasible given component ranges"""
        # Convert proportions to parts (approximate)
        ranges = np.array(self.component_ranges)
        min_vals, max_vals = ranges[:, 0], ranges[:, 1]
        
        # Rough conversion from proportions to parts
        total_parts = np.sum(min_vals + max_vals) / 2  # Approximate scaling
        point_parts = point * total_parts
        
        # Check bounds
        for i, (min_val, max_val) in enumerate(self.component_ranges):
            if point_parts[i] < min_val * 0.9 or point_parts[i] > max_val * 1.1:
                return False
        
        return True
    
    def _fit_and_analyze_main_effects(self) -> Dict:
        """Fit linear model and analyze main effects significance"""
        if len(self.design_points) < self.n_components:
            return {'r_squared': 0.0, 'significant_main_effects': [], 'model': None}
        
        X = np.array(self.design_points)
        y = np.array(self.responses)
        
        # Fit linear model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # Statistical significance testing
        n = len(y)
        p = self.n_components
        
        # Calculate t-statistics for coefficients
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - p - 1)
        
        # Approximate standard errors (simplified)
        X_centered = X - np.mean(X, axis=0)
        if np.linalg.det(X_centered.T @ X_centered) != 0:
            coeff_var = mse * np.diag(np.linalg.inv(X_centered.T @ X_centered))
            se_coeffs = np.sqrt(coeff_var)
            t_stats = model.coef_ / se_coeffs
            
            # Critical t-value
            df = n - p - 1
            t_critical = stats.t.ppf(1 - self.alpha_significance/2, df)
            
            # Identify significant main effects
            significant_main_effects = []
            for i, t_stat in enumerate(t_stats):
                if abs(t_stat) > t_critical:
                    significant_main_effects.append(f'X{i+1}')
            
            self.significant_effects['main'] = significant_main_effects
        else:
            significant_main_effects = []
        
        return {
            'r_squared': r_squared,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'significant_main_effects': significant_main_effects,
            'model': model,
            'predictions': y_pred
        }
    
    def _fit_and_analyze_interactions(self) -> Dict:
        """Fit model with interactions and analyze significance"""
        if len(self.design_points) < 10:  # Need sufficient data for interactions
            return self._fit_and_analyze_main_effects()
        
        X = np.array(self.design_points)
        y = np.array(self.responses)
        
        # Create interaction terms (2-way)
        X_with_interactions = []
        term_names = []
        
        # Add main effects
        for i in range(self.n_components):
            X_with_interactions.append(X[:, i])
            term_names.append(f'X{i+1}')
        
        # Add 2-way interactions
        for i in range(self.n_components):
            for j in range(i+1, self.n_components):
                interaction_term = X[:, i] * X[:, j]
                X_with_interactions.append(interaction_term)
                term_names.append(f'X{i+1}*X{j+1}')
        
        X_full = np.column_stack(X_with_interactions)
        
        # Fit model with interactions
        model = LinearRegression()
        model.fit(X_full, y)
        y_pred = model.predict(X_full)
        r_squared = r2_score(y, y_pred)
        
        # Identify significant interactions (simplified)
        significant_interactions = []
        n_main = self.n_components
        
        for i, coeff in enumerate(model.coef_[n_main:]):
            if abs(coeff) > np.std(model.coef_) * 1.5:  # Simple threshold
                significant_interactions.append(term_names[n_main + i])
        
        self.significant_effects['interactions'] = significant_interactions
        
        return {
            'r_squared': r_squared,
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'significant_main_effects': self.significant_effects['main'],
            'significant_interactions': significant_interactions,
            'term_names': term_names,
            'model': model,
            'predictions': y_pred
        }
    
    def _fit_final_model(self) -> Dict:
        """Fit the final comprehensive model"""
        return self._fit_and_analyze_interactions()
    
    def _should_continue_to_next_stage(self, stage_results: Dict) -> bool:
        """Determine if should continue to next stage based on results"""
        if len(self.design_points) >= self.max_total_runs:
            return False
        
        if stage_results['improvement'] < self.improvement_threshold:
            print(f"  ⏹️  Stopping: Improvement {stage_results['improvement']:.3f} < threshold {self.improvement_threshold}")
            return False
        
        return True
    
    def _calculate_efficiency_metrics(self) -> Dict:
        """Calculate various efficiency metrics"""
        final_model = self.model_history[-1]
        
        # Create design matrix for D-efficiency calculation
        X = np.array(self.design_points)
        design_matrix = []
        for point in X:
            row = evaluate_mixture_model_terms(point.tolist(), "quadratic")
            design_matrix.append(row)
        
        design_matrix = np.array(design_matrix)
        d_efficiency = calculate_d_efficiency(design_matrix, model_type='quadratic')
        
        return {
            'final_r_squared': final_model['r_squared'],
            'd_efficiency': d_efficiency,
            'total_runs': len(self.design_points),
            'runs_per_stage': [len([s for s in self.stage_info if f'Stage{i}' in s]) 
                              for i in range(1, 5)],
            'r_squared_progression': self.r_squared_history
        }
    
    def _print_final_summary(self, results: Dict):
        """Print comprehensive final summary"""
        print(f"\n{'='*80}")
        print(f"🎉 ADAPTIVE SEQUENTIAL DOE COMPLETE")
        print(f"{'='*80}")
        
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Total experimental runs: {results['total_runs']}")
        print(f"   Final R²: {results['efficiency_metrics']['final_r_squared']:.3f}")
        print(f"   D-efficiency: {results['efficiency_metrics']['d_efficiency']:.3f}")
        
        print(f"\n🎭 STAGE BREAKDOWN:")
        for i, stage_result in enumerate(results['stage_results']):
            print(f"   Stage {stage_result['stage']}: {stage_result['stage_name']}")
            print(f"     Runs added: {stage_result['runs_added']}")
            print(f"     R²: {stage_result['r_squared']:.3f}")
            print(f"     Improvement: +{stage_result['improvement']:.3f}")
            if 'significant_effects' in stage_result:
                print(f"     Significant effects: {len(stage_result['significant_effects'])}")
        
        print(f"\n📈 MODEL PROGRESSION:")
        for i, r_sq in enumerate(results['efficiency_metrics']['r_squared_progression']):
            print(f"   After Stage {i+1}: R² = {r_sq:.3f}")
        
        efficiency_grade = "Excellent" if results['efficiency_metrics']['final_r_squared'] > 0.9 else \
                          "Good" if results['efficiency_metrics']['final_r_squared'] > 0.8 else \
                          "Fair" if results['efficiency_metrics']['final_r_squared'] > 0.6 else "Poor"
        
        print(f"\n🏆 OVERALL ASSESSMENT: {efficiency_grade}")
        print(f"   The adaptive sequential DOE achieved {efficiency_grade.lower()} model accuracy")
        print(f"   using {results['total_runs']} strategically planned experiments.")
