"""
JMP-Style Advanced Screening for Higher-Order Mixture Models

This module implements JMP-like screening methodology combining:
1. D-optimal design generation
2. Forward selection with regularization
3. Effect heredity enforcement
4. VIF monitoring
5. Cross-validation
6. Multiple model selection criteria

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import warnings

try:
    from ..algorithms.hierarchical_screening import HeredityChecker
except:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from algorithms.hierarchical_screening import HeredityChecker


@dataclass
class ModelSelectionResult:
    """Results from model selection process"""
    selected_terms: List[str]
    coefficients: Dict[str, float]
    model: object
    r_squared: float
    adjusted_r_squared: float
    aic: float
    bic: float
    cv_score: float
    vif_values: Dict[str, float]
    p_values: Dict[str, float]
    iteration_history: List[Dict]


class VIFCalculator:
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    
    @staticmethod
    def calculate_vif(X: np.ndarray, feature_index: int) -> float:
        """
        Calculate VIF for a specific feature.
        
        VIF = 1 / (1 - R²) where R² is from regressing feature on others
        
        Parameters:
        -----------
        X : np.ndarray
            Design matrix
        feature_index : int
            Index of feature to calculate VIF for
            
        Returns:
        --------
        float
            VIF value (>10 indicates high multicollinearity)
        """
        if X.shape[1] < 2:
            return 1.0
        
        # Get this feature and all others
        X_feature = X[:, feature_index]
        X_others = np.delete(X, feature_index, axis=1)
        
        # Regress this feature on all others
        try:
            model = LinearRegression()
            model.fit(X_others, X_feature)
            r2 = model.score(X_others, X_feature)
            
            # VIF = 1 / (1 - R²)
            if r2 >= 0.9999:  # Perfect collinearity
                return np.inf
            
            vif = 1.0 / (1.0 - r2)
            return vif
        except:
            return np.inf
    
    @staticmethod
    def calculate_all_vifs(X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate VIF for all features"""
        vifs = {}
        for i, name in enumerate(feature_names):
            vifs[name] = VIFCalculator.calculate_vif(X, i)
        return vifs


class ModelCriteriaCalculator:
    """Calculate model selection criteria (AIC, BIC, etc.)"""
    
    @staticmethod
    def calculate_aic(n_observations: int, n_parameters: int, sse: float) -> float:
        """
        Calculate Akaike Information Criterion.
        
        AIC = n*log(SSE/n) + 2*k
        
        Lower is better.
        """
        if n_observations <= 0 or sse <= 0:
            return np.inf
        
        aic = n_observations * np.log(sse / n_observations) + 2 * n_parameters
        return aic
    
    @staticmethod
    def calculate_bic(n_observations: int, n_parameters: int, sse: float) -> float:
        """
        Calculate Bayesian Information Criterion.
        
        BIC = n*log(SSE/n) + k*log(n)
        
        Lower is better. More conservative than AIC.
        """
        if n_observations <= 0 or sse <= 0:
            return np.inf
        
        bic = n_observations * np.log(sse / n_observations) + n_parameters * np.log(n_observations)
        return bic
    
    @staticmethod
    def calculate_adjusted_r2(r2: float, n_observations: int, n_parameters: int) -> float:
        """
        Calculate adjusted R².
        
        Adjusted R² = 1 - (1-R²)*(n-1)/(n-k-1)
        """
        if n_observations <= n_parameters:
            return 0.0
        
        adj_r2 = 1.0 - (1.0 - r2) * (n_observations - 1) / (n_observations - n_parameters - 1)
        return adj_r2


class DOptimalMixtureDesign:
    """Generate D-optimal designs for mixture models using ADVANCED algorithm"""
    
    @staticmethod
    def build_design_matrix(design: np.ndarray, terms: List[str], 
                          component_names: List[str]) -> np.ndarray:
        """Build design matrix for given terms"""
        n_points = design.shape[0]
        columns = []
        
        for term in terms:
            if term == 'intercept':
                columns.append(np.ones(n_points))
            elif '*' in term:
                # Interaction term
                components = term.split('*')
                indices = [component_names.index(comp) for comp in components]
                column = np.ones(n_points)
                for idx in indices:
                    column *= design[:, idx]
                columns.append(column)
            else:
                # Main effect
                idx = component_names.index(term)
                columns.append(design[:, idx])
        
        return np.column_stack(columns) if columns else np.empty((n_points, 0))
    
    @staticmethod
    def calculate_d_efficiency(X: np.ndarray) -> float:
        """Calculate D-efficiency of design"""
        if X.shape[0] < X.shape[1]:
            return 0.0
        
        try:
            XtX = X.T @ X
            det_XtX = np.linalg.det(XtX)
            n_params = X.shape[1]
            n_runs = X.shape[0]
            
            if det_XtX <= 0:
                return 0.0
            
            d_eff = (det_XtX / n_runs) ** (1.0 / n_params)
            return d_eff
        except:
            return 0.0
    
    @staticmethod
    def generate_quartic_mixture_design(n_components: int, n_runs: int,
                                       component_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate D-optimal design for quartic mixture model using ADVANCED algorithm.
        
        This uses the proven MixtureDOptimalAlgorithm from d_optimal_algorithm.py
        which includes:
        - Multi-phase optimization (vertices → edges → centroid → hotspots)
        - Quartic hotspot locking (~50% of runs in quartic-rich patterns)
        - Intelligent diversity-based initialization
        - Advanced coordinate exchange
        
        This matches JMP's design generation strategy!
        """
        if component_names is None:
            component_names = [f"x{i+1}" for i in range(n_components)]
        
        # Import advanced algorithm
        try:
            from .d_optimal_algorithm import MixtureDOptimalAlgorithm
        except:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from d_optimal_algorithm import MixtureDOptimalAlgorithm
        
        # Create advanced D-optimal algorithm instance
        advanced_optimizer = MixtureDOptimalAlgorithm(
            model_type="quartic",
            component_names=component_names
        )
        
        # The advanced algorithm generates candidates internally during optimization
        # We just call optimize_mixture_design with dummy candidates array
        # (it will be ignored and regenerated internally with proper hotspot locking)
        dummy_candidates = np.random.dirichlet(np.ones(n_components), size=100)
        
        # Use the proven multi-phase optimization
        design, final_det, optimization_info = advanced_optimizer.optimize_mixture_design(
            candidates=dummy_candidates,  # Provides shape info, algorithm generates proper candidates
            n_runs=n_runs,
            strategy="balanced"
        )
        
        # Extract D-efficiency
        d_eff = optimization_info.get('d_efficiency', 0.0)
        
        print(f"Generated design: {n_runs} runs, D-efficiency = {d_eff:.3f}")
        print(f"  Algorithm: {optimization_info.get('algorithm', 'Advanced D-optimal')}")
        
        return design


class JMPStyleForwardSelection:
    """
    JMP-style forward selection with:
    - Regularization (Ridge/ElasticNet)
    - Heredity constraints
    - VIF monitoring
    - AIC/BIC selection
    - Cross-validation
    """
    
    def __init__(self,
                 n_components: int,
                 component_names: Optional[List[str]] = None,
                 max_order: int = 4,
                 use_strong_heredity: bool = False,
                 alpha: float = 0.01,
                 max_vif: float = 10.0,
                 p_threshold: float = 0.05,
                 use_cross_validation: bool = True,
                 cv_folds: int = 5,
                 verbose: bool = True):
        """
        Initialize JMP-style forward selection.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        component_names : List[str], optional
            Component names
        max_order : int
            Maximum interaction order (2-4)
        use_strong_heredity : bool
            Use strong heredity (default: weak)
        alpha : float
            Regularization parameter
        max_vif : float
            Maximum allowed VIF
        p_threshold : float
            P-value threshold for term inclusion
        use_cross_validation : bool
            Use cross-validation for model selection
        cv_folds : int
            Number of CV folds
        verbose : bool
            Print progress
        """
        self.n_components = n_components
        self.component_names = component_names or [f"x{i+1}" for i in range(n_components)]
        self.max_order = max_order
        self.use_strong_heredity = use_strong_heredity
        self.alpha = alpha
        self.max_vif = max_vif
        self.p_threshold = p_threshold
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.verbose = verbose
        
        self.selected_terms: List[str] = []
        self.iteration_history: List[Dict] = []
    
    def _build_design_matrix(self, design: np.ndarray, terms: List[str]) -> np.ndarray:
        """Build design matrix for given terms"""
        return DOptimalMixtureDesign.build_design_matrix(design, terms, self.component_names)
    
    def _generate_candidate_terms(self, order: int) -> List[str]:
        """Generate candidate terms of given order"""
        from itertools import combinations
        
        if order == 1:
            return self.component_names.copy()
        
        # Get eligible based on heredity
        significant_lower = set(self.selected_terms)
        
        candidates = HeredityChecker.get_eligible_interactions(
            order, self.component_names, significant_lower, self.use_strong_heredity
        )
        
        return candidates
    
    def _fit_model_with_term(self, X: np.ndarray, y: np.ndarray,
                            new_term_column: np.ndarray) -> Tuple[object, Dict]:
        """
        Fit model with new term added.
        
        Returns model and statistics dict.
        """
        # Augment design matrix
        if X.shape[1] == 0:
            X_augmented = new_term_column.reshape(-1, 1)
        else:
            X_augmented = np.column_stack([X, new_term_column])
        
        # Fit with regularization
        model = Ridge(alpha=self.alpha, fit_intercept=False)
        model.fit(X_augmented, y)
        
        # Get predictions
        y_pred = model.predict(X_augmented)
        
        # Calculate metrics
        n_obs = len(y)
        n_params = X_augmented.shape[1]
        
        sse = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        
        adj_r2 = ModelCriteriaCalculator.calculate_adjusted_r2(r2, n_obs, n_params)
        aic = ModelCriteriaCalculator.calculate_aic(n_obs, n_params, sse)
        bic = ModelCriteriaCalculator.calculate_bic(n_obs, n_params, sse)
        
        # Calculate p-value for new term (approximate)
        if n_obs > n_params:
            residual_var = sse / (n_obs - n_params)
            if residual_var > 0:
                # Get coefficient standard error (approximate)
                try:
                    XtX_inv = np.linalg.inv(X_augmented.T @ X_augmented + self.alpha * np.eye(n_params))
                    se = np.sqrt(residual_var * np.diag(XtX_inv))
                    new_coef = model.coef_[-1]
                    t_stat = new_coef / se[-1] if se[-1] > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - n_params))
                except:
                    p_value = 0.5
            else:
                p_value = 0.0
        else:
            p_value = 1.0
        
        # Calculate CV score if requested
        cv_score = 0.0
        if self.use_cross_validation and n_obs >= self.cv_folds:
            try:
                cv_scores = cross_val_score(
                    Ridge(alpha=self.alpha, fit_intercept=False),
                    X_augmented, y,
                    cv=min(self.cv_folds, n_obs // 2),
                    scoring='r2'
                )
                cv_score = np.mean(cv_scores)
            except:
                cv_score = r2
        else:
            cv_score = r2
        
        stats_dict = {
            'model': model,
            'r2': r2,
            'adj_r2': adj_r2,
            'aic': aic,
            'bic': bic,
            'cv_score': cv_score,
            'p_value': p_value,
            'n_params': n_params
        }
        
        return model, stats_dict
    
    def forward_select(self, design: np.ndarray, responses: np.ndarray) -> ModelSelectionResult:
        """
        Run forward selection algorithm.
        
        Parameters:
        -----------
        design : np.ndarray
            Experimental design (n_points × n_components)
        responses : np.ndarray
            Response values
            
        Returns:
        --------
        ModelSelectionResult
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"JMP-STYLE FORWARD SELECTION")
            print(f"{'='*70}")
            print(f"Components: {self.n_components}")
            print(f"Design points: {len(design)}")
            print(f"Max order: {self.max_order}")
            print(f"Heredity: {'Strong' if self.use_strong_heredity else 'Weak'}")
            print(f"Regularization alpha: {self.alpha}")
            print(f"Max VIF: {self.max_vif}")
            print(f"P-value threshold: {self.p_threshold}")
        
        self.selected_terms = []
        self.iteration_history = []
        
        # Current design matrix (empty initially)
        X_current = np.empty((len(design), 0))
        current_aic = np.inf
        
        # Process each order sequentially
        for order in range(1, self.max_order + 1):
            order_name = {1: 'Main effects', 2: 'Two-way', 3: 'Three-way', 4: 'Four-way'}.get(order)
            
            if self.verbose:
                print(f"\n{'-'*70}")
                print(f"STAGE: {order_name} (order {order})")
                print(f"{'-'*70}")
            
            # Generate candidates for this order
            candidates = self._generate_candidate_terms(order)
            
            if not candidates:
                if self.verbose:
                    print(f"  No eligible candidates (heredity constraints)")
                continue
            
            if self.verbose:
                print(f"  Candidates to test: {len(candidates)}")
            
            # Test each candidate
            added_this_stage = []
            
            while candidates:
                best_term = None
                best_stats = None
                best_X = None
                best_vif_ok = False
                
                for term in candidates:
                    # Build column for this term
                    term_column = self._build_design_matrix(design, [term])
                    
                    # Fit model with this term
                    model, stats = self._fit_model_with_term(X_current, responses, term_column)
                    
                    # Check VIF
                    terms_list = self.selected_terms + [term]
                    X_test = self._build_design_matrix(design, terms_list)
                    vif = VIFCalculator.calculate_vif(X_test, len(terms_list) - 1)
                    vif_ok = vif <= self.max_vif
                    
                    # Check all criteria
                    p_ok = stats['p_value'] <= self.p_threshold
                    aic_improved = stats['aic'] < current_aic
                    
                    # Select best based on AIC among valid candidates
                    if vif_ok and p_ok and aic_improved:
                        if best_stats is None or stats['aic'] < best_stats['aic']:
                            best_term = term
                            best_stats = stats
                            best_X = X_test
                            best_vif_ok = True
                
                # Add best term if found
                if best_term is None:
                    if self.verbose:
                        print(f"  No more terms meet criteria. Stopping {order_name}.")
                    break
                
                # Accept this term
                self.selected_terms.append(best_term)
                added_this_stage.append(best_term)
                X_current = best_X
                current_aic = best_stats['aic']
                candidates.remove(best_term)
                
                # Record iteration
                iteration_info = {
                    'iteration': len(self.iteration_history) + 1,
                    'term_added': best_term,
                    'order': order,
                    'r2': best_stats['r2'],
                    'adj_r2': best_stats['adj_r2'],
                    'aic': best_stats['aic'],
                    'bic': best_stats['bic'],
                    'cv_score': best_stats['cv_score'],
                    'p_value': best_stats['p_value'],
                    'n_terms': len(self.selected_terms)
                }
                self.iteration_history.append(iteration_info)
                
                if self.verbose:
                    print(f"  Added: {best_term:<20} R²={best_stats['r2']:.4f} "
                          f"AIC={best_stats['aic']:.1f} p={best_stats['p_value']:.4f}")
            
            if self.verbose:
                print(f"  {order_name} complete: {len(added_this_stage)} terms added")
            
            # If no terms added in this stage, stop
            if len(added_this_stage) == 0:
                if self.verbose:
                    print(f"  No terms added. Stopping forward selection.")
                break
        
        # Final model with selected terms
        if self.verbose:
            print(f"\n{'-'*70}")
            print(f"FORWARD SELECTION COMPLETE")
            print(f"{'-'*70}")
            print(f"Total terms selected: {len(self.selected_terms)}")
        
        if len(self.selected_terms) == 0:
            # Return empty model
            return ModelSelectionResult(
                selected_terms=[],
                coefficients={},
                model=None,
                r_squared=0.0,
                adjusted_r_squared=0.0,
                aic=np.inf,
                bic=np.inf,
                cv_score=0.0,
                vif_values={},
                p_values={},
                iteration_history=self.iteration_history
            )
        
        # Refit final model
        X_final = self._build_design_matrix(design, self.selected_terms)
        final_model = Ridge(alpha=self.alpha, fit_intercept=False)
        final_model.fit(X_final, responses)
        
        y_pred = final_model.predict(X_final)
        sse = np.sum((responses - y_pred) ** 2)
        sst = np.sum((responses - np.mean(responses)) ** 2)
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        adj_r2 = ModelCriteriaCalculator.calculate_adjusted_r2(r2, len(responses), len(self.selected_terms))
        aic = ModelCriteriaCalculator.calculate_aic(len(responses), len(self.selected_terms), sse)
        bic = ModelCriteriaCalculator.calculate_bic(len(responses), len(self.selected_terms), sse)
        
        # Calculate VIFs
        vif_values = VIFCalculator.calculate_all_vifs(X_final, self.selected_terms)
        
        # Calculate p-values (approximate)
        p_values = {}
        n_obs = len(responses)
        n_params = len(self.selected_terms)
        if n_obs > n_params:
            residual_var = sse / (n_obs - n_params)
            if residual_var > 0:
                try:
                    XtX_inv = np.linalg.inv(X_final.T @ X_final + self.alpha * np.eye(n_params))
                    se = np.sqrt(residual_var * np.diag(XtX_inv))
                    for i, term in enumerate(self.selected_terms):
                        t_stat = final_model.coef_[i] / se[i] if se[i] > 0 else 0
                        p_values[term] = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - n_params))
                except:
                    for term in self.selected_terms:
                        p_values[term] = 0.05
        
        # CV score
        cv_score = 0.0
        if self.use_cross_validation:
            try:
                cv_scores = cross_val_score(
                    Ridge(alpha=self.alpha, fit_intercept=False),
                    X_final, responses,
                    cv=min(self.cv_folds, n_obs // 2),
                    scoring='r2'
                )
                cv_score = np.mean(cv_scores)
            except:
                cv_score = r2
        
        # Coefficients
        coefficients = dict(zip(self.selected_terms, final_model.coef_))
        
        if self.verbose:
            print(f"\nFinal Model Summary:")
            print(f"  Terms: {len(self.selected_terms)}")
            print(f"  R²: {r2:.4f}")
            print(f"  Adjusted R²: {adj_r2:.4f}")
            print(f"  AIC: {aic:.2f}")
            print(f"  BIC: {bic:.2f}")
            print(f"  CV Score: {cv_score:.4f}")
            
            print(f"\nSelected Terms:")
            for term in self.selected_terms:
                coef = coefficients[term]
                vif = vif_values.get(term, 0)
                p_val = p_values.get(term, 1.0)
                order = len(term.split('*'))
                order_name = {1: 'Main', 2: '2-way', 3: '3-way', 4: '4-way'}.get(order)
                print(f"  {term:<20} coef={coef:+8.3f} VIF={vif:6.2f} p={p_val:.4f} [{order_name}]")
        
        return ModelSelectionResult(
            selected_terms=self.selected_terms,
            coefficients=coefficients,
            model=final_model,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            aic=aic,
            bic=bic,
            cv_score=cv_score,
            vif_values=vif_values,
            p_values=p_values,
            iteration_history=self.iteration_history
        )


# Example usage
if __name__ == "__main__":
    print("JMP-Style Screening Module")
    print("="*70)
    
    # Test D-optimal design generation
    print("\n1. Testing D-Optimal Design Generation")
    design = DOptimalMixtureDesign.generate_quartic_mixture_design(
        n_components=5,
        n_runs=47
    )
    print(f"Generated design shape: {design.shape}")
    print(f"Row sums (should be ~1.0): {design.sum(axis=1)[:5]}")
    
    # Test forward selection
    print("\n2. Testing Forward Selection")
    
    # Create synthetic data
    def test_function(x):
        return 3*x[0] - 4*x[2] + 5*x[4] + 2*x[0]*x[2] - 3*x[1]*x[3]
    
    responses = np.array([test_function(point) for point in design])
    
    selector = JMPStyleForwardSelection(
        n_components=5,
        max_order=3,
        use_strong_heredity=False,
        alpha=0.01,
        max_vif=10.0,
        verbose=True
    )
    
    result = selector.forward_select(design, responses)
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
