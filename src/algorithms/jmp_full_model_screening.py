"""
JMP-Style Full Model Screening with Term Removal

This implements JMP's actual approach:
1. Generate D-optimal design for FULL model (all terms)
2. Fit the full model with regularization
3. Identify non-significant terms
4. Remove them and refit
5. Iteratively refine

This is different from forward selection - it starts with everything
and removes what's not significant.
"""

import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import Ridge

from .jmp_style_screening import (
    DOptimalMixtureDesign,
    VIFCalculator,
    ModelCriteriaCalculator,
    ModelSelectionResult
)


@dataclass
class TermPruningResult:
    """Results from iterative term pruning"""
    iteration: int
    removed_terms: List[str]
    remaining_terms: List[str]
    r_squared: float
    aic: float
    bic: float
    n_insignificant: int


class JMPFullModelScreening:
    """
    JMP-style screening: Fit full model, then iteratively remove
    non-significant terms until only significant ones remain.
    
    This matches JMP's actual workflow:
    1. Generate design for full model
    2. Fit with regularization
    3. Check significance of each term
    4. Remove non-significant terms
    5. Refit and repeat until convergence
    """
    
    def __init__(self,
                 n_components: int,
                 component_names: Optional[List[str]] = None,
                 max_order: int = 4,
                 alpha: float = 0.01,
                 p_threshold: float = 0.05,
                 max_vif: float = 10.0,
                 min_terms: int = 1,
                 verbose: bool = True):
        """
        Initialize full model screening.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        component_names : List[str], optional
            Component names
        max_order : int
            Maximum interaction order
        alpha : float
            Regularization parameter for Ridge regression
        p_threshold : float
            P-value threshold for significance
        max_vif : float
            Maximum allowed VIF
        min_terms : int
            Minimum number of terms to keep
        verbose : bool
            Print progress
        """
        self.n_components = n_components
        self.component_names = component_names or [f"x{i+1}" for i in range(n_components)]
        self.max_order = max_order
        self.alpha = alpha
        self.p_threshold = p_threshold
        self.max_vif = max_vif
        self.min_terms = min_terms
        self.verbose = verbose
        
        self.pruning_history: List[TermPruningResult] = []
    
    def _generate_all_terms(self) -> List[str]:
        """Generate all possible terms up to max_order"""
        from itertools import combinations
        
        terms = []
        
        # Main effects
        terms.extend(self.component_names)
        
        # Interactions from 2-way up to max_order
        for order in range(2, self.max_order + 1):
            for combo in combinations(range(self.n_components), order):
                term_components = [self.component_names[i] for i in combo]
                terms.append('*'.join(term_components))
        
        return terms
    
    def _fit_and_get_significance(self, design: np.ndarray, responses: np.ndarray,
                                  terms: List[str]) -> Dict[str, Dict]:
        """
        Fit model with OLS and get coefficient significance for each term.
        
        CORRECTED: Uses OLS (not Ridge) for valid p-values!
        Ridge regularization biases coefficients and produces invalid p-values.
        
        Returns dict with term -> {coef, p_value, vif}
        """
        from sklearn.linear_model import LinearRegression
        
        # Build design matrix
        X = DOptimalMixtureDesign.build_design_matrix(design, terms, self.component_names)
        
        # CORRECTED: Use OLS for proper significance testing
        # Ridge regression produces biased coefficients and invalid p-values
        model = LinearRegression(fit_intercept=False)
        
        try:
            model.fit(X, responses)
            
            # Get predictions and residuals
            y_pred = model.predict(X)
            n_obs = len(responses)
            n_params = len(terms)
            
            sse = np.sum((responses - y_pred) ** 2)
            
            # Calculate p-values
            term_stats = {}
            
            if n_obs > n_params:
                residual_var = sse / (n_obs - n_params)
                
                if residual_var > 0:
                    try:
                        # CORRECTED: No regularization penalty in XtX_inv!
                        # This gives TRUE standard errors for OLS
                        XtX_inv = np.linalg.inv(X.T @ X)
                        se = np.sqrt(residual_var * np.diag(XtX_inv))
                        
                        for i, term in enumerate(terms):
                            coef = model.coef_[i]
                            t_stat = coef / se[i] if se[i] > 0 else 0
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - n_params))
                            vif = VIFCalculator.calculate_vif(X, i)
                            
                            term_stats[term] = {
                                'coefficient': coef,
                                'p_value': p_value,
                                'vif': vif,
                                't_stat': t_stat
                            }
                    except np.linalg.LinAlgError:
                        # Matrix singular - use Ridge as fallback with warning
                        model_ridge = Ridge(alpha=self.alpha, fit_intercept=False)
                        model_ridge.fit(X, responses)
                        
                        if self.verbose:
                            print(f"  ⚠️  WARNING: Singular matrix, using Ridge fallback")
                        
                        for i, term in enumerate(terms):
                            term_stats[term] = {
                                'coefficient': model_ridge.coef_[i],
                                'p_value': 0.5,  # Uncertain - mark for review
                                'vif': 999.0,
                                't_stat': 0
                            }
                        return term_stats, model_ridge, sse
                else:
                    # Zero residual variance
                    for i, term in enumerate(terms):
                        term_stats[term] = {
                            'coefficient': model.coef_[i],
                            'p_value': 0.0,  # Perfect fit
                            'vif': VIFCalculator.calculate_vif(X, i),
                            't_stat': float('inf')
                        }
            else:
                # Not enough observations
                for i, term in enumerate(terms):
                    term_stats[term] = {
                        'coefficient': model.coef_[i],
                        'p_value': 1.0,
                        'vif': VIFCalculator.calculate_vif(X, i),
                        't_stat': 0
                    }
            
            return term_stats, model, sse
            
        except Exception as e:
            # Complete fallback
            if self.verbose:
                print(f"  ⚠️  ERROR in fit: {e}")
            
            # Return Ridge as emergency fallback
            model_ridge = Ridge(alpha=self.alpha, fit_intercept=False)
            model_ridge.fit(X, responses)
            y_pred = model_ridge.predict(X)
            sse = np.sum((responses - y_pred) ** 2)
            
            term_stats = {}
            for i, term in enumerate(terms):
                term_stats[term] = {
                    'coefficient': model_ridge.coef_[i],
                    'p_value': 0.5,
                    'vif': 999.0,
                    't_stat': 0
                }
            
            return term_stats, model_ridge, sse
    
    def screen_full_model(self, design: np.ndarray, responses: np.ndarray) -> ModelSelectionResult:
        """
        Fit full model and iteratively remove non-significant terms.
        
        Parameters:
        -----------
        design : np.ndarray
            Experimental design
        responses : np.ndarray
            Response values
            
        Returns:
        --------
        ModelSelectionResult
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"JMP-STYLE FULL MODEL SCREENING")
            print(f"{'='*70}")
            print(f"Design points: {len(design)}")
            print(f"Max order: {self.max_order}")
            print(f"P-value threshold: {self.p_threshold}")
            print(f"Max VIF: {self.max_vif}")
        
        # Generate all possible terms
        all_terms = self._generate_all_terms()
        current_terms = all_terms.copy()
        
        if self.verbose:
            print(f"\n📊 FULL MODEL:")
            print(f"   Total terms: {len(all_terms)}")
            
            # Count by order
            by_order = {}
            for term in all_terms:
                order = len(term.split('*'))
                by_order[order] = by_order.get(order, 0) + 1
            
            for order in sorted(by_order.keys()):
                order_name = {1: "Main effects", 2: "2-way", 3: "3-way", 4: "4-way"}.get(order)
                print(f"   {order_name}: {by_order[order]}")
        
        # Iterative pruning
        iteration = 0
        converged = False
        
        while not converged and len(current_terms) > self.min_terms:
            iteration += 1
            
            if self.verbose:
                print(f"\n{'-'*70}")
                print(f"ITERATION {iteration}: Testing {len(current_terms)} terms")
                print(f"{'-'*70}")
            
            # Fit current model
            term_stats, model, sse = self._fit_and_get_significance(
                design, responses, current_terms
            )
            
            # Calculate model metrics
            n_obs = len(responses)
            n_params = len(current_terms)
            sst = np.sum((responses - np.mean(responses)) ** 2)
            r2 = 1.0 - sse / sst if sst > 0 else 0.0
            aic = ModelCriteriaCalculator.calculate_aic(n_obs, n_params, sse)
            bic = ModelCriteriaCalculator.calculate_bic(n_obs, n_params, sse)
            
            if self.verbose:
                print(f"  R² = {r2:.4f}, AIC = {aic:.2f}, BIC = {bic:.2f}")
            
            # Identify terms to remove - RESPECTING HIERARCHY
            # Remove higher-order terms before lower-order ones
            terms_to_remove = []
            
            for term, stats_dict in term_stats.items():
                p_val = stats_dict['p_value']
                vif = stats_dict['vif']
                order = len(term.split('*'))
                
                # Remove if not significant OR VIF too high
                # But NEVER remove main effects due to VIF (mixture constraint issue)
                if p_val > self.p_threshold:
                    terms_to_remove.append((term, 'p-value', p_val, order))
                elif vif > self.max_vif and order > 1:  # Only check VIF for interactions
                    terms_to_remove.append((term, 'VIF', vif, order))
            
            # Sort by order (remove highest order first), then by p-value
            # This respects effect hierarchy: remove 4-way before 3-way before 2-way
            terms_to_remove.sort(key=lambda x: (-x[3], -x[2]), reverse=False)
            
            if terms_to_remove:
                # Remove one term at a time (most conservative)
                term_to_remove, reason, value, order = terms_to_remove[0]
                current_terms.remove(term_to_remove)
                
                if self.verbose:
                    print(f"  Removed: {term_to_remove} ({reason}={value:.4f})")
                
                # Record pruning step
                pruning_result = TermPruningResult(
                    iteration=iteration,
                    removed_terms=[term_to_remove],
                    remaining_terms=current_terms.copy(),
                    r_squared=r2,
                    aic=aic,
                    bic=bic,
                    n_insignificant=len(terms_to_remove)
                )
                self.pruning_history.append(pruning_result)
            else:
                # No more terms to remove - converged!
                converged = True
                if self.verbose:
                    print(f"  ✓ Converged! All remaining terms are significant.")
        
        # Final model with selected terms
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"FINAL MODEL")
            print(f"{'='*70}")
            print(f"Selected {len(current_terms)} terms")
        
        # Refit final model
        term_stats, final_model, sse = self._fit_and_get_significance(
            design, responses, current_terms
        )
        
        # Calculate final metrics
        n_obs = len(responses)
        n_params = len(current_terms)
        sst = np.sum((responses - np.mean(responses)) ** 2)
        r2 = 1.0 - sse / sst if sst > 0 else 0.0
        adj_r2 = ModelCriteriaCalculator.calculate_adjusted_r2(r2, n_obs, n_params)
        aic = ModelCriteriaCalculator.calculate_aic(n_obs, n_params, sse)
        bic = ModelCriteriaCalculator.calculate_bic(n_obs, n_params, sse)
        
        # Extract info
        coefficients = {term: stats_dict['coefficient'] 
                       for term, stats_dict in term_stats.items()}
        p_values = {term: stats_dict['p_value'] 
                   for term, stats_dict in term_stats.items()}
        vif_values = {term: stats_dict['vif'] 
                     for term, stats_dict in term_stats.items()}
        
        if self.verbose:
            print(f"\nFinal Statistics:")
            print(f"  R² = {r2:.4f}")
            print(f"  Adjusted R² = {adj_r2:.4f}")
            print(f"  AIC = {aic:.2f}")
            print(f"  BIC = {bic:.2f}")
            
            print(f"\nSelected Terms:")
            for term in current_terms:
                coef = coefficients[term]
                p_val = p_values[term]
                vif = vif_values[term]
                order = len(term.split('*'))
                order_name = {1: "Main", 2: "2-way", 3: "3-way", 4: "4-way"}.get(order)
                print(f"  {term:<20} coef={coef:+8.3f} p={p_val:.4f} VIF={vif:6.2f} [{order_name}]")
        
        return ModelSelectionResult(
            selected_terms=current_terms,
            coefficients=coefficients,
            model=final_model,
            r_squared=r2,
            adjusted_r_squared=adj_r2,
            aic=aic,
            bic=bic,
            cv_score=r2,  # Could add CV if needed
            vif_values=vif_values,
            p_values=p_values,
            iteration_history=[]  # Could add if needed
        )


# Example usage
if __name__ == "__main__":
    print("JMP Full Model Screening")
    print("="*70)
    
    # Generate test data
    def test_function(x):
        return 3*x[0] - 4*x[2] + 5*x[4] + 2*x[0]*x[2] - 3*x[1]*x[3]
    
    # Generate D-optimal design for quartic model
    print("\nGenerating D-optimal design for quartic model...")
    design = DOptimalMixtureDesign.generate_quartic_mixture_design(
        n_components=5,
        n_runs=50
    )
    
    responses = np.array([test_function(point) for point in design])
    
    # Run full model screening
    screener = JMPFullModelScreening(
        n_components=5,
        max_order=4,
        alpha=0.01,
        p_threshold=0.05,
        verbose=True
    )
    
    result = screener.screen_full_model(design, responses)
    
    print("\n" + "="*70)
    print("Screening complete!")
    print(f"Started with {len(screener._generate_all_terms())} terms")
    print(f"Ended with {len(result.selected_terms)} terms")
    print(f"Removed {len(screener._generate_all_terms()) - len(result.selected_terms)} terms")
