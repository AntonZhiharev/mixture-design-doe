"""
Hierarchical Screening for Higher-Order Mixture Models

This module implements hierarchical screening methodology for estimating
higher-order interactions with limited experimental data, using:
- Lenth's Pseudo Standard Error (PSE)
- Heredity principles
- Alias detection
- Staged model building
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


@dataclass
class ScreeningStageResult:
    """Results from a single screening stage"""
    stage_name: str
    order: int
    terms_tested: List[str]
    terms_significant: List[str]
    coefficients: Dict[str, float]
    pseudo_t_ratios: Dict[str, float]
    pseudo_p_values: Dict[str, float]
    pse: float
    margin_of_error: float
    r_squared: float
    n_points: int


class LenthPSECalculator:
    """
    Calculate Lenth's Pseudo Standard Error for effect screening.
    
    Lenth's method estimates the standard error of effects in screening
    designs where classical statistical tests are not applicable due to
    limited degrees of freedom.
    """
    
    @staticmethod
    def calculate_pse(effects: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        """
        Calculate Lenth's PSE and related statistics.
        
        Parameters:
        -----------
        effects : np.ndarray
            Array of effect estimates (coefficients)
        alpha : float
            Significance level (default 0.05)
            
        Returns:
        --------
        Dict with PSE, ME, and SME
        """
        # Step 1: Calculate absolute effects
        abs_effects = np.abs(effects[effects != 0])  # Exclude zero effects
        
        if len(abs_effects) == 0:
            return {'pse': 0.0, 'me': 0.0, 'sme': 0.0}
        
        # Step 2: Calculate s₀
        s0 = 1.5 * np.median(abs_effects)
        
        # Step 3: Select effects less than 2.5 × s₀
        selected = abs_effects[abs_effects < 2.5 * s0]
        
        # Step 4: Calculate PSE
        if len(selected) == 0:
            pse = s0
        else:
            pse = 1.5 * np.median(selected)
        
        # Step 5: Calculate margin of error (ME)
        # For Lenth's method, use fixed critical values
        # ME = 2.0 * PSE (approximately t=2 for individual effects)
        me = 2.0 * pse
        
        # Step 6: Calculate simultaneous margin of error (SME)
        # SME = 3.0 * PSE (more conservative for multiple comparisons)
        # This is Lenth's recommendation for screening designs
        sme = 3.0 * pse
        
        return {
            'pse': pse,
            'me': me,
            'sme': sme,
            's0': s0,
            'n_effects': len(effects)
        }
    
    @staticmethod
    def identify_significant_effects(effects: np.ndarray, 
                                    effect_names: List[str],
                                    alpha: float = 0.05,
                                    use_simultaneous: bool = True) -> Tuple[List[str], Dict]:
        """
        Identify significant effects using Lenth's method.
        
        Parameters:
        -----------
        effects : np.ndarray
            Effect estimates
        effect_names : List[str]
            Names of effects
        alpha : float
            Significance level
        use_simultaneous : bool
            Use simultaneous margin of error (more conservative)
            
        Returns:
        --------
        Tuple of (significant_effect_names, statistics_dict)
        """
        pse_stats = LenthPSECalculator.calculate_pse(effects, alpha)
        
        # Choose threshold
        threshold = pse_stats['sme'] if use_simultaneous else pse_stats['me']
        
        # Calculate pseudo t-ratios and p-values
        pseudo_t_ratios = {}
        pseudo_p_values = {}
        significant_effects = []
        
        for i, (effect, name) in enumerate(zip(effects, effect_names)):
            if pse_stats['pse'] > 0:
                t_ratio = abs(effect) / pse_stats['pse']
                # Pseudo p-value (two-tailed)
                p_value = 2 * (1 - stats.t.cdf(t_ratio, df=len(effects)//3))
            else:
                t_ratio = 0.0
                p_value = 1.0
            
            pseudo_t_ratios[name] = t_ratio
            pseudo_p_values[name] = p_value
            
            # Check significance
            if abs(effect) > threshold:
                significant_effects.append(name)
        
        return significant_effects, {
            'pseudo_t_ratios': pseudo_t_ratios,
            'pseudo_p_values': pseudo_p_values,
            'pse_stats': pse_stats,
            'threshold': threshold
        }


class HeredityChecker:
    """
    Check heredity constraints for interaction terms.
    
    Heredity principles:
    - Strong: All parent terms must be significant
    - Weak: At least one parent term must be significant
    """
    
    @staticmethod
    def parse_interaction_term(term: str) -> List[str]:
        """Parse interaction term into component names"""
        return term.split('*')
    
    @staticmethod
    def check_strong_heredity(interaction: str, 
                            significant_terms: Set[str]) -> bool:
        """
        Check strong heredity: ALL parent terms must be significant.
        
        Example: x1*x2*x3 requires x1, x2, AND x3 to be significant
        """
        components = HeredityChecker.parse_interaction_term(interaction)
        
        # All components must be in significant_terms
        return all(comp in significant_terms for comp in components)
    
    @staticmethod
    def check_weak_heredity(interaction: str,
                          significant_terms: Set[str]) -> bool:
        """
        Check weak heredity: AT LEAST ONE parent term must be significant.
        
        Example: x1*x2*x3 requires x1 OR x2 OR x3 to be significant
        """
        components = HeredityChecker.parse_interaction_term(interaction)
        
        # At least one component must be in significant_terms
        return any(comp in significant_terms for comp in components)
    
    @staticmethod
    def get_eligible_interactions(order: int,
                                 component_names: List[str],
                                 significant_lower_order: Set[str],
                                 use_strong_heredity: bool = False) -> List[str]:
        """
        Get list of interactions that satisfy heredity constraints.
        
        Parameters:
        -----------
        order : int
            Order of interactions (2, 3, 4, 5)
        component_names : List[str]
            All component names
        significant_lower_order : Set[str]
            Significant terms from previous stages
        use_strong_heredity : bool
            Use strong heredity (default: weak)
            
        Returns:
        --------
        List of eligible interaction terms
        """
        from itertools import combinations
        
        eligible = []
        
        # Generate all possible interactions of this order
        for combo in combinations(component_names, order):
            interaction = '*'.join(combo)
            
            # Check heredity
            if use_strong_heredity:
                if HeredityChecker.check_strong_heredity(interaction, significant_lower_order):
                    eligible.append(interaction)
            else:
                if HeredityChecker.check_weak_heredity(interaction, significant_lower_order):
                    eligible.append(interaction)
        
        return eligible


class AliasDetector:
    """
    Detect aliased (confounded) terms in design matrix.
    
    Two terms are aliased if they are linearly dependent or
    highly correlated in the design space.
    """
    
    @staticmethod
    def check_aliases(design_matrix: np.ndarray,
                     term_names: List[str],
                     correlation_threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Check for aliased terms in design matrix.
        
        Parameters:
        -----------
        design_matrix : np.ndarray
            Design matrix (n_points × n_terms)
        term_names : List[str]
            Names of terms/columns
        correlation_threshold : float
            Correlation threshold for aliasing (default 0.95)
            
        Returns:
        --------
        List of tuples: (term1, term2, correlation)
        """
        if design_matrix.shape[1] != len(term_names):
            raise ValueError("Number of columns must match number of term names")
        
        aliases = []
        
        # Calculate correlation matrix
        correlations = np.corrcoef(design_matrix.T)
        
        # Find highly correlated pairs
        for i in range(len(term_names)):
            for j in range(i+1, len(term_names)):
                corr = correlations[i, j]
                if abs(corr) >= correlation_threshold:
                    aliases.append((term_names[i], term_names[j], corr))
        
        return aliases
    
    @staticmethod
    def check_linear_dependence(design_matrix: np.ndarray,
                               term_names: List[str],
                               tolerance: float = 1e-10) -> List[str]:
        """
        Check for linearly dependent columns using SVD.
        
        Returns:
        --------
        List of term names that are linearly dependent
        """
        # Compute SVD
        U, s, Vt = np.linalg.svd(design_matrix, full_matrices=False)
        
        # Find small singular values
        dependent_indices = np.where(s < tolerance)[0]
        
        if len(dependent_indices) == 0:
            return []
        
        # Map back to term names
        dependent_terms = [term_names[i] for i in dependent_indices if i < len(term_names)]
        
        return dependent_terms


class HierarchicalScreening:
    """
    Main class for hierarchical screening of mixture models.
    
    Implements stage-by-stage screening:
    1. Main effects (linear)
    2. Two-way interactions
    3. Three-way interactions
    4. Four-way interactions
    5. Five-way interactions (if applicable)
    """
    
    def __init__(self, 
                 n_components: int,
                 component_names: Optional[List[str]] = None,
                 alpha: float = 0.05,
                 use_strong_heredity: bool = False,
                 max_order: int = 4):
        """
        Initialize hierarchical screening.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        component_names : List[str], optional
            Names of components (default: x1, x2, ...)
        alpha : float
            Significance level for Lenth's method
        use_strong_heredity : bool
            Use strong heredity constraints
        max_order : int
            Maximum interaction order to test (2-5)
        """
        self.n_components = n_components
        self.component_names = component_names or [f"x{i+1}" for i in range(n_components)]
        self.alpha = alpha
        self.use_strong_heredity = use_strong_heredity
        self.max_order = max_order
        
        # Results storage
        self.stage_results: List[ScreeningStageResult] = []
        self.significant_terms: Set[str] = set()
        self.all_coefficients: Dict[str, float] = {}
        
    def _build_design_matrix(self, 
                           design: np.ndarray,
                           terms: List[str]) -> np.ndarray:
        """
        Build design matrix for specified terms.
        
        Parameters:
        -----------
        design : np.ndarray
            Experimental design (n_points × n_components)
        terms : List[str]
            List of term names to include
            
        Returns:
        --------
        Design matrix with columns for each term
        """
        n_points = design.shape[0]
        columns = []
        
        for term in terms:
            if '*' in term:
                # Interaction term
                components = term.split('*')
                indices = [self.component_names.index(comp) for comp in components]
                
                # Multiply components
                column = np.ones(n_points)
                for idx in indices:
                    column *= design[:, idx]
                columns.append(column)
            else:
                # Main effect
                idx = self.component_names.index(term)
                columns.append(design[:, idx])
        
        return np.column_stack(columns)
    
    def screen_stage(self,
                    design: np.ndarray,
                    responses: np.ndarray,
                    stage_name: str,
                    terms_to_test: List[str]) -> ScreeningStageResult:
        """
        Screen a single stage.
        
        Parameters:
        -----------
        design : np.ndarray
            Experimental design
        responses : np.ndarray
            Response values
        stage_name : str
            Name of this stage
        terms_to_test : List[str]
            Terms to test at this stage
            
        Returns:
        --------
        ScreeningStageResult
        """
        if len(terms_to_test) == 0:
            return ScreeningStageResult(
                stage_name=stage_name,
                order=0,
                terms_tested=[],
                terms_significant=[],
                coefficients={},
                pseudo_t_ratios={},
                pseudo_p_values={},
                pse=0.0,
                margin_of_error=0.0,
                r_squared=0.0,
                n_points=len(design)
            )
        
        print(f"\n{'='*70}")
        print(f"SCREENING STAGE: {stage_name}")
        print(f"{'='*70}")
        print(f"Terms to test: {len(terms_to_test)}")
        print(f"Available points: {len(design)}")
        
        # Build design matrix
        X = self._build_design_matrix(design, terms_to_test)
        
        # Check for aliases
        aliases = AliasDetector.check_aliases(X, terms_to_test)
        if aliases:
            print(f"\n⚠️  WARNING: Found {len(aliases)} aliased term pairs:")
            for term1, term2, corr in aliases[:5]:  # Show first 5
                print(f"   {term1} ↔ {term2} (r = {corr:.3f})")
        
        # Fit model
        model = LinearRegression(fit_intercept=False)
        model.fit(X, responses)
        
        # Get predictions and R²
        predictions = model.predict(X)
        r2 = r2_score(responses, predictions)
        
        # Get coefficients
        coefficients = dict(zip(terms_to_test, model.coef_))
        
        # Apply Lenth's method
        significant_terms, stats_dict = LenthPSECalculator.identify_significant_effects(
            model.coef_,
            terms_to_test,
            alpha=self.alpha,
            use_simultaneous=True
        )
        
        # Create result
        result = ScreeningStageResult(
            stage_name=stage_name,
            order=len(terms_to_test[0].split('*')) if terms_to_test else 1,
            terms_tested=terms_to_test,
            terms_significant=significant_terms,
            coefficients=coefficients,
            pseudo_t_ratios=stats_dict['pseudo_t_ratios'],
            pseudo_p_values=stats_dict['pseudo_p_values'],
            pse=stats_dict['pse_stats']['pse'],
            margin_of_error=stats_dict['threshold'],
            r_squared=r2,
            n_points=len(design)
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  R² = {r2:.4f}")
        print(f"  PSE = {result.pse:.4f}")
        print(f"  Threshold (SME) = {result.margin_of_error:.4f}")
        print(f"  Significant terms: {len(significant_terms)}/{len(terms_to_test)}")
        
        if significant_terms:
            print(f"\n  Significant effects:")
            for term in significant_terms:
                coef = coefficients[term]
                t_ratio = stats_dict['pseudo_t_ratios'][term]
                p_value = stats_dict['pseudo_p_values'][term]
                print(f"    {term:<20} coef={coef:+8.3f}  t={t_ratio:6.2f}  p={p_value:.4f}")
        
        return result
    
    def run_hierarchical_screening(self,
                                  design: np.ndarray,
                                  responses: np.ndarray) -> Dict:
        """
        Run complete hierarchical screening process.
        
        Parameters:
        -----------
        design : np.ndarray
            Experimental design (n_points × n_components)
        responses : np.ndarray
            Response values
            
        Returns:
        --------
        Dict with screening results
        """
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL SCREENING FOR HIGHER-ORDER INTERACTIONS")
        print(f"{'='*70}")
        print(f"Components: {self.n_components}")
        print(f"Design points: {len(design)}")
        print(f"Max order: {self.max_order}")
        print(f"Heredity: {'Strong' if self.use_strong_heredity else 'Weak'}")
        print(f"Significance level: {self.alpha}")
        
        self.significant_terms = set()
        self.all_coefficients = {}
        self.stage_results = []
        
        # Stage 1: Main Effects
        main_effects = self.component_names.copy()
        stage1 = self.screen_stage(design, responses, "Main Effects (Linear)", main_effects)
        self.stage_results.append(stage1)
        self.significant_terms.update(stage1.terms_significant)
        self.all_coefficients.update(stage1.coefficients)
        
        if len(stage1.terms_significant) == 0:
            print("\n⚠️  No significant main effects found. Stopping screening.")
            return self._compile_results()
        
        # Stage 2: Two-Way Interactions
        if self.max_order >= 2:
            two_way = HeredityChecker.get_eligible_interactions(
                2, self.component_names, self.significant_terms, self.use_strong_heredity
            )
            if two_way:
                stage2 = self.screen_stage(design, responses, "Two-Way Interactions", two_way)
                self.stage_results.append(stage2)
                self.significant_terms.update(stage2.terms_significant)
                self.all_coefficients.update(stage2.coefficients)
        
        # Stage 3: Three-Way Interactions
        if self.max_order >= 3:
            three_way = HeredityChecker.get_eligible_interactions(
                3, self.component_names, self.significant_terms, self.use_strong_heredity
            )
            if three_way:
                stage3 = self.screen_stage(design, responses, "Three-Way Interactions", three_way)
                self.stage_results.append(stage3)
                self.significant_terms.update(stage3.terms_significant)
                self.all_coefficients.update(stage3.coefficients)
        
        # Stage 4: Four-Way Interactions
        if self.max_order >= 4:
            four_way = HeredityChecker.get_eligible_interactions(
                4, self.component_names, self.significant_terms, self.use_strong_heredity
            )
            if four_way:
                stage4 = self.screen_stage(design, responses, "Four-Way Interactions", four_way)
                self.stage_results.append(stage4)
                self.significant_terms.update(stage4.terms_significant)
                self.all_coefficients.update(stage4.coefficients)
        
        # Stage 5: Five-Way Interactions
        if self.max_order >= 5:
            five_way = HeredityChecker.get_eligible_interactions(
                5, self.component_names, self.significant_terms, self.use_strong_heredity
            )
            if five_way:
                stage5 = self.screen_stage(design, responses, "Five-Way Interactions", five_way)
                self.stage_results.append(stage5)
                self.significant_terms.update(stage5.terms_significant)
                self.all_coefficients.update(stage5.coefficients)
        
        return self._compile_results()
    
    def _compile_results(self) -> Dict:
        """Compile final screening results"""
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL SCREENING COMPLETE")
        print(f"{'='*70}")
        print(f"Total stages: {len(self.stage_results)}")
        print(f"Total significant terms: {len(self.significant_terms)}")
        
        print(f"\nFinal significant terms:")
        for term in sorted(self.significant_terms, key=lambda x: (len(x.split('*')), x)):
            coef = self.all_coefficients.get(term, 0.0)
            order = len(term.split('*'))
            order_name = {1: "Main", 2: "2-way", 3: "3-way", 4: "4-way", 5: "5-way"}.get(order, f"{order}-way")
            print(f"  {term:<20} {coef:+8.3f}  [{order_name}]")
        
        return {
            'significant_terms': list(self.significant_terms),
            'coefficients': self.all_coefficients,
            'stage_results': self.stage_results,
            'n_stages': len(self.stage_results),
            'final_model_size': len(self.significant_terms)
        }
    
    def get_final_model_terms(self) -> List[str]:
        """Get list of terms for final model"""
        return sorted(self.significant_terms, key=lambda x: (len(x.split('*')), x))


# Example usage
if __name__ == "__main__":
    print("Hierarchical Screening Module")
    print("="*70)
    
    # Example: Test Lenth's PSE
    print("\n1. Testing Lenth's PSE Calculator")
    effects = np.array([0.5, -0.3, 2.5, 0.1, -2.8, 0.2, 3.1, -0.4])
    effect_names = [f"Effect_{i+1}" for i in range(len(effects))]
    
    significant, stats = LenthPSECalculator.identify_significant_effects(
        effects, effect_names, alpha=0.05
    )
    
    print(f"PSE = {stats['pse_stats']['pse']:.3f}")
    print(f"Threshold = {stats['threshold']:.3f}")
    print(f"Significant: {significant}")
    
    # Example: Test heredity checker
    print("\n2. Testing Heredity Checker")
    significant_lower = {'x1', 'x2', 'x4'}
    eligible = HeredityChecker.get_eligible_interactions(
        2, ['x1', 'x2', 'x3', 'x4', 'x5'], significant_lower, use_strong_heredity=False
    )
    print(f"Eligible 2-way interactions (weak heredity): {eligible}")
    
    # Example: Test alias detector
    print("\n3. Testing Alias Detector")
    design_test = np.random.rand(10, 5)
    term_names_test = ['A', 'B', 'C', 'A*B', 'A*C']
    X_test = np.column_stack([
        design_test[:, 0],
        design_test[:, 1],
        design_test[:, 2],
        design_test[:, 0] * design_test[:, 1],
        design_test[:, 0] * design_test[:, 2]
    ])
    
    aliases = AliasDetector.check_aliases(X_test, term_names_test, correlation_threshold=0.8)
    print(f"Aliases found: {len(aliases)}")
    
    print("\n" + "="*70)
    print("All tests completed successfully!")
