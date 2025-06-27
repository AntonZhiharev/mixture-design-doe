"""
Response Analysis for Design of Experiments
Comprehensive guide for analyzing experimental data after collecting responses
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import scipy.stats as stats
from scipy.optimize import minimize
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class DOEResponseAnalysis:
    """
    Class for analyzing experimental responses from DOE
    """
    
    def __init__(self, design_matrix: np.ndarray, responses: np.ndarray, 
                 factor_names: list = None, response_name: str = "Response"):
        """
        Initialize response analysis
        
        Parameters:
        design_matrix: Experimental design matrix (n_runs x n_factors)
        responses: Experimental responses (n_runs,)
        factor_names: Names of factors
        response_name: Name of response variable
        """
        self.X = design_matrix
        self.y = responses
        self.n_runs, self.n_factors = design_matrix.shape
        
        if factor_names is None:
            self.factor_names = [f'X{i+1}' for i in range(self.n_factors)]
        else:
            self.factor_names = factor_names
            
        self.response_name = response_name
        self.model = None
        self.model_type = None
        
    def fit_linear_model(self) -> dict:
        """
        Fit linear model: Y = β₀ + β₁X₁ + β₂X₂ + ... + ε
        """
        # Create design matrix with intercept
        X_design = np.column_stack([np.ones(self.n_runs), self.X])
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_design[:, 1:], self.y)  # Skip intercept column for sklearn
        self.model_type = "linear"
        
        # Calculate statistics
        y_pred = self.model.predict(self.X)
        r2 = r2_score(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        # Manual calculation for more detailed stats
        coeffs = np.linalg.lstsq(X_design, self.y, rcond=None)[0]
        residuals = self.y - X_design @ coeffs
        mse = np.sum(residuals**2) / (self.n_runs - self.n_factors - 1)
        
        # Standard errors
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(XtX_inv) * mse)
        
        # t-statistics and p-values
        t_stats = coeffs / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), self.n_runs - self.n_factors - 1))
        
        return {
            'coefficients': coeffs,
            'standard_errors': se,
            't_statistics': t_stats,
            'p_values': p_values,
            'r_squared': r2,
            'rmse': rmse,
            'residuals': residuals,
            'predictions': y_pred
        }
    
    def fit_quadratic_model(self) -> dict:
        """
        Fit quadratic model with interactions: Y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ + ε
        """
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(self.X)
        
        # Fit model
        self.model = LinearRegression()
        self.model.fit(X_poly, self.y)
        self.model_type = "quadratic"
        
        # Get feature names
        feature_names = poly.get_feature_names_out(self.factor_names)
        
        # Calculate statistics
        y_pred = self.model.predict(X_poly)
        r2 = r2_score(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        # Manual calculation for detailed stats
        coeffs = np.linalg.lstsq(X_poly, self.y, rcond=None)[0]
        residuals = self.y - X_poly @ coeffs
        dof = self.n_runs - len(coeffs)
        mse = np.sum(residuals**2) / dof if dof > 0 else np.inf
        
        # Standard errors (if possible)
        try:
            XtX_inv = np.linalg.inv(X_poly.T @ X_poly)
            se = np.sqrt(np.diag(XtX_inv) * mse)
            t_stats = coeffs / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        except:
            se = np.full_like(coeffs, np.nan)
            t_stats = np.full_like(coeffs, np.nan)
            p_values = np.full_like(coeffs, np.nan)
        
        return {
            'coefficients': coeffs,
            'feature_names': feature_names,
            'standard_errors': se,
            't_statistics': t_stats,
            'p_values': p_values,
            'r_squared': r2,
            'rmse': rmse,
            'residuals': residuals,
            'predictions': y_pred,
            'polynomial_features': X_poly
        }
    
    def anova_analysis(self, model_results: dict) -> pd.DataFrame:
        """
        Perform ANOVA analysis
        """
        if self.model_type == "linear":
            return self._linear_anova(model_results)
        elif self.model_type == "quadratic":
            return self._quadratic_anova(model_results)
    
    def _linear_anova(self, results: dict) -> pd.DataFrame:
        """ANOVA for linear model"""
        # Total sum of squares
        SS_total = np.sum((self.y - np.mean(self.y))**2)
        
        # Regression sum of squares
        y_pred = results['predictions']
        SS_reg = np.sum((y_pred - np.mean(self.y))**2)
        
        # Error sum of squares
        SS_error = np.sum(results['residuals']**2)
        
        # Degrees of freedom
        df_reg = self.n_factors
        df_error = self.n_runs - self.n_factors - 1
        df_total = self.n_runs - 1
        
        # Mean squares
        MS_reg = SS_reg / df_reg
        MS_error = SS_error / df_error
        
        # F-statistic
        F_stat = MS_reg / MS_error
        p_value = 1 - stats.f.cdf(F_stat, df_reg, df_error)
        
        anova_table = pd.DataFrame({
            'Source': ['Regression', 'Error', 'Total'],
            'SS': [SS_reg, SS_error, SS_total],
            'DF': [df_reg, df_error, df_total],
            'MS': [MS_reg, MS_error, np.nan],
            'F': [F_stat, np.nan, np.nan],
            'p-value': [p_value, np.nan, np.nan]
        })
        
        return anova_table
    
    def _quadratic_anova(self, results: dict) -> pd.DataFrame:
        """ANOVA for quadratic model"""
        # Similar to linear but more complex
        SS_total = np.sum((self.y - np.mean(self.y))**2)
        y_pred = results['predictions']
        SS_reg = np.sum((y_pred - np.mean(self.y))**2)
        SS_error = np.sum(results['residuals']**2)
        
        df_reg = len(results['coefficients']) - 1
        df_error = self.n_runs - len(results['coefficients'])
        df_total = self.n_runs - 1
        
        MS_reg = SS_reg / df_reg
        MS_error = SS_error / df_error if df_error > 0 else np.inf
        
        F_stat = MS_reg / MS_error if MS_error > 0 else np.inf
        p_value = 1 - stats.f.cdf(F_stat, df_reg, df_error) if df_error > 0 else 0
        
        anova_table = pd.DataFrame({
            'Source': ['Regression', 'Error', 'Total'],
            'SS': [SS_reg, SS_error, SS_total],
            'DF': [df_reg, df_error, df_total],
            'MS': [MS_reg, MS_error, np.nan],
            'F': [F_stat, np.nan, np.nan],
            'p-value': [p_value, np.nan, np.nan]
        })
        
        return anova_table
    
    def effects_analysis(self, model_results: dict) -> pd.DataFrame:
        """
        Calculate main effects and interactions
        """
        coeffs = model_results['coefficients']
        p_values = model_results['p_values']
        
        # Check if feature_names exists (quadratic model) or create terms for linear model
        if 'feature_names' in model_results:
            # Quadratic model
            feature_names = model_results['feature_names']
            effects_df = pd.DataFrame({
                'Term': feature_names,
                'Coefficient': coeffs,
                'p-value': p_values,
                'Significant': p_values < 0.05
            })
        else:
            # Linear model
            effects_df = pd.DataFrame({
                'Term': ['Intercept'] + self.factor_names,
                'Coefficient': coeffs,
                'p-value': p_values,
                'Significant': p_values < 0.05
            })
        
        return effects_df.sort_values('p-value')
    
    def predict_response(self, new_conditions: np.ndarray) -> np.ndarray:
        """
        Predict response for new factor combinations
        """
        if self.model is None:
            raise ValueError("No model fitted. Call fit_linear_model() or fit_quadratic_model() first.")
        
        if self.model_type == "linear":
            return self.model.predict(new_conditions)
        elif self.model_type == "quadratic":
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(new_conditions)
            return self.model.predict(X_poly)
    
    def optimize_response(self, bounds: list, maximize: bool = True) -> dict:
        """
        Find optimal factor settings
        
        Parameters:
        bounds: List of (min, max) tuples for each factor
        maximize: True to maximize, False to minimize
        """
        if self.model is None:
            raise ValueError("No model fitted.")
        
        def objective(x):
            pred = self.predict_response(x.reshape(1, -1))[0]
            return -pred if maximize else pred
        
        # Try multiple starting points
        best_result = None
        best_value = np.inf if not maximize else -np.inf
        
        for _ in range(10):  # Multiple random starts
            x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            
            result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                obj_value = -result.fun if maximize else result.fun
                if (maximize and obj_value > best_value) or (not maximize and obj_value < best_value):
                    best_value = obj_value
                    best_result = result
        
        if best_result is None:
            return {'success': False, 'message': 'Optimization failed'}
        
        optimal_x = best_result.x
        optimal_y = self.predict_response(optimal_x.reshape(1, -1))[0]
        
        return {
            'success': True,
            'optimal_factors': dict(zip(self.factor_names, optimal_x)),
            'optimal_response': optimal_y,
            'optimization_result': best_result
        }
    
    def plot_diagnostics(self, model_results: dict):
        """
        Create diagnostic plots
        """
        residuals = model_results['residuals']
        predictions = model_results['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Fitted
        axes[0, 0].scatter(predictions, residuals, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # Scale-Location plot
        sqrt_abs_residuals = np.sqrt(np.abs(residuals))
        axes[1, 0].scatter(predictions, sqrt_abs_residuals, alpha=0.7)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Residuals|')
        axes[1, 0].set_title('Scale-Location Plot')
        
        # Residuals histogram
        axes[1, 1].hist(residuals, bins=10, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Residuals Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_response_surface(self, factor1_idx: int = 0, factor2_idx: int = 1, 
                            fixed_values: dict = None, n_points: int = 50):
        """
        Plot 2D response surface for two factors
        """
        if self.n_factors < 2:
            print("Need at least 2 factors for response surface plot")
            return
        
        if self.model is None:
            print("No model fitted.")
            return
        
        # Create grid
        factor1_range = np.linspace(np.min(self.X[:, factor1_idx]), 
                                  np.max(self.X[:, factor1_idx]), n_points)
        factor2_range = np.linspace(np.min(self.X[:, factor2_idx]), 
                                  np.max(self.X[:, factor2_idx]), n_points)
        
        F1, F2 = np.meshgrid(factor1_range, factor2_range)
        
        # Create prediction matrix
        grid_points = []
        for i in range(n_points):
            for j in range(n_points):
                point = np.zeros(self.n_factors)
                point[factor1_idx] = F1[i, j]
                point[factor2_idx] = F2[i, j]
                
                # Set fixed values for other factors
                if fixed_values:
                    for factor_name, value in fixed_values.items():
                        if factor_name in self.factor_names:
                            idx = self.factor_names.index(factor_name)
                            point[idx] = value
                else:
                    # Use mean values for other factors
                    for k in range(self.n_factors):
                        if k not in [factor1_idx, factor2_idx]:
                            point[k] = np.mean(self.X[:, k])
                
                grid_points.append(point)
        
        grid_points = np.array(grid_points)
        predictions = self.predict_response(grid_points)
        Z = predictions.reshape(n_points, n_points)
        
        # Create plot
        fig = go.Figure(data=[go.Surface(z=Z, x=F1, y=F2, colorscale='Viridis')])
        
        fig.update_layout(
            title=f'Response Surface: {self.response_name}',
            scene=dict(
                xaxis_title=self.factor_names[factor1_idx],
                yaxis_title=self.factor_names[factor2_idx],
                zaxis_title=self.response_name
            ),
            height=600
        )
        
        # Add experimental points
        fig.add_trace(go.Scatter3d(
            x=self.X[:, factor1_idx],
            y=self.X[:, factor2_idx],
            z=self.y,
            mode='markers',
            marker=dict(size=8, color='red'),
            name='Experimental Data'
        ))
        
        fig.show()


class MixtureResponseAnalysis:
    """
    Specialized analysis for mixture experiments
    """
    
    def __init__(self, mixture_design: np.ndarray, responses: np.ndarray, 
                 component_names: list = None, response_name: str = "Response"):
        """
        Initialize mixture response analysis
        """
        self.X = mixture_design
        self.y = responses
        self.n_runs, self.n_components = mixture_design.shape
        
        if component_names is None:
            self.component_names = [f'Component_{i+1}' for i in range(self.n_components)]
        else:
            self.component_names = component_names
            
        self.response_name = response_name
        self.model = None
        
    def fit_scheffe_linear(self) -> dict:
        """
        Fit Scheffé linear mixture model: Y = Σβᵢxᵢ (no intercept)
        """
        # Fit model without intercept
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(self.X, self.y)
        
        # Calculate statistics
        y_pred = self.model.predict(self.X)
        r2 = r2_score(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        coeffs = self.model.coef_
        residuals = self.y - y_pred
        
        return {
            'coefficients': coeffs,
            'component_names': self.component_names,
            'r_squared': r2,
            'rmse': rmse,
            'residuals': residuals,
            'predictions': y_pred,
            'model_type': 'linear'
        }
    
    def fit_scheffe_quadratic(self) -> dict:
        """
        Fit Scheffé quadratic mixture model: Y = Σβᵢxᵢ + ΣΣβᵢⱼxᵢxⱼ
        """
        # Create design matrix for quadratic mixture model
        X_design = []
        term_names = []
        
        # Linear terms
        for i in range(self.n_components):
            X_design.append(self.X[:, i])
            term_names.append(self.component_names[i])
        
        # Interaction terms
        for i in range(self.n_components):
            for j in range(i + 1, self.n_components):
                X_design.append(self.X[:, i] * self.X[:, j])
                term_names.append(f'{self.component_names[i]}*{self.component_names[j]}')
        
        X_design = np.column_stack(X_design)
        
        # Fit model without intercept
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X_design, self.y)
        
        # Calculate statistics
        y_pred = self.model.predict(X_design)
        r2 = r2_score(self.y, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y, y_pred))
        
        coeffs = self.model.coef_
        residuals = self.y - y_pred
        
        return {
            'coefficients': coeffs,
            'term_names': term_names,
            'design_matrix': X_design,
            'r_squared': r2,
            'rmse': rmse,
            'residuals': residuals,
            'predictions': y_pred,
            'model_type': 'quadratic'
        }
    
    def plot_mixture_effects(self, model_results: dict):
        """
        Plot mixture component effects
        """
        coeffs = model_results['coefficients']
        
        if model_results['model_type'] == 'linear':
            names = model_results['component_names']
        else:
            names = model_results['term_names']
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, coeffs)
        plt.xlabel('Components/Interactions')
        plt.ylabel('Coefficient')
        plt.title('Mixture Model Coefficients')
        plt.xticks(rotation=45)
        
        # Color bars by magnitude
        colors = ['red' if c < 0 else 'blue' for c in coeffs]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.show()
    
    def predict_mixture_response(self, new_mixtures: np.ndarray) -> np.ndarray:
        """
        Predict response for new mixture compositions
        """
        if self.model is None:
            raise ValueError("No model fitted.")
        
        # Verify mixtures sum to 1
        sums = np.sum(new_mixtures, axis=1)
        if not np.allclose(sums, 1.0):
            print("Warning: Some mixtures don't sum to 1.0")
        
        return self.model.predict(new_mixtures)


# Example usage functions
def analyze_regular_doe_example():
    """
    Example: Analyzing regular DOE data
    """
    print("=== Regular DOE Analysis Example ===")
    
    # Simulate experimental data
    np.random.seed(42)
    
    # 3-factor experiment (Temperature, Pressure, pH)
    design = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0]
    ])
    
    # Simulate responses (yield = f(temp, pressure, pH) + noise)
    true_effects = [75, 5, 3, -2, 1.5, -1, 0.5, 2]  # intercept + linear + interactions
    X_full = np.column_stack([
        np.ones(len(design)),  # intercept
        design,  # linear terms
        design[:, 0] * design[:, 1],  # temp*pressure
        design[:, 0] * design[:, 2],  # temp*pH
        design[:, 1] * design[:, 2],  # pressure*pH
        design[:, 0] * design[:, 1] * design[:, 2]  # three-way interaction
    ])
    
    responses = X_full @ true_effects + np.random.normal(0, 2, len(design))
    
    # Create analyzer
    analyzer = DOEResponseAnalysis(
        design, responses, 
        factor_names=['Temperature', 'Pressure', 'pH'],
        response_name='Yield (%)'
    )
    
    # Fit models
    print("\n1. Linear Model Results:")
    linear_results = analyzer.fit_linear_model()
    print(f"R² = {linear_results['r_squared']:.4f}")
    print(f"RMSE = {linear_results['rmse']:.4f}")
    
    print("\n2. Quadratic Model Results:")
    quad_results = analyzer.fit_quadratic_model()
    print(f"R² = {quad_results['r_squared']:.4f}")
    print(f"RMSE = {quad_results['rmse']:.4f}")
    
    # ANOVA
    print("\n3. ANOVA Table (Linear Model):")
    anova = analyzer.anova_analysis(linear_results)
    print(anova.round(4))
    
    # Effects analysis
    print("\n4. Effects Analysis (Linear Model):")
    effects = analyzer.effects_analysis(linear_results)
    print(effects.round(4))
    
    # Optimization
    print("\n5. Response Optimization:")
    bounds = [(-1, 1), (-1, 1), (-1, 1)]
    opt_result = analyzer.optimize_response(bounds, maximize=True)
    if opt_result['success']:
        print(f"Optimal conditions:")
        for factor, value in opt_result['optimal_factors'].items():
            print(f"  {factor}: {value:.3f}")
        print(f"Predicted optimal response: {opt_result['optimal_response']:.2f}")
    
    # Predictions
    print("\n6. Predictions for new conditions:")
    new_conditions = np.array([[0.5, -0.5, 0.2], [-0.3, 0.8, -0.1]])
    predictions = analyzer.predict_response(new_conditions)
    for i, pred in enumerate(predictions):
        print(f"Condition {i+1}: Predicted yield = {pred:.2f}%")
    
    return analyzer, linear_results, quad_results


def analyze_mixture_example():
    """
    Example: Analyzing mixture experiment data
    """
    print("\n=== Mixture Design Analysis Example ===")
    
    # Simulate mixture experiment (3 components: A, B, C)
    np.random.seed(42)
    
    mixture_design = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.33, 0.33, 0.34],
        [0.6, 0.2, 0.2],
        [0.2, 0.6, 0.2],
        [0.2, 0.2, 0.6]
    ])
    
    # Simulate responses (strength = linear blending + synergy between A&B)
    component_effects = [50, 80, 30]  # Pure component responses
    synergy_AB = 25  # Synergistic effect between A and B
    
    responses = (mixture_design @ component_effects + 
                synergy_AB * mixture_design[:, 0] * mixture_design[:, 1] +
                np.random.normal(0, 2, len(mixture_design)))
    
    # Create analyzer
    mixture_analyzer = MixtureResponseAnalysis(
        mixture_design, responses,
        component_names=['Component_A', 'Component_B', 'Component_C'],
        response_name='Strength'
    )
    
    # Fit models
    print("\n1. Scheffé Linear Model:")
    linear_results = mixture_analyzer.fit_scheffe_linear()
    print(f"R² = {linear_results['r_squared']:.4f}")
    print("Component coefficients:")
    for name, coeff in zip(linear_results['component_names'], linear_results['coefficients']):
        print(f"  {name}: {coeff:.2f}")
    
    print("\n2. Scheffé Quadratic Model:")
    quad_results = mixture_analyzer.fit_scheffe_quadratic()
    print(f"R² = {quad_results['r_squared']:.4f}")
    print("Model terms:")
    for name, coeff in zip(quad_results['term_names'], quad_results['coefficients']):
        print(f"  {name}: {coeff:.2f}")
    
    # Predictions
    print("\n3. Mixture Predictions:")
    new_mixtures = np.array([
        [0.4, 0.4, 0.2],
        [0.7, 0.1, 0.2],
        [0.1, 0.7, 0.2]
    ])
    
    mixture_analyzer.model = mixture_analyzer.model  # Use the last fitted model
    predictions = mixture_analyzer.predict_mixture_response(new_mixtures)
    
    for i, (mixture, pred) in enumerate(zip(new_mixtures, predictions)):
        print(f"Mixture {i+1}: A={mixture[0]:.1f}, B={mixture[1]:.1f}, C={mixture[2]:.1f} → Strength = {pred:.1f}")
    
    return mixture_analyzer, linear_results, quad_results


if __name__ == "__main__":
    # Run examples
    print("DOE Response Analysis Examples")
    print("=" * 50)
    
    # Regular DOE analysis
    doe_analyzer, doe_linear, doe_quad = analyze_regular_doe_example()
    
    # Mixture analysis  
    mix_analyzer, mix_linear, mix_quad = analyze_mixture_example()
    
    print("\n" + "=" * 50)
    print("Analysis complete! Key takeaways:")
    print("✅ Always check R² and residual plots")
    print("✅ Use ANOVA to test model significance") 
    print("✅ Identify significant factors/interactions")
    print("✅ Optimize for desired response")
    print("✅ Validate predictions with new experiments")
