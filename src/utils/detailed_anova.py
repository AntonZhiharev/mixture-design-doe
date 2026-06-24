"""
Detailed ANOVA Analysis with Individual Factor Effects
======================================================
Calculates Sum of Squares, Mean Square, F-statistic, and P-Value
for each factor and interaction term
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def calculate_detailed_anova(X, y, factor_names, model_type="quadratic", analysis_type="Standard DOE", 
                            max_poly_degree=None, max_interaction_order=None):
    """
    Calculate detailed ANOVA with individual factor effects
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Design matrix with factor values
    y : array-like, shape (n_samples,)
        Response values
    factor_names : list
        Names of factors
    model_type : str
        Type of model ('linear', 'quadratic', 'cubic') - for backward compatibility
    analysis_type : str
        Type of analysis ('Standard DOE' or 'Mixture Design')
    max_poly_degree : int, optional
        Maximum degree for pure polynomial terms (X, X², X³, etc.)
    max_interaction_order : int, optional
        Maximum order for interaction terms (2=X₁X₂, 3=X₁X₂X₃, etc.)
    
    Returns:
    --------
    detailed_anova_df : DataFrame
        Detailed ANOVA table with SS, DF, MS, F, and P-value for each term
    model_summary : dict
        Overall model statistics
    """
    
    n_samples = len(y)
    n_factors = X.shape[1]
    
    # ── Mixture: enforce sum-to-1 constraint ──────────────────────────────────
    # For Scheffé models (fit_intercept=False), residuals are only guaranteed to
    # be centred around 0 when each mixture row sums to exactly 1.
    # Constrained designs exported from our tool may deviate slightly from 1,
    # causing a systematic downward shift in the residual plot.
    # We apply the standard L-pseudocomponent normalisation: x_i' = x_i / Σx_i
    mixture_was_normalized = False
    if analysis_type == "Mixture Design":
        row_sums = X.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            import warnings
            warnings.warn(
                f"⚠️ Mixture Design: component rows do not sum to 1 "
                f"(mean={np.mean(row_sums):.6f}, "
                f"min={np.min(row_sums):.6f}, max={np.max(row_sums):.6f}). "
                "Applying L-pseudocomponent normalisation (x_i' = x_i / Σx_i) "
                "so the Scheffé model intercept is correctly absorbed. "
                "This is the standard fix for constrained-mixture exports.",
                UserWarning,
                stacklevel=2,
            )
            X = X / row_sums[:, np.newaxis]
            mixture_was_normalized = True
    # ─────────────────────────────────────────────────────────────────────────

    # Calculate overall mean
    y_mean = np.mean(y)
    SS_total = np.sum((y - y_mean)**2)
    
    # Build model matrix based on type
    if analysis_type == "Mixture Design":
        # Check if we have custom degree/interaction parameters for mixture designs too
        if max_poly_degree is not None and max_interaction_order is not None:
            # Custom term builder for mixture designs
            from itertools import combinations
            design_matrix_list = []
            term_names = []
            
            # For mixtures: NO intercept (sum to 1 constraint)
            # 1. Linear terms (always include for mixtures)
            for i in range(n_factors):
                design_matrix_list.append(X[:, i])
                term_names.append(factor_names[i])
            
            # 2. Interaction terms based on max_interaction_order
            for order in range(2, max_interaction_order + 1):
                if order > n_factors:
                    break
                
                # Generate all combinations of 'order' factors
                for combo in combinations(range(n_factors), order):
                    # Calculate interaction term
                    interaction_term = np.ones(n_samples)
                    term_name_parts = []
                    
                    for factor_idx in combo:
                        interaction_term *= X[:, factor_idx]
                        term_name_parts.append(factor_names[factor_idx])
                    
                    design_matrix_list.append(interaction_term)
                    term_names.append('*'.join(term_name_parts))
            
            # Stack all terms
            design_matrix = np.column_stack(design_matrix_list)
            fit_intercept = False
            
        elif model_type == "linear":
            # Linear mixture: Y = Σβᵢxᵢ (no intercept)
            design_matrix = X.copy()
            term_names = factor_names.copy()
            fit_intercept = False
            
        else:  # quadratic
            # Quadratic mixture: Y = Σβᵢxᵢ + ΣΣβᵢⱼxᵢxⱼ
            design_matrix = []
            term_names = []
            
            # Linear terms
            for i in range(n_factors):
                design_matrix.append(X[:, i])
                term_names.append(factor_names[i])
            
            # Interaction terms
            for i in range(n_factors):
                for j in range(i + 1, n_factors):
                    design_matrix.append(X[:, i] * X[:, j])
                    term_names.append(f'{factor_names[i]}*{factor_names[j]}')
            
            design_matrix = np.column_stack(design_matrix)
            fit_intercept = False
        
    else:  # Standard DOE
        # Check if we have separate degree/interaction parameters
        if max_poly_degree is not None and max_interaction_order is not None:
            # Custom term builder - separate polynomial degree from interaction order
            design_matrix_list = []
            term_names = []
            
            # 1. Intercept
            design_matrix_list.append(np.ones(n_samples))
            term_names.append('1')
            
            # 2. Linear terms (always include)
            for i in range(n_factors):
                design_matrix_list.append(X[:, i])
                term_names.append(factor_names[i])
            
            # 3. Pure polynomial terms (X², X³, etc.)
            for degree in range(2, max_poly_degree + 1):
                for i in range(n_factors):
                    design_matrix_list.append(X[:, i] ** degree)
                    term_names.append(f"{factor_names[i]}^{degree}")
            
            # 4. Interaction terms
            from itertools import combinations
            
            for order in range(2, max_interaction_order + 1):
                if order > n_factors:
                    break  # Can't have more factors in interaction than we have factors
                
                # Generate all combinations of 'order' factors
                for combo in combinations(range(n_factors), order):
                    # Calculate interaction term
                    interaction_term = np.ones(n_samples)
                    term_name_parts = []
                    
                    for factor_idx in combo:
                        interaction_term *= X[:, factor_idx]
                        term_name_parts.append(factor_names[factor_idx])
                    
                    design_matrix_list.append(interaction_term)
                    term_names.append(' '.join(term_name_parts))
            
            # Stack all terms into design matrix
            design_matrix = np.column_stack(design_matrix_list)
            fit_intercept = False  # Already included
            
        elif model_type == "linear":
            # Linear model: Y = β₀ + Σβᵢxᵢ
            design_matrix = X.copy()
            term_names = ['Intercept'] + factor_names.copy()
            fit_intercept = True
            
        elif model_type == "quadratic":
            # Use PolynomialFeatures for quadratic standard DOE
            poly = PolynomialFeatures(degree=2, include_bias=True)
            design_matrix = poly.fit_transform(X)
            term_names = poly.get_feature_names_out(factor_names).tolist()
            fit_intercept = False  # Already included in polynomial features
            
        elif model_type == "cubic":
            # Use PolynomialFeatures for cubic standard DOE
            poly = PolynomialFeatures(degree=3, include_bias=True)
            design_matrix = poly.fit_transform(X)
            term_names = poly.get_feature_names_out(factor_names).tolist()
            fit_intercept = False  # Already included in polynomial features
            
        else:
            # Default to quadratic for any other model type
            poly = PolynomialFeatures(degree=2, include_bias=True)
            design_matrix = poly.fit_transform(X)
            term_names = poly.get_feature_names_out(factor_names).tolist()
            fit_intercept = False
    
    # Check for numerical issues
    n_terms = design_matrix.shape[1] if not fit_intercept else X.shape[1] + 1
    
    if n_samples <= n_terms:
        raise ValueError(f"Not enough observations ({n_samples}) for model with {n_terms} parameters. Need at least {n_terms + 1} observations.")
    
    # Check design matrix rank and remove redundant columns if needed
    if not fit_intercept:
        matrix_rank = np.linalg.matrix_rank(design_matrix)
        if matrix_rank < design_matrix.shape[1]:
            # Find redundant columns using QR decomposition
            Q, R, P = np.linalg.qr(design_matrix, mode='full', pivoting=True)
            
            # Identify independent columns (non-zero diagonal in R)
            tol = 1e-10
            independent_cols = np.abs(np.diag(R)) > tol
            independent_indices = P[independent_cols]
            independent_indices = np.sort(independent_indices)
            
            # Keep only independent columns
            redundant_terms = [term_names[i] for i in range(len(term_names)) if i not in independent_indices]
            
            # Categorize redundant terms for better reporting
            redundant_by_type = {
                'two_way': [],
                'three_way': [],
                'four_way': [],
                'five_way': [],
                'other': []
            }
            
            for term in redundant_terms:
                space_count = term.count(' ')
                if space_count == 1 and '^' not in term:
                    redundant_by_type['two_way'].append(term)
                elif space_count == 2 and '^' not in term:
                    redundant_by_type['three_way'].append(term)
                elif space_count == 3 and '^' not in term:
                    redundant_by_type['four_way'].append(term)
                elif space_count == 4 and '^' not in term:
                    redundant_by_type['five_way'].append(term)
                else:
                    redundant_by_type['other'].append(term)
            
            design_matrix = design_matrix[:, independent_indices]
            term_names = [term_names[i] for i in independent_indices]
            
            import warnings
            warning_msg = f"\n⚠️ Removed {len(redundant_terms)} redundant terms due to insufficient data variation:\n"
            warning_msg += f"  Design matrix rank: {matrix_rank} (expected {matrix_rank + len(redundant_terms)})\n"
            
            if redundant_by_type['two_way']:
                warning_msg += f"  - {len(redundant_by_type['two_way'])} two-way interactions\n"
            if redundant_by_type['three_way']:
                warning_msg += f"  - {len(redundant_by_type['three_way'])} three-way interactions: {redundant_by_type['three_way'][:5]}{'...' if len(redundant_by_type['three_way']) > 5 else ''}\n"
            if redundant_by_type['four_way']:
                warning_msg += f"  - {len(redundant_by_type['four_way'])} four-way interactions: {redundant_by_type['four_way'][:3]}{'...' if len(redundant_by_type['four_way']) > 3 else ''}\n"
            if redundant_by_type['five_way']:
                warning_msg += f"  - {len(redundant_by_type['five_way'])} five-way interactions: {redundant_by_type['five_way']}\n"
            if redundant_by_type['other']:
                warning_msg += f"  - {len(redundant_by_type['other'])} other terms\n"
            
            warning_msg += f"\n💡 To include higher-order interactions, you need:\n"
            warning_msg += f"   1. More observations (currently: {n_samples})\n"
            warning_msg += f"   2. Greater variation in your data\n"
            warning_msg += f"   3. Or reduce model complexity\n"
            
            warnings.warn(warning_msg)
    
    # Fit full model with error handling
    try:
        model = LinearRegression(fit_intercept=fit_intercept)
        
        if fit_intercept and analysis_type == "Standard DOE" and model_type == "linear":
            # For linear standard DOE with intercept
            model.fit(X, y)
            y_pred_full = model.predict(X)
            coefficients = np.concatenate([[model.intercept_], model.coef_])
        else:
            # For all other cases
            model.fit(design_matrix, y)
            y_pred_full = model.predict(design_matrix)
            coefficients = model.coef_
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Numerical error in model fitting: {str(e)}. This usually means the design matrix is singular or ill-conditioned. Try reducing model complexity or adding more observations.")
    
    SS_residual_full = np.sum((y - y_pred_full)**2)
    
    # Calculate sequential (Type I) sum of squares for each term
    anova_results = []
    
    # For each term, calculate its contribution
    if fit_intercept and analysis_type == "Standard DOE" and model_type == "linear":
        # Special handling for linear models with intercept
        # Start with intercept only
        SS_explained = 0
        
        for i, term_name in enumerate(term_names):
            if i == 0:  # Intercept
                # SS for intercept = n * (mean)^2
                SS_term = n_samples * (y_mean ** 2)
                df_term = 1
            else:
                # Add this term to the model
                X_subset = X[:, :i]
                model_subset = LinearRegression()
                model_subset.fit(X_subset, y)
                y_pred_subset = model_subset.predict(X_subset)
                SS_subset = np.sum((y - y_pred_subset)**2)
                
                # Previous model
                if i == 1:
                    SS_previous = SS_total
                else:
                    X_previous = X[:, :i-1]
                    model_previous = LinearRegression()
                    model_previous.fit(X_previous, y)
                    y_pred_previous = model_previous.predict(X_previous)
                    SS_previous = np.sum((y - y_pred_previous)**2)
                
                SS_term = SS_previous - SS_subset
                df_term = 1
            
            # Calculate MS, F, and p-value
            MS_term = SS_term / df_term
            df_error = n_samples - len(term_names)
            MS_error = SS_residual_full / df_error if df_error > 0 else 0
            
            if MS_error > 0:
                F_stat = MS_term / MS_error
                p_value = 1 - stats.f.cdf(F_stat, df_term, df_error)
            else:
                F_stat = 0
                p_value = 1.0
            
            # Calculate LogWorth = -log10(p_value)
            # LogWorth > 2 means p < 0.01 (highly significant)
            # LogWorth > 1.3 means p < 0.05 (significant)
            if p_value > 0 and p_value < 1:
                logworth = -np.log10(p_value)
            elif p_value == 0:
                logworth = np.inf  # Extremely small p-value
            else:
                logworth = 0  # p_value = 1 or invalid
            
            anova_results.append({
                'Source': term_name,
                'Sum_of_Squares': SS_term,
                'DF': df_term,
                'Mean_Square': MS_term,
                'F_statistic': F_stat,
                'P_Value': p_value,
                'LogWorth': logworth,
                'Coefficient': coefficients[i]
            })
    
    else:
        # For mixture models and polynomial models
        # Sequential sum of squares (Type I)
        for i, term_name in enumerate(term_names):
            # Fit model with terms up to and including current term
            X_current = design_matrix[:, :i+1]
            model_current = LinearRegression(fit_intercept=False)
            model_current.fit(X_current, y)
            y_pred_current = model_current.predict(X_current)
            SS_current = np.sum((y - y_pred_current)**2)
            
            # Fit model with terms up to but not including current term
            if i == 0:
                SS_previous = SS_total
            else:
                X_previous = design_matrix[:, :i]
                model_previous = LinearRegression(fit_intercept=False)
                model_previous.fit(X_previous, y)
                y_pred_previous = model_previous.predict(X_previous)
                SS_previous = np.sum((y - y_pred_previous)**2)
            
            # Sum of squares for this term
            SS_term = SS_previous - SS_current
            
            # Check for negative SS (indicates numerical issues or overfitting)
            if SS_term < -1e-10:  # Negative beyond numerical error
                import warnings
                warnings.warn(
                    f"\n⚠️  Negative Sum of Squares detected for {term_name}: {SS_term:.6f}\n"
                    f"    This indicates:\n"
                    f"    1. Model complexity exceeds data capacity\n"
                    f"    2. Numerical instability in sequential calculation\n"
                    f"    3. Potential overfitting ({n_samples} obs, {design_matrix.shape[1]} params)\n"
                    f"    Using absolute value, but statistical tests may be unreliable.\n"
                )
                SS_term = abs(SS_term)
            elif SS_term < 0:
                # Small negative value due to numerical precision
                SS_term = 0
            
            df_term = 1
            MS_term = SS_term / df_term
            
            # Calculate F-statistic and p-value
            n_params = design_matrix.shape[1]
            df_error = n_samples - n_params
            MS_error = SS_residual_full / df_error if df_error > 0 else 0
            
            # Check for near-zero MSE (overfitting indicator)
            if df_error > 0 and MS_error < 1e-10:
                import warnings
                warnings.warn(
                    f"\n⚠️  Near-zero residual detected (MSE={MS_error:.2e})\n"
                    f"    Model is severely overfitting:\n"
                    f"    - Parameters: {n_params}\n"
                    f"    - Observations: {n_samples}\n"
                    f"    - Ratio: {n_samples/n_params:.2f} (recommended: >5)\n"
                    f"    P-values and F-statistics are unreliable!\n"
                    f"    Recommendations:\n"
                    f"    1. Reduce model complexity (fewer parameters)\n"
                    f"    2. Collect more data\n"
                    f"    3. Use regularization techniques\n"
                )
                # Add small epsilon to prevent division issues
                MS_error = max(MS_error, 1e-10)
            
            if MS_error > 0 and df_error > 0:
                F_stat = MS_term / MS_error
                # Ensure F-stat is non-negative
                if F_stat < 0:
                    import warnings
                    warnings.warn(f"Negative F-statistic for {term_name}: {F_stat}. Setting to 0.")
                    F_stat = 0
                    p_value = 1.0
                else:
                    p_value = 1 - stats.f.cdf(F_stat, df_term, df_error)
            else:
                F_stat = 0
                p_value = 1.0
            
            # Calculate LogWorth = -log10(p_value)
            # LogWorth > 2 means p < 0.01 (highly significant)
            # LogWorth > 1.3 means p < 0.05 (significant)
            if p_value > 0 and p_value < 1:
                logworth = -np.log10(p_value)
            elif p_value == 0:
                logworth = np.inf  # Extremely small p-value
            else:
                logworth = 0  # p_value = 1 or invalid
            
            anova_results.append({
                'Source': term_name,
                'Sum_of_Squares': SS_term,
                'DF': df_term,
                'Mean_Square': MS_term,
                'F_statistic': F_stat,
                'P_Value': p_value,
                'LogWorth': logworth,
                'Coefficient': coefficients[i]
            })
    
    # Add error row
    n_params = len(term_names)
    df_error = n_samples - n_params
    MS_error = SS_residual_full / df_error if df_error > 0 else 0
    
    anova_results.append({
        'Source': 'Error (Residual)',
        'Sum_of_Squares': SS_residual_full,
        'DF': df_error,
        'Mean_Square': MS_error,
        'F_statistic': np.nan,
        'P_Value': np.nan,
        'LogWorth': np.nan,
        'Coefficient': np.nan
    })
    
    # Add total row
    anova_results.append({
        'Source': 'Total',
        'Sum_of_Squares': SS_total,
        'DF': n_samples - 1,
        'Mean_Square': np.nan,
        'F_statistic': np.nan,
        'P_Value': np.nan,
        'LogWorth': np.nan,
        'Coefficient': np.nan
    })
    
    # Create DataFrame
    detailed_anova_df = pd.DataFrame(anova_results)
    
    # Calculate Lenth's Pseudo Statistics for all terms
    # Extract effects (absolute coefficients) excluding error and total
    effects_for_pse = []
    for result in anova_results:
        if result['Source'] not in ['Error (Residual)', 'Total']:
            coef = result['Coefficient']
            if not pd.isna(coef):
                effects_for_pse.append(abs(coef))
    
    # Calculate PSE, ME, and SME using Lenth's method
    if len(effects_for_pse) >= 3:
        PSE, ME, SME = calculate_lenth_pse(effects_for_pse)
        df_lenth = max(1, len(effects_for_pse) // 3)
        
        # Add Pseudo t-ratio and Pseudo p-value columns
        pseudo_t_ratios = []
        pseudo_p_values = []
        
        from scipy import stats as scipy_stats
        
        for idx, row in detailed_anova_df.iterrows():
            if row['Source'] not in ['Error (Residual)', 'Total']:
                coef = row['Coefficient']
                if not pd.isna(coef) and PSE > 0:
                    # Pseudo t-ratio = |Effect| / PSE
                    pseudo_t = abs(coef) / PSE
                    pseudo_t_ratios.append(pseudo_t)
                    
                    # Pseudo p-value from pseudo t-ratio
                    if np.isfinite(pseudo_t):
                        pseudo_p = 2 * (1 - scipy_stats.t.cdf(pseudo_t, df_lenth))
                        pseudo_p_values.append(pseudo_p)
                    else:
                        pseudo_p_values.append(0.0 if abs(coef) > 1e-10 else 1.0)
                else:
                    pseudo_t_ratios.append(np.nan)
                    pseudo_p_values.append(np.nan)
            else:
                pseudo_t_ratios.append(np.nan)
                pseudo_p_values.append(np.nan)
        
        detailed_anova_df['Pseudo_t_Ratio'] = pseudo_t_ratios
        detailed_anova_df['Pseudo_P_Value'] = pseudo_p_values
        
        # Store PSE, ME, SME in model_summary for reference
        pse_info = {
            'PSE': PSE,
            'ME': ME,
            'SME': SME,
            'df_lenth': df_lenth
        }
    else:
        # Not enough terms for Lenth's method
        detailed_anova_df['Pseudo_t_Ratio'] = np.nan
        detailed_anova_df['Pseudo_P_Value'] = np.nan
        pse_info = None
    
    # Add Standard Error and Confidence Intervals for effect estimates
    # Standard Error: se(Effect) = sqrt(V(Effect)) = sqrt((1/n) * MS_error)
    # For factorial designs: se(Effect) = 2*sqrt(MS_error / n)
    # Confidence Interval: Effect ± t(α/2, df_error) * se(Effect)
    
    if df_error > 0 and MS_error > 0:
        from scipy import stats as scipy_stats
        
        # Calculate standard errors for each effect
        standard_errors = []
        ci_lower = []
        ci_upper = []
        t_statistics = []
        
        # t-value for 95% confidence interval
        t_crit = scipy_stats.t.ppf(0.975, df_error)  # Two-tailed, α=0.05
        
        for idx, row in detailed_anova_df.iterrows():
            if row['Source'] not in ['Error (Residual)', 'Total']:
                # Standard error for factorial design effects
                # se(Effect) = sqrt(4 * MS_error / n) for 2^k designs
                # More generally: se(Coefficient) = sqrt(MS_error * C_ii)
                # where C_ii is diagonal element of (X'X)^-1
                # Simplified: se(Effect) ≈ sqrt(MS_error / n_runs)
                
                se = np.sqrt(MS_error / n_samples)
                standard_errors.append(se)
                
                # Calculate t-statistic
                coef = row['Coefficient']
                if not pd.isna(coef) and se > 0:
                    t_stat = coef / se
                    t_statistics.append(t_stat)
                    
                    # 95% Confidence interval
                    ci_low = coef - t_crit * se
                    ci_high = coef + t_crit * se
                    ci_lower.append(ci_low)
                    ci_upper.append(ci_high)
                else:
                    t_statistics.append(np.nan)
                    ci_lower.append(np.nan)
                    ci_upper.append(np.nan)
            else:
                standard_errors.append(np.nan)
                t_statistics.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
        
        # Add to DataFrame
        detailed_anova_df['Std_Error'] = standard_errors
        detailed_anova_df['t_Statistic'] = t_statistics
        detailed_anova_df['CI_Lower_95'] = ci_lower
        detailed_anova_df['CI_Upper_95'] = ci_upper
    else:
        # No valid error term for calculating standard errors
        detailed_anova_df['Std_Error'] = np.nan
        detailed_anova_df['t_Statistic'] = np.nan
        detailed_anova_df['CI_Lower_95'] = np.nan
        detailed_anova_df['CI_Upper_95'] = np.nan
    
    # Calculate overall model statistics
    SS_regression = SS_total - SS_residual_full
    R_squared = 1 - (SS_residual_full / SS_total)
    
    # Adjusted R-squared
    if n_samples > n_params:
        R_squared_adj = 1 - ((SS_residual_full / df_error) / (SS_total / (n_samples - 1)))
    else:
        R_squared_adj = np.nan
    
    model_summary = {
        'R_squared': R_squared,
        'R_squared_adj': R_squared_adj,
        'SS_regression': SS_regression,
        'SS_residual': SS_residual_full,
        'SS_total': SS_total,
        'MS_error': MS_error,
        'df_error': df_error,
        'n_parameters': n_params,
        'n_samples': n_samples,
        'predictions': y_pred_full,
        'residuals': y - y_pred_full,
        # Flag so the UI can show a warning when auto-normalisation was applied
        'mixture_was_normalized': mixture_was_normalized,
    }
    
    return detailed_anova_df, model_summary


def calculate_lenth_pse(effects):
    """
    Calculate Lenth's Pseudo Standard Error (PSE) for unreplicated designs
    
    Lenth's PSE provides a robust estimate of effect variability without replication.
    It's particularly useful for screening experiments and unreplicated factorials.
    
    Parameters:
    -----------
    effects : array-like
        Array of absolute effect values
    
    Returns:
    --------
    PSE : float
        Pseudo Standard Error
    ME : float
        Margin of Error (individual effect threshold at α=0.05)
    SME : float
        Simultaneous Margin of Error (conservative threshold)
    """
    effects = np.array(effects)
    effects = effects[np.isfinite(effects)]  # Remove inf/nan
    
    if len(effects) < 3:
        # Too few effects for robust estimation
        return np.std(effects) if len(effects) > 0 else 1.0, None, None
    
    # Step 1: Calculate initial estimate
    s0 = 1.5 * np.median(effects)
    
    # Step 2: Identify small effects (likely noise)
    small_effects = effects[effects < 2.5 * s0]
    
    # Step 3: Calculate PSE from small effects
    if len(small_effects) > 0:
        PSE = 1.5 * np.median(small_effects)
    else:
        PSE = s0
    
    # Ensure PSE is reasonable (not too small)
    if PSE < 1e-10:
        PSE = np.std(effects) if np.std(effects) > 1e-10 else 1e-10
    
    # Calculate Margin of Error (ME) and Simultaneous ME (SME)
    # Conservative degrees of freedom
    df_lenth = max(1, len(effects) // 3)
    
    from scipy import stats as scipy_stats
    
    # ME: Individual effect threshold (α = 0.05, two-tailed)
    t_me = scipy_stats.t.ppf(0.975, df_lenth)
    ME = t_me * PSE
    
    # SME: Simultaneous threshold (Bonferroni-like correction)
    # More conservative for multiple comparisons
    t_sme = scipy_stats.t.ppf(1 - 0.05 / (2 * len(effects)), df_lenth)
    SME = t_sme * PSE
    
    return PSE, ME, SME


def identify_significant_terms(anova_df, alpha=0.05, coef_threshold=1e-10):
    """
    Identify significant terms based on p-values and coefficient magnitude
    
    **IMPORTANT**: Terms with coefficient = 0 (or near-zero) are automatically 
    considered NOT significant, regardless of p-value. This prevents false 
    positives from overfitting scenarios where terms with zero effect show 
    spuriously low p-values.
    
    Parameters:
    -----------
    anova_df : DataFrame
        Detailed ANOVA table
    alpha : float
        Significance level (default 0.05)
    coef_threshold : float
        Minimum absolute coefficient value (default 1e-10)
        Coefficients below this are considered zero/not significant
    
    Returns:
    --------
    significant_terms : list
        List of significant term names based on p-value AND non-zero coefficient
    """
    # Filter out error and total rows
    terms_df = anova_df[~anova_df['Source'].isin(['Error (Residual)', 'Total'])]
    
    # Check if Coefficient column exists
    if 'Coefficient' in terms_df.columns:
        # Filter by p-value AND non-zero coefficient
        # A term is significant ONLY if:
        # 1. P-value < alpha, AND
        # 2. Coefficient is not zero (absolute value > threshold)
        significant = terms_df[
            (terms_df['P_Value'] < alpha) &
            (terms_df['Coefficient'].abs() > coef_threshold)
        ]['Source'].tolist()
    else:
        # Fallback: filter by p-value only (for backward compatibility)
        significant = terms_df[terms_df['P_Value'] < alpha]['Source'].tolist()
    
    return significant


def identify_significant_terms_by_logworth(anova_df, logworth_threshold=1.3, coef_threshold=1e-10):
    """
    Identify significant terms based on LogWorth values and coefficient magnitude
    
    LogWorth is more robust than p-values, especially in overfitting scenarios.
    It's defined as -log10(p-value), which provides a more interpretable scale:
    
    - LogWorth > 2.0: highly significant (p < 0.01)
    - LogWorth > 1.3: significant (p < 0.05)
    - LogWorth > 1.0: marginally significant (p < 0.1)
    - LogWorth < 1.0: not significant
    
    LogWorth is particularly useful when:
    1. Model is overfitting (many parameters, few observations)
    2. P-values approach 0 or 1 (numerical extremes)
    3. Comparing effects across different scales
    
    **IMPORTANT**: Terms with coefficient = 0 (or near-zero) are automatically 
    considered NOT significant, regardless of LogWorth or p-value. This prevents 
    false positives from overfitting scenarios.
    
    Parameters:
    -----------
    anova_df : DataFrame
        Detailed ANOVA table with LogWorth column
    logworth_threshold : float
        Minimum LogWorth value for significance (default 1.3 for alpha=0.05)
        Use 2.0 for alpha=0.01, or 1.0 for alpha=0.10
    coef_threshold : float
        Minimum absolute coefficient value (default 1e-10)
        Coefficients below this are considered zero/not significant
    
    Returns:
    --------
    significant_terms : list
        List of significant term names based on LogWorth AND non-zero coefficient
    """
    # Filter out error and total rows
    terms_df = anova_df[~anova_df['Source'].isin(['Error (Residual)', 'Total'])]
    
    # Check if LogWorth column exists
    if 'LogWorth' not in terms_df.columns:
        raise ValueError("LogWorth column not found in ANOVA table. Update your ANOVA calculation to include LogWorth.")
    
    # Check if Coefficient column exists
    if 'Coefficient' not in terms_df.columns:
        raise ValueError("Coefficient column not found in ANOVA table.")
    
    # Filter by LogWorth AND non-zero coefficient
    # A term is significant ONLY if:
    # 1. LogWorth >= threshold (or inf), AND
    # 2. Coefficient is not zero (absolute value > threshold)
    significant = terms_df[
        ((terms_df['LogWorth'] >= logworth_threshold) | (terms_df['LogWorth'] == np.inf)) &
        (terms_df['Coefficient'].abs() > coef_threshold)
    ]['Source'].tolist()
    
    return significant


def rank_terms_by_importance(anova_df, metric='LogWorth', ascending=False):
    """
    Rank model terms by importance using specified metric
    
    Parameters:
    -----------
    anova_df : DataFrame
        Detailed ANOVA table
    metric : str
        Metric to use for ranking:
        - 'LogWorth': -log10(p-value), robust to overfitting (recommended)
        - 'Coefficient': absolute coefficient value
        - 'Sum_of_Squares': contribution to explained variance
        - 'F_statistic': F-test statistic
    ascending : bool
        If True, rank from smallest to largest. Default False (largest first).
    
    Returns:
    --------
    ranked_df : DataFrame
        Terms sorted by importance with rank column
    """
    # Filter out error and total rows
    terms_df = anova_df[~anova_df['Source'].isin(['Error (Residual)', 'Total'])].copy()
    
    if metric not in terms_df.columns:
        raise ValueError(f"Metric '{metric}' not found in ANOVA table. Available: {list(terms_df.columns)}")
    
    # For Coefficient, use absolute value
    if metric == 'Coefficient':
        terms_df['_sort_value'] = terms_df[metric].abs()
        sort_col = '_sort_value'
    else:
        sort_col = metric
    
    # Handle inf values in LogWorth - put them at top
    if metric == 'LogWorth':
        terms_df['_has_inf'] = terms_df[metric] == np.inf
        terms_df = terms_df.sort_values(by=['_has_inf', sort_col], ascending=[False, ascending])
        terms_df = terms_df.drop(columns=['_has_inf'])
    else:
        terms_df = terms_df.sort_values(by=sort_col, ascending=ascending)
    
    # Add rank column
    terms_df.insert(0, 'Rank', range(1, len(terms_df) + 1))
    
    # Drop temporary sort column if created
    if '_sort_value' in terms_df.columns:
        terms_df = terms_df.drop(columns=['_sort_value'])
    
    return terms_df
