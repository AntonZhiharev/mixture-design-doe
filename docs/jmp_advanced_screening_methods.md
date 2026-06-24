# JMP's Advanced Methods for Higher-Order Interaction Estimation

## Why JMP Can Estimate from 47 Points

When JMP successfully estimates higher-order interactions from only 47 points for a 5-component quartic model (which has 30 parameters), they're using several advanced techniques that go far beyond simple OLS regression.

## Key Differences from Basic Approach

### 1. Definitive Screening Designs (DSD)

**What is DSD?**
- Special 3-level design developed by Jones & Nachtsheim (2011)
- Requires only **2k + 1** runs for k factors (for 5 factors = 11 runs!)
- **Projection properties**: Any 3 factors form a full factorial
- **Estimation capacity**: Can estimate all main effects, 2-way interactions, and detect higher-order effects

**For mixture models:**
- Adapted DSD with mixture constraints
- Strategically placed points to decorrelate effects
- Maximizes information per experimental run

**Example for 5 components:**
```
Minimum DSD runs: 2(5) + 1 = 11 runs
With replication: ~15-20 runs for main + 2-way
Additional augmentation: +25-30 runs for higher-order
Total: ~47 runs
```

### 2. D-Optimal Point Selection

**Not random points!** JMP uses D-optimal algorithms to:

```python
# Pseudocode for JMP's approach
def generate_optimal_design(n_runs, model_terms):
    """
    Generate D-optimal design that maximizes:
    det(X'X) where X is design matrix
    """
    # 1. Generate large candidate set (1000+ points)
    candidates = generate_candidate_points(n=10000)
    
    # 2. Select n_runs points that maximize determinant
    selected = []
    for i in range(n_runs):
        best_point = None
        best_det = -inf
        
        for candidate in candidates:
            test_design = selected + [candidate]
            X = build_design_matrix(test_design, model_terms)
            det_value = determinant(X.T @ X)
            
            if det_value > best_det:
                best_det = det_value
                best_point = candidate
        
        selected.append(best_point)
        candidates.remove(best_point)
    
    return selected
```

**Key difference:**
- Random 47 points: D-efficiency ~0.2-0.4
- D-optimal 47 points: D-efficiency ~0.8-0.95
- **This 2-4× improvement is critical!**

### 3. Forward Selection with Regularization

JMP doesn't fit all 30 parameters at once. Instead:

**Step 1: Start with intercept only**
```
Model_0: y = β₀
```

**Step 2: Add most significant main effect**
```
Try each: y = β₀ + β₁x₁
         y = β₀ + β₁x₂
         ...
Select best based on p-value and AIC
```

**Step 3: Continue adding terms one at a time**
```
For each candidate term:
    1. Fit augmented model
    2. Calculate:
       - p-value (must be < 0.05)
       - VIF (variance inflation factor < 10)
       - AIC improvement (must decrease)
    3. Keep term if all criteria met
    4. Stop when no improvement
```

**JMP's stopping criteria:**
- p-value threshold (e.g., 0.05 or 0.10)
- AIC/BIC improvement
- Maximum allowed VIF
- Heredity constraints
- Minimum effect size

### 4. Effect Heredity Enforcement

JMP strictly enforces effect heredity:

**Strong Heredity:**
```python
def can_add_interaction(term, current_model):
    """
    Example: Can only add x1*x2*x3 if:
    - x1 is in model
    - x2 is in model  
    - x3 is in model
    """
    components = parse_term(term)
    return all(comp in current_model for comp in components)
```

**This dramatically reduces candidate terms:**
```
Without heredity:
- All 4-way interactions: C(5,4) = 5 terms

With strong heredity (only x1, x2, x4 significant):
- Eligible 4-way: only x1*x2*x4 with one more
- Result: ~2 terms instead of 5
```

### 5. Ridge Regression / Lasso

For near-collinear designs, JMP uses regularized regression:

**Ridge Regression (L2 penalty):**
```
minimize: ||y - Xβ||² + λ||β||²

Effect:
- Shrinks coefficients
- Stabilizes estimates
- Reduces overfitting
- Can handle more parameters than observations!
```

**Lasso (L1 penalty):**
```
minimize: ||y - Xβ||² + λ|β|

Effect:
- Shrinks AND selects
- Forces small coefficients to exactly zero
- Automatic variable selection
- Sparse models
```

**JMP's adaptive approach:**
```python
# JMP likely uses elastic net (combination)
alpha = 0.5  # Mix of L1 and L2
lambda_optimal = cross_validation_select(lambdas)

model = ElasticNet(alpha=alpha, lambda=lambda_optimal)
model.fit(X, y)

# Keep only non-zero coefficients
significant_terms = [term for term, coef in zip(terms, model.coef_) 
                     if abs(coef) > threshold]
```

### 6. Cross-Validation for Model Selection

JMP uses k-fold cross-validation:

```python
def select_model_terms(X_full, y, candidate_terms):
    """
    Use cross-validation to select best subset
    """
    best_score = -inf
    best_terms = []
    
    for subset in generate_subsets(candidate_terms):
        cv_scores = []
        
        for train_idx, test_idx in k_fold_split(n=5):
            X_train = X_full[train_idx][:, subset]
            y_train = y[train_idx]
            X_test = X_full[test_idx][:, subset]
            y_test = y[test_idx]
            
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)
        
        avg_score = mean(cv_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_terms = subset
    
    return best_terms
```

### 7. Variance Inflation Factor (VIF) Monitoring

JMP checks multicollinearity:

```python
def calculate_vif(X, term_index):
    """
    VIF for a term = 1 / (1 - R²)
    where R² is from regressing that term on all others
    """
    X_others = delete_column(X, term_index)
    X_target = X[:, term_index]
    
    model = LinearRegression()
    model.fit(X_others, X_target)
    R2 = model.score(X_others, X_target)
    
    VIF = 1 / (1 - R2)
    return VIF

# JMP's rule: Only keep terms with VIF < 10 (some use < 5)
for i, term in enumerate(model_terms):
    if calculate_vif(X, i) > 10:
        remove_term(term)
```

### 8. Orthogonal Encoding for Mixture Components

For mixture designs, JMP may use **Scheffé canonical polynomials** or **Cox's mixture polynomials** which have better numerical properties:

```python
# Standard encoding (we use):
x1*x2 = component1 × component2

# Scheffé encoding (JMP):
# Automatically orthogonal for mixtures
# Better conditioned matrices
```

## Complete JMP-Style Algorithm

```python
class JMPStyleScreening:
    def fit(self, design_points, responses):
        # Step 1: Generate optimal design
        optimal_design = self.d_optimal_augmentation(
            current_design=design_points,
            target_model=quartic_terms,
            n_additional=max(0, 47 - len(design_points))
        )
        
        # Step 2: Build candidate term list with heredity
        candidates = self.build_candidate_list(
            max_order=4,
            enforce_heredity='weak'
        )
        
        # Step 3: Forward selection with regularization
        selected_terms = []
        current_model = ['intercept']
        
        while len(candidates) > 0:
            best_term = None
            best_metric = -inf
            
            for term in candidates:
                # Try adding term
                trial_terms = current_model + [term]
                
                # Fit with ridge regression
                model = Ridge(alpha=0.01)
                X = self.build_matrix(optimal_design, trial_terms)
                model.fit(X, responses)
                
                # Calculate metrics
                p_value = self.calculate_p_value(model, term)
                vif = self.calculate_vif(X, len(trial_terms)-1)
                aic = self.calculate_aic(model, X, responses)
                
                # Check criteria
                if (p_value < 0.05 and 
                    vif < 10 and
                    aic < current_aic):
                    
                    if aic > best_metric:
                        best_metric = aic
                        best_term = term
            
            if best_term is None:
                break  # No improvement
            
            # Add best term
            current_model.append(best_term)
            selected_terms.append(best_term)
            candidates.remove(best_term)
            
            # Update candidates based on heredity
            candidates = self.update_candidates(
                candidates, selected_terms
            )
        
        # Step 4: Final model with only selected terms
        X_final = self.build_matrix(optimal_design, selected_terms)
        final_model = Ridge(alpha=0.001)  # Light regularization
        final_model.fit(X_final, responses)
        
        return final_model, selected_terms
```

## Why Our Simple Approach Failed

| Aspect | Our Approach | JMP Approach |
|--------|--------------|--------------|
| Design | Random mixture points | D-optimal selection |
| Fitting | All 30 params at once | Forward selection |
| Regularization | None | Ridge/Lasso |
| Collinearity | Ignored | VIF monitoring |
| Model selection | R² only | AIC + CV + p-values |
| Heredity | Not enforced | Strictly enforced |
| Points/param ratio | 2:1 | Irrelevant with regularization |

## Implementation Recommendations

To match JMP's capabilities:

1. **Replace random design with D-optimal**
   ```python
   from src.algorithms.d_optimal_algorithm import DOptimalAlgorithm
   design = DOptimalAlgorithm.generate(n_runs=47, model='quartic')
   ```

2. **Use forward selection instead of all-at-once**
   ```python
   from sklearn.feature_selection import SequentialFeatureSelector
   selector = SequentialFeatureSelector(estimator, direction='forward')
   ```

3. **Add regularization**
   ```python
   from sklearn.linear_model import Ridge, ElasticNet
   model = ElasticNet(alpha=0.01, l1_ratio=0.5)
   ```

4. **Enforce heredity constraints**
   ```python
   from src.algorithms.hierarchical_screening import HeredityChecker
   eligible_terms = HeredityChecker.get_eligible_interactions(...)
   ```

5. **Monitor VIF**
   ```python
   from statsmodels.stats.outliers_influence import variance_inflation_factor
   ```

6. **Use cross-validation for final model**
   ```python
   from sklearn.model_selection import cross_val_score
   ```

## Conclusion

JMP's success with 47 points comes from:

1. ✅ **Optimal experimental design** (not random points)
2. ✅ **Smart model building** (forward selection, not all-at-once)
3. ✅ **Regularization** (Ridge/Lasso handles collinearity)
4. ✅ **Heredity enforcement** (reduces parameter space)
5. ✅ **Multiple criteria** (p-value + AIC + VIF + CV)
6. ✅ **Advanced algorithms** (10+ years of JMP development)

**Bottom line:** It's not about having more data—it's about using the data optimally through sophisticated algorithms.
