# False Positive Interaction Analysis

## Problem Summary

When running two-stage selection on the test dataset, we observed:

**Original Problem (without hierarchical constraints):**
- ❌ False positive 3-way: `x1*x4*x5` (doesn't exist in true model)
- ❌ Missing both 4-way interactions: `x1*x2*x3*x4` and `x2*x3*x4*x5`

**After adding hierarchical constraints:**
- ✅ Fixed: `x1*x4*x5` false positive eliminated
- ❌ New issue: 4 weak 2-way false positives (`x1*x2`, `x2*x3`, `x3*x5`, `x4*x5`)
- ❌ Still missing: Both 4-way interactions

## Root Cause Analysis

### 1. **Insufficient Data for 4-Way Interactions**

With 45 runs and a full model of 30 terms (5 main + 10 two-way + 10 three-way + 5 four-way):
- Degrees of freedom: 45 - 30 = 15
- This is barely adequate for estimating the model
- 4-way interactions are the hardest to estimate due to:
  - Smallest signal in the design space
  - Highest multicollinearity with lower-order terms
  - Largest standard errors

**Evidence from diagnostic:**
```
Correlations of x1*x4*x5 (FALSE) with 4-way interactions:
x1*x2*x4*x5   +0.408203
x1*x3*x4*x5   +0.417429
x1*x2*x3*x4   +0.103809  ✅ TRUE
x2*x3*x4*x5   +0.098026  ✅ TRUE
```

The false positive `x1*x4*x5` was correlated with OTHER 4-way terms, not the true ones.

### 2. **Backward Elimination Bias Against High-Order Terms**

Backward elimination starts with all terms and removes the least significant. Early in the process:
- 4-way interactions have high p-values (due to large standard errors)
- Algorithm removes them before they can "prove" themselves
- Their signal gets absorbed by correlated 3-way or 2-way terms

From the fixed test output:
```
Iter 1: Removing x1*x2*x3*x5    (p=0.989771)
Iter 5: Removing x1*x2*x3*x4    (p=0.611892)  ❌ TRUE 4-way removed!
Iter 8: Removing x2*x3*x4*x5    (p=0.460553)  ❌ TRUE 4-way removed!
```

Both true 4-way interactions were removed early with p-values > 0.4.

### 3. **Hierarchical Constraints Create Weak Parent Terms**

When hierarchical constraints prevent removal of parent terms:
- Weak 2-way interactions remain in the model
- These are artifacts created to maintain hierarchy
- Example: `x3*x5` (coef=-0.011, p=0.73) kept because `x3*x4*x5` needs it

### 4. **The Fundamental Mixture Design Challenge**

In mixture designs where components sum to 1:
- Linear dependencies exist between terms
- The design matrix rank equals 30 (no deficiency in this case)
- But practical identifiability is still an issue for high-order terms
- 4-way interactions require very specific design points to estimate

## Solutions

### Solution 1: Use Forward Selection Instead (Recommended)

**Why it works better:**
- Starts with main effects only
- Adds terms incrementally based on significance
- Strong effects get selected first
- Hierarchy is naturally maintained
- Less prone to removing important high-order terms prematurely

**Implementation:**
```python
def forward_selection_with_hierarchy(design, responses, component_names, 
                                    p_threshold=0.05):
    """
    Forward selection that respects hierarchy:
    - Only add interaction if all parents are already in model
    - Add term with smallest p-value (most significant)
    - Stop when no more terms meet criteria
    """
    current_terms = list(component_names)  # Start with main effects
    
    while True:
        best_term, best_p = None, 1.0
        
        # Try each candidate term
        for candidate in candidate_pool:
            if candidate in current_terms:
                continue
            
            # Check hierarchy: all parents must be in model
            parents = get_parents(candidate)
            if not all(p in current_terms for p in parents):
                continue
            
            # Test this term
            test_terms = current_terms + [candidate]
            X = build_design_matrix(design, test_terms, component_names)
            p_values, _ = calculate_p_values(X, responses, test_terms)
            
            if p_values[candidate] < best_p:
                best_term = candidate
                best_p = p_values[candidate]
        
        # Add best term if significant
        if best_term and best_p < p_threshold:
            current_terms.append(best_term)
        else:
            break
    
    return current_terms
```

### Solution 2: Use Regularization (Lasso/Elastic Net)

**Why it works:**
- Handles multicollinearity naturally
- Performs automatic variable selection
- Doesn't suffer from the "remove-then-can't-add-back" problem
- Can be combined with hierarchy constraints

**Implementation:**
```python
from sklearn.linear_model import LassoCV, ElasticNetCV

def regularized_selection(design, responses, terms, component_names):
    """Use cross-validated Lasso for selection"""
    X = build_design_matrix(design, terms, component_names)
    
    # Cross-validated Lasso
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X, responses)
    
    # Select non-zero coefficients
    selected = [term for i, term in enumerate(terms) 
                if abs(lasso.coef_[i]) > 1e-6]
    
    return selected
```

### Solution 3: Increase Sample Size

For reliable estimation of 4-way interactions in a 5-component mixture:
- **Minimum recommended**: 60-80 runs
- **Ideal**: 100+ runs
- Use D-optimal or I-optimal designs optimized for high-order interactions

### Solution 4: Use Practical Significance + Cross-Validation

Instead of relying solely on p-values:
```python
def select_with_cross_validation(design, responses, component_names):
    """
    Select model using cross-validated prediction error
    - Start with simple model
    - Add terms that improve CV error
    - Use practical significance threshold
    """
    from sklearn.model_selection import cross_val_score
    
    current_terms = list(component_names)
    best_cv_score = calculate_cv_score(design, responses, current_terms)
    
    # Iteratively add terms that improve CV score
    improved = True
    while improved:
        improved = False
        for candidate in get_candidates(current_terms, component_names):
            test_terms = current_terms + [candidate]
            cv_score = calculate_cv_score(design, responses, test_terms)
            
            # Must improve by at least 1% to be worth adding
            if cv_score > best_cv_score * 1.01:
                current_terms.append(candidate)
                best_cv_score = cv_score
                improved = True
                break
    
    return current_terms
```

### Solution 5: Two-Stage Approach (Modified)

Keep your two-stage approach but modify Stage 1:
```python
def improved_two_stage_selection(design, responses, component_names):
    """
    Stage 1: Forward selection with hierarchy
    Stage 2: Remove small effects (except those required by hierarchy)
    Stage 3: Refit and validate
    """
    # Use forward selection instead of backward
    stage1_terms = forward_selection_with_hierarchy(
        design, responses, component_names, p_threshold=0.10
    )
    
    # Stage 2: Remove genuinely small effects
    stage2_terms = remove_small_effects_with_hierarchy(
        design, responses, stage1_terms, component_names,
        relative_threshold=0.05
    )
    
    # Stage 3: Cross-validate
    cv_score = cross_val_score(design, responses, stage2_terms)
    
    return stage2_terms, cv_score
```

## Recommendations

**For your specific case (45 runs, 5 components):**

1. **Short term**: Use **forward selection with hierarchical constraints**
   - Will better detect 4-way interactions
   - Naturally maintains hierarchy
   - Less prone to false positives

2. **Medium term**: Add **cross-validation** to validate selections
   - Helps catch overfitting
   - More robust than p-values alone

3. **Long term**: Increase sample size to 80-100 runs
   - Enables reliable estimation of 4-way interactions
   - Reduces multicollinearity issues

## Comparison of Methods

| Method | False Positives | Detects 4-way | Maintains Hierarchy | Handles Multicollinearity |
|--------|----------------|---------------|---------------------|---------------------------|
| Backward (original) | ❌ High (x1*x4*x5) | ❌ No | ❌ No | ❌ Poor |
| Backward + hierarchy | ⚠️ Some (weak 2-ways) | ❌ No | ✅ Yes | ⚠️ Moderate |
| Forward + hierarchy | ✅ Low | ⚠️ Sometimes | ✅ Yes | ⚠️ Moderate |
| Lasso/ElasticNet | ✅ Very Low | ⚠️ Sometimes | ⚠️ Optional | ✅ Excellent |
| Cross-validated | ✅ Very Low | ⚠️ Sometimes | ✅ Yes | ✅ Excellent |

## Conclusion

The false positive problem stems from:
1. Backward elimination removing 4-way terms too early
2. Their signal being absorbed by spurious lower-order terms
3. Insufficient data (45 runs) to reliably estimate 4-way interactions

**Best immediate fix**: Switch to forward selection with hierarchical constraints and cross-validation.
