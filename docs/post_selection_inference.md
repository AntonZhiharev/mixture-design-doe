# Post-Selection Inference for Parameter Recovery

## ⚠️ CRITICAL ISSUE: Selection Bias

After data-driven variable selection (e.g., Lasso, FDR, stepwise selection), **standard confidence intervals are INVALID**:
- They are too narrow (underestimate uncertainty)
- Coverage is < 95% (often 60-80%)
- P-values are biased (too small)
- This applies to ALL selection methods in current implementation

## Why This Happens

When you:
1. Use the SAME data to select variables
2. Then estimate parameters from selected variables
3. Traditional inference assumes variables were chosen a priori

**Result**: You're "double-dipping" the data, leading to overconfident estimates.

## Modern Solutions

### 1. Data Splitting ⭐ MOST ROBUST

**Concept**: Use different data for selection vs estimation

```python
# Split data: 50% selection, 50% estimation
n = len(design)
select_idx = np.random.choice(n, n//2, replace=False)
estimate_idx = np.setdiff1d(np.arange(n), select_idx)

# Step 1: Select using first half
screener = JMPFullModelScreening(...)
result_select = screener.screen_full_model(
    design[select_idx], 
    responses[select_idx]
)
selected_terms = result_select.selected_terms

# Step 2: Estimate using second half (FRESH DATA!)
# Build design matrix with only selected terms
X_estimate = build_design_matrix(design[estimate_idx], selected_terms)
y_estimate = responses[estimate_idx]

# Standard linear regression - CIs are now VALID!
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_estimate, y_estimate)

# Get confidence intervals (use statsmodels for CIs)
import statsmodels.api as sm
model_sm = sm.OLS(y_estimate, sm.add_constant(X_estimate)).fit()
conf_intervals = model_sm.conf_int(alpha=0.05)  # TRUE 95% CIs!
```

**Advantages**:
- ✅ Valid inference (correct coverage)
- ✅ Conceptually simple
- ✅ No special software needed

**Disadvantages**:
- ❌ Loses power (half data for estimation)
- ❌ Selection becomes less reliable with smaller n

### 2. Bootstrap Confidence Intervals

**Concept**: Resample data to estimate uncertainty

```python
def bootstrap_estimate(design, responses, selected_terms, n_bootstrap=1000):
    n = len(design)
    estimates = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_idx = np.random.choice(n, n, replace=True)
        X_boot = build_design_matrix(design[boot_idx], selected_terms)
        y_boot = responses[boot_idx]
        
        # Fit model
        model = LinearRegression().fit(X_boot, y_boot)
        estimates.append(model.coef_)
    
    # Calculate percentile confidence intervals
    estimates = np.array(estimates)
    lower = np.percentile(estimates, 2.5, axis=0)
    upper = np.percentile(estimates, 97.5, axis=0)
    
    return lower, upper
```

**Advantages**:
- ✅ Flexible, works with complex designs
- ✅ Doesn't assume normality
- ✅ Uses all data

**Disadvantages**:
- ❌ Computationally intensive
- ⚠️ Still biased if selection done on same data
- ⚠️ Underestimates uncertainty

### 3. Selective Inference (Advanced)

**Concept**: Adjust CIs to account for selection process

Requires specialized software (R package `selectiveInference`):

```R
library(selectiveInference)
# After Lasso selection
fit <- lar(X, y)
# VALID confidence intervals accounting for selection
inference_result <- larInf(fit, sigma = sigma_estimate, alpha = 0.05)
```

**Key Papers**:
- Lee et al. (2016) - "Exact post-selection inference"
- Taylor & Tibshirani (2015) - "Statistical learning and selective inference"

**Advantages**:
- ✅ Theoretically optimal
- ✅ Uses all data
- ✅ Valid inference

**Disadvantages**:
- ❌ Computationally complex
- ❌ Limited software support
- ❌ Only available for specific methods (Lasso, forward stepwise)

### 4. Bayesian Approach

**Concept**: Full posterior distribution naturally includes selection uncertainty

```python
import pystan  # or use rstanarm in R

# Bayesian regression with horseshoe prior (like Lasso)
model = """
data {
  int<lower=0> N;
  int<lower=0> P;
  matrix[N, P] X;
  vector[N] y;
}
parameters {
  vector[P] beta;
  real<lower=0> sigma;
  vector<lower=0>[P] lambda;
  real<lower=0> tau;
}
model {
  // Horseshoe prior (induces sparsity)
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, 1);
  beta ~ normal(0, tau * lambda);
  sigma ~ cauchy(0, 5);
  
  y ~ normal(X * beta, sigma);
}
"""

# Posterior credible intervals are VALID!
posterior_intervals = fit.extract()['beta']
ci_lower = np.percentile(posterior_intervals, 2.5, axis=0)
ci_upper = np.percentile(posterior_intervals, 97.5, axis=0)
```

**Advantages**:
- ✅ Natural uncertainty quantification
- ✅ Incorporates prior knowledge
- ✅ Handles selection and estimation together

**Disadvantages**:
- ❌ Computationally intensive
- ❌ Requires prior specification
- ❌ Learning curve for Bayesian methods

## Recommended Workflow

### For Research/Publication:

```
┌────────────────────────────────────────┐
│ Phase 1: EXPLORATION (Full Data)      │
│ - Use current staged analysis          │
│ - Identify important terms              │
│ - Visualize effects                     │
│ - DO NOT report these estimates!       │
└────────────────────────────────────────┘
                  ↓
┌────────────────────────────────────────┐
│ Phase 2: ESTIMATION (Choose method)   │
│                                         │
│ Option A: New Experiment (GOLD)       │
│ - Collect fresh data                   │
│ - Pre-register selected terms          │
│ - Standard inference valid             │
│                                         │
│ Option B: Data Splitting               │
│ - Re-analyze with split                │
│ - Report valid CIs                     │
│                                         │
│ Option C: Selective Inference          │
│ - Use specialized software             │
│ - Report adjusted CIs                  │
└────────────────────────────────────────┘
```

### For Process Optimization (Less Critical):

If you just need good point estimates (not CIs):
- Current approach is fine for finding important factors
- Use cross-validation to verify predictions
- Consider confidence intervals directional only

## Implementation Status

### Current Implementation:
- ❌ Standard CIs (INVALID after selection)
- ✅ Good for exploration
- ✅ Point estimates reasonable

### TODO:
- [ ] Add data splitting option
- [ ] Add bootstrap CIs
- [ ] Add warnings about CI interpretation
- [ ] Provide post-selection re-estimation tools

## References

1. **Berk et al. (2013)** - "Valid post-selection inference"
   - Shows standard inference fails after selection

2. **Taylor & Tibshirani (2015)** - "Statistical learning and selective inference"
   - Theoretical framework for post-selection inference

3. **Lee et al. (2016)** - "Exact post-selection inference for sequential regression procedures"
   - Practical methods for common selection procedures

4. **Efron (2014)** - "Estimation and accuracy after model selection"
   - Bootstrap approaches to post-selection inference

5. **Rinaldo et al. (2019)** - "Bootstrapping and sample splitting for high-dimensional, assumption-lean inference"
   - Modern review of practical methods

## Key Takeaways

1. **After variable selection, standard CIs are wrong** - typically 20-40% too narrow
2. **Data splitting is the simplest robust solution** - but loses power
3. **For critical decisions, use fresh data** - gold standard
4. **Current implementation is fine for exploration** - but not for final inference
5. **Bootstrap helps but doesn't fully solve the problem** - still underestimates uncertainty
