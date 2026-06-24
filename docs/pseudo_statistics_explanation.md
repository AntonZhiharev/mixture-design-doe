# Pseudo t-Ratio and Pseudo p-Value in DOE Analysis

## Why Use Pseudo Statistics?

In Design of Experiments (DOE), especially with **unreplicated designs** or **screening experiments**, traditional t-statistics and p-values can be **unreliable or unavailable** because:

1. **No pure error estimate** - Unreplicated designs have no repeated runs, so classical error variance cannot be calculated
2. **Overfitting risk** - Models with many parameters and few observations lead to inflated significance
3. **Assumption violations** - Traditional ANOVA assumes normally distributed errors, which may not hold

## Lenth's Method: Robust Alternative

**Lenth's Pseudo Standard Error (PSE)** provides a robust estimate of effect variability without requiring replication.

### Mathematical Foundation

#### 1. Pseudo Standard Error (PSE)

```
Step 1: Calculate initial estimate
s₀ = 1.5 × median(|effects|)

Step 2: Identify small effects
Small effects = effects where |effect| < 2.5 × s₀

Step 3: Calculate PSE
PSE = 1.5 × median(|small effects|)
```

**Key Idea:** Small effects represent random noise, so their median provides a robust error estimate.

#### 2. Pseudo t-Ratio

```
Pseudo t-ratio = |Effect| / PSE
```

This ratio indicates how many "standard errors" an effect is from zero.

#### 3. Pseudo p-Value

```
Pseudo p-value = 2 × P(T > |pseudo t-ratio|)

Where T ~ t-distribution with df = n_effects/3 (conservative)
```

## Interpretation Guide

### Pseudo t-Ratio Thresholds

| Pseudo t-Ratio | Interpretation | Action |
|----------------|----------------|--------|
| > 3.0 | Highly significant | ✅ Definitely keep |
| > 2.5 | Very significant | ✅ Keep |
| > 2.0 | Significant | ✅ Keep |
| 1.5 - 2.0 | Moderately significant | ⚠️ Consider keeping |
| < 1.5 | Not significant | ❌ Can remove |

### Pseudo p-Value Thresholds

| Pseudo p-Value | Pseudo t≈ | Significance | Decision |
|----------------|-----------|--------------|----------|
| < 0.001 | >3.5 | Extremely significant | ✅ Keep |
| < 0.01 | >2.8 | Highly significant | ✅ Keep |
| < 0.05 | >2.0 | Significant | ✅ Keep |
| < 0.10 | >1.7 | Marginally significant | ⚠️ Consider |
| ≥ 0.10 | <1.7 | Not significant | ❌ Remove |

## Advantages Over Traditional Statistics

### Traditional t-test Problems:
- ❌ Requires replication for error estimate
- ❌ Assumes normal distribution
- ❌ Sensitive to outliers
- ❌ Unreliable with df_error ≈ 0

### Pseudo Statistics Benefits:
- ✅ **No replication needed** - Works with unreplicated designs
- ✅ **Robust to outliers** - Uses median instead of mean
- ✅ **Distribution-free** - Minimal assumptions
- ✅ **Handles screening** - Automatically separates active from inactive effects
- ✅ **Prevents overfitting** - More conservative than traditional p-values

## When to Use Each Method

### Use Traditional p-values when:
- ✅ You have replicated design (df_error ≥ 5)
- ✅ Errors are normally distributed
- ✅ Model is not overfitted (n_obs >> n_params)
- ✅ You have pure error estimate

### Use Pseudo p-values when:
- ✅ Unreplicated or minimally replicated design
- ✅ Screening many factors
- ✅ Suspected overfitting (n_params close to n_obs)
- ✅ Want more conservative estimates
- ✅ Errors may not be normal

### Use Both (Recommended):
- ✅ **Compare both methods** for robustness
- ✅ **Agreement** between methods → strong conclusion
- ✅ **Disagreement** → investigate further

## Example

### Data:
- 15 runs, 10 effects
- Traditional: df_error = 4 (small!)
- Effects: [0.5, 1.2, 3.5, 0.3, 0.7, 2.1, 0.4, 0.6, 1.8, 0.2]

### Calculation:
```
1. PSE calculation:
   s₀ = 1.5 × median([0.5, 1.2, 3.5, ...]) = 1.5 × 0.7 = 1.05
   Small effects (< 2.625): [0.5, 1.2, 0.3, 0.7, 2.1, 0.4, 0.6, 1.8, 0.2]
   PSE = 1.5 × median([0.5, 1.2, ...]) = 1.5 × 0.6 = 0.9

2. Pseudo t-ratios:
   Effect 3.5: t_pseudo = 3.5 / 0.9 = 3.89 → Highly significant
   Effect 2.1: t_pseudo = 2.1 / 0.9 = 2.33 → Significant
   Effect 1.8: t_pseudo = 1.8 / 0.9 = 2.00 → Significant
   Effect 0.7: t_pseudo = 0.7 / 0.9 = 0.78 → Not significant
```

## References

1. **Lenth, R. V. (1989)**. "Quick and Easy Analysis of Unreplicated Factorials". *Technometrics*, 31(4), 469-473.

2. **Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005)**. *Statistics for Experimenters* (2nd ed.). Wiley.

3. **JMP Statistical Software** - Uses Lenth's method as default for unreplicated designs

4. **Design-Expert Software** - Offers both traditional and Lenth's pseudo statistics

## Implementation Notes

Our implementation:
- Calculates **both traditional AND pseudo statistics**
- Displays side-by-side for comparison
- Recommends which to trust based on design properties
- Flags when traditional statistics may be unreliable
- Uses conservative df for pseudo p-values (n_effects/3)
