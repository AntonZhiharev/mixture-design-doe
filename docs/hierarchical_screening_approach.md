# Hierarchical Screening Approach for Higher-Order Interactions

## The Problem with Fitting All Parameters at Once

When we try to fit a quartic model with 30 parameters using only 60 points:
- **Points/Parameter ratio**: 60/30 = 2:1 (too low!)
- **Result**: Overfitting, unreliable coefficient estimates
- **Effect**: Higher-order terms appear insignificant even when they're real

## The Proper Solution: Hierarchical Screening

Instead of fitting all 30 parameters simultaneously, we use a **staged screening approach**:

### Stage 1: Main Effects Screening (Linear Terms)
- **Model**: Linear only (5 parameters)
- **Points needed**: ~15-20 points
- **Method**: Fit y = β₁x₁ + β₂x₂ + β₃x₃ + β₄x₄ + β₅x₅
- **Significance test**: Pseudo t-ratio, Pseudo p-value
- **Decision**: Keep significant main effects for next stage

### Stage 2: Two-Way Interaction Screening
- **Model**: Linear + all 2-way interactions (15 parameters)
- **Points needed**: ~30-45 points
- **Method**: Add x₁x₂, x₁x₃, ..., x₄x₅ to significant main effects
- **Check**: Alias structure (confounding between terms)
- **Significance test**: Pseudo t-ratio, Pseudo p-value
- **Decision**: Keep significant 2-way interactions

### Stage 3: Three-Way Interaction Screening
- **Model**: Previous + selected 3-way interactions
- **Points needed**: ~50-70 points
- **Method**: Only test 3-way interactions involving terms significant in Stage 2
- **Example**: If x₁x₂ was significant, test x₁x₂x₃, x₁x₂x₄, x₁x₂x₅
- **Significance test**: Pseudo t-ratio, Pseudo p-value
- **Decision**: Keep significant 3-way interactions

### Stage 4: Four-Way Interaction Screening
- **Model**: Previous + selected 4-way interactions
- **Points needed**: ~70-100 points
- **Method**: Only test 4-way among terms significant in Stage 3
- **Significance test**: Pseudo t-ratio, Pseudo p-value
- **Decision**: Keep significant 4-way interactions

### Stage 5: Five-Way Interaction Screening (if applicable)
- **Model**: Previous + selected 5-way interactions
- **Points needed**: ~90-120 points
- **Method**: Only test if all 5 components were in significant lower-order terms
- **Significance test**: Pseudo t-ratio, Pseudo p-value

## Key Principles

### 1. Heredity Principle
- **Strong heredity**: A 2-way interaction x₁x₂ can only be significant if BOTH x₁ AND x₂ are significant
- **Weak heredity**: A 2-way interaction x₁x₂ can be significant if EITHER x₁ OR x₂ is significant
- Apply same principle to higher-order interactions

### 2. Effect Sparsity
- **Assumption**: Only a few terms are truly active
- **Implication**: We don't need to test ALL possible interactions
- **Benefit**: Greatly reduces the number of parameters to estimate

### 3. Sequential Design Augmentation
- Start with minimal design for main effects
- Add points optimally for testing 2-way interactions
- Continue adding points as needed for higher orders
- Stop when no more significant effects found

## Pseudo t-Ratio and Pseudo p-Value

### Why "Pseudo"?
When we don't have enough points for classical t-tests, we use:

1. **Pseudo t-ratio**:
   ```
   t_pseudo = |coefficient| / Pseudo_SE
   
   Where Pseudo_SE is estimated from:
   - Pooled variance of inactive terms
   - Bootstrap resampling
   - Lenth's method (PSE - Pseudo Standard Error)
   ```

2. **Pseudo p-value**:
   ```
   p_pseudo = 2 * P(|t| > |t_pseudo|)
   
   Using adjusted df based on:
   - Number of active terms
   - Sparsity assumptions
   ```

### Lenth's Pseudo Standard Error (PSE)

For screening designs with limited data:
```python
def calculate_lenth_pse(effects):
    """
    Calculate Lenth's PSE for effect significance.
    
    Steps:
    1. Calculate absolute effects
    2. Find s₀ = 1.5 × median(|effects|)
    3. Select effects < 2.5 × s₀
    4. PSE = 1.5 × median(selected effects)
    5. ME (Margin of Error) = t₀.₀₂₅ × PSE
    6. SME (Simultaneous ME) = t₀.₀₀₀₈₃ × PSE
    """
    abs_effects = np.abs(effects)
    s0 = 1.5 * np.median(abs_effects)
    
    # Select effects in range
    selected = abs_effects[abs_effects < 2.5 * s0]
    
    if len(selected) == 0:
        return s0
    
    PSE = 1.5 * np.median(selected)
    return PSE
```

## Alias Structure

### What are Aliases?
When parameters are **confounded** (can't be distinguished):
```
Example with 8 points for 3 components:
- Main effect x₁ might be aliased with x₂x₃
- Can't tell if effect is from x₁ or x₂x₃
```

### Checking for Aliases
```python
def check_alias_structure(design_matrix, term_names):
    """
    Check which terms are confounded.
    
    Two terms are aliased if their columns are:
    - Identical
    - Negatives of each other
    - Linearly dependent
    """
    # Calculate correlation matrix
    correlations = np.corrcoef(design_matrix.T)
    
    # Find high correlations (|r| > 0.95)
    aliases = []
    for i in range(len(term_names)):
        for j in range(i+1, len(term_names)):
            if abs(correlations[i,j]) > 0.95:
                aliases.append((term_names[i], term_names[j], correlations[i,j]))
    
    return aliases
```

## Complete Workflow

```
Stage 1: Main Effects
├── Fit: y ~ x₁ + x₂ + x₃ + x₄ + x₅
├── Calculate: Pseudo t-ratios using Lenth's PSE
├── Identify: Significant main effects (e.g., x₁, x₂, x₄)
└── Decision: Move to Stage 2

Stage 2: Two-Way Interactions
├── Test only: Interactions involving x₁, x₂, x₄
├── Models: x₁x₂, x₁x₄, x₂x₄
├── Check: Alias structure
├── Calculate: Pseudo t-ratios
├── Identify: Significant 2-way (e.g., x₁x₂)
└── Decision: Move to Stage 3

Stage 3: Three-Way Interactions
├── Test only: x₁x₂ with other active terms
├── Models: x₁x₂x₄ only (since x₃, x₅ not active)
├── Calculate: Pseudo t-ratios
├── Identify: Significant 3-way (if any)
└── Decision: Continue or stop

Final Model:
├── Include: All significant terms from all stages
├── Refit: Using only significant terms
├── Validate: Check fit, residuals, predictions
└── Report: Final model with confidence intervals
```

## Advantages of Hierarchical Screening

1. **Works with limited data**: Can screen 30+ potential terms with 60 points
2. **Respects effect sparsity**: Only tests plausible interactions
3. **Heredity principle**: Ensures logical model structure
4. **Computational efficiency**: Avoids fitting over-parameterized models
5. **Better power**: More power to detect active effects at each stage
6. **Interpretability**: Clear understanding of model structure

## Implementation Requirements

To implement this properly, we need:

1. **Lenth's PSE calculator** for pseudo p-values
2. **Heredity checker** to determine which interactions to test
3. **Alias detector** to identify confounded terms
4. **Staged fitting** capability
5. **Design augmentation** for each stage
6. **Reporting** of screening results at each stage

## Comparison: All-at-Once vs Hierarchical

| Aspect | All-at-Once (Current) | Hierarchical Screening |
|--------|----------------------|------------------------|
| Parameters tested | 30 simultaneously | 5 → 10 → few → fewer |
| Points needed | 90-120 | 60-80 |
| Power to detect | Low (many df) | High (focused) |
| Overfitting risk | High | Low |
| Interpretability | Low | High |
| Computational cost | High | Medium |
| Success rate | Poor with <100 pts | Good with 60+ pts |

## Conclusion

**The hierarchical screening approach is the proper method for estimating higher-order interactions with limited experimental data.** It's what professional DOE software like JMP, Design-Expert, and Minitab use for definitive screening designs.

This should be implemented in the sequential regression reconstruction to properly handle quartic and quintic models.
