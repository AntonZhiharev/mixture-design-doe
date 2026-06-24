# Why High-Order Interactions Cannot Be Estimated

## Summary

**The sequential regression reconstruction algorithm can only estimate up to 2nd-order (quadratic) interactions.** Higher-order interactions (3rd, 4th, 5th order and beyond) cannot be estimated because:

1. The model is hardcoded to fit only quadratic mixture models
2. The design matrix construction only includes linear and 2-way interaction terms
3. Higher-order models require significantly more experimental points and are not implemented

---

## The Problem

Your test generates complex mixture functions with:
- **Linear terms**: `a₁×x₁ + a₂×x₂ + ... + a₅×x₅`
- **Quadratic (2nd order)**: `b₁₂×x₁×x₂ + b₁₃×x₁×x₃ + ...`
- **Cubic (3rd order)**: `c₁₂₃×x₁×x₂×x₃ + ...` ← **Cannot estimate**
- **Quartic (4th order)**: `d₁₂₃₄×x₁×x₂×x₃×x₄ + ...` ← **Cannot estimate**
- **Quintic (5th order)**: `e₁₂₃₄₅×x₁×x₂×x₃×x₄×x₅` ← **Cannot estimate**

However, the reconstruction algorithm only fits:
```python
y = a₁×x₁ + a₂×x₂ + ... + a₅×x₅  # Linear (5 terms)
  + b₁₂×x₁×x₂ + b₁₃×x₁×x₃ + ...   # Quadratic (10 terms)
```

**Total: 15 parameters for a 5-component quadratic mixture model**

---

## Evidence from the Code

### 1. Model Type Configuration
In `sequential_regression_reconstruction.py`, line 157:
```python
model_type: str = "quadratic"  # linear, quadratic, cubic, special_cubic
```

While the config mentions "cubic" and "special_cubic", **the actual implementation never uses them**.

### 2. Analysis Method (Lines 505-524)
```python
def analyze_current_data(self) -> Dict:
    if self._is_mixture_design():
        analyzer = MixtureResponseAnalysis(...)
        
        # Only two options implemented:
        if self.config.model_type == "linear":
            model_results = analyzer.fit_scheffe_linear()
        else:
            model_results = analyzer.fit_scheffe_quadratic()  # ← Default
```

**There is no `elif` for cubic, quartic, or quintic models!**

### 3. Design Matrix Construction (Lines 568-583)
```python
def _build_mixture_quadratic_matrix(self, design: np.ndarray) -> np.ndarray:
    """Build design matrix for mixture quadratic model"""
    n_runs, n_comp = design.shape
    
    # Linear terms
    X = [design]
    
    # Interaction terms (only 2-way)
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            interaction = design[:, i] * design[:, j]
            X.append(interaction.reshape(-1, 1))
    
    return np.column_stack(X)
```

**This method explicitly builds only 2-way interactions.** It would need additional nested loops for 3-way, 4-way, and 5-way interactions.

### 4. Test File Explicitly States This (test_sequential_complex.py)
```python
print(f"   NOTE: Sequential algorithm fits quadratic mixture model (15 params)")
print(f"         but true function may have higher-order terms not captured in model")
```

And later:
```python
print(f"🚀 HIGHER-ORDER TERMS - TRUE COEFFICIENTS (NOT RECOVERED BY QUADRATIC MODEL):")
print(f"   These terms exist in the true function but cannot be estimated by quadratic regression")
```

---

## Mathematical Requirements for Higher-Order Models

For a 5-component mixture:

| Model Order | Example Term | # Parameters | Min Experiments Needed |
|-------------|--------------|--------------|------------------------|
| Linear (1st) | `x₁` | 5 | 5 |
| Quadratic (2nd) | `x₁×x₂` | 5 + 10 = **15** | 15 |
| Cubic (3rd) | `x₁×x₂×x₃` | 15 + 10 = **25** | 25 |
| Quartic (4th) | `x₁×x₂×x₃×x₄` | 25 + 5 = **30** | 30 |
| Quintic (5th) | `x₁×x₂×x₃×x₄×x₅` | 30 + 1 = **31** | 31 |

**Problem**: As model order increases:
- Number of parameters grows exponentially
- Required experimental points increase significantly
- Risk of overfitting increases
- Computational complexity increases

---

## What Happens When You Have Higher-Order Terms?

When the **true function** has higher-order terms but you fit a **quadratic model**:

1. **Model Bias**: The higher-order terms act as "noise" from the model's perspective
2. **Coefficient Distortion**: The estimated linear and quadratic coefficients try to approximate the combined effect
3. **Reduced R²**: The model cannot fully explain the variance (R² will be lower than ideal)
4. **Prediction Errors**: Predictions will have systematic errors in regions where higher-order terms are important

### Example from Your Test Output:
```
🎪 CHALLENGING TEST:
   ✗ True function has up to 5th-order interactions
   ✗ Model only captures up to 2nd order (quadratic)
   ✗ Higher-order terms create 'noise' in lower-order estimates
   ✓ Sequential algorithm still finds significant patterns
   ✓ Prediction accuracy: 2.45 avg error
```

The model can still make **useful predictions** but cannot **exactly recover** the true coefficients.

---

## Solutions

### Option 1: Accept the Limitation (Current Approach)
- Use quadratic models for most practical applications
- Recognize that higher-order effects are absorbed into estimation error
- Focus on prediction accuracy rather than perfect coefficient recovery

**When this works:**
- Higher-order effects are small
- You only care about prediction, not interpretation
- Experimental budget is limited

### Option 2: Implement Higher-Order Models (Requires Code Changes)

You would need to modify `sequential_regression_reconstruction.py`:

```python
def _build_mixture_cubic_matrix(self, design: np.ndarray) -> np.ndarray:
    """Build design matrix for mixture cubic model"""
    X = [design]  # Linear terms
    n_comp = design.shape[1]
    
    # 2-way interactions
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            X.append((design[:, i] * design[:, j]).reshape(-1, 1))
    
    # 3-way interactions (NEW)
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            for k in range(j + 1, n_comp):
                X.append((design[:, i] * design[:, j] * design[:, k]).reshape(-1, 1))
    
    return np.column_stack(X)
```

Then update `analyze_current_data()`:
```python
if self.config.model_type == "cubic":
    # Fit cubic model (25 parameters for 5 components)
    X_cubic = self._build_mixture_cubic_matrix(self.current_design)
    model_results = self._fit_model(X_cubic, self.current_responses)
```

**Challenges:**
- Need many more experimental points (25+ for cubic, 30+ for quartic)
- Risk of overfitting
- Longer computation times
- May not converge with sequential approach

### Option 3: Use Specialized Higher-Order Designs
- Create D-optimal designs specifically for higher-order models
- Use model selection (start with quadratic, test if higher-order terms are needed)
- Employ regularization (Ridge/Lasso) to handle the large number of parameters

---

## Practical Recommendations

1. **For most applications**: Quadratic models are sufficient
   - Most real-world mixture effects are dominated by linear and 2-way interactions
   - Higher-order effects are often small and hard to distinguish from noise

2. **To detect higher-order effects**:
   - Look at R² values: if R² is low (<0.80) despite good data, higher-order terms may exist
   - Examine residual patterns: systematic residuals suggest missing model terms
   - Use lack-of-fit tests

3. **If you must estimate higher-order terms**:
   - Use a classical factorial or mixture design with enough points upfront
   - Don't rely on sequential building for very high-order models
   - Consider methods like Kriging or Gaussian Processes as alternatives

4. **Best practice**:
   - Start with quadratic model
   - If inadequate, add specific higher-order terms based on domain knowledge
   - Use cross-validation to prevent overfitting

---

## Conclusion

**You cannot estimate high-order interactions because the implementation only supports quadratic (2nd-order) models.** This is:
- ✅ By design (not a bug)
- ✅ Practical for most applications
- ✅ Clearly documented in the code comments

To estimate higher-order interactions, you would need to:
1. Implement higher-order design matrix construction
2. Modify the model fitting logic
3. Use designs with sufficient experimental points
4. Accept longer computation times and risk of overfitting

For your specific tests showing 3rd, 4th, and 5th order terms: **these are intentionally created to demonstrate the model's limitations**, not because they're expected in typical mixture experiments.
