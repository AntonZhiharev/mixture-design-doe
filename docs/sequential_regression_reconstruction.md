# Sequential Regression Coefficient Reconstruction

This document describes the `SequentialRegressionReconstructor` class, a comprehensive framework for sequentially reconstructing regression function coefficients using experimental design. The process involves iterative data point generation, analysis, parameter selection, and cycle repetition.

## Overview

The Sequential Regression Reconstruction methodology implements an adaptive, iterative approach to experimental design that:

1. **Generates initial design points** using optimal design strategies
2. **Collects experimental responses** for those points
3. **Analyzes current data** to fit regression models and assess quality
4. **Selects experimental parameters** for the next iteration based on current results
5. **Generates additional design points** using adaptive strategies
6. **Repeats the cycle** until convergence criteria are met

This approach is particularly powerful for:
- **Mixture experiments** where components must sum to 1
- **Factorial experiments** with continuous factors
- **Sequential learning** where each iteration informs the next
- **Coefficient reconstruction** with known precision requirements

## Key Features

### 🎯 **Adaptive Design Generation**
- D-optimal design selection for maximum information
- Space-filling designs for exploration
- Targeted designs for significant terms
- Model improvement strategies for poor fit regions

### 📊 **Comprehensive Analysis**
- Statistical model fitting (linear, quadratic)
- D-efficiency calculation and monitoring
- Model adequacy assessment
- Significance testing for all terms

### 🔄 **Intelligent Iteration**
- Multiple convergence criteria (R², D-efficiency, coefficient stability)
- Adaptive parameter selection based on current state
- Smart stopping rules to avoid over-experimentation

### 🧪 **Flexible Experimental Support**
- Mixture designs (Scheffé models)
- Factorial designs with interactions
- Fixed component constraints
- Both proportion and parts-mode operation

## Quick Start Example

```python
from src.algorithms.sequential_regression_reconstruction import (
    SequentialRegressionReconstructor, ReconstructionConfig
)
import numpy as np

# Define your response function
def my_response_function(point):
    x1, x2, x3 = point
    return 50*x1 + 80*x2 + 30*x3 + 25*x1*x2 + np.random.normal(0, 1.0)

# Configure the reconstruction
config = ReconstructionConfig(
    n_components=3,
    model_type="quadratic",
    max_iterations=10,
    initial_batch_size=8,
    sequential_batch_size=3,
    r2_threshold=0.90,
    d_efficiency_threshold=0.80
)

# Create and run reconstructor
reconstructor = SequentialRegressionReconstructor(config)
results = reconstructor.run_sequential_reconstruction(my_response_function)

# View results
print(f"Converged: {results['converged']}")
print(f"Final R²: {results['final_analysis']['model_results']['r_squared']:.3f}")

# Get coefficient summary
coeff_summary = reconstructor.get_coefficient_summary()
print(coeff_summary)
```

## Configuration Options

### `ReconstructionConfig` Parameters

#### **Design Parameters**
- `n_components`: Number of factors/components (default: 3)
- `model_type`: "linear" or "quadratic" (default: "quadratic")
- `max_iterations`: Maximum number of iterations (default: 20)
- `min_iterations`: Minimum iterations before convergence check (default: 3)

#### **Batch Sizes**
- `initial_batch_size`: Initial number of experiments (default: 10)
- `sequential_batch_size`: Additional experiments per iteration (default: 3)

#### **Convergence Criteria**
- `r2_threshold`: Minimum R² for convergence (default: 0.95)
- `d_efficiency_threshold`: Minimum D-efficiency (default: 0.85)
- `coefficient_tolerance`: Relative change tolerance (default: 0.05)
- `prediction_accuracy_threshold`: Maximum RMSE (default: 0.02)

#### **Strategy and Method**
- `design_strategy`: "d_optimal", "space_filling", or "adaptive" (default: "d_optimal")
- `regression_method`: "ols", "ridge", or "lasso" (default: "ols")
- `significance_level`: Statistical significance threshold (default: 0.05)

#### **Constraints (Optional)**
- `lower_bounds`: Minimum values for each factor
- `upper_bounds`: Maximum values for each factor  
- `fixed_components`: Dictionary of fixed component values

## Step-by-Step Process

### 1. Initial Design Generation

The reconstructor generates an initial experimental design using one of several strategies:

**For Mixture Designs:**
- Pure component vertices
- Binary mixture edges
- Centroid points
- D-optimal selection from candidates

**For Factorial Designs:**
- Factorial vertices (±1 levels)
- Center points
- Space-filling extensions
- D-optimal augmentation

### 2. Response Collection

- Calls your response function for each design point
- Accumulates responses across iterations
- Tracks response ranges and statistics

### 3. Data Analysis

**Model Fitting:**
- Mixture models: Scheffé linear/quadratic (no intercept)
- Factorial models: Standard polynomial with intercept
- Statistical inference with standard errors and p-values

**Quality Assessment:**
- R² and adjusted R² calculation
- D-efficiency of current design
- Residual analysis and normality testing
- Identification of significant terms

### 4. Parameter Selection

Based on current analysis results, selects strategy for next iteration:

- **Exploration**: When no significant terms found
- **D-optimal**: When design efficiency is low
- **Significant terms**: Focus on active factors
- **Model improvement**: Target high-residual regions

### 5. Additional Point Generation

Generates new experimental points using selected strategy:

**D-optimal Augmentation:**
```python
# Greedy selection maximizing determinant
for each candidate:
    test_design = [current_design + candidate]
    efficiency = calculate_d_efficiency(test_design)
    select_best_efficiency()
```

**Exploration Strategy:**
- Random space-filling points
- Ensures broad coverage of factor space
- Respects mixture/factorial constraints

**Significant Terms Strategy:**
- Emphasizes factors appearing in significant interactions
- Allocates higher proportions/levels to active components

### 6. Convergence Checking

Multiple criteria must be met for convergence:

1. **R² Adequacy**: `R² ≥ r2_threshold`
2. **Design Efficiency**: `D-efficiency ≥ d_efficiency_threshold`
3. **Model Adequacy**: Residuals pass normality tests
4. **Coefficient Stability**: Relative changes < `coefficient_tolerance`
5. **Prediction Accuracy**: RMSE ≤ `prediction_accuracy_threshold`

## Advanced Usage

### Mixture Design with Fixed Components

```python
config = ReconstructionConfig(
    n_components=4,
    fixed_components={3: 0.1},  # Component 4 fixed at 10%
    model_type="quadratic"
)

# The reconstructor automatically handles the constraint
# and normalizes remaining components appropriately
```

### Factorial Design with Bounds

```python
config = ReconstructionConfig(
    n_components=3,
    lower_bounds=[-2, -1, -1],
    upper_bounds=[2, 1, 1],
    model_type="quadratic"
)

# Factors will be restricted to specified ranges
# during candidate generation and selection
```

### Custom Convergence Criteria

```python
config = ReconstructionConfig(
    r2_threshold=0.98,         # Very high fit requirement
    d_efficiency_threshold=0.9, # Very efficient design
    coefficient_tolerance=0.01, # Very stable coefficients
    max_iterations=15          # Allow more iterations
)
```

### Manual Step-by-Step Execution

```python
reconstructor = SequentialRegressionReconstructor(config)

# Manual control of each step
initial_design = reconstructor.generate_initial_design()
initial_responses = reconstructor.collect_responses(initial_design, response_func)

for iteration in range(max_iterations):
    analysis_results = reconstructor.analyze_current_data()
    
    if reconstructor.check_convergence(analysis_results):
        break
        
    selected_params = reconstructor.select_experimental_parameters(analysis_results)
    additional_points = reconstructor.gen_additional_design_points(selected_params, analysis_results)
    additional_responses = reconstructor.collect_responses(additional_points, response_func)
    
    reconstructor.save_iteration_result(iteration, analysis_results, selected_params)
```

## Results and Analysis

### Accessing Results

```python
# Run complete reconstruction
results = reconstructor.run_sequential_reconstruction(response_function)

# Key result components
print(f"Converged: {results['converged']}")
print(f"Total iterations: {results['total_iterations']}")
print(f"Total experiments: {results['total_experiments']}")
print(f"Final coefficients: {results['final_coefficients']}")

# Detailed iteration history
for i, iteration_result in enumerate(reconstructor.iteration_history):
    print(f"Iteration {i+1}: R²={iteration_result.r_squared:.3f}, "
          f"D-eff={iteration_result.d_efficiency:.3f}")
```

### Coefficient Summary

```python
# Get detailed coefficient information
summary = reconstructor.get_coefficient_summary()
print(summary)

# Output includes:
# - Term names (x1, x2, x1*x2, etc.)
# - Coefficient values
# - Standard errors
# - Significance flags
# - Confidence intervals (if available)
```

### Visualization

```python
# Plot convergence history
reconstructor.plot_convergence()

# Shows:
# - R² evolution over iterations
# - D-efficiency improvement
# - RMSE reduction
# - Number of experiments by iteration
```

### Making Predictions

```python
# Predict responses for new points
new_points = np.array([
    [0.5, 0.3, 0.2],
    [0.2, 0.6, 0.2]  
])

predictions = reconstructor.predict_response(new_points)
print(f"Predicted responses: {predictions}")
```

### Saving and Loading

```python
# Save complete results to JSON
filename = reconstructor.save_results("my_experiment_results.json")

# Results include:
# - Configuration parameters
# - Convergence status
# - Iteration summaries
# - Significant terms found
# - Final metrics
```

## Design Strategies Explained

### D-Optimal Strategy

Selects points that maximize the determinant of the information matrix (X'X):

**Advantages:**
- Minimizes parameter variance
- Provides most precise coefficient estimates
- Well-established statistical theory

**Best for:**
- Final stages of experimentation
- When you need precise coefficients
- Confirmation of model structure

### Exploration Strategy

Uses space-filling approaches to broadly sample the factor space:

**Advantages:**
- Discovers unexpected regions of interest
- Avoids local minima in optimization
- Good for early exploration

**Best for:**
- Initial phases of experimentation
- When factor interactions are unknown
- Robust model building

### Significant Terms Strategy

Focuses on regions that activate statistically significant model terms:

**Advantages:**
- Efficiently estimates important effects
- Reduces experimental burden
- Targeted coefficient refinement

**Best for:**
- After initial screening
- When some effects are already identified
- Parameter refinement phases

## Common Use Cases

### 1. Mixture Formulation Optimization

```python
# Example: Paint formulation with 3 components
def paint_performance(composition):
    pigment, binder, solvent = composition
    # Complex mixture response surface
    return (80*pigment + 60*binder + 40*solvent + 
            50*pigment*binder + 30*pigment*solvent - 20*binder*solvent)

config = ReconstructionConfig(n_components=3, model_type="quadratic")
reconstructor = SequentialRegressionReconstructor(config)
results = reconstructor.run_sequential_reconstruction(paint_performance)
```

### 2. Chemical Process Optimization

```python
# Example: Reaction with temperature, pressure, pH
def reaction_yield(factors):
    temp, pressure, ph = factors  # Coded -1 to +1 levels
    return (75 + 10*temp + 5*pressure - 3*ph + 
            4*temp*pressure - 2*temp*ph + pressure*ph)

config = ReconstructionConfig(
    n_components=3, 
    lower_bounds=[-1, -1, -1], 
    upper_bounds=[1, 1, 1]
)
```

### 3. Sequential Screening and Optimization

```python
# Start with screening (linear model)
screening_config = ReconstructionConfig(model_type="linear", max_iterations=5)
screening_results = reconstructor.run_sequential_reconstruction(response_func)

# Then optimization (quadratic model) 
optimization_config = ReconstructionConfig(model_type="quadratic")
# Use screening results to focus optimization...
```

## Troubleshooting

### Common Issues

**1. Slow Convergence**
- Decrease convergence thresholds
- Increase `sequential_batch_size`
- Check if response function has sufficient signal-to-noise ratio

**2. Poor Model Fit**
- Increase `max_iterations`
- Check for outliers in response data
- Consider model_type="linear" if quadratic is unnecessary

**3. Design Generation Failures**
- Verify constraints are feasible
- Check that fixed components sum to < 1.0
- Ensure response function handles all valid points

**4. Import Errors**
- Ensure all dependencies are installed: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`
- Check that `src` folder is in Python path

### Performance Tips

**For Large Numbers of Components (n > 5):**
- Start with smaller `initial_batch_size`
- Use `model_type="linear"` initially
- Increase `coefficient_tolerance` for faster convergence

**For Noisy Response Functions:**
- Increase batch sizes for better statistics
- Lower convergence thresholds
- Consider response averaging across multiple evaluations

**For Expensive Experiments:**
- Set conservative thresholds to minimize total experiments
- Use `min_iterations` to ensure adequate exploration
- Monitor convergence plots to stop early if needed

## Mathematical Background

### Scheffé Mixture Models

For mixture experiments where components sum to 1:

**Linear Model:**
```
y = β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

**Quadratic Model:**
```  
y = Σβᵢxᵢ + ΣΣβᵢⱼxᵢxⱼ + ε
```

Where xᵢ ≥ 0 and Σxᵢ = 1.

### D-Efficiency Calculation

D-efficiency is calculated as:
```
D-efficiency = (|X'X|/n)^(1/p)
```

Where:
- |X'X| is the determinant of the information matrix
- n is the number of experimental runs  
- p is the number of model parameters

### Convergence Criteria

Multiple criteria ensure robust convergence:

1. **Statistical Adequacy**: R² ≥ threshold
2. **Design Optimality**: D-efficiency ≥ threshold  
3. **Parameter Stability**: |Δβᵢ/βᵢ| < tolerance
4. **Prediction Accuracy**: RMSE ≤ threshold
5. **Model Validity**: Residuals ~ Normal(0,σ²)

## References and Further Reading

1. **Mixture Experiments**: Cornell, J.A. "Experiments with Mixtures: Designs, Models, and the Analysis of Mixture Data"

2. **Optimal Design Theory**: Atkinson, A.C. & Donev, A.N. "Optimum Experimental Designs"

3. **Sequential Design**: Pronzato, L. & Pázman, A. "Design of Experiments in Nonlinear Models"

4. **D-Optimal Design**: Silvey, S.D. "Optimal Design: An Introduction to the Theory for Parameter Estimation"

## Example Applications

This methodology has been successfully applied to:

- **Pharmaceutical formulation** (tablet composition optimization)
- **Food product development** (flavor and texture optimization)  
- **Chemical process optimization** (reaction condition screening)
- **Materials science** (alloy composition studies)
- **Environmental engineering** (treatment process optimization)

The sequential approach typically reduces experimental effort by 30-50% compared to traditional factorial or mixture designs while achieving equivalent or better model precision.
