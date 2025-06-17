# Optimal Design of Experiments (DOE) for n Factors and m Responses

This repository provides comprehensive implementations for generating optimal experimental designs for any number of factors (n) and analyzing multiple responses (m). The code includes both Python and R implementations with D-optimal and I-optimal design generation capabilities.

## Files Overview

### Core Implementation Files
- **`optimal_doe_python.py`** - Complete Python implementation with OptimalDOE class
- **`optimal_doe_r.R`** - Complete R implementation using AlgDesign package
- **`example_usage.py`** - Practical example demonstrating 4 factors, 3 responses
- **`README.md`** - This documentation file

## Key Features

### Design Types Supported
- **D-optimal designs** - Maximize determinant of information matrix (best for parameter estimation)
- **I-optimal designs** - Minimize average prediction variance (best for prediction)
- **Custom model formulas** - Linear, quadratic, cubic, or custom polynomial models
- **Flexible factor ranges** - Any number of factors with custom ranges

### Multiple Response Analysis
- Analyze multiple response variables simultaneously
- Compare design performance across different responses
- Statistical summary and optimization for each response
- Export results to CSV for further analysis

## Quick Start

### Python Implementation

```python
from optimal_doe_python import OptimalDOE, multiple_response_analysis

# Define experimental setup
n_factors = 3
factor_ranges = [(-1, 1), (-2, 2), (0, 10)]  # Custom ranges for each factor

# Initialize DOE generator
doe = OptimalDOE(n_factors, factor_ranges)

# Generate D-optimal design (best for parameter estimation)
d_optimal_design = doe.generate_d_optimal(n_runs=20, model_order=2, random_seed=42)

# Generate I-optimal design (best for prediction)
i_optimal_design = doe.generate_i_optimal(n_runs=20, model_order=2, random_seed=42)

# Evaluate design quality
d_results = doe.evaluate_design(d_optimal_design, model_order=2)
print(f"D-efficiency: {d_results['d_efficiency']:.4f}")
print(f"I-efficiency: {d_results['i_efficiency']:.4f}")

# Define response functions
def response1(X):
    return X[:, 0]**2 + X[:, 1] + X[:, 2] + noise

def response2(X):
    return X[:, 0]*X[:, 1] + X[:, 2]**2 + noise

# Analyze multiple responses
designs = [d_optimal_design, i_optimal_design]
design_names = ['D-optimal', 'I-optimal']
response_functions = [response1, response2]

comparison = multiple_response_analysis(designs, response_functions, design_names)
print(comparison)
```

### R Implementation

```r
source("optimal_doe_r.R")

# Generate D-optimal design
d_optimal_result <- generate_optimal_doe(
  n_factors = 3,
  n_runs = 20,
  factor_ranges = list(c(-1, 1), c(-2, 2), c(0, 10)),
  criterion = "D"
)

# Generate I-optimal design
i_optimal_result <- generate_optimal_doe(
  n_factors = 3,  
  n_runs = 20,
  factor_ranges = list(c(-1, 1), c(-2, 2), c(0, 10)),
  criterion = "I"
)

# View results
print(d_optimal_result$design)
cat("D-efficiency:", d_optimal_result$d_efficiency, "\n")
cat("I-efficiency:", d_optimal_result$i_efficiency, "\n")
```

## Detailed Usage

### Python Classes and Methods

#### OptimalDOE Class
```python
class OptimalDOE:
    def __init__(n_factors, factor_ranges=None)
    def generate_d_optimal(n_runs, model_order=2, max_iter=1000, random_seed=None)
    def generate_i_optimal(n_runs, model_order=2, max_iter=1000, random_seed=None)
    def evaluate_design(X, model_order=2)
    def plot_design_2d(X, title="Experimental Design")  # For 2 factors only
```

#### Key Parameters
- **n_factors**: Number of experimental factors
- **factor_ranges**: List of (min, max) tuples for each factor
- **n_runs**: Number of experimental runs to generate
- **model_order**: 1 for linear, 2 for quadratic models
- **criterion**: "D" for D-optimal, "I" for I-optimal (R implementation)

### R Functions

#### Main Functions
```r
generate_optimal_doe(n_factors, n_runs, factor_ranges, model_formula, criterion)
evaluate_design(design, model_formula)
multiple_response_analysis(designs, response_functions, design_names)
compare_designs(designs, design_names, model_formula)
plot_design_2d(design, title, factor_names)
```

#### Utility Functions
```r
generate_fractional_factorial(n_factors, resolution)
add_center_points(design, n_center)
```

## Response Analysis Framework

### Complete Workflow: Design → Experiment → Analyze
After generating your optimal design and collecting experimental data, use our comprehensive response analysis system to extract insights and optimize your process.

#### Additional Files for Response Analysis
- **`response_analysis.py`** - Complete statistical analysis framework
- **`tutorial_response_analysis.py`** - Step-by-step tutorial with real examples

### Loading Data from Files

Most commonly, you'll have your DOE design and experimental results in spreadsheet files. Here's how to load and prepare your data:

#### From CSV Files

```python
import pandas as pd
import numpy as np
from response_analysis import DOEResponseAnalysis, MixtureResponseAnalysis

# Option 1: DOE design and responses in separate files
design_df = pd.read_csv('doe_design.csv')
responses_df = pd.read_csv('experimental_results.csv')

# Extract design matrix and responses
experimental_design = design_df[['Temperature', 'Pressure', 'pH']].values
experimental_responses = responses_df['Yield'].values

# Option 2: Everything in one file
data_df = pd.read_csv('complete_experimental_data.csv')
# Example file structure:
# Run, Temperature, Pressure, pH, Yield, Purity, Cost
# 1,    80,         2,        6,   72.5,  98.2,   45.3
# 2,    120,        2,        6,   78.2,  97.8,   48.1

experimental_design = data_df[['Temperature', 'Pressure', 'pH']].values
experimental_responses = data_df['Yield'].values  # or 'Purity', 'Cost', etc.
```

#### From Excel Files

```python
# Read from Excel
data_df = pd.read_excel('experimental_data.xlsx', sheet_name='Results')

# Extract design and responses
experimental_design = data_df[['Factor1', 'Factor2', 'Factor3']].values
experimental_responses = data_df['Response'].values

# Handle multiple responses
yield_responses = data_df['Yield'].values
purity_responses = data_df['Purity'].values
cost_responses = data_df['Cost'].values
```

#### Data Preparation

```python
# Convert to coded units if needed (for statistical analysis)
def convert_to_coded_units(real_values, low_value, high_value):
    """Convert real factor values to coded units (-1 to +1)"""
    center = (high_value + low_value) / 2
    half_range = (high_value - low_value) / 2
    return (real_values - center) / half_range

# Example: Convert temperature from 80-120°C to -1 to +1
temp_coded = convert_to_coded_units(experimental_design[:, 0], 80, 120)
press_coded = convert_to_coded_units(experimental_design[:, 1], 2, 4)
pH_coded = convert_to_coded_units(experimental_design[:, 2], 6, 8)

coded_design = np.column_stack([temp_coded, press_coded, pH_coded])
```

### Quick Start: Response Analysis

```python
from response_analysis import DOEResponseAnalysis, MixtureResponseAnalysis

# After loading your data from files (see above)
# experimental_design and experimental_responses are now loaded

# Create analyzer
analyzer = DOEResponseAnalysis(
    experimental_design, experimental_responses,
    factor_names=['Temperature', 'Pressure', 'pH'],
    response_name='Yield (%)'
)

# Fit statistical models
linear_results = analyzer.fit_linear_model()
quad_results = analyzer.fit_quadratic_model()

print(f"Linear Model R² = {linear_results['r_squared']:.4f}")
print(f"Quadratic Model R² = {quad_results['r_squared']:.4f}")

# Statistical analysis
anova_table = analyzer.anova_analysis(linear_results)
effects = analyzer.effects_analysis(linear_results)
significant_factors = effects[effects['Significant'] == True]

print("Significant factors (p < 0.05):")
print(significant_factors)

# Optimize your process
bounds = [(-1, 1), (-1, 1), (-1, 1)]  # Factor ranges in coded units
opt_result = analyzer.optimize_response(bounds, maximize=True)

if opt_result['success']:
    print("Optimal conditions:")
    for factor, value in opt_result['optimal_factors'].items():
        print(f"  {factor}: {value:.3f}")
    print(f"Predicted optimal response: {opt_result['optimal_response']:.2f}")

# Predict new conditions
new_conditions = np.array([[0.5, -0.5, 0.2], [-0.3, 0.8, -0.1]])
predictions = analyzer.predict_response(new_conditions)
print(f"Predictions: {predictions}")
```

### Mixture Design Analysis

```python
# For mixture experiments (components sum to 100%)
mixture_design = np.array([
    [0.70, 0.20, 0.10],  # 70% A, 20% B, 10% C
    [0.50, 0.40, 0.10],  # 50% A, 40% B, 10% C
    # ... more mixture compositions
])

mixture_responses = np.array([45.2, 52.1, 58.7, ...])  # Measured properties

mixture_analyzer = MixtureResponseAnalysis(
    mixture_design, mixture_responses,
    component_names=['Polymer_A', 'Polymer_B', 'Additive'],
    response_name='Tensile Strength (MPa)'
)

# Fit Scheffé mixture models
linear_results = mixture_analyzer.fit_scheffe_linear()
quad_results = mixture_analyzer.fit_scheffe_quadratic()

print(f"Linear Model R² = {linear_results['r_squared']:.4f}")
print("Pure component effects:")
for name, coeff in zip(linear_results['component_names'], linear_results['coefficients']):
    print(f"  {name}: {coeff:.2f} MPa")
```

### Complete Tutorial

Run the comprehensive tutorial to see the entire workflow:

```bash
python tutorial_response_analysis.py
```

This tutorial shows:
1. **Data Loading**: How to organize your experimental results
2. **Model Fitting**: Linear and quadratic model comparison
3. **Statistical Testing**: ANOVA and significance testing
4. **Factor Effects**: Identify which factors matter most
5. **Optimization**: Find optimal process conditions
6. **Prediction**: Estimate responses for new conditions
7. **Mixture Analysis**: Handle mixture design constraints

### Analysis Capabilities

#### Statistical Models
- **Linear models**: Y = β₀ + β₁X₁ + β₂X₂ + ... + ε
- **Quadratic models**: Includes interactions and squared terms
- **Scheffé mixture models**: Specialized for mixture experiments

#### Statistical Tests
- **ANOVA**: Test overall model significance
- **t-tests**: Individual factor significance (p-values)
- **R² analysis**: Model fit quality assessment
- **Residual analysis**: Model assumption validation

#### Optimization Features
- **Response optimization**: Find conditions for maximum/minimum response
- **Multi-start optimization**: Robust global optimization
- **Constraint handling**: Factor bounds and mixture constraints
- **Prediction intervals**: Uncertainty quantification

#### Visualization
- **Diagnostic plots**: Residuals, Q-Q plots, scale-location
- **Response surfaces**: 3D visualization of factor effects
- **Effects plots**: Bar charts of factor importance

## Advanced Features

### Custom Model Formulas (R)
```r
# Linear model
model_formula <- ~ X1 + X2 + X3

# Quadratic model with interactions
model_formula <- ~ X1 + X2 + X3 + I(X1^2) + I(X2^2) + I(X3^2) + X1:X2 + X1:X3 + X2:X3

# Cubic model
model_formula <- ~ X1 + X2 + I(X1^2) + I(X2^2) + I(X1^3) + I(X2^3) + X1:X2
```

### Design Evaluation Metrics
- **D-efficiency**: Measure of parameter estimation precision
- **I-efficiency**: Measure of prediction accuracy
- **Condition number**: Measure of design stability
- **A-efficiency**: Measure based on trace criterion

### Multiple Response Analysis
The implementations can handle any number of responses by:
1. Defining response functions that take the design matrix as input
2. Comparing how different designs perform across all responses
3. Identifying optimal conditions for each response
4. Providing statistical summaries and comparisons

## Example Applications

### 1. Chemical Process Optimization
- **Factors**: Temperature, Pressure, Concentration, Time
- **Responses**: Yield, Purity, Cost
- **Goal**: Find conditions that maximize yield and purity while minimizing cost

### 2. Manufacturing Process
- **Factors**: Speed, Feed rate, Depth of cut, Tool angle
- **Responses**: Surface roughness, Tool wear, Power consumption
- **Goal**: Optimize multiple quality characteristics

### 3. Pharmaceutical Formulation
- **Factors**: Drug concentration, Excipient ratios, pH, Temperature
- **Responses**: Dissolution rate, Stability, Bioavailability
- **Goal**: Develop robust formulation with desired properties

## Installation Requirements

### Python
```bash
pip install numpy pandas matplotlib scipy scikit-learn
```

### R
```r
install.packages(c("AlgDesign", "rsm", "DoE.base", "ggplot2", "gridExtra", "dplyr"))
```

## Running the Examples

### Python Example
```bash
python example_usage.py
```

### R Example
```bash
Rscript optimal_doe_r.R
```

## Output Files

The examples generate:
- **d_optimal_results.csv** - Complete experimental design with predicted responses
- **d_optimal_design.png** - Visualization of D-optimal design (2D)
- **i_optimal_design.png** - Visualization of I-optimal design (2D)

## Key Advantages of Optimal Designs

1. **Efficiency**: Use fewer experimental runs than classical designs
2. **Flexibility**: Handle irregular experimental regions and constraints
3. **Optimization**: Tailored for specific objectives (parameter estimation vs. prediction)
4. **Robustness**: Better performance with limited experimental budget
5. **Scalability**: Handle any number of factors and responses

## Mathematical Background

### D-Optimal Criterion
Maximizes det(X'X)^(1/p) where X is the model matrix and p is the number of parameters.
- Minimizes generalized variance of parameter estimates
- Best for parameter estimation accuracy

### I-Optimal Criterion
Minimizes average prediction variance over the design space.
- Optimizes prediction accuracy
- Best for response surface exploration

## Tips for Use

1. **Choose criterion based on objective**:
   - D-optimal: Parameter estimation, model fitting
   - I-optimal: Prediction, response surface mapping

2. **Model order selection**:
   - Linear (order=1): For screening experiments
   - Quadratic (order=2): For optimization studies

3. **Number of runs**:
   - Minimum: Number of model parameters
   - Recommended: 1.5-2 times number of parameters

4. **Factor ranges**:
   - Use actual experimental ranges
   - Consider practical constraints

## Contributing

Feel free to extend the code for:
- Additional optimality criteria (A-optimal, G-optimal)
- Mixture designs
- Block designs
- Sequential design strategies
- Advanced visualization tools

## References

1. Atkinson, A. C., & Donev, A. N. (1992). Optimum Experimental Designs
2. Montgomery, D. C. (2017). Design and Analysis of Experiments
3. Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016). Response Surface Methodology
