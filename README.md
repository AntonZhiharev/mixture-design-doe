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
