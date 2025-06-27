# Getting Started with Mixture Design Package

This guide will help you get started with the Mixture Design package.

## Installation

Currently, the package is not available on PyPI, so you'll need to use it directly from the source code.

1. Make sure all the required files are in the same directory:
   - `base_doe.py`
   - `mixture_utils.py`
   - `mixture_base.py`
   - `mixture_algorithms.py`
   - `optimal_mixture_doe.py`
   - `fixed_parts_mixture_designs.py`
   - `mixture_designs.py`
   - `__init__.py`

2. Install the required dependencies:
   ```
   pip install numpy pandas matplotlib
   ```

## Quick Start

The easiest way to get started is to run the example script:

```
python example_usage.py
```

This will run several examples showing different types of mixture designs.

## Creating Your First Mixture Design

Here's a simple example to create a simplex lattice design:

```python
from mixture_designs import MixtureDesignGenerator

# Define your mixture components
n_components = 3
component_names = ["A", "B", "C"]
component_bounds = [(0.1, 0.8), (0.1, 0.7), (0.1, 0.6)]

# Create a simplex lattice design
design, mixture_design = MixtureDesignGenerator.create_simplex_lattice(
    n_components=n_components,
    degree=2,
    component_names=component_names,
    component_bounds=component_bounds
)

# Print the design
print("\nSimplex Lattice Design:")
for i, point in enumerate(design):
    print(f"Run {i+1}: {', '.join([f'{name}={val:.3f}' for name, val in zip(component_names, point)])}")

# Export the design to CSV
MixtureDesignGenerator.export_design_to_csv(design, mixture_design, "my_first_design.csv")
```

## Creating an Optimal Design

For more advanced designs, you can create D-optimal or I-optimal designs:

```python
# Create a D-optimal design
design, mixture_design = MixtureDesignGenerator.create_d_optimal(
    n_components=3,
    n_runs=8,
    component_names=["A", "B", "C"],
    component_bounds=[(0.1, 0.8), (0.1, 0.7), (0.1, 0.6)],
    model_type="quadratic",
    random_seed=42
)
```

## Working with Fixed Components

If you need to keep some components fixed across all runs:

```python
# Create a design with fixed components
design, mixture_design = MixtureDesignGenerator.create_fixed_parts_design(
    n_components=5,
    n_runs=8,
    component_names=["A", "B", "C", "D", "E"],
    component_bounds=[(5, 20), (10, 30), (5, 15), (1, 5), (0.5, 2)],
    fixed_components={"D": 3, "E": 1},
    design_type="d-optimal",
    model_type="quadratic"
)
```

## Next Steps

- Check the `README.md` file for more detailed information
- Look at `example_usage.py` for more examples
- Explore the API documentation in the source code
