# Mixture Design of Experiments (DOE)

This package provides tools for creating and analyzing mixture designs of experiments, including support for fixed components and parts-based formulations.

## Overview

Mixture designs are a special type of experimental design where the factors are components of a mixture, and their sum must equal 1 (or 100%). This package provides various methods for generating mixture designs, including:

- Simplex Lattice designs
- Simplex Centroid designs
- Extreme Vertices designs
- D-optimal designs
- I-optimal designs
- Fixed Parts designs

## File Structure

The package is organized into the following modules:

- `base_doe.py`: Base Design of Experiments functionality
- `mixture_utils.py`: Utility functions for mixture designs
- `mixture_base.py`: Base MixtureDesign class
- `mixture_algorithms.py`: Implementation of classical mixture design algorithms
- `optimal_mixture_doe.py`: Implementation of optimal mixture designs
- `fixed_parts_mixture_designs.py`: Implementation of fixed parts mixture designs
- `mixture_designs.py`: Main interface for all mixture design functionality

## Usage

### Basic Usage

```python
from mixture_designs import MixtureDesignGenerator

# Create a simplex lattice design
n_components = 3
component_names = ["A", "B", "C"]
component_bounds = [(0.1, 0.8), (0.1, 0.7), (0.1, 0.6)]

design, mixture_design = MixtureDesignGenerator.create_simplex_lattice(
    n_components=n_components,
    degree=2,
    component_names=component_names,
    component_bounds=component_bounds
)

# Export design to CSV
MixtureDesignGenerator.export_design_to_csv(design, mixture_design, "simplex_lattice_design.csv")
```

### D-Optimal Design

```python
# Create D-optimal design
n_components = 4
component_names = ["W", "X", "Y", "Z"]
component_bounds = [(0.2, 0.8), (0.1, 0.5), (0.05, 0.3), (0.05, 0.2)]

design, mixture_design = MixtureDesignGenerator.create_d_optimal(
    n_components=n_components,
    n_runs=10,
    component_names=component_names,
    component_bounds=component_bounds,
    model_type="quadratic",
    random_seed=42
)
```

### Fixed Parts Design

```python
# Create fixed parts design
n_components = 5
component_names = ["A", "B", "C", "D", "E"]
component_bounds = [(5, 20), (10, 30), (5, 15), (1, 5), (0.5, 2)]
fixed_components = {"D": 3, "E": 1}

design, mixture_design = MixtureDesignGenerator.create_fixed_parts_design(
    n_components=n_components,
    n_runs=8,
    component_names=component_names,
    component_bounds=component_bounds,
    fixed_components=fixed_components,
    design_type="d-optimal",
    model_type="quadratic",
    random_seed=42
)
```

## Features

### Mixture Design Types

- **Simplex Lattice**: Creates a design with evenly spaced points on the simplex
- **Simplex Centroid**: Creates a design with points at the centroid of each face of the simplex
- **Extreme Vertices**: Creates a design with points at the extreme vertices of the constrained region
- **D-Optimal**: Creates a design that maximizes the determinant of the information matrix
- **I-Optimal**: Creates a design that minimizes the average prediction variance

### Fixed Components Support

The package supports fixed components, where certain components are held constant across all runs. This is useful for formulations where some ingredients must be present at a fixed level.

### Parts Mode

The package supports "parts mode" where components are specified in parts rather than proportions. This is useful for formulations where the total amount can vary, but the relative proportions must sum to 1.

## Example

See `example_usage.py` for complete examples of how to use the package.
