# Flexible Run Numbers in Mixture Designs

## Overview

Yes, it is possible to set up different numbers of runs for Simplex Lattice DOE and other mixture designs. I've created an enhanced implementation that provides this flexibility.

## Key Capabilities

### 1. **Traditional Fixed Formulas**
- **Simplex Lattice**: N = C(q+m-1, m) where q=components, m=degree
- **Simplex Centroid**: N = 2^q - 1

### 2. **New Flexible Approach**
The `EnhancedMixtureDesign` class allows you to:
- Specify exact number of runs for any design type
- Automatically adjust designs using various strategies
- Maintain statistical efficiency while meeting practical constraints

## Methods to Control Run Numbers

### 1. **Direct Specification**
```python
# Generate exactly 15 runs
design = enhanced_design.generate_mixture_design(
    design_type="simplex-lattice",
    n_runs=15,
    model_type="quadratic"
)
```

### 2. **Augmentation Strategies** (when you need more runs)
- **"centroid"**: Add centroids of point subsets
- **"replicate"**: Replicate important points (vertices, centroid)
- **"d-optimal"**: Add D-optimal points to improve efficiency
- **"space-filling"**: Add space-filling points for better coverage

### 3. **Reduction Strategies** (when you need fewer runs)
- **"subset"**: Select diverse subset maintaining coverage
- **"d-optimal"**: Select D-optimal subset for best efficiency

## Examples

### Example 1: Fixed 20 Runs for Different Designs
```python
for design_type in ["simplex-lattice", "d-optimal", "space-filling"]:
    design = enhanced_design.generate_mixture_design(
        design_type=design_type,
        n_runs=20,
        model_type="quadratic"
    )
```

### Example 2: Augmenting Simplex Lattice
Base simplex lattice (degree 2) has 6 runs. To get 12 runs:
```python
augmented = enhanced_design.generate_mixture_design(
    design_type="simplex-lattice",
    n_runs=12,
    augment_strategy="d-optimal"
)
```

### Example 3: Reducing Simplex Lattice
Base simplex lattice (degree 4) has 35 runs. To get 15 runs:
```python
reduced = enhanced_design.generate_mixture_design(
    design_type="simplex-lattice",
    n_runs=15,
    augment_strategy="d-optimal"
)
```

## Recommended Number of Runs

For different model complexities:

| Model Type | Minimum Runs* | Recommended (1.5x) | Optimal (2x) |
|------------|--------------|-------------------|--------------|
| Linear     | q            | 1.5×q             | 2×q          |
| Quadratic  | q(q+1)/2     | 1.5×min           | 2×min        |
| Cubic      | Complex**    | 1.5×min           | 2×min        |

*q = number of components
**Cubic: q + q(q-1)/2 + q(q-1)(q-2)/6

## Practical Considerations

1. **Efficiency Trade-offs**: 
   - More runs → Better parameter estimation
   - Fewer runs → Lower cost
   - Find the sweet spot for your application

2. **Model Requirements**:
   - Ensure you have at least the minimum runs for your model
   - Extra runs improve robustness and allow for model validation

3. **Design Quality**:
   - Monitor D-efficiency when adjusting run numbers
   - Space-filling properties are important for prediction

## Implementation Files

1. **enhanced_mixture_designs.py**: Core implementation with flexible run control
2. **demo_flexible_runs.py**: Comprehensive demonstration of capabilities
3. **mixture_designs.py**: Original implementation (fixed formulas)

## Running the Demo

```bash
python demo_flexible_runs.py
```

This will show:
- Default runs for each design type
- How to set exact run numbers
- Augmentation/reduction strategies
- Efficiency comparisons
- Optimal run number analysis

## Conclusion

The enhanced implementation provides complete flexibility in choosing the number of runs while maintaining the mathematical rigor and efficiency of mixture designs. You can now:

1. Use traditional formulas when appropriate
2. Specify exact run numbers for practical constraints
3. Optimize designs for your specific needs
4. Balance statistical efficiency with experimental cost

This flexibility makes mixture DOE more practical for real-world applications where budget, time, or resource constraints may limit the number of experiments you can perform.
