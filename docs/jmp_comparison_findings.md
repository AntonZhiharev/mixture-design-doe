# JMP vs Our Design: Findings and Improvements

## Summary

Analysis comparing JMP's MixtureDesigh45Runs1Order.xlsx design with our generated design revealed key differences in point replication strategy and design optimization that affect higher-order interaction detection.

## Key Findings

### 1. JMP Replication Strategy (CRITICAL)
JMP uses strategic replication with **13 replicated points** (29% of 45-run design):
- Centroid replicated **3 times** (1 base + 2 extra) - for pure error estimation
- Several vertex points replicated 2x
- Mid-edge points (0.5/0.5) replicated 2x  
- Some ternary points replicated

### 2. JMP Point Distribution
JMP's 45-run design has:
- **10 Vertices** (22%) - pure components, 4 zeros each
- **15 Edges** (33%) - binary mixtures, 3 zeros each
- **12 Ternary** (27%) - 3-component mixtures, 2 zeros each
- **5 Quaternary** (11%) - 4-component mixtures, 1 zero each
- **3 Interior** (7%) - all 5 components active

### 3. Sparsity
JMP achieves **82.2% sparsity** (37/45 points have 2+ zeros)

### 4. D-Optimal Selection
JMP uses **D-optimal coordinate exchange** to SELECT points from a candidate set, which:
- Minimizes multicollinearity between model terms
- Optimizes the condition number of the information matrix
- Enables reliable estimation of higher-order interactions

## Improvements Made

### 1. Enhanced Replication Strategy
- Increased replication to match JMP's ~29% rate (13 replicates for 45 runs)
- Added replication of DIFFERENT points (not same point multiple times)
- Proper tracking to avoid duplicate replicates

### 2. Quaternary Coverage
- Ensured **at least 5 quaternary points** (one per C(5,4) combination)
- This is critical for 4-way interaction detection

### 3. Fixed Vertex Generation
- Fixed issue where vertices were duplicated when n_target > n_components
- Now generates exactly n_components unique vertices
- Excess vertex budget redistributed to edges

### 4. Improved Point Tracking
- Use dict-based tracking to ensure unique point selection in replication phase
- Separate unique point for each type (vertex, edge, ternary)

## Detection Rates Comparison (After Improvements)

| Interaction Order | JMP Design | Our Design (Improved) |
|-------------------|------------|----------------------|
| Main Effects      | 100%       | **100%** ✓           |
| 2-Way            | 100%       | **100%** ✓           |
| 3-Way            | 100%       | 0%                   |
| 4-Way            | 0%         | 0%                   |
| **Total**        | **84.6%**  | **69.2%**            |

### Structure Now Matches JMP:
- ✅ Same replicated points: 13
- ✅ Same unique points: 32
- ✅ Similar sparsity: 80% vs 82%
- ✅ Centroid replicated 3x
- ✅ Mid-edges (0.5/0.5) replicated

## Remaining Gap: 3-Way Detection

### Root Cause
The primary difference between JMP detecting 100% of 3-way interactions vs our 0% is:

1. **D-Optimal Point Selection**: JMP uses coordinate exchange algorithm to SELECT the optimal subset of points from a large candidate set. This optimization specifically minimizes the condition number of the design matrix.

2. **Condition Number**: JMP has condition number ~1.33e18, our design ~1.52e18 (worse by ~14%)

3. **Numerical Stability**: Our design produces near-singular matrices during 3-way term fitting, causing the screening algorithm to fall back to Ridge regression with uncertain p-values.

### Why Pattern-Based Generation Isn't Enough
Our approach generates points based on JMP's observed patterns (sparsity, proportions) but:
- Pattern matching doesn't guarantee D-optimality
- The specific RATIOS in edge/ternary/quaternary points matter
- D-optimal selection finds the BEST subset, not just a good one

## Recommendations

### For Users Needing 3-Way Detection
1. **Use the Actual JMP Design** when 3-way+ interactions are critical
2. The JMP design (MixtureDesigh45Runs1Order.xlsx) is already optimized for this

### For Our Design
1. Add **D-optimal refinement** as a post-processing step
2. Generate candidate pool → Generate initial design → D-optimal coordinate exchange
3. This would use our `MixtureDOptimalAlgorithm.optimize_mixture_design()` method

### Future Enhancement
Implement D-optimal point selection:
```python
# Pseudocode for D-optimal enhanced generation
1. Generate all candidate points (vertices, edges, ternary, quaternary)
2. Use D-optimal coordinate exchange to select best n_runs points
3. Add strategic replication to selected points
```

## Conclusion

Our improved design now matches JMP's structure (sparsity, replication, point types) but lacks D-optimal selection. For 1-2 way interactions, our design performs equally well. For 3-way+ interactions, JMP's D-optimal selection provides a significant advantage.

The fundamental limitation is that **pattern-based generation cannot match D-optimal selection** without implementing the actual optimization algorithm as a refinement step.
