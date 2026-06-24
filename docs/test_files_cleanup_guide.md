# Test Files Cleanup Guide

## Files to KEEP (Core Final Solution)

### Essential Test Files:
1. **`test_monte_carlo_recovery.py`** ✅ KEEP
   - Main Monte Carlo simulation (100 random hierarchical functions)
   - Tests backward vs forward selection
   - Shows 10 worst trials with detailed analysis
   - Includes worst prediction points with true vs predicted values
   - **This is the primary comprehensive test**

2. **`test_hierarchical_recovery.py`** ✅ KEEP
   - Single hierarchical function detailed test
   - Compares backward vs forward selection
   - Shows 10 worst predicted points in detail
   - Useful for quick single-case testing

3. **`generate_hierarchical_function.py`** ✅ KEEP (REQUIRED)
   - Generates realistic hierarchical test functions
   - Used by both test files above
   - Ensures hierarchy (heredity principle)
   - Guarantees 4-way interactions
   - **Required dependency - DO NOT DELETE**

### Diagnostic/Analysis Files:
4. **`diagnose_false_positives.py`** ✅ KEEP
   - Analyzes correlation patterns
   - Shows why false positives occur
   - Useful for understanding the problem
   - Referenced in documentation

## Files to DELETE (Intermediate/Exploratory)

### Superseded by Current Solution:
1. **`test_two_stage_selection.py`** ❌ DELETE
   - Original version with non-hierarchical function
   - **Superseded by** `test_hierarchical_recovery.py`

2. **`test_two_stage_selection_fixed.py`** ❌ DELETE
   - Intermediate version with hierarchical constraints
   - **Superseded by** `test_hierarchical_recovery.py`

### Exploratory Tests (No longer needed):
3. **`test_backward_with_practical_significance.py`** ❌ DELETE
   - Exploratory test for practical significance threshold
   - Findings integrated into main tests

4. **`test_proper_backward_elimination.py`** ❌ DELETE
   - Testing backward elimination variations
   - Not needed for final solution

5. **`test_relative_practical_significance.py`** ❌ DELETE
   - Testing relative significance thresholds
   - Exploratory, not part of final solution

6. **`test_parameter_recovery.py`** ❌ DELETE
   - Old parameter recovery test
   - **Superseded by** current hierarchical tests

7. **`test_different_regression_methods.py`** ❌ DELETE
   - Comparison of regression methods
   - Not relevant to final hierarchical testing

### Different Topics (Not related to false positive analysis):
8. **`test_d_optimal_comparison.py`** ❌ DELETE (unless needed for other work)
   - D-optimal design comparison
   - Different topic from false positive analysis

9. **`test_jmp_actual_design.py`** ❌ DELETE (unless needed for other work)
   - JMP-specific testing
   - Not related to false positive analysis

10. **`test_jmp_full_model.py`** ❌ DELETE (unless needed for other work)
    - JMP full model screening
    - Different methodology

11. **`test_jmp_style_complex.py`** ❌ DELETE (unless needed for other work)
    - JMP style testing
    - Not related to current solution

12. **`test_sparse_design_performance.py`** ❌ DELETE (unless needed for other work)
    - Performance testing for sparse designs
    - Different topic

## Final Directory Structure (Recommended)

```
DOE/
├── generate_hierarchical_function.py          ✅ KEEP
├── test_monte_carlo_recovery.py               ✅ KEEP
├── test_hierarchical_recovery.py              ✅ KEEP
├── diagnose_false_positives.py                ✅ KEEP
├── docs/
│   ├── false_positive_analysis.md             ✅ KEEP
│   ├── hierarchical_function_findings.md      ✅ KEEP
│   └── test_files_cleanup_guide.md            ✅ KEEP (this file)
└── [other project files...]
```

## Summary

**Keep:** 4 files (3 test files + 1 generator)
**Delete:** 12 files (exploratory/superseded tests)

The kept files provide:
- Comprehensive Monte Carlo testing (100 trials)
- Single-case detailed testing
- Hierarchical function generation
- Diagnostic analysis tools
- Complete documentation

All findings and best practices from the deleted exploratory tests have been incorporated into the final solution.
