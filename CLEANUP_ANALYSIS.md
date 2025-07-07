# 🧹 Project Cleanup Analysis & Plan

## Summary
- **Total Files**: ~110 files  
- **Safe to Delete**: ~25 files (22% reduction)
- **Candidates for Refactoring**: ~15 files
- **Estimated Size Reduction**: 30-40%

---

## 🗑️ FILES TO DELETE IMMEDIATELY

### 1. Obsolete Core Files
```bash
# Old versions replaced by newer implementations
rm src/core/optimal_design_generator_old.py  # Replaced by optimal_design_generator.py
rm streamlit_app.py                          # Use src/apps/streamlit_app.py instead
rm mixture_designs.py                        # Functionality moved to src/core/
```

### 2. Debug Files (Keep only if actively debugging)
```bash
rm debug_d_efficiency.py
rm debug_init.py
rm debug_coordinate_exchange.py
rm debug_d_optimal_algorithm.py
rm debug_quadratic_selection.py
rm debug_fixed_space.py
rm force_reload_streamlit.py
```

### 3. Redundant Test Files (D-Efficiency focused)
```bash
# These all test similar D-efficiency functionality - consolidate
rm test_d_efficiency_issue.py
rm test_d_efficiency_fixed.py
rm test_d_efficiency_real_data.py
rm test_d_efficiency_fix.py
rm test_i_optimal_d_efficiency.py
```

### 4. Investigative/Temporary Files
```bash
rm investigate_d_efficiency_issue.py
rm investigate_parts_mode_issue.py
rm compare_implementations.py
rm compare_approaches.py
rm compare_mixture_approaches.py
rm analyze_simplex_lattice_zeros.py
```

### 5. One-off Test/Demo Files
```bash
rm test_import_debug.py
rm test_fixed_space_solution.py
rm show_design_matrix.py
rm simple_test.py
rm demo_flexible_runs.py
```

---

## 🔄 REFACTORING OPPORTUNITIES

### 1. Mixture Design Classes (HIGH PRIORITY)

**Problem**: Too many similar classes with overlapping functionality

**Current Structure:**
```
src/core/mixture_designs.py                    # MixtureDesign, EnhancedMixtureDesign
src/core/fixed_parts_mixture_designs.py        # FixedPartsMixtureDesign
src/core/anti_clustering_mixture_design.py     # AntiClusteringMixtureDesign
src/core/correct_fixed_components_mixture.py   # CorrectFixedComponentsMixture
src/core/true_fixed_components_mixture.py      # TrueFixedComponentsMixture
src/core/improved_fixed_parts_mixture.py       # ImprovedFixedPartsMixture
src/core/simplified_mixture_design.py          # 7 different classes!
```

**Proposed Unified Structure:**
```
src/core/
├── mixture_base.py                 # Keep - Abstract base class
├── mixture_design.py               # Unified standard mixture designs
├── optimal_mixture_design.py       # D-optimal, I-optimal with all options
└── specialized/
    ├── fixed_parts_mixture.py      # Consolidate all fixed parts variants
    ├── anti_clustering_mixture.py  # Move from root
    └── constrained_mixture.py      # Extreme vertices, bounds
```

### 2. Test File Consolidation (MEDIUM PRIORITY)

**Current**: 40+ scattered test files
**Proposed**: Organized by functionality

```
tests/
├── unit/
│   ├── test_mixture_designs.py     # All basic mixture design tests
│   ├── test_optimal_designs.py     # D-optimal, I-optimal tests
│   ├── test_d_efficiency.py        # Consolidate ALL d-efficiency tests
│   └── test_parts_mode.py          # All parts mode tests
├── integration/
│   ├── test_streamlit_app.py       # UI integration tests
│   └── test_complete_workflows.py  # End-to-end tests
└── performance/
    └── test_design_comparison.py   # Performance comparisons
```

### 3. Utilities Consolidation

**Current**:
```
src/utils/d_efficiency_calculator.py
src/utils/mixture_utils.py
src/utils/response_analysis.py
src/utils/sequential_doe.py
src/utils/sequential_mixture_doe.py  # Duplicate of above?
```

**Proposed**:
```
src/utils/
├── design_metrics.py        # D-efficiency, I-efficiency calculations
├── mixture_utils.py         # Keep - mixture-specific utilities  
├── export_utils.py          # Excel export, formatting
└── visualization_utils.py   # Plotting functions
```

---

## 📊 DUPLICATION ANALYSIS

### 1. D-Efficiency Calculation
**Found in**: 5+ different files with similar implementations
**Solution**: Use `src/utils/d_efficiency_calculator.py` everywhere

### 2. Optimal Design Generation
**Found in**: 
- `optimal_design_generator.py` (current)
- `optimal_design_generator_old.py` (DELETE)
- Multiple mixture design classes

**Solution**: Standardize on current implementation

### 3. Fixed Components Logic
**Found in**:
- `fixed_parts_mixture_designs.py`
- `correct_fixed_components_mixture.py` 
- `true_fixed_components_mixture.py`
- `improved_fixed_parts_mixture.py`

**Solution**: Consolidate into single robust implementation

### 4. Matrix Calculations
**Found in**: Multiple files implementing gram_matrix, determinant, etc.
**Solution**: Create `src/utils/matrix_utils.py`

---

## 🎯 IMPLEMENTATION STEPS

### Phase 1: Safe Deletions (Low Risk)
1. Delete debug files
2. Delete old/obsolete files  
3. Delete redundant test files
4. **Estimated cleanup**: 20-25 files

### Phase 2: Test Consolidation (Medium Risk)
1. Create new organized test structure
2. Migrate important test cases
3. Delete old test files
4. **Estimated cleanup**: 15-20 files

### Phase 3: Core Refactoring (High Risk - Requires Testing)
1. Create unified mixture design classes
2. Consolidate utilities
3. Update imports throughout codebase
4. **Estimated cleanup**: 10-15 files

---

## ⚠️ SAFETY NOTES

### Before Deleting:
1. **Commit current state** to git
2. **Run existing tests** to ensure they pass
3. **Check for external dependencies** on files

### High-Risk Files (Check Before Deleting):
- Any file imported by `streamlit_app.py`
- Any file in active development
- Files with recent commits

### Backup Strategy:
```bash
# Create cleanup branch
git checkout -b project-cleanup

# Create backup of current state
git tag backup-before-cleanup

# Proceed with deletions
```

---

## 🔧 AUTOMATED CLEANUP SCRIPT

```bash
#!/bin/bash
# Phase 1: Safe deletions

echo "🧹 Starting Phase 1: Safe Deletions"

# Debug files
rm -f debug_*.py
rm -f force_reload_streamlit.py

# Old/obsolete files  
rm -f src/core/optimal_design_generator_old.py
rm -f streamlit_app.py
rm -f mixture_designs.py

# Investigative files
rm -f investigate_*.py
rm -f compare_*.py
rm -f analyze_*.py

# One-off files
rm -f test_import_debug.py
rm -f show_design_matrix.py
rm -f simple_test.py

echo "✅ Phase 1 complete. Removed ~15-20 files."
echo "⚠️  Run tests before proceeding to Phase 2"
```

---

## 📈 EXPECTED BENEFITS

### File Organization:
- **Before**: 110+ scattered files
- **After**: ~70 well-organized files
- **Reduction**: 35-40%

### Code Maintainability:
- Unified mixture design architecture
- Consolidated test suite
- Clear separation of concerns
- Reduced duplication

### Developer Experience:
- Easier to find relevant code
- Fewer import conflicts
- Clearer documentation
- Better test coverage

---

## 🚀 NEXT STEPS

1. **Review this analysis** - Confirm which files are safe to delete
2. **Create backup** - Use git tag/branch before changes  
3. **Phase 1 execution** - Start with safe deletions
4. **Test validation** - Ensure everything still works
5. **Phase 2 & 3** - Proceed with refactoring if needed

**Would you like me to start with Phase 1 (safe deletions) or would you prefer to review specific files first?**
