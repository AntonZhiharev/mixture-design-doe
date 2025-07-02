# Proportional Parts Mixture Solution

## Problem Statement

In parts mode, when boundaries are set for components, there was an issue where candidate points did not maintain proper proportional relationships between components. The issue occurred when:

1. Components had different parts ranges (e.g., Component A: 0.1-5.0, Component B: 0.2-3.0, Component C: 0.1-2.0)
2. Candidate point generation treated each component independently
3. The conversion between parts space and proportional space didn't maintain proper relationships
4. Boundaries were not properly evaluated for each component in proportion

## Solution Overview

The solution involves implementing a **ProportionalPartsMixture** approach that:

1. **Converts parts space to proportional space** while maintaining relationships
2. **Evaluates boundaries for each component in proper proportion**
3. **Ensures candidate points maintain proportional relationships**
4. **Integrates seamlessly with existing design generators**

## Key Components

### 1. ProportionalPartsMixture Class (`src/core/proportional_parts_mixture.py`)

This new class handles the core functionality:

```python
class ProportionalPartsMixture:
    def __init__(self, n_components, component_ranges, fixed_components=None):
        # Calculate proportional ranges that maintain proper relationships
        self.proportional_ranges = self._calculate_proportional_ranges()
    
    def _calculate_proportional_ranges(self):
        # Convert parts ranges to proportional ranges while ensuring
        # sum constraint is satisfied and relationships are maintained
        
    def generate_proportional_candidate(self):
        # Generate candidate points that maintain proportional relationships
        
    def convert_proportions_to_parts(self, proportions):
        # Convert proportional candidates to parts while respecting boundaries
        
    def validate_parts_candidate(self, parts_values):
        # Validate that parts values respect all boundaries
```

### 2. Enhanced OptimalDesignGenerator (`src/core/optimal_design_generator.py`)

The optimal design generator now includes:

```python
def __init__(self, ..., component_ranges=None):
    # Initialize proportional parts mixture helper if component ranges provided
    self.proportional_parts_helper = None
    if self.component_ranges and self.design_type == "mixture":
        self.proportional_parts_helper = ProportionalPartsMixture(
            n_components=num_variables,
            component_ranges=component_ranges
        )

def _generate_candidate_point(self):
    if self.design_type == "mixture":
        # Use proportional parts mixture helper if available
        if self.proportional_parts_helper is not None:
            _, proportions = self.proportional_parts_helper.generate_feasible_parts_candidate()
            return proportions
        # Fallback to standard method
```

### 3. Integration with SimplifiedMixtureDesign

The existing `DOptimalMixtureDesign` class automatically benefits from the fix when using parts mode with component bounds.

## How It Works

### Step 1: Proportional Range Calculation

When component parts ranges are specified, the system calculates feasible proportional ranges:

```
Component 1: parts [0.100, 5.000] → proportions [0.0196, 0.9434]
Component 2: parts [0.200, 3.000] → proportions [0.0278, 0.9375] 
Component 3: parts [0.100, 2.000] → proportions [0.0123, 0.8696]
```

### Step 2: Candidate Generation in Proportional Space

Candidates are generated within these proportional ranges, then normalized to sum=1:

```python
def generate_proportional_candidate(self):
    candidate = []
    for i, (min_prop, max_prop) in enumerate(self.proportional_ranges):
        prop = random.uniform(min_prop, max_prop)
        candidate.append(prop)
    
    # Normalize to sum = 1
    total = sum(candidate)
    return [x/total for x in candidate]
```

### Step 3: Validation and Conversion

The system validates that candidates can be converted back to valid parts:

```python
def _is_valid_proportional_candidate(self, proportions):
    # Try different total parts values to see if any satisfy all constraints
    for total_parts in candidate_totals:
        valid = True
        for i, prop in enumerate(proportions):
            parts_value = prop * total_parts
            min_parts, max_parts = self.component_ranges[i]
            if parts_value < min_parts or parts_value > max_parts:
                valid = False
                break
        if valid:
            return True
    return False
```

## Test Results

All comprehensive tests pass, confirming:

✅ **Basic Functionality**: ProportionalPartsMixture generates valid candidates
✅ **Integration**: OptimalDesignGenerator uses the helper when component ranges are provided  
✅ **Improvement**: Fix provides better candidate generation compared to standard approach
✅ **Compatibility**: Existing SimplifiedMixtureDesign classes work seamlessly

### Example Test Output

```
Testing candidate generation:
Run  Parts                     Proportions               Sum      Valid
---------------------------------------------------------------------------
1    [5.00, 0.38, 1.52]        [0.725, 0.055, 0.221]     1.000000 True
2    [0.20, 0.20, 0.28]        [0.290, 0.297, 0.413]     1.000000 True
3    [0.10, 0.34, 0.44]        [0.113, 0.384, 0.502]     1.000000 True
...

✅ All candidates are valid - boundaries respected and proportions sum to 1
```

## Benefits

1. **Maintains Proportional Relationships**: Candidate points properly respect the relationships between components in parts space
2. **Respects All Boundaries**: Parts values stay within specified ranges
3. **Proper Normalization**: Proportions always sum to 1.0
4. **Seamless Integration**: Works with existing design generation algorithms
5. **Backwards Compatible**: Standard mixture designs continue to work unchanged

## Usage Examples

### Basic Usage

```python
from src.core.proportional_parts_mixture import ProportionalPartsMixture

# Define component ranges
component_ranges = [
    (0.1, 5.0),   # Component A: 0.1 to 5.0 parts
    (0.2, 3.0),   # Component B: 0.2 to 3.0 parts  
    (0.1, 2.0)    # Component C: 0.1 to 2.0 parts
]

# Create proportional parts mixture
ppm = ProportionalPartsMixture(n_components=3, component_ranges=component_ranges)

# Generate candidates
parts_values, proportions = ppm.generate_feasible_parts_candidate()
```

### With OptimalDesignGenerator

```python
from src.core.optimal_design_generator import OptimalDesignGenerator

# Create generator with component ranges (automatically uses proportional parts)
generator = OptimalDesignGenerator(
    num_variables=3,
    num_runs=10,
    design_type="mixture",
    model_type="quadratic",
    component_ranges=component_ranges  # This enables the fix
)

# Generate optimal design
final_det = generator.generate_optimal_design(method="d_optimal")
```

### With SimplifiedMixtureDesign

```python
from src.core.simplified_mixture_design import DOptimalMixtureDesign

# Create designer with parts mode and bounds
designer = DOptimalMixtureDesign(
    n_components=3,
    component_names=['Component_A', 'Component_B', 'Component_C'],
    use_parts_mode=True,
    component_bounds=component_ranges  # This enables the fix
)

# Generate design
design_df = designer.generate_design(n_runs=10, model_type="quadratic")
```

## Files Modified/Created

### New Files
- `src/core/proportional_parts_mixture.py` - Core implementation
- `test_proportional_parts_fix.py` - Comprehensive test suite
- `PROPORTIONAL_PARTS_MIXTURE_SOLUTION.md` - This documentation

### Modified Files  
- `src/core/optimal_design_generator.py` - Added proportional parts helper integration

### Integration Points
- `src/core/simplified_mixture_design.py` - Automatically benefits from the fix

## Conclusion

The proportional parts mixture solution successfully addresses the parts mode proportion issue by:

1. ✅ Converting parts space to proportional space while maintaining relationships
2. ✅ Evaluating boundaries for each component in proper proportion  
3. ✅ Ensuring candidate points maintain proportional relationships
4. ✅ Providing seamless integration with existing design generators
5. ✅ Maintaining backwards compatibility

The fix is now ready for production use and provides a robust solution for handling component boundaries in parts mode while maintaining proper proportional relationships.
