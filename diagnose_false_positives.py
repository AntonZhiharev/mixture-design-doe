"""
Diagnose false positive interactions in two-stage selection
"""

import sys
sys.path.append('.')
sys.path.append('src')

import numpy as np
import pandas as pd
from itertools import combinations


def generate_test_function():
    """Test function with known interactions"""
    linear_coeffs = [3.0, -4.0, 5.0, 2.0, -3.0]
    
    interactions = {
        (0, 2): 2.5, (1, 3): -3.0, (0, 4): 1.5, (2, 3): -2.0,
        (0, 1, 2): 4.0, (2, 3, 4): -3.5,
        (0, 1, 2, 3): 5.0, (1, 2, 3, 4): -4.5,
    }
    
    def mixture_function(x, noise_level=0.01):
        result = sum(coeff * x[i] for i, coeff in enumerate(linear_coeffs))
        for interaction, coeff in interactions.items():
            term = coeff * np.prod([x[idx] for idx in interaction])
            result += term
        if noise_level > 0:
            result += np.random.normal(0, noise_level)
        return result
    
    mixture_function.linear_coeffs = linear_coeffs
    mixture_function.interactions = interactions
    return mixture_function


def main():
    print("="*100)
    print("DIAGNOSING FALSE POSITIVE INTERACTIONS")
    print("="*100)
    
    # Load design
    df = pd.read_excel('MixtureDesigh45Runs1Order.xlsx')
    design = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
    
    print(f"\nDesign shape: {design.shape}")
    
    # Generate true responses
    true_function = generate_test_function()
    np.random.seed(123)
    responses = np.array([true_function(point, noise_level=0.01) for point in design])
    
    # Build full design matrix
    component_names = ['x1', 'x2', 'x3', 'x4', 'x5']
    n_components = len(component_names)
    
    # Collect all terms and build matrix
    terms_list = []
    X_cols = []
    
    # Main effects
    for i in range(n_components):
        terms_list.append(component_names[i])
        X_cols.append(design[:, i])
    
    # 2-way
    for combo in combinations(range(n_components), 2):
        term = '*'.join([component_names[i] for i in combo])
        terms_list.append(term)
        X_cols.append(np.prod(design[:, combo], axis=1))
    
    # 3-way
    for combo in combinations(range(n_components), 3):
        term = '*'.join([component_names[i] for i in combo])
        terms_list.append(term)
        X_cols.append(np.prod(design[:, combo], axis=1))
    
    # 4-way
    for combo in combinations(range(n_components), 4):
        term = '*'.join([component_names[i] for i in combo])
        terms_list.append(term)
        X_cols.append(np.prod(design[:, combo], axis=1))
    
    X_full = np.column_stack(X_cols)
    
    print(f"\nFull model matrix shape: {X_full.shape}")
    print(f"Matrix rank: {np.linalg.matrix_rank(X_full)}")
    print(f"Rank deficiency: {X_full.shape[1] - np.linalg.matrix_rank(X_full)}")
    
    # Check correlations between the problematic terms
    print("\n" + "="*100)
    print("CORRELATION ANALYSIS - TRUE vs FALSE POSITIVE INTERACTIONS")
    print("="*100)
    
    # True 3-way interactions
    true_3way = ['x1*x2*x3', 'x3*x4*x5']
    
    # False positive
    false_3way = ['x1*x4*x5']
    
    # True 4-way interactions
    true_4way = ['x1*x2*x3*x4', 'x2*x3*x4*x5']
    
    # Build correlation matrix for these terms
    check_terms = true_3way + false_3way + true_4way
    check_indices = [terms_list.index(t) for t in check_terms]
    
    print(f"\nChecking correlations for:")
    for t in check_terms:
        idx = terms_list.index(t)
        true_exists = any(t == '*'.join([component_names[i] for i in interaction]) 
                         for interaction in true_function.interactions.keys())
        status = "✅ TRUE" if true_exists else "❌ FALSE POSITIVE"
        print(f"  {t:<20} {status}")
    
    # Calculate correlation matrix
    X_check = X_full[:, check_indices]
    corr_matrix = np.corrcoef(X_check.T)
    
    print(f"\nCorrelation matrix:")
    print(f"{'':20}", end="")
    for t in check_terms:
        print(f"{t:12}", end="")
    print()
    print("-" * 100)
    
    for i, term1 in enumerate(check_terms):
        print(f"{term1:20}", end="")
        for j, term2 in enumerate(check_terms):
            print(f"{corr_matrix[i, j]:12.4f}", end="")
        print()
    
    # Find high correlations
    print(f"\n" + "="*100)
    print("HIGH CORRELATIONS (|r| > 0.5) - POTENTIAL ALIASING")
    print("="*100)
    
    for i, term1 in enumerate(check_terms):
        for j, term2 in enumerate(check_terms):
            if i < j and abs(corr_matrix[i, j]) > 0.5:
                true1 = any(term1 == '*'.join([component_names[k] for k in interaction]) 
                           for interaction in true_function.interactions.keys())
                true2 = any(term2 == '*'.join([component_names[k] for k in interaction]) 
                           for interaction in true_function.interactions.keys())
                
                status1 = "TRUE" if true1 else "FALSE"
                status2 = "TRUE" if true2 else "FALSE"
                
                print(f"{term1:20} ({status1:5}) <-> {term2:20} ({status2:5})  r = {corr_matrix[i, j]:+.4f}")
    
    # Check if 4-way can be confounded with 3-way
    print(f"\n" + "="*100)
    print("ANALYSIS: WHY x1*x4*x5 (FALSE) IS DETECTED INSTEAD OF x1*x2*x3*x4 or x2*x3*x4*x5 (TRUE)")
    print("="*100)
    
    # Check all correlations with x1*x4*x5
    false_idx = terms_list.index('x1*x4*x5')
    
    print(f"\nCorrelations of FALSE POSITIVE 'x1*x4*x5' with ALL 4-way interactions:")
    print(f"{'4-way Term':<20} {'Correlation':>12}")
    print("-" * 40)
    
    for combo in combinations(range(n_components), 4):
        term_4way = '*'.join([component_names[i] for i in combo])
        idx_4way = terms_list.index(term_4way)
        
        col_false = X_full[:, false_idx]
        col_4way = X_full[:, idx_4way]
        
        corr = np.corrcoef(col_false, col_4way)[0, 1]
        
        is_true = any(term_4way == '*'.join([component_names[i] for i in interaction]) 
                     for interaction in true_function.interactions.keys())
        status = "✅ TRUE" if is_true else ""
        
        print(f"{term_4way:<20} {corr:+12.6f}  {status}")
    
    print(f"\n" + "="*100)
    print("ROOT CAUSE ANALYSIS")
    print("="*100)
    
    print("""
The problem is ALIASING/CONFOUNDING in mixture designs:

1. With 45 runs and 30 terms (5 main + 10 two-way + 10 three-way + 5 four-way),
   the design has rank deficiency.

2. In mixture designs, components sum to 1, which creates linear dependencies:
   - 4-way interactions can be partially confounded with 3-way interactions
   - The term x1*x4*x5 is picking up signal from the true 4-way interactions
   
3. The backward elimination algorithm cannot distinguish between:
   - A TRUE 3-way interaction x1*x4*x5
   - A FALSE 3-way that is confounded with TRUE 4-way interactions
   
4. Since 4-way interactions are harder to estimate (need more data),
   the algorithm prefers to fit the confounded 3-way term instead.

SOLUTION:
Use hierarchical model selection that respects the heredity principle:
- Only include 3-way interactions if their parent 2-ways are included
- Use regularization (Lasso/Elastic Net) to handle multicollinearity
- Increase the p-value threshold or use a different selection criterion
- Use cross-validation to detect overfitting
""")


if __name__ == "__main__":
    main()
