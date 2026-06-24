"""
Test parameter recovery with a REALISTIC hierarchical function

This addresses the issue that the original test function had interactions
that didn't respect hierarchy (e.g., x1*x2*x3 existed but x1*x2 didn't).

Now we use a properly hierarchical function where:
- 3-way interactions only exist if their parent 2-ways exist
- 4-way interactions only exist if their parent 3-ways exist
"""

import sys
sys.path.append('.')
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
from itertools import combinations
from generate_hierarchical_function import generate_hierarchical_mixture_function


def build_design_matrix(design, terms, component_names):
    """Build design matrix"""
    n_obs, n_terms = len(design), len(terms)
    X = np.zeros((n_obs, n_terms))
    
    for j, term in enumerate(terms):
        if '*' in term:
            components = term.split('*')
            indices = [component_names.index(comp) for comp in components]
            X[:, j] = np.prod(design[:, indices], axis=1)
        else:
            idx = component_names.index(term)
            X[:, j] = design[:, idx]
    
    return X


def calculate_p_values(X, y, terms):
    """Calculate p-values using OLS"""
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    n_obs, n_params = len(y), len(terms)
    
    sse = np.sum((y - y_pred) ** 2)
    residual_var = sse / (n_obs - n_params)
    
    if residual_var <= 0 or n_obs <= n_params:
        return None, None
    
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(residual_var * np.diag(XtX_inv))
        
        p_values, coefficients = {}, {}
        
        for i, term in enumerate(terms):
            coef = model.coef_[i]
            t_stat = coef / se[i] if se[i] > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n_obs - n_params))
            
            p_values[term] = p_val
            coefficients[term] = coef
        
        return p_values, coefficients
    except:
        return None, None


def get_parents(term):
    """Get all parent terms (one order lower)"""
    components = term.split('*')
    if len(components) <= 1:
        return []
    
    parents = []
    for i in range(len(components)):
        parent_components = components[:i] + components[i+1:]
        if len(parent_components) > 0:
            parents.append('*'.join(parent_components))
    
    return parents


def can_remove_term(term, current_terms, component_names):
    """Check if term can be removed without violating hierarchical constraints"""
    
    # Main effects cannot be removed
    if term in component_names:
        return False
    
    # Check if any higher-order term depends on this term
    for other_term in current_terms:
        if other_term == term:
            continue
        
        parents = get_parents(other_term)
        if term in parents:
            return False
    
    return True


def backward_elimination_hierarchical(design, responses, component_names, p_threshold=0.10):
    """Backward elimination with hierarchical constraints"""
    
    # Generate all terms
    all_terms = list(component_names)
    for order in range(2, 5):
        for combo in combinations(range(len(component_names)), order):
            all_terms.append('*'.join([component_names[i] for i in combo]))
    
    current_terms = all_terms.copy()
    
    print(f"Starting backward elimination with {len(current_terms)} terms")
    
    iteration = 0
    while True:
        iteration += 1
        
        X = build_design_matrix(design, current_terms, component_names)
        p_values, coefficients = calculate_p_values(X, responses, current_terms)
        
        if p_values is None:
            break
        
        # Find highest p-value among removable terms
        worst_term, worst_p = None, -1
        for term, p_val in p_values.items():
            if p_val > p_threshold and p_val > worst_p:
                if can_remove_term(term, current_terms, component_names):
                    worst_term, worst_p = term, p_val
        
        if worst_term is None:
            break
        
        current_terms.remove(worst_term)
        
        if iteration > 50:
            break
    
    print(f"Converged after {iteration} iterations with {len(current_terms)} terms\n")
    
    return current_terms


def forward_selection_hierarchical(design, responses, component_names, p_threshold=0.10):
    """Forward selection with hierarchical constraints"""
    
    # Generate all possible terms
    all_terms = list(component_names)
    for order in range(2, 5):
        for combo in combinations(range(len(component_names)), order):
            all_terms.append('*'.join([component_names[i] for i in combo]))
    
    current_terms = list(component_names)  # Start with main effects
    candidate_pool = [t for t in all_terms if t not in current_terms]
    
    print(f"Starting forward selection with {len(current_terms)} main effects")
    
    iteration = 0
    while True:
        iteration += 1
        
        best_term, best_p = None, 1.0
        
        # Try each candidate
        for candidate in candidate_pool:
            # Check hierarchy: all parents must be in model
            parents = get_parents(candidate)
            if not all(p in current_terms for p in parents):
                continue
            
            # Test this term
            test_terms = current_terms + [candidate]
            X = build_design_matrix(design, test_terms, component_names)
            p_values, _ = calculate_p_values(X, responses, test_terms)
            
            if p_values is None:
                continue
            
            if p_values[candidate] < best_p:
                best_term = candidate
                best_p = p_values[candidate]
        
        # Add best term if significant
        if best_term and best_p < p_threshold:
            current_terms.append(best_term)
            candidate_pool.remove(best_term)
            print(f"Iter {iteration}: Adding {best_term:<20} (p={best_p:.6f})")
        else:
            break
        
        if iteration > 50:
            break
    
    print(f"\nConverged after {iteration} iterations with {len(current_terms)} terms\n")
    
    return current_terms


def compare_methods(design, responses, true_function, component_names):
    """Compare backward vs forward selection"""
    
    print(f"{'='*100}")
    print(f"METHOD COMPARISON ON HIERARCHICAL FUNCTION")
    print(f"{'='*100}\n")
    
    # Method 1: Backward elimination
    print(f"METHOD 1: BACKWARD ELIMINATION")
    print(f"{'-'*100}")
    backward_terms = backward_elimination_hierarchical(design, responses, component_names, p_threshold=0.10)
    
    # Method 2: Forward selection
    print(f"METHOD 2: FORWARD SELECTION")
    print(f"{'-'*100}")
    forward_terms = forward_selection_hierarchical(design, responses, component_names, p_threshold=0.10)
    
    # Analyze results
    print(f"{'='*100}")
    print(f"RESULTS ANALYSIS")
    print(f"{'='*100}\n")
    
    # True interactions
    true_interactions = set()
    for interaction in true_function.interactions.keys():
        term = '*'.join([component_names[i] for i in interaction])
        true_interactions.add(term)
    
    print(f"TRUE MODEL: {true_function.n_terms} terms")
    print(f"  Main effects: {len(component_names)}")
    print(f"  Interactions: {len(true_interactions)}")
    
    for interaction in sorted(true_interactions, key=lambda x: (len(x.split('*')), x)):
        coef = true_function.interactions[tuple([component_names.index(c) for c in interaction.split('*')])]
        order = len(interaction.split('*'))
        print(f"    {interaction:<20} coef={coef:+.3f}  ({order}-way)")
    
    # Backward results
    print(f"\nBACKWARD ELIMINATION: {len(backward_terms)} terms")
    backward_interactions = set([t for t in backward_terms if '*' in t])
    
    true_pos_back = true_interactions & backward_interactions
    false_pos_back = backward_interactions - true_interactions
    false_neg_back = true_interactions - backward_interactions
    
    print(f"  True positives: {len(true_pos_back)}/{len(true_interactions)}")
    print(f"  False positives: {len(false_pos_back)}")
    print(f"  False negatives: {len(false_neg_back)}")
    
    if false_pos_back:
        print(f"  ❌ False positives:")
        for term in sorted(false_pos_back, key=lambda x: (len(x.split('*')), x)):
            print(f"     {term}")
    
    if false_neg_back:
        print(f"  ❌ Missed:")
        for term in sorted(false_neg_back, key=lambda x: (len(x.split('*')), x)):
            true_coef = true_function.interactions[tuple([component_names.index(c) for c in term.split('*')])]
            print(f"     {term:<20} (true coef={true_coef:+.3f})")
    
    # Forward results
    print(f"\nFORWARD SELECTION: {len(forward_terms)} terms")
    forward_interactions = set([t for t in forward_terms if '*' in t])
    
    true_pos_fwd = true_interactions & forward_interactions
    false_pos_fwd = forward_interactions - true_interactions
    false_neg_fwd = true_interactions - forward_interactions
    
    print(f"  True positives: {len(true_pos_fwd)}/{len(true_interactions)}")
    print(f"  False positives: {len(false_pos_fwd)}")
    print(f"  False negatives: {len(false_neg_fwd)}")
    
    if false_pos_fwd:
        print(f"  ❌ False positives:")
        for term in sorted(false_pos_fwd, key=lambda x: (len(x.split('*')), x)):
            print(f"     {term}")
    
    if false_neg_fwd:
        print(f"  ❌ Missed:")
        for term in sorted(false_neg_fwd, key=lambda x: (len(x.split('*')), x)):
            true_coef = true_function.interactions[tuple([component_names.index(c) for c in term.split('*')])]
            print(f"     {term:<20} (true coef={true_coef:+.3f})")
    
    # Calculate R²
    X_back = build_design_matrix(design, backward_terms, component_names)
    model_back = LinearRegression(fit_intercept=False)
    model_back.fit(X_back, responses)
    r2_back = model_back.score(X_back, responses)
    
    X_fwd = build_design_matrix(design, forward_terms, component_names)
    model_fwd = LinearRegression(fit_intercept=False)
    model_fwd.fit(X_fwd, responses)
    r2_fwd = model_fwd.score(X_fwd, responses)
    
    print(f"\nPREDICTIVE ACCURACY:")
    print(f"  Backward R²: {r2_back:.6f}")
    print(f"  Forward R²:  {r2_fwd:.6f}")
    
    # Detailed error analysis - 10 worst points
    print(f"\n{'='*100}")
    print(f"PREDICTION ERROR ANALYSIS - 10 WORST POINTS")
    print(f"{'='*100}\n")
    
    # Calculate true values (no noise)
    true_values = np.array([true_function(point, noise_level=0.0) for point in design])
    
    # Backward predictions
    pred_back = model_back.predict(X_back)
    errors_back = np.abs(true_values - pred_back)
    worst_idx_back = np.argsort(errors_back)[-10:][::-1]
    
    print(f"BACKWARD ELIMINATION - 10 WORST PREDICTIONS:")
    print(f"{'Rank':<6} {'Point':<50} {'True Value':<12} {'Predicted':<12} {'Error':<12}")
    print(f"{'-'*100}")
    for rank, idx in enumerate(worst_idx_back, 1):
        point = design[idx]
        point_str = f"[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}, {point[3]:.3f}, {point[4]:.3f}]"
        true_val = true_values[idx]
        pred_val = pred_back[idx]
        error = errors_back[idx]
        print(f"{rank:<6} {point_str:<50} {true_val:+10.4f}  {pred_val:+10.4f}  {error:10.4f}")
    
    print(f"\nBackward Error Statistics:")
    print(f"  Mean Absolute Error: {np.mean(errors_back):.6f}")
    print(f"  Max Error: {np.max(errors_back):.6f}")
    print(f"  Std Dev: {np.std(errors_back):.6f}")
    
    # Forward predictions  
    pred_fwd = model_fwd.predict(X_fwd)
    errors_fwd = np.abs(true_values - pred_fwd)
    worst_idx_fwd = np.argsort(errors_fwd)[-10:][::-1]
    
    print(f"\n{'-'*100}")
    print(f"FORWARD SELECTION - 10 WORST PREDICTIONS:")
    print(f"{'Rank':<6} {'Point':<50} {'True Value':<12} {'Predicted':<12} {'Error':<12}")
    print(f"{'-'*100}")
    for rank, idx in enumerate(worst_idx_fwd, 1):
        point = design[idx]
        point_str = f"[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}, {point[3]:.3f}, {point[4]:.3f}]"
        true_val = true_values[idx]
        pred_val = pred_fwd[idx]
        error = errors_fwd[idx]
        print(f"{rank:<6} {point_str:<50} {true_val:+10.4f}  {pred_val:+10.4f}  {error:10.4f}")
    
    print(f"\nForward Error Statistics:")
    print(f"  Mean Absolute Error: {np.mean(errors_fwd):.6f}")
    print(f"  Max Error: {np.max(errors_fwd):.6f}")
    print(f"  Std Dev: {np.std(errors_fwd):.6f}")
    
    print(f"\n{'='*100}")
    print(f"CONCLUSION")
    print(f"{'='*100}")
    
    if len(false_pos_back) == 0 and len(false_pos_fwd) == 0:
        print(f"✅ Both methods: NO FALSE POSITIVES with hierarchical function!")
    elif len(false_pos_back) < len(false_pos_fwd):
        print(f"⚠️  Backward has fewer false positives ({len(false_pos_back)} vs {len(false_pos_fwd)})")
    elif len(false_pos_fwd) < len(false_pos_back):
        print(f"✅ Forward has fewer false positives ({len(false_pos_fwd)} vs {len(false_pos_back)})")
    else:
        print(f"⚠️  Both have same number of false positives ({len(false_pos_back)})")
    
    if len(false_neg_fwd) < len(false_neg_back):
        print(f"✅ Forward detected more true interactions ({len(true_pos_fwd)} vs {len(true_pos_back)})")
    elif len(false_neg_back) < len(false_neg_fwd):
        print(f"⚠️  Backward detected more true interactions ({len(true_pos_back)} vs {len(true_pos_fwd)})")
    
    print(f"\n{'='*100}\n")


def main():
    """Run comparison with hierarchical function"""
    
    print("\n" + "="*100)
    print("TESTING PARAMETER RECOVERY WITH HIERARCHICAL FUNCTION")
    print("="*100 + "\n")
    
    # Generate hierarchical function (seed 789 has a 4-way interaction)
    print("Generating hierarchical test function...")
    print("-"*100)
    true_function = generate_hierarchical_mixture_function(n_components=5, seed=789)
    
    # Load JMP design
    df = pd.read_excel('MixtureDesigh45Runs1Order.xlsx')
    jmp_design = df[['X1', 'X2', 'X3', 'X4', 'X5']].values
    
    # Generate responses
    np.random.seed(123)
    responses = np.array([true_function(point, noise_level=0.01) for point in jmp_design])
    
    component_names = ['x1', 'x2', 'x3', 'x4', 'x5']
    
    # Compare methods
    compare_methods(jmp_design, responses, true_function, component_names)


if __name__ == "__main__":
    main()
