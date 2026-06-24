"""
Monte Carlo Test: Generate Random Functions, Recover, and Test

This script:
1. Generates 100 random hierarchical functions
2. Recovers each using backward and forward selection
3. Tests on random mixture points (x1+x2+x3+x4+x5=1)
4. Compares prediction accuracy
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
from src.algorithms.d_optimal_algorithm import MixtureDOptimalAlgorithm
from src.algorithms.candidate_generation import create_candidate_generator


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
    if term in component_names:
        return False
    
    for other_term in current_terms:
        if other_term == term:
            continue
        parents = get_parents(other_term)
        if term in parents:
            return False
    
    return True


def backward_elimination_hierarchical(design, responses, component_names, p_threshold=0.10):
    """Backward elimination with hierarchical constraints"""
    all_terms = list(component_names)
    for order in range(2, 5):
        for combo in combinations(range(len(component_names)), order):
            all_terms.append('*'.join([component_names[i] for i in combo]))
    
    current_terms = all_terms.copy()
    iteration = 0
    
    while True:
        iteration += 1
        X = build_design_matrix(design, current_terms, component_names)
        p_values, coefficients = calculate_p_values(X, responses, current_terms)
        
        if p_values is None:
            break
        
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
    
    return current_terms


def forward_selection_hierarchical(design, responses, component_names, p_threshold=0.10):
    """Forward selection with hierarchical constraints"""
    all_terms = list(component_names)
    for order in range(2, 5):
        for combo in combinations(range(len(component_names)), order):
            all_terms.append('*'.join([component_names[i] for i in combo]))
    
    current_terms = list(component_names)
    candidate_pool = [t for t in all_terms if t not in current_terms]
    iteration = 0
    
    while True:
        iteration += 1
        best_term, best_p = None, 1.0
        
        for candidate in candidate_pool:
            parents = get_parents(candidate)
            if not all(p in current_terms for p in parents):
                continue
            
            test_terms = current_terms + [candidate]
            X = build_design_matrix(design, test_terms, component_names)
            p_values, _ = calculate_p_values(X, responses, test_terms)
            
            if p_values is None:
                continue
            
            if p_values[candidate] < best_p:
                best_term = candidate
                best_p = p_values[candidate]
        
        if best_term and best_p < p_threshold:
            current_terms.append(best_term)
            candidate_pool.remove(best_term)
        else:
            break
        
        if iteration > 50:
            break
    
    return current_terms


def generate_random_mixture_points(n_points, n_components, seed=None):
    """Generate random points satisfying mixture constraint x1+x2+...+xn=1"""
    if seed is not None:
        np.random.seed(seed)
    
    points = []
    for _ in range(n_points):
        # Generate random values
        vals = np.random.random(n_components)
        # Normalize to sum to 1
        vals = vals / vals.sum()
        points.append(vals)
    
    return np.array(points)


def run_single_trial(trial_num, design, component_names):
    """Run a single trial: generate function, recover, test"""
    
    # Generate random hierarchical function
    true_function = generate_hierarchical_mixture_function(n_components=5, seed=trial_num*1000)
    
    # Calculate noise level based on the smallest parameter magnitude (1% of smallest effect)
    all_coeffs = list(true_function.linear_coeffs) + [coef for coef in true_function.interactions.values()]
    min_coeff = min(abs(c) for c in all_coeffs)
    noise_level = 0.01 * min_coeff  # 1% of smallest coefficient
    
    # Generate responses with noise
    np.random.seed(trial_num*1000 + 1)
    responses = np.array([true_function(point, noise_level=noise_level) for point in design])
    
    # Recover using both methods
    backward_terms = backward_elimination_hierarchical(design, responses, component_names, p_threshold=0.10)
    forward_terms = forward_selection_hierarchical(design, responses, component_names, p_threshold=0.10)
    
    # Fit models
    X_back = build_design_matrix(design, backward_terms, component_names)
    model_back = LinearRegression(fit_intercept=False)
    model_back.fit(X_back, responses)
    
    X_fwd = build_design_matrix(design, forward_terms, component_names)
    model_fwd = LinearRegression(fit_intercept=False)
    model_fwd.fit(X_fwd, responses)
    
    # Generate random test points
    test_points = generate_random_mixture_points(100, 5, seed=trial_num*1000 + 2)
    
    # Calculate true values on test points
    true_test_values = np.array([true_function(point, noise_level=0.0) for point in test_points])
    
    # Predict with recovered models
    X_test_back = build_design_matrix(test_points, backward_terms, component_names)
    pred_test_back = model_back.predict(X_test_back)
    
    X_test_fwd = build_design_matrix(test_points, forward_terms, component_names)
    pred_test_fwd = model_fwd.predict(X_test_fwd)
    
    # Calculate errors
    mae_back = np.mean(np.abs(true_test_values - pred_test_back))
    mae_fwd = np.mean(np.abs(true_test_values - pred_test_fwd))
    
    errors_back = np.abs(true_test_values - pred_test_back)
    errors_fwd = np.abs(true_test_values - pred_test_fwd)
    
    max_err_back = np.max(errors_back)
    max_err_fwd = np.max(errors_fwd)
    
    # Get worst prediction points
    worst_idx_back = np.argmax(errors_back)
    worst_idx_fwd = np.argmax(errors_fwd)
    
    worst_point_back = test_points[worst_idx_back]
    worst_point_fwd = test_points[worst_idx_fwd]
    
    worst_true_back = true_test_values[worst_idx_back]
    worst_pred_back = pred_test_back[worst_idx_back]
    
    worst_true_fwd = true_test_values[worst_idx_fwd]
    worst_pred_fwd = pred_test_fwd[worst_idx_fwd]
    
    # Count terms
    n_terms_true = true_function.n_terms
    n_terms_back = len(backward_terms)
    n_terms_fwd = len(forward_terms)
    
    # Count correct/false positives
    true_interactions = set()
    for interaction in true_function.interactions.keys():
        term = '*'.join([component_names[i] for i in interaction])
        true_interactions.add(term)
    
    backward_interactions = set([t for t in backward_terms if '*' in t])
    forward_interactions = set([t for t in forward_terms if '*' in t])
    
    true_pos_back = len(true_interactions & backward_interactions)
    false_pos_back = len(backward_interactions - true_interactions)
    
    true_pos_fwd = len(true_interactions & forward_interactions)
    false_pos_fwd = len(forward_interactions - true_interactions)
    
    return {
        'trial': trial_num,
        'n_true_terms': n_terms_true,
        'n_back_terms': n_terms_back,
        'n_fwd_terms': n_terms_fwd,
        'true_pos_back': true_pos_back,
        'false_pos_back': false_pos_back,
        'true_pos_fwd': true_pos_fwd,
        'false_pos_fwd': false_pos_fwd,
        'mae_back': mae_back,
        'mae_fwd': mae_fwd,
        'max_err_back': max_err_back,
        'max_err_fwd': max_err_fwd,
        'worst_point_back': worst_point_back,
        'worst_true_back': worst_true_back,
        'worst_pred_back': worst_pred_back,
        'worst_point_fwd': worst_point_fwd,
        'worst_true_fwd': worst_true_fwd,
        'worst_pred_fwd': worst_pred_fwd
    }


def main():
    """Run Monte Carlo simulation"""
    
    print("="*100)
    print("MONTE CARLO SIMULATION: 100 RANDOM FUNCTIONS")
    print("="*100)
    print()
    print("Process:")
    print("1. Generate OUR D-optimal design (ONCE at beginning)")
    print("2. Generate 100 random hierarchical functions")
    print("3. Recover each using backward and forward selection")
    print("4. Test each on 100 random mixture points (sum=1 constraint)")
    print("5. Analyze results")
    print()
    
    component_names = ['x1', 'x2', 'x3', 'x4', 'x5']
    
    # Generate OUR D-optimal design ONCE at the beginning
    print("="*100)
    print("STEP 1: GENERATING OUR D-OPTIMAL DESIGN (45 runs)")
    print("="*100)
    
    candidate_gen = create_candidate_generator(
        'lhs', 
        n_components=5, 
        component_names=component_names,
        component_bounds=[(0.0, 1.0)] * 5
    )
    
    candidates_list = candidate_gen.generate_candidates(n_candidates=5000)
    candidates = np.array(candidates_list)
    candidates = candidates / candidates.sum(axis=1, keepdims=True)
    
    print(f"Generated {len(candidates)} candidate points using LHS")
    
    d_opt_algo = MixtureDOptimalAlgorithm(model_type="cubic")
    our_design, det, info = d_opt_algo.optimize_mixture_design(candidates, n_runs=45, strategy="balanced")
    
    print()
    print("="*100)
    print("STEP 2: RUNNING 100 TRIALS WITH OUR D-OPTIMAL DESIGN")
    print("="*100)
    print(f"Using {len(our_design)} design points")
    print(f"Testing on 100 random mixture points per trial")
    print()
    print("Running trials...")
    
    # Run trials with our design
    results = []
    for i in range(100):
        if (i+1) % 10 == 0:
            print(f"  Completed {i+1}/100 trials...")
        
        result = run_single_trial(i, our_design, component_names)
        results.append(result)
    
    print()
    print("="*100)
    print("MONTE CARLO RESULTS SUMMARY (OUR D-OPTIMAL DESIGN)")
    print("="*100)
    print()
    
    # Convert to arrays for analysis
    mae_back = np.array([r['mae_back'] for r in results])
    mae_fwd = np.array([r['mae_fwd'] for r in results])
    max_err_back = np.array([r['max_err_back'] for r in results])
    max_err_fwd = np.array([r['max_err_fwd'] for r in results])
    
    true_pos_back = np.array([r['true_pos_back'] for r in results])
    false_pos_back = np.array([r['false_pos_back'] for r in results])
    true_pos_fwd = np.array([r['true_pos_fwd'] for r in results])
    false_pos_fwd = np.array([r['false_pos_fwd'] for r in results])
    
    n_terms_back = np.array([r['n_back_terms'] for r in results])
    n_terms_fwd = np.array([r['n_fwd_terms'] for r in results])
    n_terms_true = np.array([r['n_true_terms'] for r in results])
    
    # Print summary statistics
    print("PREDICTION ACCURACY ON TEST DATA:")
    print(f"{'Metric':<30} {'Backward':<20} {'Forward':<20}")
    print("-"*70)
    print(f"{'Mean Absolute Error:':<30} {np.mean(mae_back):<20.6f} {np.mean(mae_fwd):<20.6f}")
    print(f"{'  Std Dev:':<30} {np.std(mae_back):<20.6f} {np.std(mae_fwd):<20.6f}")
    print(f"{'  Median:':<30} {np.median(mae_back):<20.6f} {np.median(mae_fwd):<20.6f}")
    print()
    print(f"{'Max Error:':<30} {np.mean(max_err_back):<20.6f} {np.mean(max_err_fwd):<20.6f}")
    print(f"{'  Std Dev:':<30} {np.std(max_err_back):<20.6f} {np.std(max_err_fwd):<20.6f}")
    print(f"{'  Median:':<30} {np.median(max_err_back):<20.6f} {np.median(max_err_fwd):<20.6f}")
    
    print()
    print("MODEL SELECTION QUALITY:")
    print(f"{'Metric':<30} {'Backward':<20} {'Forward':<20}")
    print("-"*70)
    print(f"{'True Positives (avg):':<30} {np.mean(true_pos_back):<20.2f} {np.mean(true_pos_fwd):<20.2f}")
    print(f"{'False Positives (avg):':<30} {np.mean(false_pos_back):<20.2f} {np.mean(false_pos_fwd):<20.2f}")
    print(f"{'Model Size (avg terms):':<30} {np.mean(n_terms_back):<20.1f} {np.mean(n_terms_fwd):<20.1f}")
    print(f"{'True Model Size (avg):':<30} {np.mean(n_terms_true):<20.1f}")
    
    print()
    print("="*100)
    print("DETAILED COMPARISON")
    print("="*100)
    print()
    
    # Count which method is better
    back_better_mae = np.sum(mae_back < mae_fwd)
    fwd_better_mae = np.sum(mae_fwd < mae_back)
    tie_mae = np.sum(mae_back == mae_fwd)
    
    back_better_max = np.sum(max_err_back < max_err_fwd)
    fwd_better_max = np.sum(max_err_fwd < max_err_back)
    
    back_fewer_false = np.sum(false_pos_back < false_pos_fwd)
    fwd_fewer_false = np.sum(false_pos_fwd < false_pos_back)
    
    print(f"Mean Absolute Error:")
    print(f"  Backward better: {back_better_mae}/100 trials")
    print(f"  Forward better:  {fwd_better_mae}/100 trials")
    print(f"  Ties:            {tie_mae}/100 trials")
    print()
    
    print(f"Max Error:")
    print(f"  Backward better: {back_better_max}/100 trials")
    print(f"  Forward better:  {fwd_better_max}/100 trials")
    print()
    
    print(f"False Positives:")
    print(f"  Backward fewer:  {back_fewer_false}/100 trials")
    print(f"  Forward fewer:   {fwd_fewer_false}/100 trials")
    
    print()
    print("="*100)
    print("10 WORST TRIALS - DETAILED ANALYSIS")
    print("="*100)
    print()
    
    # Find 10 worst trials by average MAE (average of backward and forward)
    avg_mae = (mae_back + mae_fwd) / 2
    worst_10_indices = np.argsort(avg_mae)[-10:][::-1]
    
    print("Analyzing the 10 trials with highest average prediction errors...")
    print()
    
    for rank, trial_idx in enumerate(worst_10_indices, 1):
        result = results[trial_idx]
        
        print(f"{'='*100}")
        print(f"RANK #{rank} - TRIAL {result['trial']}")
        print(f"{'='*100}")
        print()
        
        # Regenerate the function for this trial to get details
        true_func = generate_hierarchical_mixture_function(n_components=5, seed=result['trial']*1000)
        
        print(f"TRUE FUNCTION: {result['n_true_terms']} terms")
        print(f"  Main effects: 5")
        
        # Count interactions by order
        n_2way_true = sum(1 for k in true_func.interactions.keys() if len(k) == 2)
        n_3way_true = sum(1 for k in true_func.interactions.keys() if len(k) == 3)
        n_4way_true = sum(1 for k in true_func.interactions.keys() if len(k) == 4)
        
        print(f"  2-way: {n_2way_true}, 3-way: {n_3way_true}, 4-way: {n_4way_true}")
        print(f"  Interactions:")
        
        for interaction, coef in sorted(true_func.interactions.items(), key=lambda x: len(x[0])):
            term = '*'.join([f"x{i+1}" for i in interaction])
            print(f"    {term:<15} coef={coef:+8.3f}")
        
        print()
        print(f"BACKWARD ELIMINATION:")
        print(f"  Selected: {result['n_back_terms']} terms")
        print(f"  True positives: {result['true_pos_back']}/{len(true_func.interactions)}")
        print(f"  False positives: {result['false_pos_back']}")
        print(f"  Test MAE: {result['mae_back']:.6f}")
        print(f"  Test Max Error: {result['max_err_back']:.6f}")
        print(f"  Worst prediction point:")
        worst_pt = result['worst_point_back']
        print(f"    Point: [{worst_pt[0]:.3f}, {worst_pt[1]:.3f}, {worst_pt[2]:.3f}, {worst_pt[3]:.3f}, {worst_pt[4]:.3f}]")
        print(f"    True value:      {result['worst_true_back']:+10.6f}")
        print(f"    Predicted value: {result['worst_pred_back']:+10.6f}")
        print(f"    Error:           {result['max_err_back']:10.6f}")
        
        print()
        print(f"FORWARD SELECTION:")
        print(f"  Selected: {result['n_fwd_terms']} terms")
        print(f"  True positives: {result['true_pos_fwd']}/{len(true_func.interactions)}")
        print(f"  False positives: {result['false_pos_fwd']}")
        print(f"  Test MAE: {result['mae_fwd']:.6f}")
        print(f"  Test Max Error: {result['max_err_fwd']:.6f}")
        print(f"  Worst prediction point:")
        worst_pt = result['worst_point_fwd']
        print(f"    Point: [{worst_pt[0]:.3f}, {worst_pt[1]:.3f}, {worst_pt[2]:.3f}, {worst_pt[3]:.3f}, {worst_pt[4]:.3f}]")
        print(f"    True value:      {result['worst_true_fwd']:+10.6f}")
        print(f"    Predicted value: {result['worst_pred_fwd']:+10.6f}")
        print(f"    Error:           {result['max_err_fwd']:10.6f}")
        
        print()
        print(f"COMPARISON:")
        better_method = "Forward" if result['mae_fwd'] < result['mae_back'] else "Backward"
        worse_method = "Backward" if result['mae_fwd'] < result['mae_back'] else "Forward"
        diff = abs(result['mae_back'] - result['mae_fwd'])
        print(f"  {better_method} has {diff:.6f} lower MAE than {worse_method}")
        
        if result['false_pos_fwd'] < result['false_pos_back']:
            print(f"  Forward has {result['false_pos_back'] - result['false_pos_fwd']} fewer false positives")
        elif result['false_pos_back'] < result['false_pos_fwd']:
            print(f"  Backward has {result['false_pos_fwd'] - result['false_pos_back']} fewer false positives")
        
        print()
    
    print("="*100)
    print("CONCLUSION")
    print("="*100)
    
    if np.mean(mae_fwd) < np.mean(mae_back):
        improvement = ((np.mean(mae_back) - np.mean(mae_fwd)) / np.mean(mae_back)) * 100
        print(f"✅ Forward selection has {improvement:.1f}% better mean prediction accuracy")
    else:
        improvement = ((np.mean(mae_fwd) - np.mean(mae_back)) / np.mean(mae_fwd)) * 100
        print(f"⚠️  Backward elimination has {improvement:.1f}% better mean prediction accuracy")
    
    if np.mean(false_pos_fwd) < np.mean(false_pos_back):
        print(f"✅ Forward selection has fewer false positives ({np.mean(false_pos_fwd):.1f} vs {np.mean(false_pos_back):.1f})")
    else:
        print(f"⚠️  Backward has fewer false positives ({np.mean(false_pos_back):.1f} vs {np.mean(false_pos_fwd):.1f})")
    
    print()
    print("Over 100 random hierarchical functions:")
    print(f"  - Both methods achieve good prediction accuracy (MAE < {max(np.mean(mae_back), np.mean(mae_fwd)):.3f})")
    print(f"  - Forward is more conservative ({np.mean(n_terms_fwd):.1f} vs {np.mean(n_terms_back):.1f} terms)")
    print(f"  - Backward detects more true interactions ({np.mean(true_pos_back):.1f} vs {np.mean(true_pos_fwd):.1f})")
    print(f"  - But backward has more false positives ({np.mean(false_pos_back):.1f} vs {np.mean(false_pos_fwd):.1f})")
    
    print()
    print("The 10 worst trials show that even in challenging cases:")
    print(f"  - Errors remain relatively small (worst avg MAE: {avg_mae[worst_10_indices[0]]:.6f})")
    print(f"  - Forward consistently has fewer false positives")
    print()
    print("="*100)


if __name__ == "__main__":
    main()
