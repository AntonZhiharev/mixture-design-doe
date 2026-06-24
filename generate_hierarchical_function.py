"""
Generate a realistic hierarchical test function for mixture DOE

Key principle: Heredity/Hierarchy
- If a 3-way interaction exists, at least some of its parent 2-ways must exist
- If a 4-way interaction exists, at least some of its parent 3-ways must exist
- Stronger lower-order effects lead to stronger higher-order effects
"""

import numpy as np
from itertools import combinations


def generate_hierarchical_mixture_function(n_components=5, seed=42):
    """
    Generate a realistic hierarchical mixture function
    
    Algorithm:
    1. Start with main effects (all exist)
    2. Randomly select some 2-way interactions
    3. For 3-way: only include if AT LEAST 2 of its parent 2-ways exist
    4. For 4-way: only include if AT LEAST 2 of its parent 3-ways exist
    
    This ensures heredity principle is respected.
    """
    np.random.seed(seed)
    
    # Main effects - all components have effects (10x larger range)
    linear_coeffs = np.random.uniform(-50, 50, n_components)
    print(f"MAIN EFFECTS:")
    for i, coef in enumerate(linear_coeffs):
        print(f"  x{i+1}: {coef:+.3f}")
    
    interactions = {}
    component_names = [f"x{i+1}" for i in range(n_components)]
    
    # Stage 1: Generate 2-way interactions (60% minimum to support 4-way)
    all_2way = list(combinations(range(n_components), 2))
    n_2way = int(len(all_2way) * 0.6)  # 60% of possible 2-ways to ensure enough parents
    selected_2way = np.random.choice(len(all_2way), n_2way, replace=False)
    
    print(f"\n2-WAY INTERACTIONS:")
    for idx in selected_2way:
        combo = all_2way[idx]
        # Coefficient magnitude related to parent main effects
        parent_strength = (abs(linear_coeffs[combo[0]]) + abs(linear_coeffs[combo[1]])) / 2
        coef = np.random.uniform(-1, 1) * parent_strength * 0.6  # 60% of parent average
        interactions[combo] = coef
        
        term_name = '*'.join([component_names[i] for i in combo])
        print(f"  {term_name}: {coef:+.3f}")
    
    # Stage 2: Generate 3-way interactions (hierarchically) - higher probability to support 4-way
    all_3way = list(combinations(range(n_components), 3))
    
    print(f"\n3-WAY INTERACTIONS:")
    n_3way_added = 0
    for combo_3way in all_3way:
        # Get all parent 2-ways
        parent_2ways = [
            (combo_3way[0], combo_3way[1]),
            (combo_3way[0], combo_3way[2]),
            (combo_3way[1], combo_3way[2])
        ]
        
        # Count how many parent 2-ways exist
        existing_parents = [p for p in parent_2ways if p in interactions]
        
        # Only add 3-way if at least 2 parent 2-ways exist
        if len(existing_parents) >= 2:
            # Higher probability (80%) to ensure enough 3-ways for 4-way generation
            if np.random.random() < 0.8:
                # Coefficient related to parent interactions
                parent_strength = np.mean([abs(interactions[p]) for p in existing_parents])
                coef = np.random.uniform(-1, 1) * parent_strength * 0.8  # 80% of parent average
                interactions[combo_3way] = coef
                
                term_name = '*'.join([component_names[i] for i in combo_3way])
                parent_names = ', '.join(['*'.join([component_names[i] for i in p]) for p in existing_parents])
                print(f"  {term_name}: {coef:+.3f}  (parents: {parent_names})")
                n_3way_added += 1
    
    if n_3way_added == 0:
        print(f"  (none - not enough parent 2-ways)")
    
    # Stage 3: Generate 4-way interactions (GUARANTEED at least one)
    all_4way = list(combinations(range(n_components), 4))
    
    print(f"\n4-WAY INTERACTIONS:")
    n_4way_added = 0
    
    # First pass: try to add with higher probability
    for combo_4way in all_4way:
        # Get all parent 3-ways
        parent_3ways = []
        for i in range(4):
            parent = tuple([combo_4way[j] for j in range(4) if j != i])
            parent_3ways.append(parent)
        
        # Count how many parent 3-ways exist
        existing_parents = [p for p in parent_3ways if p in interactions]
        
        # Only add 4-way if at least 2 parent 3-ways exist
        if len(existing_parents) >= 2:
            # Higher probability (70%) to ensure at least one 4-way
            if np.random.random() < 0.7:
                # Coefficient related to parent interactions
                parent_strength = np.mean([abs(interactions[p]) for p in existing_parents])
                coef = np.random.uniform(-1, 1) * parent_strength * 0.9  # 90% of parent average
                interactions[combo_4way] = coef
                
                term_name = '*'.join([component_names[i] for i in combo_4way])
                parent_names = ', '.join(['*'.join([component_names[i] for i in p]) for p in existing_parents])
                print(f"  {term_name}: {coef:+.3f}  (parents: {parent_names})")
                n_4way_added += 1
    
    # FORCE at least one 4-way if none were added
    if n_4way_added == 0:
        # Find a 4-way combo that has enough parent 3-ways
        for combo_4way in all_4way:
            parent_3ways = []
            for i in range(4):
                parent = tuple([combo_4way[j] for j in range(4) if j != i])
                parent_3ways.append(parent)
            
            existing_parents = [p for p in parent_3ways if p in interactions]
            
            if len(existing_parents) >= 2:
                # FORCE creation of this 4-way
                parent_strength = np.mean([abs(interactions[p]) for p in existing_parents])
                coef = np.random.uniform(-1, 1) * parent_strength * 0.9
                interactions[combo_4way] = coef
                
                term_name = '*'.join([component_names[i] for i in combo_4way])
                parent_names = ', '.join(['*'.join([component_names[i] for i in p]) for p in existing_parents])
                print(f"  {term_name}: {coef:+.3f}  (parents: {parent_names}) [FORCED]")
                n_4way_added += 1
                break  # Only add one forced 4-way
    
    if n_4way_added == 0:
        print(f"  (none - not enough parent 3-ways even after forced attempt)")
    
    # Create the function
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
    mixture_function.n_terms = n_components + len(interactions)
    
    # Print summary
    n_2way = sum(1 for k in interactions.keys() if len(k) == 2)
    n_3way = sum(1 for k in interactions.keys() if len(k) == 3)
    n_4way = sum(1 for k in interactions.keys() if len(k) == 4)
    
    print(f"\n" + "="*80)
    print(f"SUMMARY:")
    print(f"  Main effects: {n_components}")
    print(f"  2-way interactions: {n_2way}")
    print(f"  3-way interactions: {n_3way}")
    print(f"  4-way interactions: {n_4way}")
    print(f"  Total terms: {mixture_function.n_terms}")
    print(f"="*80)
    
    return mixture_function


def verify_hierarchy(mixture_function, component_names):
    """Verify that the function respects hierarchy"""
    interactions = mixture_function.interactions
    
    print(f"\n" + "="*80)
    print(f"HIERARCHY VERIFICATION")
    print(f"="*80)
    
    violations = []
    
    # Check 3-way terms
    for combo, coef in interactions.items():
        if len(combo) == 3:
            # Get parent 2-ways
            parent_2ways = [
                (combo[0], combo[1]),
                (combo[0], combo[2]),
                (combo[1], combo[2])
            ]
            
            existing_parents = [p for p in parent_2ways if p in interactions]
            term_name = '*'.join([component_names[i] for i in combo])
            
            if len(existing_parents) < 2:
                violations.append(f"3-way {term_name} has only {len(existing_parents)} parent 2-ways")
            else:
                parent_names = ', '.join(['*'.join([component_names[i] for i in p]) for p in existing_parents])
                print(f"✅ {term_name} has {len(existing_parents)}/3 parent 2-ways: {parent_names}")
    
    # Check 4-way terms
    for combo, coef in interactions.items():
        if len(combo) == 4:
            # Get parent 3-ways
            parent_3ways = []
            for i in range(4):
                parent = tuple([combo[j] for j in range(4) if j != i])
                parent_3ways.append(parent)
            
            existing_parents = [p for p in parent_3ways if p in interactions]
            term_name = '*'.join([component_names[i] for i in combo])
            
            if len(existing_parents) < 2:
                violations.append(f"4-way {term_name} has only {len(existing_parents)} parent 3-ways")
            else:
                parent_names = ', '.join(['*'.join([component_names[i] for i in p]) for p in existing_parents])
                print(f"✅ {term_name} has {len(existing_parents)}/4 parent 3-ways: {parent_names}")
    
    if violations:
        print(f"\n❌ HIERARCHY VIOLATIONS:")
        for v in violations:
            print(f"  {v}")
        return False
    else:
        print(f"\n✅ All interactions respect hierarchy!")
        return True


if __name__ == "__main__":
    print("="*80)
    print("GENERATING HIERARCHICAL MIXTURE FUNCTION")
    print("="*80)
    
    # Generate function
    func = generate_hierarchical_mixture_function(n_components=5, seed=42)
    
    # Verify hierarchy
    component_names = [f"x{i+1}" for i in range(5)]
    verify_hierarchy(func, component_names)
    
    # Test the function
    print(f"\n" + "="*80)
    print(f"TEST EVALUATION")
    print(f"="*80)
    
    test_point = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    result = func(test_point, noise_level=0.0)
    print(f"Test point: {test_point}")
    print(f"Function value (no noise): {result:.6f}")
    
    # Generate a few more for variety
    print(f"\n" + "="*80)
    print(f"TRYING DIFFERENT SEEDS")
    print(f"="*80)
    
    for seed in [123, 456, 789]:
        print(f"\nSeed {seed}:")
        func_temp = generate_hierarchical_mixture_function(n_components=5, seed=seed)
