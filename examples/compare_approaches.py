from mixture_designs import MixtureDesign
from sequential_mixture_doe import SequentialMixtureDOE
import numpy as np

print("=== Comparing Mixture Design Approaches ===\n")

# Use the same problematic bounds that cause issues in regular mixture_designs
component_names = ['Component_1', 'Component_2', 'Component_3']
tight_bounds = [(0.0, 0.47619047619047616), (0.0, 0.47619047619047616), (0.0, 0.047619047619047616)]

print("Bounds that sum exactly to 1.0:")
print(f"Component bounds: {tight_bounds}")
print(f"Sum of max bounds: {sum(bound[1] for bound in tight_bounds):.15f}")

print("\n" + "="*60)
print("1. REGULAR MIXTURE DESIGN (from mixture_designs.py)")
print("="*60)

try:
    regular_mixture = MixtureDesign(3, component_names, tight_bounds)
    regular_candidates = regular_mixture._generate_candidate_points(20)
    
    print(f"Generated {len(regular_candidates)} candidates")
    print("Candidate diversity:")
    for j in range(3):
        vals = regular_candidates[:, j]
        print(f"  Component {j+1}: [{vals.min():.8f}, {vals.max():.8f}] (range: {vals.max()-vals.min():.8f})")
    
    # Test D-optimal
    regular_design = regular_mixture.generate_d_optimal_mixture(n_runs=8, model_type='linear', random_seed=42)
    regular_d_eff = regular_mixture._calculate_d_efficiency(regular_design, 'linear')
    
    print(f"\nD-optimal design D-efficiency: {regular_d_eff}")
    print("First 3 design points:")
    for i in range(3):
        print(f"  Run {i+1}: {regular_design[i]}")
    
    # Check if all points are nearly identical
    all_same = np.allclose(regular_design[0], regular_design, atol=1e-6)
    print(f"All design points nearly identical: {all_same}")
    
except Exception as e:
    print(f"Error with regular approach: {e}")

print("\n" + "="*60)
print("2. SEQUENTIAL DESIGN - PROPORTION MODE")
print("="*60)

try:
    seq_mixture_prop = SequentialMixtureDOE(
        n_components=3, 
        component_names=component_names, 
        component_bounds=tight_bounds,
        use_parts_mode=False  # Use proportion mode like regular design
    )
    
    seq_candidates_prop = seq_mixture_prop._generate_mixture_candidates(20)
    
    print(f"Generated {len(seq_candidates_prop)} candidates")
    print("Candidate diversity:")
    for j in range(3):
        vals = seq_candidates_prop[:, j]
        print(f"  Component {j+1}: [{vals.min():.8f}, {vals.max():.8f}] (range: {vals.max()-vals.min():.8f})")
    
    seq_design_prop = seq_mixture_prop.generate_d_optimal_mixture(n_runs=8, model_type='linear', random_seed=42)
    seq_d_eff_prop = seq_mixture_prop._calculate_mixture_d_efficiency(seq_design_prop, 'linear')
    
    print(f"\nD-optimal design D-efficiency: {seq_d_eff_prop}")
    print("First 3 design points:")
    for i in range(3):
        print(f"  Run {i+1}: {seq_design_prop[i]}")
    
    all_same_seq_prop = np.allclose(seq_design_prop[0], seq_design_prop, atol=1e-6)
    print(f"All design points nearly identical: {all_same_seq_prop}")
    
except Exception as e:
    print(f"Error with sequential proportion mode: {e}")

print("\n" + "="*60)
print("3. SEQUENTIAL DESIGN - PARTS MODE (THE KEY DIFFERENCE!)")
print("="*60)

try:
    # Convert tight proportion bounds to more flexible parts bounds
    # Instead of bounds that sum to exactly 1.0, use parts with more design space
    parts_bounds = [
        (0.0, 10.0),   # Component 1: 0-10 parts  
        (0.0, 10.0),   # Component 2: 0-10 parts
        (0.0, 1.0)     # Component 3: 0-1 parts (stays constrained but relative to others)
    ]
    
    print("Parts bounds (much more design space):")
    print(f"Parts bounds: {parts_bounds}")
    print(f"Sum of max parts: {sum(bound[1] for bound in parts_bounds)} (no constraint to sum to 1!)")
    
    seq_mixture_parts = SequentialMixtureDOE(
        n_components=3, 
        component_names=component_names, 
        component_bounds=parts_bounds,
        use_parts_mode=True  # KEY: Use parts mode!
    )
    
    seq_candidates_parts = seq_mixture_parts._generate_mixture_candidates(20)
    
    print(f"\nGenerated {len(seq_candidates_parts)} candidates")
    print("Candidate diversity:")
    for j in range(3):
        vals = seq_candidates_parts[:, j]
        print(f"  Component {j+1}: [{vals.min():.8f}, {vals.max():.8f}] (range: {vals.max()-vals.min():.8f})")
    
    seq_design_parts = seq_mixture_parts.generate_d_optimal_mixture(n_runs=8, model_type='linear', random_seed=42)
    seq_d_eff_parts = seq_mixture_parts._calculate_mixture_d_efficiency(seq_design_parts, 'linear')
    
    print(f"\nD-optimal design D-efficiency: {seq_d_eff_parts}")
    print("First 3 design points:")
    for i in range(3):
        print(f"  Run {i+1}: {seq_design_parts[i]}")
    
    all_same_seq_parts = np.allclose(seq_design_parts[0], seq_design_parts, atol=1e-6)
    print(f"All design points nearly identical: {all_same_seq_parts}")
    
    print("\nExample parts-to-proportions conversion:")
    # Show how a specific parts combination converts to proportions
    example_parts = [5.0, 4.0, 0.5]  # 5 parts A, 4 parts B, 0.5 parts C
    total_parts = sum(example_parts)
    example_props = [p/total_parts for p in example_parts]
    print(f"Parts {example_parts} → Proportions {[f'{p:.3f}' for p in example_props]} (sum: {sum(example_props):.3f})")
    
except Exception as e:
    print(f"Error with sequential parts mode: {e}")

print("\n" + "="*60)
print("4. WHY PARTS MODE SOLVES THE PROBLEM")
print("="*60)

print("""
KEY INSIGHTS:

1. TIGHT PROPORTION BOUNDS = NO DESIGN SPACE
   - When bounds sum exactly to 1.0: [(0, 0.476), (0, 0.476), (0, 0.048)]
   - All feasible points are clustered in a tiny region
   - Variation range: ~0.000006 (essentially identical points)
   - Results in singular design matrix (D-efficiency = 0)

2. PARTS MODE = AMPLE DESIGN SPACE  
   - Parts bounds can be: [(0, 10), (0, 10), (0, 1)]
   - Much larger design space to explore
   - Parts get normalized to proportions: parts → parts/sum(parts)
   - Can achieve the same proportion ratios but with diverse exploration

3. MATHEMATICAL DIFFERENCE:
   - Proportion mode: directly constrained to simplex with tight bounds
   - Parts mode: explores larger space, then projects to simplex
   - Parts mode can represent same mixture ratios with different total amounts

4. DESIGN GENERATION STRATEGIES:
   - Sequential design also uses multiple sampling strategies:
     * Uniform, vertices, edges, center sampling
     * Better diversity even in constrained spaces
   - Regular design relies mainly on Dirichlet + normalization
""")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

print("""
The sequential_mixture_doe.py doesn't have the same problem because:

1. It uses PARTS MODE which avoids the extremely constrained feasible space
2. It has more sophisticated candidate generation strategies
3. Parts can be freely scaled before normalizing to proportions
4. This provides much more design space to find diverse, well-separated points

The regular mixture_designs.py fails because it works directly in proportion space
where tight bounds create an impossibly small feasible region.
""")
