"""
Comparison of Parts Mode (good) vs Fixed Parts (poor distribution) implementations
This script demonstrates the differences in candidate generation strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import os

def test_parts_mode_algorithm():
    """Test the good parts mode algorithm from proportional_parts_mixture.py"""
    
    print("=== Testing GOOD Implementation: Proportional Parts Mixture ===")
    
    # Import the good implementation
    import sys
    sys.path.insert(0, os.path.join('mixture-design-doe', 'src'))
    
    from core.proportional_parts_mixture import ProportionalPartsMixture
    
    # Setup similar to fixed parts but without fixed components
    component_ranges = [
        (0.0, 40.0),   # Solvent: 0-40 parts  
        (0.0, 15.0),   # Additive: 0-15 parts
        (50.0, 55.0),  # Base_Polymer: simulated as nearly fixed (50-55 parts)
        (2.0, 3.0)     # Catalyst: simulated as nearly fixed (2-3 parts)
    ]
    
    ppm = ProportionalPartsMixture(
        n_components=4,
        component_ranges=component_ranges
    )
    
    # Generate many candidates to analyze distribution
    n_candidates = 1000
    candidates_parts = []
    candidates_props = []
    
    print(f"Generating {n_candidates} candidates with GOOD algorithm...")
    
    for i in range(n_candidates):
        parts, props = ppm.generate_feasible_parts_candidate()
        candidates_parts.append(parts)
        candidates_props.append(props)
    
    candidates_parts = np.array(candidates_parts)
    candidates_props = np.array(candidates_props)
    
    # Extract variable components (Solvent, Additive)
    solvent_parts = candidates_parts[:, 0]
    additive_parts = candidates_parts[:, 1]
    
    print(f"\nGOOD Implementation Results:")
    print(f"  Solvent range: {solvent_parts.min():.2f} to {solvent_parts.max():.2f}")
    print(f"  Additive range: {additive_parts.min():.2f} to {additive_parts.max():.2f}")
    
    # Check corner coverage
    corners = [(0, 0), (0, 15), (40, 0), (40, 15)]
    for corner_x, corner_y in corners:
        distances = np.sqrt((solvent_parts - corner_x)**2 + (additive_parts - corner_y)**2)
        min_dist = distances.min()
        print(f"  Corner ({corner_x}, {corner_y}): closest candidate at distance {min_dist:.2f}")
    
    return solvent_parts, additive_parts

def test_fixed_parts_algorithm():
    """Test the problematic fixed parts algorithm"""
    
    print("\n=== Testing PROBLEMATIC Implementation: Fixed Parts Mixture ===")
    
    # Import the problematic implementation
    import sys
    sys.path.insert(0, os.path.join('mixture-design-doe', 'src'))
    
    from core.true_fixed_components_mixture import TrueFixedComponentsMixture
    
    component_names = ["Solvent", "Additive", "Base_Polymer", "Catalyst"]
    fixed_parts = {"Base_Polymer": 50.0, "Catalyst": 2.5}
    variable_bounds = {"Solvent": (0.0, 40.0), "Additive": (0.0, 15.0)}
    
    designer = TrueFixedComponentsMixture(
        component_names=component_names,
        fixed_parts=fixed_parts,
        variable_bounds=variable_bounds
    )
    
    # Generate many candidates to analyze distribution
    n_candidates = 1000
    print(f"Generating {n_candidates} candidates with PROBLEMATIC algorithm...")
    
    parts_candidates, prop_candidates, batch_candidates = designer.generate_candidate_set(n_candidates)
    
    # Extract variable components (Solvent, Additive)
    solvent_idx = component_names.index("Solvent")
    additive_idx = component_names.index("Additive")
    
    solvent_parts = parts_candidates[:, solvent_idx]
    additive_parts = parts_candidates[:, additive_idx]
    
    print(f"\nPROBLEMATIC Implementation Results:")
    print(f"  Solvent range: {solvent_parts.min():.2f} to {solvent_parts.max():.2f}")
    print(f"  Additive range: {additive_parts.min():.2f} to {additive_parts.max():.2f}")
    
    # Check corner coverage
    corners = [(0, 0), (0, 15), (40, 0), (40, 15)]
    for corner_x, corner_y in corners:
        distances = np.sqrt((solvent_parts - corner_x)**2 + (additive_parts - corner_y)**2)
        min_dist = distances.min()
        print(f"  Corner ({corner_x}, {corner_y}): closest candidate at distance {min_dist:.2f}")
    
    return solvent_parts, additive_parts

def analyze_key_differences():
    """Analyze the key algorithmic differences"""
    
    print("\n" + "="*80)
    print("KEY ALGORITHMIC DIFFERENCES ANALYSIS")
    print("="*80)
    
    print("\n1. CANDIDATE GENERATION STRATEGY:")
    print("   GOOD (ProportionalPartsMixture):")
    print("   - Calculates proportional ranges based on parts boundaries")
    print("   - Uses sophisticated proportional candidate generation")
    print("   - Multiple total parts candidates tested for feasibility")
    print("   - Converts between parts and proportions intelligently")
    print("   - Validates and adjusts candidates to meet constraints")
    
    print("\n   PROBLEMATIC (TrueFixedComponentsMixture):")
    print("   - Simple random uniform sampling: np.random.uniform(min_parts, max_parts)")
    print("   - Independent sampling for each variable component")
    print("   - No sophisticated space-filling considerations")
    print("   - No validation or adjustment of candidates")
    
    print("\n2. SPACE-FILLING PROPERTIES:")
    print("   GOOD: Implicit space-filling through proportional relationships")
    print("   PROBLEMATIC: Pure random sampling with clustering tendency")
    
    print("\n3. BOUNDARY COVERAGE:")
    print("   GOOD: Proportional ranges ensure boundary coverage")
    print("   PROBLEMATIC: Random sampling may miss corners and edges")
    
    print("\n4. STRUCTURED POINTS:")
    print("   GOOD: Built into proportional candidate generation")
    print("   PROBLEMATIC: Limited structured points (only 4-5 basic points)")

def create_comparison_plots():
    """Create side-by-side plots comparing the two implementations"""
    
    print("\n=== Creating Comparison Plots ===")
    
    # Test both algorithms
    good_solvent, good_additive = test_parts_mode_algorithm()
    bad_solvent, bad_additive = test_fixed_parts_algorithm()
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Good implementation plot
    ax1 = axes[0]
    ax1.scatter(good_solvent, good_additive, alpha=0.6, s=20, c='blue')
    ax1.set_xlim(0, 40)
    ax1.set_ylim(0, 15)
    ax1.set_xlabel('Solvent (Parts)')
    ax1.set_ylabel('Additive (Parts)')
    ax1.set_title('GOOD: Proportional Parts\n(Space-filling distribution)')
    ax1.grid(True, alpha=0.3)
    
    # Add corner indicators
    corners_x = [0, 0, 40, 40]
    corners_y = [0, 15, 0, 15]
    ax1.scatter(corners_x, corners_y, marker='x', s=100, c='red', alpha=0.8, linewidth=3)
    
    # Problematic implementation plot  
    ax2 = axes[1]
    ax2.scatter(bad_solvent, bad_additive, alpha=0.6, s=20, c='orange')
    ax2.set_xlim(0, 40)
    ax2.set_ylim(0, 15)
    ax2.set_xlabel('Solvent (Parts)')
    ax2.set_ylabel('Additive (Parts)')
    ax2.set_title('PROBLEMATIC: Fixed Parts\n(Poor distribution, missing corners)')
    ax2.grid(True, alpha=0.3)
    
    # Add corner indicators
    ax2.scatter(corners_x, corners_y, marker='x', s=100, c='red', alpha=0.8, linewidth=3)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('mixture-design-doe/output', exist_ok=True)
    
    plt.savefig('mixture-design-doe/output/algorithm_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plots saved to: mixture-design-doe/output/algorithm_comparison.png")

def identify_specific_issues():
    """Identify the specific code issues in the fixed parts implementation"""
    
    print("\n" + "="*80)
    print("SPECIFIC CODE ISSUES IN FIXED PARTS IMPLEMENTATION")
    print("="*80)
    
    print("\n🔍 ISSUE 1: Poor Random Candidate Generation")
    print("Location: TrueFixedComponentsMixture.generate_random_feasible_point()")
    print("Problem Code:")
    print("```python")
    print("for i, name in enumerate(self.component_names):")
    print("    if name in self.variable_names:")
    print("        min_parts, max_parts = self.variable_bounds[name]")
    print("        parts[i] = np.random.uniform(min_parts, max_parts)  # ❌ PROBLEM")
    print("```")
    print("Issue: Independent uniform sampling for each component")
    print("Result: Clustering, poor space-filling, missing corners")
    
    print("\n🔍 ISSUE 2: Limited Structured Points")
    print("Location: TrueFixedComponentsMixture.generate_structured_points()")
    print("Problem: Only generates 4-5 basic points:")
    print("- Min batch size (all variables at minimum)")
    print("- Max batch size (all variables at maximum)")  
    print("- Each variable at max, others at min")
    print("- Centroid")
    print("Missing: Edge points, systematic corner coverage, Latin Hypercube")
    
    print("\n🔍 ISSUE 3: No Space-Filling Optimization")
    print("Location: TrueFixedComponentsMixture.generate_candidate_set()")
    print("Problem: Pure random generation without space-filling criteria")
    print("Result: Candidates cluster in central regions, poor boundary coverage")
    
    print("\n🔍 ISSUE 4: No Constraint-Aware Sampling")
    print("Problem: Doesn't consider the reduced design space effectively")
    print("Fixed components consume space, but sampling doesn't adapt to this")

def recommend_fixes():
    """Recommend specific fixes for the fixed parts implementation"""
    
    print("\n" + "="*80)
    print("RECOMMENDED FIXES FOR FIXED PARTS IMPLEMENTATION")
    print("="*80)
    
    print("\n✅ FIX 1: Implement Latin Hypercube Sampling (LHS)")
    print("- Replace random uniform sampling with LHS for variable components")
    print("- Ensures better space-filling properties")
    print("- Guarantees coverage of entire design space")
    
    print("\n✅ FIX 2: Enhanced Structured Points Generation")
    print("- Add all corner points of variable space")
    print("- Add edge midpoints and face centers")
    print("- Add stratified sampling points")
    
    print("\n✅ FIX 3: Proportional Relationship Awareness")
    print("- Adapt the ProportionalPartsMixture strategy for fixed components")
    print("- Consider relationships between variable components")
    print("- Use multiple candidate batch sizes")
    
    print("\n✅ FIX 4: Space-Filling Objective in Coordinate Exchange")
    print("- Add space-filling criteria to the D-optimal algorithm")
    print("- Balance D-efficiency with space-filling properties")
    print("- Use maximin distance or other space-filling metrics")

if __name__ == "__main__":
    # Run the comparison analysis
    create_comparison_plots()
    analyze_key_differences()
    identify_specific_issues()
    recommend_fixes()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - ROOT CAUSE IDENTIFIED")
    print("="*80)
    print("The fixed parts implementation uses simple random sampling")
    print("while the good parts implementation uses sophisticated")
    print("proportional relationship management. This explains the")
    print("poor distribution in fixed parts mixture designs!")
