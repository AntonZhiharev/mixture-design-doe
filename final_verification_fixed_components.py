"""
Final verification that fixed components are working correctly
Accounts for floating-point precision issues
"""

import numpy as np
from enhanced_mixture_designs import EnhancedMixtureDesign

def final_verification():
    print("=== Final Verification: Fixed Components Working Correctly ===\n")
    
    # Create enhanced mixture design with fixed components
    enhanced_design = EnhancedMixtureDesign(
        n_components=4,
        component_names=['Polymer_A', 'Polymer_B', 'Stabilizer', 'Catalyst'],
        component_bounds=[(1, 10), (2, 8), (0.5, 3), (0.1, 1)],
        use_parts_mode=True,
        fixed_components={'Stabilizer': 2.0, 'Catalyst': 0.5}
    )
    
    # Test multiple design types and run numbers
    test_cases = [
        ("simplex-lattice", 8, "centroid"),
        ("simplex-lattice", 15, "d-optimal"),
        ("d-optimal", 12, None),
        ("space-filling", 20, None)
    ]
    
    all_passed = True
    
    for design_type, n_runs, aug_strategy in test_cases:
        print(f"Testing {design_type} with {n_runs} runs...")
        
        kwargs = {
            'design_type': design_type,
            'n_runs': n_runs,
            'model_type': 'quadratic',
            'random_seed': 42
        }
        if aug_strategy:
            kwargs['augment_strategy'] = aug_strategy
        
        design = enhanced_design.generate_mixture_design(**kwargs)
        
        # Check fixed components with proper tolerance
        stabilizer_values = design[:, 2]
        catalyst_values = design[:, 3]
        
        # Use allclose to check if all values are essentially identical
        stabilizer_fixed = np.allclose(stabilizer_values, stabilizer_values[0], atol=1e-10)
        catalyst_fixed = np.allclose(catalyst_values, catalyst_values[0], atol=1e-10)
        
        # Check if they match expected fixed values
        expected_stabilizer = enhanced_design.fixed_components['Stabilizer']
        expected_catalyst = enhanced_design.fixed_components['Catalyst']
        
        stabilizer_correct = np.allclose(stabilizer_values, expected_stabilizer, atol=1e-10)
        catalyst_correct = np.allclose(catalyst_values, expected_catalyst, atol=1e-10)
        
        # Check sums equal 1
        sums = np.sum(design, axis=1)
        sums_correct = np.allclose(sums, 1.0, atol=1e-10)
        
        test_passed = stabilizer_fixed and catalyst_fixed and stabilizer_correct and catalyst_correct and sums_correct
        
        print(f"  Shape: {design.shape}")
        print(f"  Stabilizer fixed: {'✓' if stabilizer_fixed else '❌'}")
        print(f"  Catalyst fixed: {'✓' if catalyst_fixed else '❌'}")
        print(f"  Stabilizer correct value: {'✓' if stabilizer_correct else '❌'}")
        print(f"  Catalyst correct value: {'✓' if catalyst_correct else '❌'}")
        print(f"  Sums = 1.0: {'✓' if sums_correct else '❌'}")
        print(f"  Overall: {'✓ PASS' if test_passed else '❌ FAIL'}")
        print()
        
        if not test_passed:
            all_passed = False
            # Show actual values for debugging
            print(f"    Stabilizer range: [{stabilizer_values.min():.2e}, {stabilizer_values.max():.2e}]")
            print(f"    Catalyst range: [{catalyst_values.min():.2e}, {catalyst_values.max():.2e}]")
            print()
    
    # Test original question: different numbers of runs for Simplex Lattice
    print("="*60)
    print("ANSWERING ORIGINAL QUESTION:")
    print("Can you set different numbers of runs for Simplex Lattice DOE?")
    print("="*60)
    
    simplex_runs_test = [5, 8, 12, 15, 20, 25]
    
    for n_runs in simplex_runs_test:
        try:
            design = enhanced_design.generate_mixture_design(
                design_type="simplex-lattice",
                n_runs=n_runs,
                model_type="quadratic",
                augment_strategy="d-optimal",
                random_seed=42
            )
            
            # Verify fixed components
            stabilizer_fixed = np.allclose(design[:, 2], design[0, 2], atol=1e-10)
            catalyst_fixed = np.allclose(design[:, 3], design[0, 3], atol=1e-10)
            fixed_ok = stabilizer_fixed and catalyst_fixed
            
            print(f"✓ {n_runs} runs: Generated {design.shape[0]} runs, Fixed components: {'✓' if fixed_ok else '❌'}")
            
        except Exception as e:
            print(f"❌ {n_runs} runs: Failed - {str(e)}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Fixed components are working correctly!")
        print("✓ Yes, you CAN set different numbers of runs for Simplex Lattice DOE!")
        print("✓ Fixed components remain constant across all runs!")
    else:
        print("❌ Some tests failed - further investigation needed")
    print(f"{'='*60}")

if __name__ == "__main__":
    final_verification()
