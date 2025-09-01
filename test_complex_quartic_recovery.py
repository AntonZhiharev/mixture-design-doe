import sys
sys.path.append('.')

import numpy as np
import random
from itertools import combinations
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ComplexMixtureFunctionGenerator:
    """Generates complex mixture functions with guaranteed 4th and 5th order interactions."""
    
    def __init__(self, n_components=5, seed=None):
        self.n_components = n_components
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_complex_quartic_function(self, noise_level=0.05):
        """
        Generate a mixture function with guaranteed quartic (4th order) and quintic (5th order) interactions.
        """
        
        # Linear coefficients (always include these)
        linear_coeffs = [3, 2, 6, 5, 4]  # Fixed for reproducibility
        
        # Quadratic interactions (select key ones)
        quadratic_interactions = {
            (0, 1): -4,    # x1*x2
            (1, 2): -6,    # x2*x3  
            (2, 3): 3,     # x3*x4
            (0, 4): -3,    # x1*x5
            (1, 4): 4,     # x2*x5
        }
        
        # Cubic interactions (select important ones)
        cubic_interactions = {
            (0, 1, 2): -5,    # x1*x2*x3
            (1, 2, 3): 7,     # x2*x3*x4
            (0, 2, 4): -4,    # x1*x3*x5
            (1, 3, 4): 6,     # x2*x4*x5
        }
        
        # **QUARTIC INTERACTIONS** (4th order) - This is what you wanted to see!
        quartic_interactions = {
            (0, 1, 2, 3): 8,     # x1*x2*x3*x4 - MAJOR 4th order interaction
            (1, 2, 3, 4): -7,    # x2*x3*x4*x5 - Another 4th order
            (0, 1, 3, 4): 5,     # x1*x2*x4*x5 - Third 4th order
        }
        
        # **QUINTIC INTERACTION** (5th order) 
        quintic_interaction = {
            (0, 1, 2, 3, 4): -6   # x1*x2*x3*x4*x5 - FULL 5th order interaction
        }
        
        # Combine all interactions
        all_interactions = {}
        all_interactions.update(quadratic_interactions)
        all_interactions.update(cubic_interactions)
        all_interactions.update(quartic_interactions)
        all_interactions.update(quintic_interaction)
        
        def complex_mixture_function(x):
            """The generated complex mixture function with guaranteed high-order interactions."""
            result = 0
            
            # Linear terms
            for i in range(self.n_components):
                result += linear_coeffs[i] * x[i]
            
            # All interaction terms
            for interaction, coeff in all_interactions.items():
                term = coeff
                for component_idx in interaction:
                    term *= x[component_idx]
                result += term
            
            # Add noise
            if noise_level > 0:
                result += np.random.normal(0, noise_level)
            
            return result
        
        # Store function metadata
        complex_mixture_function.linear_coeffs = linear_coeffs
        complex_mixture_function.interactions = list(all_interactions.keys())
        complex_mixture_function.interaction_coeffs = all_interactions
        complex_mixture_function.quartic_interactions = quartic_interactions
        complex_mixture_function.quintic_interaction = quintic_interaction
        
        return complex_mixture_function

def generate_large_strategic_design(n_components=5, n_points=60):
    """Generate large, comprehensive design specifically for complex interaction detection."""
    
    print(f"\n🎯 GENERATING LARGE STRATEGIC DESIGN: {n_points} POINTS")
    
    strategic_points = []
    
    # Strategy 1: All possible 3-way interactions (10 points)
    print("Generating 3-way interaction points...")
    three_way_combinations = list(combinations(range(n_components), 3))
    for i, (comp1, comp2, comp3) in enumerate(three_way_combinations):
        point = np.full(n_components, 0.025)
        point[comp1] = 0.3
        point[comp2] = 0.3
        point[comp3] = 0.3
        point = point / np.sum(point)
        strategic_points.append(point.tolist())
        print(f"  Point {len(strategic_points)}: x{comp1+1}*x{comp2+1}*x{comp3+1}")
    
    # Strategy 2: ALL POSSIBLE 4-WAY INTERACTIONS (5 points) - **This addresses your question!**
    print("\nGenerating 4-way interaction points (QUARTIC FOCUS)...")
    four_way_combinations = list(combinations(range(n_components), 4))
    for i, (comp1, comp2, comp3, comp4) in enumerate(four_way_combinations):
        point = np.full(n_components, 0.02)
        # Equal allocation to the four components
        point[comp1] = 0.24
        point[comp2] = 0.24
        point[comp3] = 0.24
        point[comp4] = 0.24
        point = point / np.sum(point)
        strategic_points.append(point.tolist())
        print(f"  Point {len(strategic_points)}: x{comp1+1}*x{comp2+1}*x{comp3+1}*x{comp4+1} (QUARTIC)")
    
    # Strategy 3: 5-way interaction (1 point)
    print("\nGenerating 5-way interaction point (QUINTIC FOCUS)...")
    point = np.full(n_components, 0.2)  # Equal allocation for all components
    strategic_points.append(point.tolist())
    print(f"  Point {len(strategic_points)}: x1*x2*x3*x4*x5 (QUINTIC)")
    
    # Strategy 4: High-leverage 2-way interactions (10 points)
    print("\nGenerating 2-way interaction points...")
    two_way_combinations = list(combinations(range(n_components), 2))
    for i, (comp1, comp2) in enumerate(two_way_combinations):
        point = np.full(n_components, 0.05)
        point[comp1] = 0.45
        point[comp2] = 0.45
        point = point / np.sum(point)
        strategic_points.append(point.tolist())
        print(f"  Point {len(strategic_points)}: x{comp1+1}*x{comp2+1}")
    
    # Strategy 5: Component-dominant points (5 points)
    print("\nGenerating component-dominant points...")
    for i in range(n_components):
        point = np.full(n_components, 0.05)
        point[i] = 0.8
        point = point / np.sum(point)
        strategic_points.append(point.tolist())
        print(f"  Point {len(strategic_points)}: x{i+1} dominant")
    
    # Strategy 6: Balanced interior points with perturbations
    print("\nGenerating balanced interior points...")
    remaining_points = n_points - len(strategic_points)
    for i in range(remaining_points):
        if i % 3 == 0:
            # Centroid with small perturbation
            point = np.full(n_components, 0.2)
            perturbation = np.random.normal(0, 0.02, n_components)
            point += perturbation
        elif i % 3 == 1:
            # Two components dominant
            comps = random.sample(range(n_components), 2)
            point = np.full(n_components, 0.05)
            point[comps[0]] = 0.4
            point[comps[1]] = 0.4
        else:
            # Three components moderate
            comps = random.sample(range(n_components), 3)
            point = np.full(n_components, 0.05)
            for comp in comps:
                point[comp] = 0.25
        
        point = np.abs(point)  # Ensure non-negative
        point = point / np.sum(point)  # Normalize
        strategic_points.append(point.tolist())
    
    print(f"\nTotal strategic points generated: {len(strategic_points)}")
    print(f"  - 3-way interactions: 10 points")
    print(f"  - 4-way interactions: 5 points (QUARTIC FOCUS)")
    print(f"  - 5-way interaction: 1 point (QUINTIC FOCUS)")
    print(f"  - 2-way interactions: 10 points")
    print(f"  - Component-dominant: 5 points")
    print(f"  - Balanced interior: {remaining_points} points")
    
    return strategic_points

def test_complex_quartic_recovery():
    """Test coefficient recovery on complex functions with guaranteed 4th and 5th order interactions."""
    
    print(f"{'='*90}")
    print("COMPLEX QUARTIC & QUINTIC INTERACTION RECOVERY TEST")
    print("="*90)
    
    # Generate complex function with guaranteed quartic interactions
    generator = ComplexMixtureFunctionGenerator(n_components=5, seed=42)
    complex_func = generator.generate_complex_quartic_function(noise_level=0.03)
    
    # Print detailed function structure
    print(f"🎯 COMPLEX FUNCTION STRUCTURE:")
    print(f"  Linear coefficients: {complex_func.linear_coeffs}")
    
    print(f"\n  📊 INTERACTION BREAKDOWN:")
    print(f"     Quadratic interactions: {len([k for k in complex_func.interaction_coeffs.keys() if len(k) == 2])}")
    print(f"     Cubic interactions: {len([k for k in complex_func.interaction_coeffs.keys() if len(k) == 3])}")
    print(f"     QUARTIC interactions: {len([k for k in complex_func.interaction_coeffs.keys() if len(k) == 4])} ⭐")
    print(f"     QUINTIC interactions: {len([k for k in complex_func.interaction_coeffs.keys() if len(k) == 5])} ⭐")
    
    print(f"\n  🔥 QUARTIC INTERACTIONS (4th Order):")
    for interaction, coeff in complex_func.quartic_interactions.items():
        component_names = [f"x{i+1}" for i in interaction]
        print(f"     {coeff:+.1f} * {'*'.join(component_names)}")
    
    print(f"\n  🌟 QUINTIC INTERACTION (5th Order):")
    for interaction, coeff in complex_func.quintic_interaction.items():
        component_names = [f"x{i+1}" for i in interaction]
        print(f"     {coeff:+.1f} * {'*'.join(component_names)}")
    
    print(f"\n  📋 ALL INTERACTIONS ({len(complex_func.interactions)}):")
    for interaction, coeff in complex_func.interaction_coeffs.items():
        component_names = [f"x{i+1}" for i in interaction]
        order_marker = ""
        if len(interaction) == 4:
            order_marker = " [QUARTIC]"
        elif len(interaction) == 5:
            order_marker = " [QUINTIC]"
        print(f"     {coeff:+.1f} * {'*'.join(component_names)}{order_marker}")
    
    # Generate large strategic design - **This addresses your first question about design size!**
    design_size = 70  # **MUCH LARGER DESIGN**
    print(f"\n🚀 GENERATING LARGE DESIGN: {design_size} EXPERIMENTAL POINTS")
    
    strategic_points = generate_large_strategic_design(n_points=design_size)
    
    # Also add some standard optimal design points
    from src.algorithms.jmp_style_mixture_design import JMPStyleMixtureDesignOptimizer
    design_optimizer = JMPStyleMixtureDesignOptimizer(n_components=5, model_type="extended_quadratic")
    standard_design = design_optimizer.generate_superior_design(n_runs=20, optimization_strategy="multi_objective")
    standard_points = [point.tolist() if hasattr(point, 'tolist') else point for point in standard_design['design_points']]
    
    # Combine designs
    all_design_points = strategic_points + standard_points
    all_responses = [complex_func(point) for point in all_design_points]
    
    total_points = len(all_design_points)
    print(f"\n📊 FINAL DESIGN SUMMARY:")
    print(f"  ✅ Strategic points: {len(strategic_points)}")
    print(f"  ✅ Standard optimal: {len(standard_points)}")
    print(f"  🎯 TOTAL DESIGN: {total_points} points")  # **This answers your question!**
    print(f"  📈 Points per parameter: {total_points/31:.1f} (31 total parameters)")
    
    # Enhanced analysis
    from enhanced_coefficient_recovery import EnhancedCoefficientRecovery
    recovery_analyzer = EnhancedCoefficientRecovery(n_components=5)
    results = recovery_analyzer.analyze_coefficient_recovery_enhanced(
        all_design_points, all_responses, complex_func
    )
    
    # Special analysis for quartic and quintic terms
    print(f"\n🔬 SPECIAL ANALYSIS: HIGH-ORDER INTERACTION RECOVERY")
    print("="*60)
    
    true_coeffs = recovery_analyzer._get_true_coefficient_vector(complex_func)
    estimated_coeffs = results['best_result']['coefficients']
    
    # Find quartic and quintic terms
    quartic_terms = []
    quintic_terms = []
    
    for i, name in enumerate(recovery_analyzer.coefficient_names):
        if name.count('*') == 3:  # 4-way interaction
            quartic_terms.append((i, name))
        elif name.count('*') == 4:  # 5-way interaction
            quintic_terms.append((i, name))
    
    print(f"\n🔥 QUARTIC INTERACTION RECOVERY:")
    for idx, name in quartic_terms:
        true_val = true_coeffs[idx]
        est_val = estimated_coeffs[idx]
        if abs(true_val) > 1e-10:  # Only if truly active
            error = abs(est_val - true_val)
            rel_error = (error / abs(true_val)) * 100
            detection_threshold = np.std(estimated_coeffs) * 0.05
            status = "✅ DETECTED" if abs(est_val) > detection_threshold else "❌ MISSED"
            print(f"  {name:<20} True: {true_val:+.3f} | Est: {est_val:+.6f} | Error: {rel_error:6.1f}% | {status}")
    
    print(f"\n🌟 QUINTIC INTERACTION RECOVERY:")
    for idx, name in quintic_terms:
        true_val = true_coeffs[idx]
        est_val = estimated_coeffs[idx]
        if abs(true_val) > 1e-10:  # Only if truly active
            error = abs(est_val - true_val)
            rel_error = (error / abs(true_val)) * 100
            detection_threshold = np.std(estimated_coeffs) * 0.05
            status = "✅ DETECTED" if abs(est_val) > detection_threshold else "❌ MISSED"
            print(f"  {name:<20} True: {true_val:+.3f} | Est: {est_val:+.6f} | Error: {rel_error:6.1f}% | {status}")
    
    # **Print final design points table**
    print_design_points_table(all_design_points, complex_func)
    
    return results, complex_func, all_design_points

def print_design_points_table(design_points, mixture_func):
    """Print detailed table of all design points with responses and interaction values."""
    
    print(f"\n{'='*120}")
    print("FINAL DESIGN POINTS TABLE")
    print("="*120)
    
    responses = [mixture_func(point) for point in design_points]
    
    # Calculate key interaction values for each point
    interaction_data = []
    for i, point in enumerate(design_points):
        x = np.array(point)
        
        # Calculate specific interaction terms
        x1_x2_x5 = x[0] * x[1] * x[4]          # 3rd order
        x2_x3_x4 = x[1] * x[2] * x[3]          # 3rd order  
        x1_x2_x3_x4 = x[0] * x[1] * x[2] * x[3]   # 4th order (QUARTIC)
        x2_x3_x4_x5 = x[1] * x[2] * x[3] * x[4]   # 4th order (QUARTIC)
        x1_x2_x3_x4_x5 = x[0] * x[1] * x[2] * x[3] * x[4]  # 5th order (QUINTIC)
        
        interaction_data.append({
            'x1_x2_x5': x1_x2_x5,
            'x2_x3_x4': x2_x3_x4,
            'x1_x2_x3_x4': x1_x2_x3_x4,
            'x2_x3_x4_x5': x2_x3_x4_x5,
            'x1_x2_x3_x4_x5': x1_x2_x3_x4_x5
        })
    
    # Print header
    print(f"{'Run':<4} {'x1':<8} {'x2':<8} {'x3':<8} {'x4':<8} {'x5':<8} {'Response':<10} {'x1*x2*x5':<10} {'x2*x3*x4':<10} {'x1*x2*x3*x4':<12} {'x2*x3*x4*x5':<12} {'x1*x2*x3*x4*x5':<14}")
    print("-" * 120)
    
    # Print each design point
    for i, (point, response, interactions) in enumerate(zip(design_points, responses, interaction_data)):
        run_num = i + 1
        x1, x2, x3, x4, x5 = point
        
        print(f"{run_num:<4} {x1:<8.4f} {x2:<8.4f} {x3:<8.4f} {x4:<8.4f} {x5:<8.4f} {response:<10.4f} "
              f"{interactions['x1_x2_x5']:<10.6f} {interactions['x2_x3_x4']:<10.6f} "
              f"{interactions['x1_x2_x3_x4']:<12.8f} {interactions['x2_x3_x4_x5']:<12.8f} "
              f"{interactions['x1_x2_x3_x4_x5']:<14.10f}")
    
    print("-" * 120)
    
    # Summary statistics
    total_points = len(design_points)
    
    # Count points with significant interaction values
    significant_3rd = sum(1 for interactions in interaction_data 
                         if max(interactions['x1_x2_x5'], interactions['x2_x3_x4']) > 0.001)
    significant_4th = sum(1 for interactions in interaction_data 
                         if max(interactions['x1_x2_x3_x4'], interactions['x2_x3_x4_x5']) > 0.0001)
    significant_5th = sum(1 for interactions in interaction_data 
                         if interactions['x1_x2_x3_x4_x5'] > 0.00001)
    
    print(f"\n📊 DESIGN POINTS SUMMARY:")
    print(f"  Total experimental points: {total_points}")
    print(f"  Points with significant 3rd-order interactions (>0.001): {significant_3rd}/{total_points} ({100*significant_3rd/total_points:.1f}%)")
    print(f"  Points with significant 4th-order interactions (>0.0001): {significant_4th}/{total_points} ({100*significant_4th/total_points:.1f}%)")
    print(f"  Points with significant 5th-order interactions (>0.00001): {significant_5th}/{total_points} ({100*significant_5th/total_points:.1f}%)")
    
    # Interaction value statistics
    all_3rd = [max(interactions['x1_x2_x5'], interactions['x2_x3_x4']) for interactions in interaction_data]
    all_4th = [max(interactions['x1_x2_x3_x4'], interactions['x2_x3_x4_x5']) for interactions in interaction_data]
    all_5th = [interactions['x1_x2_x3_x4_x5'] for interactions in interaction_data]
    
    print(f"\n📈 INTERACTION VALUE STATISTICS:")
    print(f"  3rd-order: Max={max(all_3rd):.6f}, Mean={np.mean(all_3rd):.6f}, Std={np.std(all_3rd):.6f}")
    print(f"  4th-order: Max={max(all_4th):.8f}, Mean={np.mean(all_4th):.8f}, Std={np.std(all_4th):.8f}")
    print(f"  5th-order: Max={max(all_5th):.10f}, Mean={np.mean(all_5th):.10f}, Std={np.std(all_5th):.10f}")
    
    # Response statistics
    print(f"\n📋 RESPONSE STATISTICS:")
    print(f"  Response range: [{min(responses):.4f}, {max(responses):.4f}]")
    print(f"  Response mean: {np.mean(responses):.4f}")
    print(f"  Response std: {np.std(responses):.4f}")

def test_multiple_design_sizes():
    """Test how performance scales with design size."""
    
    print(f"\n{'='*80}")
    print("DESIGN SIZE SCALING ANALYSIS")
    print("="*80)
    
    # Test different design sizes
    design_sizes = [40, 60, 80, 100]
    
    # Use same complex function for all tests
    generator = ComplexMixtureFunctionGenerator(n_components=5, seed=42)
    complex_func = generator.generate_complex_quartic_function(noise_level=0.03)
    
    results_summary = []
    
    for size in design_sizes:
        print(f"\n🔬 Testing design size: {size} points")
        
        # Generate design of specified size
        strategic_points = generate_large_strategic_design(n_points=size-15)
        
        # Add standard points
        from src.algorithms.jmp_style_mixture_design import JMPStyleMixtureDesignOptimizer
        design_optimizer = JMPStyleMixtureDesignOptimizer(n_components=5, model_type="extended_quadratic")
        standard_design = design_optimizer.generate_superior_design(n_runs=15, optimization_strategy="multi_objective")
        standard_points = [point.tolist() if hasattr(point, 'tolist') else point for point in standard_design['design_points']]
        
        all_points = strategic_points + standard_points
        all_responses = [complex_func(point) for point in all_points]
        
        print(f"  Actual design size: {len(all_points)} points")
        
        # Quick analysis
        from enhanced_coefficient_recovery import EnhancedCoefficientRecovery
        analyzer = EnhancedCoefficientRecovery(n_components=5)
        
        # Build design matrix and fit simple ElasticNet
        X = analyzer.build_design_matrix(all_points)
        y = np.array(all_responses)
        
        model = ElasticNetCV(alphas=np.logspace(-8, 0, 15), l1_ratio=[0.1, 0.3, 0.5], 
                            fit_intercept=False, cv=3, max_iter=10000)
        model.fit(X, y)
        
        # Calculate recovery stats
        true_coeffs = analyzer._get_true_coefficient_vector(complex_func)
        estimated_coeffs = model.coef_
        
        true_active_indices = np.where(np.abs(true_coeffs) > 1e-10)[0]
        detection_threshold = np.std(estimated_coeffs) * 0.05
        detected = sum(1 for idx in true_active_indices if abs(estimated_coeffs[idx]) > detection_threshold)
        
        detection_rate = detected / len(true_active_indices) * 100
        r2 = r2_score(y, model.predict(X))
        
        results_summary.append({
            'size': len(all_points),
            'detection_rate': detection_rate,
            'r2': r2
        })
        
        print(f"  Detection rate: {detection_rate:.1f}%")
        print(f"  R² score: {r2:.4f}")
    
    print(f"\n📊 DESIGN SIZE SCALING SUMMARY:")
    print(f"{'Size':<8} {'Detection %':<12} {'R² Score':<10}")
    print("-" * 35)
    for result in results_summary:
        print(f"{result['size']:<8} {result['detection_rate']:<12.1f} {result['r2']:<10.4f}")
    
    return results_summary

if __name__ == "__main__":
    # Test complex quartic recovery
    print("🎯 PART 1: Complex Quartic & Quintic Recovery")
    results, func, design = test_complex_quartic_recovery()
    
    print("\n" + "="*80)
    print("🎯 PART 2: Design Size Scaling Analysis")
    scaling_results = test_multiple_design_sizes()
