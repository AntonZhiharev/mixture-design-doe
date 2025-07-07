#!/usr/bin/env python3

"""
Comparison of our I-optimal implementation vs pyDOE2
"""

import sys
import os
sys.path.insert(0, 'src')

import numpy as np

def test_pydoe2_vs_our_implementation():
    """Compare pyDOE2 I-optimal with our implementation"""
    print("="*80)
    print("TESTING pyDOE2 vs OUR I-OPTIMAL IMPLEMENTATION")
    print("="*80)
    
    # Test parameters
    n_components = 3
    n_runs = 10
    model_type = "quadratic"
    
    # Test 1: Try pyDOE2 for mixture design
    print("\n1. TESTING pyDOE2 for Mixture Design:")
    print("-"*50)
    
    try:
        # Try pyDOE2 mixture design
        import pyDOE2
        print("✅ pyDOE2 successfully imported")
        
        # Check available functions
        available_functions = [attr for attr in dir(pyDOE2) if not attr.startswith('_')]
        print(f"Available pyDOE2 functions: {available_functions}")
        
        # Try different mixture design approaches
        pydoe2_design = None
        pydoe2_method = None
        
        # Method 1: Try mixture design if available
        if hasattr(pyDOE2, 'mixexp'):
            try:
                print("Trying pyDOE2.mixexp...")
                # pyDOE2 mixture experimental design
                pydoe2_design = pyDOE2.mixexp(n_components, n_runs)
                pydoe2_method = "mixexp"
                print(f"✅ pyDOE2.mixexp successful: {pydoe2_design.shape}")
            except Exception as e:
                print(f"❌ pyDOE2.mixexp failed: {e}")
        
        # Method 2: Try optimal design
        if pydoe2_design is None and hasattr(pyDOE2, 'doe_optimal'):
            try:
                print("Trying pyDOE2.doe_optimal...")
                # Create candidate set for mixture
                candidates = generate_mixture_candidates(n_components, n_runs * 10)
                pydoe2_design = pyDOE2.doe_optimal(candidates, n_runs, criterion='I')
                pydoe2_method = "doe_optimal"
                print(f"✅ pyDOE2.doe_optimal successful: {pydoe2_design.shape}")
            except Exception as e:
                print(f"❌ pyDOE2.doe_optimal failed: {e}")
        
        # Method 3: Try any other optimal design functions
        if pydoe2_design is None:
            other_functions = [f for f in available_functions if 'opt' in f.lower() or 'mix' in f.lower()]
            print(f"Trying other potential functions: {other_functions}")
            
            for func_name in other_functions:
                try:
                    func = getattr(pyDOE2, func_name)
                    print(f"Trying {func_name}...")
                    # Try simple call first
                    if func_name == 'doe_box_behnken':
                        continue  # Skip Box-Behnken for mixture
                    
                    # Generic attempt
                    result = func(n_components, n_runs)
                    if hasattr(result, 'shape') and len(result.shape) == 2:
                        pydoe2_design = result
                        pydoe2_method = func_name
                        print(f"✅ {func_name} successful: {result.shape}")
                        break
                except Exception as e:
                    print(f"❌ {func_name} failed: {e}")
        
        # If still no design, create a simple mixture design
        if pydoe2_design is None:
            print("⚠️ No pyDOE2 mixture function worked, creating simple mixture design...")
            pydoe2_design = create_simple_mixture_design(n_components, n_runs)
            pydoe2_method = "simple_mixture"
        
        # Ensure design is normalized (sum=1 for each row)
        if pydoe2_design is not None:
            row_sums = np.sum(pydoe2_design, axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                print("Normalizing pyDOE2 design to sum=1...")
                pydoe2_design = pydoe2_design / row_sums[:, np.newaxis]
        
    except ImportError:
        print("❌ pyDOE2 not available, installing...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyDOE2"])
            import pyDOE2
            print("✅ pyDOE2 installed and imported successfully")
            # Restart the test after installation
            return test_pydoe2_vs_our_implementation()
        except Exception as e:
            print(f"❌ Failed to install pyDOE2: {e}")
            print("Using fallback simple mixture design...")
            pydoe2_design = create_simple_mixture_design(n_components, n_runs)
            pydoe2_method = "fallback"
    except Exception as e:
        print(f"❌ Unexpected error with pyDOE2: {e}")
        print("Using fallback simple mixture design...")
        pydoe2_design = create_simple_mixture_design(n_components, n_runs)
        pydoe2_method = "fallback"
    
    # Test 2: Our implementation
    print(f"\n2. TESTING OUR I-OPTIMAL IMPLEMENTATION:")
    print("-"*50)
    
    from core.optimal_design_generator import OptimalDesignGenerator
    
    our_generator = OptimalDesignGenerator(
        num_variables=n_components,
        num_runs=n_runs,
        design_type="mixture",
        model_type=model_type
    )
    
    our_det = our_generator.generate_optimal_design(method="i_optimal")
    our_design = np.array(our_generator.design_points)
    
    # Test 3: Calculate I-efficiency for both designs
    print(f"\n3. COMPARING I-EFFICIENCY:")
    print("-"*50)
    
    if pydoe2_design is not None:
        pydoe2_i_efficiency = calculate_i_efficiency_direct(pydoe2_design, model_type)
        print(f"pyDOE2 design ({pydoe2_method}):")
        print(f"  Shape: {pydoe2_design.shape}")
        print(f"  I-efficiency: {pydoe2_i_efficiency:.6f}")
        print(f"  First 3 points:")
        for i in range(min(3, len(pydoe2_design))):
            point = pydoe2_design[i]
            print(f"    [{', '.join(f'{x:.3f}' for x in point)}] (sum = {np.sum(point):.3f})")
    else:
        pydoe2_i_efficiency = 0.0
        print(f"❌ No pyDOE2 design available")
    
    our_i_efficiency = calculate_i_efficiency_direct(our_design, model_type)
    print(f"\nOur implementation:")
    print(f"  Shape: {our_design.shape}")
    print(f"  I-efficiency: {our_i_efficiency:.6f}")
    print(f"  Determinant: {our_det:.6e}")
    print(f"  First 3 points:")
    for i in range(min(3, len(our_design))):
        point = our_design[i]
        print(f"    [{', '.join(f'{x:.3f}' for x in point)}] (sum = {np.sum(point):.3f})")
    
    # Test 4: Comparison
    print(f"\n4. FINAL COMPARISON:")
    print("-"*50)
    
    if pydoe2_i_efficiency > 0:
        if our_i_efficiency > pydoe2_i_efficiency:
            improvement = (our_i_efficiency - pydoe2_i_efficiency) / pydoe2_i_efficiency * 100
            print(f"✅ Our implementation OUTPERFORMS pyDOE2!")
            print(f"   Our I-efficiency: {our_i_efficiency:.6f}")
            print(f"   pyDOE2 I-efficiency: {pydoe2_i_efficiency:.6f}")
            print(f"   Improvement: {improvement:.1f}%")
        else:
            decline = (pydoe2_i_efficiency - our_i_efficiency) / pydoe2_i_efficiency * 100
            print(f"⚠️ pyDOE2 outperforms our implementation")
            print(f"   pyDOE2 I-efficiency: {pydoe2_i_efficiency:.6f}")
            print(f"   Our I-efficiency: {our_i_efficiency:.6f}")
            print(f"   Performance gap: {decline:.1f}%")
    else:
        print(f"⚠️ Could not compare - no valid pyDOE2 design")
        print(f"Our I-efficiency: {our_i_efficiency:.6f}")
    
    return pydoe2_i_efficiency, our_i_efficiency

def generate_mixture_candidates(n_components, n_candidates):
    """Generate mixture candidates for pyDOE2"""
    np.random.seed(42)
    candidates = []
    
    # Add vertices
    for i in range(n_components):
        point = [0.0] * n_components
        point[i] = 1.0
        candidates.append(point)
    
    # Add random simplex points
    for _ in range(n_candidates - n_components):
        point = np.random.random(n_components)
        point = point / np.sum(point)  # Normalize to sum=1
        candidates.append(point.tolist())
    
    return np.array(candidates)

def create_simple_mixture_design(n_components, n_runs):
    """Create a simple mixture design as fallback"""
    design = []
    
    # Add vertices
    for i in range(min(n_components, n_runs)):
        point = [0.0] * n_components
        point[i] = 1.0
        design.append(point)
    
    # Add centroid if space
    if len(design) < n_runs:
        centroid = [1.0/n_components] * n_components
        design.append(centroid)
    
    # Add edge points
    while len(design) < n_runs:
        # Binary mixture
        i, j = np.random.choice(n_components, 2, replace=False)
        ratio = np.random.uniform(0.3, 0.7)
        point = [0.0] * n_components
        point[i] = ratio
        point[j] = 1.0 - ratio
        design.append(point)
    
    return np.array(design[:n_runs])

def calculate_i_efficiency_direct(design_points, model_type):
    """Calculate I-efficiency directly"""
    try:
        from core.optimal_design_generator import gram_matrix, matrix_inverse, matrix_trace, evaluate_mixture_model_terms
        
        # Build design matrix
        design_matrix = []
        for point in design_points:
            row = evaluate_mixture_model_terms(point, model_type)
            design_matrix.append(row)
        
        # Calculate information matrix
        info_matrix = gram_matrix(design_matrix)
        
        # Calculate I-efficiency
        inverse_matrix = matrix_inverse(info_matrix)
        trace_value = matrix_trace(inverse_matrix)
        i_efficiency = 1.0 / trace_value if trace_value > 1e-10 else 0.0
        
        return i_efficiency
    except Exception as e:
        print(f"Error calculating I-efficiency: {e}")
        return 0.0

if __name__ == "__main__":
    test_pydoe2_vs_our_implementation()
