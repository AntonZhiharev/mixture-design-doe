"""
Investigate D-efficiency issue in parts mode specifically
Since parts don't have sum constraint, full quadratic model applies
"""

import numpy as np
import pandas as pd
from src.core.simplified_mixture_design import SimplexLatticeDesign, DOptimalMixtureDesign
from src.core.optimal_design_generator import OptimalDesignGenerator

def test_parts_mode_comparison():
    """Test D-efficiency comparison in parts mode for quadratic model with 9 runs"""
    
    print("="*80)
    print("INVESTIGATING D-EFFICIENCY IN PARTS MODE")
    print("="*80)
    
    # Parameters exactly as user specified
    n_components = 3
    n_runs = 9
    model_type = "quadratic"
    
    # Define parts bounds (typical parts ranges)
    component_bounds = [
        (0.1, 10.0),  # Component 1 parts range
        (0.2, 8.0),   # Component 2 parts range  
        (0.5, 5.0)    # Component 3 parts range
    ]
    
    print(f"Setup: {n_components} components, {n_runs} runs, {model_type} model")
    print(f"Parts bounds: {component_bounds}")
    
    # Test 1: Simplex Lattice in parts mode
    print(f"\n{'='*60}")
    print("1. SIMPLEX LATTICE DESIGN IN PARTS MODE")
    print("="*60)
    
    lattice_design = SimplexLatticeDesign(
        n_components=n_components,
        use_parts_mode=True,
        component_bounds=component_bounds
    )
    
    lattice_df = lattice_design.generate_design(degree=2)
    print(f"Simplex lattice generated {len(lattice_df)} points")
    print("\nLattice design (proportions):")
    print(lattice_df)
    
    # Get parts design if available
    if hasattr(lattice_design, 'parts_design') and lattice_design.parts_design is not None:
        print(f"\nLattice design (parts):")
        for i, row in enumerate(lattice_design.parts_design):
            print(f"  Run {i+1}: [{', '.join(f'{x:.3f}' for x in row)}]")
    
    # Test 2: D-Optimal in parts mode  
    print(f"\n{'='*60}")
    print("2. D-OPTIMAL DESIGN IN PARTS MODE")
    print("="*60)
    
    dopt_design = DOptimalMixtureDesign(
        n_components=n_components,
        use_parts_mode=True,
        component_bounds=component_bounds
    )
    
    dopt_df = dopt_design.generate_design(n_runs=n_runs, model_type=model_type)
    print(f"D-optimal generated {len(dopt_df)} points")
    print("\nD-optimal design (proportions):")
    print(dopt_df)
    
    # Get parts design if available
    if hasattr(dopt_design, 'parts_design') and dopt_design.parts_design is not None:
        print(f"\nD-optimal design (parts):")
        for i, row in enumerate(dopt_design.parts_design):
            print(f"  Run {i+1}: [{', '.join(f'{x:.3f}' for x in row)}]")
    
    # Test 3: Calculate D-efficiency properly for parts mode
    print(f"\n{'='*60}")
    print("3. D-EFFICIENCY CALCULATION FOR PARTS")
    print("="*60)
    
    def calculate_d_efficiency_parts(parts_design, model_type="quadratic"):
        """Calculate D-efficiency for parts-based design using full quadratic model"""
        
        if parts_design is None or len(parts_design) == 0:
            print("No parts design available")
            return 0.0
            
        n_runs, n_vars = parts_design.shape
        print(f"Parts design shape: {parts_design.shape}")
        
        # For parts mode, use full polynomial model (no sum constraint)
        if model_type == "quadratic":
            # Build full quadratic model matrix
            model_terms = []
            
            # Linear terms
            for i in range(n_vars):
                model_terms.append(parts_design[:, i])
            
            # Quadratic terms (pure squares)
            for i in range(n_vars):
                model_terms.append(parts_design[:, i]**2)
            
            # Interaction terms  
            for i in range(n_vars):
                for j in range(i+1, n_vars):
                    model_terms.append(parts_design[:, i] * parts_design[:, j])
            
            X = np.column_stack(model_terms)
            n_params = X.shape[1]
            
            print(f"Full quadratic model for parts:")
            print(f"  Linear terms: {n_vars}")
            print(f"  Quadratic terms: {n_vars}")  
            print(f"  Interaction terms: {n_vars * (n_vars - 1) // 2}")
            print(f"  Total parameters: {n_params}")
            
        else:
            X = parts_design
            n_params = n_vars
        
        # Calculate information matrix and determinant
        try:
            XTX = X.T @ X
            det_value = np.linalg.det(XTX)
            
            print(f"Determinant: {det_value:.8f}")
            
            if det_value <= 0:
                print("Non-positive determinant - singular design")
                return 0.0
            
            # D-efficiency
            d_efficiency = (det_value / n_runs) ** (1 / n_params)
            
            print(f"D-efficiency = (det/n_runs)^(1/p) = ({det_value:.8f}/{n_runs})^(1/{n_params}) = {d_efficiency:.6f}")
            
            return d_efficiency
            
        except Exception as e:
            print(f"Error calculating D-efficiency: {e}")
            return 0.0
    
    # Calculate D-efficiency for both designs in parts mode
    print(f"\nSimplex Lattice D-efficiency (parts mode):")
    lattice_parts = lattice_design.get_parts_design()
    lattice_d_eff = calculate_d_efficiency_parts(lattice_parts, model_type)
    
    print(f"\nD-Optimal D-efficiency (parts mode):")
    dopt_parts = dopt_design.get_parts_design()
    dopt_d_eff = calculate_d_efficiency_parts(dopt_parts, model_type)
    
    # Test 4: Direct comparison with OptimalDesignGenerator
    print(f"\n{'='*60}")
    print("4. DIRECT OPTIMALDESIGNGENERATOR TEST")
    print("="*60)
    
    # Use OptimalDesignGenerator directly in parts bounds
    print(f"Testing OptimalDesignGenerator with parts bounds...")
    
    generator = OptimalDesignGenerator(
        num_variables=n_components,
        model_type=model_type,
        num_runs=n_runs,
        component_ranges=component_bounds
    )
    
    final_det = generator.generate_optimal_design()
    
    # Get the D-efficiency from the generator
    if hasattr(generator, '_last_generator'):
        generator_d_eff = generator._last_generator._calculate_d_efficiency(
            np.array(generator.design_points), model_type)
        print(f"OptimalDesignGenerator D-efficiency: {generator_d_eff:.6f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    
    print(f"Simplex Lattice D-efficiency:     {lattice_d_eff:.6f}")
    print(f"D-Optimal D-efficiency:           {dopt_d_eff:.6f}")
    
    if lattice_d_eff > 0 and dopt_d_eff > 0:
        if dopt_d_eff > lattice_d_eff:
            improvement = (dopt_d_eff / lattice_d_eff - 1) * 100
            print(f"D-optimal is {improvement:.1f}% better (as expected)")
        else:
            decline = (1 - dopt_d_eff / lattice_d_eff) * 100
            print(f"⚠️  D-optimal is {decline:.1f}% WORSE than lattice")
            print(f"This suggests an issue with the D-optimal algorithm for parts mode")
    
    print(f"\nPossible issues to investigate:")
    print(f"1. Candidate generation in parts space")
    print(f"2. Optimization algorithm effectiveness")
    print(f"3. Model matrix construction")
    print(f"4. Bounds handling in optimization")

if __name__ == "__main__":
    test_parts_mode_comparison()
