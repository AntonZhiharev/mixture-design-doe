"""
Investigation script to understand why D-optimal is showing lower D-efficiency than simple lattice
for quadratic models with 9 runs
"""

import numpy as np
import pandas as pd
from src.core.simplified_mixture_design import SimplexLatticeDesign, DOptimalMixtureDesign
from src.core.optimal_design_generator import OptimalDesignGenerator, gram_matrix, calculate_determinant

def calculate_d_efficiency_mixture_model(design_matrix, model_type="quadratic"):
    """Calculate D-efficiency specifically for mixture models"""
    n_runs, n_components = design_matrix.shape
    
    print(f"\nCalculating D-efficiency for {model_type} mixture model:")
    print(f"Design matrix shape: {design_matrix.shape}")
    print("Design matrix:")
    for i, row in enumerate(design_matrix):
        print(f"  Run {i+1}: [{', '.join(f'{x:.3f}' for x in row)}] (sum = {sum(row):.3f})")
    
    # Build model matrix for mixture models
    if model_type == "linear":
        # For mixture models, linear terms are just the proportions
        X = design_matrix
        
    elif model_type == "quadratic":
        # For mixture models, quadratic includes:
        # - Linear terms: x_i (the proportions)
        # - Two-factor interactions: x_i * x_j for i < j
        # NOTE: We don't include pure quadratic terms x_i^2 because of mixture constraint
        
        model_terms = []
        
        # Linear terms (proportions)
        for i in range(n_components):
            model_terms.append(design_matrix[:, i])
        
        # Two-factor interactions
        for i in range(n_components):
            for j in range(i+1, n_components):
                interaction = design_matrix[:, i] * design_matrix[:, j]
                model_terms.append(interaction)
        
        X = np.column_stack(model_terms)
        
        print(f"\nMixture quadratic model terms:")
        print(f"  Linear terms: {n_components}")
        print(f"  Interaction terms: {n_components * (n_components - 1) // 2}")
        print(f"  Total parameters: {X.shape[1]}")
    
    print(f"\nModel matrix X shape: {X.shape}")
    print("Model matrix X:")
    for i, row in enumerate(X):
        print(f"  Run {i+1}: [{', '.join(f'{x:.4f}' for x in row)}]")
    
    # Calculate information matrix X'X
    XTX = X.T @ X
    print(f"\nInformation matrix X'X shape: {XTX.shape}")
    print("Information matrix X'X:")
    for i, row in enumerate(XTX):
        print(f"  Row {i+1}: [{', '.join(f'{x:.4f}' for x in row)}]")
    
    # Calculate determinant
    det_value = np.linalg.det(XTX)
    print(f"\nDeterminant of X'X: {det_value:.8f}")
    
    # Calculate D-efficiency
    n_params = X.shape[1]
    d_efficiency = (det_value / n_runs) ** (1/n_params) if det_value > 0 else 0.0
    
    print(f"D-efficiency = (det/n_runs)^(1/p) = ({det_value:.8f}/{n_runs})^(1/{n_params}) = {d_efficiency:.6f}")
    
    return d_efficiency, det_value, X, XTX

def calculate_d_efficiency_general_model(design_matrix, model_type="quadratic"):
    """Calculate D-efficiency using the general polynomial approach (like OptimalDesignGenerator)"""
    n_runs, n_components = design_matrix.shape
    
    print(f"\nCalculating D-efficiency using GENERAL polynomial model approach:")
    
    # Convert mixture proportions to [-1,1] range for general polynomial
    # This is what OptimalDesignGenerator expects
    design_std = 2 * design_matrix - 1  # Convert [0,1] to [-1,1]
    
    print(f"Converted to [-1,1] range:")
    for i, row in enumerate(design_std):
        print(f"  Run {i+1}: [{', '.join(f'{x:.3f}' for x in row)}]")
    
    # Use OptimalDesignGenerator approach to build model matrix
    if model_type == "quadratic":
        model_terms = []
        
        # Linear terms
        for i in range(n_components):
            model_terms.append(design_std[:, i])
        
        # Pure quadratic terms (x_i^2) - THIS IS THE DIFFERENCE!
        for i in range(n_components):
            model_terms.append(design_std[:, i]**2)
        
        # Interaction terms
        for i in range(n_components):
            for j in range(i+1, n_components):
                model_terms.append(design_std[:, i] * design_std[:, j])
        
        X_general = np.column_stack(model_terms)
        
        print(f"\nGeneral quadratic model terms:")
        print(f"  Linear terms: {n_components}")
        print(f"  Pure quadratic terms: {n_components}")
        print(f"  Interaction terms: {n_components * (n_components - 1) // 2}")
        print(f"  Total parameters: {X_general.shape[1]}")
    
    print(f"\nGeneral model matrix shape: {X_general.shape}")
    
    # Calculate using gram matrix approach (like OptimalDesignGenerator)
    XTX_general = gram_matrix(X_general.tolist())
    det_general = calculate_determinant(XTX_general)
    
    print(f"Determinant (general approach): {det_general:.8f}")
    
    # Calculate D-efficiency
    n_params_general = X_general.shape[1]
    d_efficiency_general = (det_general / n_runs) ** (1/n_params_general) if det_general > 0 else 0.0
    
    print(f"D-efficiency (general) = ({det_general:.8f}/{n_runs})^(1/{n_params_general}) = {d_efficiency_general:.6f}")
    
    return d_efficiency_general, det_general, X_general

def main():
    print("="*80)
    print("INVESTIGATING D-EFFICIENCY DISCREPANCY")
    print("="*80)
    
    # Test parameters
    n_components = 3
    n_runs = 9
    model_type = "quadratic"
    
    print(f"Test setup: {n_components} components, {n_runs} runs, {model_type} model")
    
    # Generate Simplex Lattice Design
    print("\n" + "="*60)
    print("1. SIMPLEX LATTICE DESIGN")
    print("="*60)
    
    lattice_design = SimplexLatticeDesign(n_components)
    lattice_df = lattice_design.generate_design(degree=2)
    
    print(f"Generated {len(lattice_df)} lattice points")
    print(lattice_df)
    
    # Calculate D-efficiency for lattice using mixture model approach
    d_eff_lattice_mixture, det_lattice_mixture, X_lattice_mixture, XTX_lattice = calculate_d_efficiency_mixture_model(
        lattice_df.values, model_type)
    
    # Generate D-Optimal Design
    print("\n" + "="*60)
    print("2. D-OPTIMAL DESIGN")
    print("="*60)
    
    dopt_design = DOptimalMixtureDesign(n_components)
    dopt_df = dopt_design.generate_design(n_runs=n_runs, model_type=model_type)
    
    print(f"Generated {len(dopt_df)} D-optimal points")
    print(dopt_df)
    
    # Calculate D-efficiency for D-optimal using mixture model approach
    d_eff_dopt_mixture, det_dopt_mixture, X_dopt_mixture, XTX_dopt = calculate_d_efficiency_mixture_model(
        dopt_df.values, model_type)
    
    # Also test with general polynomial approach
    print("\n" + "="*60)
    print("3. TESTING GENERAL POLYNOMIAL APPROACH")
    print("="*60)
    
    print("\nFor Simplex Lattice:")
    d_eff_lattice_general, det_lattice_general, X_lattice_general = calculate_d_efficiency_general_model(
        lattice_df.values, model_type)
    
    print("\nFor D-Optimal:")
    d_eff_dopt_general, det_dopt_general, X_dopt_general = calculate_d_efficiency_general_model(
        dopt_df.values, model_type)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"\nMIXTURE MODEL APPROACH (correct for mixture designs):")
    print(f"  Simplex Lattice D-efficiency: {d_eff_lattice_mixture:.6f}")
    print(f"  D-Optimal D-efficiency:      {d_eff_dopt_mixture:.6f}")
    print(f"  Winner: {'Simplex Lattice' if d_eff_lattice_mixture > d_eff_dopt_mixture else 'D-Optimal'}")
    
    print(f"\nGENERAL POLYNOMIAL APPROACH (what OptimalDesignGenerator uses):")
    print(f"  Simplex Lattice D-efficiency: {d_eff_lattice_general:.6f}")
    print(f"  D-Optimal D-efficiency:      {d_eff_dopt_general:.6f}")
    print(f"  Winner: {'Simplex Lattice' if d_eff_lattice_general > d_eff_dopt_general else 'D-Optimal'}")
    
    print(f"\nDIAGNOSIS:")
    print(f"The issue likely stems from:")
    print(f"1. Model definition differences:")
    print(f"   - Mixture models don't include x_i^2 terms (due to sum constraint)")
    print(f"   - General polynomial models include x_i^2 terms")
    print(f"2. OptimalDesignGenerator optimizes for general polynomials, not mixture constraints")
    print(f"3. Different parameter counts: mixture quadratic has {X_lattice_mixture.shape[1]} params, general has {X_lattice_general.shape[1]} params")
    
    # Additional analysis
    print(f"\nADDITIONAL ANALYSIS:")
    print(f"Parameter count comparison:")
    print(f"  Mixture quadratic model: {X_lattice_mixture.shape[1]} parameters")
    print(f"  General quadratic model: {X_lattice_general.shape[1]} parameters")
    
    if X_lattice_mixture.shape[1] != X_lattice_general.shape[1]:
        print(f"⚠️  Different parameter counts explain the D-efficiency differences!")
        print(f"     The D-optimal algorithm is optimizing for the wrong model structure!")

if __name__ == "__main__":
    main()
