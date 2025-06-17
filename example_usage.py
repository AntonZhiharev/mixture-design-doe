"""
Example usage of Optimal DOE for n factors and m responses
"""

import numpy as np
import pandas as pd
from optimal_doe_python import OptimalDOE, multiple_response_analysis

# Example: 4 factors, 3 responses

def main():
    print("=== Optimal DOE Example: 4 Factors, 3 Responses ===\n")
    
    # Define the experimental setup
    n_factors = 4
    factor_ranges = [
        (-2, 2),    # Factor 1: Temperature
        (10, 100),  # Factor 2: Pressure  
        (0.1, 1.0), # Factor 3: Concentration
        (1, 10)     # Factor 4: Time
    ]
    
    factor_names = ['Temperature', 'Pressure', 'Concentration', 'Time']
    
    # Initialize DOE generator
    doe = OptimalDOE(n_factors, factor_ranges)
    
    # Generate designs
    print("Generating optimal designs...")
    n_runs = 25  # Number of experimental runs
    
    # D-optimal design (best for parameter estimation)
    d_optimal = doe.generate_d_optimal(n_runs=n_runs, model_order=2, random_seed=42)
    
    # I-optimal design (best for prediction)
    i_optimal = doe.generate_i_optimal(n_runs=n_runs, model_order=2, random_seed=42)
    
    # Evaluate designs
    d_results = doe.evaluate_design(d_optimal, model_order=2)
    i_results = doe.evaluate_design(i_optimal, model_order=2)
    
    print(f"D-optimal design: D-eff = {d_results['d_efficiency']:.4f}, I-eff = {d_results['i_efficiency']:.4f}")
    print(f"I-optimal design: D-eff = {i_results['d_efficiency']:.4f}, I-eff = {i_results['i_efficiency']:.4f}")
    
    # Define 3 response functions (representing different process outcomes)
    def response_yield(X):
        """Response 1: Process yield (%)"""
        # Complex response with interactions
        temp, press, conc, time = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        yield_response = (
            80 + 
            2*temp + 0.1*press + 15*conc + 1.5*time +
            -0.5*temp**2 - 0.001*press**2 - 8*conc**2 - 0.1*time**2 +
            0.01*temp*press + 2*temp*conc + 0.05*press*conc +
            np.random.normal(0, 2, len(X))
        )
        return np.clip(yield_response, 0, 100)  # Yield between 0-100%
    
    def response_purity(X):
        """Response 2: Product purity (%)"""
        temp, press, conc, time = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        purity_response = (
            90 +
            -0.5*temp + 0.05*press + 5*conc + 0.8*time +
            -0.1*temp**2 - 0.0005*press**2 - 3*conc**2 - 0.05*time**2 +
            0.02*conc*time +
            np.random.normal(0, 1.5, len(X))
        )
        return np.clip(purity_response, 70, 100)  # Purity between 70-100%
    
    def response_cost(X):
        """Response 3: Production cost ($/kg)"""
        temp, press, conc, time = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        cost_response = (
            50 +
            0.8*np.abs(temp) + 0.02*press + 20*conc + 2*time +
            0.1*temp**2 + 0.0001*press**2 + 10*conc**2 + 0.1*time**2 +
            np.random.normal(0, 3, len(X))
        )
        return np.maximum(cost_response, 10)  # Cost >= $10/kg
    
    # Analyze multiple responses for both designs
    designs = [d_optimal, i_optimal]
    design_names = ['D-optimal', 'I-optimal']
    response_functions = [response_yield, response_purity, response_cost]
    response_names = ['Yield (%)', 'Purity (%)', 'Cost ($/kg)']
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Analyze responses
    comparison_df = multiple_response_analysis(designs, response_functions, design_names)
    
    print("\n=== Multiple Response Analysis ===")
    print(comparison_df.round(2))
    
    # Create detailed design matrices with factor names
    print("\n=== D-Optimal Design Matrix ===")
    d_design_df = pd.DataFrame(d_optimal, columns=factor_names)
    print(d_design_df.round(3))
    
    print("\n=== I-Optimal Design Matrix ===")
    i_design_df = pd.DataFrame(i_optimal, columns=factor_names)
    print(i_design_df.round(3))
    
    # Calculate responses for each design point (D-optimal)
    print("\n=== D-Optimal Design with Predicted Responses ===")
    np.random.seed(42)  # Reset seed for consistent results
    d_responses = pd.DataFrame({
        'Run': range(1, n_runs + 1),
        'Temperature': d_optimal[:, 0],
        'Pressure': d_optimal[:, 1], 
        'Concentration': d_optimal[:, 2],
        'Time': d_optimal[:, 3],
        'Yield (%)': response_yield(d_optimal),
        'Purity (%)': response_purity(d_optimal),
        'Cost ($/kg)': response_cost(d_optimal)
    })
    print(d_responses.round(2))
    
    # Summary statistics
    print("\n=== Response Summary Statistics (D-Optimal) ===")
    summary_stats = d_responses[['Yield (%)', 'Purity (%)', 'Cost ($/kg)']].describe()
    print(summary_stats.round(2))
    
    # Identify best conditions for each response
    print("\n=== Optimal Conditions for Each Response ===")
    
    best_yield_idx = d_responses['Yield (%)'].idxmax()
    best_purity_idx = d_responses['Purity (%)'].idxmax()
    best_cost_idx = d_responses['Cost ($/kg)'].idxmin()
    
    print(f"Best Yield: {d_responses.loc[best_yield_idx, 'Yield (%)']:.1f}% at run {best_yield_idx+1}")
    print(f"  Conditions: T={d_responses.loc[best_yield_idx, 'Temperature']:.2f}, P={d_responses.loc[best_yield_idx, 'Pressure']:.1f}, C={d_responses.loc[best_yield_idx, 'Concentration']:.3f}, t={d_responses.loc[best_yield_idx, 'Time']:.2f}")
    
    print(f"Best Purity: {d_responses.loc[best_purity_idx, 'Purity (%)']:.1f}% at run {best_purity_idx+1}")
    print(f"  Conditions: T={d_responses.loc[best_purity_idx, 'Temperature']:.2f}, P={d_responses.loc[best_purity_idx, 'Pressure']:.1f}, C={d_responses.loc[best_purity_idx, 'Concentration']:.3f}, t={d_responses.loc[best_purity_idx, 'Time']:.2f}")
    
    print(f"Best Cost: ${d_responses.loc[best_cost_idx, 'Cost ($/kg)']:.2f}/kg at run {best_cost_idx+1}")
    print(f"  Conditions: T={d_responses.loc[best_cost_idx, 'Temperature']:.2f}, P={d_responses.loc[best_cost_idx, 'Pressure']:.1f}, C={d_responses.loc[best_cost_idx, 'Concentration']:.3f}, t={d_responses.loc[best_cost_idx, 'Time']:.2f}")
    
    # Save results to CSV
    d_responses.to_csv('d_optimal_results.csv', index=False)
    print(f"\nResults saved to 'd_optimal_results.csv'")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()
