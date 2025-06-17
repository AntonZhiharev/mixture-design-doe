"""
TUTORIAL: How to Analyze Your Experimental Data After Collecting Responses

This tutorial shows you step-by-step how to use our DOE system to analyze
your experimental results and extract meaningful insights.
"""

import numpy as np
import pandas as pd
from response_analysis import DOEResponseAnalysis, MixtureResponseAnalysis

def tutorial_regular_doe():
    """
    TUTORIAL: Regular DOE Analysis Workflow
    
    Scenario: You ran a chemical process optimization experiment
    Factors: Temperature (°C), Pressure (bar), pH
    Response: Yield (%)
    """
    
    print("🧪 TUTORIAL: Regular DOE Analysis")
    print("=" * 50)
    
    # STEP 1: Load your experimental data
    print("\n📊 STEP 1: Load Your Experimental Data")
    print("-" * 30)
    
    # This is YOUR experimental design (from our DOE generator)
    experimental_design = np.array([
        [80, 2, 6],    # Run 1: 80°C, 2 bar, pH 6
        [120, 2, 6],   # Run 2: 120°C, 2 bar, pH 6
        [80, 4, 6],    # Run 3: 80°C, 4 bar, pH 6
        [120, 4, 6],   # Run 4: 120°C, 4 bar, pH 6
        [80, 2, 8],    # Run 5: 80°C, 2 bar, pH 8
        [120, 2, 8],   # Run 6: 120°C, 2 bar, pH 8
        [80, 4, 8],    # Run 7: 80°C, 4 bar, pH 8
        [120, 4, 8],   # Run 8: 120°C, 4 bar, pH 8
        [100, 3, 7],   # Run 9: Center point
        [100, 3, 7]    # Run 10: Center point
    ])
    
    # These are YOUR experimental results (what you measured in the lab)
    experimental_responses = np.array([
        72.5,  # Run 1 result
        78.2,  # Run 2 result
        75.1,  # Run 3 result
        82.3,  # Run 4 result
        74.8,  # Run 5 result
        80.1,  # Run 6 result
        77.9,  # Run 7 result
        85.4,  # Run 8 result
        79.2,  # Run 9 result
        78.8   # Run 10 result
    ])
    
    print("Experimental Design:")
    design_df = pd.DataFrame(experimental_design, 
                           columns=['Temperature (°C)', 'Pressure (bar)', 'pH'])
    design_df['Yield (%)'] = experimental_responses
    design_df.index = [f'Run {i+1}' for i in range(len(design_df))]
    print(design_df)
    
    # STEP 2: Convert to coded units (standard practice in DOE)
    print("\n🔄 STEP 2: Convert to Coded Units")
    print("-" * 30)
    
    # Convert to -1, +1 scale for analysis
    coded_design = np.zeros_like(experimental_design)
    
    # Temperature: 80-120°C → -1 to +1
    coded_design[:, 0] = (experimental_design[:, 0] - 100) / 20
    
    # Pressure: 2-4 bar → -1 to +1  
    coded_design[:, 1] = (experimental_design[:, 1] - 3) / 1
    
    # pH: 6-8 → -1 to +1
    coded_design[:, 2] = (experimental_design[:, 2] - 7) / 1
    
    print("Coded Design Matrix:")
    coded_df = pd.DataFrame(coded_design, 
                          columns=['Temp (coded)', 'Press (coded)', 'pH (coded)'])
    print(coded_df.round(2))
    
    # STEP 3: Create analyzer and fit models
    print("\n📈 STEP 3: Fit Statistical Models")
    print("-" * 30)
    
    analyzer = DOEResponseAnalysis(
        coded_design, 
        experimental_responses,
        factor_names=['Temperature', 'Pressure', 'pH'],
        response_name='Yield (%)'
    )
    
    # Fit linear model
    linear_results = analyzer.fit_linear_model()
    print(f"Linear Model R² = {linear_results['r_squared']:.4f}")
    print(f"Linear Model RMSE = {linear_results['rmse']:.2f}%")
    
    # Fit quadratic model
    quad_results = analyzer.fit_quadratic_model()
    print(f"Quadratic Model R² = {quad_results['r_squared']:.4f}")
    print(f"Quadratic Model RMSE = {quad_results['rmse']:.2f}%")
    
    # Choose best model
    best_model = "Quadratic" if quad_results['r_squared'] > linear_results['r_squared'] else "Linear"
    print(f"\n🏆 Best Model: {best_model}")
    
    # STEP 4: Statistical significance testing
    print("\n📊 STEP 4: Statistical Analysis")
    print("-" * 30)
    
    # ANOVA table
    anova_table = analyzer.anova_analysis(linear_results)
    print("ANOVA Table:")
    print(anova_table.round(4))
    
    # Effects analysis
    effects = analyzer.effects_analysis(linear_results)
    print("\nFactor Effects:")
    print(effects.round(4))
    
    # Interpretation
    significant_factors = effects[effects['Significant'] == True]['Term'].tolist()
    print(f"\n🎯 Significant factors (p < 0.05): {significant_factors}")
    
    # STEP 5: Optimize the process
    print("\n🎯 STEP 5: Process Optimization")
    print("-" * 30)
    
    # Find conditions for maximum yield
    bounds = [(-1, 1), (-1, 1), (-1, 1)]  # Coded units
    opt_result = analyzer.optimize_response(bounds, maximize=True)
    
    if opt_result['success']:
        print("Optimal Conditions (coded units):")
        for factor, value in opt_result['optimal_factors'].items():
            print(f"  {factor}: {value:.3f}")
        
        # Convert back to real units
        print("\nOptimal Conditions (real units):")
        temp_optimal = opt_result['optimal_factors']['Temperature'] * 20 + 100
        press_optimal = opt_result['optimal_factors']['Pressure'] * 1 + 3
        pH_optimal = opt_result['optimal_factors']['pH'] * 1 + 7
        
        print(f"  Temperature: {temp_optimal:.1f}°C")
        print(f"  Pressure: {press_optimal:.1f} bar")
        print(f"  pH: {pH_optimal:.1f}")
        print(f"  Predicted Yield: {opt_result['optimal_response']:.1f}%")
    
    # STEP 6: Make predictions for new conditions
    print("\n🔮 STEP 6: Predictions for New Conditions")
    print("-" * 30)
    
    # You want to test these new conditions
    new_conditions_real = np.array([
        [90, 2.5, 6.5],   # New condition 1
        [110, 3.5, 7.5],  # New condition 2
        [105, 2.8, 7.2]   # New condition 3
    ])
    
    # Convert to coded units
    new_conditions_coded = np.zeros_like(new_conditions_real)
    new_conditions_coded[:, 0] = (new_conditions_real[:, 0] - 100) / 20
    new_conditions_coded[:, 1] = (new_conditions_real[:, 1] - 3) / 1
    new_conditions_coded[:, 2] = (new_conditions_real[:, 2] - 7) / 1
    
    predictions = analyzer.predict_response(new_conditions_coded)
    
    print("Predictions:")
    for i, (real_cond, pred) in enumerate(zip(new_conditions_real, predictions)):
        print(f"Condition {i+1}: {real_cond[0]:.0f}°C, {real_cond[1]:.1f}bar, pH{real_cond[2]:.1f} → {pred:.1f}% yield")
    
    return analyzer, linear_results, quad_results


def tutorial_mixture_doe():
    """
    TUTORIAL: Mixture DOE Analysis Workflow
    
    Scenario: You're optimizing a polymer blend formulation
    Components: Polymer A, Polymer B, Additive
    Response: Tensile Strength (MPa)
    """
    
    print("\n🧬 TUTORIAL: Mixture DOE Analysis")
    print("=" * 50)
    
    # STEP 1: Load your mixture experimental data
    print("\n📊 STEP 1: Load Your Mixture Data")
    print("-" * 30)
    
    # Mixture compositions (must sum to 1.0 or 100%)
    mixture_design = np.array([
        [0.70, 0.20, 0.10],  # Run 1: 70% A, 20% B, 10% Additive
        [0.50, 0.40, 0.10],  # Run 2: 50% A, 40% B, 10% Additive
        [0.30, 0.60, 0.10],  # Run 3: 30% A, 60% B, 10% Additive
        [0.60, 0.20, 0.20],  # Run 4: 60% A, 20% B, 20% Additive
        [0.40, 0.40, 0.20],  # Run 5: 40% A, 40% B, 20% Additive
        [0.20, 0.60, 0.20],  # Run 6: 20% A, 60% B, 20% Additive
        [0.50, 0.20, 0.30],  # Run 7: 50% A, 20% B, 30% Additive
        [0.30, 0.40, 0.30],  # Run 8: 30% A, 40% B, 30% Additive
        [0.10, 0.60, 0.30],  # Run 9: 10% A, 60% B, 30% Additive
        [0.33, 0.33, 0.34]   # Run 10: Centroid blend
    ])
    
    # Your experimental results (tensile strength in MPa)
    mixture_responses = np.array([
        45.2,  # Run 1 result
        52.1,  # Run 2 result
        58.7,  # Run 3 result
        38.9,  # Run 4 result
        48.3,  # Run 5 result
        54.6,  # Run 6 result
        32.1,  # Run 7 result
        42.8,  # Run 8 result
        48.5,  # Run 9 result
        46.7   # Run 10 result
    ])
    
    print("Mixture Experimental Data:")
    mixture_df = pd.DataFrame(mixture_design, 
                            columns=['Polymer A (%)', 'Polymer B (%)', 'Additive (%)'])
    mixture_df = mixture_df * 100  # Convert to percentages
    mixture_df['Tensile Strength (MPa)'] = mixture_responses
    mixture_df.index = [f'Run {i+1}' for i in range(len(mixture_df))]
    print(mixture_df.round(1))
    
    # Verify mixture constraint
    sums = np.sum(mixture_design, axis=1)
    print(f"\n✅ Mixture constraint check: All sums = {sums.round(3)} (should be 1.0)")
    
    # STEP 2: Create mixture analyzer
    print("\n📈 STEP 2: Fit Mixture Models")
    print("-" * 30)
    
    mixture_analyzer = MixtureResponseAnalysis(
        mixture_design, 
        mixture_responses,
        component_names=['Polymer_A', 'Polymer_B', 'Additive'],
        response_name='Tensile Strength (MPa)'
    )
    
    # Fit Scheffé linear model
    linear_results = mixture_analyzer.fit_scheffe_linear()
    print(f"Scheffé Linear Model R² = {linear_results['r_squared']:.4f}")
    print(f"Scheffé Linear Model RMSE = {linear_results['rmse']:.2f} MPa")
    
    print("\nLinear Model Coefficients (Pure Component Effects):")
    for name, coeff in zip(linear_results['component_names'], linear_results['coefficients']):
        print(f"  {name}: {coeff:.1f} MPa")
    
    # Fit Scheffé quadratic model
    quad_results = mixture_analyzer.fit_scheffe_quadratic()
    print(f"\nScheffé Quadratic Model R² = {quad_results['r_squared']:.4f}")
    print(f"Scheffé Quadratic Model RMSE = {quad_results['rmse']:.2f} MPa")
    
    print("\nQuadratic Model Terms:")
    for name, coeff in zip(quad_results['term_names'], quad_results['coefficients']):
        print(f"  {name}: {coeff:.1f} MPa")
    
    # STEP 3: Interpret the results
    print("\n🔍 STEP 3: Interpret Results")
    print("-" * 30)
    
    # Pure component effects
    pure_effects = linear_results['coefficients']
    component_names = linear_results['component_names']
    
    best_component = component_names[np.argmax(pure_effects)]
    worst_component = component_names[np.argmin(pure_effects)]
    
    print(f"Best pure component: {best_component} ({max(pure_effects):.1f} MPa)")
    print(f"Worst pure component: {worst_component} ({min(pure_effects):.1f} MPa)")
    
    # Interaction effects (from quadratic model)
    interaction_coeffs = quad_results['coefficients'][len(component_names):]
    interaction_names = quad_results['term_names'][len(component_names):]
    
    print("\nInteraction Effects:")
    for name, coeff in zip(interaction_names, interaction_coeffs):
        effect_type = "Synergistic" if coeff > 0 else "Antagonistic"
        print(f"  {name}: {coeff:.1f} MPa ({effect_type})")
    
    # STEP 4: Find optimal mixture
    print("\n🎯 STEP 4: Find Optimal Mixture")
    print("-" * 30)
    
    # Find the mixture composition that gives maximum strength
    best_run_idx = np.argmax(mixture_responses)
    best_mixture = mixture_design[best_run_idx]
    best_response = mixture_responses[best_run_idx]
    
    print(f"Best experimental mixture (Run {best_run_idx + 1}):")
    for i, comp_name in enumerate(['Polymer_A', 'Polymer_B', 'Additive']):
        print(f"  {comp_name}: {best_mixture[i]*100:.1f}%")
    print(f"  Tensile Strength: {best_response:.1f} MPa")
    
    # STEP 5: Predict new mixture compositions
    print("\n🔮 STEP 5: Predict New Mixtures")
    print("-" * 30)
    
    # Test some new mixture compositions
    new_mixtures = np.array([
        [0.40, 0.50, 0.10],  # High Polymer B
        [0.35, 0.55, 0.10],  # Even higher Polymer B
        [0.25, 0.65, 0.10],  # Maximum Polymer B with constraints
    ])
    
    # Use the quadratic model for predictions
    mixture_analyzer.model = mixture_analyzer.model  # Ensure model is set
    predictions = []
    
    # Manual prediction using quadratic model coefficients
    for mixture in new_mixtures:
        # Linear terms
        pred = np.sum(mixture * quad_results['coefficients'][:3])
        
        # Interaction terms
        interaction_idx = 3
        for i in range(3):
            for j in range(i + 1, 3):
                pred += quad_results['coefficients'][interaction_idx] * mixture[i] * mixture[j]
                interaction_idx += 1
        
        predictions.append(pred)
    
    print("New Mixture Predictions:")
    for i, (mixture, pred) in enumerate(zip(new_mixtures, predictions)):
        print(f"Mixture {i+1}: A={mixture[0]*100:.0f}%, B={mixture[1]*100:.0f}%, Add={mixture[2]*100:.0f}% → {pred:.1f} MPa")
    
    return mixture_analyzer, linear_results, quad_results


def practical_workflow_guide():
    """
    PRACTICAL WORKFLOW: Step-by-step guide for YOUR experiments
    """
    
    print("\n📋 PRACTICAL WORKFLOW GUIDE")
    print("=" * 50)
    
    print("""
🔄 COMPLETE DOE WORKFLOW:

1. DESIGN PHASE (Using our DOE system):
   ✅ Use optimal_doe_python.py or streamlit_app.py
   ✅ Generate D-optimal or I-optimal design
   ✅ Export design matrix as CSV
   ✅ Run experiments in laboratory

2. DATA COLLECTION PHASE:
   ✅ Follow the experimental design exactly
   ✅ Measure your response variable(s)
   ✅ Record data in spreadsheet or CSV file

3. ANALYSIS PHASE (Using response_analysis.py):
   ✅ Load your experimental data
   ✅ Fit statistical models (linear/quadratic)
   ✅ Check model adequacy (R², residuals)
   ✅ Identify significant factors
   ✅ Optimize response

4. IMPLEMENTATION PHASE:
   ✅ Use optimal conditions in production
   ✅ Validate with confirmation experiments
   ✅ Monitor process performance

📊 YOUR DATA FORMAT:
   For Regular DOE:
   - Design matrix: [temperature, pressure, pH, ...]
   - Responses: [yield1, yield2, yield3, ...]
   
   For Mixture DOE:
   - Mixture matrix: [comp1_fraction, comp2_fraction, ...]
   - Responses: [strength1, strength2, strength3, ...]

💡 KEY TIPS:
   ✅ Always check R² > 0.80 for good model fit
   ✅ Use residual plots to validate assumptions
   ✅ Focus on statistically significant factors (p < 0.05)
   ✅ Validate optimal conditions with new experiments
   ✅ Consider interaction effects in complex systems
    """)


if __name__ == "__main__":
    print("🧪 DOE RESPONSE ANALYSIS TUTORIAL")
    print("=" * 60)
    
    # Run regular DOE tutorial
    regular_analyzer, regular_linear, regular_quad = tutorial_regular_doe()
    
    # Run mixture DOE tutorial
    mixture_analyzer, mixture_linear, mixture_quad = tutorial_mixture_doe()
    
    # Show practical workflow
    practical_workflow_guide()
    
    print("\n🎯 TUTORIAL COMPLETE!")
    print("=" * 60)
    print("Now you know how to:")
    print("✅ Load and prepare experimental data")
    print("✅ Fit statistical models to your responses")
    print("✅ Identify significant factors and interactions")
    print("✅ Optimize your process conditions")
    print("✅ Make predictions for new conditions")
    print("✅ Handle both regular and mixture experiments")
    print("\nNext: Apply these methods to YOUR experimental data! 🚀")
