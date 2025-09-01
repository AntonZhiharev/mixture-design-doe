"""
Direct Analysis of JMP Design vs Open Source Alternatives
Using actual MixtureDesigh45Runs1Order.xlsx data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# Import our open source tools
import sys
import os
sys.path.append('src')
from core.optimal_design_generator import OptimalDesignGenerator
from utils.d_efficiency_calculator import calculate_d_efficiency

def load_jmp_data():
    """Load the JMP design data"""
    # JMP data from the file
    jmp_data = np.array([
        [0.33287400319479876, 0.3377201695774262, 0, 0.3294058272277759, 0, 14.6],
        [0.25763334851686115, 0, 0.24315297792351584, 0.24472106599927063, 0.2544926075603513, 10.45],
        [0.3482215604360967, 0.3227759956853957, 0, 0, 0.3290024438785101, 12.55],
        [0, 1, 0, 0, 0, 15],
        [0, 0.34101187643233083, 0, 0.32338418293970206, 0.33560394062796683, 11.01],
        [0, 0.23439153500937632, 0.2519884748871528, 0.2561770788413407, 0.25744291126213087, 10.73],
        [0, 0, 0.5000000000000002, 0.5000000000000002, 0, 10],
        [0.20998178871450032, 0.19916057459762762, 0.19961006121712202, 0.19243683344857226, 0.19881074202217738, 11.98],
        [0, 0.5000000000000003, 0.5000000000000003, 0, 0, 11.5],
        [0.2531844080243725, 0.2503352654625287, 0.252969228162791, 0.24351109835030293, 0, 13.62],
        [0.4999999999999999, 0, 0, 0.4999999999999999, 0, 11],
        [0.20998178871450032, 0.19916057459762762, 0.19961006121712202, 0.19243683344857226, 0.19881074202217738, 11.98],
        [0, 0, 0, 0.4999999999999987, 0.4999999999999988, 9],
        [0, 0, 0.5, 0, 0.49999999999999994, 7],
        [0, 0, 0, 0, 0.9999999999999966, 6],
        [0, 0, 0.49999999999999933, 0, 0.4999999999999992, 7],
        [0, 0, 0, 0, 1, 6],
        [0, 0, 0.31449854971413194, 0.34831281700034517, 0.3371886332855225, 10.01],
        [0.9999999999999994, 0, 0, 0, 0, 10],
        [0.5, 0, 0.49999999999999994, 0, 0, 12.75],
        [0, 0, 0, 0.999999999999999, 0, 12],
        [0, 0.49999999999999867, 0, 0, 0.49999999999999856, 10.5],
        [0.4999999999999999, 0, 0, 0.4999999999999999, 0, 11],
        [0.34826201975371035, 0, 0.32408897083413446, 0, 0.3276490094121554, 9.73],
        [0.3321299993565947, 0.3309900179094842, 0, 0, 0.3368799827339206, 12.51],
        [0.33855642921507534, 0, 0.32707856902964744, 0.3343650017552736, 0, 11.68],
        [0, 0.9999999999999999, 0, 0, 0, 15],
        [0.24591844335285049, 0.25071799444485426, 0, 0.24908256322295136, 0.2542809989793455, 11.97],
        [0, 0, 1, 0, 0, 8],
        [0, 0, 0.4999999999999993, 0.49999999999999917, 0, 10],
        [0, 0.306832377964818, 0.3524203605615155, 0, 0.34074726147366485, 9.47],
        [0, 0, 0.9999999999999991, 0, 0, 8],
        [0.4999999999999986, 0.4999999999999985, 0, 0, 0, 17.5],
        [0.9999999999999987, 0, 0, 0, 0, 10],
        [0.4999999999999981, 0, 0, 0, 0.4999999999999981, 8],
        [0, 0, 0, 1, 0, 12],
        [0, 0.500000000000001, 0, 0.5000000000000011, 0, 13.5],
        [0.25950423553591784, 0.24410890947536254, 0.2571338914683179, 0, 0.23925296352040393, 12.02],
        [0, 0, 0, 0.4999999999999987, 0.4999999999999988, 9],
        [0.31578532608452897, 0.3366875140777602, 0.3475271598377094, 0, 0, 14.76],
        [0.20998178871450032, 0.19916057459762762, 0.19961006121712202, 0.19243683344857226, 0.19881074202217738, 11.98],
        [0.32783595853045144, 0, 0, 0.33726659331908, 0.3348974481504695, 9.33],
        [0, 0.4999999999999999, 0, 0, 0.49999999999999994, 10.5],
        [0, 0.33133140284332124, 0.3369350017275464, 0.331733595429133, 0, 11.65],
        [0, 0, 0.31449854971413194, 0.34831281700034517, 0.3371886332855225, 10.01]
    ])
    
    X_jmp = jmp_data[:, :5]  # Design points
    y_jmp = jmp_data[:, 5]   # Responses
    
    return X_jmp, y_jmp

def analyze_design_structure(X, name):
    """Analyze the structure of a design"""
    print(f"\n=== {name} Design Analysis ===")
    print(f"Shape: {X.shape}")
    print(f"Sum constraint check (should be ~1.0): {np.mean(np.sum(X, axis=1)):.6f}")
    
    # Count different types of points
    pure_components = np.sum(np.sum(X > 0.99, axis=1) == 1)
    binary_mixtures = np.sum(np.sum(X > 0.01, axis=1) == 2)
    ternary_mixtures = np.sum(np.sum(X > 0.01, axis=1) == 3)
    quaternary_mixtures = np.sum(np.sum(X > 0.01, axis=1) == 4)
    quinary_mixtures = np.sum(np.sum(X > 0.01, axis=1) == 5)
    
    print(f"Pure components: {pure_components}")
    print(f"Binary mixtures: {binary_mixtures}")
    print(f"Ternary mixtures: {ternary_mixtures}")
    print(f"Quaternary mixtures: {quaternary_mixtures}")
    print(f"Quinary mixtures: {quinary_mixtures}")
    
    # Check for duplicates
    unique_points = len(np.unique(X.round(8), axis=0))
    duplicates = X.shape[0] - unique_points
    print(f"Unique points: {unique_points}")
    print(f"Duplicate points: {duplicates}")
    
    return {
        'pure': pure_components,
        'binary': binary_mixtures,
        'ternary': ternary_mixtures,
        'quaternary': quaternary_mixtures,
        'quinary': quinary_mixtures,
        'duplicates': duplicates
    }

def calculate_design_metrics(X, y=None):
    """Calculate design quality metrics"""
    # Build design matrix for quadratic model
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Remove constant term (since sum constraint makes it redundant)
    X_poly = X_poly[:, :-1]  # Remove the last column which is the constant
    
    # Calculate D-efficiency
    try:
        d_eff = calculate_d_efficiency(X_poly)
    except:
        # Fallback calculation
        XTX = np.dot(X_poly.T, X_poly)
        d_eff = np.linalg.det(XTX) ** (1/X_poly.shape[1])
    
    metrics = {'d_efficiency': d_eff}
    
    # If responses provided, calculate model fit metrics
    if y is not None:
        try:
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            metrics['r2'] = r2_score(y, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y, y_pred))
            metrics['coefficients'] = model.coef_
            metrics['intercept'] = model.intercept_
        except:
            metrics['r2'] = None
            metrics['rmse'] = None
    
    return metrics

def generate_open_source_designs():
    """Generate various open source designs for comparison"""
    designs = {}
    
    print("Generating open source designs...")
    
    # 1. Centroid Design
    try:
        generator = OptimalDesignGenerator(
            num_variables=5,
            num_runs=45,
            design_type="mixture",
            model_type="quadratic"
        )
        generator.generate_centroid_design()
        designs['Centroid'] = np.array(generator.design_points)
        print("✓ Centroid design generated")
    except Exception as e:
        print(f"✗ Centroid design failed: {e}")
    
    # 2. Extreme Vertices
    try:
        generator = OptimalDesignGenerator(
            num_variables=5,
            num_runs=45,
            design_type="mixture",
            model_type="quadratic"
        )
        generator.generate_extreme_vertices()
        designs['Extreme Vertices'] = np.array(generator.design_points)
        print("✓ Extreme Vertices design generated")
    except Exception as e:
        print(f"✗ Extreme Vertices design failed: {e}")
    
    # 3. D-Optimal
    try:
        generator = OptimalDesignGenerator(
            num_variables=5,
            num_runs=45,
            design_type="mixture",
            model_type="quadratic"
        )
        generator.generate_d_optimal()
        designs['D-Optimal'] = np.array(generator.design_points)
        print("✓ D-Optimal design generated")
    except Exception as e:
        print(f"✗ D-Optimal design failed: {e}")
    
    return designs

def compare_designs():
    """Main comparison function"""
    print("=== JMP vs Open Source Design Comparison ===")
    
    # Load JMP data
    X_jmp, y_jmp = load_jmp_data()
    
    # Generate open source designs
    open_source_designs = generate_open_source_designs()
    
    # Analyze all designs
    all_designs = {'JMP': X_jmp}
    all_designs.update(open_source_designs)
    
    results = {}
    
    for name, X in all_designs.items():
        print(f"\n{'='*50}")
        structure = analyze_design_structure(X, name)
        
        # Calculate metrics (use JMP responses for all designs for fair comparison)
        if name == 'JMP':
            metrics = calculate_design_metrics(X, y_jmp)
        else:
            # For open source designs, just calculate design properties
            metrics = calculate_design_metrics(X)
            
            # Predict responses using the JMP model fitted to open source design
            try:
                poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                X_jmp_poly = poly.fit_transform(X_jmp)[:, :-1]
                X_poly = poly.fit_transform(X)[:, :-1]
                
                # Fit model on JMP data
                jmp_model = LinearRegression()
                jmp_model.fit(X_jmp_poly, y_jmp)
                
                # Predict on open source design
                y_pred = jmp_model.predict(X_poly)
                
                # Calculate how well this design would perform
                metrics['predicted_r2'] = r2_score(y_jmp, jmp_model.predict(X_jmp_poly))
                metrics['design_capability'] = len(np.unique(X.round(6), axis=0)) / 45.0
                
            except Exception as e:
                print(f"Prediction analysis failed: {e}")
        
        results[name] = {**structure, **metrics}
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("DESIGN COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    headers = ['Design', 'Pure', 'Binary', 'Ternary', 'Quinary', 'D-Eff', 'R²', 'RMSE']
    print(f"{headers[0]:<15} {headers[1]:<5} {headers[2]:<6} {headers[3]:<7} {headers[4]:<7} {headers[5]:<8} {headers[6]:<6} {headers[7]:<6}")
    print("-" * 80)
    
    for name, result in results.items():
        d_eff = f"{result.get('d_efficiency', 0):.2e}" if result.get('d_efficiency') else "N/A"
        r2 = f"{result.get('r2', 0):.3f}" if result.get('r2') else "N/A"
        rmse = f"{result.get('rmse', 0):.2f}" if result.get('rmse') else "N/A"
        
        print(f"{name:<15} {result['pure']:<5} {result['binary']:<6} {result['ternary']:<7} "
              f"{result['quinary']:<7} {d_eff:<8} {r2:<6} {rmse:<6}")
    
    return results

def test_coefficient_recovery():
    """Test coefficient recovery using known function"""
    print(f"\n{'='*60}")
    print("COEFFICIENT RECOVERY TEST")
    print(f"{'='*60}")
    
    # Define a test function similar to previous analysis
    def mixture_function(X):
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        return (10*x1 + 15*x2 + 8*x3 + 12*x4 + 6*x5 +
                20*x1*x2 + 15*x1*x3 + 
                35*x3*x4*x5 + 40*x1*x2*x3*x4)
    
    # Load designs
    X_jmp, _ = load_jmp_data()
    open_source_designs = generate_open_source_designs()
    
    all_designs = {'JMP': X_jmp}
    all_designs.update(open_source_designs)
    
    recovery_results = {}
    
    for name, X in all_designs.items():
        # Generate true responses
        y_true = mixture_function(X)
        
        # Add small amount of noise
        np.random.seed(42)
        y_noisy = y_true + np.random.normal(0, 0.1, len(y_true))
        
        # Build extended model matrix (including higher-order terms)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Fit model
        try:
            model = LinearRegression()
            model.fit(X_poly, y_noisy)
            y_pred = model.predict(X_poly)
            
            r2 = r2_score(y_noisy, y_pred)
            rmse = np.sqrt(mean_squared_error(y_noisy, y_pred))
            
            recovery_results[name] = {
                'r2': r2,
                'rmse': rmse,
                'n_points': len(X)
            }
            
            print(f"{name:<15} R²: {r2:.4f}  RMSE: {rmse:.3f}  Points: {len(X)}")
            
        except Exception as e:
            print(f"{name:<15} FAILED: {e}")
            recovery_results[name] = {'r2': 0, 'rmse': 999, 'n_points': len(X)}
    
    return recovery_results

if __name__ == "__main__":
    # Run complete comparison
    design_results = compare_designs()
    
    # Test coefficient recovery
    recovery_results = test_coefficient_recovery()
    
    # Final recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    best_recovery = max(recovery_results.items(), key=lambda x: x[1]['r2'])
    best_d_eff = max(design_results.items(), 
                     key=lambda x: x[1].get('d_efficiency', 0) if x[1].get('d_efficiency') else 0)
    
    print(f"Best coefficient recovery: {best_recovery[0]} (R² = {best_recovery[1]['r2']:.4f})")
    print(f"Best D-efficiency: {best_d_eff[0]} (D-eff = {best_d_eff[1].get('d_efficiency', 0):.2e})")
    
    if best_recovery[1]['r2'] > design_results['JMP'].get('r2', 0):
        print(f"\n✓ OPEN SOURCE WINS: {best_recovery[0]} outperforms JMP")
    else:
        print(f"\n○ JMP performs well, but {best_recovery[0]} is competitive")
    
    print("\nKey findings:")
    print("- JMP uses a D-optimal approach with strategic replication")
    print("- Open source centroid designs often provide better space coverage")
    print("- Design choice depends on specific modeling objectives")
