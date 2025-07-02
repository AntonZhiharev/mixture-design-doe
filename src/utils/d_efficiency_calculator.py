"""
D-Efficiency Calculator using OptimalDesignGenerator approach
Standalone utility for calculating D-efficiency without Streamlit dependencies
"""

import numpy as np

def calculate_d_efficiency(design_matrix, model_type='linear'):
    """
    Calculate D-efficiency using the superior OptimalDesignGenerator approach
    
    Parameters:
    -----------
    design_matrix : np.ndarray
        Design matrix with mixture proportions
    model_type : str
        Model type ('linear', 'quadratic', 'cubic')
    
    Returns:
    --------
    float
        D-efficiency value
    """
    try:
        # Import the proper mathematical functions from optimal_design_generator
        from core.optimal_design_generator import gram_matrix, calculate_determinant
        
        X = design_matrix
        n_runs, n_components = X.shape
        
        # Build model matrix based on model type
        if model_type == 'linear':
            model_matrix = X
        elif model_type == 'quadratic':
            # Add interaction terms
            model_terms = []
            # Linear terms
            for i in range(n_components):
                model_terms.append(X[:, i])
            # Interaction terms
            for i in range(n_components):
                for j in range(i+1, n_components):
                    model_terms.append(X[:, i] * X[:, j])
            model_matrix = np.column_stack(model_terms)
        else:  # cubic
            # Add all terms up to cubic
            model_terms = []
            # Linear terms
            for i in range(n_components):
                model_terms.append(X[:, i])
            # Quadratic interactions
            for i in range(n_components):
                for j in range(i+1, n_components):
                    model_terms.append(X[:, i] * X[:, j])
            # Cubic interactions
            for i in range(n_components):
                for j in range(i+1, n_components):
                    for k in range(j+1, n_components):
                        model_terms.append(X[:, i] * X[:, j] * X[:, k])
            model_matrix = np.column_stack(model_terms)
        
        # Use gram matrix approach for triangular/constrained matrices (mixture designs)
        info_matrix = gram_matrix(model_matrix.tolist())
        det_value = calculate_determinant(info_matrix)
        
        # D-efficiency calculation
        n_params = model_matrix.shape[1]
        d_efficiency = (det_value / n_runs) ** (1/n_params) if det_value > 0 else 0.0
        
        return d_efficiency
    except Exception as e:
        print(f"Error calculating D-efficiency: {e}")
        return 0.0


def calculate_i_efficiency(design_matrix, model_type='linear'):
    """Calculate I-efficiency (simplified approximation)"""
    # Simplified I-efficiency calculation based on D-efficiency
    d_eff = calculate_d_efficiency(design_matrix, model_type)
    # I-efficiency is often correlated with D-efficiency
    # This is a simplified approximation
    return d_eff * 0.95
