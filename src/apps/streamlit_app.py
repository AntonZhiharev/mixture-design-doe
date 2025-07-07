"""
Streamlit App for Simplified Mixture Design
Uses the new simplified "one method - one class" architecture
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from simplified mixture design
from core.simplified_mixture_design import (
    create_mixture_design,
    SimplexLatticeDesign,
    SimplexCentroidDesign,
    DOptimalMixtureDesign,
    ExtremeVerticesDesign,
    AugmentedDesign,
    CustomMixtureDesign
)

# Import the NEW correct fixed components implementation
from core.fixed_parts_mixture_designs import FixedPartsMixtureDesign

# Import D-efficiency calculator
from utils.d_efficiency_calculator import calculate_d_efficiency, calculate_i_efficiency

# Import enhanced Excel export functionality
from utils.mixture_utils import create_enhanced_excel_export

# Import regular DOE (keep existing functionality)
try:
    from core.base_doe import OptimalDOE, multiple_response_analysis
except ImportError:
    # Fallback if base_doe is not available
    OptimalDOE = None
    multiple_response_analysis = None

# Page configuration
st.set_page_config(
    page_title="Mixture Design Generator (Simplified)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧪 Mixture Design of Experiments</h1>', unsafe_allow_html=True)
st.markdown("**Generate mixture experimental designs using simplified architecture**")

# Helper functions - now using standalone D-efficiency calculator
# (Functions imported from utils.d_efficiency_calculator)

# Sidebar for navigation
st.sidebar.title("Navigation")
design_type = st.sidebar.selectbox(
    "Choose Design Type",
    ["Mixture Design", "Standard DOE", "D-Optimal Analysis", "Design Comparison", "About"]
)

if design_type == "Mixture Design":
    st.markdown('<h2 class="sub-header">🧬 Mixture Design</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Mixture Parameters")
        
        # Number of components
        n_components = st.number_input(
            "Number of Components", 
            min_value=2, 
            max_value=10, 
            value=3,
            help="Number of mixture components"
        )
        
        # Component names
        st.write("**Component Names:**")
        component_names = []
        for i in range(n_components):
            name = st.text_input(
                f"Component {i+1} name",
                value=f"Component_{i+1}",
                key=f"comp_name_{i}"
            )
            component_names.append(name)
        
        # Parts mode option
        use_parts_mode = st.checkbox(
            "Use Parts Mode",
            value=False,
            help="Work with absolute parts instead of proportions only"
        )
        
        # Design method selection - conditional based on parts mode
        if use_parts_mode:
            # For parts mode, only allow optimal methods that work well with fixed components
            available_methods = ["d-optimal", "i-optimal"]
            method_descriptions = {
                "d-optimal": "D-Optimal",
                "i-optimal": "I-Optimal"
            }
            st.info("ℹ️ **Parts Mode**: Only D-optimal and I-optimal methods are available in parts mode as they are designed for optimal experimental design with fixed components.")
        else:
            # For proportions mode, all methods are available
            available_methods = ["simplex-lattice", "simplex-centroid", "d-optimal", "i-optimal", "extreme-vertices"]
            method_descriptions = {
                "simplex-lattice": "Simplex Lattice",
                "simplex-centroid": "Simplex Centroid", 
                "d-optimal": "D-Optimal",
                "i-optimal": "I-Optimal",
                "extreme-vertices": "Extreme Vertices"
            }
        
        design_method = st.selectbox(
            "Design Method",
            available_methods,
            format_func=lambda x: method_descriptions[x]
        )
        
        # Fixed components (only for parts mode)
        fixed_components = {}
        if use_parts_mode:
            st.write("**Fixed Components (Parts):**")
            for i, comp_name in enumerate(component_names):
                fix_comp = st.checkbox(f"Fix {comp_name}", key=f"fix_{i}")
                if fix_comp:
                    fixed_value = st.number_input(
                        f"{comp_name} fixed value (parts)",
                        min_value=0.01,
                        value=1.0,
                        step=0.1,
                        key=f"fixed_val_{i}"
                    )
                    fixed_components[comp_name] = fixed_value
        
        # Model type for evaluation (moved earlier to use in method-specific parameters)
        model_type = st.selectbox(
            "Model Type (for efficiency calculation)",
            ["linear", "quadratic", "cubic"],
            index=1
        )
        
        # Component bounds (for parts mode or extreme vertices)
        component_bounds = None
        if use_parts_mode or design_method == "extreme-vertices":
            st.write("**Component Bounds:**")
            bounds_mode = "Parts" if use_parts_mode else "Proportions"
            max_val = 10.0 if use_parts_mode else 1.0
            default_min = 0.1 if use_parts_mode else 0.0
            
            # For parts mode with fixed components, add smart bounds warning
            if use_parts_mode and fixed_components:
                total_fixed_parts = sum(fixed_components.values())
                estimated_max_batch = total_fixed_parts + (len(component_names) - len(fixed_components)) * max_val
                min_meaningful_parts = estimated_max_batch * 0.01  # 1% proportion
                
                st.info(f"""
                ⚠️ **Parts Mode Bounds Guidance:**
                - Total fixed parts: {total_fixed_parts:.1f}
                - Estimated max batch size: ~{estimated_max_batch:.1f} parts
                - For ≥1% proportion: use min bounds ≥ {min_meaningful_parts:.2f} parts
                - **Avoid very small bounds** (like 0.1) that become tiny proportions!
                """)
            
            lower_bounds = []
            upper_bounds = []
            bounds_warnings = []
            
            for i, comp_name in enumerate(component_names):
                # Skip fixed components
                if comp_name in fixed_components:
                    continue
                    
                col_a, col_b = st.columns(2)
                with col_a:
                    lower = st.number_input(
                        f"{comp_name} min ({bounds_mode.lower()})",
                        min_value=0.0,
                        max_value=max_val,
                        value=default_min,
                        step=0.01,
                        key=f"bounds_lower_{i}_parts" if use_parts_mode else f"bounds_lower_{i}_props"
                    )
                    lower_bounds.append(lower)
                with col_b:
                    upper = st.number_input(
                        f"{comp_name} max ({bounds_mode.lower()})",
                        min_value=0.01,
                        max_value=max_val,
                        value=max_val if use_parts_mode else 1.0,
                        step=0.01,
                        key=f"bounds_upper_{i}_parts" if use_parts_mode else f"bounds_upper_{i}_props"
                    )
                    upper_bounds.append(upper)
                
                # Check for potential proportion issues in parts mode
                if use_parts_mode and fixed_components:
                    total_fixed_parts = sum(fixed_components.values())
                    max_possible_batch = total_fixed_parts + sum(upper_bounds) + (max_val * (len(component_names) - len(fixed_components) - len(upper_bounds)))
                    min_proportion = lower / max_possible_batch if max_possible_batch > 0 else 0
                    
                    if min_proportion < 0.005:  # Less than 0.5% proportion
                        bounds_warnings.append(f"⚠️ {comp_name}: min {lower:.2f} parts = {min_proportion:.1%} proportion (very small!)")
            
            # Show bounds warnings
            if bounds_warnings:
                st.warning("**Potential Issues with Small Bounds:**\n" + "\n".join(bounds_warnings))
                st.info("💡 **Tip**: Increase minimum bounds to ensure components have meaningful proportions (≥1-2%)")
            
            component_bounds = list(zip(lower_bounds, upper_bounds))
        
        # Method-specific parameters
        if design_method == "simplex-lattice":
            # Dynamic degree limits based on model type for evaluation
            if model_type == "cubic":
                max_degree = 10
                default_degree = 6
                help_text = "Degree of the simplex lattice (higher degree needed for cubic models)"
            elif model_type == "quadratic":
                max_degree = 7
                default_degree = 4
                help_text = "Degree of the simplex lattice (moderate degree for quadratic models)"
            else:  # linear
                max_degree = 5
                default_degree = 3
                help_text = "Degree of the simplex lattice"
            
            degree = st.number_input(
                "Lattice Degree",
                min_value=2,
                max_value=max_degree,
                value=default_degree,
                help=help_text
            )
            
            # Add warning for insufficient points
            from math import comb
            expected_points = comb(n_components + degree - 1, degree)
            if model_type == "cubic":
                linear_terms = n_components
                quad_interactions = (n_components * (n_components - 1)) // 2
                cubic_interactions = (n_components * (n_components - 1) * (n_components - 2)) // 6
                min_params = linear_terms + quad_interactions + cubic_interactions
                
                if expected_points < min_params:
                    st.warning(f"⚠️ Lattice degree {degree} will generate ~{expected_points} points, but cubic model needs ≥{min_params} points. Consider degree ≥{6}")
            elif model_type == "quadratic":
                linear_terms = n_components
                quad_interactions = (n_components * (n_components - 1)) // 2
                min_params = linear_terms + quad_interactions
                
                if expected_points < min_params:
                    st.warning(f"⚠️ Lattice degree {degree} will generate ~{expected_points} points, but quadratic model needs ≥{min_params} points. Consider degree ≥{4}")
            
            additional_params = {"degree": degree}
            
        elif design_method in ["d-optimal", "i-optimal"]:
            n_runs = st.number_input(
                "Number of Runs",
                min_value=n_components + 1,
                max_value=1000,
                value=min(15, 3 * n_components),
                help="Number of experimental runs"
            )
            include_interior = st.checkbox(
                "Include interior points",
                value=True,
                help="Include points inside the simplex (not just corners)"
            )
            
            if design_method == "i-optimal":
                i_model_type = st.selectbox(
                    "Model Type for I-optimality",
                    ["linear", "quadratic", "cubic"],
                    index=1,
                    help="Model type used for I-optimal criterion"
                )
                additional_params = {"n_runs": n_runs, "include_interior": include_interior, "model_type": i_model_type}
            else:
                additional_params = {"n_runs": n_runs, "include_interior": include_interior}
            
        elif design_method == "extreme-vertices":
            # This case is now handled above in the combined bounds section
            additional_params = {
                "lower_bounds": np.array([bound[0] for bound in component_bounds]) if component_bounds else np.zeros(n_components),
                "upper_bounds": np.array([bound[1] for bound in component_bounds]) if component_bounds else np.ones(n_components)
            }
        else:
            additional_params = {}
        
        # Generate button
        generate_button = st.button("🚀 Generate Design", type="primary")
    
    with col2:
        if generate_button:
            with st.spinner("Generating mixture design..."):
                try:
                    # Prepare parameters for design generation
                    design_params = {
                        'method': design_method,
                        'n_components': n_components,
                        'component_names': component_names,
                        **additional_params
                    }
                    
                    # Add model type only for optimization methods that actually use it
                    if design_method in ['d-optimal', 'i-optimal']:
                        design_params['model_type'] = model_type
                    
                    # Add parts mode and bounds if applicable
                    if use_parts_mode:
                        design_params['use_parts_mode'] = True
                        if component_bounds:
                            design_params['component_bounds'] = component_bounds
                        if fixed_components:
                            design_params['fixed_components'] = fixed_components
                    elif component_bounds and design_method == "extreme-vertices":
                        design_params['component_bounds'] = component_bounds
                    
                    # For D-optimal method with fixed components, use our NEW correct implementation
                    if design_method == 'd-optimal' and use_parts_mode and fixed_components:
                        st.info("🔧 Using NEW Fixed Components implementation (FixedPartsMixtureDesign)")
                        
                        # Convert component_bounds to variable_bounds for non-fixed components
                        variable_bounds = {}
                        if component_bounds:
                            # component_bounds only contains bounds for non-fixed components
                            # so we need to map them correctly
                            bounds_index = 0
                            for comp_name in component_names:
                                if comp_name not in fixed_components:
                                    if bounds_index < len(component_bounds):
                                        variable_bounds[comp_name] = component_bounds[bounds_index]
                                        bounds_index += 1
                        
                        # Create the NEW correct fixed components designer
                        fixed_designer = FixedPartsMixtureDesign(
                            component_names=component_names,
                            fixed_parts=fixed_components,
                            variable_bounds=variable_bounds
                        )
                        
                        # Generate design using the correct implementation
                        design_df = fixed_designer.generate_design(
                            n_runs=additional_params.get('n_runs', 12),
                            design_type="d-optimal",
                            model_type=model_type,
                            random_seed=42
                        )
                        
                        # Store the correct designer
                        st.session_state.fixed_designer = fixed_designer
                        st.session_state.d_optimal_designer = fixed_designer  # For backward compatibility
                        
                        # Extract just the parts columns for the parts design
                        full_design = fixed_designer.get_parts_design()
                        
                        # Check if it's a DataFrame or numpy array
                        if hasattr(full_design, 'columns'):
                            # It's a DataFrame
                            parts_cols = [col for col in full_design.columns if '_Parts' in col]
                            if parts_cols:  # Only proceed if we found parts columns
                                parts_design = full_design[parts_cols].values  # Extract as numpy array
                            else:
                                st.error("No parts columns found in design!")
                                parts_design = None
                        else:
                            # It's already a numpy array - use directly
                            if full_design.size > 0:  # Only proceed if array has data
                                parts_design = full_design
                            else:
                                st.error("Empty design array received!")
                                parts_design = None
                        
                        # Additional safety check for array dimensions
                        if parts_design is not None and parts_design.shape[1] != len(component_names):
                            st.warning(f"Design shape mismatch: {parts_design.shape[1]} columns vs {len(component_names)} components")
                            # Adjust component names to match design shape
                            component_names = component_names[:parts_design.shape[1]]
                            
                        if parts_design is not None:
                            st.session_state.correct_parts_design = parts_design
                    
                    # For D-optimal method WITHOUT fixed components, use old implementation
                    elif design_method == 'd-optimal':
                        from core.simplified_mixture_design import DOptimalMixtureDesign
                        
                        # Create the D-optimal designer once
                        d_optimal_designer = DOptimalMixtureDesign(
                            n_components, 
                            component_names, 
                            use_parts_mode, 
                            component_bounds, 
                            fixed_components
                        )
                        
                        # Generate design and capture OptimalDesignGenerator (single call)
                        design_df = d_optimal_designer.generate_design(**{k: v for k, v in design_params.items() 
                                                                         if k not in ['method', 'n_components', 'component_names', 
                                                                                     'use_parts_mode', 'component_bounds', 'fixed_components']})
                        
                        # Store in session state for later access
                        st.session_state.d_optimal_designer = d_optimal_designer
                        
                        # Access the OptimalDesignGenerator instance that was created internally
                        if hasattr(d_optimal_designer, '_last_generator'):
                            generator = d_optimal_designer._last_generator
                            # Use OptimalDesignGenerator's exact determinant and design points
                            det_value = generator.determinant_history[-1] if generator.determinant_history else 0.0
                            optimal_design_points = generator.design_points
                            
                            st.write(f"🚀 Using OptimalDesignGenerator's exact values:")
                            st.write(f"🔍 Debug: OptimalDesignGenerator determinant: {det_value:.6f}")
                            st.write(f"🔍 Debug: OptimalDesignGenerator design points shape: {len(optimal_design_points)}x{len(optimal_design_points[0]) if optimal_design_points else 0}")
                            
                            # Calculate info matrix from design_matrix (fixed approach)
                            try:
                                from core.optimal_design_generator import gram_matrix
                                if generator.design_matrix and len(generator.design_matrix) > 0:
                                    optimal_info_matrix = gram_matrix(generator.design_matrix)
                                    gram_np = np.array(optimal_info_matrix)
                                    n_params = len(optimal_info_matrix) if optimal_info_matrix else 0
                                else:
                                    gram_np = np.array([])
                                    n_params = 0
                            except Exception as e:
                                st.warning(f"Could not calculate info matrix: {e}")
                                gram_np = np.array([])
                                n_params = 0
                            
                        else:
                            st.warning("Could not access OptimalDesignGenerator instance - falling back to calculation")
                            det_value = 0.0
                            gram_np = np.array([])
                            n_params = 0
                    else:
                        # For other methods, use the standard approach
                        design_df = create_mixture_design(**design_params)
                        
                        # Get the design array in mixture proportion space (normalized to sum=1)
                        design_array = design_df.values
                        n_runs, n_components_actual = design_array.shape
                        
                        try:
                            from core.optimal_design_generator import gram_matrix, calculate_determinant
                            
                            # Convert mixture proportions [0,1] to [-1,1] space like OptimalDesignGenerator
                            X_standard = 2 * design_array - 1  # Convert [0,1] to [-1,1]
                            
                            # Build model matrix EXACTLY like OptimalDesignGenerator._evaluate_polynomial_terms
                            model_matrix_rows = []
                            for row_idx in range(n_runs):
                                x_values = X_standard[row_idx]  # Single row
                                
                                if model_type == 'linear':
                                    # Linear terms only for mixture
                                    row = []
                                    for i in range(n_components_actual):
                                        row.append(x_values[i])
                                
                                elif model_type == 'quadratic':
                                    # MIXTURE quadratic model (NO pure quadratic terms due to sum constraint)
                                    row = []
                                    # Linear terms
                                    for i in range(n_components_actual):
                                        row.append(x_values[i])
                                    # Two-way interaction terms ONLY (no xi² terms for mixtures)
                                    for i in range(n_components_actual):
                                        for j in range(i+1, n_components_actual):
                                            row.append(x_values[i] * x_values[j])
                                
                                else:  # cubic
                                    # MIXTURE cubic model (Scheffé canonical polynomials)
                                    row = []
                                    # Linear terms
                                    for i in range(n_components_actual):
                                        row.append(x_values[i])
                                    # Two-way interactions
                                    for i in range(n_components_actual):
                                        for j in range(i+1, n_components_actual):
                                            row.append(x_values[i] * x_values[j])
                                    # Three-way interactions
                                    for i in range(n_components_actual):
                                        for j in range(i+1, n_components_actual):
                                            for k in range(j+1, n_components_actual):
                                                row.append(x_values[i] * x_values[j] * x_values[k])
                                
                                model_matrix_rows.append(row)
                            
                            model_matrix = np.array(model_matrix_rows)
                            
                            # Calculate gram matrix in [-1,1] space (consistent with OptimalDesignGenerator)
                            info_matrix = gram_matrix(model_matrix.tolist())
                            det_value = calculate_determinant(info_matrix)
                            
                            # Additional metrics
                            gram_np = np.array(info_matrix)
                            n_params = model_matrix.shape[1]
                            
                            st.write(f"🔍 Debug: Coordinate space: [-1,1] (consistent with OptimalDesignGenerator)")
                            st.write(f"🔍 Debug: Model matrix shape: {model_matrix.shape}")
                            st.write(f"🔍 Debug: Determinant in [-1,1] space: {det_value:.6e}")
                        except Exception as e:
                            det_value = 0.0
                            gram_np = np.array([])
                            n_params = 0
                    
                    # Calculate D-efficiency and I-efficiency for all methods
                    # For fixed components design, extract only proportion columns for design_array
                    if design_method == 'd-optimal' and use_parts_mode and fixed_components:
                        # Extract proportion columns for efficiency calculations
                        prop_cols = [col for col in design_df.columns if '_Prop' in col]
                        if prop_cols:
                            design_array = design_df[prop_cols].values
                        else:
                            # Fallback: calculate proportions from the design matrix
                            design_array = design_df.values[:, :n_components]  # First n_components columns
                    else:
                        design_array = design_df.values
                    
                    d_efficiency = calculate_d_efficiency(design_array, model_type)
                    i_efficiency = calculate_i_efficiency(design_array, model_type)
                    
                    # Safe calculations for gram matrix metrics
                    if 'gram_np' in locals() and gram_np.size > 0 and len(gram_np.shape) == 2 and gram_np.shape[0] > 0:
                        condition_number = np.linalg.cond(gram_np)
                        trace_value = np.trace(gram_np)
                        eigenvals = np.linalg.eigvals(gram_np)
                        min_eigenvalue = np.real(eigenvals.min())
                        max_eigenvalue = np.real(eigenvals.max()) 
                        matrix_rank = np.linalg.matrix_rank(gram_np)
                        a_efficiency = n_params / trace_value if trace_value > 0 and 'n_params' in locals() else 0.0
                    else:
                        condition_number = float('inf')
                        trace_value = 0.0
                        min_eigenvalue = 0.0
                        max_eigenvalue = 0.0
                        matrix_rank = 0
                        a_efficiency = 0.0
                        if 'det_value' not in locals():
                            det_value = 0.0
                        if 'n_params' not in locals():
                            n_params = 0
                    
                    # Store in session state
                    st.session_state.design = design_df
                    st.session_state.design_array = design_array
                    st.session_state.d_efficiency = d_efficiency
                    st.session_state.i_efficiency = i_efficiency
                    st.session_state.component_names = component_names
                    st.session_state.design_method = design_method
                    # Store gram matrix metrics
                    model_matrix_shape = (n_runs, n_params) if 'model_matrix' in locals() else (0, 0)
                    
                    st.session_state.gram_metrics = {
                        'determinant': det_value,
                        'condition_number': condition_number,
                        'trace': trace_value,
                        'min_eigenvalue': min_eigenvalue,
                        'max_eigenvalue': max_eigenvalue,
                        'matrix_rank': matrix_rank,
                        'a_efficiency': a_efficiency,
                        'n_parameters': n_params,
                        'model_matrix_shape': model_matrix_shape,
                        'gram_matrix_shape': gram_np.shape if 'gram_np' in locals() else (0, 0)
                    }
                    
                    st.success("✅ Design generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating design: {str(e)}")
        
        # Display results if available
        if 'design' in st.session_state:
            design_df = st.session_state.design
            design_array = st.session_state.design_array
            d_efficiency = st.session_state.d_efficiency
            i_efficiency = st.session_state.i_efficiency
            
            # Metrics
            st.subheader("Design Metrics")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("D-Efficiency", f"{d_efficiency:.4f}")
            with col_b:
                st.metric("I-Efficiency", f"{i_efficiency:.4f}")
            with col_c:
                st.metric("Runs", len(design_df))
            
            # Gram Matrix Metrics
            if 'gram_metrics' in st.session_state:
                st.subheader("📊 Gram Matrix Metrics")
                gram_metrics = st.session_state.gram_metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Determinant", f"{gram_metrics['determinant']:.2e}")
                    st.metric("Matrix Rank", gram_metrics['matrix_rank'])
                with col2:
                    st.metric("Condition Number", f"{gram_metrics['condition_number']:.2e}")
                    st.metric("A-Efficiency", f"{gram_metrics['a_efficiency']:.4f}")
                with col3:
                    st.metric("Trace", f"{gram_metrics['trace']:.4f}")
                    st.metric("Min Eigenvalue", f"{gram_metrics['min_eigenvalue']:.4f}")
                with col4:
                    st.metric("Max Eigenvalue", f"{gram_metrics['max_eigenvalue']:.4f}")
                    st.metric("Parameters", gram_metrics['n_parameters'])
                
                # Additional details in expander
                with st.expander("🔬 Detailed Gram Matrix Information"):
                    col_detail1, col_detail2 = st.columns(2)
                    with col_detail1:
                        st.write("**Model Matrix Shape:**", gram_metrics['model_matrix_shape'])
                        st.write("**Gram Matrix Shape:**", gram_metrics['gram_matrix_shape'])
                        st.write("**Number of Parameters:**", gram_metrics['n_parameters'])
                        
                        # Add model explanation for mixture designs
                        st.write("**📋 Mixture Model Structure:**")
                        if 'model_type' in locals():
                            if model_type == "linear":
                                theoretical_params = n_components
                                effective_params = n_components - 1  # Due to sum constraint
                                st.write(f"- **Linear**: {n_components} components")
                                st.write("- Terms: x₁, x₂, x₃, ...")
                                st.write(f"- **Theoretical: {theoretical_params} parameters**")
                                st.warning(f"⚠️ **Effective: {effective_params} parameters** (due to ∑xᵢ = 1 constraint)")
                            elif model_type == "quadratic":
                                linear_terms = n_components
                                interaction_terms = (n_components * (n_components - 1)) // 2
                                theoretical_params = linear_terms + interaction_terms
                                # In mixture models, we typically lose 1 degree of freedom due to sum constraint
                                effective_params = theoretical_params - 1
                                st.write(f"- **Quadratic**: {n_components} linear + {interaction_terms} interactions")
                                st.write("- Linear: x₁, x₂, x₃, ...")
                                st.write("- Interactions: x₁x₂, x₁x₃, x₂x₃, ...")
                                st.write(f"- **Theoretical: {theoretical_params} parameters**")
                                st.warning(f"⚠️ **Effective: {effective_params} parameters** (due to ∑xᵢ = 1 constraint)")
                            elif model_type == "cubic":
                                linear_terms = n_components
                                quad_interactions = (n_components * (n_components - 1)) // 2
                                cubic_interactions = (n_components * (n_components - 1) * (n_components - 2)) // 6
                                theoretical_params = linear_terms + quad_interactions + cubic_interactions
                                effective_params = theoretical_params - 1
                                st.write(f"- **Cubic**: {linear_terms} linear + {quad_interactions} quadratic + {cubic_interactions} cubic")
                                st.write("- Linear: x₁, x₂, x₃, ...")
                                st.write("- Quadratic: x₁x₂, x₁x₃, ...")
                                st.write("- Cubic: x₁x₂x₃, ...")
                                st.write(f"- **Theoretical: {theoretical_params} parameters**")
                                st.warning(f"⚠️ **Effective: {effective_params} parameters** (due to ∑xᵢ = 1 constraint)")
                        
                        # Add explanation about mixture constraint (outside expander to avoid nesting)
                        st.info("""
                        **ℹ️ Why Parameter Reduction in Mixtures?**
                        
                        **Mixture Constraint: x₁ + x₂ + x₃ + ... = 1**
                        
                        In mixture experiments, the components must sum to 1 (100%), which creates a **linear dependency**:
                        
                        • If you know (n-1) components, the last one is determined: xₙ = 1 - x₁ - x₂ - ... - x₍ₙ₋₁₎
                        • This reduces the **effective degrees of freedom** by 1
                        • The model matrix becomes **rank-deficient** without this adjustment
                        • **No intercept term** is needed (unlike standard regression)
                        
                        **Example for 3 components:**
                        • Linear: 3 → 2 effective parameters (x₃ = 1 - x₁ - x₂)
                        • Quadratic: 6 → 5 effective parameters  
                        
                        This is **normal and expected** for mixture designs!
                        """)
                    with col_detail2:
                        st.write("**Matrix Properties:**")
                        st.write(f"- Determinant: {gram_metrics['determinant']:.6e}")
                        st.write(f"- Condition Number: {gram_metrics['condition_number']:.2e}")
                        st.write(f"- Matrix Rank: {gram_metrics['matrix_rank']}")
                        
                        # Condition number interpretation
                        cond_num = gram_metrics['condition_number']
                        if cond_num < 100:
                            st.success("✅ Well-conditioned matrix")
                        elif cond_num < 1000:
                            st.info("ℹ️ Moderately conditioned matrix")
                        elif cond_num < 10000:
                            st.warning("⚠️ Poorly conditioned matrix")
                        else:
                            st.error("❌ Severely ill-conditioned matrix")
                        
                        st.write("**🎯 Parameter Interpretation:**")
                        st.write("- **More parameters** = More complex model")
                        st.write("- **Quadratic models** capture curvature")
                        st.write("- **Interactions** show component synergies")
                        st.write("- **Need ≥ parameters runs** for estimation")
            
            # Design matrix
            st.subheader("Design Matrix - Proportions")
            
            # For fixed components design, extract only the proportion columns
            if design_method == 'd-optimal' and use_parts_mode and fixed_components:
                # Extract only proportion columns for clean display
                prop_cols = [col for col in design_df.columns if '_Prop' in col]
                if prop_cols:
                    # Create clean dataframe with just proportions
                    clean_df = design_df[prop_cols].copy()
                    # Rename columns to remove _Prop suffix
                    clean_df.columns = [col.replace('_Prop', '') for col in prop_cols]
                    display_df = clean_df
                else:
                    # Fallback: extract only the first n_components columns from design_array
                    if design_array.shape[1] >= n_components:
                        display_df = pd.DataFrame(design_array[:, :n_components], columns=component_names)
                    else:
                        # Last resort: use design_df directly with appropriate columns
                        available_cols = design_df.columns[:n_components] if len(design_df.columns) >= n_components else design_df.columns
                        display_df = design_df[available_cols].copy()
                        if len(available_cols) < n_components:
                            # Pad with component names if needed
                            display_df.columns = component_names[:len(available_cols)]
            else:
                # For other methods, ensure design_array has correct dimensions
                if design_array.shape[1] >= n_components:
                    # Always extract only the first n_components columns to match component_names
                    display_df = pd.DataFrame(design_array[:, :n_components], columns=component_names)
                else:
                    # If somehow we have fewer columns than components, use what we have
                    available_cols = min(design_array.shape[1], len(component_names))
                    display_df = pd.DataFrame(
                        design_array[:, :available_cols], 
                        columns=component_names[:available_cols]
                    )
            
            # **APPLY PRECISION CLEANUP** - Fix 0.9998/0.0002 issues
            from utils.mixture_utils import clean_numerical_precision
            display_df = clean_numerical_precision(display_df)
            
            # Add percentage columns
            for col_name in display_df.columns:
                display_df[f"{col_name} (%)"] = (display_df[col_name] * 100).round(1)
            
            st.dataframe(display_df.round(4))
            
            # Verify sum to 1
            sums = design_array.sum(axis=1)
            if np.allclose(sums, 1.0):
                st.success("✅ All mixtures sum to 100%")
            else:
                st.warning("⚠️ Some mixtures don't sum exactly to 100%")
            
            # MANUFACTURING WORKSHEETS - Available for ALL design methods!
            st.markdown("---")
            st.subheader("🏭 Manufacturing Worksheets")
            st.info("✨ **New**: Manufacturing worksheets are now available for ALL design methods, not just parts mode!")
            
            # Generate manufacturing quantities from ANY design
            # Convert design proportions to parts for manufacturing
            parts_design = None
            
            # First check if we have our NEW correct parts design (for fixed components in parts mode)
            if use_parts_mode and 'correct_parts_design' in st.session_state:
                parts_design = st.session_state.correct_parts_design
                conversion_method = "Fixed Components Parts Mode"
            
            # Then check if D-optimal designer has parts design (old implementation in parts mode)
            elif (use_parts_mode and design_method == 'd-optimal' and 
                'd_optimal_designer' in st.session_state and 
                hasattr(st.session_state.d_optimal_designer, 'parts_design') and 
                st.session_state.d_optimal_designer.parts_design is not None):
                parts_design = st.session_state.d_optimal_designer.parts_design
                conversion_method = "D-Optimal Parts Mode"
            
            # For ANY other design method (including non-parts mode), convert proportions to manufacturing quantities
            else:
                # For mixture designs, we ALWAYS need "parts per 100" format for manufacturing calculations
                # This ensures batch size scaling works correctly: actual_quantities = parts_design * batch_size / 100.0
                parts_design = design_array * 100.0
                
                if use_parts_mode and 'component_bounds' in locals() and component_bounds is not None:
                    # Display component bounds info for reference
                    total_parts_budget = sum(max_val for _, max_val in component_bounds)
                    conversion_method = f"Parts Mode: 100-part scaling (component bounds reference: {total_parts_budget} total)"
                else:
                    conversion_method = "Proportion-to-Manufacturing Conversion: scaled to 100 parts total"
            
            # Display parts design and manufacturing worksheets for ALL methods
            if parts_design is not None:
                
                # Show conversion method used
                st.info(f"✅ Conversion method: {conversion_method}")
                
                # Parts per 100 table
                st.markdown("### 📊 Parts per 100 Total")
                
                # Ensure shape compatibility between parts_design and component_names
                actual_n_components = parts_design.shape[1]
                if len(component_names) != actual_n_components:
                    if len(component_names) > actual_n_components:
                        # More component names than design columns - truncate names
                        effective_component_names = component_names[:actual_n_components]
                        st.warning(f"⚠️ Design has {actual_n_components} components but {len(component_names)} were specified. Using first {actual_n_components} component names.")
                    else:
                        # Fewer component names than design columns - generate additional names
                        effective_component_names = component_names.copy()
                        for i in range(len(component_names), actual_n_components):
                            effective_component_names.append(f"Component_{i+1}")
                        st.warning(f"⚠️ Design has {actual_n_components} components but only {len(component_names)} were specified. Generated additional component names.")
                else:
                    effective_component_names = component_names
                
                parts_df = pd.DataFrame(parts_design, columns=effective_component_names)
                parts_df.index = [f"Run_{i+1}" for i in range(len(parts_df))]
                
                # **APPLY PRECISION CLEANUP** - Fix 0.9998/0.0002 issues in Parts table
                from utils.mixture_utils import clean_numerical_precision
                parts_df = clean_numerical_precision(parts_df, preserve_mixture_constraint=False)
                
                # Add totals verification
                parts_df['Total_Parts'] = parts_design.sum(axis=1)
                
                st.dataframe(parts_df.round(3))
                
                # Manufacturing quantities with batch sizes
                st.markdown("### 🏭 Manufacturing Worksheets")
                
                # Batch size selector
                col_batch1, col_batch2, col_batch3 = st.columns(3)
                with col_batch1:
                    batch_size_1 = st.number_input("Batch Size 1 (kg)", min_value=0.1, value=1.0, step=0.1, key="batch_1")
                with col_batch2:
                    batch_size_2 = st.number_input("Batch Size 2 (kg)", min_value=0.1, value=5.0, step=0.5, key="batch_2")
                with col_batch3:
                    batch_size_3 = st.number_input("Batch Size 3 (kg)", min_value=0.1, value=10.0, step=1.0, key="batch_3")
                
                batch_sizes = [batch_size_1, batch_size_2, batch_size_3]
                
                # Create tabs for different batch sizes
                tabs = st.tabs([f"Batch {batch_size} kg" for batch_size in batch_sizes])
                
                manufacturing_worksheets = {}
                
                for idx, (tab, batch_size) in enumerate(zip(tabs, batch_sizes)):
                    with tab:
                        # Calculate actual quantities
                        # For fixed components design, we need to scale properly based on total parts per run
                        if use_parts_mode and 'correct_parts_design' in st.session_state:
                            # Our fixed components design has actual parts, need to scale by batch_size/total_parts
                            total_parts_per_run = parts_design.sum(axis=1, keepdims=True)
                            actual_quantities = parts_design * batch_size / total_parts_per_run
                        else:
                            # Standard design: parts are in "per 100" format
                            actual_quantities = parts_design * batch_size / 100.0
                        
                        # Create manufacturing worksheet
                        worksheet_df = pd.DataFrame()
                        worksheet_df['Run_ID'] = [f'EXP_{i+1:02d}' for i in range(len(actual_quantities))]
                        
                        # Add percentages first, then kg quantities for each component
                        for j, comp_name in enumerate(effective_component_names):
                            worksheet_df[f'{comp_name}_%'] = (parts_design[:, j]).round(2)
                            worksheet_df[f'{comp_name}_kg'] = actual_quantities[:, j].round(4)
                        
                        # Add totals and verification
                        worksheet_df['Total_kg'] = actual_quantities.sum(axis=1).round(4)
                        worksheet_df['Weight_Check'] = ['✓' if abs(total - batch_size) < 1e-3 else '✗' 
                                                       for total in worksheet_df['Total_kg']]
                        
                        # **APPLY PRECISION CLEANUP** - Fix 0.9998/0.0002 issues in Manufacturing Worksheets
                        from utils.mixture_utils import clean_numerical_precision
                        # Only clean numeric columns, skip text columns like Run_ID and Weight_Check
                        numeric_cols = [col for col in worksheet_df.columns if col not in ['Run_ID', 'Weight_Check']]
                        worksheet_df[numeric_cols] = clean_numerical_precision(worksheet_df[numeric_cols], preserve_mixture_constraint=False)
                        
                        st.dataframe(worksheet_df)
                        
                        # Material requirements summary
                        st.markdown("#### 📋 Material Requirements Summary")
                        total_materials = actual_quantities.sum(axis=0)
                        
                        summary_df = pd.DataFrame({
                            'Component': effective_component_names,
                            'Total_Required_kg': total_materials.round(3),
                            'Min_per_Run_kg': actual_quantities.min(axis=0).round(3),
                            'Max_per_Run_kg': actual_quantities.max(axis=0).round(3),
                            'Order_with_20%_Buffer_kg': (total_materials * 1.2).round(3)
                        })
                        
                        st.dataframe(summary_df)
                        
                        col_summary1, col_summary2 = st.columns(2)
                        with col_summary1:
                            st.metric("Total Material (all runs)", f"{total_materials.sum():.2f} kg")
                        with col_summary2:
                            st.metric("With 20% Buffer", f"{total_materials.sum() * 1.2:.2f} kg")
                        
                        # Store for download
                        manufacturing_worksheets[f"batch_{batch_size}kg"] = {
                            'worksheet': worksheet_df,
                            'summary': summary_df,
                            'batch_size': batch_size
                        }
                
                # Store manufacturing data in session state for downloads
                st.session_state.manufacturing_worksheets = manufacturing_worksheets
                st.session_state.parts_design_df = parts_df
            
            else:
                st.warning("⚠️ Manufacturing worksheets not available. Please regenerate the design.")
            
            # ENHANCED DOWNLOAD SECTION WITH RUN NUMBERS AND BATCH QUANTITIES
            st.markdown("---")
            st.subheader("📥 Enhanced Download Options")
            st.info("🆕 Excel files now include run numbers and manufacturing worksheets with batch quantities!")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                st.markdown("#### 📊 Basic Design (with Run Numbers)")
                
                # Enhanced CSV with run numbers
                enhanced_basic_df = display_df.copy()
                enhanced_basic_df.insert(0, 'Run_Number', range(1, len(display_df) + 1))
                csv_enhanced = enhanced_basic_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Design Matrix + Run Numbers (CSV)",
                    data=csv_enhanced,
                    file_name="mixture_design_with_run_numbers.csv",
                    mime="text/csv"
                )
                
                # Enhanced Excel with formatting and run numbers
                try:
                    # FIXED: Extract actual component names from display_df to avoid empty columns
                    # Filter out percentage columns and get base component names
                    actual_component_cols = [col for col in display_df.columns if not col.endswith(' (%)')]
                    actual_component_names = actual_component_cols[:n_components]  # Limit to expected number
                    
                    # Create a clean DataFrame with only the base proportion columns and correct names
                    clean_design_df = display_df[actual_component_names].copy()
                    
                    excel_enhanced_data = create_enhanced_excel_export(
                        design_df=clean_design_df,
                        component_names=actual_component_names,
                        use_parts_mode=False,
                        filename=None
                    )
                    
                    st.download_button(
                        label="📈 Enhanced Design Matrix (Excel)",
                        data=excel_enhanced_data,
                        file_name="enhanced_mixture_design.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    st.warning(f"Enhanced Excel export not available: {e}")
                    # Fallback to basic Excel
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        enhanced_basic_df.to_excel(writer, sheet_name='Design_Matrix', index=False)
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📈 Basic Design Matrix (Excel)",
                        data=excel_data,
                        file_name="mixture_design_basic.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with download_col2:
                if use_parts_mode:
                    st.markdown("#### 🏭 Parts Mode Downloads")
                    
                    # Get parts design for enhanced export
                    export_parts_design = None
                    if 'correct_parts_design' in st.session_state:
                        export_parts_design = st.session_state.correct_parts_design
                    elif parts_design is not None:
                        export_parts_design = parts_design
                    
                    if export_parts_design is not None:
                        # Enhanced parts CSV with run numbers
                        # Ensure shape compatibility between export_parts_design and component_names
                        actual_n_components = export_parts_design.shape[1]
                        if len(component_names) != actual_n_components:
                            if len(component_names) > actual_n_components:
                                # More component names than design columns - truncate names
                                effective_component_names = component_names[:actual_n_components]
                            else:
                                # Fewer component names than design columns - generate additional names
                                effective_component_names = component_names.copy()
                                for i in range(len(component_names), actual_n_components):
                                    effective_component_names.append(f"Component_{i+1}")
                        else:
                            effective_component_names = component_names
                        
                        parts_with_runs_df = pd.DataFrame(export_parts_design, columns=effective_component_names)
                        parts_with_runs_df.insert(0, 'Run_Number', range(1, len(export_parts_design) + 1))
                        parts_with_runs_df['Total_Parts'] = export_parts_design.sum(axis=1)
                        
                        csv_parts_enhanced = parts_with_runs_df.to_csv(index=False)
                        st.download_button(
                            label="🏭 Parts Design + Run Numbers (CSV)",
                            data=csv_parts_enhanced,
                            file_name="mixture_parts_with_run_numbers.csv",
                            mime="text/csv"
                        )
                        
                        # Enhanced Excel with parts mode
                        try:
                            # FIXED: Extract actual component names from display_df to avoid empty columns
                            actual_component_cols = [col for col in display_df.columns if not col.endswith(' (%)')]
                            actual_component_names = actual_component_cols[:n_components]  # Limit to expected number
                            
                            # Create a clean DataFrame with only the base proportion columns and correct names
                            clean_design_df = display_df[actual_component_names].copy()
                            
                            excel_parts_enhanced_data = create_enhanced_excel_export(
                                design_df=clean_design_df,
                                component_names=actual_component_names,
                                use_parts_mode=True,
                                parts_design=export_parts_design,
                                filename=None
                            )
                            
                            st.download_button(
                                label="📊 Enhanced Parts Design (Excel)",
                                data=excel_parts_enhanced_data,
                                file_name="enhanced_parts_design.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.warning(f"Enhanced parts Excel export not available: {e}")
                    else:
                        st.info("Parts design not available for enhanced export")
                else:
                    st.markdown("#### 🏭 Parts Mode Downloads")
                    st.info("Enable parts mode to access enhanced parts downloads")
            
            with download_col3:
                if use_parts_mode and parts_design is not None:
                    st.markdown("#### 🏗️ Complete Manufacturing Package")
                    
                    # Get batch sizes from the interface
                    batch_sizes_for_export = []
                    if 'manufacturing_worksheets' in st.session_state:
                        batch_sizes_for_export = [
                            data['batch_size'] for data in st.session_state.manufacturing_worksheets.values()
                        ]
                    else:
                        # Use default batch sizes if manufacturing worksheets not created
                        batch_sizes_for_export = [1.0, 5.0, 10.0]
                    
                    # Get parts design for export
                    export_parts_design = None
                    if 'correct_parts_design' in st.session_state:
                        export_parts_design = st.session_state.correct_parts_design
                    else:
                        export_parts_design = parts_design
                    
                    try:
                        # FIXED: Extract actual component names from display_df to avoid empty columns
                        actual_component_cols = [col for col in display_df.columns if not col.endswith(' (%)')]
                        actual_component_names = actual_component_cols[:n_components]  # Limit to expected number
                        
                        # Create a clean DataFrame with only the base proportion columns and correct names
                        clean_design_df = display_df[actual_component_names].copy()
                        
                        # Create complete manufacturing package with batch quantities
                        excel_manufacturing_complete = create_enhanced_excel_export(
                            design_df=clean_design_df,
                            component_names=actual_component_names,
                            use_parts_mode=True,
                            parts_design=export_parts_design,
                            batch_sizes=batch_sizes_for_export,
                            filename=None
                        )
                        
                        st.download_button(
                            label="🏭 Complete Manufacturing Package (Excel)",
                            data=excel_manufacturing_complete,
                            file_name="complete_manufacturing_package.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("✅ Includes:")
                        st.write("• Design matrix with run numbers")
                        st.write("• Parts design with totals")
                        st.write("• Manufacturing worksheets for each batch size")
                        st.write("• Material requirement summaries")
                        st.write("• Professional formatting and verification")
                        
                    except Exception as e:
                        st.warning(f"Complete package not available: {e}")
                        st.info("Using fallback individual downloads...")
                        
                        # Fallback to individual downloads
                        if 'manufacturing_worksheets' in st.session_state:
                            manufacturing_worksheets = st.session_state.manufacturing_worksheets
                            
                            for key, data in manufacturing_worksheets.items():
                                batch_size = data['batch_size']
                                worksheet_df = data['worksheet']
                                csv_worksheet = worksheet_df.to_csv(index=False)
                                
                                st.download_button(
                                    label=f"📋 Manufacturing {batch_size}kg (CSV)",
                                    data=csv_worksheet,
                                    file_name=f"manufacturing_{batch_size}kg.csv",
                                    mime="text/csv",
                                    key=f"fallback_download_{key}"
                                )
                else:
                    st.markdown("#### 🏗️ Complete Manufacturing Package")
                    st.info("Enable parts mode and generate manufacturing worksheets to access the complete package")
            
            # Visualization
            st.subheader("Design Visualization")
            
            if n_components == 3:
                # Ternary plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatterternary({
                    'mode': 'markers+text',
                    'a': design_array[:, 0],
                    'b': design_array[:, 1],
                    'c': design_array[:, 2],
                    'text': [f"R{i+1}" for i in range(len(design_array))],
                    'textposition': "top center",
                    'marker': {
                        'symbol': 'circle',
                        'size': 12,
                        'color': 'red',
                        'line': {'width': 2, 'color': 'black'}
                    }
                }))
                
                fig.update_layout({
                    'ternary': {
                        'sum': 1,
                        'aaxis': {'title': component_names[0]},
                        'baxis': {'title': component_names[1]},
                        'caxis': {'title': component_names[2]}
                    },
                    'height': 500,
                    'title': f"{design_method.replace('-', ' ').title()} Design"
                })
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif n_components == 2:
                # 2D scatter plot
                fig = px.scatter(
                    x=design_array[:, 0],
                    y=design_array[:, 1],
                    text=[f"R{i+1}" for i in range(len(design_array))],
                    labels={'x': component_names[0], 'y': component_names[1]},
                    title=f"{design_method.replace('-', ' ').title()} Design"
                )
                fig.update_traces(textposition='top center', marker_size=10)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Parallel coordinates for > 3 components
                fig = go.Figure()
                
                for i in range(len(design_array)):
                    fig.add_trace(go.Scatter(
                        x=list(range(n_components)),
                        y=design_array[i],
                        mode='lines+markers',
                        name=f'Run {i+1}',
                        showlegend=(i < 10)  # Only show first 10 in legend
                    ))
                
                fig.update_layout(
                    title="Parallel Coordinates Plot",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(n_components)),
                        ticktext=component_names
                    ),
                    yaxis_title="Proportion",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

elif design_type == "Standard DOE":
    st.markdown('<h2 class="sub-header">⚙️ Standard Design of Experiments</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Standard DOE Parameters")
        
        # Number of variables (not components - these are independent factors)
        n_variables = st.number_input(
            "Number of Variables", 
            min_value=2, 
            max_value=10, 
            value=3,
            help="Number of independent variables (factors)"
        )
        
        # Variable names
        st.write("**Variable Names:**")
        variable_names = []
        for i in range(n_variables):
            name = st.text_input(
                f"Variable {i+1} name",
                value=f"X{i+1}",
                key=f"std_var_name_{i}"
            )
            variable_names.append(name)
        
        # Design type selection
        design_method = st.selectbox(
            "Design Method",
            ["d-optimal", "i-optimal"],
            format_func=lambda x: {
                "d-optimal": "D-Optimal (Parameter Estimation)",
                "i-optimal": "I-Optimal (Prediction)"
            }[x],
            help="Optimization criterion for design generation"
        )
        
        # Model type
        model_type = st.selectbox(
            "Model Type",
            ["linear", "quadratic", "cubic"],
            index=1,
            help="Polynomial model type"
        )
        
        # Number of runs
        n_runs = st.number_input(
            "Number of Runs",
            min_value=n_variables + 1,
            max_value=1000,
            value=max(15, 2 * n_variables),
            help="Number of experimental runs"
        )
        
        # Generate button
        generate_standard_button = st.button("🚀 Generate Standard DOE", type="primary")
    
    with col2:
        if generate_standard_button:
            with st.spinner("Generating standard DOE design..."):
                try:
                    # Import OptimalDesignGenerator directly
                    from core.optimal_design_generator import OptimalDesignGenerator
                    
                    # Create OptimalDesignGenerator for standard DOE with selected method
                    generator = OptimalDesignGenerator(
                        num_variables=n_variables,
                        num_runs=n_runs,
                        design_type="standard",  # Standard DOE (not mixture)
                        model_type=model_type
                    )
                    
                    # Generate optimal design using selected method
                    if design_method == "d-optimal":
                        final_det = generator.generate_optimal_design(method="d_optimal")
                    else:  # i-optimal
                        final_det = generator.generate_optimal_design(method="i_optimal")
                    
                    # Get design points (in [-1,1] range)
                    design_points = generator.design_points
                    design_array = np.array(design_points)
                    
                    # Create DataFrame with variable names
                    design_df = pd.DataFrame(design_array, columns=variable_names)
                    
                    # Get determinant and other metrics
                    det_value = generator.determinant_history[-1] if generator.determinant_history else 0.0
                    
                    # Calculate D-efficiency
                    n_params = generator.num_parameters
                    d_efficiency = (det_value / n_runs) ** (1/n_params) if det_value > 0 and n_params > 0 else 0.0
                    
                    # Calculate information matrix from design_matrix
                    try:
                        from core.optimal_design_generator import gram_matrix
                        
                        if generator.design_matrix and len(generator.design_matrix) > 0:
                            info_matrix = gram_matrix(generator.design_matrix)
                            gram_np = np.array(info_matrix)
                            condition_number = np.linalg.cond(gram_np)
                            trace_value = np.trace(gram_np)
                            eigenvals = np.linalg.eigvals(gram_np)
                            min_eigenvalue = np.real(eigenvals.min())
                            max_eigenvalue = np.real(eigenvals.max()) 
                            matrix_rank = np.linalg.matrix_rank(gram_np)
                            a_efficiency = n_params / trace_value if trace_value > 0 else 0.0
                        else:
                            condition_number = float('inf')
                            trace_value = 0.0
                            min_eigenvalue = 0.0
                            max_eigenvalue = 0.0
                            matrix_rank = 0
                            a_efficiency = 0.0
                            gram_np = np.array([])
                    except:
                        condition_number = float('inf')
                        trace_value = 0.0
                        min_eigenvalue = 0.0
                        max_eigenvalue = 0.0
                        matrix_rank = 0
                        a_efficiency = 0.0
                        gram_np = np.array([])
                    
                    # Store in session state
                    st.session_state.standard_design = design_df
                    st.session_state.standard_design_array = design_array
                    st.session_state.standard_d_efficiency = d_efficiency
                    st.session_state.standard_variable_names = variable_names
                    st.session_state.standard_model_type = model_type
                    
                    # Store gram matrix metrics
                    st.session_state.standard_gram_metrics = {
                        'determinant': det_value,
                        'condition_number': condition_number,
                        'trace': trace_value,
                        'min_eigenvalue': min_eigenvalue,
                        'max_eigenvalue': max_eigenvalue,
                        'matrix_rank': matrix_rank,
                        'a_efficiency': a_efficiency,
                        'n_parameters': n_params,
                        'model_matrix_shape': (n_runs, n_params),
                        'gram_matrix_shape': gram_np.shape if gram_np.size > 0 else (0, 0)
                    }
                    
                    st.success("✅ Standard DOE design generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating standard DOE design: {str(e)}")
        
        # Display results if available
        if 'standard_design' in st.session_state:
            design_df = st.session_state.standard_design
            design_array = st.session_state.standard_design_array
            d_efficiency = st.session_state.standard_d_efficiency
            variable_names = st.session_state.standard_variable_names
            model_type = st.session_state.standard_model_type
            
            # Metrics
            st.subheader("Design Metrics")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("D-Efficiency", f"{d_efficiency:.4f}")
            with col_b:
                st.metric("Runs", len(design_df))
            with col_c:
                st.metric("Variables", len(variable_names))
            
            # Gram Matrix Metrics
            if 'standard_gram_metrics' in st.session_state:
                st.subheader("📊 Gram Matrix Metrics")
                gram_metrics = st.session_state.standard_gram_metrics
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Determinant", f"{gram_metrics['determinant']:.2e}")
                    st.metric("Matrix Rank", gram_metrics['matrix_rank'])
                with col2:
                    st.metric("Condition Number", f"{gram_metrics['condition_number']:.2e}")
                    st.metric("A-Efficiency", f"{gram_metrics['a_efficiency']:.4f}")
                with col3:
                    st.metric("Trace", f"{gram_metrics['trace']:.4f}")
                    st.metric("Min Eigenvalue", f"{gram_metrics['min_eigenvalue']:.4f}")
                with col4:
                    st.metric("Max Eigenvalue", f"{gram_metrics['max_eigenvalue']:.4f}")
                    st.metric("Parameters", gram_metrics['n_parameters'])
                
                # Additional details in expander
                with st.expander("🔬 Detailed Information"):
                    col_detail1, col_detail2 = st.columns(2)
                    with col_detail1:
                        st.write("**Design Type:** Standard DOE (Independent Variables)")
                        st.write("**Variable Range:** [-1, 1] for each variable")
                        st.write("**Model Type:**", model_type.title())
                        st.write("**Parameters:**", gram_metrics['n_parameters'])
                        
                        # Add model explanation for standard DOE
                        st.write("**📋 Standard DOE Model Structure:**")
                        if model_type == "linear":
                            st.write(f"- **Linear**: {len(variable_names)} variables")
                            st.write("- Terms: x₁, x₂, x₃, ...")
                            st.write(f"- **Total: {len(variable_names)} parameters**")
                        elif model_type == "quadratic":
                            linear_terms = len(variable_names)
                            quad_terms = len(variable_names)
                            interaction_terms = (len(variable_names) * (len(variable_names) - 1)) // 2
                            total_params = linear_terms + quad_terms + interaction_terms
                            st.write(f"- **Quadratic**: {linear_terms} linear + {quad_terms} quadratic + {interaction_terms} interactions")
                            st.write("- Linear: x₁, x₂, x₃, ...")
                            st.write("- Quadratic: x₁², x₂², x₃², ...")
                            st.write("- Interactions: x₁x₂, x₁x₃, x₂x₃, ...")
                            st.write(f"- **Total: {total_params} parameters**")
                        elif model_type == "cubic":
                            linear_terms = len(variable_names)
                            quad_terms = len(variable_names)
                            interaction_terms = (len(variable_names) * (len(variable_names) - 1)) // 2
                            cubic_terms = len(variable_names)
                            quad_linear_interactions = len(variable_names) * (len(variable_names) - 1)
                            cubic_interactions = (len(variable_names) * (len(variable_names) - 1) * (len(variable_names) - 2)) // 6
                            total_params = linear_terms + quad_terms + interaction_terms + cubic_terms + quad_linear_interactions + cubic_interactions
                            st.write(f"- **Cubic**: Complex polynomial with {total_params} terms")
                            st.write("- Linear: x₁, x₂, x₃, ...")
                            st.write("- Quadratic: x₁², x₂², ..., x₁x₂, ...")
                            st.write("- Cubic: x₁³, x₂³, ..., x₁²x₂, ..., x₁x₂x₃, ...")
                            st.write(f"- **Total: {total_params} parameters**")
                    with col_detail2:
                        st.write("**Matrix Properties:**")
                        st.write(f"- Determinant: {gram_metrics['determinant']:.6e}")
                        st.write(f"- Condition Number: {gram_metrics['condition_number']:.2e}")
                        st.write(f"- Matrix Rank: {gram_metrics['matrix_rank']}")
                        
                        # Condition number interpretation
                        cond_num = gram_metrics['condition_number']
                        if cond_num < 100:
                            st.success("✅ Well-conditioned matrix")
                        elif cond_num < 1000:
                            st.info("ℹ️ Moderately conditioned matrix")
                        elif cond_num < 10000:
                            st.warning("⚠️ Poorly conditioned matrix")
                        else:
                            st.error("❌ Severely ill-conditioned matrix")
                        
                        st.write("**🎯 Parameter Interpretation:**")
                        st.write("- **More parameters** = More complex model")
                        st.write("- **Quadratic models** capture curvature & interactions")
                        st.write("- **Cubic models** capture even more complex relationships")
                        st.write("- **Need ≥ parameters runs** for model estimation")
                        st.write("- **Variables are independent** (unlike mixture constraints)")
            
            # Design matrix
            st.subheader("Design Matrix")
            st.info("Variables are in [-1, 1] range (standard for DOE)")
            
            st.dataframe(design_df.round(4))
            
            # Check range
            min_vals = design_array.min(axis=0)
            max_vals = design_array.max(axis=0)
            if np.all(min_vals >= -1.1) and np.all(max_vals <= 1.1):
                st.success("✅ All variables are within [-1, 1] range")
            else:
                st.warning("⚠️ Some variables are outside the expected [-1, 1] range")
            
            # Download button
            csv = design_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Standard DOE Design",
                data=csv,
                file_name="standard_doe_design.csv",
                mime="text/csv"
            )
            
            # Visualization
            st.subheader("Design Visualization")
            
            if len(variable_names) == 2:
                # 2D scatter plot
                fig = px.scatter(
                    x=design_array[:, 0],
                    y=design_array[:, 1],
                    text=[f"R{i+1}" for i in range(len(design_array))],
                    labels={'x': variable_names[0], 'y': variable_names[1]},
                    title=f"Standard DOE Design ({model_type.title()} Model)"
                )
                fig.update_traces(textposition='top center', marker_size=10)
                fig.update_layout(
                    xaxis=dict(range=[-1.2, 1.2]),
                    yaxis=dict(range=[-1.2, 1.2])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif len(variable_names) == 3:
                # 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=design_array[:, 0],
                    y=design_array[:, 1],
                    z=design_array[:, 2],
                    mode='markers+text',
                    text=[f"R{i+1}" for i in range(len(design_array))],
                    textposition='top center',
                    marker=dict(
                        size=8,
                        color='blue',
                        line=dict(width=2, color='black')
                    )
                )])
                
                fig.update_layout(
                    title=f"Standard DOE Design ({model_type.title()} Model)",
                    scene=dict(
                        xaxis=dict(title=variable_names[0], range=[-1.2, 1.2]),
                        yaxis=dict(title=variable_names[1], range=[-1.2, 1.2]),
                        zaxis=dict(title=variable_names[2], range=[-1.2, 1.2])
                    ),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Parallel coordinates for > 3 variables
                fig = go.Figure()
                
                for i in range(len(design_array)):
                    fig.add_trace(go.Scatter(
                        x=list(range(len(variable_names))),
                        y=design_array[i],
                        mode='lines+markers',
                        name=f'Run {i+1}',
                        showlegend=(i < 10)  # Only show first 10 in legend
                    ))
                
                fig.update_layout(
                    title="Parallel Coordinates Plot",
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(len(variable_names))),
                        ticktext=variable_names
                    ),
                    yaxis_title="Variable Value",
                    yaxis=dict(range=[-1.2, 1.2]),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

elif design_type == "D-Optimal Analysis":
    st.markdown('<h2 class="sub-header">📊 D-Optimal Design Analysis</h2>', unsafe_allow_html=True)
    
    st.info("""
    This section demonstrates the improvement in D-optimal design by including interior points
    instead of only corner points (vertices).
    """)
    
    # Settings
    col1, col2 = st.columns(2)
    
    with col1:
        n_components = st.number_input(
            "Number of Components",
            min_value=3,
            max_value=5,
            value=3
        )
        
        n_runs = st.number_input(
            "Number of Runs",
            min_value=5,
            max_value=30,
            value=10
        )
    
    with col2:
        compare_button = st.button("🔍 Compare D-Optimal Designs", type="primary")
    
    if compare_button:
        # Generate both designs
        st.subheader("Comparison Results")
        
        # Design without interior points (corners only)
        with st.spinner("Generating corner-only design..."):
            designer1 = DOptimalMixtureDesign(n_components)
            design1 = designer1.generate_design(n_runs=n_runs, include_interior=False)
            d_eff1 = calculate_d_efficiency(design1.values)
        
        # Design with interior points
        with st.spinner("Generating design with interior points..."):
            designer2 = DOptimalMixtureDesign(n_components)
            design2 = designer2.generate_design(n_runs=n_runs, include_interior=True)
            d_eff2 = calculate_d_efficiency(design2.values)
        
        # Display comparison
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("❌ Corners Only")
            st.metric("D-Efficiency", f"{d_eff1:.4f}")
            
            # Analyze point types
            corner_points = 0
            for point in design1.values:
                if np.sum(point > 0.001) == 1:
                    corner_points += 1
            
            st.metric("Corner Points", f"{corner_points}/{len(design1)} ({corner_points/len(design1)*100:.0f}%)")
            st.dataframe(design1.round(4))
        
        with col_b:
            st.subheader("✅ With Interior Points")
            st.metric("D-Efficiency", f"{d_eff2:.4f}")
            
            # Analyze point types
            corner_points = 0
            interior_points = 0
            for point in design2.values:
                if np.sum(point > 0.001) == 1:
                    corner_points += 1
                elif np.all(point > 0.001):
                    interior_points += 1
            
            st.metric("Interior Points", f"{interior_points}/{len(design2)} ({interior_points/len(design2)*100:.0f}%)")
            st.dataframe(design2.round(4))
        
        # Improvement
        improvement = (d_eff2 - d_eff1) / d_eff1 * 100
        if improvement > 0:
            st.success(f"✅ Including interior points improved D-efficiency by {improvement:.1f}%!")
        
        # Visualization if 3 components
        if n_components == 3:
            st.subheader("Visual Comparison")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Corners Only", "With Interior Points"],
                specs=[[{'type': 'ternary'}, {'type': 'ternary'}]]
            )
            
            # Plot design 1
            fig.add_trace(
                go.Scatterternary({
                    'mode': 'markers+text',
                    'a': design1.values[:, 0],
                    'b': design1.values[:, 1],
                    'c': design1.values[:, 2],
                    'text': [f"{i+1}" for i in range(len(design1))],
                    'marker': {'size': 10, 'color': 'red'}
                }),
                row=1, col=1
            )
            
            # Plot design 2
            fig.add_trace(
                go.Scatterternary({
                    'mode': 'markers+text',
                    'a': design2.values[:, 0],
                    'b': design2.values[:, 1],
                    'c': design2.values[:, 2],
                    'text': [f"{i+1}" for i in range(len(design2))],
                    'marker': {'size': 10, 'color': 'blue'}
                }),
                row=1, col=2
            )
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif design_type == "Design Comparison":
    st.markdown('<h2 class="sub-header">⚖️ Design Method Comparison</h2>', unsafe_allow_html=True)
    
    # Settings
    n_components = st.slider("Number of Components", 3, 5, 3)
    
    if st.button("Compare All Methods", type="primary"):
        # Generate designs with all methods
        methods = ["simplex-lattice", "simplex-centroid", "d-optimal", "extreme-vertices"]
        results = []
        
        with st.spinner("Generating designs..."):
            for method in methods:
                try:
                    if method == "simplex-lattice":
                        design = create_mixture_design(method, n_components, degree=3)
                    elif method == "d-optimal":
                        design = create_mixture_design(method, n_components, n_runs=15, include_interior=True)
                    elif method == "extreme-vertices":
                        # Simple bounds
                        lower = np.zeros(n_components)
                        upper = np.ones(n_components)
                        design = create_mixture_design(method, n_components, 
                                                     lower_bounds=lower, upper_bounds=upper)
                    else:
                        design = create_mixture_design(method, n_components)
                    
                    # Calculate metrics
                    d_eff = calculate_d_efficiency(design.values, "quadratic")
                    
                    results.append({
                        "Method": method.replace("-", " ").title(),
                        "Runs": len(design),
                        "D-Efficiency": d_eff,
                        "Design": design
                    })
                except Exception as e:
                    st.warning(f"Error with {method}: {e}")
        
        # Display comparison
        if results:
            comparison_df = pd.DataFrame([
                {
                    "Method": r["Method"],
                    "Runs": r["Runs"],
                    "D-Efficiency": f"{r['D-Efficiency']:.4f}"
                }
                for r in results
            ])
            
            st.dataframe(comparison_df)
            
            # Bar chart
            fig = px.bar(
                comparison_df,
                x="Method",
                y="D-Efficiency",
                title="D-Efficiency Comparison",
                text="D-Efficiency"
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

elif design_type == "About":
    st.markdown('<h2 class="sub-header">ℹ️ About Simplified Mixture Design</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Simplified Architecture
    
    This application uses a **"one method - one class"** architecture for clarity and maintainability.
    
    ### Available Design Classes
    
    - **SimplexLatticeDesign**: Systematic coverage of mixture space
    - **SimplexCentroidDesign**: Focus on centroids of all component subsets
    - **DOptimalMixtureDesign**: Optimized for parameter estimation (includes interior points!)
    - **ExtremeVerticesDesign**: For constrained mixture regions
    - **AugmentedDesign**: Add axial points to existing designs
    - **CustomMixtureDesign**: User-specified design points
    
    ### Key Improvements
    
    ✅ **D-Optimal Fix**: Now includes interior points for better efficiency (0.54+ instead of 0.33)  
    ✅ **Clean Architecture**: Each design method is a separate class  
    ✅ **Easy to Use**: Simple factory function `create_mixture_design()`  
    ✅ **Extensible**: Easy to add new design methods  
    
    ### D-Efficiency Explanation
    
    D-efficiency measures how well a design estimates model parameters:
    - **1.0** = Perfect efficiency (theoretical maximum)
    - **0.5+** = Good efficiency for practical use
    - **0.3** = Poor efficiency (original corner-only problem)
    
    The improvement from including interior points is significant!
    
    ### Usage Example
    
    ```python
    from simplified_mixture_design import create_mixture_design
    
    # Create a D-optimal design with interior points
    design = create_mixture_design(
        method='d-optimal',
        n_components=3,
        n_runs=10,
        include_interior=True  # This is the key!
    )
    ```
    
    ---
    
    **Created with simplified, modular architecture for better maintainability**
    """)

# Footer
st.markdown("---")
st.markdown("**Simplified Mixture Design Generator** | Clean architecture, better results")
