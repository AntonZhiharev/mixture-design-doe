"""
Interactive Streamlit Application for Sequential Regression Reconstruction

This app provides a visual interface for:
- Setting up mixture experiments with constraints
- Viewing design points in real-time
- Analyzing responses and model fitting
- Monitoring convergence progress
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.sequential_regression_reconstruction import (
    SequentialRegressionReconstructor,
    ReconstructionConfig
)

# Configure page
st.set_page_config(
    page_title="Sequential Regression Reconstruction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_constrained_mixture_function(n_components=3):
    """Create a mixture function with realistic constraints for any number of components"""
    
    def mixture_response_function(point, noise_level=0.5):
        """
        Realistic mixture response function with constraints
        
        Dynamic response function that adapts to any number of components
        """
        point = np.array(point)
        
        # Validate mixture constraints
        if not np.isclose(sum(point), 1.0, atol=1e-6):
            st.error(f"Invalid mixture: components sum to {sum(point):.4f}, should be 1.0")
            return None
            
        if any(x < 0 for x in point):
            st.error("Invalid mixture: negative components not allowed")
            return None
        
        # Generate realistic coefficients based on component count
        # Linear effects: different impact levels for each component
        linear_coeffs = np.array([75, 85, 45, 60, 70][:n_components])  # Extend as needed
        if len(linear_coeffs) < n_components:
            # Generate additional coefficients if needed
            additional_coeffs = 50 + 30 * np.random.random(n_components - len(linear_coeffs))
            linear_coeffs = np.concatenate([linear_coeffs, additional_coeffs])
        
        linear = np.sum(linear_coeffs * point)
        
        # Quadratic interactions between all pairs of components
        quadratic = 0.0
        if n_components >= 2:
            # Generate interaction coefficients
            for i in range(n_components):
                for j in range(i+1, n_components):
                    if i == 0 and j == 1:
                        coeff = 40  # Strong synergy
                    elif i == 0 and j == 2:
                        coeff = 20  # Moderate synergy
                    elif i == 1 and j == 2:
                        coeff = -15  # Antagonism
                    else:
                        # Random interaction for additional components
                        coeff = np.random.uniform(-20, 30)
                    
                    quadratic += coeff * point[i] * point[j]
        
        # Add realistic noise
        noise = np.random.normal(0, noise_level)
        
        response = linear + quadratic + noise
        
        return max(0, response)  # Ensure non-negative response
    
    return mixture_response_function

def plot_design_points_3d(design_points, responses=None, iteration_colors=None, title="Design Points"):
    """Plot design points in 3D simplex space"""
    
    if len(design_points) == 0:
        st.warning("No design points to display")
        return
    
    df = pd.DataFrame(design_points, columns=['x1', 'x2', 'x3'])
    
    if responses is not None:
        df['Response'] = responses
        color_var = 'Response'
        color_scale = 'Viridis'
    elif iteration_colors is not None:
        df['Iteration'] = iteration_colors
        color_var = 'Iteration'
        color_scale = 'Set1'
    else:
        color_var = None
        color_scale = None
    
    # Create 3D scatter plot in simplex space
    fig = go.Figure()
    
    # Add design points
    if color_var:
        fig.add_trace(go.Scatter3d(
            x=df['x1'], y=df['x2'], z=df['x3'],
            mode='markers',
            marker=dict(
                size=8,
                color=df[color_var],
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title=color_var)
            ),
            text=[f"Point {i+1}<br>x1={row['x1']:.3f}<br>x2={row['x2']:.3f}<br>x3={row['x3']:.3f}" + 
                  (f"<br>Response={row['Response']:.2f}" if 'Response' in row else "")
                  for i, row in df.iterrows()],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Design Points'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=df['x1'], y=df['x2'], z=df['x3'],
            mode='markers',
            marker=dict(size=8, color='blue'),
            text=[f"Point {i+1}: ({row['x1']:.3f}, {row['x2']:.3f}, {row['x3']:.3f})" 
                  for i, row in df.iterrows()],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Design Points'
        ))
    
    # Add simplex boundaries (triangle edges)
    vertices = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0]])  # Close the triangle
    fig.add_trace(go.Scatter3d(
        x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
        mode='lines',
        line=dict(color='red', width=3),
        name='Simplex Boundary'
    ))
    
    # Add vertex labels
    labels = ['Pure x1', 'Pure x2', 'Pure x3']
    for i, (vertex, label) in enumerate(zip(vertices[:3], labels)):
        fig.add_trace(go.Scatter3d(
            x=[vertex[0]], y=[vertex[1]], z=[vertex[2]],
            mode='markers+text',
            marker=dict(size=12, color='red', symbol='diamond'),
            text=[label],
            textposition='top center',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Component 1 (x1)',
            yaxis_title='Component 2 (x2)', 
            zaxis_title='Component 3 (x3)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    return fig

def plot_response_analysis(reconstructor):
    """Create comprehensive response analysis plots"""
    
    if not reconstructor.iteration_history:
        st.warning("No iteration history to analyze")
        return
    
    # Get latest results
    latest_result = reconstructor.iteration_history[-1]
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Convergence History',
            'Response vs Predicted',
            'Residuals Distribution',
            'Coefficient Values'
        ),
        specs=[[{"secondary_y": True}, {}], [{}, {}]]
    )
    
    # 1. Convergence History
    iterations = [r.iteration for r in reconstructor.iteration_history]
    r_squared = [r.r_squared for r in reconstructor.iteration_history]
    d_efficiency = [r.d_efficiency for r in reconstructor.iteration_history]
    
    fig.add_trace(
        go.Scatter(x=iterations, y=r_squared, name='R²', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=iterations, y=d_efficiency, name='D-efficiency', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    
    # 2. Response vs Predicted
    model_results = reconstructor.iteration_history[-1]
    if hasattr(reconstructor, 'current_responses') and reconstructor.current_responses is not None:
        y_true = reconstructor.current_responses
        
        # Use appropriate feature matrix based on model type
        if reconstructor._is_mixture_design() and reconstructor.config.model_type == "quadratic":
            X_matrix = reconstructor._build_mixture_quadratic_matrix(reconstructor.current_design)
        elif reconstructor._is_mixture_design() and reconstructor.config.model_type == "cubic":
            # For now, treat cubic mixture model as quadratic (simplified)
            X_matrix = reconstructor._build_mixture_quadratic_matrix(reconstructor.current_design)
        else:
            X_matrix = reconstructor.current_design
        
        y_pred = reconstructor.current_model.predict(X_matrix)
        
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predicted vs Actual'),
            row=1, col=2
        )
        # Add perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction', line=dict(dash='dash')),
            row=1, col=2
        )
    
    # 3. Residuals Distribution
    if hasattr(latest_result, 'model_adequacy'):
        residuals = reconstructor.current_responses - y_pred
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals', nbinsx=15),
            row=2, col=1
        )
    
    # 4. Coefficient Values
    coeff_summary = reconstructor.get_coefficient_summary()
    fig.add_trace(
        go.Bar(x=coeff_summary['Term'], y=coeff_summary['Coefficient'], 
               name='Coefficients'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=1)
    fig.update_yaxes(title_text="D-efficiency", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Actual Response", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Response", row=1, col=2)
    fig.update_xaxes(title_text="Residual Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    fig.update_xaxes(title_text="Model Terms", row=2, col=2)
    fig.update_yaxes(title_text="Coefficient Value", row=2, col=2)
    
    return fig

def plot_ternary_design(design_points, responses=None, iteration_colors=None):
    """Plot design points on ternary diagram for 3-component mixtures"""
    
    if len(design_points) == 0:
        return None
    
    df = pd.DataFrame(design_points, columns=['x1', 'x2', 'x3'])
    
    if responses is not None:
        df['Response'] = responses
        color_var = 'Response'
    elif iteration_colors is not None:
        df['Iteration'] = iteration_colors  
        color_var = 'Iteration'
    else:
        color_var = None
    
    # Create ternary plot
    fig = go.Figure(go.Scatterternary(
        a=df['x1'], b=df['x2'], c=df['x3'],
        mode='markers',
        marker=dict(
            size=10,
            color=df[color_var] if color_var else 'blue',
            colorscale='Viridis' if color_var == 'Response' else 'Set1',
            showscale=True if color_var else False,
            colorbar=dict(title=color_var) if color_var else None
        ),
        text=[f"Point {i+1}<br>({row['x1']:.3f}, {row['x2']:.3f}, {row['x3']:.3f})" + 
              (f"<br>Response: {row['Response']:.2f}" if 'Response' in row else "")
              for i, row in df.iterrows()],
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title="Design Points on Ternary Diagram",
        ternary=dict(
            sum=1,
            aaxis=dict(title='Component 1 (x1)', min=0),
            baxis=dict(title='Component 2 (x2)', min=0), 
            caxis=dict(title='Component 3 (x3)', min=0)
        ),
        height=500
    )
    
    return fig

def generate_parameter_names(n_components, model_type):
    """Generate proper parameter names for mixture models."""
    names = []
    
    # Linear terms: β₁, β₂, β₃, ...
    for i in range(n_components):
        names.append(f"β_{i+1}")
    
    if model_type in ['quadratic', 'cubic']:
        # Interaction terms: β₁₂, β₁₃, β₂₃, ...
        for i in range(n_components):
            for j in range(i+1, n_components):
                names.append(f"β_{i+1}{j+1}")
    
    if model_type == 'cubic':
        # Cubic terms: β₁₂₃, β₁₂₄, ... (for 3+ components)
        if n_components >= 3:
            for i in range(n_components):
                for j in range(i+1, n_components):
                    for k in range(j+1, n_components):
                        names.append(f"β_{i+1}{j+1}{k+1}")
    
    return names


def extract_true_coefficients(response_function, n_components, model_type):
    """Extract true coefficients from the response function."""
    # For the mixture function, we know the structure from create_constrained_mixture_function
    coeffs = []
    
    # Linear terms - from the function definition
    if n_components == 3:
        linear_coeffs = [75, 85, 45]
    elif n_components == 4:
        linear_coeffs = [75, 85, 45, 60]
    elif n_components == 5:
        linear_coeffs = [75, 85, 45, 60, 70]
    else:
        linear_coeffs = [75, 85][:n_components]
    
    coeffs.extend(linear_coeffs[:n_components])
    
    if model_type in ['quadratic', 'cubic']:
        # Interaction terms - from the function definition
        interactions = []
        for i in range(n_components):
            for j in range(i+1, n_components):
                if i == 0 and j == 1:
                    coeff = 40  # Strong synergy
                elif i == 0 and j == 2:
                    coeff = 20  # Moderate synergy
                elif i == 1 and j == 2:
                    coeff = -15  # Antagonism
                else:
                    # Use default values for additional interactions
                    coeff = 10  # Default moderate interaction
                interactions.append(coeff)
        coeffs.extend(interactions)
    
    return coeffs


def main():
    """Main Streamlit application"""
    
    st.title("🧪 Sequential Regression Reconstruction")
    st.markdown("Interactive interface for mixture design experiments")
    
    # Initialize session state
    if 'reconstructor' not in st.session_state:
        st.session_state.reconstructor = None
    if 'response_function' not in st.session_state:
        st.session_state.response_function = None
    if 'experiment_running' not in st.session_state:
        st.session_state.experiment_running = False
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model configuration
    st.sidebar.subheader("Model Parameters")
    n_components = st.sidebar.slider("Number of Components", 2, 5, 3)
    
    # Calculate parameters for each model type
    def calculate_parameters(n_comp, model):
        if model == "linear":
            return n_comp
        elif model == "quadratic":
            return n_comp + (n_comp * (n_comp - 1)) // 2
        elif model == "cubic":
            linear = n_comp
            quadratic = (n_comp * (n_comp - 1)) // 2
            cubic = (n_comp * (n_comp - 1) * (n_comp - 2)) // 6
            return linear + quadratic + cubic
        return n_comp
    
    # Model type options with parameter counts
    model_options = []
    for model in ["linear", "quadratic", "cubic"]:
        params = calculate_parameters(n_components, model)
        if params <= 20:  # Reasonable limit for parameter estimation
            model_options.append(f"{model.capitalize()} ({params} parameters)")
    
    # Display model selection
    model_selection = st.sidebar.selectbox("Model Type", model_options)
    model_type = model_selection.split(" ")[0].lower()
    
    # Show parameter breakdown
    with st.sidebar.expander("📊 Parameter Details"):
        st.write(f"**{model_type.capitalize()} Model for {n_components} components:**")
        
        if model_type == "linear":
            st.write(f"• **Linear terms: {n_components}**")
            for i in range(n_components):
                st.write(f"  - β{i+1}: Effect of component {i+1}")
            st.write(f"• **Total parameters: {calculate_parameters(n_components, model_type)}**")
            st.write("Model form: Y = β₁x₁ + β₂x₂ + ... + βₙxₙ")
            
        elif model_type == "quadratic":
            linear_params = n_components
            quad_params = (n_components * (n_components - 1)) // 2
            st.write(f"• **Linear terms: {linear_params}**")
            for i in range(n_components):
                st.write(f"  - β{i+1}: Main effect of component {i+1}")
            st.write(f"• **Interaction terms: {quad_params}**")
            interaction_count = 0
            for i in range(n_components):
                for j in range(i+1, n_components):
                    interaction_count += 1
                    st.write(f"  - β{linear_params + interaction_count}: Interaction x{i+1} × x{j+1}")
            st.write(f"• **Total parameters: {linear_params + quad_params}**")
            st.write("Model form: Y = Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ")
            
        elif model_type == "cubic":
            linear_params = n_components
            quad_params = (n_components * (n_components - 1)) // 2
            cubic_params = (n_components * (n_components - 1) * (n_components - 2)) // 6
            st.write(f"• **Linear terms: {linear_params}**")
            for i in range(n_components):
                st.write(f"  - β{i+1}: Main effect of component {i+1}")
            st.write(f"• **Quadratic interactions: {quad_params}**")
            st.write(f"• **Cubic interactions: {cubic_params}**")
            st.write(f"• **Total parameters: {linear_params + quad_params + cubic_params}**")
            st.write("Model form: Y = Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢⱼₖxᵢxⱼxₖ")
    
    max_iterations = st.sidebar.slider("Max Iterations", 3, 20, 8)
    
    # Batch sizes
    st.sidebar.subheader("Experimental Design")
    
    # Calculate recommended batch sizes based on parameters
    total_params = calculate_parameters(n_components, model_type)
    min_initial = max(total_params + 2, 5)  # At least parameters + 2 for DOF
    recommended_initial = min(min_initial * 2, 20)  # Roughly 2x parameters
    
    st.sidebar.write(f"**Recommended minimum:** {min_initial} points")
    initial_batch = st.sidebar.slider(
        "Initial Batch Size", 
        min_initial, 
        25, 
        recommended_initial,
        help=f"Minimum {min_initial} points needed for {total_params} parameters"
    )
    
    sequential_batch = st.sidebar.slider("Sequential Batch Size", 2, 10, 3)
    
    # Convergence criteria
    st.sidebar.subheader("Convergence Criteria")
    r2_threshold = st.sidebar.slider("R² Threshold", 0.70, 0.99, 0.85)
    d_efficiency_threshold = st.sidebar.slider("D-efficiency Threshold", 0.50, 0.95, 0.70)
    
    # Mixture constraints
    st.sidebar.subheader("Mixture Constraints")
    use_bounds = st.sidebar.checkbox("Use Component Bounds")
    
    lower_bounds = None
    upper_bounds = None
    if use_bounds and n_components == 3:
        st.sidebar.markdown("**Component Bounds:**")
        x1_min, x1_max = st.sidebar.slider("Component 1 Range", 0.0, 1.0, (0.1, 0.8), step=0.05)
        x2_min, x2_max = st.sidebar.slider("Component 2 Range", 0.0, 1.0, (0.0, 0.7), step=0.05)
        x3_min, x3_max = st.sidebar.slider("Component 3 Range", 0.0, 1.0, (0.1, 0.6), step=0.05)
        
        lower_bounds = [x1_min, x2_min, x3_min]
        upper_bounds = [x1_max, x2_max, x3_max]
        
        # Validate constraints
        if sum(lower_bounds) >= 1.0:
            st.sidebar.error("Lower bounds sum exceeds 1.0! Adjust constraints.")
        if sum(upper_bounds) < 1.0:
            st.sidebar.error("Upper bounds sum less than 1.0! Some constraints impossible.")
    
    # Fixed components
    use_fixed = st.sidebar.checkbox("Fix Some Components")
    fixed_components = None
    if use_fixed and n_components >= 3:
        fixed_comp_idx = st.sidebar.selectbox("Fixed Component", range(n_components))
        fixed_value = st.sidebar.slider("Fixed Value", 0.01, 0.50, 0.10)
        fixed_components = {fixed_comp_idx: fixed_value}
        
        remaining = 1.0 - fixed_value
        st.sidebar.info(f"Remaining components must sum to {remaining:.2f}")
    
    # Response function parameters
    st.sidebar.subheader("Response Function")
    noise_level = st.sidebar.slider("Noise Level", 0.0, 2.0, 0.5, step=0.1)
    
    # Create configuration
    config = ReconstructionConfig(
        n_components=n_components,
        model_type=model_type,
        max_iterations=max_iterations,
        initial_batch_size=initial_batch,
        sequential_batch_size=sequential_batch,
        r2_threshold=r2_threshold,
        d_efficiency_threshold=d_efficiency_threshold,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        fixed_components=fixed_components
    )
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Experiment", "📊 Design Points", "📈 Analysis", "🔄 Stage Analysis", "📋 Results"])
    
    with tab1:
        st.header("Experiment Control")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Initialize New Experiment", type="primary"):
                # Create response function
                st.session_state.response_function = create_constrained_mixture_function(n_components)
                
                # Create reconstructor
                st.session_state.reconstructor = SequentialRegressionReconstructor(config)
                
                # Generate initial design
                with st.spinner("Generating initial design..."):
                    initial_design = st.session_state.reconstructor.generate_initial_design()
                    
                    # Collect initial responses
                    initial_responses = st.session_state.reconstructor.collect_responses(
                        initial_design, 
                        lambda x: st.session_state.response_function(x, noise_level)
                    )
                
                st.success("✅ Experiment initialized!")
                st.rerun()
        
        with col2:
            if (st.session_state.reconstructor is not None and 
                st.button("➡️ Run Next Iteration")):
                
                reconstructor = st.session_state.reconstructor
                
                with st.spinner("Running iteration..."):
                    # Analyze current data
                    analysis_results = reconstructor.analyze_current_data()
                    
                    # Check convergence
                    converged = reconstructor.check_convergence(analysis_results)
                    
                    if converged:
                        st.success("🎉 Experiment converged!")
                    else:
                        # Select parameters and generate additional points
                        selected_params = reconstructor.select_experimental_parameters(analysis_results)
                        additional_points = reconstructor.gen_additional_design_points(
                            selected_params, analysis_results
                        )
                        additional_responses = reconstructor.collect_responses(
                            additional_points,
                            lambda x: st.session_state.response_function(x, noise_level)
                        )
                        
                        # Save iteration
                        iteration = len(reconstructor.iteration_history) + 1
                        reconstructor.save_iteration_result(iteration, analysis_results, selected_params)
                        
                        st.success(f"✅ Iteration {iteration} completed!")
                
                st.rerun()
        
        # Display current status
        if st.session_state.reconstructor is not None:
            reconstructor = st.session_state.reconstructor
            
            st.subheader("📊 Current Status")
            
            if reconstructor.iteration_history:
                latest = reconstructor.iteration_history[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Iterations", len(reconstructor.iteration_history))
                with col2:
                    st.metric("Total Experiments", len(reconstructor.current_design))
                with col3:
                    st.metric("R²", f"{latest.r_squared:.4f}")
                with col4:
                    st.metric("D-efficiency", f"{latest.d_efficiency:.4f}")
                
                # Convergence status
                if reconstructor.converged:
                    st.success("🎉 **CONVERGED** - Experiment complete!")
                else:
                    st.info("🔄 **RUNNING** - Continue iterations to improve model")
                
                # Show constraints satisfaction
                if n_components == 3:
                    st.subheader("🔍 Constraint Validation")
                    design_sums = np.sum(reconstructor.current_design, axis=1)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Sum Constraint (should = 1.0):**")
                        constraint_ok = np.allclose(design_sums, 1.0, atol=1e-6)
                        if constraint_ok:
                            st.success(f"✅ All points sum correctly (range: {design_sums.min():.6f} - {design_sums.max():.6f})")
                        else:
                            st.error(f"❌ Sum constraint violated (range: {design_sums.min():.6f} - {design_sums.max():.6f})")
                    
                    with col2:
                        st.write("**Non-negativity Constraint:**")
                        non_negative = np.all(reconstructor.current_design >= -1e-10)  # Allow small numerical errors
                        if non_negative:
                            st.success("✅ All components non-negative")
                        else:
                            st.error("❌ Negative components detected")
    
    with tab2:
        st.header("📊 Design Points Visualization")
        
        if st.session_state.reconstructor is not None:
            reconstructor = st.session_state.reconstructor
            
            if n_components == 3:
                viz_type = st.selectbox("Visualization Type", 
                                       ["Ternary Diagram", "3D Simplex", "Data Table"])
                
                # Prepare iteration colors
                iteration_colors = []
                current_iter = 1
                for i, design_batch in enumerate(reconstructor.design_history):
                    iteration_colors.extend([current_iter] * len(design_batch))
                    if i > 0:  # Don't increment for initial design
                        current_iter += 1
                
                if viz_type == "Ternary Diagram":
                    fig = plot_ternary_design(
                        reconstructor.current_design, 
                        reconstructor.current_responses,
                        iteration_colors
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "3D Simplex":
                    fig = plot_design_points_3d(
                        reconstructor.current_design,
                        reconstructor.current_responses,
                        iteration_colors,
                        "All Design Points by Iteration"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Data Table":
                    df = pd.DataFrame(
                        reconstructor.current_design, 
                        columns=[f'Component_{i+1}' for i in range(n_components)]
                    )
                    df['Response'] = reconstructor.current_responses
                    df['Iteration'] = iteration_colors
                    df['Sum_Check'] = df[[f'Component_{i+1}' for i in range(n_components)]].sum(axis=1)
                    
                    st.dataframe(df, use_container_width=True)
                    
                    # Download data
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Design Data",
                        data=csv,
                        file_name="design_points.csv",
                        mime="text/csv"
                    )
            else:
                st.info(f"Visualization requires 3 components. Currently using {n_components} components.")
        else:
            st.info("Initialize an experiment to view design points")
    
    with tab3:
        st.header("📈 Response Analysis")
        
        if st.session_state.reconstructor is not None and st.session_state.reconstructor.iteration_history:
            fig = plot_response_analysis(st.session_state.reconstructor)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show coefficient summary with true coefficients
            st.subheader("🔢 Current Model Coefficients")
            
            # Use consistent approach: manually build model using the same method as Stage Analysis
            reconstructor = st.session_state.reconstructor
            
            if len(reconstructor.current_design) > 0 and len(reconstructor.current_responses) > 0:
                try:
                    # Use the same approach as in Stage Analysis for consistency
                    current_design = np.array(reconstructor.current_design)
                    current_responses = np.array(reconstructor.current_responses)
                    
                    # Build feature matrix using the same method as Stage Analysis
                    if config.model_type == "quadratic" and reconstructor._is_mixture_design():
                        X_matrix = reconstructor._build_mixture_quadratic_matrix(current_design)
                    elif config.model_type == "cubic" and reconstructor._is_mixture_design():
                        # For now, treat cubic mixture model as quadratic (simplified)
                        X_matrix = reconstructor._build_mixture_quadratic_matrix(current_design)
                    else:
                        X_matrix = current_design
                    
                    # Fit model using the same approach
                    from sklearn.linear_model import LinearRegression
                    current_model = LinearRegression()
                    current_model.fit(X_matrix, current_responses)
                    
                    if hasattr(current_model, 'coef_'):
                        param_names = generate_parameter_names(config.n_components, config.model_type)
                        
                        # Extract true coefficients
                        true_coeffs = extract_true_coefficients(
                            st.session_state.response_function,
                            config.n_components,
                            config.model_type
                        )
                        
                        coeff_df = pd.DataFrame({
                            'Parameter': param_names[:len(current_model.coef_)],
                            'True_Coefficient': true_coeffs[:len(current_model.coef_)],
                            'Recovered_Coefficient': current_model.coef_,
                            'Difference': np.array(true_coeffs[:len(current_model.coef_)]) - current_model.coef_,
                            'Absolute_Error': np.abs(np.array(true_coeffs[:len(current_model.coef_)]) - current_model.coef_),
                            'Relative_Error_%': np.where(np.array(true_coeffs[:len(current_model.coef_)]) != 0, 
                                                       100 * np.abs(np.array(true_coeffs[:len(current_model.coef_)]) - current_model.coef_) / np.abs(np.array(true_coeffs[:len(current_model.coef_)])), 
                                                       np.abs(current_model.coef_) * 100)
                        })
                        
                        st.dataframe(coeff_df, use_container_width=True)
                        
                        # Add a note about the approach
                        st.info("ℹ️ **Note:** Coefficients are computed using the same procedure as in Stage Analysis for consistency.")
                    else:
                        st.info("Model coefficients not available yet.")
                
                except Exception as e:
                    st.error(f"Could not compute model coefficients: {str(e)}")
                    # Fallback to original method
                    coeff_summary = reconstructor.get_coefficient_summary()
                    st.dataframe(coeff_summary, use_container_width=True)
            else:
                st.info("No experiment data available yet.")
            
            # Model adequacy
            if st.session_state.reconstructor.iteration_history:
                latest = st.session_state.reconstructor.iteration_history[-1]
                adequacy = latest.model_adequacy
                
                st.subheader("✅ Model Adequacy Assessment")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    r2_ok = adequacy.get('r_squared_adequate', False)
                    st.metric("R² Adequate", "✅ Yes" if r2_ok else "❌ No")
                
                with col2:
                    normal_ok = adequacy.get('normality_adequate', False)
                    st.metric("Residuals Normal", "✅ Yes" if normal_ok else "❌ No")
                
                with col3:
                    overall_ok = adequacy.get('overall_adequate', False)
                    st.metric("Overall Adequate", "✅ Yes" if overall_ok else "❌ No")
        else:
            st.info("Run experiment to view response analysis")
    
    with tab4:
        st.header("🔄 Stage-by-Stage Analysis")
        
        if st.session_state.reconstructor is not None and st.session_state.reconstructor.iteration_history:
            reconstructor = st.session_state.reconstructor
            
            # Stage selector
            stage_options = ["Initial"] + [f"Iteration {i+1}" for i in range(len(reconstructor.iteration_history))]
            selected_stage = st.selectbox("Select Stage", stage_options)
            
            if selected_stage == "Initial":
                stage_idx = -1  # Initial stage
                st.subheader("📊 Initial Design Stage")
                st.info("This shows the initial design before any iterations.")
            else:
                stage_idx = int(selected_stage.split()[-1]) - 1
                st.subheader(f"📊 {selected_stage} Results")
            
            # Get stage-specific data
            if stage_idx == -1:
                # Initial stage data - before any iterations
                if len(reconstructor.design_history) > 0:
                    cumulative_design = reconstructor.design_history[0]
                    cumulative_responses = reconstructor.response_history[0]
                    stage_design = cumulative_design
                    stage_responses = cumulative_responses
                    stage_model = None  # No model yet for initial
                else:
                    st.warning("No design history available")
                    cumulative_design = []
                    cumulative_responses = []
                    stage_design = []
                    stage_responses = []
                    stage_model = None
            else:
                # Specific iteration data
                if stage_idx < len(reconstructor.iteration_history):
                    # Get cumulative design up to this stage
                    cumulative_design = []
                    cumulative_responses = []
                    for i in range(min(stage_idx + 2, len(reconstructor.design_history))):
                        if i < len(reconstructor.design_history):
                            cumulative_design.extend(reconstructor.design_history[i])
                            cumulative_responses.extend(reconstructor.response_history[i])
                    
                    # Get this iteration's new points
                    if stage_idx + 1 < len(reconstructor.design_history):
                        stage_design = reconstructor.design_history[stage_idx + 1]
                        stage_responses = reconstructor.response_history[stage_idx + 1]
                    else:
                        stage_design = []
                        stage_responses = []
                    
                    # Get iteration results for model coefficients
                    stage_iteration_result = reconstructor.iteration_history[stage_idx]
                else:
                    st.error("Invalid stage selection")
                    cumulative_design = []
                    cumulative_responses = []
                    stage_design = []
                    stage_responses = []
                    stage_iteration_result = None
            
            if len(cumulative_design) > 0:
                # Display stage information
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("New Points This Stage", len(stage_design) if len(stage_design) > 0 else 0)
                with col2:
                    st.metric("Cumulative Points", len(cumulative_design))
                with col3:
                    if stage_idx >= 0 and stage_idx < len(reconstructor.iteration_history):
                        stage_result = reconstructor.iteration_history[stage_idx]
                        st.metric("Stage R²", f"{stage_result.r_squared:.4f}")
                    else:
                        st.metric("Stage R²", "Initial")
                
                # 1. DESIGN POINTS AT THIS STAGE
                st.subheader("🎯 Design Points at This Stage")
                
                if len(stage_design) > 0:
                    # Show new points added in this stage
                    stage_df = pd.DataFrame(
                        stage_design, 
                        columns=[f'Component_{i+1}' for i in range(n_components)]
                    )
                    stage_df['Response'] = stage_responses
                    stage_df['Point_Type'] = f"New in {selected_stage}"
                    st.dataframe(stage_df, use_container_width=True)
                else:
                    st.info("No new points added in this stage")
                
                # Show cumulative design
                st.subheader("📊 Cumulative Design Points")
                cumulative_df = pd.DataFrame(
                    cumulative_design,
                    columns=[f'Component_{i+1}' for i in range(n_components)]
                )
                cumulative_df['Response'] = cumulative_responses
                
                # Add iteration labels
                iteration_labels = []
                current_iter = 0
                for i, design_batch in enumerate(reconstructor.design_history[:stage_idx + 2]):
                    batch_label = "Initial" if i == 0 else f"Iter_{i}"
                    iteration_labels.extend([batch_label] * len(design_batch))
                
                cumulative_df['Stage'] = iteration_labels[:len(cumulative_design)]
                st.dataframe(cumulative_df, use_container_width=True)
                
                # 2. RECOVERED PARAMETERS AT THIS STAGE
                st.subheader("🔧 Recovered Parameters")
                
                # For the initial stage, still fit a model if we have enough data
                # For iteration stages, use the data available up to that stage
                if len(cumulative_design) >= n_components and len(cumulative_responses) >= n_components:
                    try:
                        # Build temporary reconstructor with stage data
                        temp_design = np.array(cumulative_design)
                        temp_responses = np.array(cumulative_responses)
                        
                        if reconstructor.config.model_type == "quadratic" and reconstructor._is_mixture_design():
                            X_matrix = reconstructor._build_mixture_quadratic_matrix(temp_design)
                        elif reconstructor.config.model_type == "cubic" and reconstructor._is_mixture_design():
                            # For now, treat cubic mixture model as quadratic (simplified)
                            X_matrix = reconstructor._build_mixture_quadratic_matrix(temp_design)
                        else:
                            X_matrix = temp_design
                        
                        # Fit model
                        from sklearn.linear_model import LinearRegression
                        temp_model = LinearRegression()
                        temp_model.fit(X_matrix, temp_responses)
                        
                        # Get coefficient names using the proper naming function
                        coeff_names = generate_parameter_names(n_components, reconstructor.config.model_type)[:len(temp_model.coef_)]
                        
                        # Calculate comprehensive statistical analysis
                        y_pred = temp_model.predict(X_matrix)
                        n_obs = len(temp_responses)
                        n_params = len(temp_model.coef_)
                        df_residual = n_obs - n_params
                        
                        # Residuals and standard error
                        residuals = temp_responses - y_pred
                        mse = np.sum(residuals**2) / df_residual if df_residual > 0 else np.inf
                        
                        # Standard errors of coefficients
                        try:
                            # Calculate variance-covariance matrix
                            XTX_inv = np.linalg.inv(X_matrix.T @ X_matrix)
                            var_coeff = mse * np.diag(XTX_inv)
                            std_errors = np.sqrt(var_coeff)
                            
                            # t-statistics and p-values
                            from scipy import stats
                            t_stats = temp_model.coef_ / std_errors
                            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))
                            
                            # 95% Confidence intervals
                            t_critical = stats.t.ppf(0.975, df_residual)
                            ci_lower = temp_model.coef_ - t_critical * std_errors
                            ci_upper = temp_model.coef_ + t_critical * std_errors
                            
                        except (np.linalg.LinAlgError, ZeroDivisionError):
                            # Fallback if matrix is singular
                            std_errors = np.full(n_params, np.nan)
                            t_stats = np.full(n_params, np.nan)
                            p_values = np.full(n_params, np.nan)
                            ci_lower = np.full(n_params, np.nan)
                            ci_upper = np.full(n_params, np.nan)
                        
                        # R-squared and adjusted R-squared
                        ss_tot = np.sum((temp_responses - np.mean(temp_responses))**2)
                        ss_res = np.sum(residuals**2)
                        r2_score = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                        adj_r2 = 1 - (ss_res / df_residual) / (ss_tot / (n_obs - 1)) if df_residual > 0 and n_obs > 1 else 0
                        
                        # Overall F-statistic
                        ms_reg = ((ss_tot - ss_res) / (n_params - 1)) if n_params > 1 else 0
                        f_stat = ms_reg / mse if mse > 0 and n_params > 1 else 0
                        f_p_value = 1 - stats.f.cdf(f_stat, n_params - 1, df_residual) if df_residual > 0 and n_params > 1 else 1
                        
                        # Create comprehensive coefficients table
                        stage_coeffs = pd.DataFrame({
                            'Term': coeff_names,
                            'Coefficient': temp_model.coef_,
                            'Std_Error': std_errors,
                            't_Statistic': t_stats,
                            'p_Value': p_values,
                            'CI_Lower_95': ci_lower,
                            'CI_Upper_95': ci_upper,
                            'Significant': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else '' 
                                          for p in p_values]
                        })
                        
                        # Format the table for display
                        display_coeffs = stage_coeffs.copy()
                        for col in ['Coefficient', 'Std_Error', 't_Statistic', 'CI_Lower_95', 'CI_Upper_95']:
                            display_coeffs[col] = display_coeffs[col].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
                        display_coeffs['p_Value'] = display_coeffs['p_Value'].apply(
                            lambda x: f"{x:.4f}" if not np.isnan(x) and x >= 0.001 else "<0.001" if not np.isnan(x) else "N/A"
                        )
                        
                        st.dataframe(display_coeffs, use_container_width=True)
                        
                        # Add significance codes explanation
                        st.caption("**Significance codes:** *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
                        
                        # Model summary statistics
                        st.subheader("📊 Model Summary Statistics")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Goodness of Fit:**")
                            st.write(f"• R²: {r2_score:.4f}")
                            st.write(f"• Adjusted R²: {adj_r2:.4f}")
                            st.write(f"• RMSE: {np.sqrt(mse):.4f}")
                            st.write(f"• Residual Standard Error: {np.sqrt(mse):.4f}")
                        
                        with col2:
                            st.markdown("**Model Statistics:**")
                            st.write(f"• F-statistic: {f_stat:.4f}")
                            if f_p_value < 0.001:
                                st.write(f"• F p-value: <0.001 ***")
                            else:
                                st.write(f"• F p-value: {f_p_value:.4f}")
                            st.write(f"• Degrees of freedom: {n_params-1}, {df_residual}")
                            st.write(f"• Observations: {n_obs}")
                        
                        # ANOVA table
                        st.subheader("📈 Analysis of Variance (ANOVA)")
                        
                        anova_data = {
                            'Source': ['Regression', 'Residual', 'Total'],
                            'DF': [n_params - 1, df_residual, n_obs - 1],
                            'Sum_of_Squares': [ss_tot - ss_res, ss_res, ss_tot],
                            'Mean_Square': [ms_reg, mse, ss_tot / (n_obs - 1) if n_obs > 1 else 0],
                            'F_Value': [f_stat, '', ''],
                            'p_Value': [f"{f_p_value:.4f}" if f_p_value >= 0.001 else "<0.001", '', '']
                        }
                        
                        anova_df = pd.DataFrame(anova_data)
                        # Format numeric columns
                        for col in ['Sum_of_Squares', 'Mean_Square']:
                            anova_df[col] = anova_df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                        anova_df['F_Value'] = anova_df['F_Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x != 0 else x)
                        
                        st.dataframe(anova_df, use_container_width=True)
                        
                        # Partial F-tests for parameter groups
                        if reconstructor.config.model_type == "quadratic" and n_components > 1:
                            st.subheader("🧮 Partial F-tests for Parameter Groups")
                            
                            # Test significance of interaction terms vs linear model
                            try:
                                # Fit linear-only model
                                X_linear = temp_design
                                linear_model = LinearRegression()
                                linear_model.fit(X_linear, temp_responses)
                                linear_pred = linear_model.predict(X_linear)
                                ss_res_linear = np.sum((temp_responses - linear_pred)**2)
                                
                                # Test for interaction effects
                                ss_interaction = ss_res_linear - ss_res
                                n_interactions = (n_components * (n_components - 1)) // 2
                                ms_interaction = ss_interaction / n_interactions if n_interactions > 0 else 0
                                f_interaction = ms_interaction / mse if mse > 0 else 0
                                p_interaction = 1 - stats.f.cdf(f_interaction, n_interactions, df_residual) if df_residual > 0 else 1
                                
                                partial_f_data = {
                                    'Test': ['Interaction Terms vs Linear'],
                                    'Hypothesis': ['H₀: All β_ij = 0'],
                                    'DF_Numerator': [n_interactions],
                                    'DF_Denominator': [df_residual],
                                    'F_Statistic': [f"{f_interaction:.4f}"],
                                    'p_Value': [f"{p_interaction:.4f}" if p_interaction >= 0.001 else "<0.001"],
                                    'Significance': ['***' if p_interaction < 0.001 else '**' if p_interaction < 0.01 else 
                                                   '*' if p_interaction < 0.05 else '.' if p_interaction < 0.1 else 'ns']
                                }
                                
                                partial_f_df = pd.DataFrame(partial_f_data)
                                st.dataframe(partial_f_df, use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"Could not compute partial F-tests: {str(e)}")
                        
                        # Compare with true coefficients if available
                        st.subheader("🎯 True vs Recovered Coefficients")
                        
                        # Get true coefficients from response function using our extraction function
                        if hasattr(st.session_state, 'response_function'):
                            true_coeffs_list = extract_true_coefficients(
                                st.session_state.response_function,
                                n_components,
                                reconstructor.config.model_type
                            )
                            
                            comparison_data = []
                            for i, row in stage_coeffs.iterrows():
                                term = row['Term']
                                recovered = row['Coefficient']
                                # Get true coefficient by index if available
                                true_val = true_coeffs_list[i] if i < len(true_coeffs_list) else 0
                                error = abs(recovered - true_val)
                                error_pct = (error / abs(true_val) * 100) if abs(true_val) > 1e-6 else error * 100
                                
                                comparison_data.append({
                                    'Parameter': term,
                                    'True_Coefficient': true_val,
                                    'Recovered_Coefficient': recovered,
                                    'Absolute_Error': error,
                                    'Relative_Error_%': f"{error_pct:.1f}%" if error_pct != float('inf') else "N/A"
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            st.dataframe(comparison_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Could not compute coefficients for this stage: {str(e)}")
                        st.info("This can happen for early stages with insufficient data.")
                        
                else:
                    st.warning(f"Insufficient data for model fitting at this stage: {len(cumulative_design)} points, need at least {n_components}")
                
                # 3. RESPONSE PREDICTION ERRORS
                st.subheader("📉 Response vs Prediction Analysis")
                
                if stage_idx >= 0:
                    try:
                        # Get predictions from stage model
                        stage_result = reconstructor.iteration_history[stage_idx]
                        
                        # Calculate predictions vs actual
                        actual_responses = np.array(cumulative_responses)
                        if reconstructor.config.model_type == "quadratic" and reconstructor._is_mixture_design():
                            X_matrix = reconstructor._build_mixture_quadratic_matrix(np.array(cumulative_design))
                        else:
                            X_matrix = np.array(cumulative_design)
                        
                        predicted_responses = temp_model.predict(X_matrix)
                        errors = actual_responses - predicted_responses
                        
                        # Create error analysis DataFrame
                        error_df = pd.DataFrame({
                            'Point_Index': range(len(cumulative_design)),
                            'Actual_Response': actual_responses,
                            'Predicted_Response': predicted_responses,
                            'Error': errors,
                            'Absolute_Error': np.abs(errors),
                            'Squared_Error': errors**2
                        })
                        
                        st.dataframe(error_df, use_container_width=True)
                        
                        # Error statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean Error", f"{np.mean(errors):.4f}")
                        with col2:
                            st.metric("RMSE", f"{np.sqrt(np.mean(errors**2)):.4f}")
                        with col3:
                            st.metric("Max Error", f"{np.max(np.abs(errors)):.4f}")
                        with col4:
                            st.metric("R²", f"{stage_result.r_squared:.4f}")
                        
                        # Plot actual vs predicted
                        fig = go.Figure()
                        
                        # Add points
                        fig.add_trace(go.Scatter(
                            x=actual_responses,
                            y=predicted_responses,
                            mode='markers',
                            name='Predictions',
                            marker=dict(size=8, color='blue'),
                            text=[f"Point {i+1}<br>Actual: {actual:.3f}<br>Predicted: {pred:.3f}<br>Error: {err:.3f}"
                                  for i, (actual, pred, err) in enumerate(zip(actual_responses, predicted_responses, errors))],
                            hovertemplate='<b>%{text}</b><extra></extra>'
                        ))
                        
                        # Add perfect prediction line
                        min_val = min(actual_responses.min(), predicted_responses.min())
                        max_val = max(actual_responses.max(), predicted_responses.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red')
                        ))
                        
                        fig.update_layout(
                            title=f"Actual vs Predicted Responses - {selected_stage}",
                            xaxis_title="Actual Response",
                            yaxis_title="Predicted Response",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Could not generate prediction analysis: {str(e)}")
                else:
                    st.info("No predictions available for initial stage")
                    
            else:
                st.warning("No experiment data available for analysis")
        else:
            st.info("Run experiment iterations to view stage-by-stage analysis")
    
    with tab5:
        st.header("📋 Results Summary")
        
        if st.session_state.reconstructor is not None:
            reconstructor = st.session_state.reconstructor
            
            # Configuration summary
            st.subheader("⚙️ Experiment Configuration")
            total_params = calculate_parameters(config.n_components, config.model_type)
            config_df = pd.DataFrame([
                ["Components", str(config.n_components)],
                ["Model Type", str(config.model_type.capitalize())],
                ["Model Parameters", str(total_params)],
                ["Max Iterations", str(config.max_iterations)],
                ["R² Threshold", str(config.r2_threshold)],
                ["D-efficiency Threshold", str(config.d_efficiency_threshold)],
                ["Initial Batch", str(config.initial_batch_size)],
                ["Sequential Batch", str(config.sequential_batch_size)]
            ], columns=["Parameter", "Value"])
            st.table(config_df)
            
            # Results summary
            if reconstructor.iteration_history:
                st.subheader("📊 Final Results")
                
                latest = reconstructor.iteration_history[-1]
                results_df = pd.DataFrame([
                    ["Status", "Converged" if reconstructor.converged else "Max iterations reached"],
                    ["Total Iterations", str(len(reconstructor.iteration_history))],
                    ["Total Experiments", str(len(reconstructor.current_design))],
                    ["Final R²", f"{latest.r_squared:.4f}"],
                    ["Final D-efficiency", f"{latest.d_efficiency:.4f}"],
                    ["Final RMSE", f"{latest.prediction_error:.4f}"],
                    ["Significant Terms", str(len(latest.significant_terms))]
                ], columns=["Metric", "Value"])
                st.table(results_df)
                
                # Save results
                if st.button("💾 Save Complete Results"):
                    filename = reconstructor.save_results()
                    st.success(f"Results saved to {filename}")
                    
                    # Also save design points as CSV
                    design_df = pd.DataFrame(
                        reconstructor.current_design,
                        columns=[f'x{i+1}' for i in range(config.n_components)]
                    )
                    design_df['Response'] = reconstructor.current_responses
                    design_csv = design_df.to_csv(index=False)
                    
                    st.download_button(
                        "📥 Download Design Matrix",
                        data=design_csv,
                        file_name="final_design_matrix.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Initialize experiment to view results")

if __name__ == "__main__":
    main()
