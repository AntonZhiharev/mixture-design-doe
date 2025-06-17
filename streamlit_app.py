"""
User-Friendly Web Interface for Optimal Design of Experiments
Includes regular DOE, mixture designs, and multiple response analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import base64

# Import our DOE classes
try:
    from optimal_doe_python import OptimalDOE, multiple_response_analysis
    from mixture_designs import MixtureDesign, mixture_response_analysis
except ImportError:
    st.error("Please ensure optimal_doe_python.py and mixture_designs.py are in the same directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Optimal DOE Generator", 
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
    .stExpander {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🧪 Optimal Design of Experiments</h1>', unsafe_allow_html=True)
st.markdown("**Generate D-optimal, I-optimal, and mixture designs with interactive visualization**")

# Sidebar for navigation
st.sidebar.title("Navigation")
design_type = st.sidebar.selectbox(
    "Choose Design Type",
    ["Regular DOE", "Mixture Design", "Compare Designs", "About"]
)

if design_type == "Regular DOE":
    st.markdown('<h2 class="sub-header">📊 Regular Optimal Design</h2>', unsafe_allow_html=True)
    
    # Add warning about mixture experiments
    st.warning("⚠️ **Important**: This section is for **independent factors** (temperature, pressure, time, etc.). "
               "If you're working with **mixture components** (ingredients, formulations) that must sum to 100%, "
               "please use the **Mixture Design** tab instead!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Design Parameters")
        
        # Add experiment type selection
        experiment_type = st.radio(
            "Experiment Type",
            ["Independent Factors", "Mixture Components"],
            help="Independent factors can vary independently. Mixture components must sum to 100%."
        )
        
        if experiment_type == "Mixture Components":
            st.error("🚫 For mixture experiments, please use the **Mixture Design** tab. "
                    "Regular DOE cannot handle mixture constraints properly.")
            st.stop()
        
        # Number of factors
        n_factors = st.number_input("Number of Factors", min_value=1, max_value=10, value=3)
        
        # Factor ranges
        st.write("**Factor Ranges:**")
        factor_ranges = []
        factor_names = []
        
        for i in range(n_factors):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                name = st.text_input(f"Factor {i+1} Name", value=f"Factor_{i+1}", key=f"name_{i}")
                factor_names.append(name)
            with col_b:
                min_val = st.number_input(f"Min", value=-1.0, key=f"min_{i}")
            with col_c:
                max_val = st.number_input(f"Max", value=1.0, key=f"max_{i}")
            factor_ranges.append((min_val, max_val))
        
        # Add examples for clarity
        with st.expander("💡 Examples of Independent Factors"):
            st.markdown("""
            **Process Variables:**
            - Temperature: 80°C to 120°C
            - Pressure: 1 to 5 bar
            - pH: 3 to 9
            - Time: 30 to 120 minutes
            
            **Material Properties:**
            - Thickness: 1mm to 5mm
            - Speed: 100 to 500 rpm
            - Concentration: 0.1M to 1.0M
            
            These factors can be set **independently** of each other.
            """)
        
        # Design settings
        n_runs = st.number_input("Number of Experimental Runs", min_value=5, max_value=100, value=15)
        model_order = st.selectbox("Model Order", [1, 2], index=1, format_func=lambda x: f"{'Linear' if x==1 else 'Quadratic'}")
        criterion = st.selectbox("Optimality Criterion", ["D-optimal", "I-optimal"])
        random_seed = st.number_input("Random Seed (for reproducibility)", value=42)
        
        generate_button = st.button("🚀 Generate Design", type="primary")
    
    with col2:
        if generate_button:
            with st.spinner("Generating optimal design..."):
                try:
                    # Create DOE generator
                    doe = OptimalDOE(n_factors, factor_ranges)
                    
                    # Generate design based on criterion
                    if criterion == "D-optimal":
                        design = doe.generate_d_optimal(n_runs, model_order, random_seed=random_seed)
                    else:
                        design = doe.generate_i_optimal(n_runs, model_order, random_seed=random_seed)
                    
                    # Evaluate design
                    results = doe.evaluate_design(design, model_order)
                    
                    # Store in session state
                    st.session_state.regular_design = design
                    st.session_state.regular_results = results
                    st.session_state.regular_factor_names = factor_names
                    
                    st.success("✅ Design generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating design: {str(e)}")
        
        # Display results if available
        if 'regular_design' in st.session_state:
            design = st.session_state.regular_design
            results = st.session_state.regular_results
            factor_names = st.session_state.regular_factor_names
            
            st.subheader("Design Metrics")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("D-Efficiency", f"{results['d_efficiency']:.4f}")
            with col_b:
                st.metric("I-Efficiency", f"{results['i_efficiency']:.4f}")
            with col_c:
                st.metric("Runs", results['n_runs'])
            
            st.subheader("Design Matrix")
            design_df = pd.DataFrame(design, columns=factor_names)
            design_df.index = [f"Run {i+1}" for i in range(len(design_df))]
            st.dataframe(design_df.round(3))
            
            # Download button
            csv = design_df.to_csv()
            st.download_button(
                label="📥 Download Design Matrix",
                data=csv,
                file_name="optimal_design.csv",
                mime="text/csv"
            )
            
            # Visualization for 2D designs
            if n_factors == 2:
                st.subheader("Design Visualization")
                fig = px.scatter(
                    design_df, 
                    x=factor_names[0], 
                    y=factor_names[1],
                    title=f"{criterion} Design ({n_runs} runs)",
                    text=design_df.index
                )
                fig.update_traces(textposition="top center", marker_size=10)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Response simulation
            st.subheader("Response Simulation")
            with st.expander("Simulate Responses"):
                n_responses = st.number_input("Number of Responses", min_value=1, max_value=5, value=2)
                
                if st.button("Simulate Responses"):
                    np.random.seed(random_seed)
                    responses_df = design_df.copy()
                    
                    for r in range(n_responses):
                        # Generate random coefficients for response
                        coeffs = np.random.uniform(-5, 5, n_factors)
                        response = np.sum(design * coeffs, axis=1) + np.random.normal(0, 1, len(design))
                        responses_df[f'Response_{r+1}'] = response
                    
                    st.dataframe(responses_df.round(3))
                    
                    # Response distribution plots
                    fig = make_subplots(rows=1, cols=n_responses, 
                                      subplot_titles=[f'Response {i+1}' for i in range(n_responses)])
                    
                    for r in range(n_responses):
                        fig.add_trace(
                            go.Histogram(x=responses_df[f'Response_{r+1}'], name=f'Response {r+1}'),
                            row=1, col=r+1
                        )
                    
                    fig.update_layout(height=400, title_text="Response Distributions")
                    st.plotly_chart(fig, use_container_width=True)

elif design_type == "Mixture Design":
    st.markdown('<h2 class="sub-header">🧬 Mixture Design</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Mixture Parameters")
        
        # Number of components
        n_components = st.number_input("Number of Components", min_value=2, max_value=8, value=3)
        
        # Component details
        st.write("**Component Details:**")
        component_names = []
        component_bounds = []
        
        for i in range(n_components):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                name = st.text_input(f"Component {i+1} Name", value=f"Component_{i+1}", key=f"comp_name_{i}")
                component_names.append(name)
            with col_b:
                min_val = st.number_input(f"Min %", value=0.0, min_value=0.0, max_value=1.0, step=0.01, key=f"comp_min_{i}")
            with col_c:
                max_val = st.number_input(f"Max %", value=1.0, min_value=0.0, max_value=1.0, step=0.01, key=f"comp_max_{i}")
            component_bounds.append((min_val, max_val))
        
        # Validate bounds
        sum_min = sum(bound[0] for bound in component_bounds)
        sum_max = sum(bound[1] for bound in component_bounds)
        
        if sum_min > 1:
            st.error("⚠️ Sum of minimum bounds exceeds 100% - adjust bounds!")
        elif sum_max < 1:
            st.error("⚠️ Sum of maximum bounds less than 100% - adjust bounds!")
        else:
            st.success("✅ Bounds are feasible")
        
        # Design settings
        design_method = st.selectbox(
            "Design Method", 
            ["D-optimal", "I-optimal", "Simplex Lattice", "Simplex Centroid", "Extreme Vertices"]
        )
        
        if design_method in ["D-optimal", "I-optimal"]:
            n_runs = st.number_input("Number of Runs", min_value=5, max_value=50, value=12)
            model_type = st.selectbox("Model Type", ["linear", "quadratic", "cubic"], index=1)
        elif design_method == "Simplex Lattice":
            lattice_degree = st.number_input("Lattice Degree", min_value=2, max_value=5, value=3)
        
        random_seed = st.number_input("Random Seed", value=42, key="mixture_seed")
        
        generate_mixture_button = st.button("🧬 Generate Mixture Design", type="primary")
    
    with col2:
        if generate_mixture_button and sum_min <= 1 and sum_max >= 1:
            with st.spinner("Generating mixture design..."):
                try:
                    # Create mixture design generator
                    mixture = MixtureDesign(n_components, component_names, component_bounds)
                    
                    # Generate design based on method
                    if design_method == "D-optimal":
                        design = mixture.generate_d_optimal_mixture(n_runs, model_type, random_seed=random_seed)
                        results = mixture.evaluate_mixture_design(design, model_type)
                    elif design_method == "I-optimal":
                        design = mixture.generate_i_optimal_mixture(n_runs, model_type, random_seed=random_seed)
                        results = mixture.evaluate_mixture_design(design, model_type)
                    elif design_method == "Simplex Lattice":
                        design = mixture.generate_simplex_lattice(lattice_degree)
                        results = mixture.evaluate_mixture_design(design, "quadratic")
                    elif design_method == "Simplex Centroid":
                        design = mixture.generate_simplex_centroid()
                        results = mixture.evaluate_mixture_design(design, "quadratic")
                    elif design_method == "Extreme Vertices":
                        design = mixture.generate_extreme_vertices()
                        results = mixture.evaluate_mixture_design(design, "quadratic")
                    
                    # Store in session state
                    st.session_state.mixture_design = design
                    st.session_state.mixture_results = results
                    st.session_state.mixture_component_names = component_names
                    st.session_state.mixture_generator = mixture
                    
                    st.success(f"✅ {design_method} mixture design generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating mixture design: {str(e)}")
        
        # Display results if available
        if 'mixture_design' in st.session_state:
            design = st.session_state.mixture_design
            results = st.session_state.mixture_results
            component_names = st.session_state.mixture_component_names
            
            st.subheader("Design Metrics")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("D-Efficiency", f"{results['d_efficiency']:.4f}")
            with col_b:
                st.metric("I-Efficiency", f"{results['i_efficiency']:.4f}")
            with col_c:
                st.metric("Runs", results['n_runs'])
            
            st.subheader("Mixture Design Matrix")
            mixture_df = pd.DataFrame(design, columns=component_names)
            mixture_df.index = [f"Run {i+1}" for i in range(len(mixture_df))]
            
            # Add percentage columns
            for col in component_names:
                mixture_df[f"{col} (%)"] = (mixture_df[col] * 100).round(1)
            
            st.dataframe(mixture_df.round(4))
            
            # Verify sum to 1
            sums = np.sum(design, axis=1)
            if np.allclose(sums, 1.0):
                st.success("✅ All mixtures sum to 100%")
            else:
                st.warning("⚠️ Some mixtures don't sum exactly to 100% (rounding)")
            
            # Download button
            csv = mixture_df.to_csv()
            st.download_button(
                label="📥 Download Mixture Design",
                data=csv,
                file_name="mixture_design.csv",
                mime="text/csv"
            )
            
            # Ternary plot for 3 components
            if n_components == 3:
                st.subheader("Ternary Plot")
                
                # Create ternary plot using plotly
                fig = go.Figure()
                
                # Add scatter points
                fig.add_trace(go.Scatterternary({
                    'mode': 'markers+text',
                    'a': design[:, 0],
                    'b': design[:, 1], 
                    'c': design[:, 2],
                    'text': [f"R{i+1}" for i in range(len(design))],
                    'textposition': "middle center",
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
                    'title': f"{design_method} Mixture Design"
                })
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Response simulation for mixtures
            st.subheader("Mixture Response Simulation")
            with st.expander("Simulate Mixture Responses"):
                response_type = st.selectbox(
                    "Response Model Type",
                    ["Linear Blending", "Synergistic", "Antagonistic", "Custom"]
                )
                
                if st.button("Simulate Mixture Responses"):
                    np.random.seed(random_seed)
                    sim_df = mixture_df.copy()
                    
                    if response_type == "Linear Blending":
                        # Simple linear blending
                        coeffs = np.random.uniform(50, 100, n_components)
                        response = np.sum(design * coeffs, axis=1) + np.random.normal(0, 2, len(design))
                    elif response_type == "Synergistic":
                        # Include positive interactions
                        coeffs = np.random.uniform(50, 80, n_components)
                        response = np.sum(design * coeffs, axis=1)
                        # Add synergistic interactions
                        for i in range(n_components):
                            for j in range(i+1, n_components):
                                response += 20 * design[:, i] * design[:, j]
                        response += np.random.normal(0, 2, len(design))
                    elif response_type == "Antagonistic":
                        # Include negative interactions
                        coeffs = np.random.uniform(70, 100, n_components)
                        response = np.sum(design * coeffs, axis=1)
                        # Add antagonistic interactions
                        for i in range(n_components):
                            for j in range(i+1, n_components):
                                response -= 15 * design[:, i] * design[:, j]
                        response += np.random.normal(0, 2, len(design))
                    
                    sim_df['Response'] = response.round(2)
                    
                    st.dataframe(sim_df)
                    
                    # Response surface plot for 3 components
                    if n_components == 3:
                        fig = go.Figure(go.Scatterternary({
                            'mode': 'markers',
                            'a': design[:, 0],
                            'b': design[:, 1], 
                            'c': design[:, 2],
                            'marker': {
                                'symbol': 'circle',
                                'size': 10,
                                'color': response,
                                'colorscale': 'Viridis',
                                'showscale': True,
                                'colorbar': {'title': 'Response'}
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
                            'title': f"Response Surface - {response_type}"
                        })
                        
                        st.plotly_chart(fig, use_container_width=True)

elif design_type == "Compare Designs":
    st.markdown('<h2 class="sub-header">⚖️ Design Comparison</h2>', unsafe_allow_html=True)
    
    # Check if designs are available
    has_regular = 'regular_design' in st.session_state
    has_mixture = 'mixture_design' in st.session_state
    
    if not has_regular and not has_mixture:
        st.info("Please generate at least one design first using the Regular DOE or Mixture Design tabs.")
    else:
        col1, col2 = st.columns(2)
        
        if has_regular:
            with col1:
                st.subheader("Regular DOE Design")
                regular_results = st.session_state.regular_results
                st.metric("Runs", regular_results['n_runs'])
                st.metric("Factors", regular_results['n_factors'])
                st.metric("D-Efficiency", f"{regular_results['d_efficiency']:.4f}")
                st.metric("I-Efficiency", f"{regular_results['i_efficiency']:.4f}")
        
        if has_mixture:
            with col2:
                st.subheader("Mixture Design")
                mixture_results = st.session_state.mixture_results
                st.metric("Runs", mixture_results['n_runs'])
                st.metric("Components", mixture_results['n_components'])
                st.metric("D-Efficiency", f"{mixture_results['d_efficiency']:.4f}")
                st.metric("I-Efficiency", f"{mixture_results['i_efficiency']:.4f}")
        
        # Design efficiency comparison
        if has_regular and has_mixture:
            st.subheader("Efficiency Comparison")
            
            comparison_data = {
                'Design Type': ['Regular DOE', 'Mixture Design'],
                'D-Efficiency': [regular_results['d_efficiency'], mixture_results['d_efficiency']],
                'I-Efficiency': [regular_results['i_efficiency'], mixture_results['i_efficiency']],
                'Runs': [regular_results['n_runs'], mixture_results['n_runs']]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # Efficiency plots
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=['D-Efficiency', 'I-Efficiency'])
            
            fig.add_trace(
                go.Bar(x=comparison_df['Design Type'], y=comparison_df['D-Efficiency'], 
                      name='D-Efficiency', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=comparison_df['Design Type'], y=comparison_df['I-Efficiency'], 
                      name='I-Efficiency', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, title_text="Design Efficiency Comparison")
            st.plotly_chart(fig, use_container_width=True)

elif design_type == "About":
    st.markdown('<h2 class="sub-header">ℹ️ About Optimal DOE</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## What is Design of Experiments (DOE)?
    
    Design of Experiments is a systematic approach to planning experiments to maximize information while minimizing cost and time.
    
    ### Regular DOE
    - **D-optimal**: Maximizes the determinant of the information matrix
        - Best for **parameter estimation accuracy**
        - Minimizes confidence intervals of model coefficients
        - Ideal when your goal is to fit a precise model
    
    - **I-optimal**: Minimizes average prediction variance
        - Best for **prediction accuracy** across the design space
        - Optimizes response surface mapping
        - Ideal when your goal is to predict responses
    
    ### Mixture Designs
    Special designs where components must sum to 100%:
    - **Simplex Lattice**: Systematic coverage of mixture space
    - **Simplex Centroid**: Focus on centroids of mixture subspaces
    - **Extreme Vertices**: Based on constraint boundaries
    - **D-optimal Mixture**: Optimized for parameter estimation
    - **I-optimal Mixture**: Optimized for prediction
    
    ### Key Benefits
    ✅ **Efficiency**: Fewer experiments needed than traditional methods  
    ✅ **Flexibility**: Handle irregular experimental regions and constraints  
    ✅ **Optimization**: Tailored for specific objectives  
    ✅ **Quality**: Better statistical properties  
    
    ### When to Use Each Design
    - **D-optimal**: When you need precise model parameters
    - **I-optimal**: When you need good predictions everywhere
    - **Mixture designs**: When dealing with formulations (recipes, blends, etc.)
    
    ### Example Applications
    - **Chemical processes**: Temperature, pressure, concentration optimization
    - **Material science**: Composite formulations, alloy development
    - **Pharmaceutical**: Drug formulation, process optimization
    - **Manufacturing**: Quality improvement, process robustness
    
    ---
    
    **Created with ❤️ using Streamlit and advanced DOE algorithms**
    """)
    
    with st.expander("Technical Details"):
        st.markdown("""
        ### Algorithms Used
        
        **Coordinate Exchange Algorithm**: 
        - Iterative optimization of design points
        - Maximizes design efficiency metrics
        - Handles complex constraints and bounds
        
        **Efficiency Metrics**:
        - D-efficiency: det(X'X)^(1/p) / n
        - I-efficiency: Average prediction variance
        - Condition number: Matrix stability measure
        
        **Model Types**:
        - Linear: β₀ + Σβᵢxᵢ
        - Quadratic: Linear + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ
        - Mixture: Scheffé polynomials (no intercept)
        """)

# Footer
st.markdown("---")
st.markdown("**Optimal DOE Generator** | Advanced experimental design made simple")
