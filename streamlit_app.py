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
    from sequential_doe import SequentialDOE, create_sequential_plan
    from sequential_mixture_doe import SequentialMixtureDOE
except ImportError:
    st.error("Please ensure all required files are in the same directory")
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
    ["Regular DOE", "Mixture Design", "Sequential DOE", "Sequential Mixture DOE", "Compare Designs", "About"]
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
        
        # Add replicates option
        st.write("**Replication Settings:**")
        n_replicates = st.number_input(
            "Number of Replicates per Run", 
            min_value=1, max_value=10, value=1,
            help="Each unique experimental condition will be repeated this many times"
        )
        
        # Show total experiments
        total_experiments = n_runs * n_replicates
        if n_replicates > 1:
            st.info(f"Total experiments: {n_runs} runs × {n_replicates} replicates = **{total_experiments}** experiments")
        else:
            st.info(f"Total experiments: **{total_experiments}** (no replicates)")
        
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
                    
                    # Generate replicated design if requested
                    if n_replicates > 1:
                        replicated_design = []
                        for run in design:
                            for _ in range(n_replicates):
                                replicated_design.append(run)
                        replicated_design = np.array(replicated_design)
                        
                        # Evaluate replicated design
                        replicated_results = doe.evaluate_design(replicated_design, model_order)
                    else:
                        replicated_design = design
                        replicated_results = results
                    
                    # Store in session state
                    st.session_state.regular_design = design
                    st.session_state.regular_results = results
                    st.session_state.regular_factor_names = factor_names
                    st.session_state.n_replicates = n_replicates
                    st.session_state.replicated_design = replicated_design
                    st.session_state.replicated_results = replicated_results
                    
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
            
            # Show replication impact if replicates were requested
            if 'n_replicates' in st.session_state and st.session_state.n_replicates > 1:
                n_replicates = st.session_state.n_replicates
                replicated_results = st.session_state.replicated_results
                
                st.subheader("📊 Impact of Replicates on Design Efficiency")
                
                # Create comparison
                comparison_data = {
                    'Design': ['Base Design', f'With {n_replicates} Replicates'],
                    'Total Experiments': [results['n_runs'], replicated_results['n_runs']],
                    'D-Efficiency': [results['d_efficiency'], replicated_results['d_efficiency']],
                    'I-Efficiency': [results['i_efficiency'], replicated_results['i_efficiency']]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.dataframe(comparison_df.round(4))
                
                # Efficiency improvement metrics
                d_improvement = replicated_results['d_efficiency'] / results['d_efficiency']
                i_improvement = replicated_results['i_efficiency'] / results['i_efficiency']
                
                col_d, col_i = st.columns(2)
                with col_d:
                    if d_improvement > 1:
                        st.success(f"D-Efficiency improved by {((d_improvement-1)*100):.1f}%")
                    else:
                        st.info(f"D-Efficiency ratio: {d_improvement:.3f}")
                
                with col_i:
                    if i_improvement > 1:
                        st.success(f"I-Efficiency improved by {((i_improvement-1)*100):.1f}%")
                    else:
                        st.info(f"I-Efficiency ratio: {i_improvement:.3f}")
                
                # Explanation
                with st.expander("💡 Understanding Replication Effects"):
                    st.markdown(f"""
                    **Key Insights:**
                    
                    🔍 **Base Design:** {results['n_runs']} unique experimental conditions
                    🔄 **With Replicates:** {replicated_results['n_runs']} total experiments ({n_replicates} × {results['n_runs']})
                    
                    **Why Replicates Matter:**
                    - **Error Estimation**: Replicates allow you to estimate pure experimental error
                    - **Improved Precision**: More data points → better parameter estimates
                    - **Statistical Power**: Higher confidence in your results
                    - **Lack-of-Fit Testing**: Can detect if your model is adequate
                    
                    **Trade-offs:**
                    - ✅ **Pros**: Better precision, error estimation, validation capability
                    - ⚠️ **Cons**: More experiments = higher cost and time
                    
                    **Recommendation**: 
                    - Use 2-3 replicates for critical experiments
                    - Use 1 replicate for screening studies
                    - Consider center point replicates for pure error estimation
                    """)
                
                # Visualization of efficiency comparison
                fig = make_subplots(rows=1, cols=2, 
                                  subplot_titles=['D-Efficiency', 'I-Efficiency'])
                
                fig.add_trace(
                    go.Bar(x=comparison_df['Design'], y=comparison_df['D-Efficiency'], 
                          name='D-Efficiency', marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=comparison_df['Design'], y=comparison_df['I-Efficiency'], 
                          name='I-Efficiency', marker_color='lightcoral'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, title_text="Efficiency Impact of Replicates", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
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
        
        # Add replicates option for mixture designs
        st.write("**Replication Settings:**")
        n_mixture_replicates = st.number_input(
            "Number of Replicates per Mixture", 
            min_value=1, max_value=10, value=1,
            help="Each unique mixture composition will be repeated this many times",
            key="mixture_replicates"
        )
        
        # Show total experiments for mixtures
        if design_method in ["D-optimal", "I-optimal"]:
            total_mixture_experiments = n_runs * n_mixture_replicates
            if n_mixture_replicates > 1:
                st.info(f"Total experiments: {n_runs} mixtures × {n_mixture_replicates} replicates = **{total_mixture_experiments}** experiments")
            else:
                st.info(f"Total experiments: **{total_mixture_experiments}** (no replicates)")
        
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
                    
                    # Generate replicated mixture design if requested
                    if n_mixture_replicates > 1:
                        replicated_mixture_design = []
                        for mixture_run in design:
                            for _ in range(n_mixture_replicates):
                                replicated_mixture_design.append(mixture_run)
                        replicated_mixture_design = np.array(replicated_mixture_design)
                        
                        # Evaluate replicated mixture design
                        replicated_mixture_results = mixture.evaluate_mixture_design(replicated_mixture_design, model_type if design_method in ["D-optimal", "I-optimal"] else "quadratic")
                    else:
                        replicated_mixture_design = design
                        replicated_mixture_results = results
                    
                    # Store in session state
                    st.session_state.mixture_design = design
                    st.session_state.mixture_results = results
                    st.session_state.mixture_component_names = component_names
                    st.session_state.mixture_generator = mixture
                    st.session_state.n_mixture_replicates = n_mixture_replicates
                    st.session_state.replicated_mixture_design = replicated_mixture_design
                    st.session_state.replicated_mixture_results = replicated_mixture_results
                    
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
            
            # Show replication impact for mixture designs if replicates were requested
            if 'n_mixture_replicates' in st.session_state and st.session_state.n_mixture_replicates > 1:
                n_mixture_replicates = st.session_state.n_mixture_replicates
                replicated_mixture_results = st.session_state.replicated_mixture_results
                
                st.subheader("📊 Impact of Replicates on Mixture Design Efficiency")
                
                # Create comparison
                mixture_comparison_data = {
                    'Design': ['Base Mixture Design', f'With {n_mixture_replicates} Replicates'],
                    'Total Experiments': [results['n_runs'], replicated_mixture_results['n_runs']],
                    'D-Efficiency': [results['d_efficiency'], replicated_mixture_results['d_efficiency']],
                    'I-Efficiency': [results['i_efficiency'], replicated_mixture_results['i_efficiency']]
                }
                mixture_comparison_df = pd.DataFrame(mixture_comparison_data)
                
                # Display comparison table
                st.dataframe(mixture_comparison_df.round(4))
                
                # Efficiency improvement metrics
                mixture_d_improvement = replicated_mixture_results['d_efficiency'] / results['d_efficiency']
                mixture_i_improvement = replicated_mixture_results['i_efficiency'] / results['i_efficiency']
                
                col_d, col_i = st.columns(2)
                with col_d:
                    if mixture_d_improvement > 1:
                        st.success(f"D-Efficiency improved by {((mixture_d_improvement-1)*100):.1f}%")
                    else:
                        st.info(f"D-Efficiency ratio: {mixture_d_improvement:.3f}")
                
                with col_i:
                    if mixture_i_improvement > 1:
                        st.success(f"I-Efficiency improved by {((mixture_i_improvement-1)*100):.1f}%")
                    else:
                        st.info(f"I-Efficiency ratio: {mixture_i_improvement:.3f}")
                
                # Explanation for mixture replicates
                with st.expander("💡 Understanding Mixture Replication Effects"):
                    st.markdown(f"""
                    **Key Insights for Mixture Experiments:**
                    
                    🔍 **Base Design:** {results['n_runs']} unique mixture compositions
                    🔄 **With Replicates:** {replicated_mixture_results['n_runs']} total experiments ({n_mixture_replicates} × {results['n_runs']})
                    
                    **Why Replicates Matter in Mixture Designs:**
                    - **Blend Variability**: Account for mixing and preparation errors
                    - **Component Interactions**: Better estimate synergistic/antagonistic effects
                    - **Process Validation**: Ensure mixture formulations are reproducible
                    - **Pure Error Estimation**: Separate measurement error from model inadequacy
                    
                    **Mixture-Specific Benefits:**
                    - ✅ **Formulation Robustness**: Test consistency of blend properties
                    - ✅ **Interaction Detection**: More power to detect component interactions
                    - ✅ **Scaling Validation**: Verify mixture behavior at different scales
                    
                    **Recommendation for Mixtures**: 
                    - Use 2-3 replicates for formulation development
                    - Use 3-5 replicates for critical blend validation
                    - Consider process replicates (different batches) vs. analytical replicates
                    """)
                
                # Visualization of mixture efficiency comparison
                mixture_fig = make_subplots(rows=1, cols=2, 
                                          subplot_titles=['D-Efficiency', 'I-Efficiency'])
                
                mixture_fig.add_trace(
                    go.Bar(x=mixture_comparison_df['Design'], y=mixture_comparison_df['D-Efficiency'], 
                          name='D-Efficiency', marker_color='lightgreen'),
                    row=1, col=1
                )
                
                mixture_fig.add_trace(
                    go.Bar(x=mixture_comparison_df['Design'], y=mixture_comparison_df['I-Efficiency'], 
                          name='I-Efficiency', marker_color='lightsalmon'),
                    row=1, col=2
                )
                
                mixture_fig.update_layout(height=400, title_text="Mixture Design Efficiency Impact of Replicates", showlegend=False)
                st.plotly_chart(mixture_fig, use_container_width=True)
            
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

elif design_type == "Sequential DOE":
    st.markdown('<h2 class="sub-header">🔄 Sequential Design of Experiments</h2>', unsafe_allow_html=True)
    
    st.info("**Sequential DOE** allows you to start with a screening design (Stage 1), analyze results, and then add targeted experiments (Stage 2) based on what you learned.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sequential Design Setup")
        
        # Number of factors
        n_factors = st.number_input("Number of Factors", min_value=2, max_value=10, value=6, key="seq_n_factors")
        
        # Factor setup
        st.write("**Factor Details:**")
        factor_names = []
        factor_ranges = []
        
        for i in range(n_factors):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                name = st.text_input(f"Factor {i+1} Name", value=f"Factor_{i+1}", key=f"seq_name_{i}")
                factor_names.append(name)
            with col_b:
                min_val = st.number_input(f"Min", value=-1.0, key=f"seq_min_{i}")
            with col_c:
                max_val = st.number_input(f"Max", value=1.0, key=f"seq_max_{i}")
            factor_ranges.append((min_val, max_val))
        
        st.write("**Stage Configuration:**")
        
        # Get recommendations based on rules of thumb
        temp_seq_doe = SequentialDOE(n_factors, factor_ranges)
        seq_recommendations = temp_seq_doe.get_sequential_recommendations()
        
        # Quality level selection
        quality_level = st.selectbox(
            "Experiment Quality Level",
            ["minimum", "recommended", "excellent"],
            index=1,  # Default to recommended
            help="Minimum: 1x parameters | Recommended: 1.5x parameters | Excellent: 2x parameters"
        )
        
        # Stage 1 settings
        with st.expander("Stage 1: Screening", expanded=True):
            st.write(f"**{seq_recommendations['stage1']['purpose']}**")
            
            col_min, col_rec, col_exc = st.columns(3)
            with col_min:
                st.metric("Minimum", seq_recommendations['stage1']['minimum'])
            with col_rec:
                st.metric("Recommended", seq_recommendations['stage1']['recommended'])
            with col_exc:
                st.metric("Excellent", seq_recommendations['stage1']['excellent'])
            
            # Set default based on quality level
            stage1_default = {
                'minimum': seq_recommendations['stage1']['minimum'],
                'recommended': seq_recommendations['stage1']['recommended'],
                'excellent': seq_recommendations['stage1']['excellent']
            }[quality_level]
            
            stage1_runs = st.number_input(
                "Stage 1 Runs", 
                min_value=seq_recommendations['stage1']['minimum'], 
                max_value=50, 
                value=stage1_default,
                help=seq_recommendations['stage1']['can_fit']
            )
            
            # Show what can be fitted
            st.info(f"✓ Can fit: {seq_recommendations['stage1']['can_fit']}")
            
            # Show expected efficiency
            stage1_rec_details = temp_seq_doe.get_recommended_runs(1, quality_level)
            st.success(f"Expected D-efficiency: {stage1_rec_details['efficiency_expected']}")
        
        # Stage 2 settings
        with st.expander("Stage 2: Optimization", expanded=True):
            st.write(f"**{seq_recommendations['stage2']['purpose']}**")
            
            # Calculate recommended additional runs
            quad_params = temp_seq_doe._count_parameters(2)
            total_recommended = int(np.ceil(quad_params * {'minimum': 1.0, 'recommended': 1.5, 'excellent': 2.0}[quality_level]))
            stage2_recommended = max(total_recommended - stage1_runs, 5)
            
            st.info(f"Based on Stage 1 ({stage1_runs} runs), recommended additional: **{stage2_recommended}** runs")
            st.info(f"Total experiments: {stage1_runs + stage2_recommended} to fit quadratic model with {quad_params} parameters")
            
            stage2_runs = st.number_input(
                "Stage 2 Additional Runs", 
                min_value=5, 
                max_value=50, 
                value=stage2_recommended,
                help=f"Additional runs for quadratic model ({quad_params} parameters total)"
            )
            
            # Show total and efficiency
            total_runs = stage1_runs + stage2_runs
            run_to_param_ratio = total_runs / quad_params
            
            col_total, col_ratio = st.columns(2)
            with col_total:
                st.metric("Total Runs", total_runs)
            with col_ratio:
                st.metric("Run/Parameter Ratio", f"{run_to_param_ratio:.2f}")
                
            if run_to_param_ratio < 1.0:
                st.error("⚠️ Not enough runs to fit quadratic model!")
            elif run_to_param_ratio < 1.5:
                st.warning("⚠️ Minimum runs - expect 70-80% efficiency")
            elif run_to_param_ratio < 2.0:
                st.success("✓ Good coverage - expect 85-95% efficiency")
            else:
                st.success("✓ Excellent coverage - expect 95-99% efficiency")
            
            # Focus region option
            use_focus = st.checkbox("Focus on specific region", value=True)
            if use_focus:
                st.write("**Focus Region (after Stage 1 analysis):**")
                focus_factors = st.multiselect(
                    "Factors to focus on", 
                    options=factor_names,
                    default=factor_names[:2] if len(factor_names) >= 2 else factor_names
                )
        
        # Show benefits of sequential approach
        with st.expander("💡 Why Sequential DOE?", expanded=False):
            st.markdown("### Benefits of Sequential Approach:")
            for benefit in seq_recommendations['benefits']:
                st.write(f"• {benefit}")
        
        random_seed = st.number_input("Random Seed", value=42, key="seq_seed")
        
        # Response names
        n_responses = st.number_input("Number of Responses", min_value=1, max_value=5, value=3, key="seq_n_resp")
        response_names = []
        for i in range(n_responses):
            resp_name = st.text_input(f"Response {i+1} Name", value=f"Response_{i+1}", key=f"seq_resp_{i}")
            response_names.append(resp_name)
        
        generate_sequential = st.button("🔄 Generate Sequential Plan", type="primary")
    
    with col2:
        if generate_sequential:
            with st.spinner("Generating sequential DOE plan..."):
                try:
                    # Create sequential DOE generator
                    seq_doe = SequentialDOE(n_factors, factor_ranges)
                    
                    # Generate Stage 1 (screening)
                    stage1_design = seq_doe.generate_d_optimal(
                        n_runs=stage1_runs, 
                        model_order=1,  # Linear for screening
                        random_seed=random_seed
                    )
                    
                    # Prepare focus region if selected
                    focus_region = None
                    if use_focus and focus_factors:
                        # Create focus region centered around midpoints
                        focus_indices = [factor_names.index(f) for f in focus_factors]
                        focus_ranges = []
                        for idx in focus_indices:
                            min_val, max_val = factor_ranges[idx]
                            center = (min_val + max_val) / 2
                            width = (max_val - min_val) * 0.6  # Focus on 60% of range
                            focus_ranges.append((center - width/2, center + width/2))
                        
                        focus_region = {
                            'factor_indices': focus_indices,
                            'ranges': focus_ranges
                        }
                    
                    # Generate Stage 2 (augmentation)
                    stage2_design = seq_doe.augment_design(
                        stage1_design,
                        n_additional_runs=stage2_runs,
                        model_order=2,  # Quadratic for optimization
                        criterion="D-optimal",
                        focus_region=focus_region,
                        random_seed=random_seed + 1
                    )
                    
                    # Store in session state
                    st.session_state.seq_doe = seq_doe
                    st.session_state.stage1_design = stage1_design
                    st.session_state.stage2_design = stage2_design
                    st.session_state.seq_factor_names = factor_names
                    st.session_state.seq_response_names = response_names
                    
                    st.success("✅ Sequential DOE plan generated!")
                    
                except Exception as e:
                    st.error(f"Error generating sequential design: {str(e)}")
        
        # Display results if available
        if 'seq_doe' in st.session_state:
            seq_doe = st.session_state.seq_doe
            stage1_design = st.session_state.stage1_design
            stage2_design = st.session_state.stage2_design
            factor_names = st.session_state.seq_factor_names
            response_names = st.session_state.seq_response_names
            
            # Efficiency analysis
            st.subheader("📊 Sequential Efficiency Analysis")
            
            # Evaluate designs
            stage1_eval_linear = seq_doe.evaluate_design(stage1_design, model_order=1)
            stage1_eval_quad = seq_doe.evaluate_design(stage1_design, model_order=2)
            
            combined_design = np.vstack([stage1_design, stage2_design])
            combined_eval_linear = seq_doe.evaluate_design(combined_design, model_order=1)
            combined_eval_quad = seq_doe.evaluate_design(combined_design, model_order=2)
            
            # Create efficiency comparison
            efficiency_data = []
            
            efficiency_data.append({
                'Stage': 'Stage 1 Only',
                'Total Runs': len(stage1_design),
                'Linear D-Eff': stage1_eval_linear['d_efficiency'],
                'Quadratic D-Eff': stage1_eval_quad['d_efficiency'],
                'Can Fit': 'Linear model ✓'
            })
            
            efficiency_data.append({
                'Stage': 'Stage 1 + 2',
                'Total Runs': len(combined_design),
                'Linear D-Eff': combined_eval_linear['d_efficiency'],
                'Quadratic D-Eff': combined_eval_quad['d_efficiency'],
                'Can Fit': 'Full quadratic model ✓'
            })
            
            efficiency_df = pd.DataFrame(efficiency_data)
            st.dataframe(efficiency_df.round(4))
            
            # Visualization
            fig = go.Figure()
            
            # Stage progression
            stages = ['Stage 1', 'Stage 1+2']
            runs = [len(stage1_design), len(combined_design)]
            linear_eff = [stage1_eval_linear['d_efficiency'], combined_eval_linear['d_efficiency']]
            quad_eff = [stage1_eval_quad['d_efficiency'], combined_eval_quad['d_efficiency']]
            
            fig.add_trace(go.Scatter(
                x=runs, y=linear_eff,
                mode='lines+markers',
                name='Linear D-Efficiency',
                line=dict(color='blue', width=2),
                marker=dict(size=10)
            ))
            
            fig.add_trace(go.Scatter(
                x=runs, y=quad_eff,
                mode='lines+markers',
                name='Quadratic D-Efficiency',
                line=dict(color='red', width=2),
                marker=dict(size=10)
            ))
            
            # Add stage labels
            for i, (r, stage) in enumerate(zip(runs, stages)):
                fig.add_annotation(x=r, y=0.5, text=stage,
                                 showarrow=False, yshift=-30)
            
            fig.update_layout(
                title="Efficiency Progression in Sequential DOE",
                xaxis_title="Total Number of Runs",
                yaxis_title="D-Efficiency",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display designs
            st.subheader("📋 Sequential Design Details")
            
            tab1, tab2, tab3 = st.tabs(["Stage 1: Screening", "Stage 2: Augmentation", "Combined Design"])
            
            with tab1:
                st.write("**Stage 1: Screening Design** (Linear Model)")
                stage1_df = pd.DataFrame(stage1_design, columns=factor_names)
                stage1_df.insert(0, 'Run', range(1, len(stage1_df) + 1))
                stage1_df.insert(1, 'Stage', 'Screening')
                
                # Add response columns
                for resp in response_names:
                    stage1_df[resp] = np.nan
                
                st.dataframe(stage1_df.round(3))
                
                # Download button
                csv1 = stage1_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Stage 1 Design",
                    data=csv1,
                    file_name="stage1_screening.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.write("**Stage 2: Augmentation Design** (For Quadratic Model)")
                stage2_df = pd.DataFrame(stage2_design, columns=factor_names)
                stage2_df.insert(0, 'Run', range(len(stage1_df) + 1, len(stage1_df) + len(stage2_df) + 1))
                stage2_df.insert(1, 'Stage', 'Augmentation')
                
                # Add response columns
                for resp in response_names:
                    stage2_df[resp] = np.nan
                
                st.dataframe(stage2_df.round(3))
                
                # Download button
                csv2 = stage2_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Stage 2 Design",
                    data=csv2,
                    file_name="stage2_augmentation.csv",
                    mime="text/csv"
                )
            
            with tab3:
                st.write("**Combined Design** (All Experiments)")
                combined_df = pd.concat([stage1_df, stage2_df], ignore_index=True)
                st.dataframe(combined_df.round(3))
                
                # Download complete plan
                csv_all = combined_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Sequential Plan",
                    data=csv_all,
                    file_name="sequential_doe_complete.csv",
                    mime="text/csv"
                )
            
            # Workflow guide
            st.subheader("📚 Sequential DOE Workflow")
            
            with st.expander("How to Use Your Sequential Design", expanded=True):
                st.markdown(f"""
                ### 🚀 Sequential Experimentation Workflow
                
                **Stage 1: Screening ({len(stage1_design)} experiments)**
                1. Run all Stage 1 experiments
                2. Record responses for each run
                3. Analyze results to identify:
                   - Important factors (large effects)
                   - Unimportant factors (small effects)
                   - Promising regions of the design space
                
                **Analysis After Stage 1:**
                - Fit a linear model to identify main effects
                - Use ANOVA to test factor significance
                - Create main effects plots
                - Decide which factors to focus on
                
                **Stage 2: Augmentation ({len(stage2_design)} experiments)**
                1. Based on Stage 1 results, you may:
                   - Focus on important factors
                   - Explore promising regions
                   - Add points for quadratic effects
                2. Run Stage 2 experiments
                3. Combine with Stage 1 data
                
                **Final Analysis:**
                - Fit full quadratic model
                - Test for curvature and interactions
                - Optimize responses
                - Validate optimal conditions
                
                ### 💡 Benefits of This Approach:
                - **Efficiency**: {len(combined_design)} total runs vs. {seq_doe._count_parameters(2)} minimum for quadratic
                - **Flexibility**: Can stop after Stage 1 if linear model sufficient
                - **Learning**: Stage 2 design informed by Stage 1 results
                - **Risk Reduction**: Smaller initial investment
                
                ### 📊 Model Comparison:
                - **After Stage 1**: Can fit linear model with {n_factors + 1} parameters
                - **After Stage 2**: Can fit full quadratic with {seq_doe._count_parameters(2)} parameters
                """)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            # Simulate some responses for demonstration
            np.random.seed(random_seed)
            simulated_responses = np.random.randn(len(stage1_design))
            
            recommendations = seq_doe.recommend_next_stage(
                stage1_design, 
                simulated_responses,
                target_efficiency=0.9
            )
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Current D-Efficiency", f"{recommendations['current_efficiency']:.3f}")
            with col_b:
                st.metric("Target D-Efficiency", f"{recommendations['target_efficiency']:.3f}")
            
            if recommendations['efficiency_gap'] > 0:
                st.info(f"Stage 2 will improve efficiency by approximately {recommendations['efficiency_gap']:.3f}")

elif design_type == "Sequential Mixture DOE":
    st.markdown('<h2 class="sub-header">🧬🔄 Sequential Mixture Design</h2>', unsafe_allow_html=True)
    
    st.info("**Sequential Mixture DOE** is specifically designed for mixture experiments where components must sum to 100%. It supports fixed components and ensures all values are non-negative.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Mixture Sequential Setup")
        
        # Number of variable components
        n_variable_components = st.number_input(
            "Number of Variable Components", 
            min_value=2, 
            max_value=10, 
            value=5, 
            key="seq_mix_n_var_comp",
            help="Components that will have variable proportions (with bounds)"
        )
        
        # Parts mode selection FIRST (before fixed components)
        use_parts_mode = st.checkbox(
            "Use parts instead of proportions", 
            value=True,
            help="Parts mode: Specify ALL components in parts (e.g., 1, 0.3, 0.025) which will be normalized to proportions"
        )
        
        # Fixed components setup
        st.write("**Fixed Components (Optional):**")
        st.info("Fixed components will be added to your variable components")
        
        use_fixed = st.checkbox("Use fixed components", value=False)
        fixed_component_names = []
        fixed_components = {}
        fixed_parts = {}
        
        if use_fixed:
            n_fixed_components = st.number_input(
                "Number of Fixed Components", 
                min_value=1, 
                max_value=5, 
                value=1, 
                key="seq_mix_n_fixed_comp"
            )
            
            st.write("**Fixed Component Details:**")
            for i in range(n_fixed_components):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fixed_name = st.text_input(
                        f"Fixed Component {i+1} Name", 
                        value=f"Fixed_{i+1}", 
                        key=f"seq_mix_fixed_name_{i}"
                    )
                    fixed_component_names.append(fixed_name)
                with col_b:
                    if use_parts_mode:
                        fixed_val = st.number_input(
                            f"Value (parts)", 
                            value=0.05, 
                            min_value=0.0,
                            step=0.001,
                            format="%.3f",
                            key=f"seq_mix_fixed_parts_{i}"
                        )
                        fixed_parts[fixed_name] = fixed_val
                    else:
                        fixed_val = st.number_input(
                            f"Value (proportion)", 
                            value=0.02, 
                            min_value=0.0, 
                            max_value=1.0, 
                            step=0.01,
                            key=f"seq_mix_fixed_val_{i}"
                        )
                        fixed_components[fixed_name] = fixed_val
            
            # Validate fixed components
            if use_parts_mode and fixed_parts:
                # In parts mode, we need to estimate proportions
                # This is approximate since we don't know variable parts yet
                st.info("Fixed component proportions will be calculated based on total mixture")
            else:
                # In proportion mode, validate sum
                fixed_sum = sum(fixed_components.values())
                if fixed_sum >= 1.0:
                    st.error(f"⚠️ Fixed components sum to {fixed_sum:.2f} - no room for variable components!")
                else:
                    st.success(f"✅ Fixed components sum to {fixed_sum:.2f}, leaving {1-fixed_sum:.2f} for variable components")
        
        # Calculate total number of components
        n_components = n_variable_components + len(fixed_component_names)
        
        # Variable component setup
        st.write("**Variable Component Details:**")
        # Store all component names (fixed + variable)
        all_component_names = fixed_component_names.copy()  # Start with fixed names
        variable_component_names = []  # Track variable component names separately
        component_bounds = []
        component_bounds_parts = []  # Store original parts values
        
        # Add bounds for fixed components first (they have fixed bounds)
        for fixed_name in fixed_component_names:
            if use_parts_mode:
                # Fixed components have their value as both min and max
                fixed_val = fixed_parts.get(fixed_name, 0.0)
                component_bounds_parts.append((fixed_val, fixed_val))
                component_bounds.append((0.0, 1.0))  # Will be converted later
            else:
                fixed_val = fixed_components.get(fixed_name, 0.0)
                component_bounds.append((fixed_val, fixed_val))
        
        # Now add variable components
        for i in range(n_variable_components):
            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                name = st.text_input(f"Variable Component {i+1} Name", value=f"Component_{i+1}", key=f"seq_mix_name_{i}")
                all_component_names.append(name)
                variable_component_names.append(name)
            with col_b:
                if use_parts_mode:
                    min_val = st.number_input(
                        f"Min (parts)", 
                        value=0.0, 
                        min_value=0.0,
                        step=0.001,
                        format="%.3f",
                        key=f"seq_mix_min_parts_{i}"
                    )
                else:
                    min_val = st.number_input(
                        f"Min", 
                        value=0.0, 
                        min_value=0.0, 
                        max_value=1.0, 
                        step=0.01, 
                        key=f"seq_mix_min_{i}"
                    )
            with col_c:
                if use_parts_mode:
                    max_val = st.number_input(
                        f"Max (parts)", 
                        value=1.0 if i < 2 else 0.1,  # Default higher for main components
                        min_value=0.0,
                        step=0.001,
                        format="%.3f",
                        key=f"seq_mix_max_parts_{i}"
                    )
                else:
                    max_val = st.number_input(
                        f"Max", 
                        value=1.0, 
                        min_value=0.0, 
                        max_value=1.0, 
                        step=0.01, 
                        key=f"seq_mix_max_{i}"
                    )
            
            if use_parts_mode:
                component_bounds_parts.append((min_val, max_val))
                # Will convert to proportions after getting all values
                component_bounds.append((0.0, 1.0))  # Placeholder
            else:
                component_bounds.append((min_val, max_val))
        
        # Validate bounds for variable components only
        if use_parts_mode:
            # For parts mode, we need at least one valid combination
            sum_min_parts = sum(bound[0] for bound in component_bounds_parts)
            sum_max_parts = sum(bound[1] for bound in component_bounds_parts)
            
            if sum_min_parts == 0 and sum_max_parts == 0:
                st.error("⚠️ All bounds are zero - impossible mixture!")
            else:
                st.success("✅ Variable component bounds are feasible")
                
                # Show conversion preview
                with st.expander("Preview of proportion ranges for variable components"):
                    st.write("**Estimated proportion ranges (variable components only):**")
                    # Calculate bounds considering only variable components will sum to (1 - fixed_sum)
                    available_for_variable = 1.0 - sum(fixed_components.values())
                    
                    min_props = []
                    max_props = []
                    
                    # Get only variable component bounds (skip fixed components)
                    variable_bounds_parts = component_bounds_parts[len(fixed_component_names):]
                    
                    # Calculate sum of variable component bounds only
                    var_sum_min_parts = sum(bound[0] for bound in variable_bounds_parts)
                    var_sum_max_parts = sum(bound[1] for bound in variable_bounds_parts)
                    
                    for i in range(n_variable_components):
                        # Calculate proportions within the available space
                        if var_sum_max_parts > 0:
                            min_prop = (variable_bounds_parts[i][0] / var_sum_max_parts) * available_for_variable
                            max_prop = (variable_bounds_parts[i][1] / var_sum_min_parts if var_sum_min_parts > 0 else 1.0) * available_for_variable
                        else:
                            min_prop = 0
                            max_prop = 0
                        
                        min_props.append(min_prop)
                        max_props.append(max_prop)
                    
                    # Only show variable components in preview
                    # variable_component_names already contains only variable components
                    
                    preview_df = pd.DataFrame({
                        'Component': variable_component_names,
                        'Min Parts': [b[0] for b in variable_bounds_parts],
                        'Max Parts': [b[1] for b in variable_bounds_parts],
                        'Min %': [f"{p*100:.1f}%" for p in min_props],
                        'Max %': [f"{p*100:.1f}%" for p in max_props]
                    })
                    st.dataframe(preview_df)
                
                # Convert parts bounds to proportion bounds for the algorithm
                component_bounds = [(min_prop, max_prop) for min_prop, max_prop in zip(min_props, max_props)]
        else:
            # For proportion mode, variable components must sum to (1 - fixed_sum)
            available_for_variable = 1.0 - sum(fixed_components.values())
            sum_min = sum(bound[0] for bound in component_bounds)
            sum_max = sum(bound[1] for bound in component_bounds)
            
            if sum_min > available_for_variable:
                st.error(f"⚠️ Variable component minimums sum to {sum_min:.2f} but only {available_for_variable:.2f} is available!")
            elif sum_max < available_for_variable:
                st.error(f"⚠️ Variable component maximums sum to {sum_max:.2f} but need to fill {available_for_variable:.2f}!")
            else:
                st.success("✅ Variable component bounds are feasible")
        
        # Batch size input
        st.write("**Batch Size:**")
        batch_size = st.number_input(
            "Batch Size (for quantity calculations)", 
            value=100.0, 
            min_value=1.0,
            step=1.0,
            help="Total batch size in your preferred units (kg, lb, etc.)",
            key="batch_size"
        )
        
        # Calculate number of variable components
        n_variable = n_components - len(fixed_components)
        
        st.write("**Stage Configuration:**")
        
        # Create temporary object to get recommendations
        try:
            temp_seq_mix = SequentialMixtureDOE(
                n_components, 
                all_component_names, 
                component_bounds if not use_parts_mode else component_bounds_parts,
                fixed_components if fixed_components else None,
                use_parts_mode=use_parts_mode
            )
            mix_recommendations = temp_seq_mix.get_mixture_recommendations(n_variable)
            
            # Quality level selection
            quality_level_mix = st.selectbox(
                "Experiment Quality Level",
                ["minimum", "recommended", "excellent"],
                index=1,
                key="seq_mix_quality",
                help="Minimum: 1x parameters | Recommended: 1.5x parameters | Excellent: 2x parameters"
            )
            
            # Stage 1 settings
            with st.expander("Stage 1: Screening", expanded=True):
                st.write(f"**{mix_recommendations['stage1']['purpose']}**")
                
                col_min, col_rec, col_exc = st.columns(3)
                with col_min:
                    st.metric("Minimum", mix_recommendations['stage1']['minimum'])
                with col_rec:
                    st.metric("Recommended", mix_recommendations['stage1']['recommended'])
                with col_exc:
                    st.metric("Excellent", mix_recommendations['stage1']['excellent'])
                
                # Set default based on quality level
                stage1_mix_default = {
                    'minimum': mix_recommendations['stage1']['minimum'],
                    'recommended': mix_recommendations['stage1']['recommended'],
                    'excellent': mix_recommendations['stage1']['excellent']
                }[quality_level_mix]
                
                stage1_mix_runs = st.number_input(
                    "Stage 1 Runs", 
                    min_value=mix_recommendations['stage1']['minimum'], 
                    max_value=50, 
                    value=stage1_mix_default,
                    key="seq_mix_stage1_runs",
                    help=mix_recommendations['stage1']['can_fit']
                )
                
                st.info(f"✓ Can fit: {mix_recommendations['stage1']['can_fit']}")
                st.warning(f"⚠️ {mix_recommendations['stage1']['note']}")
            
            # Stage 2 settings
            with st.expander("Stage 2: Optimization", expanded=True):
                st.write(f"**{mix_recommendations['stage2']['purpose']}**")
                
                stage2_mix_runs = st.number_input(
                    "Stage 2 Additional Runs", 
                    min_value=5, 
                    max_value=50, 
                    value=mix_recommendations['stage2']['recommended_additional'],
                    key="seq_mix_stage2_runs",
                    help=f"Additional runs for quadratic mixture model"
                )
                
                # Focus components option
                use_focus_mix = st.checkbox("Focus on specific components", value=True, key="seq_mix_focus")
                focus_components = []
                if use_focus_mix:
                    # Only exclude components that are actually fixed (checked and have values)
                    actually_fixed = []
                    if use_fixed and fixed_component_names:
                        actually_fixed = fixed_component_names
                    
                    variable_components = [c for c in all_component_names if c not in actually_fixed]
                    focus_components = st.multiselect(
                        "Components to focus on", 
                        options=variable_components,
                        default=variable_components[:2] if len(variable_components) >= 2 else variable_components,
                        key="seq_mix_focus_comp"
                    )
                
                # Show mixture-specific info
                st.info(f"📊 Mixture Constraints:")
                st.write(f"• Fixed components: {len(fixed_components)}")
                st.write(f"• Variable components: {n_variable}")
                st.write(f"• All components sum to 1")
                st.write(f"• All values ≥ 0")
        
        except Exception as e:
            st.error(f"Configuration error: {str(e)}")
            temp_seq_mix = None
            mix_recommendations = None
        
        random_seed_mix = st.number_input("Random Seed", value=42, key="seq_mix_seed")
        
        # Response names
        n_responses_mix = st.number_input("Number of Responses", min_value=1, max_value=5, value=2, key="seq_mix_n_resp")
        response_names_mix = []
        for i in range(n_responses_mix):
            resp_name = st.text_input(f"Response {i+1} Name", value=f"Property_{i+1}", key=f"seq_mix_resp_{i}")
            response_names_mix.append(resp_name)
        
        generate_sequential_mix = st.button("🧬🔄 Generate Sequential Mixture Plan", type="primary")
    
    with col2:
        # Check if bounds are valid based on mode
        bounds_valid = False
        if use_parts_mode:
            # In parts mode, just need non-zero max parts
            sum_max_parts = sum(bound[1] for bound in component_bounds_parts) if 'component_bounds_parts' in locals() else 0
            bounds_valid = sum_max_parts > 0
        else:
            # In proportion mode, need feasible bounds
            bounds_valid = sum_min <= 1 and sum_max >= 1
        
        if generate_sequential_mix and temp_seq_mix is not None and bounds_valid:
            with st.spinner("Generating sequential mixture DOE plan..."):
                try:
                    # Create sequential mixture DOE generator
                    # In parts mode, pass fixed_parts instead of fixed_components
                    if use_parts_mode and fixed_parts:
                        # The SequentialMixtureDOE will handle conversion from parts to proportions
                        seq_mix_doe = SequentialMixtureDOE(
                            n_components,
                            all_component_names,
                            component_bounds_parts,
                            fixed_parts,  # Pass parts, not proportions
                            use_parts_mode=True
                        )
                    else:
                        seq_mix_doe = SequentialMixtureDOE(
                            n_components,
                            all_component_names,
                            component_bounds,
                            fixed_components if fixed_components else None,
                            use_parts_mode=False
                        )
                    
                    # Generate Stage 1 (screening)
                    stage1_mix_design = seq_mix_doe.generate_d_optimal_mixture(
                        n_runs=stage1_mix_runs,
                        model_type="linear",
                        random_seed=random_seed_mix
                    )
                    
                    # Apply fixed components
                    stage1_mix_design = seq_mix_doe._adjust_for_fixed_components(stage1_mix_design)
                    
                    # Generate Stage 2 (augmentation)
                    stage2_mix_design = seq_mix_doe.augment_mixture_design(
                        stage1_mix_design,
                        n_additional_runs=stage2_mix_runs,
                        model_type="quadratic",
                        focus_components=focus_components if use_focus_mix else None,
                        random_seed=random_seed_mix + 1
                    )
                    
                    # Store in session state
                    st.session_state.seq_mix_doe = seq_mix_doe
                    st.session_state.stage1_mix_design = stage1_mix_design
                    st.session_state.stage2_mix_design = stage2_mix_design
                    st.session_state.seq_mix_component_names = all_component_names
                    st.session_state.seq_mix_response_names = response_names_mix
                    st.session_state.seq_mix_fixed_components = fixed_components
                    st.session_state.seq_mix_use_parts_mode = use_parts_mode
                    st.session_state.seq_mix_fixed_parts = fixed_parts if use_parts_mode else {}
                    st.session_state.seq_mix_batch_size = batch_size
                    
                    st.success("✅ Sequential mixture DOE plan generated!")
                    
                except Exception as e:
                    st.error(f"Error generating sequential mixture design: {str(e)}")
        
        # Display results if available
        if 'seq_mix_doe' in st.session_state:
            seq_mix_doe = st.session_state.seq_mix_doe
            stage1_mix_design = st.session_state.stage1_mix_design
            stage2_mix_design = st.session_state.stage2_mix_design
            component_names = st.session_state.seq_mix_component_names if 'seq_mix_component_names' in st.session_state else all_component_names
            response_names_mix = st.session_state.seq_mix_response_names
            fixed_components = st.session_state.seq_mix_fixed_components
            use_parts_mode = st.session_state.seq_mix_use_parts_mode
            fixed_parts = st.session_state.seq_mix_fixed_parts
            batch_size = st.session_state.seq_mix_batch_size
            
            # Verification
            st.subheader("✅ Design Verification")
            
            all_designs = np.vstack([stage1_mix_design, stage2_mix_design])
            
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                sums_ok = np.allclose(all_designs.sum(axis=1), 1.0)
                st.metric("Sum to 100%", "✓ Pass" if sums_ok else "✗ Fail")
            with col_v2:
                non_neg = np.all(all_designs >= 0)
                st.metric("Non-negative", "✓ Pass" if non_neg else "✗ Fail")
            with col_v3:
                st.metric("Total Mixtures", len(all_designs))
            
            # Display fixed components
            if fixed_components:
                st.subheader("🔒 Fixed Components")
                if use_parts_mode and fixed_parts:
                    # Show parts and calculated proportions
                    fixed_data = []
                    for comp_name, parts in fixed_parts.items():
                        fixed_data.append({
                            'Component': comp_name,
                            'Parts': parts,
                            'Proportion': fixed_components.get(comp_name, 0),
                            'Percentage': f"{fixed_components.get(comp_name, 0) * 100:.1f}%"
                        })
                    fixed_df = pd.DataFrame(fixed_data)
                    st.dataframe(fixed_df)
                else:
                    fixed_df = pd.DataFrame.from_dict(fixed_components, orient='index', columns=['Fixed Value'])
                    fixed_df['Percentage'] = (fixed_df['Fixed Value'] * 100).round(1).astype(str) + '%'
                    st.dataframe(fixed_df)
            
            # Calculate and show batch quantities
            st.subheader("📦 Batch Quantity Calculator")
            col_calc1, col_calc2 = st.columns(2)
            
            with col_calc1:
                show_stage = st.selectbox(
                    "Show quantities for:",
                    ["Stage 1", "Stage 2", "Combined"],
                    key="batch_calc_stage"
                )
            
            with col_calc2:
                custom_batch = st.number_input(
                    "Custom batch size:",
                    value=batch_size,
                    min_value=1.0,
                    step=1.0,
                    key="custom_batch"
                )
            
            # Select design based on choice
            if show_stage == "Stage 1":
                selected_design = stage1_mix_design
            elif show_stage == "Stage 2":
                selected_design = stage2_mix_design
            else:
                selected_design = all_designs
            
            # Calculate quantities
            quantities_df = seq_mix_doe.calculate_batch_quantities(selected_design, custom_batch)
            
            # Add run labels
            if show_stage == "Stage 1":
                quantities_df['Stage'] = 'Screening'
            elif show_stage == "Stage 2":
                quantities_df['Stage'] = 'Augmentation'
                quantities_df['Mixture'] = quantities_df['Mixture'] + len(stage1_mix_design)
            else:
                quantities_df['Stage'] = ['Screening'] * len(stage1_mix_design) + ['Augmentation'] * len(stage2_mix_design)
            
            # Reorder columns
            cols = ['Mixture', 'Stage'] + component_names + ['Total']
            quantities_df = quantities_df[cols]
            
            st.dataframe(quantities_df.round(2))
            
            # Download quantities
            csv_quantities = quantities_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {show_stage} Quantities",
                data=csv_quantities,
                file_name=f"{show_stage.lower().replace(' ', '_')}_quantities_batch_{custom_batch}.csv",
                mime="text/csv",
                key=f"dl_quantities_{show_stage}"
            )
            
            # Display designs
            st.subheader("📋 Sequential Mixture Design Details")
            
            tab1, tab2, tab3 = st.tabs(["Stage 1: Screening", "Stage 2: Augmentation", "Combined Design"])
            
            with tab1:
                st.write("**Stage 1: Linear Mixture Model**")
                stage1_mix_df = pd.DataFrame(stage1_mix_design, columns=component_names)
                stage1_mix_df.insert(0, 'Run', range(1, len(stage1_mix_df) + 1))
                stage1_mix_df.insert(1, 'Stage', 'Screening')
                
                # Add percentage columns
                for col in component_names:
                    stage1_mix_df[f"{col} (%)"] = (stage1_mix_df[col] * 100).round(1)
                
                # Add response columns
                for resp in response_names_mix:
                    stage1_mix_df[resp] = np.nan
                
                st.dataframe(stage1_mix_df.round(3))
                
                # Download button
                csv1 = stage1_mix_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Stage 1 Mixture Design",
                    data=csv1,
                    file_name="stage1_mixture_screening.csv",
                    mime="text/csv",
                    key="dl_mix_stage1"
                )
            
            with tab2:
                st.write("**Stage 2: Quadratic Mixture Model**")
                stage2_mix_df = pd.DataFrame(stage2_mix_design, columns=component_names)
                stage2_mix_df.insert(0, 'Run', range(len(stage1_mix_df) + 1, len(stage1_mix_df) + len(stage2_mix_df) + 1))
                stage2_mix_df.insert(1, 'Stage', 'Augmentation')
                
                # Add percentage columns
                for col in component_names:
                    stage2_mix_df[f"{col} (%)"] = (stage2_mix_df[col] * 100).round(1)
                
                # Add response columns
                for resp in response_names_mix:
                    stage2_mix_df[resp] = np.nan
                
                st.dataframe(stage2_mix_df.round(3))
                
                # Download button
                csv2 = stage2_mix_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Stage 2 Mixture Design",
                    data=csv2,
                    file_name="stage2_mixture_augmentation.csv",
                    mime="text/csv",
                    key="dl_mix_stage2"
                )
            
            with tab3:
                st.write("**Combined Mixture Design**")
                combined_mix_df = pd.concat([stage1_mix_df, stage2_mix_df], ignore_index=True)
                st.dataframe(combined_mix_df.round(3))
                
                # Download complete plan
                csv_all = combined_mix_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Sequential Mixture Plan",
                    data=csv_all,
                    file_name="sequential_mixture_complete.csv",
                    mime="text/csv",
                    key="dl_mix_all"
                )
            
            # Ternary plot for 3 components (if no fixed components)
            if n_components == 3 and len(fixed_components) == 0:
                st.subheader("📊 Mixture Design Visualization")
                
                fig = go.Figure()
                
                # Stage 1 points
                fig.add_trace(go.Scatterternary({
                    'mode': 'markers+text',
                    'a': stage1_mix_design[:, 0],
                    'b': stage1_mix_design[:, 1], 
                    'c': stage1_mix_design[:, 2],
                    'text': [f"S1-{i+1}" for i in range(len(stage1_mix_design))],
                    'textposition': "top center",
                    'marker': {
                        'symbol': 'circle',
                        'size': 10,
                        'color': 'blue',
                        'line': {'width': 2, 'color': 'darkblue'}
                    },
                    'name': 'Stage 1'
                }))
                
                # Stage 2 points
                fig.add_trace(go.Scatterternary({
                    'mode': 'markers+text',
                    'a': stage2_mix_design[:, 0],
                    'b': stage2_mix_design[:, 1], 
                    'c': stage2_mix_design[:, 2],
                    'text': [f"S2-{i+1}" for i in range(len(stage2_mix_design))],
                    'textposition': "bottom center",
                    'marker': {
                        'symbol': 'square',
                        'size': 10,
                        'color': 'red',
                        'line': {'width': 2, 'color': 'darkred'}
                    },
                    'name': 'Stage 2'
                }))
                
                fig.update_layout({
                    'ternary': {
                        'sum': 1,
                        'aaxis': {'title': component_names[0]},
                        'baxis': {'title': component_names[1]},
                        'caxis': {'title': component_names[2]}
                    },
                    'height': 500,
                    'title': "Sequential Mixture Design",
                    'showlegend': True
                })
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Workflow guide
            st.subheader("📚 Sequential Mixture DOE Workflow")
            
            with st.expander("How to Use Your Sequential Mixture Design", expanded=True):
                st.markdown(f"""
                ### 🧬 Sequential Mixture Experimentation
                
                **Stage 1: Screening ({len(stage1_mix_design)} mixtures)**
                1. Prepare all Stage 1 mixtures
                2. Test each mixture and record properties
                3. Analyze to identify:
                   - Which components have largest effects
                   - Promising composition regions
                   - Component interactions
                
                **Special Considerations for Mixtures:**
                - **Mixing Order**: May affect results - be consistent
                - **Preparation Method**: Document thoroughly
                - **Fixed Components**: Add at consistent point in process
                - **Quality Control**: Verify actual compositions
                
                **Stage 2: Optimization ({len(stage2_mix_design)} mixtures)**
                1. Based on Stage 1:
                   - Focus on important components
                   - Explore promising regions
                   - Test for synergistic effects
                2. Prepare and test Stage 2 mixtures
                
                **Analysis Approach:**
                - Use Scheffé models (no intercept)
                - Check for synergistic/antagonistic effects
                - Consider practical constraints
                - Validate optimal formulation
                
                ### 🔬 Mixture-Specific Benefits:
                - **Efficiency**: {len(all_designs)} total mixtures
                - **Constraints**: Handles fixed components
                - **Learning**: Stage 2 focuses on key components
                - **Practical**: All compositions are feasible
                
                ### 📊 Fixed Components:
                {f"• {len(fixed_components)} components fixed" if fixed_components else "• No fixed components"}
                {f"• {n_variable} components variable" if fixed_components else ""}
                {f"• Fixed sum: {sum(fixed_components.values()):.2%}" if fixed_components else ""}
                """)

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
