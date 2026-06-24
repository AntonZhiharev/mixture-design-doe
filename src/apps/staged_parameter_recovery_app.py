"""
Streamlit App for Interactive Staged Parameter Recovery

This provides a graphical interface for users to make decisions at each stage
of the parameter recovery process.
"""

import streamlit as st
import sys
sys.path.append('.')
sys.path.append('src')

import numpy as np
import pandas as pd
from src.algorithms.jmp_style_mixture_design import generate_jmp_sparse_design
from src.algorithms.jmp_full_model_screening import JMPFullModelScreening


def generate_test_function():
    """Generate test function with known parameters"""
    linear_coeffs = [3.0, -4.0, 5.0, 2.0, -3.0]
    
    interactions = {
        (0, 2): 2.5,   # x1*x3
        (1, 3): -3.0,  # x2*x4
        (0, 4): 1.5,   # x1*x5
        (2, 3): -2.0,  # x3*x4
        (0, 1, 2): 4.0,   # x1*x2*x3
        (2, 3, 4): -3.5,  # x3*x4*x5
        (0, 1, 2, 3): 5.0,    # x1*x2*x3*x4
        (1, 2, 3, 4): -4.5,   # x2*x3*x4*x5
    }
    
    def mixture_function(x, noise_level=0.01):
        result = sum(coeff * x[i] for i, coeff in enumerate(linear_coeffs))
        for interaction, coeff in interactions.items():
            term = coeff * np.prod([x[idx] for idx in interaction])
            result += term
        if noise_level > 0:
            result += np.random.normal(0, noise_level)
        return result
    
    mixture_function.linear_coeffs = linear_coeffs
    mixture_function.interactions = interactions
    return mixture_function


def main():
    st.title("🔬 Staged Parameter Recovery Interface")
    st.markdown("### Interactive Decision-Making for Parameter Screening")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.current_stage = 0
        st.session_state.active_components = set(range(5))
        st.session_state.stage_results = []
        st.session_state.design = None
        st.session_state.responses = None
        st.session_state.true_function = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        n_components = st.number_input("Number of Components", 3, 10, 5)
        n_runs = st.number_input("Number of Runs", 20, 100, 45)
        noise_level = st.slider("Noise Level", 0.0, 0.1, 0.01, 0.001)
        significance_threshold = st.slider("Significance Threshold (α)", 0.01, 0.10, 0.05, 0.01)
        
        if st.button("🚀 Initialize Experiment", type="primary"):
            # Generate design and responses
            st.session_state.true_function = generate_test_function()
            np.random.seed(123)
            st.session_state.design = generate_jmp_sparse_design(n_runs=n_runs, n_components=n_components)
            st.session_state.responses = np.array([
                st.session_state.true_function(point, noise_level=noise_level) 
                for point in st.session_state.design
            ])
            st.session_state.initialized = True
            st.session_state.current_stage = 1
            st.session_state.active_components = set(range(n_components))
            st.session_state.stage_results = []
            st.success(f"✅ Initialized with {n_runs} runs")
        
        st.markdown("---")
        st.markdown("### 📊 Current Status")
        if st.session_state.initialized:
            st.metric("Current Stage", st.session_state.current_stage)
            st.metric("Active Components", len(st.session_state.active_components))
            st.metric("Stages Completed", len(st.session_state.stage_results))
    
    # Main content
    if not st.session_state.initialized:
        st.info("👈 Configure parameters and click 'Initialize Experiment' to begin")
        st.markdown("""
        ### How it works:
        1. **Configure** your experiment parameters in the sidebar
        2. **Initialize** to generate design and responses
        3. **Review** results at each stage
        4. **Decide** which parameters to continue investigating
        5. **Progress** through stages until complete
        
        At each stage, you'll see:
        - 📋 True parameter values (for comparison)
        - 📊 Estimated coefficients and p-values
        - ✅ Significance indicators
        - 📌 Recommendations for next steps
        """)
        return
    
    # Stage execution
    st.header(f"📍 Stage {st.session_state.current_stage}: Analysis")
    
   
    component_names = [f"x{i+1}" for i in range(n_components)]
    
    if st.session_state.current_stage == 1:
        # Stage 1: Main Effects
        st.subheader("Main Effects Screening")
        st.info(f"Testing all {n_components} components for main effects")
        
        # Run screening
        screener = JMPFullModelScreening(
            n_components=n_components,
            max_order=1,
            alpha=0.05,
            p_threshold=0.05,
            max_vif=10.0,
            verbose=False
        )
        result = screener.screen_full_model(st.session_state.design, st.session_state.responses)
        
        # Display results table
        data = []
        significant_terms = []
        for i, comp in enumerate(component_names):
            true_val = st.session_state.true_function.linear_coeffs[i] if i < len(st.session_state.true_function.linear_coeffs) else 0.0
            if comp in result.selected_terms:
                coef = result.coefficients[comp]
                p_val = result.p_values.get(comp, 0.0)
                is_sig = p_val < significance_threshold
                data.append({
                    "Term": comp,
                    "True Value": f"{true_val:+.3f}",
                    "Estimated": f"{coef:+.3f}",
                    "P-Value": f"{p_val:.4f}",
                    "Significant": "✅" if is_sig else "⚠️"
                })
                if is_sig:
                    significant_terms.append(comp)
            else:
                data.append({
                    "Term": comp,
                    "True Value": f"{true_val:+.3f}",
                    "Estimated": "N/A",
                    "P-Value": "N/A",
                    "Significant": "❌"
                })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Significant Effects", f"{len(significant_terms)}/{n_components}")
        with col2:
            st.metric("Model R²", f"{result.r_squared:.4f}")
        with col3:
            detection_rate = len(significant_terms) / n_components * 100
            st.metric("Detection Rate", f"{detection_rate:.0f}%")
        
        # Recommendation
        st.markdown("### 📋 Recommendation")
        if len(significant_terms) == n_components:
            st.success(f"✅ All components show significant main effects. Continue with all {n_components} components for 2-way interactions.")
            selected_components = set(range(n_components))
        elif len(significant_terms) > 1:
            st.warning(f"⚠️ Only {len(significant_terms)} components show significant main effects.")
            st.info(f"Recommend testing 2-way interactions only among: {significant_terms}")
            selected_components = {int(term[1])-1 for term in significant_terms}
        else:
            st.error("❌ Insufficient significant effects. Consider reviewing data quality.")
            selected_components = set()
        
        # User decision
        st.markdown("### 🎯 Your Decision")
        
        if len(selected_components) > 1:
            proceed = st.button("➡️ Proceed to Stage 2 (2-Way Interactions)", type="primary")
            if proceed:
                st.session_state.current_stage = 2
                st.session_state.active_components = selected_components
                st.session_state.stage_results.append({
                    "stage": 1,
                    "significant_terms": significant_terms,
                    "r_squared": result.r_squared
                })
                st.rerun()
        else:
            st.error("Cannot proceed: Need at least 2 significant components for interactions")
    
    elif st.session_state.current_stage == 2:
        # Stage 2: 2-Way Interactions
        st.subheader("2-Way Interaction Screening")
        active_comp_list = sorted(list(st.session_state.active_components))
        st.info(f"Testing 2-way interactions among {len(active_comp_list)} components: {[component_names[i] for i in active_comp_list]}")
        
        # Run screening
        screener = JMPFullModelScreening(
            n_components=n_components,
            max_order=2,
            alpha=0.05,
            p_threshold=0.05,
            max_vif=10.0,
            verbose=False
        )
        result = screener.screen_full_model(st.session_state.design, st.session_state.responses)
        
        # Extract 2-way terms
        two_way_terms = [term for term in result.selected_terms 
                        if '*' in term and len(term.split('*')) == 2]
        
        # Get true 2-way interactions
        true_2way = {f"x{k[0]+1}*x{k[1]+1}": v for k, v in st.session_state.true_function.interactions.items() if len(k) == 2}
        
        # Display results
        if two_way_terms:
            data = []
            significant_interactions = []
            for term in two_way_terms:
                coef = result.coefficients[term]
                p_val = result.p_values.get(term, 0.0)
                is_sig = p_val < significance_threshold
                true_val = true_2way.get(term, 0.0)
                
                data.append({
                    "Term": term,
                    "True Value": f"{true_val:+.3f}" if true_val != 0 else "0.000",
                    "Estimated": f"{coef:+.3f}",
                    "P-Value": f"{p_val:.4f}",
                    "Significant": "✅" if is_sig else "⚠️"
                })
                if is_sig:
                    significant_interactions.append(term)
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No 2-way interactions detected")
            significant_interactions = []
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            n_possible = len(active_comp_list) * (len(active_comp_list) - 1) // 2
            st.metric("Significant Interactions", f"{len(significant_interactions)}/{n_possible}")
        with col2:
            st.metric("Model R²", f"{result.r_squared:.4f}")
        with col3:
            if n_possible > 0:
                detection_rate = len(significant_interactions) / n_possible * 100
                st.metric("Detection Rate", f"{detection_rate:.0f}%")
        
        # Recommendation
        st.markdown("### 📋 Recommendation")
        if len(significant_interactions) > 0:
            st.success(f"✅ Found {len(significant_interactions)} significant 2-way interaction(s)")
            if len(st.session_state.active_components) > 2:
                st.info("Can proceed to test 3-way interactions if desired")
        else:
            st.warning("⚠️ No significant 2-way interactions found. Main effects model may be sufficient.")
        
        # User decision
        st.markdown("### 🎯 Your Decision")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Complete Analysis", type="primary"):
                st.session_state.stage_results.append({
                    "stage": 2,
                    "significant_terms": significant_interactions,
                    "r_squared": result.r_squared
                })
                st.session_state.current_stage = "complete"
                st.rerun()
        with col2:
            if len(st.session_state.active_components) > 2 and len(significant_interactions) > 0:
                if st.button("➡️ Continue to Stage 3"):
                    st.session_state.current_stage = 3
                    st.session_state.stage_results.append({
                        "stage": 2,
                        "significant_terms": significant_interactions,
                        "r_squared": result.r_squared
                    })
                    st.rerun()
    
    elif st.session_state.current_stage == "complete":
        st.success("🎉 Analysis Complete!")
        st.balloons()
        
        # Display summary
        st.header("📊 Final Summary")
        
        total_significant = sum(len(stage["significant_terms"]) for stage in st.session_state.stage_results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stages Completed", len(st.session_state.stage_results))
        with col2:
            st.metric("Total Significant Terms", total_significant)
        with col3:
            if st.session_state.stage_results:
                final_r2 = st.session_state.stage_results[-1]["r_squared"]
                st.metric("Final R²", f"{final_r2:.4f}")
        
        # Stage-by-stage results
        st.subheader("Stage-by-Stage Results")
        for stage in st.session_state.stage_results:
            with st.expander(f"Stage {stage['stage']}: {len(stage['significant_terms'])} significant terms (R²={stage['r_squared']:.4f})"):
                st.write("**Significant terms:**")
                for term in stage['significant_terms']:
                    st.write(f"- {term}")
        
        if st.button("🔄 Start New Analysis"):
            st.session_state.initialized = False
            st.rerun()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Staged Parameter Recovery",
        page_icon="🔬",
        layout="wide"
    )
    main()
