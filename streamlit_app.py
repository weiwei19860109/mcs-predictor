import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="MCS Prediction System",
    page_icon="🧠",
    layout="wide"
)

# ==================== Model Parameters ====================
INTERCEPT = 1.167968
THRESHOLD = 0.4502  # Best threshold from Youden index
YOUDEN_INDEX = 0.3276  # Youden's index

# Model coefficients
COEFFICIENTS = {
    'global_alpha_clustering_coefficient': -12.879485,
    'global_alpha_degree_centrality_mean': 0.293723,
    'global_alpha_mean_plv': 5.287009,
    'global_alpha_characteristic_path_length': -0.150704,
    'global_alpha_global_efficiency': -0.223109,
    'global_beta_degree_centrality_mean': 0.170574,
    'global_beta_mean_plv': 3.070339,
    'beta_Fp2-O2_PLV': -2.299571,
    'gamma_Fp2-F8_PLV': -0.622310,
    'gamma_Fp2-O2_PLV': 3.693667,
    'gamma_F7-O2_PLV': 1.201478
}

# Feature names in order
FEATURES = list(COEFFICIENTS.keys())

# Feature display names (for UI)
FEATURE_DISPLAY_NAMES = {
    'global_alpha_clustering_coefficient': 'Alpha Clustering Coeff',
    'global_alpha_degree_centrality_mean': 'Alpha Degree Centrality',
    'global_alpha_mean_plv': 'Alpha Mean PLV',
    'global_alpha_characteristic_path_length': 'Alpha Path Length',
    'global_alpha_global_efficiency': 'Alpha Global Efficiency',
    'global_beta_degree_centrality_mean': 'Beta Degree Centrality',
    'global_beta_mean_plv': 'Beta Mean PLV',
    'beta_Fp2-O2_PLV': 'Beta Fp2-O2 PLV',
    'gamma_Fp2-F8_PLV': 'Gamma Fp2-F8 PLV',
    'gamma_Fp2-O2_PLV': 'Gamma Fp2-O2 PLV',
    'gamma_F7-O2_PLV': 'Gamma F7-O2 PLV'
}

# ==================== Preprocessing Conditions ====================
PREPROCESSING_CONDITIONS = {
    'Sampling Rate': 'Resampled to 250 Hz (if original > 250 Hz)',
    'Bandpass Filter': '0.5 - 100 Hz (firwin design)',
    'Notch Filter': '50 Hz (power line noise removal)',
    'Reference': 'Average reference',
    'Channel Selection': 'Standard 19 EEG channels (10-20 system)',
    'Channel Renaming': 'Standardized naming convention',
    'Valid Channels': 'Minimum 10 channels required'
}

# ==================== Prediction Function ====================
def predict(features_dict):
    """Calculate MCS probability"""
    logit = INTERCEPT
    for feature, value in features_dict.items():
        if feature in COEFFICIENTS:
            logit += COEFFICIENTS[feature] * value
    probability = 1 / (1 + np.exp(-logit))
    return probability, logit

# ==================== Initialize session state ====================
def init_session_state():
    """Initialize session state with default values"""
    if 'features_initialized' not in st.session_state:
        for feature in FEATURES:
            if feature not in st.session_state:
                st.session_state[feature] = 0.5
        st.session_state['features_initialized'] = True

# ==================== Main UI ====================
def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🧠 MCS Prediction System")
    st.markdown("**Minimally Conscious State Prediction Model** based on EEG Network Features")
    st.markdown("---")
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Performance")
        
        st.metric("AUC", "0.7118 (95% CI: 0.6161-0.8076)")
        st.metric("Accuracy", "66.67%")
        st.metric("Precision", "63.38%")
        st.metric("Recall/Sensitivity", "81.82%")
        st.metric("Specificity", "50.94%")
        st.metric("F1 Score", "0.7143")
        st.metric("Best Threshold", f"{THRESHOLD:.4f}")
        st.metric("Youden's Index", f"{YOUDEN_INDEX:.4f}")
        
        st.markdown("---")
        st.markdown("**Model Statistics**")
        st.metric("Log-Likelihood", "-65.9417")
        st.metric("McFadden R²", "0.1189")
        st.metric("AIC", "155.8834")
        st.metric("BIC", "188.0690")
        
        st.markdown("---")
        st.markdown("**Model Formula**")
        st.latex(r"Logit(P) = 1.168 - 12.879 \times \alpha_{clust}")
        st.latex(r"+ 0.294 \times \alpha_{deg} + 5.287 \times \alpha_{plv}")
        st.latex(r"- 0.151 \times \alpha_{path} - 0.223 \times \alpha_{effic}")
        st.latex(r"+ 0.171 \times \beta_{deg} + 3.070 \times \beta_{plv}")
        st.latex(r"- 2.300 \times \beta_{Fp2-O2} - 0.622 \times \gamma_{Fp2-F8}")
        st.latex(r"+ 3.694 \times \gamma_{Fp2-O2} + 1.201 \times \gamma_{F7-O2}")
    
    # Main area - Input tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Single Prediction", "📊 Batch Prediction", "⚙️ Preprocessing Conditions", "📖 Model Info"])
    
    # ==================== Tab 1: Single Prediction ====================
    with tab1:
        st.markdown("### Input EEG Network Features")
        st.markdown("Enter values for all 11 features (range: 0-1)")
        
        # Create 3 columns for input organization
        col1, col2, col3 = st.columns(3)
        
        # Alpha band features
        with col1:
            st.markdown("#### 🧬 Alpha Band Features")
            
            alpha_clust = st.number_input(
                FEATURE_DISPLAY_NAMES['global_alpha_clustering_coefficient'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_alpha_clustering_coefficient', 0.5), 
                step=0.01, format="%.4f",
                key="alpha_clust_input",
                help="Global alpha band clustering coefficient"
            )
            st.session_state['global_alpha_clustering_coefficient'] = alpha_clust
            
            alpha_deg = st.number_input(
                FEATURE_DISPLAY_NAMES['global_alpha_degree_centrality_mean'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_alpha_degree_centrality_mean', 0.5),
                step=0.01, format="%.4f",
                key="alpha_deg_input",
                help="Mean degree centrality in alpha band"
            )
            st.session_state['global_alpha_degree_centrality_mean'] = alpha_deg
            
            alpha_plv = st.number_input(
                FEATURE_DISPLAY_NAMES['global_alpha_mean_plv'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_alpha_mean_plv', 0.5),
                step=0.01, format="%.4f",
                key="alpha_plv_input",
                help="Mean Phase Locking Value in alpha band"
            )
            st.session_state['global_alpha_mean_plv'] = alpha_plv
            
            alpha_path = st.number_input(
                FEATURE_DISPLAY_NAMES['global_alpha_characteristic_path_length'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_alpha_characteristic_path_length', 0.5),
                step=0.01, format="%.4f",
                key="alpha_path_input",
                help="Characteristic path length in alpha band"
            )
            st.session_state['global_alpha_characteristic_path_length'] = alpha_path
            
            alpha_effic = st.number_input(
                FEATURE_DISPLAY_NAMES['global_alpha_global_efficiency'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_alpha_global_efficiency', 0.5),
                step=0.01, format="%.4f",
                key="alpha_effic_input",
                help="Global efficiency in alpha band"
            )
            st.session_state['global_alpha_global_efficiency'] = alpha_effic
        
        # Beta and Gamma features
        with col2:
            st.markdown("#### 🧬 Beta Band Features")
            
            beta_deg = st.number_input(
                FEATURE_DISPLAY_NAMES['global_beta_degree_centrality_mean'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_beta_degree_centrality_mean', 0.5),
                step=0.01, format="%.4f",
                key="beta_deg_input",
                help="Mean degree centrality in beta band"
            )
            st.session_state['global_beta_degree_centrality_mean'] = beta_deg
            
            beta_plv = st.number_input(
                FEATURE_DISPLAY_NAMES['global_beta_mean_plv'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('global_beta_mean_plv', 0.5),
                step=0.01, format="%.4f",
                key="beta_plv_input",
                help="Mean Phase Locking Value in beta band"
            )
            st.session_state['global_beta_mean_plv'] = beta_plv
            
            beta_fp2_o2 = st.number_input(
                FEATURE_DISPLAY_NAMES['beta_Fp2-O2_PLV'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('beta_Fp2-O2_PLV', 0.5),
                step=0.01, format="%.4f",
                key="beta_fp2_o2_input",
                help="Beta band PLV between Fp2 and O2 channels"
            )
            st.session_state['beta_Fp2-O2_PLV'] = beta_fp2_o2
            
            st.markdown("#### 🧬 Gamma Band Features")
            
            gamma_fp2_f8 = st.number_input(
                FEATURE_DISPLAY_NAMES['gamma_Fp2-F8_PLV'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('gamma_Fp2-F8_PLV', 0.5),
                step=0.01, format="%.4f",
                key="gamma_fp2_f8_input",
                help="Gamma band PLV between Fp2 and F8 channels"
            )
            st.session_state['gamma_Fp2-F8_PLV'] = gamma_fp2_f8
            
            gamma_fp2_o2 = st.number_input(
                FEATURE_DISPLAY_NAMES['gamma_Fp2-O2_PLV'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('gamma_Fp2-O2_PLV', 0.5),
                step=0.01, format="%.4f",
                key="gamma_fp2_o2_input",
                help="Gamma band PLV between Fp2 and O2 channels"
            )
            st.session_state['gamma_Fp2-O2_PLV'] = gamma_fp2_o2
        
        with col3:
            st.markdown("#### 🧬 Additional Features")
            
            gamma_f7_o2 = st.number_input(
                FEATURE_DISPLAY_NAMES['gamma_F7-O2_PLV'],
                min_value=0.0, max_value=1.0, value=st.session_state.get('gamma_F7-O2_PLV', 0.5),
                step=0.01, format="%.4f",
                key="gamma_f7_o2_input",
                help="Gamma band PLV between F7 and O2 channels"
            )
            st.session_state['gamma_F7-O2_PLV'] = gamma_f7_o2
            
            st.markdown("#### 🚀 Quick Presets")
            
            if st.button("📊 Average Features", use_container_width=True, key="preset_avg"):
                for feature in FEATURES:
                    st.session_state[feature] = 0.5
                st.rerun()
            
            if st.button("⬆️ High Risk Preset", use_container_width=True, key="preset_high"):
                high_risk_values = {
                    'global_alpha_clustering_coefficient': 0.2,
                    'global_alpha_degree_centrality_mean': 0.7,
                    'global_alpha_mean_plv': 0.8,
                    'global_alpha_characteristic_path_length': 0.7,
                    'global_alpha_global_efficiency': 0.3,
                    'global_beta_degree_centrality_mean': 0.6,
                    'global_beta_mean_plv': 0.7,
                    'beta_Fp2-O2_PLV': 0.3,
                    'gamma_Fp2-F8_PLV': 0.4,
                    'gamma_Fp2-O2_PLV': 0.8,
                    'gamma_F7-O2_PLV': 0.7
                }
                for feature, value in high_risk_values.items():
                    st.session_state[feature] = value
                st.rerun()
            
            if st.button("⬇️ Low Risk Preset", use_container_width=True, key="preset_low"):
                low_risk_values = {
                    'global_alpha_clustering_coefficient': 0.7,
                    'global_alpha_degree_centrality_mean': 0.3,
                    'global_alpha_mean_plv': 0.2,
                    'global_alpha_characteristic_path_length': 0.3,
                    'global_alpha_global_efficiency': 0.7,
                    'global_beta_degree_centrality_mean': 0.4,
                    'global_beta_mean_plv': 0.3,
                    'beta_Fp2-O2_PLV': 0.7,
                    'gamma_Fp2-F8_PLV': 0.6,
                    'gamma_Fp2-O2_PLV': 0.2,
                    'gamma_F7-O2_PLV': 0.3
                }
                for feature, value in low_risk_values.items():
                    st.session_state[feature] = value
                st.rerun()
        
        # Prediction button
        st.markdown("---")
        if st.button("🔍 Predict MCS Probability", type="primary", use_container_width=True, key="predict_btn"):
            # Build features dictionary from session state
            features_dict = {feature: st.session_state.get(feature, 0.5) for feature in FEATURES}
            
            # Calculate prediction
            probability, logit = predict(features_dict)
            
            # Display results
            st.markdown("---")
            st.subheader("📈 Prediction Result")
            
            col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
            
            with col_res1:
                st.metric("MCS Probability", f"{probability:.2%}")
                st.progress(probability)
            
            with col_res2:
                if probability >= THRESHOLD:
                    st.success(f"✅ MCS Positive")
                    st.caption(f"Probability ≥ {THRESHOLD:.1%}")
                else:
                    st.error(f"❌ MCS Negative")
                    st.caption(f"Probability < {THRESHOLD:.1%}")
            
            with col_res3:
                st.metric("Logit Value", f"{logit:.4f}")
            
            # Detailed calculation
            with st.expander("📐 View Detailed Calculation"):
                st.markdown("**Input Features:**")
                for feature in FEATURES:
                    st.write(f"• {FEATURE_DISPLAY_NAMES[feature]}: {features_dict.get(feature, 0.5):.4f}")
                
                st.markdown("**Calculation Steps:**")
                st.write(f"Intercept: {INTERCEPT:.6f}")
                calc_logit = INTERCEPT
                for feature in FEATURES:
                    coef = COEFFICIENTS[feature]
                    val = features_dict.get(feature, 0.5)
                    term = coef * val
                    calc_logit += term
                    st.write(f"+ {coef:.6f} × {val:.4f} = {term:.6f}")
                st.write(f"**Logit(P) = {calc_logit:.6f}**")
                st.write(f"**P(MCS) = 1/(1+e^(-{calc_logit:.6f})) = {probability:.6f} ({probability:.2%})**")
            
            st.info("⚠️ Prediction is for research purposes only. Please consult clinical professionals for medical decisions.")
    
    # ==================== Tab 2: Batch Prediction ====================
    with tab2:
        st.markdown("### Batch Prediction")
        st.markdown("Upload a CSV file with the following 11 feature columns:")
        
        st.code("global_alpha_clustering_coefficient, global_alpha_degree_centrality_mean, global_alpha_mean_plv, global_alpha_characteristic_path_length, global_alpha_global_efficiency, global_beta_degree_centrality_mean, global_beta_mean_plv, beta_Fp2-O2_PLV, gamma_Fp2-F8_PLV, gamma_Fp2-O2_PLV, gamma_F7-O2_PLV")
        
        # Template download
        template_df = pd.DataFrame({feature: [0.5] for feature in FEATURES})
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="mcs_prediction_template.csv",
            mime="text/csv"
        )
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("**Data Preview:**", df.head())
            
            if all(feature in df.columns for feature in FEATURES):
                if st.button("Run Batch Prediction", type="primary", key="batch_predict_btn"):
                    results = []
                    for _, row in df.iterrows():
                        features_dict = {feature: row[feature] for feature in FEATURES}
                        prob, logit = predict(features_dict)
                        pred = "Positive" if prob >= THRESHOLD else "Negative"
                        results.append({"Probability": prob, "Logit": logit, "Prediction": pred})
                    
                    result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
                    st.write("**Prediction Results:**", result_df)
                    
                    # Summary statistics
                    st.markdown("**Summary:**")
                    col_s1, col_s2, col_s3 = st.columns(3)
                    positive_count = sum(1 for r in results if r["Prediction"] == "Positive")
                    with col_s1:
                        st.metric("Positive Cases", f"{positive_count}/{len(results)}")
                    with col_s2:
                        avg_prob = np.mean([r["Probability"] for r in results])
                        st.metric("Average Probability", f"{avg_prob:.2%}")
                    with col_s3:
                        st.metric("Threshold", f"{THRESHOLD:.2%}")
                    
                    csv_result = result_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv_result, "predictions.csv", "text/csv")
            else:
                missing_cols = [col for col in FEATURES if col not in df.columns]
                st.error(f"Missing columns: {missing_cols[:5]}...")
    
    # ==================== Tab 3: Preprocessing Conditions ====================
    with tab3:
        st.markdown("### EEG Preprocessing Conditions")
        st.markdown("The following preprocessing steps are applied to raw EEG data before feature extraction:")
        
        # Create two columns for preprocessing info
        pre_col1, pre_col2 = st.columns(2)
        
        with pre_col1:
            st.markdown("#### 📊 Signal Processing")
            for key, value in list(PREPROCESSING_CONDITIONS.items())[:4]:
                st.markdown(f"**{key}:**")
                st.markdown(f"- {value}")
                st.markdown("")
        
        with pre_col2:
            st.markdown("#### 🔧 Channel Processing")
            for key, value in list(PREPROCESSING_CONDITIONS.items())[4:]:
                st.markdown(f"**{key}:**")
                st.markdown(f"- {value}")
                st.markdown("")
        
        st.markdown("---")
        st.markdown("#### 📝 Detailed Preprocessing Steps")
        st.markdown("""
        **1. Channel Selection**
        - Select EEG channels matching standard 10-20 system
        - Minimum 10 channels required for valid processing
        - Channels are renamed to standardized nomenclature
        
        **2. Resampling**
        - If original sampling rate > 250 Hz, resample to 250 Hz
        - Uses numpy padding for anti-aliasing
        
        **3. Filtering**
        - Bandpass filter: 0.5 - 100 Hz (FIR filter, firwin design)
        - Notch filter: 50 Hz for power line interference removal
        
        **4. Re-referencing**
        - Average reference applied to all channels
        - No projection mode for stability
        
        **5. Channel Standardization**
        - Channels mapped to standard 19-channel montage
        - Unmatched channels are dropped
        - Final output uses standard 10-20 channel names
        """)
        
        st.info("ℹ️ These preprocessing steps ensure data quality and consistency across different EEG recordings.")
    
    # ==================== Tab 4: Model Information ====================
    with tab4:
        st.markdown("### Model Information")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("#### 📊 Model Performance Metrics")
            st.markdown(f"""
            - **AUC:** 0.7118 (95% CI: 0.6161-0.8076)
            - **Accuracy:** 66.67%
            - **Precision:** 63.38%
            - **Recall (Sensitivity):** 81.82%
            - **Specificity:** 50.94%
            - **F1 Score:** 0.7143
            - **Best Threshold:** {THRESHOLD:.4f} (Youden's Index)
            - **Youden's Index:** {YOUDEN_INDEX:.4f}
            """)
        
        with col_info2:
            st.markdown("#### 📈 Model Fit Statistics")
            st.markdown(f"""
            - **Log-Likelihood:** -65.9417
            - **Null Log-Likelihood:** -74.8414
            - **McFadden Pseudo R²:** 0.1189
            - **AIC:** 155.8834
            - **BIC:** 188.0690
            """)
        
        st.markdown("---")
        st.markdown("#### 🧬 Feature Importance")
        
        # Display feature coefficients
        coef_df = pd.DataFrame([
            {"Feature": FEATURE_DISPLAY_NAMES[f], "Coefficient": COEFFICIENTS[f], "Effect": "Positive" if COEFFICIENTS[f] > 0 else "Negative"}
            for f in FEATURES
        ])
        coef_df["Absolute"] = coef_df["Coefficient"].abs()
        coef_df = coef_df.sort_values("Absolute", ascending=False)
        
        st.dataframe(coef_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 📐 Model Formula")
        st.latex(r"Logit(P) = 1.168 - 12.879 \times \alpha_{clust} + 0.294 \times \alpha_{deg} + 5.287 \times \alpha_{plv}")
        st.latex(r"- 0.151 \times \alpha_{path} - 0.223 \times \alpha_{effic} + 0.171 \times \beta_{deg}")
        st.latex(r"+ 3.070 \times \beta_{plv} - 2.300 \times \beta_{Fp2-O2} - 0.622 \times \gamma_{Fp2-F8}")
        st.latex(r"+ 3.694 \times \gamma_{Fp2-O2} + 1.201 \times \gamma_{F7-O2}")
        st.latex(r"P(MCS) = \frac{1}{1 + e^{-Logit(P)}}")
        
        st.markdown("---")
        st.markdown("#### 📖 Interpretation Guide")
        st.markdown("""
        - **MCS Positive:** Probability ≥ 0.4502 (Youden's optimal threshold)
        - **MCS Negative:** Probability < 0.4502
        - **Sensitivity 81.82%:** Model correctly identifies 81.82% of MCS patients
        - **Specificity 50.94%:** Model correctly identifies 50.94% of non-MCS patients
        
        **Clinical Note:** This prediction model is intended for research purposes only. Clinical decisions should be made by qualified healthcare professionals based on comprehensive patient evaluation.
        """)
    
    # Footer
    st.markdown("---")
    st.caption("© 2025 MCS Prediction System | Logistic Regression Model | Version 2.0 | For Research Use Only")

if __name__ == "__main__":
    main()
streamlit
numpy
