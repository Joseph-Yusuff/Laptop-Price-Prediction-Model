import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ----------------------------- #
# 🚀 Load Model & Data
# ----------------------------- #

# Load the pre-trained pipeline (includes preprocessing + model)
pipeline = joblib.load('pipeline.joblib')

# Load dataset to extract unique values for UI components
data = pd.read_csv("traineddata.csv")

# ----------------------------- #
# 🎛️ Sidebar Inputs
# ----------------------------- #

st.sidebar.header("💻 Laptop Specifications")

company = st.sidebar.selectbox('Brand', sorted(data['Company'].unique()))
type_name = st.sidebar.selectbox('Type', sorted(data['TypeName'].unique()))
ram = st.sidebar.selectbox('RAM (in GB)', sorted([2, 4, 6, 8, 12, 16, 24, 32, 64]))
os = st.sidebar.selectbox('Operating System', sorted(data['OpSys'].unique()))
weight = st.sidebar.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.sidebar.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.sidebar.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.sidebar.number_input('Screen Size (in inches)', min_value=10.0, max_value=20.0, step=0.1)
resolution = st.sidebar.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', 
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.sidebar.selectbox('CPU', sorted(data['CPU_name'].unique()))
hdd = st.sidebar.selectbox('HDD (in GB)', sorted([0, 128, 256, 512, 1024, 2048]))
ssd = st.sidebar.selectbox('SSD (in GB)', sorted([0, 8, 128, 256, 512, 1024]))
gpu = st.sidebar.selectbox('GPU Brand', sorted(data['Gpu brand'].unique()))

# ----------------------------- #
# 💻 Main Title
# ----------------------------- #

st.title("💻 Laptop Price Predictor")

# ----------------------------- #
# 🔧 Feature Engineering
# ----------------------------- #

touchscreen = 1 if touchscreen == 'Yes' else 0
ips = 1 if ips == 'Yes' else 0
X_res, Y_res = map(int, resolution.split('x'))
ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size if screen_size > 0 else 0

input_features = pd.DataFrame([[
    company, type_name, ram, weight, touchscreen, ips, ppi,
    cpu, hdd, ssd, gpu, os
]], columns=[
    'Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen', 'IPS', 'PPI',
    'CPU_name', 'HDD', 'SSD', 'Gpu brand', 'OpSys'
])

# ----------------------------- #
# 🎯 Predict Button & Output
# ----------------------------- #

if st.button("🎯 Predict Price"):
    # Predict the log price and inverse it
    prediction = pipeline.predict(input_features)
    std_error = 0.15  # Mean Absolute error in log units

    # Extract the log price and convert to the actual price
    log_price = prediction[0]
    predicted_price = np.exp(log_price)

    # Calculate the confidence interval
    lower_bound = int(np.exp(log_price - std_error))
    upper_bound = int(np.exp(log_price + std_error))

    # Display the actual predicted price
    st.markdown("### 💰 Estimated Laptop Price")
    st.success(f"£{predicted_price:.2f}")  # Format the price to 2 decimal places

    # Display the confidence interval
    st.markdown(f"**Confidence Interval:**")
    st.caption(f"Based on model prediction ±0.15 log-units, the estimated price range is £{lower_bound} - £{upper_bound}.")


# ----------------------------- #
# 📊 SHAP Explanation
# ----------------------------- #

# Divider and section header
st.markdown("---")
st.subheader("🧠 SHAP Feature Explanation")

# ✅ Cache the SHAP explainer to avoid recomputation
@st.cache_resource
def get_shap_explainer(_pipeline):
    """
    Cached SHAP explainer that works with full sklearn pipeline.
    The underscore prevents Streamlit from trying to hash the pipeline.
    """
    xgb_model = _pipeline.named_steps['model'].named_estimators_['xgb']
    booster_model = xgb_model.get_booster()
    return shap.Explainer(booster_model)

# User wants to see SHAP insights
if st.checkbox("📊 Explain Prediction with SHAP"):
    
    # Let user pick which SHAP plot to view
    plot_choice = st.radio("Choose SHAP Visualization:", ["Bar Chart", "Waterfall Plot"])
    # Divider for clarity
    st.markdown("---")
    try:
        # Extract pipeline steps
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['model']

        # Feature names used in training
        categorical_cols = ['Company', 'TypeName', 'OpSys', 'CPU_name', 'Gpu brand']
        numerical_cols = ['Ram', 'Weight', 'TouchScreen', 'IPS', 'PPI', 'HDD', 'SSD']
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_feature_names = numerical_cols + list(cat_features)

        # Initialize cached explainer
        explainer = get_shap_explainer(pipeline)

        # === GLOBAL BAR CHART ===
        if plot_choice == "Bar Chart":
            st.markdown("### 🌍 Global Feature Importance (Bar Chart)")
            st.markdown("""
This chart shows the **average impact of each feature** on laptop prices across all data:
- 📈 The longer the bar, the more it affects predictions.
- 🔴 Red = increases price, 🔵 Blue = decreases price.
            """)

            # Load training data and remove target
            raw_data = pd.read_csv("traineddata.csv")
            X_train_raw = raw_data.drop(columns='Price_GBP', errors='ignore')

            # Transform and rebuild as DataFrame with correct feature names
            X_train_processed = preprocessor.transform(X_train_raw)
            X_train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)

            # Get SHAP values for training data
            shap_values_train = explainer(X_train_df)

            # Display bar chart
            fig_bar, ax = plt.subplots(figsize=(9, 5))
            shap.plots.bar(shap_values_train, max_display=12, show=False)
            st.pyplot(fig_bar)

        # === LOCAL WATERFALL EXPLANATION ===
        elif plot_choice == "Waterfall Plot":
            st.markdown("### 🌊 Waterfall Plot for Your Laptop")
            st.markdown("""
This explains **how each feature of your selected laptop** influenced the predicted price:
- 🔴 Red = feature increases the price
- 🔵 Blue = feature decreases the price
- It starts from the average laptop price and adjusts based on each feature
            """)

            # Transform and prepare the user's input
            input_processed = preprocessor.transform(input_features)
            input_df = pd.DataFrame(input_processed, columns=all_feature_names)

            # Compute SHAP values for the single input
            shap_values = explainer(input_df)

            # Plot waterfall chart
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], max_display=15,  show=False)
            st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Could not generate SHAP plot: {str(e)}")