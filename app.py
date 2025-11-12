import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model, label encoder, and scaler
@st.cache_resource
def load_assets():
    model = pickle.load(open("Best_lgbm_model.pkl", "rb"))
    label_encoder = pickle.load(open("LabelEncoder.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, label_encoder, scaler

model, label_encoder, scaler = load_assets()


def main():
    # --- Sidebar Inputs ---
    st.sidebar.title("Machine Sensor Inputs")

    air_temp = st.sidebar.number_input("Air Temperature [K]", value=310.0)
    process_temp = st.sidebar.number_input("Process Temperature [K]", value=320.0)
    rotational_speed = st.sidebar.number_input("Rotational Speed [rpm]", value=1450.0)
    torque = st.sidebar.number_input("Torque [Nm]", value=45.0)
    tool_wear = st.sidebar.number_input("Tool Wear [min]", value=180.0)
    machine_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"], index=1)

    # --- Derived Features ---
    temp_diff = process_temp - air_temp
    power = rotational_speed * torque

    # Normalize safely for progress bar
    temp_diff_norm = int(min(abs(temp_diff) / 150 * 100, 100))
    power_norm = int(min(power / 300000 * 100, 100))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Derived Features (Auto-Calculated)")
    st.sidebar.write(f"Temperature Difference [K]: **{temp_diff:.2f}**")
    st.sidebar.progress(temp_diff_norm)

    st.sidebar.write(f"Power [rpm·Nm]: **{power:.2f}**")
    st.sidebar.progress(power_norm)

    # Encode type
    encoded_type = label_encoder.transform([machine_type])[0]

    # Prepare input data
    input_data = pd.DataFrame({
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear],
        "Type": [encoded_type],
        "Temp_Diff": [temp_diff],
        "Power": [power]
    })

    # Align with scaler
    try:
        expected_features = scaler.feature_names_in_
        input_data = input_data.reindex(columns=expected_features)
    except AttributeError:
        pass

    # --- Main UI ---
    st.title("Predictive Maintenance Dashboard")
    st.markdown("""
    This system predicts whether a machine is likely to **fail** or **operate normally**  
    based on real-time sensor readings and operational parameters.
    """)

    st.markdown("### Input Overview")
    st.dataframe(input_data, use_container_width=True)
    st.markdown("---")

    if st.button("Run Failure Prediction", use_container_width=True):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        probabilities = model.predict_proba(scaled_input)[0]

        p_normal, p_failure = probabilities[0], probabilities[1]

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f" Machine Failure Detected\nFailure Probability: {p_failure:.4f}")
        else:
            st.success(f" Machine Operating Normally\nNormal Probability: {p_normal:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal Probability", f"{p_normal:.2%}")
        with col2:
            st.metric("Failure Probability", f"{p_failure:.2%}")

        st.info("Monitor torque and temperature differences closely — large variations can signal early wear or stress.")

    # --- Footer ---
    st.markdown("""
    ---
    **Developed by: Team PrediTech**  
    **Model:** Optimized LightGBM  
    **Purpose:** Predictive Maintenance for Industrial Equipment  
    """)


if __name__ == "__main__":
    main()
