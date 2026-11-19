# ==============================================================================
# STEP 10: STREAMLIT WEB APPLICATION (FINAL DEPLOYMENT)
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime

st.set_page_config(page_title="Coimbatore RideFlow AI", page_icon="🛺", layout="wide")

@st.cache_resource
def load_models():
    surge_model = joblib.load('surge_rf_model.pkl')
    cancel_model = joblib.load('cancel_log_model.pkl')
    ml_scaler = joblib.load('ml_feature_scaler.pkl')
    feature_cols = joblib.load('rideflow_features.pkl')
    eta_model = tf.keras.models.load_model('eta_lstm_model.h5', compile=False)
    eta_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    dl_scaler = joblib.load('eta_lstm_scaler.pkl')
    return surge_model, cancel_model, ml_scaler, feature_cols, eta_model, dl_scaler

try:
    reg_model, clf_model, ml_scaler, all_columns, lstm_model, dl_scaler = load_models()
    model_status = "🟢 ALL MODELS LOADED SUCCESSFULLY"
except Exception as e:
    model_status = f"🔴 ERROR LOADING MODELS: {str(e)}"
    reg_model, clf_model, ml_scaler, all_columns, lstm_model, dl_scaler = None, None, None, None, None, None

# Sidebar Controls
st.sidebar.markdown(f"### **System Status:**\n`{model_status}`")
st.sidebar.header("🛺 Dispatch Control Panel")

zones = ['Saravanampatti', 'Gandhipuram', 'Peelamedu', 'RS Puram', 'Ukkadam']
pickup = st.sidebar.selectbox("Select Pickup Zone", zones, index=0)
drop = st.sidebar.selectbox("Select Drop Zone", zones, index=1)

current_hour = st.sidebar.slider("Hour of Day (24-Hr Format)", 0, 23, datetime.now().hour)
distance = st.sidebar.slider("Estimated Route Distance (km)", 1.0, 25.0, 6.0, step=0.5)
demand = st.sidebar.slider("Local Demand Score (1-10)", 1, 10, 7)
supply = st.sidebar.slider("Available Auto Drivers (1-10)", 1, 10, 4)

rain_input = st.sidebar.selectbox("Weather Condition", ["Clear Sky", "Heavy Rain Storm"])
fest_input = st.sidebar.selectbox("Day Classification", ["Normal Day", "Festival Day"])
extra_input = st.sidebar.selectbox("Driver Tariff Dispute Status", ["No Dispute", "Driver Asks Extra Cash"])

is_raining = 1.0 if rain_input == "Heavy Rain Storm" else 0.0
is_festival = 1.0 if fest_input == "Festival Day" else 0.0
asks_extra = 1.0 if extra_input == "Driver Asks Extra Cash" else 0.0
current_day = datetime.now().strftime('%A')

# Main Dashboard
st.title("🛺 Coimbatore RideFlow AI Operations Room")
st.markdown("### **End-to-End Intelligent Ride Optimization System (ML + DL)**")
st.divider()

if "🟢" in model_status:
    col1, col2, col3, col4 = st.columns(4)

    # Prepare Feature Dictionary
    input_data = {col: 0.0 for col in all_columns}
    input_data['Hour_of_Day'] = float(current_hour)
    input_data['Distance_km'] = float(distance)
    input_data['Demand_Score'] = float(demand)
    input_data['Supply_Score'] = float(supply)
    input_data['Demand_Supply_Gap'] = float(demand - supply)
    input_data['Is_Raining'] = is_raining
    input_data['Is_Festival'] = is_festival
    input_data['Driver_Asks_Extra'] = asks_extra

    if f"Pickup_Zone_{pickup}" in input_data: input_data[f"Pickup_Zone_{pickup}"] = 1.0
    if f"Drop_Zone_{drop}" in input_data: input_data[f"Drop_Zone_{drop}"] = 1.0
    if f"Day_of_Week_{current_day}" in input_data: input_data[f"Day_of_Week_{current_day}"] = 1.0

    X_query = pd.DataFrame([input_data])[all_columns]

    # Step A: Deep Learning LSTM ETA Prediction
    dl_cols = ['Distance_km', 'Demand_Score', 'Supply_Score', 'Is_Raining', 'Hour_of_Day']
    dl_features = pd.DataFrame([[distance, demand, supply, is_raining, current_hour]], columns=dl_cols)
    dl_features_scaled = dl_scaler.transform(dl_features)
    X_lstm_query = np.reshape(dl_features_scaled, (1, 1, 5))

    predicted_eta = max(1.0, round(float(lstm_model.predict(X_lstm_query, verbose=0).item()), 1))
    X_query['ETA_minutes'] = predicted_eta

    # Step B: Machine Learning Surge & Cancellation Prediction
    predicted_surge = max(1.0, min(2.2, round(float(reg_model.predict(X_query).item()), 2)))

    if distance <= 2:
        base_fare = 40.0
    elif distance <= 10:
        base_fare = 40.0 + (distance - 2) * 15.0
    else:
        base_fare = 40.0 + (8 * 15.0) + (distance - 10) * 14.0

    if (current_hour >= 22) or (current_hour < 6): base_fare *= 1.5
    if is_raining == 1.0: base_fare += 50.0
    final_computed_fare = round(base_fare * predicted_surge, 2)

    # Scale X_query for Logistic Regression Cancellation Model
    X_query_scaled = ml_scaler.transform(X_query)
    cancellation_probability = float(clf_model.predict_proba(X_query_scaled)[:, 1].item())

    # Display Metrics
    with col1:
        st.metric(label="⏱️ LSTM Predicted ETA", value=f"{predicted_eta} Mins")
    with col2:
        st.metric(label="📈 Dynamic Surge Multiplier", value=f"{predicted_surge}x")
    with col3:
        st.metric(label="💰 Final Computed Fare", value=f"₹{final_computed_fare}")
    with col4:
        st.metric(label="🚨 Cancellation Risk", value=f"{round(cancellation_probability * 100, 1)}%")

    st.divider()
    st.subheader("🤖 Fleet Predictive Analytics Dispatch Summary")
    st.progress(min(1.0, max(0.0, cancellation_probability)))
    st.markdown(f"**Route Vector:** `{pickup}` ➡️ `{drop}` | **Time:** `{current_day}, {current_hour}:00 hrs`")

    if cancellation_probability > 0.60:
        st.error("🚨 HIGH DISPATCH RISK: Elevated risk of driver cancellation due to tariff dispute, high ETA, or peak demand.")
    else:
        st.success("✨ OPTIMAL ROUTE: Low cancellation probability. High ride completion rate expected.")
else:
    st.error("Please ensure all model files (.pkl, .h5) are present in the repository root.")
