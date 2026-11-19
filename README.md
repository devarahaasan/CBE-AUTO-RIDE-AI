# 🛺 Coimbatore Auto-RideFlow AI: End-to-End Intelligent Ride Optimization System (ML & DL)

[![Open in Streamlit](https://streamlit.io)](https://streamlit.io)
![Python](https://shields.io)
![Scikit-Learn](https://shields.io)
![TensorFlow](https://shields.io)

### 🚀 Direct One-Click Deployment & Live App Link
* **Instant Cloud Deploy Link:** [Click Here to Deploy Directly on Streamlit Cloud](https://cbe-auto-ride-ai-zfwkca7urafopmrond3syk.streamlit.app/)
* **Live App URL (After Deploying):** `https://streamlit.app` *(Replace with your active `.streamlit.app` URL once deployed)*

---

## 📌 Project Overview
**Coimbatore Auto-RideFlow AI** is an end-to-end Machine Learning and Deep Learning system built to optimize urban auto-rickshaw dispatch in Coimbatore. Grounded in real-world street-level mobility insights—combining firsthand on-ground delivery experience and local auto-rickshaw operations—the system predicts **Real-Time ETA**, **Dynamic Surge Pricing**, and **Ride Cancellation Risk** through a chained AI pipeline.

---

## 🏗️ System Architecture (Model Chaining Pipeline)
1. **Deep Learning ETA Prediction (`LSTM`):** Uses route distance, demand, driver supply, weather, and hour of day to predict arrival time (`ETA_minutes`).
2. **Dynamic Surge Pricing (`Tuned Random Forest Regressor`):** Takes the predicted ETA alongside `Demand_Supply_Gap` and zone encodings to calculate surge multipliers (`1.0x – 2.2x`) and final fares.
3. **Cancellation Risk Prediction (`Tuned Logistic Regression`):** Evaluates tariff dispute status (`Driver_Asks_Extra`), predicted ETA, surge multiplier, and weather conditions to flag high-risk dispatches before cancellation occurs.

---

## 📊 Key Results & Model Performance

| Task | Model Selected | Primary Metric | Validation / Stability |
| :--- | :--- | :--- | :--- |
| **ETA Prediction (DL)** | **LSTM Neural Network** | **1.03 mins MAE** | Smooth convergence across 8 epochs with `Dropout(0.2)` |
| **Surge Pricing (ML Regression)** | **Random Forest Regressor** *(Tuned)* | **0.9622 R²** (`0.0383 MAE`) | Benchmarked against 6 regression algorithms |
| **Cancellation Risk (ML Classification)** | **Logistic Regression** *(Tuned)* | **91.32% Accuracy** (`0.9309 F1`) | **0.9728 ROC-AUC** & **91.02% 5-Fold CV** |

---

## 📂 Repository Structure
```text
├── rideflow_app.py              # Main Streamlit web application
├── surge_rf_model.pkl           # Tuned Random Forest Regressor (Surge Multiplier)
├── cancel_log_model.pkl         # Tuned Logistic Regression Classifier (Cancellation Risk)
├── ml_feature_scaler.pkl        # Fitted StandardScaler for ML inference
├── rideflow_features.pkl        # Exact One-Hot Encoded feature schema
├── eta_lstm_model.h5            # Trained TensorFlow/Keras LSTM model (ETA Prediction)
├── eta_lstm_scaler.pkl          # Fitted StandardScaler for LSTM inference
├── requirements.txt             # Python dependencies for Streamlit Cloud
└── README.md                    # Project documentation
```

---

## 💻 Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd YOUR-REPO-NAME
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the Streamlit app:**
   ```bash
   streamlit run rideflow_app.py
   ```
