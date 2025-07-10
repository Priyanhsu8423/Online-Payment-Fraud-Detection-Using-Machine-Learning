
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("💳 Online Payment Fraud Detection")
st.write("Enter transaction details to check if it's fraud or not.")

# Input form
with st.form("fraud_form"):
    step = st.number_input("Step (Time)", min_value=0.0)
    type_ = st.selectbox("Transaction Type", ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"])
    amount = st.number_input("Amount", min_value=0.0)
    oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0)
    newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0)
    oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0)
    newbalanceDest = st.number_input("New Balance Destination", min_value=0.0)
    submitted = st.form_submit_button("Check Fraud")

if submitted:
    # One-hot encode transaction type manually
    type_mapping = {"CASH_OUT": [1,0,0,0,0], "TRANSFER": [0,1,0,0,0],
                    "PAYMENT": [0,0,1,0,0], "DEBIT": [0,0,0,1,0], "CASH_IN": [0,0,0,0,1]}
    type_encoded = type_mapping[type_]
    
    # Create input row
    input_data = [step] + type_encoded + [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
    input_df = pd.DataFrame([input_data], columns=[
        "step", "type_CASH_OUT", "type_TRANSFER", "type_PAYMENT", "type_DEBIT", "type_CASH_IN",
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"
    ])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Fraud Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction is Safe (Probability: {prob:.2f})")
