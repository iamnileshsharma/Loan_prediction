# streamlit_app.py

import streamlit as st
import joblib
import numpy as np

from Scripts.predict import predict_loan

# Load trained model and scaler
model = joblib.load('Models/model.pkl')
scaler = joblib.load('Models/scaler.pkl')

st.title("🏦 Loan Approval Predictor")
st.markdown("Enter applicant information below to predict loan status.")

# Define input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self-Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term (in days)", min_value=0.0)
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])

# Map inputs to numerical values
input_data = [
    1 if gender == 'Male' else 0,
    1 if married == 'Yes' else 0,
    3 if dependents == '3+' else int(dependents),
    1 if education == 'Graduate' else 0,
    1 if self_employed == 'Yes' else 0,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[property_area]
]

# Predict on button click
if st.button("Predict Loan Status"):
    result = predict_loan(model, scaler, [input_data])
    st.success("✅ Loan Approved" if result[0] == 1 else "❌ Loan Rejected")
