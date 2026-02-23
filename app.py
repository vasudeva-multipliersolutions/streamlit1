import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("🏦 Customer Churn Prediction App")
st.write("Enter customer details to predict churn")

# Load model and encoders
model = load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("le.pkl", "rb") as f:
    le = pickle.load(f)

with open("one_hot_encoder_geo.pkl", "rb") as f:
    one_hot_encoder_geo = pickle.load(f)

# ---------------- UI INPUT ----------------
credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)
num_products = st.slider("Number of Products", 1, 4, 2)
has_card = st.selectbox("Has Credit Card", [1, 0])
is_active = st.selectbox("Is Active Member", [1, 0])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# ------------- PREDICT BUTTON -------------
if st.button("Predict Churn"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'CreditScore':[credit_score],
        'Geography':[geography],
        'Gender':[gender],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_products],
        'HasCrCard':[has_card],
        'IsActiveMember':[is_active],
        'EstimatedSalary':[salary]
    })

    # Encode gender
    input_data['Gender'] = le.transform(input_data['Gender'])

    # One hot encode geography
    geo_encoded = one_hot_encoder_geo.transform(input_data[['Geography']]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Combine
    input_data = pd.concat([input_data.drop("Geography", axis=1), geo_df], axis=1)

    # Add missing columns
    for col in scaler.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # Correct order
    input_data = input_data[scaler.feature_names_in_]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error(f"⚠️ Customer likely to churn (Probability: {prediction:.2f})")
    else:
        st.success(f"✅ Customer not likely to churn (Probability: {prediction:.2f})")
