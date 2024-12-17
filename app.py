import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("customer_data.csv")

# Preprocess the data
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})  # Convert target variable to numeric
df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to dummy variables

# Split the data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)  # Slider for tenure
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, format="%.2f")
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'gender_Male': [1 if gender == "Male" else 0],
    'SeniorCitizen': [senior_citizen],
    'Partner_Yes': [1 if partner == "Yes" else 0],
    'Dependents_Yes': [1 if dependents == "Yes" else 0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'DeviceProtection_Yes': [1 if device_protection == "Yes" else 0],
    'DeviceProtection_No internet service': [1 if device_protection == "No internet service" else 0],
})

# Ensure all features are present
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns with default value 0

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")