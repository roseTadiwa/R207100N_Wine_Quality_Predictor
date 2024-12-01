import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('wine_quality_model.pkl')

# Title of the app
st.title("R207100N : Wine Quality Prediction")

# Display your registration number
registration_number = "R207100N"  # Replace with your actual registration number
st.write(f"**Registration Number:** {registration_number}")

# User inputs for wine characteristics
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, max_value=14.0, value=7.0)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, max_value=1.6, value=0.3)
citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.2)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, max_value=65.8, value=2.0)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=0.6, value=0.05)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, max_value=50.0, value=15.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, max_value=300.0, value=50.0)
density = st.number_input("Density", min_value=0.99, max_value=1.04, value=1.0)
pH = st.number_input("pH", min_value=2.0, max_value=4.0, value=3.0)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, value=0.5)
alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.0)
# Input for the quality feature
quality = st.number_input("Quality", min_value=0, max_value=10, value=5)  # Adjust range as necessary


# Button to predict
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol],
        'quality': [quality] 
    })

    # Perform prediction
    prediction = model.predict(input_data)
    
    # Display the result
    wine_type = "White Wine" if prediction[0] == 0 else "Red Wine"
    st.write(f"The predicted wine type is: **{wine_type}**")