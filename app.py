import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Rent Prediction App")

st.title("🏠 House Rent Prediction")

# Load model safely
try:
    data = joblib.load("rent_model.joblib")
    model = data["model"]
    le_location = data["le_location"]
    le_furnishing = data["le_furnishing"]
except:
    st.error("❌ Model file not found! Run train_model.py first.")
    st.stop()

# Inputs
area = st.number_input("Area (sqft)", 300, 5000, 1000)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
parking = st.selectbox("Parking", [0, 1])

location = st.selectbox("Location", le_location.classes_)
furnishing = st.selectbox("Furnishing", le_furnishing.classes_)

# Encode
location_encoded = le_location.transform([location])[0]
furnishing_encoded = le_furnishing.transform([furnishing])[0]

# Predict
if st.button("Predict Rent"):
    features = np.array([[area, bedrooms, bathrooms, parking, location_encoded, furnishing_encoded]])
    prediction = model.predict(features)[0]

    st.success(f"💰 Estimated Rent: ₹ {int(prediction)}")