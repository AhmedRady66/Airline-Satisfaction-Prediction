import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ----------------------------
# Load saved objects
# ----------------------------
with open("randomforest.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

with open("features_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

with open("mappings.pkl", "rb") as f:
    mappings = pickle.load(f)


# -----------------------------
# Streamlit app
# -----------------------------
st.title("Airline Satisfaction Prediction ✈️")

st.write("Please enter the following information:")

# Create input fields
Gender = st.selectbox("Gender", options=list(mappings['Gender'].keys()))
Customer_Type = st.selectbox("Customer Type", options=list(mappings['Customer Type'].keys()))
Age = st.number_input("Age", min_value=0, max_value=120, value=30)
Type_of_Travel = st.selectbox("Type of Travel", options=list(mappings['Type of Travel'].keys()))
Class = st.selectbox("Class", options=list(mappings['Class'].keys()))
Flight_Distance = st.number_input("Flight Distance", min_value=0, value=500)
Inflight_wifi_service = st.slider("Inflight wifi service", 0, 5, 3)
Departure_Arrival_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)
Ease_of_Online_booking = st.slider("Ease of Online booking", 0, 5, 3)
Gate_location = st.slider("Gate location", 0, 5, 3)
Food_and_drink = st.slider("Food and drink", 0, 5, 3)
Online_boarding = st.slider("Online boarding", 0, 5, 3)
Seat_comfort = st.slider("Seat comfort", 0, 5, 3)
Inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)
On_board_service = st.slider("On-board service", 0, 5, 3)
Leg_room_service = st.slider("Leg room service", 0, 5, 3)
Baggage_handling = st.slider("Baggage handling", 0, 5, 3)
Checkin_service = st.slider("Checkin service", 0, 5, 3)
Inflight_service = st.slider("Inflight service", 0, 5, 3)
Cleanliness = st.slider("Cleanliness", 0, 5, 3)
Departure_Delay_in_Minutes = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
Arrival_Delay_in_Minutes = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Satisfaction"):

    # 1️⃣ Prepare input data as DataFrame
    input_data = pd.DataFrame({
        "Gender": [mappings['Gender'][Gender]],
        "Customer Type": [mappings['Customer Type'][Customer_Type]],
        "Age": [Age],
        "Type of Travel": [mappings['Type of Travel'][Type_of_Travel]],
        "Class": [mappings['Class'][Class]],
        "Flight Distance": [Flight_Distance],
        "Inflight wifi service": [Inflight_wifi_service],
        "Departure/Arrival time convenient": [Departure_Arrival_time_convenient],
        "Ease of Online booking": [Ease_of_Online_booking],
        "Gate location": [Gate_location],
        "Food and drink": [Food_and_drink],
        "Online boarding": [Online_boarding],
        "Seat comfort": [Seat_comfort],
        "Inflight entertainment": [Inflight_entertainment],
        "On-board service": [On_board_service],
        "Leg room service": [Leg_room_service],
        "Baggage handling": [Baggage_handling],
        "Checkin service": [Checkin_service],
        "Inflight service": [Inflight_service],
        "Cleanliness": [Cleanliness],
        "Departure Delay in Minutes": [Departure_Delay_in_Minutes],
        "Arrival Delay in Minutes": [Arrival_Delay_in_Minutes]
    })

    # 2️⃣ Ensure feature order
    input_data = input_data[feature_order]

    # 3️⃣ Scale numeric features
    input_scaled = scaler.transform(input_data)

    # 4️⃣ Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][prediction]

    # 5️⃣ Show result
    result = "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"
    st.success(
    f"Prediction: {result}\n"
    f"\nPrediction Probability: {probability*100:.2f}%"
)

