# -*- coding: utf-8 -*-
"""streamlit_app.py"""

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load("main_final_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title of the app
st.title("Player Rating Predictor")
st.markdown(
    """
    <style>
    .main { 
        background-color: #f5f5f5; 
        color: #333333; 
        font-family: Arial, sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

st.header("Enter Player Attributes")

def usersinput():
    with st.form(key='user_input_form'):
        col1, col2, col3 = st.columns(3)

        with col1:
            attribute1 = st.number_input("Movement Reactions", min_value=0, max_value=100, step=1)
            attribute2 = st.number_input("Potential", min_value=0, max_value=100, step=1)
            attribute3 = st.number_input("Wage (EUR)", min_value=0, step=1000)
            attribute4 = st.number_input("Shot Power", min_value=0, max_value=100, step=1)

        with col2:
            attribute5 = st.number_input("Value (EUR)", min_value=0, step=1000)
            attribute6 = st.number_input("Passing Accuracy", min_value=0, max_value=100, step=1)
            attribute7 = st.number_input("Mentality Vision", min_value=0, max_value=100, step=1)
            attribute8 = st.number_input("International Reputation", min_value=1, max_value=5, step=1)

        with col3:
            attribute9 = st.number_input("Long Passing", min_value=0, max_value=100, step=1)
            attribute10 = st.number_input("Physic", min_value=0, max_value=100, step=1)
            attribute11 = st.number_input("Ball Control", min_value=0, max_value=100, step=1)

        submit_button = st.form_submit_button(label='Predict Rating')

    if submit_button:
        dict = {
            'movement_reactions': attribute1,
            'potential': attribute2,
            'wage_eur': attribute3,
            'power_shot_power': attribute4,
            'value_eur': attribute5,
            'passing': attribute6,
            'mentality_vision': attribute7,
            'international_reputation': attribute8,
            'skill_long_passing': attribute9,
            'physic': attribute10,
            'skill_ball_control': attribute11
        }
        df = pd.DataFrame(dict, index=[0])
        return df
    else:
        return None

userinput = usersinput()

if userinput is not None:
    st.subheader('User Input Parameters')
    st.write(userinput)

    try:
        scaled_dataframe = scaler.transform(userinput)

        # Predict the rating using the trained model
        pred = model.predict(scaled_dataframe)

        st.subheader('Prediction')
        st.success(f"The predicted rating is: {pred[0]:.2f}")

    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    st.warning("Please enter the player attributes and click 'Predict Rating'")

