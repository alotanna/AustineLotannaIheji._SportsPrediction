# -*- coding: utf-8 -*-
"""streamlit_app.py"""

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load("main_final_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Player Rating Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: #ffffff;
    }
    .css-17eq0hr a { 
        color: #ffffff;
        text-decoration: none;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    .stNumberInput>div>input {
        background-color: #e9ecef;
        color: #495057;
    }
    .stSlider>div>div>div>div {
        background-color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

# Header and description
st.title("⚽ Player Rating Predictor")
st.markdown("Use this app to predict the rating of a player based on their attributes.")

# Sidebar for input
st.sidebar.header("Player Attributes Input")
st.sidebar.markdown("Enter the attributes to get a predicted player rating.")

def usersinput():
    with st.sidebar.form(key='user_input_form'):
        attribute1 = st.slider("Movement Reactions", min_value=0, max_value=100, value=50, step=1)
        attribute2 = st.slider("Potential", min_value=0, max_value=100, value=50, step=1)
        attribute3 = st.number_input("Wage (EUR)", min_value=0, value=5000, step=1000)
        attribute4 = st.slider("Shot Power", min_value=0, max_value=100, value=50, step=1)
        attribute5 = st.number_input("Value (EUR)", min_value=0, value=1000000, step=100000)
        attribute6 = st.slider("Passing Accuracy", min_value=0, max_value=100, value=50, step=1)
        attribute7 = st.slider("Mentality Vision", min_value=0, max_value=100, value=50, step=1)
        attribute8 = st.slider("International Reputation", min_value=1, max_value=5, value=1, step=1)
        attribute9 = st.slider("Long Passing", min_value=0, max_value=100, value=50, step=1)
        attribute10 = st.slider("Physic", min_value=0, max_value=100, value=50, step=1)
        attribute11 = st.slider("Ball Control", min_value=0, max_value=100, value=50, step=1)

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
        st.error(f"Error occurred during prediction: {str(e)}")
else:
    st.warning("Please enter the player attributes and click 'Predict Rating'")

# Footer
st.markdown(
    """
    <div style='text-align: center; padding-top: 2rem;'>
        <hr>
        <p>&copy; 2024 Player Rating Predictor. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
