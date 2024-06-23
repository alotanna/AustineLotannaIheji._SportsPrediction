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
    /* Main layout styling */
    .main {
        background-color: #f8f9fa; /* Light grey background */
        font-family: 'Arial', sans-serif;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: #ffffff;
    }
    .sidebar .sidebar-content h2 {
        color: #ffcc00;
    }
    /* Input elements styling */
    .stNumberInput>div>input,
    .stSlider>div>div>div>div,
    .stTextInput>div>input {
        border-radius: 5px;
        background-color: #e9ecef;
        color: #495057;
    }
    /* Button styling */
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    /* Table styling */
    .css-1d391kg {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .css-1d391kg table {
        width: 100%;
        text-align: center;
        background-color: #ffffff;
    }
    /* Prediction output styling */
    .prediction-box {
        background-color: #007bff;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .prediction-box h1 {
        margin: 0;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        font-size: 14px;
        color: #868e96;
    }
    .stSlider>div>div>div>div {
        background-color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

# Header and description
st.title("⚽ Player Rating Predictor")
st.markdown("### Predict a player's rating using advanced machine learning. Just enter the player's attributes and get an instant prediction!")

# Sidebar for input
st.sidebar.header("Enter Player Attributes")
st.sidebar.markdown("Fill in the details below to get a predicted rating for the player.")

# Input form
def usersinput():
    with st.sidebar.form(key='user_input_form'):
        st.markdown("#### Core Attributes")
        attribute1 = st.slider("Movement Reactions", min_value=0, max_value=100, value=50, step=1)
        attribute2 = st.slider("Potential", min_value=0, max_value=100, value=50, step=1)
        attribute3 = st.number_input("Wage (EUR)", min_value=0, value=5000, step=1000)
        
        st.markdown("#### Skills and Abilities")
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

# Main content
if userinput is not None:
    st.subheader('User Input Parameters')
    st.dataframe(userinput.style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#343a40'), ('color', 'white')]}]
    ).highlight_max(axis=0, color='#d9edf7'))

    try:
        scaled_dataframe = scaler.transform(userinput)

        # Predict the rating using the trained model
        pred = model.predict(scaled_dataframe)

        st.markdown('<div class="prediction-box"><h1>Predicted Rating: {:.2f}</h1></div>'.format(pred[0]), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error occurred during prediction: {str(e)}")
else:
    st.info("Please enter the player attributes and click 'Predict Rating'")

# Footer
st.markdown(
    """
    <div class='footer'>
        <hr>
        <p>&copy; 2024 Player Rating Predictor. All rights reserved. | Built with ❤️ using Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)
