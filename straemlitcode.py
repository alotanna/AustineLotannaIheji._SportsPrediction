# -*- coding: utf-8 -*-
"""straemlitcode.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dTunbPF-HCSxTsGeCucr6DovRPoFRI8a
"""

#!pip install streamlit
import streamlit as st
import numpy as np
import joblib
import pandas as pd

#load the model
model = joblib.load("C:/Users/austi/Intro to AI Codes and dataset/main_final_model.pkl")
scaler = joblib.load("C:/Users/austi/Intro to AI Codes and dataset/scaler.pkl")

def usersinput():
    attribute1 = st.number_input("Enter the movement reactions")
    attribute2 = st.number_input("Enter the potential")
    attribute3 = st.number_input("Enter the wage")
    attribute4 = st.number_input("Enter the shot power")
    attribute5 = st.number_input("Enter the value of the player in euros")
    attribute6 = st.number_input("Enter the passing accuracy score")
    attribute7 = st.number_input("Enter the mentality vision")
    attribute8 = st.number_input("Enter the international reputation")
    attribute9 = st.number_input("Enter the long passing")
    attribute10 = st.number_input("Enter the shot power")
    attribute11 = st.number_input("Enter the physic")
    attribute12 = st.number_input("Enter the ball control")

    dict = {
        'movement_reactions': attribute1,
        'potential': attribute2,
        'wage_eur': attribute3,
        'power_shot_power': attribute4,
        'value_eur': attribute5,
        'passing': attribute6,
        'mentality_vision': attribute7,
        'international_reputation' : attribute8,
        'skill_long_passing': attribute9,
        'power_shot_power': attribute10,
        'physic': attribute11 ,
        'skill_ball_control' : attribute12
    }
    df = pd.DataFrame(dict, index=[0])
    return df

userinput = usersinput()

st.subheader('User Input parameters')
st.write(userinput)

try:

  scaled_dataframe = scaler.transform(userinput)

  #predict the rating using the trained model
  pred = model.predict(scaled_dataframe)


  st.subheader('Prediction')
  st.write("The predicted rating is:  " + str(pred))



except Exception as e:
  st.error("Error occured: " + str(e))

