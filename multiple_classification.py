# -*- coding: utf-8 -*-

import pickle
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu

# -------------------- LOAD MODELS --------------------
loan_model = pickle.load(open("C:/Users/Lab/Desktop/ML/loan_model.sav",'rb'))
ridingmower_model = pickle.load(open("C:/Users/Lab/Desktop/ML/RidingMowers_model.sav",'rb'))
stress_model = pickle.load(open("Stress_model.sav",'rb'))

# -------------------- SIDEBAR --------------------
with st.sidebar:
    selected = option_menu(
        'Classification System',
        ['LOAN','RidingMower','Stress'],
        icons=['cash','car-front','emoji-stress'],
        default_index=0
    )

# ==========================================================
# ======================= STRESS ============================
# ==========================================================
if selected == 'Stress':
    st.title('Stress Level Prediction')

    Age = st.number_input('Age', min_value=0)
    Gender = st.selectbox('Gender', ['Male','Female','Other'])
    Occupation = st.selectbox('Occupation', ['Student','Employee','Freelancer'])
    Device_Type = st.selectbox('Device Type', ['Android','iOS'])
    Daily_Phone_Hours = st.number_input('Daily Phone Hours', min_value=0.0)
    Social_Media_Hours = st.number_input('Social Media Hours', min_value=0.0)
    Work_Productivity_Score = st.number_input('Work Productivity Score', min_value=0.0)
    Sleep_Hours = st.number_input('Sleep Hours', min_value=0.0)
    App_Usage_Count = st.number_input('App Usage Count', min_value=0)
    Caffeine_Intake_Cups = st.number_input('Caffeine Intake Cups', min_value=0)
    Weekend_Screen_Time_Hours = st.number_input('Weekend Screen Time Hours', min_value=0.0)

    # ----- Manual Encoding (ต้องตรงกับตอน train โมเดล) -----
    gender_map = {'Female':0, 'Male':1, 'Other':2}
    occupation_map = {'Employee':0, 'Freelancer':1, 'Student':2}
    device_map = {'Android':0, 'iOS':1}

    if st.button('Predict Stress'):
        input_data = np.array([[
            Age,
            gender_map[Gender],
            occupation_map[Occupation],
            device_map[Device_Type],
            Daily_Phone_Hours,
            Social_Media_Hours,
            Work_Productivity_Score,
            Sleep_Hours,
            App_Usage_Count,
            Caffeine_Intake_Cups,
            Weekend_Screen_Time_Hours
        ]])

        prediction = stress_model.predict(input_data)

        st.success(f"Predicted Stress Level: {prediction[0]}")


# ==========================================================
# =================== RIDING MOWER =========================
# ==========================================================
if selected == 'RidingMower':
    st.title('Riding Mower Prediction')

    Income = st.number_input('Income')
    LotSize = st.number_input('Lot Size')

    if st.button('Predict RidingMower'):
        input_data = np.array([[Income, LotSize]])
        prediction = ridingmower_model.predict(input_data)

        if prediction[0] == 0:
            st.success('Prediction: Non Owner')
        else:
            st.success('Prediction: Owner')


# ==========================================================
# ======================== LOAN ============================
# ==========================================================
if selected == 'LOAN':
    st.title('Loan Approval Prediction')

    person_age = st.number_input('Person Age')
    person_income = st.number_input('Person Income')
    credit_score = st.number_input('Credit Score')

    if st.button('Predict Loan'):
        input_data = np.array([[person_age, person_income, credit_score]])
        prediction = loan_model.predict(input_data)

        if prediction[0] == 0:
            st.success('Prediction: Loan Reject')
        else:
            st.success('Prediction: Loan Accept')