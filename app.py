import streamlit as st
import numpy as np
import joblib

# =========================
# Load trained model and scaler
# =========================
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title='ডায়াবেটিস পূর্বাভাস (Diabetes Prediction)',
    page_icon='🩺',
    layout='centered'
)

# =========================
# Title and description
# =========================
st.title('🩺 ডায়াবেটিস পূর্বাভাস (Diabetes Prediction)')
st.write('রোগীর তথ্য দিন এবং **পূর্বাভাস দেখুন (Predict)** বাটনে ক্লিক করুন।')

# =========================
# Input fields (Bangla + English)
# =========================

pregnancies = st.number_input(
    'গর্ভধারণের সংখ্যা (Pregnancies)',
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    'রক্তে গ্লুকোজের মাত্রা (Glucose)',
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    'রক্তচাপ (Blood Pressure)',
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    'ত্বকের পুরুত্ব (Skin Thickness)',
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    'ইনসুলিনের মাত্রা (Insulin)',
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    'বডি মাস ইনডেক্স (BMI)',
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    'ডায়াবেটিস পারিবারিক ঝুঁকি মান (Diabetes Pedigree Function)',
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    'বয়স (Age)',
    min_value=1,
    max_value=120,
    value=30
)

# =========================
# Prediction button
# =========================
if st.button('পূর্বাভাস দেখুন (Predict)'):

    # Create input array
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    # Standardize input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Show result
    st.subheader('পূর্বাভাসের ফলাফল (Prediction Result)')

    if prediction[0] == 0:
        st.success('✅ এই ব্যক্তির **ডায়াবেটিস নেই (NOT Diabetic)**।')
    else:
        st.error('⚠️ এই ব্যক্তি **ডায়াবেটিসে আক্রান্ত হতে পারেন (Diabetic)**।')

# =========================
# Footer
# =========================
st.markdown('---')
st.caption('AI-based Diabetes Prediction System')