import streamlit as st
import pandas as pd
import numpy as np
import keras
import joblib
from sklearn.ensemble import RandomForestRegressor
from prediction import get_prediction, ordinal_encoder
from keras.models import load_model

model = load_model('Patient_Survival_NN_Model.h5')

st.set_page_config(page_title="Patient Survival Prediction App",
                   page_icon="🚧", layout="wide")


st.markdown("<h1 style='text-align: center;'>Patient Survival Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        apache_4a_hospital_death_prob = st.text_input("apache_4a_hospital_death_prob", value="", max_chars=6)
        apache_4a_icu_death_prob = st.text_input("apache_4a_icu_death_prob", value="", max_chars=6)
        gcs_motor_apache = st.text_input("gcs_motor_apache", value="", max_chars=6)
        gcs_eyes_apache = st.text_input("gcs_eyes_apache", value="", max_chars=6)
        gcs_verbal_apache = st.text_input("gcs_verbal_apache", value="", max_chars=6)
        d1_spo2_min = st.text_input("d1_spo2_min", value="", max_chars=6)
        d1_sysbp_min = st.text_input("d1_sysbp_min", value="", max_chars=6)
        d1_diasbp_noninvasive_min = st.text_input("d1_diasbp_noninvasive_min", value="", max_chars=6)
        d1_temp_min = st.text_input("d1_temp_min", value="", max_chars=6)
        d1_mbp_min = st.text_input("d1_mbp_min", value="", max_chars=6)
        submit = st.form_submit_button("Predict")


    if submit:
        data = np.array([apache_4a_hospital_death_prob,apache_4a_icu_death_prob,gcs_motor_apache,gcs_eyes_apache,
        gcs_verbal_apache,d1_spo2_min,d1_sysbp_min,d1_diasbp_noninvasive_min,d1_temp_min,d1_mbp_min]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted survival of patient is:  {pred}")

if __name__ == '__main__':
    main()
