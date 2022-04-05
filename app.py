import streamlit as st
import pandas as pd
import numpy as np
import keras
import joblib
from sklearn.ensemble import RandomForestRegressor
from prediction import get_prediction, ordinal_encoder
from keras.models import load_model

model = load_model(r'Model/Patient_Survival_NN_Model.h5')

st.set_page_config(page_title="Patient Survival Prediction App",
                   page_icon="🚧", layout="wide")


st.markdown("<h1 style='text-align: center;'>Patient Survival Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        apache_4a_hospital_death_prob = st.text_input("The APACHE IVa probabilistic prediction of in-hospital mortality", value="", max_chars=6)
        apache_4a_icu_death_prob = st.text_input("The APACHE IVa probabilistic prediction of in ICU mortality", value="", max_chars=6)
        gcs_motor_apache = st.text_input("The motor component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score", value="", max_chars=6)
        gcs_eyes_apache = st.text_input("The eye opening component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score", value="", max_chars=6)
        gcs_verbal_apache = st.text_input("The verbal component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score", value="", max_chars=6)
        d1_spo2_min = st.text_input("The patient's lowest peripheral oxygen saturation during the first 24 hours of their unit stay", value="", max_chars=6)
        d1_sysbp_min = st.text_input("The patient's lowest systolic blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured", value="", max_chars=6)
        d1_diasbp_noninvasive_min = st.text_input("The patient's lowest diastolic blood pressure during the first 24 hours of their unit stay, non-invasively measured", value="", max_chars=6)
        d1_temp_min = st.text_input("The patient's lowest core temperature during the first 24 hours of their unit stay", value="", max_chars=6)
        d1_mbp_min = st.text_input("The patient's lowest mean blood pressure during the first 24 hours of their unit stay, either non-invasively or invasively measured", value="", max_chars=6)
        submit = st.form_submit_button("Predict")


    if submit:
        data = np.array([float(apache_4a_hospital_death_prob),float(apache_4a_icu_death_prob),float(gcs_motor_apache),float(gcs_eyes_apache),
        float(gcs_verbal_apache),float(d1_spo2_min),float(d1_sysbp_min),float(d1_diasbp_noninvasive_min),float(d1_temp_min),float(d1_mbp_min)]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted survival of patient is:  {pred}")

if __name__ == '__main__':
    main()
