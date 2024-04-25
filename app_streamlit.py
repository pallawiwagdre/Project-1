import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

pickle_in = open("load_car_sale_price_model.pkl", "rb")
classifier = pickle.load(pickle_in)


def welcome():
    return 'Welecome All'


def predict_note_authentication(registered_year, engine_capacity, insurance, transmission_type, 
                                kms_driven, owner_type, fuel_type, max_power, mileage, body_type, seats):
    prediction = classifier.predict([[registered_year, engine_capacity, insurance,
                                    transmission_type, kms_driven, owner_type, fuel_type, max_power, mileage, body_type,seats]])
    print(prediction)
    return prediction


def main():
    st.title('CAR RESALE PRICE PREDICTION')
    html_temp = """
    <div style = 'backgroud-color:red;padding:10px'>
    <h2 style = 'color:white;text-align:centre;'>Streamlit CAR RESALE PRICE PREDICTION ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    registered_year = st.text_input('registered_year')
    engine_capacity = st.text_input('engine_capacity')
    insurance = st.text_input('insurance')
    transmission_type = st.text_input('transmission_type')
    kms_driven = st.text_input('kms_driven')
    owner_type = st.text_input('owner_type')
    fuel_type = st.text_input('fuel_type')
    max_power = st.text_input('max_power')
    mileage = st.text_input('mileage')
    body_type = st.text_input('body_type')
    seats = st.text_input('seats')

    result = ""
    if st.button('Predict'):
        result = predict_note_authentication(eval(registered_year), eval(engine_capacity), eval(insurance), eval(transmission_type), 
                eval(kms_driven), eval(owner_type), eval(fuel_type), eval(max_power), eval(mileage), eval(body_type), eval(seats))
        st.success('The output is {}'.format(result))
        if st.button('About'):
            st.text('Lets Learn')
            st.text('This is about Car Price Prediction')


if __name__ == '__main__':
    main()
