import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st

model = pk.load(open('model.pk','rb'))

st.header('Car Price Prediction-Machine Learning Model')

cars_data = pd.read_csv('Cardetails.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

cars_data['name'] = cars_data['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', cars_data['name'].unique(),)

year = st.slider('Car Manufactured Year', 1994,2024,)

km_driven = st.slider('No of kms Driven', 11,200000)

fuel = st.selectbox('Fuel type', cars_data['fuel'].unique())

seller_type = st.selectbox('Seller  type', cars_data['seller_type'].unique())

transmission = st.selectbox('Transmission type', cars_data['transmission'].unique())

owner = st.selectbox('Owner', cars_data['owner'].unique())

mileage = st.slider('Car Mileage', 10,40)

engine = st.slider('Engine CC', 700,5000)

max_power = st.slider('Max Power', 10,200)

seats = st.slider('No of Seats', 5,10)

if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    
    input_data_model['name'].replace(cars_data['name'].unique(), range(1, len(cars_data['name'].unique()) + 1), inplace=True)
    input_data_model['fuel'].replace(cars_data['fuel'].unique(), range(1, len(cars_data['fuel'].unique()) + 1), inplace=True)
    input_data_model['seller_type'].replace(cars_data['seller_type'].unique(), range(1, len(cars_data['seller_type'].unique()) + 1), inplace=True)
    input_data_model['transmission'].replace(cars_data['transmission'].unique(), range(1, len(cars_data['transmission'].unique()) + 1), inplace=True)
    input_data_model['owner'].replace(cars_data['owner'].unique(), range(1, len(cars_data['owner'].unique()) + 1), inplace=True)

    car_price = model.predict(input_data_model)
    
    car_price = abs(car_price[0])
    
    st.subheader('Predicted Car Price is')
    st.markdown(f'<p style="color:#ff6961; font-size: 35px;">₹{car_price:,.0f}</p>', unsafe_allow_html=True)
