import streamlit as st
import numpy as np
import pickle

# Load the saved model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Create a simple Streamlit app to use the model
st.title('Flood Prediction App')
st.image('flood.jpg', use_column_width=True)

# Input features for prediction
temperature = st.number_input('Temperature', value=25)
humidity = st.number_input('Humidity', value=80)
rain_no_of_days = st.number_input('Rain No. of Days', value=6)
measure_of_rain = st.number_input('Measure of Rain', value=70)

# Make prediction
if st.button('Predict'):
    input_data = np.array([[temperature, humidity, rain_no_of_days, measure_of_rain]])
    prediction = model.predict(input_data)
    if(prediction[0]==1):
        st.write("Yes there is a chance to flood.")
    
    else:
        st.write("No there is no chance to flood")
    
    # st.write(f'Predicted class: {prediction[0]}')
