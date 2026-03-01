import streamlit as st
import pandas as pd

@st.cache_data
def get_data():
    data = pd.read_csv("dataset/archive/DailyDelhiClimateTrain.csv")
    data.columns = ['date', 'temperature', 'humidity', 'wind_speed', 'pressure']
    return data

st.title('Homepage')
data = get_data()
weather_variable = st.selectbox('Weather variable', data.columns[1:])
st.line_chart(data.set_index('date')[weather_variable])

