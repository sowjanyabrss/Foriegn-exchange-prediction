import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from fbprophet import Prophet



def kats():
    st.title("Foreign Exchange Prediction")
    uploaded_file = st.file_uploader("Upload CSV" , type = ['csv'])
    if uploaded_file:
        file = pd.read_csv(uploaded_file)
        st.write(file)
        columns = file.columns


        df = pd.DataFrame(columns = ["ds" , "y"])
        df["ds"] = file["Time Serie"]
        country = st.selectbox("Select the Value Column", columns[::-1], key = "a")
        df["y"] = file[country]

        df['ds'] = pd.to_datetime(df['ds'])
        new_df = df[df["y"] != "ND"]

        new_df = new_df.reset_index()
        
        m = Prophet(interval_width = 0.95,yearly_seasonality = True,weekly_seasonality = True,daily_seasonality = False,holidays = None,changepoint_prior_scale = 0.095)
        m.fit(new_df)

        new_date = st.date_input("Enter a Date to Predict", datetime.now())
        val = new_df["ds"][len(new_df)-1].strftime('%Y-%m-%d')
        old_date = datetime.strptime(val,"%Y-%m-%d")
        
        days = (new_date - old_date.date()).days

        if st.button("forecast"):
            future = m.make_future_dataframe(periods=int(days))
            forecast = m.predict(future.tail(int(days)))
            st.write(forecast.tail(1))
            #st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


if __name__ == "__main__":
    kats()
