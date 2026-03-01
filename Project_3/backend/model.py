# import datetime
# import joblib
# import pandas as pd
# import os
# from prophet import Prophet
 
# BASE_DIR = os.getcwd()
# TODAY = datetime.date.today()
# DATA_PATH = "dataset/archive/DailyDelhiClimateTrain.csv"

# def train(weather_variable='temperature'):
#     data = pd.read_csv(DATA_PATH)
#     data.columns = ['date', 'temperature', 'humidity', 'wind_speed', 'pressure']
#     df_forecast = data.copy()
#     df_forecast['ds'] = pd.to_datetime(df_forecast['date'])
#     df_forecast['y'] = df_forecast[weather_variable]
#     df_forecast = df_forecast[['ds','y']]
#     model = Prophet()
#     model.fit(df_forecast)

#     joblib.dump(model, os.path.join(BASE_DIR, f'{weather_variable}.joblib'))

# def predict(weather_variable='temperature', days=7):
#     model_file = os.path.join(BASE_DIR, f'{weather_variable}.joblib')
#     model = joblib.load(model_file)

#     future = TODAY + datetime.timedelta(days=days)
#     dates = pd.date_range(
#         start='2017-01-01',
#         end = future.strftime('%m/%d/%Y')

#     )

#     df = pd.DataFrame({'ds': dates})
#     forecast = model.predict(df)

#     return forecast.tail(days).to_dict('records')

# def convert(prediction_list):
#     output = {}
#     for data in prediction_list:
#         date = data['ds'].strftime('%m/%d/%Y')
#         output[date] = data['yhat']
#     return output


# train()
# train('wind_speed')
# train('pressure')
# train('humidity')

import datetime
import joblib
import pandas as pd
import os
from prophet import Prophet

# Make paths relative to this file
BASE_DIR = os.path.dirname(__file__)
TODAY = datetime.date.today()
DATA_PATH = os.path.join(BASE_DIR, "dataset/archive/DailyDelhiClimateTrain.csv")

def train(weather_variable='temperature'):
    data = pd.read_csv(DATA_PATH)
    data.columns = ['date', 'temperature', 'humidity', 'wind_speed', 'pressure']

    df_forecast = data.copy()
    df_forecast['ds'] = pd.to_datetime(df_forecast['date'])
    df_forecast['y'] = df_forecast[weather_variable]
    df_forecast = df_forecast[['ds','y']]

    model = Prophet()
    model.fit(df_forecast)

    joblib.dump(model, os.path.join(BASE_DIR, f'{weather_variable}.joblib'))

def predict(weather_variable='temperature', days=7):
    model_file = os.path.join(BASE_DIR, f'{weather_variable}.joblib')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model {model_file} not found. Please train it first.")
    
    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)
    dates = pd.date_range(
        start='2017-01-01',
        end=future.strftime('%m/%d/%Y')
    )

    df = pd.DataFrame({'ds': dates})
    forecast = model.predict(df)

    return forecast.tail(days).to_dict('records')

def convert(prediction_list):
    output = {}
    for data in prediction_list:
        date = data['ds'].strftime('%m/%d/%Y')
        output[date] = data['yhat']
    return output

# Optional: train only if running locally
if __name__ == "__main__":
    train()
    train('wind_speed')
    train('pressure')
    train('humidity')

