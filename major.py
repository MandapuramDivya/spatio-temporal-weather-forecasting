import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib

# Load dataset
df = pd.read_csv(r'synthetic_weather_data.csv')
df.columns = df.columns.str.strip()
df['datetime'] = pd.to_datetime(df['Date'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

if 'Time' in df.columns:
    df['hour'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
else:
    df['hour'] = 12

# Encode categorical variables
le_location = LabelEncoder()
df['Location'] = le_location.fit_transform(df['Location'])
le_weather = LabelEncoder()
df['Weather Condition'] = le_weather.fit_transform(df['Weather Condition'])

# Features and Targets
X = df[['year', 'month', 'day', 'hour', 'Location']]
y_weather = df['Weather Condition']
y_max_temp = df['Max_Temperature']
y_min_temp = df['Min_Temperature']

# Train Models
X_train, X_test, y_train_weather, y_test_weather = train_test_split(X, y_weather, test_size=0.2, random_state=42)
_, _, y_train_max, y_test_max = train_test_split(X, y_max_temp, test_size=0.2, random_state=42)
_, _, y_train_min, y_test_min = train_test_split(X, y_min_temp, test_size=0.2, random_state=42)

# Model paths
MODEL_FILE_WEATHER = "xgboost_weather_model.json"
MODEL_FILE_MAX_TEMP = "xgboost_max_temp_model.json"
MODEL_FILE_MIN_TEMP = "xgboost_min_temp_model.json"

# Load or Train Models
def load_or_train_model(model_file, X_train, y_train, model_type='classifier'):
    if os.path.exists(model_file):
        print(f"Loading saved model: {model_file}")
        model = xgb.XGBClassifier() if model_type == 'classifier' else xgb.XGBRegressor()
        model.load_model(model_file)
    else:
        print(f"Training new model: {model_file}")
        model = xgb.XGBClassifier(eval_metric='mlogloss') if model_type == 'classifier' else xgb.XGBRegressor()
        model.fit(X_train, y_train)
        model.save_model(model_file)
    return model

model_weather = load_or_train_model(MODEL_FILE_WEATHER, X_train, y_train_weather, 'classifier')
model_max_temp = load_or_train_model(MODEL_FILE_MAX_TEMP, X_train, y_train_max, 'regressor')
model_min_temp = load_or_train_model(MODEL_FILE_MIN_TEMP, X_train, y_train_min, 'regressor')

# Predict and Visualize
def predict_and_plot_weather(date, time, location):
    try:
        datetime_input = pd.to_datetime(date + ' ' + time)
    except Exception as e:
        print(f"Invalid date/time format: {e}")
        return

    if location not in le_location.classes_:
        print(f"Location '{location}' not found in training data.")
        return

    location_encoded = le_location.transform([location])[0]

    # Normalize to start forecast from the exact input date
    dates = pd.date_range(start=datetime_input.normalize(), periods=7, freq='D')
    hours = [datetime_input.hour] * 7

    input_data = pd.DataFrame({
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'hour': hours,
        'Location': [location_encoded] * 7
    })

    max_temps = model_max_temp.predict(input_data)
    min_temps = model_min_temp.predict(input_data)
    predicted_weather = model_weather.predict(input_data)
    weather_conditions = le_weather.inverse_transform(predicted_weather)

    # Display output for the selected day
    print(f"Weather Condition on {date}: {weather_conditions[0]}")
    print(f"Max Temperature: {max_temps[0]:.2f}°C")
    print(f"Min Temperature: {min_temps[0]:.2f}°C")

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(dates, max_temps, label='Max Temperature (°C)', color='red', marker='o')
    plt.plot(dates, min_temps, label='Min Temperature (°C)', color='blue', marker='x')

    for i, condition in enumerate(weather_conditions):
        plt.text(dates[i], max_temps[i], f'{condition}', fontsize=9, ha='center')

    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title(f'7-Day Weather Forecast for {location}')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

# User Input
date = input('Enter date (YYYY-MM-DD): ')
time = input('Enter time (HH:MM): ')
location = input('Enter location: ')
predict_and_plot_weather(date, time, location)
