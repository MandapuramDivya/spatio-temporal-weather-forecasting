from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
from mysql.connector import Error
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "your_secret_key"
app.permanent_session_lifetime = timedelta(days=7)

# Database Connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="major"
        )
        if conn.is_connected():
            print("Database Connected Successfully!")
        return conn
    except Error as e:
        print(f"Database Connection Failed: {e}")
        return None

# Route for Home Page
@app.route("/")
def home():
    return render_template("stchome.html")

# User Registration Route
@app.route('/registeruser', methods=['GET', 'POST'])
def registeruser():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash("All fields are required!", "warning")
            return redirect(url_for('registeruser'))

        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    flash("Email already exists! Please log in.", "danger")
                    return redirect(url_for('registeruser'))

                cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, password))
                conn.commit()
                flash("Registration successful! You can now log in.", "success")
                return redirect(url_for('userlogin'))
            except Error as e:
                flash("Database error occurred.", "danger")
            finally:
                cursor.close()
                conn.close()

    return render_template("registeruser.html")

# User Login Route
@app.route("/userlogin", methods=['GET', 'POST'])
def userlogin():
    if request.method == 'POST':
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')

            if not username or not password:
                return jsonify({"success": False, "message": "Username and password are required!"})

            conn = get_db_connection()
            if conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
                user = cursor.fetchone()

                if user:
                    session.permanent = True
                    session['user'] = user['username']
                    return jsonify({"success": True, "message": "Login successful!"})
                return jsonify({"success": False, "message": "Invalid username or password"}), 401
        except Error as e:
            return jsonify({"success": False, "message": "Database error occurred."}), 500
        finally:
            cursor.close()
            conn.close()
    return render_template("userlogin.html")

# Load Models Safely
MODEL_FILE_WEATHER = "xgboost_weather_model.json"
MODEL_FILE_MAX_TEMP = "xgboost_max_temp_model.json"
MODEL_FILE_MIN_TEMP = "xgboost_min_temp_model.json"
ENCODER_LOCATION_FILE = "location_encoder.pkl"
ENCODER_WEATHER_FILE = "weather_encoder.pkl"

model_weather, model_max_temp, model_min_temp = None, None, None
le_location, le_weather = None, None

if os.path.exists(MODEL_FILE_WEATHER):
    model_weather = xgb.XGBClassifier()
    model_weather.load_model(MODEL_FILE_WEATHER)
    print("Weather model loaded.")

if os.path.exists(MODEL_FILE_MAX_TEMP) and os.path.exists(MODEL_FILE_MIN_TEMP):
    model_max_temp = xgb.XGBRegressor()
    model_min_temp = xgb.XGBRegressor()
    model_max_temp.load_model(MODEL_FILE_MAX_TEMP)
    model_min_temp.load_model(MODEL_FILE_MIN_TEMP)
    print("Temperature models loaded.")

if os.path.exists(ENCODER_LOCATION_FILE):
    le_location = joblib.load(ENCODER_LOCATION_FILE)

if os.path.exists(ENCODER_WEATHER_FILE):
    le_weather = joblib.load(ENCODER_WEATHER_FILE)

# Prediction Function
@app.route('/predict', methods=['POST'])
def get_prediction():
    try:
        data = request.get_json()
        location = data.get('location')
        datetime_str = data.get('datetime')

        if not location or not datetime_str:
            return jsonify({"success": False, "message": "Missing input data."}), 400

        date, time = datetime_str.split(' ')
        datetime_input = pd.to_datetime(f"{date} {time}")

        if location not in le_location.classes_:
            return jsonify({"success": False, "message": "Location not found in training data."}), 400

        input_data = pd.DataFrame({
            'year': [datetime_input.year],
            'month': [datetime_input.month],
            'day': [datetime_input.day],
            'hour': [datetime_input.hour],
            'Location': [le_location.transform([location])[0]]
        })

        predicted_weather = model_weather.predict(input_data)
        weather_condition = le_weather.inverse_transform(predicted_weather)[0]
        max_temp = float(model_max_temp.predict(input_data)[0])
        min_temp = float(model_min_temp.predict(input_data)[0])

        return jsonify({
            "success": True,
            "weather_condition": weather_condition,
            "max_temperature": round(max_temp, 2),
            "min_temperature": round(min_temp, 2)
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# Helper Function for 7-Day Forecast
def get_7day_forecast_data(location, start_date):
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    forecast_data = []

    for i in range(7):
        forecast_date = start_date_obj + timedelta(days=i)
        input_data = pd.DataFrame({
            'year': [forecast_date.year],
            'month': [forecast_date.month],
            'day': [forecast_date.day],
            'hour': [12],  # Midday
            'Location': [le_location.transform([location])[0]]
        })

        predicted_weather = model_weather.predict(input_data)
        weather_condition = le_weather.inverse_transform(predicted_weather)[0]
        max_temp = float(model_max_temp.predict(input_data)[0])
        min_temp = float(model_min_temp.predict(input_data)[0])

        forecast_data.append({
            'date': forecast_date.strftime('%Y-%m-%d'),
            'weather_condition': weather_condition,
            'max_temperature': round(max_temp, 2),
            'min_temperature': round(min_temp, 2)
        })

    return forecast_data

# 7-Day Forecast Endpoint
@app.route('/get_7day_forecast', methods=['POST'])
def get_7day_forecast():
    try:
        data = request.get_json()
        location = data.get('location')
        start_date = data.get('date')  # Format: 'YYYY-MM-DD'

        # Check if location and start date are provided
        if not location or not start_date:
            return jsonify({"success": False, "message": "Location and date are required."}), 400

        # Check if the location exists in the encoded locations
        if location not in le_location.classes_:
            return jsonify({"success": False, "message": "Invalid location."}), 400

        # Fetch the forecast data for the 7 days
        forecast_data = get_7day_forecast_data(location, start_date)
        
        # Return the forecast data as a JSON response
        return jsonify({"success": True, "forecast": forecast_data})

    except Exception as e:
        # Return any errors encountered during the process
        return jsonify({"success": False, "message": str(e)}), 500

# Main Page Route
@app.route('/main')
def main():
    if 'user' in session:
        return render_template("main.html", username=session['user'])
    flash("⚠ Please log in to continue.", "warning")
    return redirect(url_for('userlogin'))

# ✅ Visualize Route (Updated)
@app.route('/visualize')
def visualize():
    return render_template("visualize.html")

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('userlogin'))

# Run Flask App
if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:5001/")
    app.run(host="0.0.0.0", port=5001, debug=True)
