from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cloudburst-prediction-system-2025'

# Load the trained model and preprocessing objects
try:
    model = joblib.load('models/best_cloudburst_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    imputer = joblib.load('models/imputer.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None

# Wind direction encoding mapping
wind_directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                   'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'Unknown']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'MinimumTemperature': float(request.form.get('min_temp')),
                'MaximumTemperature': float(request.form.get('max_temp')),
                'Rainfall': float(request.form.get('rainfall')),
                'Evaporation': float(request.form.get('evaporation', 0)),
                'Sunshine': float(request.form.get('sunshine', 0)),
                'WindGustDirection': wind_directions.index(request.form.get('wind_gust_dir', 'Unknown')),
                'WindGustSpeed': float(request.form.get('wind_gust_speed')),
                'WindDirection9am': wind_directions.index(request.form.get('wind_dir_9am', 'Unknown')),
                'WindDirection3pm': wind_directions.index(request.form.get('wind_dir_3pm', 'Unknown')),
                'WindSpeed9am': float(request.form.get('wind_speed_9am')),
                'WindSpeed3pm': float(request.form.get('wind_speed_3pm')),
                'Humidity9am': float(request.form.get('humidity_9am')),
                'Humidity3pm': float(request.form.get('humidity_3pm')),
                'Pressure9am': float(request.form.get('pressure_9am')),
                'Pressure3pm': float(request.form.get('pressure_3pm')),
                'Cloud9am': float(request.form.get('cloud_9am', 0)),
                'Cloud3pm': float(request.form.get('cloud_3pm', 0)),
                'Temperature9am': float(request.form.get('temp_9am')),
                'Temperature3pm': float(request.form.get('temp_3pm'))
            }

            # Create DataFrame
            input_df = pd.DataFrame([data])

            # Apply imputer
            input_imputed = imputer.transform(input_df)

            # Scale the data
            input_scaled = scaler.transform(input_imputed)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            # Decode prediction
            prediction_label = label_encoder.inverse_transform([prediction])[0]
            confidence = max(prediction_proba) * 100

            result = {
                'prediction': prediction_label,
                'confidence': round(confidence, 2),
                'probability_no': round(prediction_proba[0] * 100, 2),
                'probability_yes': round(prediction_proba[1] * 100, 2),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'input_data': data
            }

            return render_template('results.html', result=result)

        except Exception as e:
            return render_template('predict.html', 
                                 wind_directions=wind_directions,
                                 error=f"Error making prediction: {str(e)}")

    return render_template('predict.html', wind_directions=wind_directions)

@app.route('/dashboard')
def dashboard():
    # Load model comparison results if available
    try:
        results_df = pd.read_csv('model_comparison_results.csv')
        model_data = results_df.to_dict('records')
    except:
        model_data = []

    stats = {
        'total_features': 19,
        'training_samples': 116368,
        'testing_samples': 29092,
        'best_model': 'XGBoost',
        'best_accuracy': 84.43,
        'best_f1_score': 83.26
    }

    return render_template('dashboard.html', stats=stats, model_data=model_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        # Create input array (adjust order based on your model's feature order)
        input_data = pd.DataFrame([data])
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        prediction_label = label_encoder.inverse_transform([prediction])[0]

        return jsonify({
            'success': True,
            'prediction': prediction_label,
            'confidence': round(max(prediction_proba) * 100, 2),
            'probabilities': {
                'No': round(prediction_proba[0] * 100, 2),
                'Yes': round(prediction_proba[1] * 100, 2)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
