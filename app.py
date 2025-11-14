from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load trained models
MODEL_DIR = os.path.dirname(__file__)
TEMP_MODEL_PATH = os.path.join(MODEL_DIR, 'homagama_temp_model.pkl')
RAIN_MODEL_PATH = os.path.join(MODEL_DIR, 'homagama_rain_model.pkl')

temp_model = None
rain_model = None

try:
    with open(TEMP_MODEL_PATH, 'rb') as f:
        temp_model = pickle.load(f, encoding='latin1')
    print(f"✓ Loaded temperature model from {TEMP_MODEL_PATH}")
except Exception as e:
    print(f"⚠ Could not load temp model (will use heuristic fallback): {e}")

try:
    with open(RAIN_MODEL_PATH, 'rb') as f:
        rain_model = pickle.load(f, encoding='latin1')
    print(f"✓ Loaded rain model from {RAIN_MODEL_PATH}")
except Exception as e:
    print(f"⚠ Could not load rain model (will use heuristic fallback): {e}")


@app.route('/predict', methods=['POST'])
def predict():
    """Accepts JSON features (daily + hourly) and returns predictions using trained models.
    
    Expected JSON keys: 
    - Daily: tavg, tmin, tmax, prcp, temp_range, month, day_of_year
    - Hourly (optional): hourly_temps, hourly_rain_prob, hourly_wind, hourly_humidity
    
    Returns: { 
        predicted_temp: <float>,
        predicted_rain_chance: <float>,
        today_data: { tmax, tmin, tavg, prcp },
        features: { ... },
        model_info: { temp_model, rain_model, timestamp },
        hourly_analysis: { avg_wind, avg_humidity, max_rain_prob, temp_volatility }
    }
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    required = ['tavg', 'tmin', 'tmax', 'prcp', 'temp_range', 'month', 'day_of_year']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': 'Missing features', 'missing': missing}), 400

    try:
        tavg = float(data['tavg'])
        tmin = float(data['tmin'])
        tmax = float(data['tmax'])
        prcp = float(data['prcp'])
        temp_range = float(data['temp_range'])
        day_of_year = float(data['day_of_year'])
        month = int(data['month'])
    except Exception as e:
        return jsonify({'error': 'Invalid feature types', 'details': str(e)}), 400

    # Extract hourly data if provided (for enhanced predictions)
    hourly_temps = data.get('hourly_temps', [])
    hourly_rain_prob = data.get('hourly_rain_prob', [])
    hourly_wind = data.get('hourly_wind', [])
    hourly_humidity = data.get('hourly_humidity', [])
    
    # Calculate hourly statistics for better predictions
    temp_volatility = 0
    avg_wind = 0
    avg_humidity = 0
    max_rain_prob = 0
    
    if hourly_temps and len(hourly_temps) > 1:
        # Calculate temperature volatility (standard deviation-like measure)
        temp_mean = sum(hourly_temps) / len(hourly_temps)
        temp_volatility = sum((t - temp_mean) ** 2 for t in hourly_temps) / len(hourly_temps)
        temp_volatility = temp_volatility ** 0.5  # Square root for standard deviation
    
    if hourly_wind:
        avg_wind = sum(hourly_wind) / len(hourly_wind)
    
    if hourly_humidity:
        avg_humidity = sum(hourly_humidity) / len(hourly_humidity)
    
    if hourly_rain_prob:
        max_rain_prob = max(hourly_rain_prob)

    # Prepare feature vector for models (enhanced with hourly insights)
    # Original 7 features + 4 hourly-derived features
    features = [tavg, tmin, tmax, prcp, temp_range, month, day_of_year, 
                temp_volatility, avg_wind, avg_humidity, max_rain_prob / 100 if max_rain_prob > 0 else 0]

    predicted_temp = None
    predicted_rain = None
    temp_model_name = "Not loaded"
    rain_model_name = "Not loaded"

    # Use trained temp model if available, else fallback to heuristic
    if temp_model is not None:
        try:
            # Try with full feature set first, fallback to original 7 features if model expects that
            predicted_temp = float(temp_model.predict([features])[0])
            temp_model_name = type(temp_model).__name__
        except Exception as e:
            print(f"⚠ Error predicting with full features, trying with basic features: {e}")
            try:
                features_basic = [tavg, tmin, tmax, prcp, temp_range, month, day_of_year]
                predicted_temp = float(temp_model.predict([features_basic])[0])
                temp_model_name = type(temp_model).__name__ + " (basic features)"
            except Exception as e2:
                print(f"Error predicting with temp model: {e2}")
                predicted_temp = tavg + 0.5  # Fallback
    else:
        # Fallback heuristic: incorporate hourly volatility and wind
        seasonal_adjust = 3.0 * math.cos(2 * math.pi * (day_of_year / 365.25))
        wind_effect = 0.3 * avg_wind  # High wind can affect temperature
        volatility_effect = 0.2 * temp_volatility  # Higher volatility suggests dynamic weather
        predicted_temp = tavg + seasonal_adjust - 0.1 * temp_range - 0.2 * prcp + wind_effect - volatility_effect

    # Use trained rain model if available, else fallback to simple heuristic
    if rain_model is not None:
        try:
            predicted_rain = float(rain_model.predict([features])[0])
            rain_model_name = type(rain_model).__name__
            # Clamp to 0-100 range for percentage
            predicted_rain = max(0, min(100, predicted_rain))
        except Exception as e:
            print(f"⚠ Error predicting with full features, trying with basic features: {e}")
            try:
                features_basic = [tavg, tmin, tmax, prcp, temp_range, month, day_of_year]
                predicted_rain = float(rain_model.predict([features_basic])[0])
                predicted_rain = max(0, min(100, predicted_rain))
                rain_model_name = type(rain_model).__name__ + " (basic features)"
            except Exception as e2:
                print(f"Error predicting with rain model: {e2}")
                predicted_rain = 30 if prcp > 0 else 10  # Fallback
    else:
        # Fallback heuristic: use hourly rain probability and precipitation
        base_rain_chance = 30 if prcp > 0 else 10
        predicted_rain = min(100, base_rain_chance + max_rain_prob * 0.5)
        predicted_rain = min(100, 10 + prcp * 20)

    return jsonify({
        'predicted_temp': round(predicted_temp, 1),
        'predicted_rain_chance': round(predicted_rain, 1),
        'today_data': {
            'temp_max': tmax,
            'temp_min': tmin,
            'temp_avg': round(tavg, 1),
            'precipitation': prcp
        },
        'features_used': {
            'tavg': tavg,
            'tmin': tmin,
            'tmax': tmax,
            'prcp': prcp,
            'temp_range': temp_range,
            'month': month,
            'day_of_year': day_of_year
        },
        'hourly_analysis': {
            'avg_wind': round(avg_wind, 2),
            'avg_humidity': round(avg_humidity, 1),
            'max_rain_probability': round(max_rain_prob, 1),
            'temperature_volatility': round(temp_volatility, 2)
        },
        'model_info': {
            'temp_model': temp_model_name,
            'rain_model': rain_model_name,
            'timestamp': datetime.now().isoformat()
        }
    })


@app.route('/', methods=['GET'])
def hello():
    return jsonify({'status': 'ok', 'message': 'HomagamaWeather backend is running'})


if __name__ == '__main__':
    # Bind to 127.0.0.1:5000 to match the frontend URL
    app.run(host='127.0.0.1', port=5000, debug=True)
