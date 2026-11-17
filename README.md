# Homagama Weather Backend

A Flask-based REST API for weather prediction using machine learning models. This backend provides temperature and rainfall prediction endpoints for the Homagama region based on historical weather data.

## Overview

This backend service integrates scikit-learn trained models to predict:
- **Temperature predictions**: Based on daily weather metrics
- **Rain probability predictions**: Percentage chance of precipitation

The API accepts both daily weather features and optional hourly data for enhanced predictions, providing a robust fallback mechanism when models aren't available.

## Features

- ✅ Machine Learning-based predictions using trained models
- ✅ Support for daily and hourly weather features
- ✅ Intelligent fallback heuristics when models fail
- ✅ CORS-enabled for cross-origin requests
- ✅ Comprehensive error handling and validation
- ✅ Real-time model loading and status reporting
- ✅ Hourly analysis metrics (wind, humidity, volatility)

## Project Structure

```
homagama-weather-backend/
├── app.py                          # Main Flask application
├── homagama_temp_model.pkl         # Pre-trained temperature prediction model
├── homagama_rain_model.pkl         # Pre-trained rainfall probability model
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/itz-im45h4/homagama-weather-backend.git
   cd homagama-weather-backend
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- **Flask** (>=2.0.0): Web framework for building the REST API
- **Flask-CORS** (>=3.0.10): Handle cross-origin requests
- **scikit-learn** (>=1.0.0): Machine learning library for model predictions

## Usage

### Running the Server

Start the Flask development server:

```bash
python app.py
```

The server will run on `http://127.0.0.1:5000`

### API Endpoints

#### 1. Health Check
- **Endpoint**: `GET /`
- **Description**: Check if the backend is running
- **Response**:
  ```json
  {
    "status": "ok",
    "message": "HomagamaWeather backend is running"
  }
  ```

#### 2. Weather Prediction
- **Endpoint**: `POST /predict`
- **Description**: Get temperature and rainfall predictions

**Request Body (JSON)**:
```json
{
  "tavg": 25.5,
  "tmin": 20.0,
  "tmax": 31.0,
  "prcp": 5.2,
  "temp_range": 11.0,
  "month": 1,
  "day_of_year": 15,
  "hourly_temps": [20.1, 22.5, 24.8, 26.2, 25.9, 23.1],
  "hourly_rain_prob": [10, 15, 20, 25, 30, 15],
  "hourly_wind": [5.2, 6.1, 7.3, 6.8, 5.5, 4.2],
  "hourly_humidity": [75, 70, 65, 60, 65, 72]
}
```

**Required Fields**:
- `tavg` (float): Average temperature (°C)
- `tmin` (float): Minimum temperature (°C)
- `tmax` (float): Maximum temperature (°C)
- `prcp` (float): Precipitation amount (mm)
- `temp_range` (float): Temperature range (°C)
- `month` (int): Month (1-12)
- `day_of_year` (int): Day of year (1-366)

**Optional Fields** (for enhanced predictions):
- `hourly_temps` (list): Hourly temperatures throughout the day
- `hourly_rain_prob` (list): Hourly rain probability (0-100)
- `hourly_wind` (list): Hourly wind speeds (km/h)
- `hourly_humidity` (list): Hourly humidity percentages (0-100)

**Response**:
```json
{
  "predicted_temp": 25.8,
  "predicted_rain_chance": 35.5,
  "today_data": {
    "temp_max": 31.0,
    "temp_min": 20.0,
    "temp_avg": 25.5,
    "precipitation": 5.2
  },
  "features_used": {
    "tavg": 25.5,
    "tmin": 20.0,
    "tmax": 31.0,
    "prcp": 5.2,
    "temp_range": 11.0,
    "month": 1,
    "day_of_year": 15
  },
  "hourly_analysis": {
    "avg_wind": 5.85,
    "avg_humidity": 67.8,
    "max_rain_probability": 30.0,
    "temperature_volatility": 2.34
  },
  "model_info": {
    "temp_model": "RandomForestRegressor",
    "rain_model": "GradientBoostingRegressor",
    "timestamp": "2026-01-07T10:30:45.123456"
  }
}
```

## Model Information

### Temperature Model
- **File**: `homagama_temp_model.pkl`
- **Purpose**: Predicts average temperature based on weather features
- **Fallback**: If the model fails to load or predict, a heuristic calculation is used incorporating seasonal adjustments and wind effects

### Rain Model
- **File**: `homagama_rain_model.pkl`
- **Purpose**: Predicts probability of rainfall as a percentage (0-100)
- **Fallback**: If the model fails, a simple heuristic based on precipitation amount is used

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: 
  - Invalid JSON format
  - Missing required fields
  - Invalid data types for features

- **Model Fallback**: 
  - If a model fails to load, the system logs a warning and uses heuristic calculations
  - If prediction fails with full features, it automatically retries with basic features

**Example Error Response**:
```json
{
  "error": "Missing features",
  "missing": ["tavg", "month"]
}
```

## Prediction Algorithm

### Temperature Prediction
1. **With ML Model**: Uses trained scikit-learn model (RandomForest, etc.)
2. **Fallback Heuristic**: 
   - Base: Average temperature (tavg)
   - Adjustments:
     - Seasonal adjustment using cosine function
     - Temperature range effect
     - Precipitation cooling effect
     - Wind effect on temperature
     - Volatility adjustment

### Rainfall Prediction
1. **With ML Model**: Uses trained scikit-learn model (GradientBoosting, etc.)
2. **Fallback Heuristic**:
   - Base: 30% if precipitation > 0, else 10%
   - Adjusted by maximum hourly rain probability and precipitation amount
   - Clamped to 0-100 range

## Performance Metrics

The response includes hourly analysis for better insights:
- **Temperature Volatility**: Standard deviation of hourly temperatures
- **Average Wind**: Mean wind speed across the day
- **Average Humidity**: Mean humidity across the day
- **Max Rain Probability**: Highest rain probability from hourly data

## Development

### Running in Debug Mode
The server runs in debug mode by default:
```bash
python app.py
```

This enables:
- Auto-reloading on code changes
- Detailed error messages
- Interactive debugger

### Testing with cURL

Test the health endpoint:
```bash
curl http://127.0.0.1:5000/
```

Test predictions:
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tavg": 25.5,
    "tmin": 20.0,
    "tmax": 31.0,
    "prcp": 5.2,
    "temp_range": 11.0,
    "month": 1,
    "day_of_year": 15
  }'
```

## Troubleshooting

### Models not loading
- Verify `.pkl` files are in the same directory as `app.py`
- Check file permissions
- Ensure pickle protocol compatibility with your Python version

### CORS errors
- The API has CORS enabled for all origins
- If issues persist, check browser console for detailed errors

### Prediction errors
- Verify all required fields are present
- Ensure data types match (floats for numeric values, ints for month)
- Check model fallback messages in server logs

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Contact

For questions or support, please reach out (｡ •̀ᴗ-)✧ 

---

**Repository**: [homagama-weather-backend](https://github.com/itz-im45h4/homagama-weather-backend)
