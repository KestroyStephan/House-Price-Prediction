# House Price Prediction API

## Problem
Predict house prices using features: area, bedrooms, bathrooms, stories.

## Model
Random Forest Regressor trained on Kaggle housing dataset.

## API Usage

### Health check
GET /

### Prediction
POST /predict
Example input:
{
  "area": 2500,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 2
}

### Model Info
GET /model-info
Returns model type and features used.

## Run the API
1. Install dependencies: pip install -r requirements.txt
2. Run: uvicorn main:app --reload
3. Open: http://127.0.0.1:8000/docs. 
