from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json

# Load model & metadata when app starts
model = joblib.load("models/model.pkl")
with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

# Initialize FastAPI
app = FastAPI(title="House Price Prediction API",
              description="Predicts house price based on input features.")

# Define input schema
class HouseInput(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int

# Define output schema
class PredictionOutput(BaseModel):
    prediction: float

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "House Price Prediction API is running"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: HouseInput):
    try:
        features = np.array([[input_data.area, input_data.bedrooms,
                              input_data.bathrooms, input_data.stories]])
        prediction = model.predict(features)[0]
        return PredictionOutput(prediction=float(prediction))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model info endpoint
@app.get("/model-info")
def model_info():
    return metadata
