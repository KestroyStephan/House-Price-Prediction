import json
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import numpy as np
from datetime import datetime

# Load Dataset

import pandas as pd

df = pd.read_csv("data/housing.csv")
print(df.columns)

# Select features and target
X = df[["area", "bedrooms", "bathrooms", "stories"]]   # input features
y = df["price"] 

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees
model.fit(X_train, y_train)

#Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

#Save metadata
metadata = {
    "model_type": "Random Forest Regressor",
    "problem_type": "Regression (House Price Prediction)",
    "features": list(X.columns),
    "n_estimators": 100
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f)

print("✅ Random Forest model training complete! Model and metadata saved.")