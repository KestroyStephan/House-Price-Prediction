import json
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("data/housing.csv")
print("Available columns:", df.columns)

# Select features and target
X = df[["area", "bedrooms", "bathrooms", "stories"]]   # input features
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

# Train & evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"✅ {name} trained. MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")

# Select best model (highest R² score)
best_model_name = max(results, key=lambda name: results[name]["R2"])
best_model = models[best_model_name]
print("\n🎯 Best model is:", best_model_name, "with R² =", results[best_model_name]["R2"])

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")

# Save metadata (with all results)
metadata = {
    "best_model": best_model_name,
    "results": results,
    "features": list(X.columns)
}
with open("models/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("✅ Training complete! Best model and metadata saved.")
