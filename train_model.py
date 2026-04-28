import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Dataset
data = pd.DataFrame({
    "area": [600, 800, 1200, 1500, 1000],
    "bedrooms": [1, 2, 3, 3, 2],
    "bathrooms": [1, 1, 2, 3, 2],
    "location": ["Hyderabad", "Hyderabad", "Hyderabad", "Bangalore", "Bangalore"],
    "furnishing": ["Semi", "Furnished", "Furnished", "Furnished", "Semi"],
    "parking": [0, 1, 1, 1, 1],
    "rent": [7000, 12000, 18000, 25000, 15000]
})

# Encode categorical data
le_location = LabelEncoder()
le_furnishing = LabelEncoder()

data["location"] = le_location.fit_transform(data["location"])
data["furnishing"] = le_furnishing.fit_transform(data["furnishing"])

# Features & target
X = data[["area", "bedrooms", "bathrooms", "parking", "location", "furnishing"]]
y = data["rent"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save EVERYTHING (important!)
joblib.dump({
    "model": model,
    "le_location": le_location,
    "le_furnishing": le_furnishing
}, "rent_model.joblib")

print("✅ Model + encoders saved successfully!")