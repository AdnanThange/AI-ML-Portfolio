import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("boston_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as boston_model.pkl")
