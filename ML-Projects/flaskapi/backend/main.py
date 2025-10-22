from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle, os

# -------------------- Load Model --------------------
model_path = os.path.join(os.path.dirname(__file__), "boston_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# -------------------- FastAPI Setup --------------------
app = FastAPI(title="🏡 Boston House Price Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- Pydantic Models --------------------
class BostonInput(BaseModel):
    RM: float      # Average number of rooms
    LSTAT: float   # % lower status of the population
    PTRATIO: float # Pupil-teacher ratio

class PredictionResponse(BaseModel):
    predicted_price: float

# -------------------- Routes --------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(data: BostonInput):
    features = np.array([[data.RM, data.LSTAT, data.PTRATIO]])
    price = float(model.predict(features)[0])
    return {"predicted_price": price}
