import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import PredictRequest, PredictResponse
from features import engineer_features

app = FastAPI(title="Adult Income Classifier")

# Load model from file
model = joblib.load(os.path.join(os.path.dirname(__file__), 'model.pkl'))

@app.get("/")
def root():
    return {"status": "ok", "model": "adult-income-classifier"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    df = pd.DataFrame([{
        "age":            request.age,
        "workclass":      request.workclass,
        "fnlwgt":         request.fnlwgt,
        "education":      request.education,
        "marital-status": request.marital_status,
        "occupation":     request.occupation,
        "relationship":   request.relationship,
        "race":           request.race,
        "sex":            request.sex,
        "capital-gain":   request.capital_gain,
        "capital-loss":   request.capital_loss,
        "hours-per-week": request.hours_per_week,
        "native-country": request.native_country,
    }])

    df = engineer_features(df)
    prediction  = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    return PredictResponse(prediction=prediction, probability=probability)