import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

class DriverFeatures(BaseModel):
    driverId: int
    days_since_last_trip: int = None
    completed_deliveries_last_30_days: int
    avg_rating_last_30_days: float
    total_earnings_last_30_days: float
    wallet_balance: float
    tenure_in_days: int


app = FastAPI(title="Driver Churn Prediction API")
model = joblib.load("../models/tuned_XGBoost.joblib")

@app.post("/predict_batch", response_model=List[int])
def predict_batch(drivers: List[DriverFeatures]):

    #Threshhold value from analysis
    THRESHOLD = 0.41

    input_df = pd.DataFrame([driver.dict() for driver in drivers])


    feature_columns = [
        'days_since_last_trip', 'completed_deliveries_last_30_days',
        'avg_rating_last_30_days', 'total_earnings_last_30_days',
        'wallet_balance', 'tenure_in_days'
    ]


    X_predict = input_df[feature_columns]

    
    
    probabilities = model.predict_proba(X_predict)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    results = []
    for index, prediction in enumerate(predictions):
        if bool(prediction):
            driver_id = input_df['driverId'][index]
            results.append(driver_id)

    print(results)
    return results