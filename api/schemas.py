from pydantic import BaseModel

class PredictRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class PredictResponse(BaseModel):
    prediction: int        # 0 = <=50K, 1 = >50K
    probability: float     # confidence score