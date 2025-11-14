# src/service_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel

from .infer import MedicalSpecialtyPredictor

app = FastAPI(title="Medical Specialty Classifier", version="0.1.0")
predictor = MedicalSpecialtyPredictor("artifacts")


class PredictIn(BaseModel):
    text: str
    topk: int = 3


class PredictOut(BaseModel):
    labels: list[str]


@app.post("/predict", response_model=PredictOut)
def predict(item: PredictIn):
    labels = predictor.predict(item.text, topk=item.topk)
    if isinstance(labels, str):
        labels = [labels]
    elif isinstance(labels, list) and labels and isinstance(labels[0], list):
        labels = labels[0]
    labels = [str(x) for x in labels]
    return PredictOut(labels=labels)
