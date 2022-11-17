import joblib
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from api.route.train import preprocessor, runTrain

routes = APIRouter()

class ResponseModel(BaseModel):
    status      : Optional[bool] = True
    description : Optional[str]  = ""

class RequestModel(BaseModel):
    text        : Optional[str] = ""

def LoadModel(filename):
    with open(filename, 'rb') as f:
        return joblib.load(f)

    return None

@routes.post("/train")
def train():
    return runTrain("api/route/storage/dataset.csv")

@routes.post("/predict")
def predict(req: RequestModel):

    model       = LoadModel("api/models/best_model.pkl")
    massage     = preprocessor(req.text)
    prediction  = True if model.predict([massage]) == "ham" else False

    return ResponseModel.parse_obj({"status": prediction, "description": "Geçerli bir posta içeriği" if prediction else "Sahte Posta"})



