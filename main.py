from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.ml.model import inference, load

MODEL_DIR = Path(__file__).resolve().parent / "model"
CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

model = load(MODEL_DIR / 'model.joblib')
encoder = load(MODEL_DIR / 'encoder.joblib')
lb = load(MODEL_DIR / 'lb.joblib')


class CensusRecord(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }]
        },
    )


app = FastAPI()


@app.get("/")
def welcome() -> dict[str, str]:
    return {'greeting': 'Hello'}


@app.post("/predict")
def predict(record: CensusRecord) -> dict[str, str]:
    df = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        df, CAT_FEATURES, training=False, encoder=encoder, lb=lb
    )
    pred = inference(model, X)
    return {"prediction": lb.inverse_transform(pred)[0]}
