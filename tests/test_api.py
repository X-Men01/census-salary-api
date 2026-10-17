from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE = {
    "age": 39,
    "capital-gain": 2174,
    "capital-loss": 0,
    "education": "Bachelors",
    "education-num": 13,
    "fnlgt": 77516,
    "hours-per-week": 40,
    "marital-status": "Never-married",
    "native-country": "United-States",
    "occupation": "Adm-clerical",
    "race": "White",
    "relationship": "Not-in-family",
    "sex": "Male",
    "workclass": "State-gov",
}


def test_greeting():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'greeting': 'Hello'}


def test_predict_low():
    response = client.post('/predict', json=SAMPLE)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_predict_high():
    high = SAMPLE | {
        "age": 45,
        "capital-gain": 15000,
        "hours-per-week": 50,
        "marital-status": "Married-civ-spouse",
        "relationship": "Husband",
    }
    response = client.post('/predict', json=high)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
