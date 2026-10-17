import requests

URL = "https://census-salary-api.onrender.com/predict"

sample = {
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

response = requests.post(URL, json=sample, timeout=90)
print("Status code:", response.status_code)
print("Prediction:", response.json()["prediction"])
