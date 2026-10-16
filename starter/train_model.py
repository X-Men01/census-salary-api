# Script to train machine learning model.

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics, inference, save, train_model
)
# Add the necessary imports for the starter code.

# Load the data from the project root.
data_path = Path(__file__).resolve().parents[1] / "data" / "census.csv"
data = pd.read_csv(data_path, skipinitialspace=True)

# Optional: use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=0)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb,
)
# Train and save a model.

model = train_model(X_train, y_train)

y_preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

print(f'Precision {precision}, recall {recall}, fbeta {fbeta}')

model_path = Path(__file__).resolve().parents[1] / "model" / "model.joblib"

save(model, model_path)

encoder_path = Path(__file__).resolve().parents[1] / "model" / "encoder.joblib"

save(encoder, encoder_path)

lb_path = Path(__file__).resolve().parents[1] / "model" / "lb.joblib"

save(lb, lb_path)
