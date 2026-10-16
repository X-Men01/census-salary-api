import joblib
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    clf = RandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=0
    )
    return clf.fit(X_train, y_train)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def performance_on_slices(df, feature, model, encoder, lb, cat_features):
    """Compute metrics for each unique value of a categorical feature.

    Returns one formatted line per slice.
    """
    lines = []
    for value in df[feature].unique():
        slice_df = df[df[feature] == value]
        X, y, _, _ = process_data(
            slice_df, categorical_features=cat_features, label="salary",
            training=False, encoder=encoder, lb=lb,
        )
        preds = inference(model, X)
        p, r, f1 = compute_model_metrics(y, preds)
        lines.append(
            f"{feature}={value} n={len(X)} P={p:.3f} R={r:.3f} F1={f1:.3f}"
        )
    return lines


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save(obj, path):
    """Save a model, encoder or label binarizer to `path` with joblib."""
    joblib.dump(obj, path)


def load(path):
    """Load an object saved with `save`."""
    return joblib.load(path)
