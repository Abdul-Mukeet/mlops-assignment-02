import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = "models/model.pkl"


def train_model(csv_path: str):
    """Train logistic regression model and save to MODEL_PATH."""
    data = pd.read_csv(csv_path)
    target_column = "label"
    X = data.drop(columns=target_column)
    y = data[target_column]

    model = LogisticRegression()
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


if __name__ == "__main__":
    train_model("data/dataset.csv")
    print(f"Training completed, model saved as {MODEL_PATH}")
