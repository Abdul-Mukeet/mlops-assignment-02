import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib


MODEL_PATH = "models/model.pkl"


def get_feature_count(csv_path):
    """Return number of features (exclude target column) from CSV."""
    data = pd.read_csv(csv_path)
    return data.shape[1] - 1  # exclude target column


def train_model(csv_path):
    """Train a Logistic Regression model and save it to disk."""
    data = pd.read_csv(csv_path)
    target_column = "label"  # change if needed
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    model = LogisticRegression()
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model


def main():
    """Main function to train the model."""
    train_model("data/dataset.csv")
    print(f"Training completed, model saved as {MODEL_PATH}")


if __name__ == "__main__":
    main()
