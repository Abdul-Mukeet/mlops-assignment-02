import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

MODEL_PATH = "models/model.pkl"

def get_feature_count(csv_path):
    data = pd.read_csv(csv_path)
    return data.shape[1] - 1  # exclude target column

def train_model(csv_path):
    data = pd.read_csv(csv_path)
    target_column = 'label'  # change if needed
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    model = LogisticRegression()
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

if __name__ == "__main__":
    train_model("data/dataset.csv")
    print(f"Training completed, model saved as {MODEL_PATH}")
