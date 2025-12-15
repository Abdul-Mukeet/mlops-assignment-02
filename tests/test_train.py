import os
import sys
import pandas as pd
from src.train import train_model
from sklearn.linear_model import LogisticRegression

# Add project root to Python path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)


def test_data_loading():
    data_path = "data/dataset.csv"
    data = pd.read_csv(data_path)
    assert not data.empty, "Dataset is empty"
    assert "label" in data.columns, "Target column 'label' missing"


def test_model_training():
    X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    y = pd.Series([0, 1])
    df = X.copy()
    df["label"] = y
    df.to_csv("data/temp_dataset.csv", index=False)

    model = train_model("data/temp_dataset.csv")
    assert isinstance(model, LogisticRegression)

    os.remove("data/temp_dataset.csv")


def test_model_prediction_shape():
    X = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    y = pd.Series([0, 1])
    df = X.copy()
    df["label"] = y
    df.to_csv("data/temp_dataset.csv", index=False)

    model = train_model("data/temp_dataset.csv")
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape

    os.remove("data/temp_dataset.csv")
