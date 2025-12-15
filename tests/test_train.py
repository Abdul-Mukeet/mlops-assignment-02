import os
import pytest
import pandas as pd
import joblib
from train import train_model  # assuming you have train_model function

DATA_PATH = "data/dataset.csv"
MODEL_PATH = "models/model.pkl"

def test_data_loading():
    assert os.path.exists(DATA_PATH), "Dataset not found"
    data = pd.read_csv(DATA_PATH)
    assert not data.empty, "Dataset is empty"

def test_model_training():
    model = train_model(DATA_PATH)
    assert model is not None, "Model training failed"

def test_model_saving():
    assert os.path.exists(MODEL_PATH), "Model file not saved"

def test_model_shape():
    model = joblib.load(MODEL_PATH)
    # example: check number of features (adjust to your dataset)
    from train import get_feature_count
    assert model.n_features_in_ == get_feature_count(DATA_PATH)
