import os
import pandas as pd


def test_data_loading():
    base_dir = os.path.dirname(__file__)
    # Use sample dataset for CI
    data_path = os.path.join(base_dir, "..", "data", "sample_dataset.csv")
    data = pd.read_csv(data_path)
    assert not data.empty
