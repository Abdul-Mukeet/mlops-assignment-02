import os
import pandas as pd


def test_data_loading():
    # Make path relative to test file
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "data", "dataset.csv")

    data = pd.read_csv(data_path)
    assert not data.empty
