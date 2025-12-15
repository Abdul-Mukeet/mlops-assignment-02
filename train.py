# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# ---------------------------
# 1. Load dataset
# ---------------------------
data_path = "data/dataset.csv"  # path to your CSV
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

data = pd.read_csv(data_path)

# ---------------------------
# 2. Define features and target
# ---------------------------
# Use the correct target column from your dataset
target_column = "label"

if target_column not in data.columns:
    raise ValueError(
        f"Target column '{target_column}' not found in dataset. "
        f"Columns available: {data.columns.tolist()}"
    )

X = data.drop(target_column, axis=1)
y = data[target_column]

# ---------------------------
# 3. Split into train and test sets
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 4. Train model
# ---------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------
# 5. Save trained model
# ---------------------------
output_path = "model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model, f)

print(f"Training completed, model saved as {output_path}")
