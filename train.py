import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import os

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Replace 'label' with your actual target column name
target_column = 'label'
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found. Columns: {data.columns.tolist()}")

X = data.drop(target_column, axis=1)
y = data[target_column]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training completed, model saved as models/model.pkl")
