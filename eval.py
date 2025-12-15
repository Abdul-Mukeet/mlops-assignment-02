import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Replace 'label' with your actual target column name
target_column = "label"
X = data.drop(columns=[target_column])
y = data[target_column]

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X)

# Calculate regression metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Save metrics to a file
metrics = {"MSE": mse, "MAE": mae, "R2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
print("Evaluation completed. Metrics saved in metrics.json")
