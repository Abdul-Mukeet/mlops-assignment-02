import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # example model
from sklearn.metrics import accuracy_score  # metric
import joblib
import json

# Load dataset
data = pd.read_csv('data/dataset.csv')
X = data.drop('target', axis=1)  # replace 'target' with your actual target column
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.pkl')

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save metrics
metrics = {'accuracy': accuracy}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("Training completed. Model saved as model.pkl and metrics saved as metrics.json")
