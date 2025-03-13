import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset_path = r"C:\Users\krish\Downloads\archive\robot_inverse_kinematics_dataset.csv"  # Update this with the actual file path
data = pd.read_csv(dataset_path)

# Check for missing values
data = data.dropna()  # Removes any rows with missing values

# Define features (joint angles) and target (end-effector position)
X = data[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']]  # Input: Joint angles
y = data[['x', 'y', 'z']]  # Output: End-effector position

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Visualize Predictions vs. Actual Values (for x-coordinate as an example)
plt.figure(figsize=(8, 5))
plt.scatter(y_test['x'], y_pred[:, 0], alpha=0.7, label="Predicted vs. Actual X")
plt.xlabel("Actual X Position")
plt.ylabel("Predicted X Position")
plt.title("Predicted vs. Actual End-Effector X Position")
plt.legend()
plt.show()
