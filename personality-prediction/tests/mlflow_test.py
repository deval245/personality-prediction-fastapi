import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model import PersonalityNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow.pytorch

"""
FAANG-Level Test Script for Personality Classification Model
Covers: Dataset loading, preprocessing, model validation, edge case testing, and MLflow model testing
"""

# Constants
MODEL_PATH = "models/personality_model.pth"
TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "PersonalityPrediction"

# Setup MLflow tracking
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load dataset
df = pd.read_csv("data/personality_dataset_clean.csv")

# Preprocessing
df = df.dropna()
df = df.replace({"Yes": 1, "No": 0, "Introvert": 1, "Extrovert": 0})

X = df.drop("Personality", axis=1)
y = df["Personality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to tensor
test_features = torch.tensor(X_test, dtype=torch.float32)
test_labels = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

test_dataset = TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model
model = PersonalityNet(input_size=X.shape[1])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# MLflow model logging for testing
with mlflow.start_run(run_name="TestModel"):
    mlflow.log_param("Test Batch Size", 32)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    mlflow.log_metric("Test Accuracy", accuracy)
    print(f"✅ Model test accuracy: {accuracy:.2f}%")

    # Edge case tests
    edge_cases = [
        torch.zeros((1, X.shape[1])),
        torch.ones((1, X.shape[1])) * 100,
        torch.ones((1, X.shape[1])) * -100
    ]

    for idx, case in enumerate(edge_cases):
        pred = model(case)
        print(f"Edge Case {idx + 1} Prediction: {pred.item():.4f}")

    # Log model to MLflow
    mlflow.pytorch.log_model(model, "personality_model")
