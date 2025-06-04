"""
FAANG-Level End-to-End Model Testing Script
This script tests model loading, predictions, edge cases, and expected performance using PyTorch.
Use this as part of your CI pipeline or for pre-deployment sanity testing.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import os
import warnings

# Silence expected warnings
warnings.filterwarnings("ignore")

# Constants
WEIGHTS_PATH = "model_weights.pth"  # Must be pre-trained and saved
ACCURACY_THRESHOLD = 0.90

# Step 1: Load Dataset
def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
    df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
    df['Personality'] = df['Personality'].map({'Introvert': 0, 'Extrovert': 1})
    X = df.drop("Personality", axis=1)
    y = df["Personality"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).unsqueeze(1), scaler

# Step 2: Define Model
class PersonalityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

# Step 3: Load model and weights
def load_model(input_dim):
    model = PersonalityClassifier(input_dim)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()
    return model

# Step 4: Evaluate model on real test data
def evaluate_model(model, X, y):
    with torch.no_grad():
        preds = model(X)
        preds_class = (preds >= 0.5).float()
        acc = (preds_class == y).float().mean().item()
        print(f"\n✅ Model test accuracy: {acc * 100:.2f}%")
        assert acc >= ACCURACY_THRESHOLD, f"Model accuracy {acc:.2f} is below threshold of {ACCURACY_THRESHOLD}"
        return preds

# Step 5: Edge Case Testing
def test_edge_cases(model, input_dim):
    edge_cases = {
        "All-zero input": torch.zeros((1, input_dim)),
        "Large input": torch.full((1, input_dim), 1000.0),
        "Small input": torch.full((1, input_dim), -1000.0)
    }
    print("\n🔍 Edge Case Predictions:")
    for desc, val in edge_cases.items():
        with torch.no_grad():
            output = model(val)
        print(f"{desc}: {output.item():.4f}")

# Step 6: Sanity tests for model properties
def test_model_structure(model):
    assert isinstance(model, nn.Module), "Model must be a PyTorch nn.Module"
    assert hasattr(model, 'forward'), "Model must have a forward method"
    print("\n✅ Model structure sanity checks passed.")

# Step 7: Run all tests
def run_all_tests():
    print("\n🚀 Running FAANG-level model tests...")
    dataset_path = "personality_dataset.csv"  # Adjust if moved
    X, y, scaler = load_data(dataset_path)
    input_dim = X.shape[1]
    model = load_model(input_dim)

    test_model_structure(model)
    evaluate_model(model, X, y)
    test_edge_cases(model, input_dim)
    print("\n🎯 All tests passed successfully.")

if __name__ == "__main__":
    run_all_tests()
