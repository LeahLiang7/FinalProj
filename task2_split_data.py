"""
Split Task 2 data into training set (80%) and new unseen data (20%)
Training set: already used in task2_ai_model.py
Test set: simulates new user input (no Energy column)
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load full data with Energy
filtered_data = pd.read_csv(os.path.join(base_dir, 'filtered_energy_data.csv'))

print("="*70)
print("SPLITTING DATA FOR TASK 2")
print("="*70)
print(f"Total data: {len(filtered_data):,} rows")

# Prepare features
X = filtered_data[['BS', 'Time', 'load', 'RUType']]
y = filtered_data['Energy']

# Split 80/20 with same random_state as task2_ai_model.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: {len(X_train):,} rows (80%)")
print(f"Test set: {len(X_test):,} rows (20%)")

# Save training set (with Energy)
train_data = X_train.copy()
train_data['Energy'] = y_train
output_train = os.path.join(base_dir, 'task2_train_data.csv')
train_data.to_csv(output_train, index=False)
print(f"\n✓ Saved training data: task2_train_data.csv")

# Save test set WITHOUT Energy (simulates new user input)
test_input = X_test.copy()
output_test_input = os.path.join(base_dir, 'task2_new_data_input.csv')
test_input.to_csv(output_test_input, index=False)
print(f"✓ Saved new data input (no Energy): task2_new_data_input.csv")

# Save ground truth separately for evaluation
ground_truth = X_test.copy()
ground_truth['actual_energy'] = y_test
output_ground_truth = os.path.join(base_dir, 'task2_ground_truth.csv')
ground_truth.to_csv(output_ground_truth, index=False)
print(f"✓ Saved ground truth (for evaluation): task2_ground_truth.csv")

print("\n" + "="*70)
print("DATA SPLIT COMPLETE")
print("="*70)
print("\nNext steps:")
print("1. Training is already done (trained_models.pkl exists)")
print("2. Run task2_predict_new_data.py with task2_new_data_input.csv")
print("3. Compare predictions with task2_ground_truth.csv")
