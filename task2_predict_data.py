"""
Task 2: Predict Energy for New Data
Input: CSV with only [BS, Time, load, RUType] (NO Energy column)
Output: CSV with [BS, Time, load, RUType, predicted_energy]

This simulates a real deployment scenario where users provide new data without ground truth.
"""

import pandas as pd
import numpy as np
import os
import pickle

base_dir = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("TASK 2: ENERGY PREDICTION FOR NEW DATA")
print("="*70)

# Load new input data (no Energy column)
input_file = os.path.join(base_dir, 'task2_new_data_input.csv')

if not os.path.exists(input_file):
    print("\nERROR: task2_new_data_input.csv not found!")
    print("Please run task2_split_data.py first to create the input file.")
    exit(1)

df_input = pd.read_csv(input_file)

print(f"\nLoaded new data: {len(df_input):,} rows")
print(f"Columns: {df_input.columns.tolist()}")

# Verify no Energy column exists (real deployment scenario)
if 'Energy' in df_input.columns:
    print("\nWARNING: Energy column found in input! Removing it (deployment simulation).")
    df_input = df_input.drop(columns=['Energy'])

print(f"\nData summary:")
print(f"  Base Stations: {df_input['BS'].nunique()}")
print(f"  RUTypes: {sorted(df_input['RUType'].unique())}")
print(f"  Load range: [{df_input['load'].min():.4f}, {df_input['load'].max():.4f}]")

# Prepare features (same encoding as training)
X = df_input[['load', 'RUType']].copy()
X_encoded = pd.get_dummies(X, columns=['RUType'], drop_first=True)

print(f"\nFeatures after encoding: {list(X_encoded.columns)}")

# Load trained XGBoost model
print("\n" + "="*70)
print("Loading trained XGBoost model...")
print("="*70)

model_file = os.path.join(base_dir, 'trained_models.pkl')
if not os.path.exists(model_file):
    print("ERROR: trained_models.pkl not found!")
    print("Please run task2_ai_model.py first to train the model.")
    exit(1)

with open(model_file, 'rb') as f:
    saved_data = pickle.load(f)
    xgboost_model = saved_data['models']['XGBoost']
    
print("✓ XGBoost model loaded successfully")

# Make predictions
print("\nMaking predictions...")
predictions = xgboost_model.predict(X_encoded)

# Create output dataframe
df_output = df_input.copy()
df_output['predicted_energy'] = predictions

# Save predictions
output_file = os.path.join(base_dir, 'task2_prediction_output.csv')
df_output.to_csv(output_file, index=False)

print("\n" + "="*70)
print("PREDICTION COMPLETE")
print("="*70)
print(f"Output file: task2_prediction_output.csv")
print(f"Total predictions: {len(df_output):,}")

print("\nSample predictions:")
print(df_output.head(10).to_string(index=False))

print("\nPrediction statistics:")
print(f"  Min predicted energy: {predictions.min():.2f} kW")
print(f"  Max predicted energy: {predictions.max():.2f} kW")
print(f"  Mean predicted energy: {predictions.mean():.2f} kW")
print(f"  Std predicted energy: {predictions.std():.2f} kW")

# If ground truth exists, evaluate performance
ground_truth_file = os.path.join(base_dir, 'task2_ground_truth.csv')
if os.path.exists(ground_truth_file):
    print("\n" + "="*70)
    print("EVALUATING PREDICTIONS (ground truth available)")
    print("="*70)
    
    df_truth = pd.read_csv(ground_truth_file)
    
    # Merge predictions with ground truth
    df_eval = df_output.merge(df_truth[['BS', 'Time', 'actual_energy']], 
                              on=['BS', 'Time'], how='left')
    
    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    
    mask = df_eval['actual_energy'].notna()
    y_true = df_eval.loc[mask, 'actual_energy']
    y_pred = df_eval.loc[mask, 'predicted_energy']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nPerformance on test set ({mask.sum():,} samples):")
    print(f"  MAE: {mae:.2f} kW")
    print(f"  RMSE: {rmse:.2f} kW")
    print(f"  R²: {r2:.4f}")
    
    # Save evaluation results
    df_eval['prediction_error'] = df_eval['actual_energy'] - df_eval['predicted_energy']
    df_eval['absolute_error'] = np.abs(df_eval['prediction_error'])
    
    eval_output = os.path.join(base_dir, 'task2_prediction_evaluation.csv')
    df_eval.to_csv(eval_output, index=False)
    print(f"\n✓ Evaluation results saved: task2_prediction_evaluation.csv")
    
    # Error analysis by RUType
    print("\nError analysis by RUType:")
    error_stats = df_eval[df_eval['actual_energy'].notna()].groupby('RUType').agg({
        'absolute_error': ['mean', 'std'],
        'BS': 'count'
    }).round(2)
    error_stats.columns = ['MAE', 'Std', 'Count']
    print(error_stats.to_string())

print("\n" + "="*70)
print("TASK 2 PREDICTION COMPLETE!")
print("="*70)
