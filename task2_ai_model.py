"""
Task 2: Simple AI models that predict the energy usage from load and type only.
Comparing: Linear Regression, Decision Tree, Random Forest, XGBoost
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


# Use the cleaned, filtered dataset for modeling
base_dir = os.path.dirname(os.path.abspath(__file__))
filtered_data = pd.read_csv(os.path.join(base_dir, 'filtered_energy_data.csv'))

print(f"Loaded filtered data: {len(filtered_data):,} rows")
print(f"Unique BS count: {filtered_data['BS'].nunique()}")
print(f"Time range: Hours {filtered_data['Hours'].min()} to {filtered_data['Hours'].max()}")

# Prepare data for AI model
print("Preparing data for AI model")
X = filtered_data[['load', 'RUType']]
y = filtered_data['Energy']

# Encode categorical variable RUType
X = pd.get_dummies(X, columns=['RUType'], drop_first=True)

# Check if X and y are valid before splitting
if X.empty or y.empty:
    raise ValueError("Input features X or target y are empty. Please check the filtered data.")

if len(X) != len(y):
    raise ValueError(f"Mismatch in lengths: X has {len(X)} rows, but y has {len(y)} rows.")

print(f"Dataset size before splitting: {len(X)} samples")

# Split data randomly into training (80%) and testing (20%) sets
# Random split recommended by advisor to avoid temporal pattern bias
# (e.g., weekday/weekend differences, weather changes across days)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# Define models to train
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
}

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, return predictions and metrics"""
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return y_train_pred, y_test_pred, metrics

# Train all models and collect results

print("\nTraining and Evaluating Models")

results = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    y_train_pred, y_test_pred, metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[model_name] = {
        'model': model,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'metrics': metrics
    }
    
    print(f"  Train - MAE: {metrics['train_mae']:.2f}, MAPE: {metrics['train_mape']:.2f}%, RMSE: {metrics['train_rmse']:.2f}, R²: {metrics['train_r2']:.4f}")
    print(f"  Test  - MAE: {metrics['test_mae']:.2f}, MAPE: {metrics['test_mape']:.2f}%, RMSE: {metrics['test_rmse']:.2f}, R²: {metrics['test_r2']:.4f}")

# Save trained models for reuse
import pickle
print("\nSaving trained models...")
with open(os.path.join(base_dir, 'trained_models.pkl'), 'wb') as f:
    pickle.dump({
        'models': {name: result['model'] for name, result in results.items()},
        'predictions': {name: {'y_train_pred': result['y_train_pred'], 
                               'y_test_pred': result['y_test_pred']} 
                       for name, result in results.items()},
        'data_info': {
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'feature_names': list(X_train.columns)
        }
    }, f)
print("Saved: trained_models.pkl")

# Print Summary Table
print("\n" + "="*90)
print("MODEL PERFORMANCE SUMMARY (TEST SET)")
print("="*90)
print(f"{'Model':<20} {'MAE':>12} {'MAPE':>12} {'RMSE':>12} {'R²':>12}")
print("-"*90)
for model_name, result in results.items():
    metrics = result['metrics']
    print(f"{model_name:<20} {metrics['test_mae']:>12.2f} {metrics['test_mape']:>11.2f}% {metrics['test_rmse']:>12.2f} {metrics['test_r2']:>12.4f}")
print("="*90)
print("Note: Lower MAE, MAPE, and RMSE indicate better performance. Higher R² indicates better fit.")
print("="*90 + "\n")

# Save metrics comparison
metrics_df = pd.DataFrame({
    model_name: result['metrics'] 
    for model_name, result in results.items()
}).T
metrics_df.to_csv(os.path.join(base_dir, 'task2_models_comparison.csv'))
print(f"\nModel comparison saved to: task2_models_comparison.csv")

# 1. Create evaluation metrics comparison plot for all models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metric_names = ['MAE', 'RMSE', 'R²']
metric_keys = [('train_mae', 'test_mae'), ('train_rmse', 'test_rmse'), ('train_r2', 'test_r2')]

for idx, (ax, metric_name, (train_key, test_key)) in enumerate(zip(axes, metric_names, metric_keys)):
    model_names = list(results.keys())
    train_vals = [results[m]['metrics'][train_key] for m in model_names]
    test_vals = [results[m]['metrics'][test_key] for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, train_vals, width, label='Train', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, test_vals, width, label='Test', color='lightcoral', alpha=0.8)
    
    ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task2_evaluation_metrics.png'), dpi=150)
plt.close()
print("Saved: task2_evaluation_metrics.png")

# 2. Create prediction vs. true value plots for all models (Test Set Only)
n_models = len(results)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5.5), squeeze=False) # squeeze=False to always get a 2D array

for idx, (model_name, result) in enumerate(results.items()):
    y_test_pred = result['y_test_pred']
    metrics = result['metrics']
    
    # Test predictions
    ax_test = axes[0, idx]
    ax_test.scatter(y_test, y_test_pred, color='lightcoral', alpha=0.3, s=5, label='Test Predictions')
    ax_test.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax_test.set_title(f'{model_name}\nTest: R²={metrics["test_r2"]:.4f}, MAE={metrics["test_mae"]:.2f}', 
                     fontsize=10, fontweight='bold')
    ax_test.set_xlabel('True Energy (kW)')
    ax_test.set_ylabel('Predicted Energy (kW)')
    ax_test.grid(alpha=0.3)
    ax_test.legend()
    ax_test.set_aspect('equal', adjustable='box')

fig.suptitle('Model Performance on Test Set (Generalization Ability)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task2_test_set_predictions.png'), dpi=150)
plt.close()
print("Saved: task2_test_set_predictions.png")

# 3. Create residuals plot for all models (test set)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))

for idx, (model_name, result) in enumerate(results.items()):
    y_test_pred = result['y_test_pred']
    residuals = y_test - y_test_pred
    
    ax = axes[idx] if n_models > 1 else axes
    ax.scatter(y_test_pred, residuals, color='purple', alpha=0.3, s=5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title(f'{model_name}\nResiduals (Test Set)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Energy')
    ax.set_ylabel('Residuals')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task2_residuals.png'), dpi=150)
plt.close()
print("Saved: task2_residuals.png")

# 4. Create feature importance plot (for tree-based models)
fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))

for idx, (model_name, result) in enumerate(results.items()):
    model = result['model']
    ax = axes[idx] if n_models > 1 else axes
    
    # Get feature importance based on model type
    if hasattr(model, 'coef_'):  # Linear Regression
        importance = model.coef_
        colors = ['green' if c > 0 else 'red' for c in importance]
    elif hasattr(model, 'feature_importances_'):  # Tree-based models
        importance = model.feature_importances_
        colors = 'steelblue'
    
    ax.barh(X.columns, importance, color=colors)
    if hasattr(model, 'coef_'):
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title(f'{model_name}\nFeature Importance', fontsize=10, fontweight='bold')
    ax.set_xlabel('Importance Value')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task2_feature_importance.png'), dpi=150)
plt.close()
print("Saved: task2_feature_importance.png")

print("Done.")