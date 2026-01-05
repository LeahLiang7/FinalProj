"""
Compare AI model performance across different RUTypes
Analyze MAE, RMSE, R2 for each RUType separately in both train and test sets
Loads pre-trained models from task2_ai_model.py to avoid redundant training
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load filtered data
base_dir = os.path.dirname(os.path.abspath(__file__))
filtered_data = pd.read_csv(os.path.join(base_dir, 'filtered_energy_data.csv'))

print(f"Loaded filtered data: {len(filtered_data):,} rows")
print(f"Unique RUTypes: {sorted(filtered_data['RUType'].unique())}")

# Prepare features and target
X = filtered_data[['load', 'RUType']]
y = filtered_data['Energy']
X_encoded = pd.get_dummies(X, columns=['RUType'], drop_first=True)

# Split data randomly (80/20) - same as task2_ai_model.py
# Random split recommended by advisor to avoid temporal bias
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Keep original RUType for grouping
rutype_train = filtered_data.loc[X_train.index, 'RUType']
rutype_test = filtered_data.loc[X_test.index, 'RUType']

print(f"\nTraining set: {len(X_train):,} samples ({len(X_train)/len(X_encoded)*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X_encoded)*100:.1f}%)")

# Load pre-trained models
print("\n" + "="*70)
print("Loading pre-trained models from task2_ai_model.py...")
print("="*70)

model_file = os.path.join(base_dir, 'trained_models.pkl')
if not os.path.exists(model_file):
    print(f"\nERROR: trained_models.pkl not found!")
    print("Please run task2_ai_model.py first to train and save the models.")
    exit(1)

with open(model_file, 'rb') as f:
    saved_data = pickle.load(f)
    models = saved_data['models']
    model_predictions = saved_data['predictions']
    
print(f"Loaded {len(models)} pre-trained models: {list(models.keys())}")
print("Using cached predictions from task2_ai_model.py")

# Analyze performance by RUType
print("\n" + "="*70)
print("Analyzing performance by RUType...")
print("="*70)

rutypes = sorted(filtered_data['RUType'].unique())

# Create results dataframe for each model
results_list = []

for model_name in models.keys():
    y_train_pred = model_predictions[model_name]['y_train_pred']
    y_test_pred = model_predictions[model_name]['y_test_pred']
    
    print(f"\n{model_name}:")
    print("-" * 70)
    
    for rutype in rutypes:
        # Train set metrics
        train_idx = rutype_train == rutype
        if train_idx.sum() > 0:
            train_mae = mean_absolute_error(y_train[train_idx], y_train_pred[train_idx])
            train_rmse = np.sqrt(mean_squared_error(y_train[train_idx], y_train_pred[train_idx]))
            train_r2 = r2_score(y_train[train_idx], y_train_pred[train_idx])
            train_n = train_idx.sum()
        else:
            train_mae = train_rmse = train_r2 = train_n = 0
        
        # Test set metrics
        test_idx = rutype_test == rutype
        if test_idx.sum() > 0:
            test_mae = mean_absolute_error(y_test[test_idx], y_test_pred[test_idx])
            test_rmse = np.sqrt(mean_squared_error(y_test[test_idx], y_test_pred[test_idx]))
            test_r2 = r2_score(y_test[test_idx], y_test_pred[test_idx])
            test_n = test_idx.sum()
        else:
            test_mae = test_rmse = test_r2 = test_n = 0
        
        print(f"  {rutype}: Train(n={train_n}) MAE={train_mae:.2f}, RMSE={train_rmse:.2f}, R2={train_r2:.4f} | "
              f"Test(n={test_n}) MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R2={test_r2:.4f}")
        
        results_list.append({
            'Model': model_name,
            'RUType': rutype,
            'Train_n': train_n,
            'Train_MAE': train_mae,
            'Train_RMSE': train_rmse,
            'Train_R2': train_r2,
            'Test_n': test_n,
            'Test_MAE': test_mae,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2
        })

# Save detailed results
results_df = pd.DataFrame(results_list)
results_df.to_csv(os.path.join(base_dir, 'rutype_performance_comparison.csv'), index=False)
print(f"\nDetailed results saved to: rutype_performance_comparison.csv")

# Create visualization: MAE comparison by RUType
print("\nCreating visualizations...")

# 1. MAE comparison across all models and RUTypes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
models_list = list(models.keys())

for idx, model_name in enumerate(models_list):
    ax = axes[idx // 2, idx % 2]
    
    model_data = results_df[results_df['Model'] == model_name]
    
    x = np.arange(len(rutypes))
    width = 0.35
    
    train_mae = [model_data[model_data['RUType'] == rt]['Train_MAE'].values[0] for rt in rutypes]
    test_mae = [model_data[model_data['RUType'] == rt]['Test_MAE'].values[0] for rt in rutypes]
    
    bars1 = ax.bar(x - width/2, train_mae, width, label='Train (Hours 1-48)', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_mae, width, label='Test (Hours 49-72)', color='lightcoral', alpha=0.8)
    
    ax.set_title(f'{model_name} - MAE by RUType', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
    ax.set_xlabel('RUType', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(rutypes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'rutype_mae_comparison.png'), dpi=150)
plt.close()
print("Saved: rutype_mae_comparison.png")

# 2. R2 comparison across all models and RUTypes
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, model_name in enumerate(models_list):
    ax = axes[idx // 2, idx % 2]
    
    model_data = results_df[results_df['Model'] == model_name]
    
    x = np.arange(len(rutypes))
    width = 0.35
    
    train_r2 = [model_data[model_data['RUType'] == rt]['Train_R2'].values[0] for rt in rutypes]
    test_r2 = [model_data[model_data['RUType'] == rt]['Test_R2'].values[0] for rt in rutypes]
    
    bars1 = ax.bar(x - width/2, train_r2, width, label='Train (Hours 1-48)', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_r2, width, label='Test (Hours 49-72)', color='lightcoral', alpha=0.8)
    
    ax.set_title(f'{model_name} - R2 by RUType', fontsize=12, fontweight='bold')
    ax.set_ylabel('R2 Score', fontsize=11)
    ax.set_xlabel('RUType', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(rutypes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'rutype_r2_comparison.png'), dpi=150)
plt.close()
print("Saved: rutype_r2_comparison.png")

# 3. Sample count by RUType
fig, ax = plt.subplots(figsize=(12, 6))

model_data = results_df[results_df['Model'] == models_list[0]]  # Use any model, counts are the same
x = np.arange(len(rutypes))
width = 0.35

train_counts = [model_data[model_data['RUType'] == rt]['Train_n'].values[0] for rt in rutypes]
test_counts = [model_data[model_data['RUType'] == rt]['Test_n'].values[0] for rt in rutypes]

bars1 = ax.bar(x - width/2, train_counts, width, label='Train (Hours 1-48)', color='skyblue', alpha=0.8)
bars2 = ax.bar(x + width/2, test_counts, width, label='Test (Hours 49-72)', color='lightcoral', alpha=0.8)

ax.set_title('Sample Distribution by RUType', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=11)
ax.set_xlabel('RUType', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(rutypes, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height):,}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'rutype_sample_distribution.png'), dpi=150)
plt.close()
print("Saved: rutype_sample_distribution.png")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY: Best and Worst Performing RUTypes (Test Set MAE)")
print("="*70)

for model_name in models_list:
    model_data = results_df[(results_df['Model'] == model_name) & (results_df['Test_n'] > 0)]
    
    best_rutype = model_data.loc[model_data['Test_MAE'].idxmin()]
    worst_rutype = model_data.loc[model_data['Test_MAE'].idxmax()]
    
    print(f"\n{model_name}:")
    print(f"  Best:  {best_rutype['RUType']} (MAE={best_rutype['Test_MAE']:.2f}, R2={best_rutype['Test_R2']:.4f}, n={int(best_rutype['Test_n'])})")
    print(f"  Worst: {worst_rutype['RUType']} (MAE={worst_rutype['Test_MAE']:.2f}, R2={worst_rutype['Test_R2']:.4f}, n={int(worst_rutype['Test_n'])})")

# 4. Generate train vs test prediction scatter plots for each RUType (like task2_train_test_predictions.png)
print("\n4. Generating prediction scatter plots by RUType...")
print("-" * 70)

rutypes = sorted(filtered_data['RUType'].unique())
n_rutypes = len(rutypes)

# Reset index for rutype arrays to match predictions
rutype_train = rutype_train.reset_index(drop=True)
rutype_test = rutype_test.reset_index(drop=True)

# Create one plot per model, showing all RUTypes
for model_name in models_list:
    y_train_pred = model_predictions[model_name]['y_train_pred']
    y_test_pred = model_predictions[model_name]['y_test_pred']
    
    # Create figure with 2 rows (train/test) and n_rutypes columns
    fig, axes = plt.subplots(2, n_rutypes, figsize=(4*n_rutypes, 8))
    
    for idx, rutype in enumerate(rutypes):
        # Train set for this RUType
        train_idx = (rutype_train == rutype).values
        test_idx = (rutype_test == rutype).values
        
        # Train plot
        ax_train = axes[0, idx] if n_rutypes > 1 else axes[0]
        
        if train_idx.sum() > 0:
            y_train_true = y_train.iloc[train_idx]
            y_train_pred_rt = y_train_pred[train_idx]
            
            train_r2 = r2_score(y_train_true, y_train_pred_rt)
            train_mae = mean_absolute_error(y_train_true, y_train_pred_rt)
            
            ax_train.scatter(y_train_true, y_train_pred_rt, color='skyblue', alpha=0.4, s=10)
            ax_train.plot([y_train_true.min(), y_train_true.max()], 
                         [y_train_true.min(), y_train_true.max()], 
                         color='red', linestyle='--', linewidth=2)
            ax_train.set_title(f'{rutype} - Train\nR2={train_r2:.4f}, MAE={train_mae:.2f}\nn={train_idx.sum()}', 
                              fontsize=9, fontweight='bold')
        else:
            ax_train.set_title(f'{rutype} - Train\nNo data', fontsize=9, fontweight='bold')
        
        ax_train.set_xlabel('True Energy', fontsize=8)
        ax_train.set_ylabel('Predicted Energy', fontsize=8)
        ax_train.grid(alpha=0.3)
        
        # Test plot
        ax_test = axes[1, idx] if n_rutypes > 1 else axes[1]
        
        if test_idx.sum() > 0:
            y_test_true = y_test.iloc[test_idx]
            y_test_pred_rt = y_test_pred[test_idx]
            
            test_r2 = r2_score(y_test_true, y_test_pred_rt)
            test_mae = mean_absolute_error(y_test_true, y_test_pred_rt)
            
            ax_test.scatter(y_test_true, y_test_pred_rt, color='lightcoral', alpha=0.4, s=10)
            ax_test.plot([y_test_true.min(), y_test_true.max()], 
                        [y_test_true.min(), y_test_true.max()], 
                        color='red', linestyle='--', linewidth=2)
            ax_test.set_title(f'{rutype} - Test\nR2={test_r2:.4f}, MAE={test_mae:.2f}\nn={test_idx.sum()}', 
                             fontsize=9, fontweight='bold')
        else:
            ax_test.set_title(f'{rutype} - Test\nNo data', fontsize=9, fontweight='bold')
        
        ax_test.set_xlabel('True Energy', fontsize=8)
        ax_test.set_ylabel('Predicted Energy', fontsize=8)
        ax_test.grid(alpha=0.3)
    
    plt.suptitle(f'{model_name} - Predictions by RUType', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filename = f'rutype_predictions_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(os.path.join(base_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

print("\n" + "="*70)
print("SUMMARY: Best and Worst Performing RUTypes (Test Set MAE)")
print("="*70)

for model_name in models_list:
    model_data = results_df[(results_df['Model'] == model_name) & (results_df['Test_n'] > 0)]
    
    best_rutype = model_data.loc[model_data['Test_MAE'].idxmin()]
    worst_rutype = model_data.loc[model_data['Test_MAE'].idxmax()]
    
    print(f"\n{model_name}:")
    print(f"  Best:  {best_rutype['RUType']} (MAE={best_rutype['Test_MAE']:.2f}, R2={best_rutype['Test_R2']:.4f}, n={int(best_rutype['Test_n'])})")
    print(f"  Worst: {worst_rutype['RUType']} (MAE={worst_rutype['Test_MAE']:.2f}, R2={worst_rutype['Test_R2']:.4f}, n={int(worst_rutype['Test_n'])})")

print("\n" + "="*70)
print("Done! Check the output files for detailed analysis.")
print("="*70)
