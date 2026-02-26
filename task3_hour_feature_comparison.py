"""
Task 3: Compare impact of hour feature on early morning anomalies
Version A: load + RUType (Task 2 baseline)
Version B: load + RUType + hour (Task 3 with time)
Focus: Does adding hour improve prediction for Type4 at hours 0-5?
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb

base_dir = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("TASK 3: HOUR FEATURE IMPACT ANALYSIS")
print("="*70)

# ⚠️ Use filtered_energy_data.csv (clean original data without pre-computed stats)
df = pd.read_csv(os.path.join(base_dir, 'filtered_energy_data.csv'))

# Extract hour feature from Time column
df['Time'] = pd.to_datetime(df['Time'])
df['hour'] = df['Time'].dt.hour  # Extract hour of day (0-23)

print(f"\nLoaded data: {len(df):,} rows")
print(f"Available features: {df.columns.tolist()}")
print(f"\n✓ Hour feature corrected: now ranges 0-23 (was 1-72)")

# Prepare features for both versions
# Version A: load + RUType (Task 2 baseline)
X_v1 = df[['load', 'RUType']].copy()
X_v1_encoded = pd.get_dummies(X_v1, columns=['RUType'], drop_first=True)

# Version B: load + RUType + hour
X_v2 = df[['load', 'RUType', 'hour']].copy()
X_v2_encoded = pd.get_dummies(X_v2, columns=['RUType'], drop_first=True)

y = df['Energy']

# Keep metadata for analysis
df_meta = df[['BS', 'Time', 'Hours', 'RUType', 'hour']].copy()

print(f"\nVersion A features: {list(X_v1_encoded.columns)}")
print(f"Version B features: {list(X_v2_encoded.columns)}")

# Split data (same as Task 2)
X_v1_train, X_v1_test, y_train, y_test = train_test_split(
    X_v1_encoded, y, test_size=0.2, random_state=42
)
X_v2_train, X_v2_test, _, _ = train_test_split(
    X_v2_encoded, y, test_size=0.2, random_state=42
)

# Keep metadata for test set
_, meta_test, _, _ = train_test_split(
    df_meta, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {len(X_v1_train):,} samples")
print(f"Test set: {len(X_v1_test):,} samples")

# Train Version A (Task 2 baseline)
print("\n" + "="*70)
print("Training Version A: load + RUType")
print("="*70)

model_v1 = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
model_v1.fit(X_v1_train, y_train)
y_pred_v1 = model_v1.predict(X_v1_test)

mae_v1 = mean_absolute_error(y_test, y_pred_v1)
rmse_v1 = np.sqrt(mean_squared_error(y_test, y_pred_v1))
r2_v1 = r2_score(y_test, y_pred_v1)

print(f"Version A Performance:")
print(f"  MAE:  {mae_v1:.2f} kW")
print(f"  RMSE: {rmse_v1:.2f} kW")
print(f"  R²:   {r2_v1:.4f}")

# Train Version B (with hour)
print("\n" + "="*70)
print("Training Version B: load + RUType + hour")
print("="*70)

model_v2 = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
model_v2.fit(X_v2_train, y_train)
y_pred_v2 = model_v2.predict(X_v2_test)

mae_v2 = mean_absolute_error(y_test, y_pred_v2)
rmse_v2 = np.sqrt(mean_squared_error(y_test, y_pred_v2))
r2_v2 = r2_score(y_test, y_pred_v2)

print(f"Version B Performance:")
print(f"  MAE:  {mae_v2:.2f} kW")
print(f"  RMSE: {rmse_v2:.2f} kW")
print(f"  R²:   {r2_v2:.4f}")

# Overall improvement
print("\n" + "="*70)
print("OVERALL IMPROVEMENT")
print("="*70)
print(f"MAE improvement:  {mae_v1 - mae_v2:.2f} kW ({(mae_v1-mae_v2)/mae_v1*100:.1f}%)")
print(f"RMSE improvement: {rmse_v1 - rmse_v2:.2f} kW ({(rmse_v1-rmse_v2)/rmse_v1*100:.1f}%)")
print(f"R² improvement:   {r2_v2 - r2_v1:.4f}")

# Create analysis dataframe
df_results = meta_test.copy()
df_results['actual_energy'] = y_test.values
df_results['pred_v1'] = y_pred_v1
df_results['pred_v2'] = y_pred_v2
df_results['error_v1'] = np.abs(y_test.values - y_pred_v1)
df_results['error_v2'] = np.abs(y_test.values - y_pred_v2)
df_results['improvement'] = df_results['error_v1'] - df_results['error_v2']

# Define a function for MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Focus on early morning Type4 anomalies
print("\n" + "="*70)
print("EARLY MORNING ANOMALY ANALYSIS (Hours 0-5)")
print("="*70)

for rutype in sorted(df_results['RUType'].unique()):
    mask_early = (df_results['hour'] <= 5) & (df_results['RUType'] == rutype)
    
    if mask_early.sum() > 0:
        # MAE
        mae_v1_early = df_results[mask_early]['error_v1'].mean()
        mae_v2_early = df_results[mask_early]['error_v2'].mean()
        improvement = mae_v1_early - mae_v2_early
        
        # MAPE
        mape_v1_early = mean_absolute_percentage_error(
            df_results[mask_early]['actual_energy'], df_results[mask_early]['pred_v1']
        )
        mape_v2_early = mean_absolute_percentage_error(
            df_results[mask_early]['actual_energy'], df_results[mask_early]['pred_v2']
        )
        
        print(f"\n{rutype} (n={mask_early.sum()}):")
        print(f"  Version A MAE: {mae_v1_early:.2f} kW | MAPE: {mape_v1_early:.2f}%")
        print(f"  Version B MAE: {mae_v2_early:.2f} kW | MAPE: {mape_v2_early:.2f}%")
        print(f"  Improvement:   {improvement:.2f} kW ({improvement/mae_v1_early*100:.1f}%)")

# Save detailed results
output_file = os.path.join(base_dir, 'task3_hour_comparison_results.csv')
df_results.to_csv(output_file, index=False)
print(f"\n✓ Detailed results saved: task3_hour_comparison_results.csv")

# Visualization 1: Error comparison by hour and RUType
print("\n" + "="*70)
print("GENERATING HEATMAP COMPARISONS")
print("="*70)

rutypes = sorted(df_results['RUType'].unique())
hours = sorted(df_results['hour'].unique())

# Create error matrices for both versions
error_matrix_v1 = np.zeros((len(hours), len(rutypes)))
error_matrix_v2 = np.zeros((len(hours), len(rutypes)))

for i, hour in enumerate(hours):
    for j, rutype in enumerate(rutypes):
        mask = (df_results['hour'] == hour) & (df_results['RUType'] == rutype)
        if mask.sum() > 0:
            error_matrix_v1[i, j] = df_results[mask]['error_v1'].mean()
            error_matrix_v2[i, j] = df_results[mask]['error_v2'].mean()

# Plot heatmaps side by side
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Heatmap 1: Version A (without hour)
ax1 = axes[0]
im1 = ax1.imshow(error_matrix_v1, aspect='auto', cmap='YlOrRd', origin='upper')
ax1.set_xticks(range(len(rutypes)))
ax1.set_xticklabels(rutypes, rotation=45)
ax1.set_yticks(range(0, len(hours), 2))
ax1.set_yticklabels(hours[::2])
ax1.set_xlabel('RUType', fontsize=12)
ax1.set_ylabel('Hour of Day', fontsize=12)
ax1.set_title('Version A: MAE by Hour × RUType\n(load + RUType only)', 
              fontsize=13, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Mean Absolute Error (kW)', fontsize=11)

# Heatmap 2: Version B (with hour)
ax2 = axes[1]
im2 = ax2.imshow(error_matrix_v2, aspect='auto', cmap='YlOrRd', origin='upper')
ax2.set_xticks(range(len(rutypes)))
ax2.set_xticklabels(rutypes, rotation=45)
ax2.set_yticks(range(0, len(hours), 2))
ax2.set_yticklabels(hours[::2])
ax2.set_xlabel('RUType', fontsize=12)
ax2.set_ylabel('Hour of Day', fontsize=12)
ax2.set_title('Version B: MAE by Hour × RUType\n(load + RUType + hour)', 
              fontsize=13, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Mean Absolute Error (kW)', fontsize=11)

# Heatmap 3: Improvement (V1 - V2)
ax3 = axes[2]
improvement_matrix = error_matrix_v1 - error_matrix_v2
im3 = ax3.imshow(improvement_matrix, aspect='auto', cmap='RdYlGn', origin='upper', vmin=-1, vmax=3)
ax3.set_xticks(range(len(rutypes)))
ax3.set_xticklabels(rutypes, rotation=45)
ax3.set_yticks(range(0, len(hours), 2))
ax3.set_yticklabels(hours[::2])
ax3.set_xlabel('RUType', fontsize=12)
ax3.set_ylabel('Hour of Day', fontsize=12)
ax3.set_title('Improvement (Version A - Version B)\nGreen = Better with hour feature', 
              fontsize=13, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=ax3)
cbar3.set_label('MAE Reduction (kW)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task3_hour_feature_heatmap_comparison.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved: task3_hour_feature_heatmap_comparison.png")

# Visualization 1.5: Task 2 style - Worst 10% sample proportion over time
print("\n" + "="*70)
print("GENERATING WORST 10% SAMPLES PROPORTION (BASED ON MAPE)")
print("="*70)

# Calculate relative error (percentage) for each sample
df_results['relative_error_v1'] = np.abs(
    (df_results['actual_energy'] - df_results['pred_v1']) / df_results['actual_energy']
) * 100
df_results['relative_error_v2'] = np.abs(
    (df_results['actual_energy'] - df_results['pred_v2']) / df_results['actual_energy']
) * 100

# Calculate 90th percentile threshold for worst 10% (based on relative error)
threshold_v1 = np.percentile(df_results['relative_error_v1'], 90)
threshold_v2 = np.percentile(df_results['relative_error_v2'], 90)

print(f"Version A - 90th percentile MAPE threshold: {threshold_v1:.2f}%")
print(f"Version B - 90th percentile MAPE threshold: {threshold_v2:.2f}%")

# Get unique times and RUTypes for full timeline
all_times = sorted(df_results['Time'].unique())
time_labels = [str(t) for t in all_times]

# Create proportion matrices for worst 10% samples
worst_prop_v1 = np.zeros((len(rutypes), len(all_times)))
worst_prop_v2 = np.zeros((len(rutypes), len(all_times)))

for i, rutype in enumerate(rutypes):
    for j, time in enumerate(all_times):
        mask = (df_results['RUType'] == rutype) & (df_results['Time'] == time)
        total_samples = mask.sum()
        
        if total_samples > 0:
            # Count samples exceeding threshold (worst 10% based on relative error)
            worst_count_v1 = ((df_results[mask]['relative_error_v1'] >= threshold_v1).sum())
            worst_count_v2 = ((df_results[mask]['relative_error_v2'] >= threshold_v2).sum())
            
            # Calculate proportion
            worst_prop_v1[i, j] = worst_count_v1 / total_samples
            worst_prop_v2[i, j] = worst_count_v2 / total_samples

# Plot side-by-side comparison
fig, axes = plt.subplots(2, 1, figsize=(28, 14))

# Heatmap 1: Version A worst samples proportion
ax1 = axes[0]
sns.heatmap(worst_prop_v1, ax=ax1, cmap='YlOrRd', annot=True, fmt='.1%',
            xticklabels=time_labels, yticklabels=rutypes,
            cbar_kws={'label': 'Proportion of Worst 10% Samples (by MAPE)'},
            annot_kws={'fontsize': 7}) 
ax1.set_xlabel('Time', fontsize=13)
ax1.set_ylabel('RUType', fontsize=13)
ax1.set_title('Version A: Proportion of Worst 10% Samples by Time and RUType\n(load + RUType only, threshold based on relative error)', 
              fontsize=14, fontweight='bold', pad=15)
# Rotate x-axis labels and adjust size
plt.setp(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=7)
plt.setp(ax1.get_yticklabels(), fontsize=10)

# Heatmap 2: Version B worst samples proportion
ax2 = axes[1]
sns.heatmap(worst_prop_v2, ax=ax2, cmap='YlOrRd', annot=True, fmt='.1%',
            xticklabels=time_labels, yticklabels=rutypes,
            cbar_kws={'label': 'Proportion of Worst 10% Samples (by MAPE)'},
            annot_kws={'fontsize': 7})
ax2.set_xlabel('Time', fontsize=13)
ax2.set_ylabel('RUType', fontsize=13)
ax2.set_title('Version B: Proportion of Worst 10% Samples by Time and RUType\n(load + RUType + hour, threshold based on relative error)', 
              fontsize=14, fontweight='bold', pad=15)
# Rotate x-axis labels and adjust size
plt.setp(ax2.get_xticklabels(), rotation=90, ha='center', fontsize=7)
plt.setp(ax2.get_yticklabels(), fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task3_worst_samples_proportion_heatmap.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved: task3_worst_samples_proportion_heatmap.png")

# Summary statistics (This section is less meaningful for proportions, but kept for consistency)
total_worst_v1 = (worst_prop_v1 > 0).sum()
total_worst_v2 = (worst_prop_v2 > 0).sum()
reduction = total_worst_v1 - total_worst_v2

print(f"\nTime-RUType slots with worst samples:")
print(f"  Version A: {int(total_worst_v1)} slots")
print(f"  Version B: {int(total_worst_v2)} slots")
print(f"  Reduction: {int(reduction)} slots")

# Visualization 2: Box plots for early morning hours
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

early_morning = df_results[df_results['hour'] <= 5]

ax1 = axes[0]
data_v1 = [early_morning[early_morning['RUType']==rt]['error_v1'].values 
           for rt in rutypes]
bp1 = ax1.boxplot(data_v1, labels=rutypes, patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('lightcoral')
ax1.set_ylabel('Absolute Error (kW)', fontsize=12)
ax1.set_xlabel('RUType', fontsize=12)
ax1.set_title('Version A: Early Morning Errors (Hours 0-5)\nWithout hour feature', 
              fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
data_v2 = [early_morning[early_morning['RUType']==rt]['error_v2'].values 
           for rt in rutypes]
bp2 = ax2.boxplot(data_v2, labels=rutypes, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor('lightgreen')
ax2.set_ylabel('Absolute Error (kW)', fontsize=12)
ax2.set_xlabel('RUType', fontsize=12)
ax2.set_title('Version B: Early Morning Errors (Hours 0-5)\nWith hour feature', 
              fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task3_early_morning_error_comparison.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved: task3_early_morning_error_comparison.png")

# Feature importance comparison
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Version B)")
print("="*70)

feature_names = list(X_v2_encoded.columns)
importance = model_v2.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10).to_string(index=False))

# Save feature importance
feature_importance.to_csv(os.path.join(base_dir, 'task3_feature_importance.csv'), index=False)
print("\n✓ Saved: task3_feature_importance.csv")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
top_10 = feature_importance.head(10)
ax.barh(range(len(top_10)), top_10['importance'], color='steelblue', alpha=0.8)
ax.set_yticks(range(len(top_10)))
ax.set_yticklabels(top_10['feature'])
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 10 Feature Importance (Version B with hour)', 
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task3_feature_importance_plot.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved: task3_feature_importance_plot.png")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nKey findings:")
print(f"1. Overall R² improved from {r2_v1:.4f} to {r2_v2:.4f}")
print(f"2. Overall MAE reduced from {mae_v1:.2f} to {mae_v2:.2f} kW")
print(f"3. Check heatmaps to see which RUTypes benefited most from hour feature")
print(f"4. Early morning anomalies (hours 0-5) show specific improvements")
print(f"\nNext steps:")
print(f"- Review the heatmap to identify remaining anomalies")

# Visualization 4: Energy Distribution by RUType
print("\n" + "="*70)
print("GENERATING ENERGY DISTRIBUTION BOXPLOT")
print("="*70)

fig, ax = plt.subplots(figsize=(24, 16))
bp = ax.boxplot([df_results[df_results['RUType']==rt]['actual_energy'].values 
                  for rt in sorted(df_results['RUType'].unique())],
                 labels=sorted(df_results['RUType'].unique()),
                 patch_artist=True,
                 widths=0.6,
                 # Make lines thicker
                 boxprops=dict(linewidth=3, edgecolor='black'),
                 whiskerprops=dict(linewidth=3, color='black'),
                 capprops=dict(linewidth=3, color='black'),
                 medianprops=dict(linewidth=4, color='red'),
                 # Make outlier points larger
                 flierprops=dict(marker='o', markersize=10, markerfacecolor='black', 
                                markeredgecolor='black', alpha=0.5, linewidth=2))

# Color the boxes
colors = ['steelblue'] * len(sorted(df_results['RUType'].unique()))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# ax.set_title('Energy Consumption Distribution by RUType', fontsize=14, fontweight='bold')
ax.set_xlabel('RUType', fontsize=40, fontweight='bold')
ax.set_ylabel('Energy', fontsize=40, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=36, width=2, length=8)
ax.grid(axis='y', alpha=0.3, linewidth=1.5)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'task3_energy_distribution_by_rutype.png'), 
            dpi=300, bbox_inches='tight')
print("✓ Saved: task3_energy_distribution_by_rutype.png")
print("Note: This plot helps verify if high MAE for a RUType is due to high energy consumption.")
print(f"- Consider adding more features (equipment config, interactions)")
print(f"- Investigate if measurement issues remain in certain time-RUType combinations")
