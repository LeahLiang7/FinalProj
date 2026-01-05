"""
Option 3: Prediction Error Analysis
Identify worst 10% prediction samples, analyze error characteristics and distribution patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import r2_score, mean_absolute_error

# Set matplotlib to use English
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

# Load data
print("Loading data...")
energy_data = pd.read_csv('filtered_energy_data.csv')

# Load trained models
print("Loading models...")
with open('trained_models.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    models = saved_data['models']

# Prepare features
X = energy_data[['load', 'RUType']].copy()
X = pd.get_dummies(X, columns=['RUType'], prefix='RUType', drop_first=True)
y = energy_data['Energy'].values

# Use Random Forest and XGBoost models for prediction
print("\nUsing Random Forest and XGBoost models for prediction...")
rf_model = models['Random Forest']
xgb_model = models['XGBoost']

y_pred_rf = rf_model.predict(X)
y_pred_xgb = xgb_model.predict(X)

# Calculate errors
energy_data['pred_rf'] = y_pred_rf
energy_data['pred_xgb'] = y_pred_xgb
energy_data['error_rf'] = y_pred_rf - y
energy_data['error_xgb'] = y_pred_xgb - y
energy_data['abs_error_rf'] = np.abs(energy_data['error_rf'])
energy_data['abs_error_xgb'] = np.abs(energy_data['error_xgb'])
energy_data['pct_error_rf'] = (energy_data['abs_error_rf'] / y) * 100
energy_data['pct_error_xgb'] = (energy_data['abs_error_xgb'] / y) * 100

print(f"\nOverall Performance (Random Forest):")
print(f"  R2 = {r2_score(y, y_pred_rf):.4f}")
print(f"  MAE = {mean_absolute_error(y, y_pred_rf):.2f} kW")
print(f"  Average Percentage Error = {energy_data['pct_error_rf'].mean():.2f}%")

print(f"\nOverall Performance (XGBoost):")
print(f"  R2 = {r2_score(y, y_pred_xgb):.4f}")
print(f"  MAE = {mean_absolute_error(y, y_pred_xgb):.2f} kW")
print(f"  Average Percentage Error = {energy_data['pct_error_xgb'].mean():.2f}%")

# ==================== 1. Identify Worst 10% Predictions ====================
print("\n" + "="*60)
print("1. Identify Worst 10% Prediction Samples (Based on Random Forest)")
print("="*60)

# Sort by absolute error, take worst 10%
threshold_90 = energy_data['abs_error_rf'].quantile(0.9)
worst_10pct = energy_data[energy_data['abs_error_rf'] >= threshold_90].copy()

print(f"\nWorst 10% sample count: {len(worst_10pct)}")
print(f"Error threshold: >= {threshold_90:.2f} kW")
print(f"\nStatistical characteristics of worst 10% samples:")
print(worst_10pct[['Energy', 'pred_rf', 'error_rf', 'abs_error_rf', 'load', 'RUType']].describe())

# filtered_energy_data.csv already contains all equipment parameters

# Save worst samples
worst_10pct.to_csv('worst_10pct_predictions.csv', index=False)
print("\nSaved: worst_10pct_predictions.csv")

# ==================== 2. RUType Distribution Analysis ====================
print("\n" + "="*60)
print("2. RUType Distribution of Error Samples")
print("="*60)

rutype_error_dist = worst_10pct['RUType'].value_counts()
rutype_total_dist = energy_data['RUType'].value_counts()

rutype_comparison = pd.DataFrame({
    'Error_Samples': rutype_error_dist,
    'Total_Samples': rutype_total_dist,
    'Error_Rate_%': (rutype_error_dist / rutype_total_dist * 100).round(2)
}).sort_values('Error_Rate_%', ascending=False)

print(rutype_comparison)
print("\nSaved: rutype_error_distribution.csv")
rutype_comparison.to_csv('rutype_error_distribution.csv')

# ==================== 3. Load Range Analysis ====================
print("\n" + "="*60)
print("3. Load Distribution of Error Samples")
print("="*60)

# Define load intervals
load_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
load_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

energy_data['load_bin'] = pd.cut(energy_data['load'], bins=load_bins, labels=load_labels)
worst_10pct['load_bin'] = pd.cut(worst_10pct['load'], bins=load_bins, labels=load_labels)

load_error_dist = worst_10pct['load_bin'].value_counts()
load_total_dist = energy_data['load_bin'].value_counts()

load_comparison = pd.DataFrame({
    'Error_Samples': load_error_dist,
    'Total_Samples': load_total_dist,
    'Error_Rate_%': (load_error_dist / load_total_dist * 100).round(2)
}).sort_index()

print(load_comparison)
print("\nSaved: load_range_error_distribution.csv")
load_comparison.to_csv('load_range_error_distribution.csv')

# ==================== 4. Time Distribution Analysis ====================
print("\n" + "="*60)
print("4. Time Distribution of Error Samples")
print("="*60)

hour_error_dist = worst_10pct['Time'].value_counts().sort_index()
hour_total_dist = energy_data['Time'].value_counts().sort_index()

hour_comparison = pd.DataFrame({
    'Error_Samples': hour_error_dist,
    'Total_Samples': hour_total_dist,
    'Error_Rate_%': (hour_error_dist / hour_total_dist * 100).round(2)
})

print(f"\nTime periods with highest error rates (Top 5):")
print(hour_comparison.nlargest(5, 'Error_Rate_%'))

print("\nSaved: hour_error_distribution.csv")
hour_comparison.to_csv('hour_error_distribution.csv')

# ==================== 5. Error Direction Analysis ====================
print("\n" + "="*60)
print("5. Error Direction Analysis (Overestimate vs Underestimate)")
print("="*60)

overestimate = worst_10pct[worst_10pct['error_rf'] > 0]
underestimate = worst_10pct[worst_10pct['error_rf'] < 0]

print(f"\nOverestimate samples: {len(overestimate)} ({len(overestimate)/len(worst_10pct)*100:.1f}%)")
print(f"  Average overestimation: {overestimate['error_rf'].mean():.2f} kW")
print(f"  Average true energy: {overestimate['Energy'].mean():.2f} kW")
print(f"  Average load: {overestimate['load'].mean():.3f}")
print(f"  Main RUType: {overestimate['RUType'].mode().values}")

print(f"\nUnderestimate samples: {len(underestimate)} ({len(underestimate)/len(worst_10pct)*100:.1f}%)")
print(f"  Average underestimation: {underestimate['error_rf'].mean():.2f} kW")
print(f"  Average true energy: {underestimate['Energy'].mean():.2f} kW")
print(f"  Average load: {underestimate['load'].mean():.3f}")
print(f"  Main RUType: {underestimate['RUType'].mode().values}")

# ==================== 6. Visualization ====================
print("\n" + "="*60)
print("6. Generating Visualization Charts")
print("="*60)

# Figure 1: Error Distribution Overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Error distribution histogram
ax1 = axes[0, 0]
ax1.hist(energy_data['error_rf'], bins=100, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax1.axvline(x=threshold_90, color='orange', linestyle='--', linewidth=2, label=f'90th percentile: {threshold_90:.2f} kW')
ax1.axvline(x=-threshold_90, color='orange', linestyle='--', linewidth=2)
ax1.set_xlabel('Prediction Error (kW)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Prediction Error Distribution (Random Forest)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cumulative distribution of absolute errors
ax2 = axes[0, 1]
sorted_errors = np.sort(energy_data['abs_error_rf'])
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax2.plot(sorted_errors, cumulative, linewidth=2, color='green')
ax2.axvline(x=threshold_90, color='red', linestyle='--', linewidth=2, label=f'Top 10% threshold')
ax2.axhline(y=90, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Absolute Error (kW)', fontsize=12)
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax2.set_title('Cumulative Distribution of Absolute Errors', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# RUType error rate
ax3 = axes[1, 0]
rutype_comparison['Error_Rate_%'].sort_values(ascending=True).plot(kind='barh', ax=ax3, color='coral')
ax3.set_xlabel('Error Sample Rate (%)', fontsize=12)
ax3.set_ylabel('RUType', fontsize=12)
ax3.set_title('Error Sample Rate by RUType', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Load range error rate
ax4 = axes[1, 1]
load_comparison['Error_Rate_%'].plot(kind='bar', ax=ax4, color='skyblue', edgecolor='black')
ax4.set_xlabel('Load Range', fontsize=12)
ax4.set_ylabel('Error Sample Rate (%)', fontsize=12)
ax4.set_title('Error Sample Rate by Load Range', fontsize=14, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('error_analysis_overview.png', dpi=300, bbox_inches='tight')
print("Saved: error_analysis_overview.png")
plt.close()

# Figure 2: True vs Predicted scatter plot (highlighting worst samples)
fig, ax = plt.subplots(figsize=(10, 10))

# Plot all samples
ax.scatter(y, y_pred_rf, alpha=0.3, s=10, color='gray', label='All samples')

# Highlight worst 10% samples
worst_y = worst_10pct['Energy'].values
worst_pred = worst_10pct['pred_rf'].values
ax.scatter(worst_y, worst_pred, alpha=0.7, s=50, color='red', 
          edgecolors='darkred', linewidths=1.5, label='Worst 10% samples')

# Plot perfect prediction line
min_val = min(y.min(), y_pred_rf.min())
max_val = max(y.max(), y_pred_rf.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')

# Plot ±error threshold lines
ax.plot([min_val, max_val], [min_val + threshold_90, max_val + threshold_90], 
       'orange', linestyle='--', linewidth=1, alpha=0.7, label=f'±{threshold_90:.1f} kW')
ax.plot([min_val, max_val], [min_val - threshold_90, max_val - threshold_90], 
       'orange', linestyle='--', linewidth=1, alpha=0.7)

ax.set_xlabel('True Energy (kW)', fontsize=12)
ax.set_ylabel('Predicted Energy (kW)', fontsize=12)
ax.set_title('True vs Predicted Values (Worst 10% Highlighted)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('prediction_scatter_with_worst_samples.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_scatter_with_worst_samples.png")
plt.close()

# Figure 3: Error boxplot by RUType
fig, ax = plt.subplots(figsize=(12, 6))

rutype_order = sorted(energy_data['RUType'].unique())
data_by_rutype = [energy_data[energy_data['RUType'] == rt]['abs_error_rf'].values 
                  for rt in rutype_order]

bp = ax.boxplot(data_by_rutype, tick_labels=rutype_order, patch_artist=True,
                showmeans=True, meanline=True)

# Set colors
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

ax.axhline(y=threshold_90, color='red', linestyle='--', linewidth=2, 
          label=f'Top 10% threshold: {threshold_90:.2f} kW')

ax.set_xlabel('RUType', fontsize=12)
ax.set_ylabel('Absolute Error (kW)', fontsize=12)
ax.set_title('Prediction Error Distribution by RUType', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rutype_error_boxplot.png', dpi=300, bbox_inches='tight')
print("Saved: rutype_error_boxplot.png")
plt.close()

# Figure 4: Load vs Error scatter plot (colored by RUType)
fig, ax = plt.subplots(figsize=(12, 7))

rutypes = sorted(energy_data['RUType'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(rutypes)))

for i, rutype in enumerate(rutypes):
    rutype_data = energy_data[energy_data['RUType'] == rutype]
    ax.scatter(rutype_data['load'], rutype_data['abs_error_rf'], 
              alpha=0.4, s=20, color=colors[i], label=rutype)

ax.axhline(y=threshold_90, color='red', linestyle='--', linewidth=2, 
          label=f'Top 10% threshold: {threshold_90:.2f} kW')

ax.set_xlabel('Load (normalized)', fontsize=12)
ax.set_ylabel('Absolute Error (kW)', fontsize=12)
ax.set_title('Load vs Absolute Error (Colored by RUType)', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('load_vs_error_by_rutype.png', dpi=300, bbox_inches='tight')
print("Saved: load_vs_error_by_rutype.png")
plt.close()

# Figure 5: Time-RUType distribution heatmap
fig, ax = plt.subplots(figsize=(14, 6))

hour_rutype_errors = worst_10pct.groupby(['Time', 'RUType']).size().unstack(fill_value=0)
sns.heatmap(hour_rutype_errors.T, cmap='YlOrRd', annot=True, fmt='d', 
           cbar_kws={'label': 'Error Sample Count'}, ax=ax)

ax.set_xlabel('Time (Hour)', fontsize=12)
ax.set_ylabel('RUType', fontsize=12)
ax.set_title('Time-RUType Distribution Heatmap of Worst 10% Samples', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('error_time_rutype_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: error_time_rutype_heatmap.png")
plt.close()

# ==================== 7. Top 20 Worst Samples Detailed Analysis ====================
print("\n" + "="*60)
print("7. Top 20 Worst Samples Detailed Information")
print("="*60)

top20_worst = worst_10pct.nlargest(20, 'abs_error_rf')[
    ['BS', 'Time', 'RUType', 'load', 'Energy', 'pred_rf', 'error_rf', 
     'Frequency', 'Bandwidth', 'Antennas', 'TXpower']
]

print(top20_worst.to_string(index=False))
print("\nSaved: top20_worst_predictions.csv")
top20_worst.to_csv('top20_worst_predictions.csv', index=False)

# ==================== 8. Key Findings Summary ====================
print("\n" + "="*60)
print("8. Key Findings Summary")
print("="*60)

summary = f"""
【Prediction Error Analysis Summary】

1. Overall Error Statistics:
   - Random Forest MAE: {mean_absolute_error(y, y_pred_rf):.2f} kW
   - Worst 10% sample threshold: >= {threshold_90:.2f} kW
   - Worst 10% sample count: {len(worst_10pct)} / {len(energy_data)}

2. RUType Distribution:
   - Highest error rate RUType: {rutype_comparison['Error_Rate_%'].idxmax()} ({rutype_comparison['Error_Rate_%'].max():.2f}%)
   - Lowest error rate RUType: {rutype_comparison['Error_Rate_%'].idxmin()} ({rutype_comparison['Error_Rate_%'].min():.2f}%)
   - Top 3 high error rates:
{rutype_comparison.nlargest(3, 'Error_Rate_%')[['Error_Rate_%']].to_string()}

3. Load Range Distribution:
   - Highest error rate load range: {load_comparison['Error_Rate_%'].idxmax()} ({load_comparison['Error_Rate_%'].max():.2f}%)
   - Low load (0-0.2) error rate: {load_comparison.loc['0-0.2', 'Error_Rate_%']:.2f}%
   - High load (0.8-1.0) error rate: {load_comparison.loc['0.8-1.0', 'Error_Rate_%']:.2f}%

4. Error Direction:
   - Overestimate samples: {len(overestimate)} ({len(overestimate)/len(worst_10pct)*100:.1f}%)
     Average overestimation: {overestimate['error_rf'].mean():.2f} kW
   - Underestimate samples: {len(underestimate)} ({len(underestimate)/len(worst_10pct)*100:.1f}%)
     Average underestimation: {abs(underestimate['error_rf'].mean()):.2f} kW

5. Time Distribution:
   - Top 3 time periods with highest error rates:
{hour_comparison.nlargest(3, 'Error_Rate_%')[['Error_Rate_%']].to_string()}


6.Genrated files:
- worst_10pct_predictions.csv: Worst 10% sample details
- top20_worst_predictions.csv: Top 20 worst samples
- rutype_error_distribution.csv: RUType error distribution
- load_range_error_distribution.csv: Load range error distribution
- hour_error_distribution.csv: Time error distribution
- error_analysis_overview.png: Error analysis overview chart
- prediction_scatter_with_worst_samples.png: Prediction scatter plot (worst samples highlighted)
- rutype_error_boxplot.png: RUType error boxplot
- load_vs_error_by_rutype.png: Load vs error scatter plot
- error_time_rutype_heatmap.png: Time-RUType error heatmap
"""

print(summary)

with open('error_analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("\nSummary saved: error_analysis_summary.txt")

print("\n" + "="*60)
print("Prediction Error Analysis Complete!")
print("="*60)
