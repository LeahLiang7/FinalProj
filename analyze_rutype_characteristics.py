"""
Option 1: RUType Characteristic and Performance Root Cause Analysis
Analyze equipment parameter characteristics of different RUTypes to explore root causes of performance differences
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

# filtered_energy_data.csv already contains all equipment parameters
energy_with_specs = energy_data.copy()

print(f"\nData shape: {energy_with_specs.shape}")
print(f"Missing values:\n{energy_with_specs.isnull().sum()}")

# ==================== 1. Equipment Parameter Statistics ====================
print("\n" + "="*60)
print("1. Equipment Parameter Statistics by RUType")
print("="*60)

rutype_specs = energy_with_specs.groupby('RUType').agg({
    'Frequency': lambda x: f"{x.min()}-{x.max()}",
    'Bandwidth': lambda x: f"{x.min()}-{x.max()}",
    'Antennas': lambda x: f"{x.min()}-{x.max()}",
    'TXpower': lambda x: f"{x.min()}-{x.max()}",
    'BS': 'nunique',
    'load': 'count'
}).rename(columns={'BS': 'Num_BS', 'load': 'Num_Samples'})

print(rutype_specs)
print("\nSaving equipment parameter statistics...")
rutype_specs.to_csv('rutype_equipment_specs.csv')

# ==================== 2. Baseline Energy Analysis ====================
print("\n" + "="*60)
print("2. Baseline Energy Analysis (Energy at load~0)")
print("="*60)

# Select low load samples (load < 0.05)
low_load_samples = energy_with_specs[energy_with_specs['load'] < 0.05]

baseline_stats = low_load_samples.groupby('RUType')['Energy'].agg([
    ('Baseline_Mean_kW', 'mean'),
    ('Baseline_Median_kW', 'median'),
    ('Baseline_Std_kW', 'std'),
    ('Num_Samples', 'count')
]).round(2)

print(baseline_stats)
print("\nSaving baseline energy statistics...")
baseline_stats.to_csv('rutype_baseline_energy.csv')

# ==================== 3. Load-Energy Correlation Analysis ====================
print("\n" + "="*60)
print("3. Load-Energy Correlation by RUType")
print("="*60)

correlation_results = []
for rutype in sorted(energy_data['RUType'].unique()):
    rutype_data = energy_data[energy_data['RUType'] == rutype]
    corr = rutype_data['load'].corr(rutype_data['Energy'])
    correlation_results.append({
        'RUType': rutype,
        'Correlation': corr,
        'Num_Samples': len(rutype_data)
    })

corr_df = pd.DataFrame(correlation_results)
print(corr_df)
print("\nSaving correlation analysis results...")
corr_df.to_csv('rutype_load_energy_correlation.csv', index=False)

# ==================== 4. Model Performance vs Equipment Parameters ====================
print("\n" + "="*60)
print("4. Calculate Model Performance Metrics by RUType")
print("="*60)

# Prepare data
X = energy_data[['load', 'RUType']].copy()
X = pd.get_dummies(X, columns=['RUType'], prefix='RUType', drop_first=True)
y = energy_data['Energy'].values

# Use Random Forest model for prediction
rf_model = models['Random Forest']
y_pred = rf_model.predict(X)

# Calculate performance for each RUType
performance_results = []
for rutype in sorted(energy_data['RUType'].unique()):
    mask = energy_data['RUType'] == rutype
    y_true_rutype = y[mask]
    y_pred_rutype = y_pred[mask]
    
    r2 = r2_score(y_true_rutype, y_pred_rutype)
    mae = mean_absolute_error(y_true_rutype, y_pred_rutype)
    
    # Get average equipment parameters for this RUType
    rutype_specs = energy_with_specs[energy_with_specs['RUType'] == rutype]
    avg_freq = rutype_specs['Frequency'].mean()
    avg_bw = rutype_specs['Bandwidth'].mean()
    avg_ant = rutype_specs['Antennas'].mean()
    avg_txpwr = rutype_specs['TXpower'].mean()
    avg_baseline = low_load_samples[low_load_samples['RUType'] == rutype]['Energy'].mean()
    
    performance_results.append({
        'RUType': rutype,
        'R2': round(r2, 4),
        'MAE_kW': round(mae, 2),
        'Avg_Freq_MHz': round(avg_freq, 0) if not pd.isna(avg_freq) else None,
        'Avg_BW_MHz': round(avg_bw, 0) if not pd.isna(avg_bw) else None,
        'Avg_Antennas': round(avg_ant, 0) if not pd.isna(avg_ant) else None,
        'Avg_TXpower_W': round(avg_txpwr, 0) if not pd.isna(avg_txpwr) else None,
        'Baseline_kW': round(avg_baseline, 2) if not pd.isna(avg_baseline) else None,
        'Num_Samples': mask.sum()
    })

perf_df = pd.DataFrame(performance_results)
print(perf_df.to_string(index=False))
print("\nSaving performance vs equipment parameters table...")
perf_df.to_csv('rutype_performance_vs_specs.csv', index=False)

# ==================== 5. Visualization ====================
print("\n" + "="*60)
print("5. Generating Visualization Charts")
print("="*60)

# Figure 1: Equipment parameter heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Frequency distribution
ax1 = axes[0, 0]
freq_by_type = energy_with_specs.groupby('RUType')['Frequency'].mean().sort_values()
ax1.barh(freq_by_type.index, freq_by_type.values, color='skyblue')
ax1.set_xlabel('Average Frequency (MHz)', fontsize=12)
ax1.set_ylabel('RUType', fontsize=12)
ax1.set_title('Average Frequency by RUType', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Bandwidth distribution
ax2 = axes[0, 1]
bw_by_type = energy_with_specs.groupby('RUType')['Bandwidth'].mean().sort_values()
ax2.barh(bw_by_type.index, bw_by_type.values, color='lightgreen')
ax2.set_xlabel('Average Bandwidth (MHz)', fontsize=12)
ax2.set_ylabel('RUType', fontsize=12)
ax2.set_title('Average Bandwidth by RUType', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Antenna distribution
ax3 = axes[1, 0]
ant_by_type = energy_with_specs.groupby('RUType')['Antennas'].mean().sort_values()
ax3.barh(ant_by_type.index, ant_by_type.values, color='lightcoral')
ax3.set_xlabel('Average Number of Antennas', fontsize=12)
ax3.set_ylabel('RUType', fontsize=12)
ax3.set_title('Average Antennas by RUType', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# TX power distribution
ax4 = axes[1, 1]
txpwr_by_type = energy_with_specs.groupby('RUType')['TXpower'].mean().sort_values()
ax4.barh(txpwr_by_type.index, txpwr_by_type.values, color='plum')
ax4.set_xlabel('Average TX Power (W)', fontsize=12)
ax4.set_ylabel('RUType', fontsize=12)
ax4.set_title('Average TX Power by RUType', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('rutype_equipment_parameters.png', dpi=300, bbox_inches='tight')
print("Saved: rutype_equipment_parameters.png")
plt.close()

# Figure 2: Baseline energy vs Model performance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Baseline energy vs R2
baseline_vals = perf_df['Baseline_kW'].values
r2_vals = perf_df['R2'].values
ax1.scatter(baseline_vals, r2_vals, s=200, alpha=0.6, c=range(len(perf_df)), cmap='viridis')
for i, row in perf_df.iterrows():
    ax1.annotate(row['RUType'], (row['Baseline_kW'], row['R2']), 
                fontsize=10, ha='center', va='bottom')
ax1.set_xlabel('Baseline Energy (kW)', fontsize=12)
ax1.set_ylabel('R2 Score', fontsize=12)
ax1.set_title('Baseline Energy vs Model Performance (R2)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Load correlation vs R2
corr_vals = corr_df.set_index('RUType').loc[perf_df['RUType'], 'Correlation'].values
ax2.scatter(corr_vals, r2_vals, s=200, alpha=0.6, c=range(len(perf_df)), cmap='viridis')
for i, row in perf_df.iterrows():
    corr = corr_df[corr_df['RUType'] == row['RUType']]['Correlation'].values[0]
    ax2.annotate(row['RUType'], (corr, row['R2']), 
                fontsize=10, ha='center', va='bottom')
ax2.set_xlabel('Load-Energy Correlation', fontsize=12)
ax2.set_ylabel('R2 Score', fontsize=12)
ax2.set_title('Load Correlation vs Model Performance (R2)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rutype_baseline_vs_performance.png', dpi=300, bbox_inches='tight')
print("Saved: rutype_baseline_vs_performance.png")
plt.close()

# Figure 3: Radar chart comparison
fig = plt.figure(figsize=(12, 10))

# Select best and worst RUType
best_rutype = perf_df.loc[perf_df['R2'].idxmax(), 'RUType']
worst_rutype = perf_df.loc[perf_df['R2'].idxmin(), 'RUType']

print(f"\nPerformance Comparison: {best_rutype} (Best) vs {worst_rutype} (Worst)")

# Normalize indicators for radar chart
categories = ['R2', 'MAE\n(Normalized)', 'Baseline Energy\n(Normalized)', 'Load Correlation']

def normalize_for_radar(values, inverse=False):
    """Normalize to 0-1, inverse=True when lower is better"""
    vmin, vmax = np.min(values), np.max(values)
    if vmax == vmin:
        return np.ones_like(values)
    normalized = (values - vmin) / (vmax - vmin)
    if inverse:
        normalized = 1 - normalized
    return normalized

# Prepare data
r2_norm = perf_df['R2'].values
mae_norm = normalize_for_radar(perf_df['MAE_kW'].values, inverse=True)
baseline_norm = normalize_for_radar(perf_df['Baseline_kW'].values, inverse=True)
corr_norm = corr_df.set_index('RUType').loc[perf_df['RUType'], 'Correlation'].values

best_idx = perf_df[perf_df['RUType'] == best_rutype].index[0]
worst_idx = perf_df[perf_df['RUType'] == worst_rutype].index[0]

best_values = [r2_norm[best_idx], mae_norm[best_idx], baseline_norm[best_idx], corr_norm[best_idx]]
worst_values = [r2_norm[worst_idx], mae_norm[worst_idx], baseline_norm[worst_idx], corr_norm[worst_idx]]

# Create radar chart
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
best_values += best_values[:1]
worst_values += worst_values[:1]
angles += angles[:1]

ax = fig.add_subplot(111, projection='polar')
ax.plot(angles, best_values, 'o-', linewidth=2, label=f'{best_rutype} (Best)', color='green')
ax.fill(angles, best_values, alpha=0.25, color='green')
ax.plot(angles, worst_values, 'o-', linewidth=2, label=f'{worst_rutype} (Worst)', color='red')
ax.fill(angles, worst_values, alpha=0.25, color='red')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.set_title('Best vs Worst RUType Performance Comparison\n(Normalized Indicators)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True)

plt.tight_layout()
plt.savefig('rutype_radar_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: rutype_radar_comparison.png")
plt.close()

# ==================== 6. Key Findings Summary ====================
print("\n" + "="*60)
print("6. Key Findings Summary")
print("="*60)

summary = f"""
【RUType Characteristic and Performance Root Cause Analysis Summary】

1. Performance Ranking:
   - Best: {perf_df.loc[perf_df['R2'].idxmax(), 'RUType']} (R2={perf_df['R2'].max():.4f}, MAE={perf_df.loc[perf_df['R2'].idxmax(), 'MAE_kW']:.2f} kW)
   - Worst: {perf_df.loc[perf_df['R2'].idxmin(), 'RUType']} (R2={perf_df['R2'].min():.4f}, MAE={perf_df.loc[perf_df['R2'].idxmin(), 'MAE_kW']:.2f} kW)

2. Baseline Energy Impact:
   - High baseline RUType ({worst_rutype}: {perf_df.loc[worst_idx, 'Baseline_kW']:.2f} kW) 
     shows poor prediction performance (R2={perf_df.loc[worst_idx, 'R2']:.4f})
   - Low baseline RUType ({best_rutype}: {perf_df.loc[best_idx, 'Baseline_kW']:.2f} kW) 
     shows better prediction performance (R2={perf_df.loc[best_idx, 'R2']:.4f})

3. Load Correlation:
   - Strongest correlation: {corr_df.loc[corr_df['Correlation'].idxmax(), 'RUType']} (correlation={corr_df['Correlation'].max():.4f})
   - Weakest correlation: {corr_df.loc[corr_df['Correlation'].idxmin(), 'RUType']} (correlation={corr_df['Correlation'].min():.4f})
   - RUTypes with strong correlation generally have better prediction performance

4. Equipment Parameter Differences:
   - Frequency range: {energy_with_specs['Frequency'].min():.0f}-{energy_with_specs['Frequency'].max():.0f} MHz
   - Bandwidth range: {energy_with_specs['Bandwidth'].min():.0f}-{energy_with_specs['Bandwidth'].max():.0f} MHz
   - Antenna count range: {energy_with_specs['Antennas'].min():.0f}-{energy_with_specs['Antennas'].max():.0f}
   - TX power range: {energy_with_specs['TXpower'].min():.0f}-{energy_with_specs['TXpower'].max():.0f} W

5. Improvement Recommendations:
   - For high baseline RUTypes, consider adding equipment parameter features
   - For low correlation RUTypes, add time features or sleep mode indicators
   - Consider training separate models for different RUTypes

Generated Files:
- rutype_equipment_specs.csv: Equipment parameter statistics
- rutype_baseline_energy.csv: Baseline energy statistics
- rutype_load_energy_correlation.csv: Load-energy correlation
- rutype_performance_vs_specs.csv: Performance vs equipment parameters
- rutype_equipment_parameters.png: Equipment parameter distribution chart
- rutype_baseline_vs_performance.png: Baseline energy vs performance chart
- rutype_radar_comparison.png: Best vs worst RUType radar comparison
"""

print(summary)

with open('rutype_characteristics_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)
print("\nSummary saved: rutype_characteristics_summary.txt")

print("\n" + "="*60)
print("RUType Characteristic Analysis Complete!")
print("="*60)
