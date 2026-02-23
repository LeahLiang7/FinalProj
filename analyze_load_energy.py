"""
Load vs Energy Analysis by RUType
Analyzes the relationship between load and energy consumption for each RUType.
Investigates whether flat-line base stations have consistently low load.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'filtered_energy_data.csv')
OUTPUT_PLOT = os.path.join(BASE_DIR, 'load_energy_analysis.png')

# --- 1. Load Data ---
print(f"Loading cleaned data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
print("Data loaded successfully.\n")

# --- 2. Identify Flat-line Base Stations ---
print("Identifying flat-line base stations...")
flat_line_threshold = 2.0
bs_stats = []

for bs in df['BS'].unique():
    bs_data = df[df['BS'] == bs]
    energy_std = bs_data['Energy'].std()
    load_mean = bs_data['load'].mean()
    load_std = bs_data['load'].std()
    rutype = bs_data['RUType'].iloc[0]
    
    is_flat = energy_std < flat_line_threshold
    
    bs_stats.append({
        'BS': bs,
        'RUType': rutype,
        'Energy_Std': energy_std,
        'Load_Mean': load_mean,
        'Load_Std': load_std,
        'Is_FlatLine': is_flat
    })

bs_stats_df = pd.DataFrame(bs_stats)

# --- 3. Statistical Summary ---
print("="*70)
print("Load Analysis for Flat-line vs Normal Base Stations")
print("="*70)

for rutype in sorted(df['RUType'].unique()):
    rutype_stats = bs_stats_df[bs_stats_df['RUType'] == rutype]
    flat_stats = rutype_stats[rutype_stats['Is_FlatLine'] == True]
    normal_stats = rutype_stats[rutype_stats['Is_FlatLine'] == False]
    
    print(f"\n{rutype}:")
    print(f"  Flat-line BS: {len(flat_stats)} / {len(rutype_stats)}")
    
    if len(flat_stats) > 0:
        print(f"    Average Load: {flat_stats['Load_Mean'].mean():.3f}")
        print(f"    Load Std Dev: {flat_stats['Load_Std'].mean():.3f}")
    
    if len(normal_stats) > 0:
        print(f"  Normal BS: {len(normal_stats)} / {len(rutype_stats)}")
        print(f"    Average Load: {normal_stats['Load_Mean'].mean():.3f}")
        print(f"    Load Std Dev: {normal_stats['Load_Std'].mean():.3f}")

# --- 4. Visualization: Overall Load vs Energy (All RUTypes Combined) ---
print("\n" + "="*70)
print("Generating visualizations...")
print("="*70 + "\n")

print("Creating overall load vs energy plot (all RUTypes combined)...")

# Get all flat-line BS
flat_bs_list_all = bs_stats_df[bs_stats_df['Is_FlatLine'] == True]['BS'].tolist()

# Separate flat-line and normal data for all RUTypes
flat_data_all = df[df['BS'].isin(flat_bs_list_all)]
normal_data_all = df[~df['BS'].isin(flat_bs_list_all)]

# Create overall plot
fig_overall, ax_overall = plt.subplots(figsize=(12, 8))

if len(normal_data_all) > 0:
    ax_overall.scatter(normal_data_all['load'], normal_data_all['Energy'], 
              alpha=0.2, s=8, c='steelblue', label=f'Normal BS (n={len(normal_data_all)})')

if len(flat_data_all) > 0:
    ax_overall.scatter(flat_data_all['load'], flat_data_all['Energy'], 
              alpha=0.4, s=12, c='red', label=f'Flat-line BS (n={len(flat_data_all)})', marker='x')

ax_overall.set_title('Load vs Energy Consumption - All RUTypes Combined\n(Red = Flat-line BS, Blue = Normal BS)', 
            fontsize=14, fontweight='bold')
ax_overall.set_xlabel('Load', fontsize=12)
ax_overall.set_ylabel('Energy Consumption (kW)', fontsize=12)
ax_overall.legend(fontsize=11)
ax_overall.grid(True, alpha=0.3)

# Add statistics box
flat_load_mean = flat_data_all['load'].mean() if len(flat_data_all) > 0 else 0
normal_load_mean = normal_data_all['load'].mean() if len(normal_data_all) > 0 else 0
stats_text = f'Flat-line BS avg load: {flat_load_mean:.3f}\nNormal BS avg load: {normal_load_mean:.3f}'
ax_overall.text(0.98, 0.02, stats_text, transform=ax_overall.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
overall_plot_path = os.path.join(BASE_DIR, 'load_energy_overall.png')
plt.savefig(overall_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Overall load vs energy plot saved to: {overall_plot_path}\n")
plt.close()

# --- 5. Visualization: Load vs Energy by RUType ---
print("Creating load vs energy plots by RUType...")

rutypes = sorted(df['RUType'].unique())
n_types = len(rutypes)
n_cols = 3
n_rows = (n_types + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
axes = axes.flatten()

for idx, rutype in enumerate(rutypes):
    ax = axes[idx]
    rutype_data = df[df['RUType'] == rutype]
    
    # Get flat-line BS list for this RUType
    flat_bs_list = bs_stats_df[(bs_stats_df['RUType'] == rutype) & 
                               (bs_stats_df['Is_FlatLine'] == True)]['BS'].tolist()
    
    # Separate flat-line and normal data
    flat_data = rutype_data[rutype_data['BS'].isin(flat_bs_list)]
    normal_data = rutype_data[~rutype_data['BS'].isin(flat_bs_list)]
    
    # Plot
    if len(normal_data) > 0:
        ax.scatter(normal_data['load'], normal_data['Energy'], 
                  alpha=0.3, s=10, c='steelblue', label='Normal BS')
    
    if len(flat_data) > 0:
        ax.scatter(flat_data['load'], flat_data['Energy'], 
                  alpha=0.3, s=10, c='red', label='Flat-line BS')
    
    # Customize
    ax.set_title(f'{rutype}\n({len(flat_bs_list)} flat-line / {rutype_data["BS"].nunique()} total BS)', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Load', fontsize=11)
    ax.set_ylabel('Energy Consumption', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_types, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Load vs Energy Consumption Analysis by RUType\n(Red = Flat-line BS, Blue = Normal BS)', 
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"✓ Load vs energy by RUType plot saved to: {OUTPUT_PLOT}")
plt.close()

# --- 6. Additional Analysis: Load Distribution ---
print("\n" + "="*70)
print("Load Distribution Summary")
print("="*70)

for rutype in sorted(df['RUType'].unique()):
    rutype_data = df[df['RUType'] == rutype]
    print(f"\n{rutype}:")
    print(f"  Load range: {rutype_data['load'].min():.3f} - {rutype_data['load'].max():.3f}")
    print(f"  Load mean: {rutype_data['load'].mean():.3f}")
    print(f"  Load std: {rutype_data['load'].std():.3f}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
