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
fig_overall, ax_overall = plt.subplots(figsize=(16, 12))

if len(normal_data_all) > 0:
    ax_overall.scatter(normal_data_all['load'], normal_data_all['Energy'], 
              alpha=0.2, s=16, c='steelblue')

if len(flat_data_all) > 0:
    ax_overall.scatter(flat_data_all['load'], flat_data_all['Energy'], 
              alpha=0.4, s=24, c='red', marker='x')

# Add legend in top-left corner
legend_text = 'Red = Flat-line BS\nBlue = Normal BS'
ax_overall.text(0.02, 0.98, legend_text, transform=ax_overall.transAxes,
                fontsize=28, verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5))

ax_overall.set_xlabel('Load', fontsize=32)
ax_overall.set_ylabel('Energy Consumption', fontsize=32)
ax_overall.tick_params(axis='both', which='major', labelsize=28)
ax_overall.grid(True, alpha=0.3)

plt.tight_layout()
overall_plot_path = os.path.join(BASE_DIR, 'load_energy_overall.png')
plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Overall load vs energy plot saved to: {overall_plot_path}\n")
plt.close()

# --- 5. Visualization: Load vs Energy by RUType ---
print("Creating load vs energy plots by RUType...")

rutypes = sorted(df['RUType'].unique())
n_types = len(rutypes)
n_cols = 2
n_rows = (n_types + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 12*n_rows))
if n_types == 1:
    axes = [axes]
else:
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
                  alpha=0.3, s=20, c='steelblue')
    
    if len(flat_data) > 0:
        ax.scatter(flat_data['load'], flat_data['Energy'], 
                  alpha=0.3, s=20, c='red', marker='x')
    
    # Customize
    ax.set_title(f'{rutype}\n({len(flat_bs_list)} flat-line / {rutype_data["BS"].nunique()} total BS)', 
                fontsize=24, fontweight='bold')
    ax.set_xlabel('Load', fontsize=22)
    ax.set_ylabel('Energy Consumption', fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_types, len(axes)):
    axes[idx].axis('off')

plt.suptitle('Load vs Energy Consumption Analysis by RUType\n(Red = Flat-line BS, Blue = Normal BS)', 
            fontsize=32, fontweight='bold', y=0.998)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
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

# --- 7. Time Series Analysis: Average Load over Time (All RUTypes Combined) ---
print("\nGenerating time series plots...")

# Calculate average load and energy for each hour across all RUTypes
time_series = df.groupby('Hours').agg({
    'load': 'mean',
    'Energy': 'mean'
}).reset_index()

# Plot 1: Average Load over Time
fig1, ax1 = plt.subplots(1, 1, figsize=(24, 16))

ax1.plot(time_series['Hours'], time_series['load'], 
         linewidth=3, color='steelblue', marker='o', markersize=8)

ax1.set_xlabel('Hour (72h = 3 days)', fontsize=36, fontweight='bold')
ax1.set_ylabel('Average Load', fontsize=36, fontweight='bold')

# Set x-ticks every 6 hours with 24-hour format labels
x_ticks = range(1, 73, 6)
x_labels = [f"Day{((h-1)//24)+1}\n{((h-1)%24):02d}:00" for h in x_ticks]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=32)

ax1.tick_params(axis='y', labelsize=32)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(1, 72)

plt.tight_layout()
load_time_plot = os.path.join(BASE_DIR, 'average_load_over_time.png')
plt.savefig(load_time_plot, dpi=150, bbox_inches='tight')
print(f"✓ Average load over time plot saved to: {load_time_plot}")
plt.close()

# Plot 2: Average Energy over Time
fig2, ax2 = plt.subplots(1, 1, figsize=(24, 16))

ax2.plot(time_series['Hours'], time_series['Energy'], 
         linewidth=3, color='darkgreen', marker='o', markersize=8)

ax2.set_xlabel('Hour (72h = 3 days)', fontsize=36, fontweight='bold')
ax2.set_ylabel('Average Energy Consumption', fontsize=36, fontweight='bold')

# Set x-ticks every 6 hours with 24-hour format labels
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=32)

ax2.tick_params(axis='y', labelsize=32)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(1, 72)

plt.tight_layout()
energy_time_plot = os.path.join(BASE_DIR, 'average_energy_over_time.png')
plt.savefig(energy_time_plot, dpi=150, bbox_inches='tight')
print(f"✓ Average energy over time plot saved to: {energy_time_plot}")
plt.close()

# Plot 3: Load and Energy Correlation - Dual Y-axis Time Series
fig3, ax3 = plt.subplots(1, 1, figsize=(24, 16))

# Plot load on primary y-axis
color_load = 'steelblue'
ax3.plot(time_series['Hours'], time_series['load'], 
         linewidth=4, color=color_load, marker='o', markersize=10, label='Average Load')
ax3.set_xlabel('Hour', fontsize=40, fontweight='bold')
ax3.set_ylabel('Average Load', fontsize=40, fontweight='bold', color=color_load)
ax3.tick_params(axis='y', labelsize=36, labelcolor=color_load, width=2, length=8)
ax3.tick_params(axis='x', labelsize=36, width=2, length=8)

# Set x-ticks every 6 hours
ax3.set_xticks(x_ticks)
ax3.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=32)
ax3.set_xlim(1, 72)

# Create secondary y-axis for energy
ax3_secondary = ax3.twinx()
color_energy = 'darkgreen'
ax3_secondary.plot(time_series['Hours'], time_series['Energy'], 
                   linewidth=4, color=color_energy, marker='s', markersize=10, 
                   label='Average Energy', linestyle='--')
ax3_secondary.set_ylabel('Average Energy Consumption', fontsize=40, fontweight='bold', 
                         color=color_energy)
ax3_secondary.tick_params(axis='y', labelsize=36, labelcolor=color_energy, width=2, length=8)

# Add grid
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

# Add legend
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_secondary.get_legend_handles_labels()
legend = ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=32, 
                    framealpha=0.9, edgecolor='black')
legend.get_frame().set_linewidth(2)

plt.tight_layout()
load_energy_correlation_plot = os.path.join(BASE_DIR, 'load_energy_correlation_timeseries.png')
plt.savefig(load_energy_correlation_plot, dpi=150, bbox_inches='tight')
print(f"✓ Load-Energy correlation time series plot saved to: {load_energy_correlation_plot}")
plt.close()

# Plot 4: Load vs Energy Scatter Plot (All Data Points)
print("\nGenerating load vs energy scatter plot...")
fig4, ax4 = plt.subplots(1, 1, figsize=(24, 16))

# Plot all data points with larger markers
ax4.scatter(df['load'], df['Energy'], alpha=0.3, s=80, c='steelblue', edgecolors='none')

ax4.set_xlabel('Load', fontsize=40, fontweight='bold')
ax4.set_ylabel('Energy Consumption', fontsize=40, fontweight='bold')
ax4.tick_params(axis='both', which='major', labelsize=36, width=2, length=8)
ax4.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

plt.tight_layout()
scatter_plot_path = os.path.join(BASE_DIR, 'load_energy_scatter.png')
plt.savefig(scatter_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Load vs Energy scatter plot saved to: {scatter_plot_path}")
plt.close()

print("\n" + "="*70)
print("All visualizations completed!")
print("="*70)
