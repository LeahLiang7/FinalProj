"""
Script to explore and visualize data anomalies related to RUType and time of day.
This script:
1. Generates statistical analysis of each RUType (sample size, variability, flat-line %)
2. Plots individual energy consumption curves for each RUType over 72 hours (3 days)
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'filtered_energy_data.csv')
OUTPUT_PLOT = os.path.join(BASE_DIR, 'rutype_energy_anomaly_plot.png')

# --- 1. Load Data ---
print(f"Loading cleaned data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {INPUT_CSV} was not found.")
    print("Please run the data_filter_clean.py script first to generate it.")
    exit()

# --- 2. Statistical Analysis ---
print("\n" + "="*70)
print("RUType Statistical Analysis")
print("="*70)

for rutype in sorted(df['RUType'].unique()):
    rutype_data = df[df['RUType'] == rutype]
    
    print(f"\n{'='*70}")
    print(f"RUType: {rutype}")
    print(f"{'='*70}")
    
    # Basic statistics
    num_bs = rutype_data['BS'].nunique()
    total_records = len(rutype_data)
    
    print(f"Number of base stations: {num_bs}")
    print(f"Total records: {total_records:,}")
    print(f"Records per BS (avg): {total_records/num_bs:.1f}")
    
    # Energy statistics
    print(f"\nEnergy Statistics:")
    print(f"  Mean: {rutype_data['Energy'].mean():.2f}")
    print(f"  Std Dev: {rutype_data['Energy'].std():.2f}")
    print(f"  Min: {rutype_data['Energy'].min():.2f}")
    print(f"  Max: {rutype_data['Energy'].max():.2f}")
    
    # Analyze variability per base station
    bs_variability = []
    flat_line_bs = []
    
    for bs in rutype_data['BS'].unique():
        bs_data = rutype_data[rutype_data['BS'] == bs]['Energy']
        std_dev = bs_data.std()
        bs_variability.append(std_dev)
        
        # Consider a BS as "flat line" if std dev < 2.0
        if std_dev < 2.0:
            flat_line_bs.append(bs)
    
    print(f"\nVariability Analysis (Energy Std Dev per BS):")
    print(f"  Average: {np.mean(bs_variability):.2f}")
    print(f"  Median: {np.median(bs_variability):.2f}")
    print(f"  Min: {np.min(bs_variability):.2f}")
    print(f"  Max: {np.max(bs_variability):.2f}")
    
    flat_line_pct = (len(flat_line_bs) / num_bs) * 100
    print(f"\nFlat-line Base Stations (std dev < 2.0):")
    print(f"  Count: {len(flat_line_bs)} / {num_bs}")
    print(f"  Percentage: {flat_line_pct:.1f}%")

print(f"\n{'='*70}")
print("Statistical Analysis Complete")
print(f"{'='*70}\n")

# --- 3. Convert Hours to 24-hour format (0-23) and calculate Day ---
df['Hour_of_Day'] = (df['Hours'] - 1) % 24  # Convert to 0-23 hour format
df['Day'] = ((df['Hours'] - 1) // 24) + 1  # Calculate which day (1, 2, 3)

# --- 4. Get unique RUTypes ---
rutypes = sorted(df['RUType'].unique())
num_types = len(rutypes)

print(f"Found {num_types} unique RUTypes: {rutypes}")

# --- 5. Create subplots - one for each RUType ---
print("Generating visualization...")
# Calculate optimal grid layout
n_cols = min(3, num_types)  # Max 3 columns
n_rows = (num_types + n_cols - 1) // n_cols  # Ceiling division

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
if num_types == 1:
    axes = np.array([axes])
axes = axes.flatten()

# --- 6. Plot each RUType in a separate subplot ---
for idx, rutype in enumerate(rutypes):
    ax = axes[idx]
    
    # Filter data for this RUType
    rutype_data = df[df['RUType'] == rutype].copy()
    
    # Plot each individual energy curve (each base station)
    for bs in rutype_data['BS'].unique():
        bs_data = rutype_data[rutype_data['BS'] == bs].sort_values('Hours')
        ax.plot(bs_data['Hours'], bs_data['Energy'], alpha=0.3, linewidth=0.8)
    
    # Customize subplot
    ax.set_title(f'{rutype} - Energy Patterns (3 Days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour (72h = 3 days)', fontsize=12)
    ax.set_ylabel('Energy Consumption', fontsize=12)
    
    # Set x-ticks every 6 hours with 24-hour format labels
    x_ticks = range(1, 73, 6)
    x_labels = [f"Day{((h-1)//24)+1}\n{((h-1)%24):02d}:00" for h in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(1, 72)

# Hide unused subplots
for idx in range(num_types, len(axes)):
    axes[idx].axis('off')

# --- 7. Save and Show Plot ---
plt.suptitle('Energy Consumption Anomaly Analysis by RUType', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
print(f"Anomaly exploration plot saved to: {OUTPUT_PLOT}")

# Optional: Show the plot
# plt.show()

print("\nAnalysis complete. Please check the generated plot to identify anomalies.")
print(f"Each subplot shows all individual energy curves for a specific RUType over 3 days.")
