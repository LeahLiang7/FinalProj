"""
Data Preprocessing Script - Generate Energy Data Coverage Heatmap
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load and process data
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load data files
df_bsinfo = pd.read_csv(os.path.join(base_dir, 'BSinfo.csv'))
df_cldata = pd.read_csv(os.path.join(base_dir, 'CLdata.csv'))
df_ecdata = pd.read_csv(os.path.join(base_dir, 'ECdata.csv'))



# 1. Parse time columns and generate 'Hours' field
df_cldata['Time'] = pd.to_datetime(df_cldata['Time'])
df_ecdata['Time'] = pd.to_datetime(df_ecdata['Time'])
start_time = pd.to_datetime('1/1/2023 1:00')
df_cldata['Hours'] = ((df_cldata['Time'] - start_time).dt.total_seconds() / 3600).astype(int) + 1
df_ecdata['Hours'] = ((df_ecdata['Time'] - start_time).dt.total_seconds() / 3600).astype(int) + 1

# 2. Merge CLdata with BSinfo, then merge with ECdata
df_step1 = pd.merge(df_cldata, df_bsinfo, on=['BS', 'CellName'], how='left')
df_merged = pd.merge(df_step1, df_ecdata[['Time', 'BS', 'Energy', 'Hours']], on=['Time', 'BS'], how='inner')

# 3. Extract BS number for filtering
df_merged['BS_num'] = df_merged['BS'].str.extract(r'(\d+)').astype(int)

# 4. Filter data: BS_num <= 809, Hours <= 72, CellName == 'Cell0'
df_merged = df_merged[(df_merged['BS_num'] <= 809) &
                     (df_merged['Hours_x'] <= 72) &
                     (df_merged['CellName'] == 'Cell0')].copy()

# 5. Flag multi-cell cases
cell_counts = df_merged.groupby(['BS', 'Time']).size().reset_index(name='num_cells_for_this_energy')
df_merged = df_merged.merge(cell_counts, on=['BS', 'Time'], how='left')
df_merged['energy_shared_by_multiple_cells'] = df_merged['num_cells_for_this_energy'] > 1

# 6. Save the cleaned data
output_path = os.path.join(base_dir, 'energy_data_final.csv')
important_cols = ['energy_shared_by_multiple_cells', 'num_cells_for_this_energy',
                  'Time', 'Hours_x', 'BS', 'BS_num', 'CellName', 'Energy', 'load']
other_cols = [col for col in df_merged.columns if col not in important_cols]
df_merged = df_merged[important_cols + other_cols]
df_merged = df_merged.rename(columns={'Hours_x': 'Hours'})
df_merged.to_csv(output_path, index=False)

print(f"Data processing complete. Saved {len(df_merged):,} rows to {output_path}")

# Create comprehensive data coverage heatmap for ALL base stations
print("Generating data coverage heatmap...")

# Get all possible BS IDs (B_0 to B_1019, total 1020 base stations)
all_bs_ids = [f'B_{i}' for i in range(1020)]

# Get all time points from the data
all_times = sorted(df_merged['Time'].unique())
num_hours = len(all_times)

print(f"Total base stations: {len(all_bs_ids)}, Total time points: {num_hours} hours")

# Create coverage matrix: rows=hours, columns=ALL base stations (B_0 to B_1019)
coverage_matrix = np.zeros((num_hours, len(all_bs_ids)))

# Get data for single-cell base stations (only Cell0, no multiple cells)
single_cell_bs = set(df_bsinfo[df_bsinfo['CellName'] == 'Cell0']['BS'].unique())
multi_cell_bs = set(df_bsinfo.groupby('BS')['CellName'].nunique()[df_bsinfo.groupby('BS')['CellName'].nunique() > 1].index)

# Create lookup for energy data availability
energy_data_lookup = set(zip(df_merged['BS'], df_merged['Time']))

# Fill the matrix: 1 = green (single cell + has energy data), 0 = blue (missing/multiple cells)
for time_idx, time_point in enumerate(all_times):
    for bs_idx, bs_id in enumerate(all_bs_ids):
        # Green: BS has only Cell0 AND has energy data at this time point
        if bs_id in single_cell_bs and (bs_id, time_point) in energy_data_lookup:
            coverage_matrix[time_idx, bs_idx] = 1
        # Blue: Either multiple cells OR no energy data (including missing BS)
        else:
            coverage_matrix[time_idx, bs_idx] = 0

# Create large figure to show all 1020 base stations clearly
plt.figure(figsize=(20, 10))

# Create custom colormap: 0=dark blue (missing/multiple cells), 1=green (single cell + data)
colors = ['darkblue', 'green']
custom_cmap = ListedColormap(colors)

plt.imshow(coverage_matrix, aspect='auto', cmap=custom_cmap, origin='upper')

# Customize the plot
plt.title("Energy Data Coverage for All Base Stations (B_0 to B_1019)", fontsize=16, fontweight='bold')
plt.xlabel("Base Station Index (B_0 to B_1019)", fontsize=14)
plt.ylabel(f"Hours from Start (Total: {num_hours} hours)", fontsize=14)

# Add detailed colorbar
cbar = plt.colorbar(ticks=[0, 1], shrink=0.6)
cbar.set_ticklabels(['Missing Data / Multiple Cells', 'Single Cell + Data Present'])
cbar.ax.tick_params(labelsize=12)

# Set axis ticks for better readability
# X-axis: show every 100th base station
x_tick_positions = range(0, len(all_bs_ids), 100)
x_tick_labels = [f'B_{i*100}' for i in range(len(x_tick_positions))]
plt.xticks(x_tick_positions, x_tick_labels, rotation=45)

# Y-axis: show every 24 hours (daily intervals)
y_tick_positions = range(0, num_hours, 24)
y_tick_labels = [f'{h}h' for h in range(0, num_hours, 24)]
plt.yticks(y_tick_positions, y_tick_labels)

# Add grid for better readability
plt.grid(True, alpha=0.2)

# Add text annotation with statistics
total_possible = num_hours * len(all_bs_ids)
green_cells = np.sum(coverage_matrix)
coverage_percentage = (green_cells / total_possible) * 100

plt.figtext(0.02, 0.02, f'Statistics: {int(green_cells):,}/{total_possible:,} cells ({coverage_percentage:.2f}%) have single-cell energy data', 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Save the plot
output_plot_path = os.path.join(base_dir, 'comprehensive_energy_data_coverage_heatmap.png')
plt.tight_layout()
plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {output_plot_path}")

print(f"Detailed Statistics:")
print(f"  Total base stations displayed: {len(all_bs_ids)}")
print(f"  Total time points: {num_hours} hours")
print(f"  Coverage percentage: {coverage_percentage:.2f}%")

# Show the plot
plt.show()