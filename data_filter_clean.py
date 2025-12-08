"""
Data filtering and cleaning script for 5G energy dataset.
This script processes the raw CLdata, BSinfo, and ECdata files and outputs a clean filtered dataset for downstream tasks.
"""

import pandas as pd
import os

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load raw data
cldata = pd.read_csv(os.path.join(base_dir, 'CLdata.csv'))
bsinfo = pd.read_csv(os.path.join(base_dir, 'BSinfo.csv'))
ecdata = pd.read_csv(os.path.join(base_dir, 'ECdata.csv'))

# Parse time columns
cldata['Time'] = pd.to_datetime(cldata['Time'])
ecdata['Time'] = pd.to_datetime(ecdata['Time'])

# Calculate hours from start (1/1/2023 1:00)
start_time = pd.to_datetime('1/1/2023 1:00')
cldata['Hours'] = ((cldata['Time'] - start_time).dt.total_seconds() / 3600).astype(int) + 1
ecdata['Hours'] = ((ecdata['Time'] - start_time).dt.total_seconds() / 3600).astype(int) + 1

# Merge datasets
merged = cldata.merge(ecdata[['Time', 'BS', 'Energy']], on=['Time', 'BS'], how='inner')
merged = merged.merge(bsinfo, on=['BS', 'CellName'], how='left')

# Extract BS number for filtering
merged['BS_num'] = merged['BS'].str.extract(r'(\d+)').astype(int)

# Identify base stations that ONLY have Cell0 (no other cells)
bs_cell_counts = bsinfo.groupby('BS')['CellName'].nunique()
single_cell_bs = bs_cell_counts[bs_cell_counts == 1].index.tolist()

# Further filter: only keep BS that have Cell0 as their only cell
single_cell0_bs = bsinfo[(bsinfo['BS'].isin(single_cell_bs)) & 
                         (bsinfo['CellName'] == 'Cell0')]['BS'].unique()

print(f"Total BS with only one cell: {len(single_cell_bs)}")
print(f"BS with only Cell0: {len(single_cell0_bs)}")

# Filter data: BS 0-809, Hours 1-72, BS that only have Cell0
filtered = merged[(merged['BS_num'] <= 809) &
                  (merged['Hours'] <= 72) &
                  (merged['BS'].isin(single_cell0_bs)) &
                  (merged['CellName'] == 'Cell0')].copy()

print(f"Filtered data: {len(filtered):,} rows")
print(f"Unique BS count: {filtered['BS'].nunique()}")
print(f"Time range after filter: Hours {filtered['Hours'].min()} to {filtered['Hours'].max()}")

# Save filtered data for downstream tasks
filtered_csv_path = os.path.join(base_dir, 'filtered_energy_data.csv')
filtered.to_csv(filtered_csv_path, index=False)
print(f"Filtered data saved to: {filtered_csv_path}")
