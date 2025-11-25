"""
Data Preprocessing Script - Exploratory Version
Strategy:
1. Merge BSinfo + CLdata (by BS, CellName)
2. Merge result + ECdata (by BS, Time)
3. Flag cases where multiple rows share the same Energy value
"""

import pandas as pd
import os

# 1. Load Data
base_dir = os.path.dirname(os.path.abspath(__file__))

print("\nLoading data files...")
df_bsinfo = pd.read_csv(os.path.join(base_dir, 'BSinfo.csv'))
df_cldata = pd.read_csv(os.path.join(base_dir, 'CLdata.csv'))
df_ecdata = pd.read_csv(os.path.join(base_dir, 'ECdata.csv'))

print(f"  BSinfo: {df_bsinfo.shape} - Columns: {df_bsinfo.columns.tolist()}")
print(f"  CLdata: {df_cldata.shape} - Columns: {df_cldata.columns.tolist()}")
print(f"  ECdata: {df_ecdata.shape} - Columns: {df_ecdata.columns.tolist()}")


# 2. First Merge: CLdata + BSinfo (by BS, CellName)

print("\n[MERGE 1] CLdata + BSinfo (on BS, CellName)")

df_step1 = pd.merge(
    df_cldata, 
    df_bsinfo, 
    on=['BS', 'CellName'], 
    how='left'
)

print(f"  Rows after merge: {df_step1.shape[0]:,}")
print(f"  Columns after merge: {df_step1.shape[1]}")
print(f"  Unique (BS, CellName, Time): {len(df_step1.groupby(['BS', 'CellName', 'Time'])):,}")

# Check for unmatched records
unmatched = df_step1['Frequency'].isna().sum()
if unmatched > 0:
    print(f"  WARNING: {unmatched} rows not matched to BSinfo")


# 3. Second Merge: Step1 result + ECdata (by BS, Time)

print("\n[MERGE 2] Step1 result + ECdata (on BS, Time)")

df_merged = pd.merge(
    df_step1,
    df_ecdata,
    on=['BS', 'Time'],
    how='left'
)

print(f"  Rows after merge: {df_merged.shape[0]:,}")
print(f"  Columns after merge: {df_merged.shape[1]}")

# Check for unmatched Energy values
unmatched_energy = df_merged['Energy'].isna().sum()
if unmatched_energy > 0:
    print(f"  {unmatched_energy} rows not matched to Energy (CLdata has Time not in ECdata)")
    df_merged = df_merged.dropna(subset=['Energy'])
    print(f"  Rows after removal: {df_merged.shape[0]:,}")


# 4. Flag Multiple Cell Cases

print("\n[ANALYSIS] Multiple Cell Cases")

# Count how many rows (cells) per (BS, Time)
cell_counts = df_merged.groupby(['BS', 'Time']).size().reset_index(name='num_cells_for_this_energy')

# Merge back
df_merged = df_merged.merge(cell_counts, on=['BS', 'Time'], how='left')

# Add flag: whether this Energy is shared by multiple cells
df_merged['energy_shared_by_multiple_cells'] = df_merged['num_cells_for_this_energy'] > 1

# Statistics
total_records = len(df_merged)
multi_cell_records = df_merged['energy_shared_by_multiple_cells'].sum()
unique_energy_values = df_merged.groupby(['BS', 'Time']).size().shape[0]

print(f"\nStatistics:")
print(f"  Total records: {total_records:,}")
print(f"  Unique (BS, Time) combinations: {unique_energy_values:,}")
print(f"  Records with shared Energy: {multi_cell_records:,} ({multi_cell_records/total_records*100:.2f}%)")


# 5. Save to Single CSV File
# Update output_path to use a relative path
output_path = os.path.join(base_dir, 'energy_data_final.csv')

print(f"\nSaving data to: {output_path}")

# Reorder columns - put flag columns first for easy viewing
important_cols = ['energy_shared_by_multiple_cells', 'num_cells_for_this_energy', 
                  'Time', 'BS', 'CellName', 'Energy', 'load']
other_cols = [col for col in df_merged.columns if col not in important_cols]
df_merged = df_merged[important_cols + other_cols]

df_merged.to_csv(output_path, index=False)

print(f"Successfully saved {len(df_merged):,} rows")