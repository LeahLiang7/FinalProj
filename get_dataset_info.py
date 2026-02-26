import pandas as pd

df = pd.read_csv('filtered_energy_data.csv')

print('Dataset Overview:')
print('='*80)
print(f'Total observations: {len(df):,}')
print(f'Number of base stations: {df["BS"].nunique()}')
print(f'Time period: {df["Time"].min()} to {df["Time"].max()}')
print(f'Number of Radio Unit types: {df["RUType"].nunique()}')
print(f'\nRadio Unit Types: {sorted(df["RUType"].unique())}')
print(f'\nData fields: {list(df.columns)}')
print(f'\nLoad range: {df["load"].min():.4f} to {df["load"].max():.4f}')
print(f'Energy range: {df["Energy"].min():.4f} to {df["Energy"].max():.4f}')
