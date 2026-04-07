"""
Re-generate the ESMode cumulative distribution plot, strictly following the user's
provided logic and reference image.

Key changes:
1. Filter for ESMode > 0 before plotting the CDF. This focuses only on active durations.
2. Use plt.step() to create the characteristic stepped appearance of an empirical CDF.
3. Set the x-axis to log scale and customize ticks to match the reference image.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set plot style to be more visually appealing and closer to the reference
mpl.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5


# 1. Load data
print("Loading energy_data_final.csv...")
df = pd.read_csv('energy_data_final.csv')
print(f"Loaded {len(df)} records.")

# 2. Convert proportion to seconds AND filter for active durations (> 0)
print("Filtering for active durations (ESMode > 0)...")
esm1_seconds = df[df['ESMode1'] > 0]['ESMode1'] * 3600
esm2_seconds = df[df['ESMode2'] > 0]['ESMode2'] * 3600
print(f"Found {len(esm1_seconds)} active records for ESMode1.")
print(f"Found {len(esm2_seconds)} active records for ESMode2.")


# 3. Define a function to calculate the CDF values for step plotting
def get_cdf_for_step(data):
    """
    Calculates the x and y coordinates for a step-style CDF plot.
    Returns sorted data points and their corresponding cumulative probabilities.
    """
    # Add a 0 at the beginning to make the plot start from the y-axis
    sorted_data = np.sort(data)
    # The y-values are the cumulative probabilities
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y

# Calculate CDF for both modes
x1, y1 = get_cdf_for_step(esm1_seconds)
x2, y2 = get_cdf_for_step(esm2_seconds)

# 4. Create the plot
print("Generating the plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Use plt.step to draw the CDF, which is crucial for the "staircase" look
ax.step(x1, y1, label='ESMode1', where='post', linewidth=2.5, color='#1f77b4')
ax.step(x2, y2, label='ESMode2', where='post', linewidth=2.5, color='#ff7f0e')

# 5. Set scales and labels exactly as in the reference image
ax.set_xscale('log')
ax.set_xlabel('Activation duration (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative probability', fontsize=12, fontweight='bold')
ax.set_title('ESMode Activation Duration CDF (Active Samples Only)', fontsize=14, fontweight='bold')

# Set axis limits
ax.set_ylim(0, 1.0)
ax.set_xlim(left=0.8) # Start just before 1s to see the first step

# Customize X-axis ticks to match the reference: 1s, 5s, 1m, 5m, 15m, 30m, 1h
xticks = [1, 5, 60, 300, 900, 1800, 3600]
xtick_labels = ['1 s', '5 s', '1 m', '5 m', '15 m', '30 m', '1 h']
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels, fontsize=11)

# Customize grid to match the reference (light, dashed)
ax.grid(True, which="both", ls="--", alpha=0.5)

# Customize legend
legend = ax.legend(loc='upper left', fontsize=12, frameon=True, edgecolor='black')
legend.get_frame().set_linewidth(1.5)

# Make the plot frame more prominent
for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()

# Save the final plot
output_filename = 'esmode_cdf_replication.png'
plt.savefig(output_filename, dpi=300)
print(f"\n[SUCCESS] Plot saved as {output_filename}")
plt.close()

# --- Statistical Verification ---
print("\n--- Verifying Key Thresholds from Data ---")

# ESMode1: What's the minimum activation time?
min_esm1_sec = esm1_seconds.min()
print(f"ESMode1 Minimum Activation Duration: {min_esm1_sec:.2f} seconds")
if 4.9 < min_esm1_sec < 5.1:
    print("  -> This confirms the ~5s activation threshold for ESMode1.")

# ESMode2: How many samples are below 5 seconds?
esm2_below_5s = (esm2_seconds < 5).sum()
esm2_below_5s_percent = 100 * esm2_below_5s / len(esm2_seconds)
print(f"\nESMode2 samples with duration < 5 seconds: {esm2_below_5s} ({esm2_below_5s_percent:.2f}% of active samples)")
if esm2_below_5s > 0:
    print("  -> This confirms that ESMode2 can be active for very short durations (<5s).")

print("\nAnalysis complete. The generated plot should now match the reference image.")
