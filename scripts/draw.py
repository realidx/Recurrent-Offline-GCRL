import matplotlib.pyplot as plt
import numpy as np

# --- Data Definition (from the provided table image) ---
# Each array corresponds to [ALS, AGN, Scene] datasets.

# Data for m=1 (setting standard)
hiql_mean_m1 = np.array([0.497, 0.605, 0.480])
hiql_std_m1 = np.array([0.043, 0.056, 0.028])

crl_mean_m1 = np.array([0.504, 0.331, 0.288])
crl_std_m1 = np.array([0.096, 0.056, 0.045])

# Data for m=2 (target setting)
hiql_mean_m2 = np.array([0.691, 0.722, 0.565])
hiql_std_m2 = np.array([0.108, 0.043, 0.065])

crl_mean_m2 = np.array([0.380, 0.173, 0.293])
crl_std_m2 = np.array([0.035, 0.034, 0.038])

# --- Calculate Differences and Propagated Errors (m2 - m1) ---
# Formula for difference: Diff = Mean2 - Mean1
# Formula for error propagation: Err_diff = sqrt(Err1^2 + Err2^2)

# HIQL Difference calculations
hiql_diff = hiql_mean_m2 - hiql_mean_m1
hiql_diff_std = np.sqrt(hiql_std_m1**2 + hiql_std_m2**2)

# CRL Difference calculations
crl_diff = crl_mean_m2 - crl_mean_m1
crl_diff_std = np.sqrt(crl_std_m1**2 + crl_std_m2**2)

# Print values for verification (optional)
# print("HIQL Diffs (m=2 - m=1):", hiql_diff)
# print("HIQL Diff Stds:", hiql_diff_std)
# print("CRL Diffs (m=2 - m=1):", crl_diff)
# print("CRL Diff Stds:", crl_diff_std)

# --- Define Plotting Elements ---
datasets = ['ALS', 'AGN', 'Scene']
algorithms = ['HIQL', 'CRL']

# Choose "pretty" but distinct colors to differentiate the algorithms.
# Grouped bar charts require distinct colors to be readable.
color_hiql = '#1f77b4'  # A nice, standard, deep blue
color_crl = '#ff7f0e'   # A complementary, pleasing orange

# Set bar positions and widths for the grouped layout
x = np.arange(len(datasets))  # The label locations: [0, 1, 2]
width = 0.35  # The width of each bar

# --- Start Creating the Figure ---
# Set some global aesthetics for a clean look
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars with error bars (yerr)
# Bars are centered at x - width/2 (HIQL) and x + width/2 (CRL)
rects1 = ax.bar(x - width/2, hiql_diff, width, 
                label='HIQL (m=2 - m=1)', 
                color=color_hiql, 
                alpha=0.8, # slight transparency for a softer look
                edgecolor='black', # clear outlines
                yerr=hiql_diff_std, # standard deviations for the difference
                capsize=5, # error bar cap size
                ecolor='black', # error bar color
                error_kw={'alpha': 0.6}) # error bar transparency

rects2 = ax.bar(x + width/2, crl_diff, width, 
                label='CRL (m=2 - m=1)', 
                color=color_crl, 
                alpha=0.8, 
                edgecolor='black', 
                yerr=crl_diff_std, 
                capsize=5, 
                ecolor='black', 
                error_kw={'alpha': 0.6})

# --- Add Labels, Formatting, and Annotations ---
# Y-axis label explicitly defines the difference being plotted.
ax.set_ylabel('Change in Success Rate (m=2 - m=1)', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_title('Improvement in Performance from m=1 to m=2', fontsize=14, fontweight='bold', pad=20)

# Set the x-axis tick labels to be the dataset names
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)

# Add a prominent horizontal line at y=0.
# Bars above this show improvement, bars below show regression.
ax.axhline(0, color='black', linewidth=1.5, linestyle='-')

# Add minor grid lines for readability along the y-axis
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)

# Place the legend in a location that doesn't overlap data, like the lower right.
ax.legend(loc='lower right', frameon=True, fontsize=10)

# Ensure everything fits within the figure area nicely.
fig.tight_layout()

# Save the figure as a high-quality PNG.
plt.savefig('performance_difference_grouped_bar_chart.png', dpi=300)

# Display the plot.
plt.show()