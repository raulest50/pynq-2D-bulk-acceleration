# results_plot.py
# Recreate the FPGA vs CPU performance/energy figure as a 2x2 bar chart grid.

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# --- Raw measurements (from paper) ---
# Per-PSF latency (seconds)
time_sec = {
    "FPGA (KV260)": 1.567,
    "AMD Ryzen (5300U)": 7.791853,
    "Intel Xeon (w3-2425)": 4.202998,  # Measured from console_outputs.txt
}

# Power (Watts)
power_w = {
    "FPGA (KV260)": 4.451,   # post-implementation on-chip (PS+PL)
    "AMD Ryzen (5300U)": 15.0,     # TDP assumption
    "Intel Xeon (w3-2425)": 90.0,  # Estimated based on workload
}

# --- Derived metrics ---
labels = list(time_sec.keys())
throughput_psfs = {k: 1.0 / v for k, v in time_sec.items()}         # PSF/s
energy_per_psf_j = {k: power_w[k] * time_sec[k] for k in labels}    # J/PSF

# Define a professional color scheme (colorblind-friendly)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
hatches = ['', '///', '...']  # Different patterns for grayscale printing

# --- Helper to make a simple bar chart ---
def bar_ax(ax, values_dict, title, ylabel):
    xs = range(len(labels))
    vals = [values_dict[l] for l in labels]

    # Create bars with different colors and hatches
    bars = ax.bar(xs, vals, color=colors, hatch=hatches, edgecolor='black', linewidth=0.5)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs, labels, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # annotate values on top of bars with larger font
    for rect, v in zip(bars, vals):
        ax.annotate(f"{v:.3g}",
                    xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight='bold')  # Increased from 9 to 10 and made bold

# --- Plot ---
# Update font settings for better readability
plt.rcParams.update({
    'font.size': 12,           # Increase base font size from 11 to 12
    'font.weight': 'bold',     # Make all text bold by default
    'axes.titlesize': 13,      # Slightly larger subplot titles
    'axes.titleweight': 'bold',# Bold subplot titles
    'axes.labelsize': 12,      # Ensure axis labels are readable
    'axes.labelweight': 'bold',# Bold axis labels
    'xtick.labelsize': 11,     # Readable tick labels
    'ytick.labelsize': 11,     # Readable tick labels
    'legend.fontsize': 12,     # Larger legend text
    'figure.titlesize': 16,    # Larger figure title
    'figure.titleweight': 'bold' # Bold figure title
})

# Slightly increase figure size for better spacing
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
plt.subplots_adjust(wspace=0.3, hspace=0.5)  # Slightly more spacing between subplots

bar_ax(axes[0, 0], time_sec,       "Time per PSF (s)",        "Seconds")
bar_ax(axes[0, 1], throughput_psfs,"Throughput (PSF/s)",       "PSF/s")
bar_ax(axes[1, 0], power_w,        "Power Consumption (W)",    "Watts")
bar_ax(axes[1, 1], energy_per_psf_j,"Energy per PSF (J/PSF)",  "Joules")

# Optional: tidy up y-lims to add a bit of headroom for the labels
for ax in axes.flat:
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

# Add a single legend for the entire figure
legend_elements = [Patch(facecolor=colors[i], hatch=hatches[i], edgecolor='black', label=labels[i]) 
                  for i in range(len(labels))]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
          ncol=3, frameon=False)

# Add a figure title
fig.suptitle('Performance and Energy Comparison', fontsize=14, y=1.02)

fig.tight_layout()
# Save as high-resolution PNG (600 DPI for better print quality)
fig.savefig("results.png", dpi=600, bbox_inches='tight')
# Also save as PDF (vector format for publication-quality graphics)
fig.savefig("results.pdf", bbox_inches='tight')
print("Saved figures to results.png and results.pdf")
# plt.show()  # uncomment if you want an interactive window
