import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import xarray as xr

eta_index, xi_index = 29, 8 # y->eta direction, x->xi direction
v_transform = 2

# --- Read NetCDF files ---
ds_u = xr.open_dataset(
    f'../output/ROMS_u_errors_eta{eta_index}_xi{xi_index}.nc'
)
ds_v = xr.open_dataset(
    f'../output/ROMS_v_errors_eta{eta_index}_xi{xi_index}.nc'
)

# --- Extract error matrices natively ---
# Xarray set dimensions to (time, layer).
# The layer axis is flipped to keep higher numbers at the top.
u_matrix = ds_u['error'].transpose('layer', 'time').values[::-1, :]
v_matrix = ds_v['error'].transpose('layer', 'time').values[::-1, :]

# --- Define Plot Extent ---
# Convert Xarray/numpy datetime64 times to matplotlib numerical dates
times_num = mdates.date2num(ds_u['time'].values)
layers = ds_u['layer'].values
extent = [times_num[0], times_num[-1], layers[0], layers[-1]]

# --- Calc error extremes for reference --- 
err_min_v = np.nanmin(v_matrix).item()
err_max_v = np.nanmax(v_matrix).item()
err_min_u = np.nanmin(u_matrix).item()
err_max_u = np.nanmax(u_matrix).item()
print(f"U min/max: {err_min_u}, {err_max_u} | V min/max: {err_min_v}, {err_max_v}")

# ------------------------------------------------------------
# PLOT HEATMAPS
# ------------------------------------------------------------
fig_map, (ax_m1, ax_m2) = plt.subplots(
    2, 1, figsize=(12, 10), sharex=True
)

# calcluate and plot u-velocity error heatmap
u_vmax = (
    np.nanpercentile(u_matrix, 95)
    if not np.isnan(u_matrix).all()
    else 1.0
)
if u_vmax <= 0:
    u_vmax = 1.0
u_im = ax_m1.imshow(
    u_matrix,
    aspect="auto",
    extent=extent,
    cmap="YlOrRd",
    vmin=0,
    vmax=5e-6,#u_vmax,
    interpolation="nearest",
)

# Force colorbar structure to match subplot height precisely
divider1 = make_axes_locatable(ax_m1)
cax1 = divider1.append_axes("right", size="3%", pad=0.15)
fig_map.colorbar(u_im, cax=cax1, label="Absolute Difference")

# Add title
ax_m1.set_title(
    "u-velocity Absolute Error Heatmap (Capped at "
    f"95th percentile: {u_vmax:.4f})",
    fontsize=12,
    fontweight="bold",
)
ax_m1.set_ylabel("Vertical Grid Layer Index", fontsize=11)
ax_m1.text(
    0.02,
    0.95,
    (f"eta,xi indices: {eta_index}, {xi_index}\n"
    f"Min error: {err_min_u:4.2e}\n" 
    f"Max error: {err_max_u:4.2e}"),
    transform=ax_m1.transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    fontsize = 12, 
    linespacing=1.5,
    zorder=5,
    bbox=dict(
        facecolor="white", alpha=0.7, edgecolor="none", pad=3.0
    ),
)

# Calculate and plot v-velocity absolute error heatmap
v_vmax = (
    np.nanpercentile(v_matrix, 95)
    if not np.isnan(v_matrix).all()
    else 1.0
)
if v_vmax <= 0:
    v_vmax = 1.0
v_im = ax_m2.imshow(
    v_matrix,
    aspect="auto",
    extent=extent,
    cmap="YlOrRd",
    vmin=0,
    vmax=5e-6,#v_vmax,
    interpolation="nearest",
)
# Manage colorbar
divider2 = make_axes_locatable(ax_m2)
cax2 = divider2.append_axes("right", size="3%", pad=0.15)
fig_map.colorbar(v_im, cax=cax2, label="Absolute Difference")
# Add title
ax_m2.set_title(
    "v-velocity Absolute Error Heatmap (Capped at "
    f"95th percentile: {v_vmax:.4f})",
    fontsize=12,
    fontweight="bold",
)
ax_m2.set_ylabel("Vertical Grid Layer Index", fontsize=11)
ax_m2.set_xlabel("Date / Time", fontsize=11)
ax_m2.text(
    0.02,
    0.95,
    (f"eta,xi indices: {eta_index}, {xi_index}\n"
    f"Min error: {err_min_v:4.2e}\n" 
    f"Max error: {err_max_v:4.2e}"),
    transform=ax_m2.transAxes,
    fontsize = 12,
    linespacing=1.5,
    verticalalignment="top",
    horizontalalignment="left",
    zorder=5,
    bbox=dict(
        facecolor="white", alpha=0.7, edgecolor="none", pad=3.0
    ),
)
# format axis
ax_m2.xaxis_date()
ax_m2.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y-%m-%d %H:%M")
)
fig_map.autofmt_xdate()

# save graphic
plt.tight_layout()
fig_map.subplots_adjust(right=0.90)

fig_map.savefig(
    f"../graphics/velocity_abs_errors_heatmap_vt{v_transform}.png", bbox_inches="tight"
)
plt.close(fig_map)

# Close datasets
ds_u.close()
ds_v.close()

print(f"Saved: velocity_errors_heatmap_vt{v_transform}.png")