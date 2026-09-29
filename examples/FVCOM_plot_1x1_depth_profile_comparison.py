import pandas as pd
import numpy as np
import xarray as xr
import pdb
import matplotlib.pyplot as plt
from utils import calc_depth_profile
import gridded.plotting.mpl_plotting as gplt
from pathlib import Path

netcdf_name = "SFBOFS_3D_07_2026"

# Generate a list of node and centroid locations to query  
# based off interactive graphic
csv_path = gplt.fvcom_inspector( "../input_files/SFBOFS_3D_07_2026.nc")
#csv_path = Path(f"../input_files/{netcdf_name}_selected_cell.csv")

# Load CSV with index and lat/lon information of selection by rows of:
# 'nele', 'nele_lat', 'nele_lon', <----- Face/centroid ID
# 'node1', 'node1_lat', 'node1_lon'-.
# 'node2', 'node2_lat', 'node2_lon' | <-- Surrounding node IDs
# 'node3', 'node3_lat', 'node3_lon'-'
gpts=pd.read_csv(csv_path)

# Calculate depth profile at one of the locations in the CSV file
nele = gpts['nele'][0]
input_netcdf = f"../input_files/{netcdf_name}.nc"
profile_netcdf = f"../output/{netcdf_name}_depthprofiles_nele{nele}.nc"

ds, nc_path = calc_depth_profile(
    model_input = input_netcdf, 
    model_type = "fvcom", 
    index = nele, 
    output_netcdf = profile_netcdf
)

# =================================================================
# SETUP
# =================================================================
# Load netcdf created by FVCOM_calc_depth_profile.py
ds = xr.open_dataset(nc_path)
times = ds.time.values
gridded_depth_u = ds.gridded_depth_u.values
gridded_depth_v = ds.gridded_depth_v.values
# marker size for line plot markers
ms = 8 
# line plot colors: u-velocities (cool), v-velocities (warm) 
colors = {
    "model_u": "teal",
    "gridded_u": "lightseagreen",
    "model_v": "#c65911",  # Deep Burnt Orange
    "gridded_v": "#ffc000",  # Bright Tangerine/Yellow-Orange
}

# =================================================================
# PLOT FVCOM PROFILE VS GRIDDED INTERPOLATION (sample of timesteps)
# =================================================================
Ntimes = len(times) - 1 #number of time steps to plot
nrows, ncols = 3, 3

fig1, axes = plt.subplots(
    figsize=(5, 7), 
    # sharey=True, 
    # sharex=True
)

ax = axes
t_str = pd.to_datetime(times[Ntimes]).strftime("%Y-%m-%d %H:%M")

# -------------
# FVCOM PROFILES (Dashed-dotted lines, with markers)
# -------------
# u-velocity component 
ax.plot(
    ds["model_u"][Ntimes].values,
    ds["model_depth_u"][Ntimes].values,
    color=colors["model_u"],
    alpha=0.5,
    linestyle="-",
    marker=".",
    markersize=ms,
    linewidth=1.5,
    label="Model u",
)

# v-velocity component
ax.plot(
    ds["model_v"][Ntimes].values,
    ds["model_depth_v"][Ntimes].values,
    color=colors["model_v"],
    alpha=0.5,
    linestyle="-",
    marker=".",
    markersize=ms,
    linewidth=1,
    label="Model v",
)

# ----------------
# GRIDDED PROFILES (Solid lines, NO markers)
# ----------------
# u-velocity component 
ax.plot(
    ds["gridded_u"][Ntimes].values,
    gridded_depth_u,
    color=colors["gridded_u"],
    linestyle="-",
    linewidth=1.8,
    label="Gridded u",
)

# v-velocity component
ax.plot(
    ds["gridded_v"][Ntimes].values,
    gridded_depth_v,
    color=colors["gridded_v"],
    linestyle="-",
    linewidth=1.8,
    label="Gridded v",
)

# Subplot styling
ax.set_title(t_str, fontsize=11, weight="bold")
ax.grid(True, linestyle=":", alpha=0.6)
ax.axvline(0, color="black", linestyle="-", alpha=0.4)
ax.invert_yaxis()  # Surface (0m) at the top

# Add labels
ax.set_ylabel("Depth (m)", fontsize=11)
ax.set_xlabel("Velocity (m/s)", fontsize=11)

# # Trim any trailing empty subplots
# for j in range(i + 1, len(axes)):
#     fig1.delaxes(axes[j])

# Place legend at the top
fig1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 0.92),
    ncol=2,
    fontsize=11,
    frameon=True,
)
model_name = ds.attrs.get("model name", "Hydrodynamic Model").upper()
plt.suptitle(
    f"Velocity Depth Profile\n({model_name} vs Gridded)",
    fontsize=16,
    weight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(
    f"../graphics/{netcdf_name}_vs_Gridded_1x1_profile_comparison.png", dpi=300
)
plt.show()
plt.close()