from pathlib import Path
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import gridded
from roms_tools import roms_grid_velocity_averages
from roms_qaqc_plots import map_roms_error_with_vectors
# -----------------------------------------------------------------------
#
# ROMS Arakawa-C grid dimensions 101:
# - RHO .. [M, L]
# - u .... [M, L-1]
# - v .... [M-1, L]
# - psi .. [M-1, L-1]

# Once u-velocity is manually averaged to rho, v, and psi locations,
# the original rho, v, and psi grids need to be trimmed accordinginly
# in order to align with the manually-averaged locations:

# - RHO .. [0:, 1:-1]   -> final dimensions [M, L-2]
# - v .... [0:-1, 1:-1] -> final dimensions [M-1, L-2]
# - psi .. [0:, 0:]     -> final dimensions [M-1, L-1]

# Similarly with the v-velocity
# - RHO .. [1:-1, 0:]   -> final dimensions [M-2, L]
# - u .... [1:-1,0:]    -> final dimensions [M-2, L]
# - psi .. [0:,0:]      -> final dimensions [M-1, M-1]

# -----------------------------------------------------------------------




# --- Configuration & Input Setup ---
roms_file = Path("../../gnome_test_files/gridded_test_files/ROMS_wcofs_May2026_3D.nc")
input_file = roms_file
ds = xr.load_dataset(roms_file)
n_t, n_z, n_eta, n_xi = ds["u"].shape
time_index = 0
depth_index = n_z-1

# location of depth transect in "ROMS_calc_z_velocity_errors.py"
eta_index, xi_index = 29, 8

gridded_result, u_trim_roms, v_trim_roms = roms_grid_velocity_averages(
    roms_file, 
    time_index = 0,
    depth_index = depth_index, # sfc
    gridded_depth = 0.25
)

# --------------
#  PLOT RESULTS
# --------------

for grid_loc in ["u", "v", "rho", "psi"]:

    map_roms_error_with_vectors(
        gridded_result[grid_loc],
        time_idx = 0,
        grid_loc = grid_loc
    )

    plot_coord_averaging_error(
        roms_file,
        grid_loc,
        "u",
        u_trim_roms,
        lat_u_roms_dict,
        lon_u_roms_dict
    )

    plot_coord_averaging_error(
        roms_file,
        grid_loc,
        "v",
        v_trim_roms,
        lat_v_roms_dict,
        lon_v_roms_dict
    )

    # create scatter plot of averaged vs. gridded results
    plot_velocity_error_scatter(gridded_result[grid_loc], grid_loc)

    for vel_dir in ["u","v"]:
        # --- Compare grid locations ---
        plot_coord_error_ROMSaveraging_vs_gridded(
            gridded_result, grid_loc, vel_dir
        )
         # --- Map velocity errors ---
        map_roms_error(gridded_result[grid_loc], vel_dir, grid_loc)

