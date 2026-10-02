from pathlib import Path
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import gridded
from roms_tools import roms_grid_velocity_averages as staggered_avg
import roms_tools as r_tools
import roms_qaqc_plots as r_plot 
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
depth_index = -1#n_z-1

# location of depth transect in "ROMS_calc_z_velocity_errors.py"
eta_index, xi_index = 29, 8

# Average ROMS velocities to staggered grid
gridded_result, u_trim_roms, v_trim_roms = staggered_avg(
    roms_file, 
    time_index = 0,
    depth_index = -1, # sfc
    gridded_depth = 0.25
)

# Calculate velocities and coords at staggered locations using u-,v-values and grids
u_stg, u_lat_stg, u_lon_stg = r_tools.avg_u_to_staggered(
        ds, time_index, depth_index
    )
v_stg, v_lat_stg, v_lon_stg = r_tools.avg_v_to_staggered(
        ds, time_index, depth_index
    )
# --------------
#  PLOT RESULTS
# --------------

for grid_loc in ["u", "v", "rho", "psi"]:

    # Create maps of current speeds and direction
    r_plot.map_roms_error_with_vectors(
        gridded_result[grid_loc],
        time_idx = 0,
        grid_loc = grid_loc
    )

    # Validate method using lat/lon averaging
    # check u-velocity averaging    
    r_plot.plot_coord_averaging_error(
        roms_file,
        grid_loc,
        "u",
        u_trim_roms,
        u_lat_stg,
        u_lon_stg
    )
    # check v-velocity averaging
    r_plot.plot_coord_averaging_error(
        roms_file,
        grid_loc,
        "v",
        v_trim_roms,
        v_lat_stg,
        v_lon_stg
    )

    # create scatter plot of averaged vs. gridded results
    r_plot.plot_velocity_error_scatter(
        gridded_result[grid_loc], grid_loc
    )

    for vel_dir in ["u","v"]:
        # --- Compare grid locations ---
        r_plot.plot_coord_error_ROMSaveraging_vs_gridded(
            gridded_result, grid_loc, vel_dir
        )
         # --- Map velocity errors ---
        r_plot.map_roms_error(gridded_result[grid_loc], vel_dir, grid_loc)

