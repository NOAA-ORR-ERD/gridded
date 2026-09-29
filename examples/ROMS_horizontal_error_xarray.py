import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pdb
from utils import calc_depth_profile
from pathlib import Path
import gridded
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# -----------------------------------------------------------------------
#
# ROMS Arakawa-C grid dimensions 101: 
# - RHO .. [M, L]
# - u .... [M, L-1]
# - v .... [M-1, L]
# - psi .. [M-1, L-1]

# Once u-velocity is manually averaged to rho, v, and psi locations, the original rho, v, and psi grids need to be trimmed accordinginly in order to align with the manually-averaged locations:

# - RHO .. [0:, 1:-1]   -> final dimensions [M, L-2]
# - v .... [0:-1, 1:-1] -> final dimensions [M-1, L-2]
# - psi .. [0:, 0:]     -> final dimensions [M-1, L-1]

# Similarly with the v-velocity
# - RHO .. [1:-1, 0:]   -> final dimensions [M-2, L]
# - u .... [1:-1,0:]    -> final dimensions [M-2, L]
# - psi .. [0:,0:]      -> final dimensions [M-1, M-1]

# -----------------------------------------------------------------------

# location of depth transect in "ROMS_calc_z_velocity_errors.py"
eta_index, xi_index = 29, 8

def avg_xi(arr):
    """Average values along the longitude/xi axis."""
    return 0.5 * (arr[:, :-1] + arr[:, 1:])

def avg_eta(arr):
    """Average values along the latitude/eta axis."""
    return 0.5 * (arr[:-1, :] + arr[1:, :])

def avg_4pt(arr):
    """4-point average across 2x2 cells, resulting in a diagonal shift."""
    return 0.25 * (
        arr[:-1, :-1] + arr[1:, :-1] + 
        arr[:-1, 1:]  + arr[1:, 1:]
    )
    
def map_roms_error(ds_result: xr.Dataset, vel_dir: str, grid_loc: str) -> None:
    """
    Plot and save a map of absolute velocity errors.
    
    Parameters
    ----------
    ds_result : xr.Dataset
        Dataset containing latitude, longitude, and error variables.
    vel_dir : str
        Velocity direction identifier (e.g., 'u' or 'v').
    grid_loc : str
        Grid point location identifier (e.g., 'rho', 'u', 'v', 'psi').
    """
    # assign local variables
    lon = ds_result[f"lon_{vel_dir}_gridded_{grid_loc}"].data
    lat = ds_result[f"lat_{vel_dir}_gridded_{grid_loc}"].data
    error = ds_result[f"{vel_dir}_error_{grid_loc}"].data

    # plot map of error
    fig, ax = plt.subplots(figsize=(8, 6))#, constrained_layout=True)
    pcm = ax.pcolormesh(lon,lat,error, vmin=0, vmax=0.1)
    ax.plot(
        lon[eta_index, xi_index], lat[eta_index, xi_index],
        "*", color = "red",
        label = "location of vertical\ndepth transect",)

    # add contour where errors >= 0.5
    if np.any(error >= 0.1):
        contour = ax.contour(
            lon,
            lat,
            error,
            levels=[0.5],
            colors="white",
            linewidths=2,
            zorder = 5
        )
        ax.clabel(contour, inline=True, fmt="0.5 m/s", fontsize=9)

    # set title and axes information
    ax.set_title(
        f"ROMS v. Gridded \n"
        f"{vel_dir}-velocity errors at {grid_loc.upper()} locations"
    )
    plt.legend()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.tick_params(axis='x', labelrotation=45)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("absolute error (m/s)")
    plt.savefig(f"../graphics/horizontal_interp/ROMS_sfc_{vel_dir}_velocity_at_{grid_loc}_points.png", dpi=300)

def map_roms_error_with_vectors(
    ds_result: xr.Dataset, 
    grid_loc: str,
    time_idx: int = 0,
    step: int = 2,
    scale: float = 10.0
) -> None:
    """Plots velocity vectors with ROMS-averaged (green) and Gridded (purple).

    Parameters
    ----------
    gridded_result : dict
        Dictionary of xarray Datasets keyed by ['u', 'v', 'rho', 'psi'].
    time_idx : int
        Time step index to plot if datasets contain a time dimension.
    step : int
        Subsampling step size for spatial vectors (e.g., 2 = plot every 2nd vector).
    scale : float
        Quiver scaling factor (larger value = smaller arrow length).
    """
    # Assign velocities
    u_roms = ds_result[f"u_roms_{grid_loc}"].data
    v_roms = ds_result[f"v_roms_{grid_loc}"].data
    u_grid = ds_result[f"u_gridded_{grid_loc}"].data
    v_grid = ds_result[f"v_gridded_{grid_loc}"].data
    
    print("gridded check: ", np.nansum(u_grid[:]), np.nansum(v_grid[:]))
    
    # Handle the case where time dimension is included
    if u_roms.ndim == 3:
        u_roms, v_roms = u_roms[time_idx], v_roms[time_idx]
        u_grid, v_grid = u_grid[time_idx], v_grid[time_idx]

    # Assign lat/lon coordinates
    lon_u = ds_result[f"lon_u_gridded_{grid_loc}"].data
    lat_u = ds_result[f"lat_u_gridded_{grid_loc}"].data
    lon_v = ds_result[f"lon_v_gridded_{grid_loc}"].data
    lat_v = ds_result[f"lat_v_gridded_{grid_loc}"].data

    # If u and v shapes match (e.g. psi grid), use lon_u directly
    if lon_u.shape == lon_v.shape:
        lon_q, lat_q = lon_u, lat_u
        u_r_sub, v_r_sub = u_roms, v_roms
        u_g_sub, v_g_sub = u_grid, v_grid
    else:
        # Crop to matching interior dimensions if u and v shapes differ 
        # (e.g., on u or v grid)
        min_eta = min(u_roms.shape[0], v_roms.shape[0])
        min_xi = min(u_roms.shape[1], v_roms.shape[1])

        lon_q = lon_u[:min_eta, :min_xi]
        lat_q = lat_u[:min_eta, :min_xi]

        u_r_sub, v_r_sub = u_roms[:min_eta, :min_xi], v_roms[:min_eta, :min_xi]
        u_g_sub, v_g_sub = u_grid[:min_eta, :min_xi], v_grid[:min_eta, :min_xi]

    
    # Apply spatial subsampling (step)
    sub = (slice(None, None, step), slice(None, None, step))
    plot_lon = lon_q[sub]
    plot_lat = lat_q[sub]

    fig, ax = plt.subplots(
        figsize=(10, 4), 
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=100
    )

    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=3)
    ax.add_feature(cfeature.BORDERS, linestyle=":", zorder=3)
    # Green ROMS vectors
    q_roms = ax.quiver(
        plot_lon, plot_lat,
        u_r_sub[sub],
        v_r_sub[sub],
        color="green",
        scale=scale,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        #width=0.004,
        label="ROMS-averaged",
    )
    # Purple Gridded vectors
    q_gridded = ax.quiver(
        plot_lon, plot_lat,
        u_g_sub[sub],
        v_g_sub[sub],
        color="purple",
        scale=scale,
        alpha=0.8,
        transform=ccrs.PlateCarree(),
        #width=0.004,
        label="Gridded .at()",
    )
    ax.quiverkey(
            q_roms,
            X=0.85,
            Y=0.05,
            U=0.5,
            label="0.5 m/s",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 9},
        )

    ax.set_title(
        f"Staggering: {grid_loc.upper()} Locations", 
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    
    # # Add gridlines and axis labels
    # gl = ax.gridlines(
    #     crs=ccrs.PlateCarree(), 
    #     draw_labels=True, 
    #     linewidth=0.5, 
    #     color="gray", 
    #     alpha=0.5, 
    #     linestyle="--"
    # )
    # gl.top_labels = False
    # gl.right_labels = False

    fig.suptitle(
        f"{grid_loc.upper()} surface velocity comparison (t={time_idx})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
     )
    plt.tight_layout()

    # Save output
    out_path = (
        Path("../graphics/horizontal_interp") / 
        f"quiver_comparison_{grid_loc}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    

def plot_velocity_error_scatter(ds_result: xr.Dataset, grid_loc: str
) -> None:
    """Scatter plot comparing manually averaged velocity values to gridded .at()
    results of velocities at the given locations.  

    Parameters
    ----------
    ds_result : xr.Dataset
        Dataset containing latitude, longitude, and error variables.
    grid_loc : str
        Grid point location identifier (e.g., 'rho', 'u', 'v', 'psi').

    """
    # assign local variables
    velocities = {
        "u" : {
                "averaged": ds_result[f"u_roms_{grid_loc}"].data,
                "gridded": ds_result[f"u_gridded_{grid_loc}"].data,
                "longitude": ds_result[f"lon_u_gridded_{grid_loc}"].data
            },
        "v" : {
                "averaged": ds_result[f"v_roms_{grid_loc}"].data,
                "gridded": ds_result[f"v_gridded_{grid_loc}"].data,
                "longitude": ds_result[f"lon_v_gridded_{grid_loc}"].data
            }

    }
        
    # setup figure
    fig, axs = plt.subplots(1,2,figsize=(8,4))

    # loop through the arakawa-c grid locations and plot ROMS grid
    # values vs. the estimated locations from averaging u- or v-coordinates
    for ax, (velocity, result) in zip(axs, velocities.items()):
        # assign values
        avg_vals = result["averaged"].ravel()
        gridded_vals = result["gridded"].ravel()
        longitude = result["longitude"].ravel()
        error = np.abs(avg_vals - gridded_vals)
        max_error = np.nanmax(error)
        # plot 1:1 line
        sc = ax.scatter(
            avg_vals, gridded_vals, 
            c = error, vmin=0, vmax=0.5
        )
        # add labels
        ax.set_title(
            f"{velocity}-velocity absolute error\n "
            rf"Max |${velocity}_{{avg}} - {velocity}_{{gridded}}$| " 
            f"= {max_error:4.4f} (m/s)"
        )
        ax.set_xlabel(f"Averaged {velocity}-velocity ")
        ax.set_ylabel(f"Gridded.at() {velocity}-velocity ")
        cbar = plt.colorbar(
            sc, label = rf"|${velocity}_{{avg}} - {velocity}_{{gridded}}$| (m/s)",
        )

    # tie a bow and save figure
    fig.suptitle(f"{grid_loc.upper()} grid location surface level velocity error evaluation for t=0")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6)
    fig.savefig(
        f"../graphics/horizontal_interp/{grid_loc.upper()}_location_velocity_error_scatter.png", 
        dpi=300
    )#, bbox_inches="tight")
    plt.close(fig)
    
def plot_coord_averaging_error(
    roms_file: Path | str,
    grid_loc: str,
    vel_dir: str,
    trim_dict: dict,
    avg_lat_dict: dict,
    avg_lon_dict: dict,
) -> None:
    """Scatter plot comparing manually averaged latitude values from either u- or v-velocity
    grid coordinates against true ROMS coordinate value for rho, psi, and either v or u.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing dataset coordinate variables.
    grid_loc : str
        Grid point location identifier (e.g., 'rho', 'u', 'v').
    vel_dir : str
        Velocity direction identifier ('u' or 'v').
    trim_dict : dict
        Dictionary mapping grid location keys to trimming slices.
    avg_lat_dict : dict
        Dictionary mapping grid location keys to averaged latitude DataArrays.
    avg_lon_dict : dict
        Dictionary mapping grid location keys to averaged longitude DataArrays.
    """

    # load ROMS file into data array
    ds = xr.load_dataset(Path(roms_file))
    
    # create dictionary with information for both coordinate axes
    indices = trim_dict.get(grid_loc, None)
    coords = {
        "latitude": {
            "roms": ds[f"lat_{grid_loc}"][*indices] if indices else ds[f"lat_{grid_loc}"],
            "avg": avg_lat_dict[grid_loc],
        },
        "longitude": {
            "roms": ds[f"lon_{grid_loc}"][*indices] if indices else ds[f"lon_{grid_loc}"],
            "avg": avg_lon_dict[grid_loc],
        },
    }

    # setup figure
    fig, axs = plt.subplots(1,2,figsize=(8,4))

    # loop through the arakawa-c grid locations and plot ROMS grid
    # values vs. the estimated locations from averaging u- or v-coordinates
    for ax, (coord_name, values) in zip(axs, coords.items()):
        avg_vals = values["avg"].data.ravel()
        roms_vals = values["roms"].data.ravel()
        max_error = np.max(np.abs(avg_vals - roms_vals))
        # plot 1:1 line
        ax.scatter(avg_vals, roms_vals)
        # add labels
        ax.set_title(f"{coord_name.capitalize()} Max |$\Delta$| = {max_error:4.2e}")
        ax.set_xlabel(f"Averaged {vel_dir}-grid {coord_name}")
        ax.set_ylabel(f"ROMS {grid_loc} grid {coord_name}")

    # tie a bow and save figure
    fig.suptitle(f"{grid_loc.upper()} grid location evaluation for \n{vel_dir}-velocity averaging\n")
    plt.tight_layout()
    fig.savefig(f"../graphics/horizontal_interp/{vel_dir}_coord_comparison_{grid_loc}.png", dpi=300)#, bbox_inches="tight")
    plt.close(fig)

def plot_coord_error_ROMSaveraging_vs_gridded(
    gridded_result: dict,
    grid_loc: str,
    vel_dir: str,
) -> None:
    """This function plots averaged ROMS grid locations against gridded grid locations
    for all four arakawa-c grids to ensure that the velocity averaging of ROMS velocity values
    is working by ensuring that the averaging is to the right location and that any error in the 
    velocity comparison is in a true velocity difference rather than a location difference. 

    Parameters
    ----------
    gridded_result : dict
        Dictionary containing datasets/dictionaries keyed by grid location.
    grid_loc : str
        Grid point location identifier (e.g., 'rho', 'u', 'v').
    vel_dir : str
        Velocity direction identifier ('u' or 'v').
    """
    # extract dataset/dictionary for the given grid location
    grid_data = gridded_result[grid_loc]

    # create dictionary with information for both coordinate axes
    coords = {
        "latitude": {
            "roms_avg": grid_data[f"lat_{vel_dir}_roms_avg_{grid_loc}"],
            "gridded": grid_data[f"lat_{vel_dir}_gridded_{grid_loc}"],
        },
        "longitude": {
            "roms_avg": grid_data[f"lon_{vel_dir}_roms_avg_{grid_loc}"],
            "gridded": grid_data[f"lon_{vel_dir}_gridded_{grid_loc}"],
        },
    }

    # setup figure
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # loop through coordinates and plot gridded values vs. ROMS averaged locations
    for ax, (coord_name, values) in zip(axs, coords.items()):
        roms_avg_vals = values["roms_avg"].data.ravel()
        gridded_vals = values["gridded"].data.ravel()
        max_error = np.max(np.abs(roms_avg_vals - gridded_vals))

        # plot scatter
        ax.scatter(roms_avg_vals, gridded_vals)

        # add labels
        ax.set_title(f"{coord_name.capitalize()} Max |$\Delta$| = {max_error:4.2e}")
        ax.set_xlabel(f"Gridded {grid_loc}-grid {coord_name}")
        ax.set_ylabel(f"ROMS averaged {grid_loc} grid {coord_name}")

    # tie a bow and save figure
    fig.suptitle(f"{grid_loc.upper()} grid location evaluation for \n{vel_dir}-velocity averaging vs. Gridded\n")
    plt.tight_layout()
    fig.savefig(f"../graphics/horizontal_interp/{vel_dir}_coord_comparison-avg_vs_gridded-at_{grid_loc}.png", dpi=300)
    plt.close(fig)    
    
# --- Configuration & Input Setup ---
roms_file = Path("../input_files/ROMS_wcofs_May2026_3D.nc")
input_file = roms_file
ds = xr.load_dataset(roms_file)
n_t, n_z, n_eta, n_xi = ds["u"].shape
time_index = 0
depth_index = n_z-1

# get rho-location [M, L] dimensions for reference and full u-, v- arrays
M, L = ds["h"].shape
roms_u = ds["u"][time_index, depth_index,...]
roms_v = ds["v"][time_index, depth_index,...] 
lat_u = ds["lat_u"]
lon_u = ds["lon_u"]
lat_v = ds["lat_v"]
lon_v = ds["lon_v"] 

# --- Calculate ROMS u-velocities at u, rho, psi, v locations ---
u_roms_dict = {
    "u":   roms_u,
    "rho": avg_xi(roms_u),  # [M, L-2]
    "psi": avg_eta(roms_u), # [M-1, L-1]...same as psi, no trim needed
    "v":   avg_4pt(roms_u), # [M-1, L-2]
} 
lat_u_roms_dict = {
    "u":   lat_u,
    "rho": avg_xi(lat_u), 
    "psi": avg_eta(lat_u),     
    "v":   avg_4pt(lat_u),
} 
lon_u_roms_dict = {
    "u":   lon_u,
    "rho": avg_xi(lon_u), 
    "psi": avg_eta(lon_u),    
    "v":   avg_4pt(lon_u), 
} 

# define how the roms lat/lon arrays need to be sliced to match the u-averaged locations
# these will apply to Gridded results as well since they are based on ROMS' grid locations
u_trim_roms = {
    "rho": (slice(0, M), slice(1, L - 1)),
    "v":   (slice(0, M - 1), slice(1, L - 1)),
}

# --- Calculate ROMS v-velocities at v, rho, psi, u locations ---
v_roms_dict = {
    "v":   roms_v,
    "rho": avg_eta(roms_v),  
    "psi": avg_xi(roms_v),
    "u":   avg_4pt(roms_v),
}
lat_v_roms_dict = {
    "v":   lat_v,
    "rho": avg_eta(lat_v), 
    "psi": avg_xi(lat_v),     
    "u":   avg_4pt(lat_v),
} 
lon_v_roms_dict = {
    "v":   lon_v,
    "rho": avg_eta(lon_v), 
    "psi": avg_xi(lon_v),    
    "u":   avg_4pt(lon_v), 
} 

# define how the roms lat/lon arrays need to be sliced to match the u-averaged locations
# these will apply to Gridded results as well since they are based on ROMS' grid locations
v_trim_roms = {
    "rho": (slice(1, M - 1), slice(0, L)),
    "u":   (slice(1, M - 1), slice(0, L)),
}


# --- Load gridded variables and trim to align values for error calculation ---
gridded_ds = gridded.Dataset(str(roms_file))
u_var = gridded_ds.variables["u"]
v_var = gridded_ds.variables["v"]

# Create coordinate arrays at u, v, rho, and psi locations for
# Gridded interpoloation
gridded_result = {}
for grid_loc in ["u", "v", "rho", "psi"]:
    full_lats = ds[f"lat_{grid_loc}"].data
    full_lons = ds[f"lon_{grid_loc}"].data
    
    # create coordinate array to use in Gridded interpolation
    loc_coords = np.column_stack(
        (full_lons.ravel(), full_lats.ravel(), -0.5*np.ones(full_lons.size))
    )   
    
    # interpolate velocities, reshape into 2D array and trim down to match locations 
    # of manual interpolation
    
    # -> u-velocities
    u_gridded = u_var.at(
        loc_coords, time=u_var.time.data[time_index]
    ).reshape(full_lons.shape)
    if (grid_loc == "rho") or (grid_loc == "v"):
        u_gridded = u_gridded[*u_trim_roms[grid_loc]]
    # -> v-velocities
    v_gridded = v_var.at(
        loc_coords, time=v_var.time.data[time_index]
    ).reshape(full_lons.shape)
    if grid_loc == "rho" or grid_loc == "u":  
        v_gridded = v_gridded[*v_trim_roms[grid_loc]]
        
    # assign manually averaged values for calculating errors
    u_roms, v_roms =  u_roms_dict[grid_loc].data, v_roms_dict[grid_loc].data   

    # create lat/lon arrays to save
    indices_u = u_trim_roms.get(grid_loc, None)
    lat_u = full_lats[*indices_u] if indices_u else full_lats
    lon_u = full_lons[*indices_u] if indices_u else full_lons
    indices_v = v_trim_roms.get(grid_loc, None)
    lat_v = full_lats[*indices_v] if indices_v else full_lats
    lon_v = full_lons[*indices_v] if indices_v else full_lons

    print(" --- ", grid_loc, " --- ")
    print(f"Shape of manual u-avg: {u_roms.shape}")
    print(f"Shape of gridded u-avg: {u_gridded.shape}")
    print(f"Shape of lat/lon_u: {lat_u.shape}, {lon_u.shape}")
    print(f"Shape of manual v-avg: {v_roms.shape}")   
    print(f"Shape of gridded v-avg: {v_gridded.shape}")
    
    # Creata a DataArray to store values
    gridded_result[grid_loc] = xr.Dataset(
        data_vars={
            f"u_roms_{grid_loc}": (("eta_u", "xi_u"), u_roms),
            f"v_roms_{grid_loc}": (("eta_v", "xi_v"), v_roms),
            f"u_gridded_{grid_loc}": (("eta_u", "xi_u"), u_gridded),
            f"v_gridded_{grid_loc}": (("eta_v", "xi_v"), v_gridded),
            f"u_error_{grid_loc}": (("eta_u", "xi_u"), np.abs(u_gridded - u_roms)),
            f"v_error_{grid_loc}": (("eta_v", "xi_v"), np.abs(v_gridded - v_roms)),
        },
        coords={
            f"lat_v_gridded_{grid_loc}": (("eta_v", "xi_v"), lat_v),
            f"lon_v_gridded_{grid_loc}": (("eta_v", "xi_v"), lon_v),
            f"lat_u_gridded_{grid_loc}": (("eta_u", "xi_u"), lat_u),
            f"lon_u_gridded_{grid_loc}": (("eta_u", "xi_u"), lon_u),
            
            f"lat_v_roms_avg_{grid_loc}": (("eta_v", "xi_v"), lat_v_roms_dict[grid_loc].data),
            f"lon_v_roms_avg_{grid_loc}": (("eta_v", "xi_v"), lon_v_roms_dict[grid_loc].data),
            f"lat_u_roms_avg_{grid_loc}": (("eta_u", "xi_u"), lat_u_roms_dict[grid_loc].data),
            f"lon_u_roms_avg_{grid_loc}": (("eta_u", "xi_u"), lon_u_roms_dict[grid_loc].data),
        },
        attrs={
            "grid_staggering": grid_loc,
            "description": f"grid_staggering at {grid_loc} locations",
        }
    )


for grid_loc in ["u", "v", "rho", "psi"]:
    for vel_dir in ["u","v"]:
        print(
            gridded_result[grid_loc][f"lat_{vel_dir}_gridded_{grid_loc}"].shape
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

  