import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
    # add eta_index, xi_index here if a point location on map is useful
    # eta_index, xi_index = 29, 8
    # ax.plot(
    #     lon[eta_index, xi_index], lat[eta_index, xi_index],
    #     "*", color = "red",
    #     label = "location of vertical\ndepth transect",)

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
    ## legend goes with commented out marker
    #plt.legend()
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
    """Plots velocity vectors with ROMS-averaged (green) and Gridded (purple)
    using the results from `roms_tools.roms_grid_velocity_averages()`.

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

    if np.nansum(u_grid[:]) == 0:
        err_msg = (
            "Gridded velocities are all NaN, which can happen if the "
            "gridded_depth used in `roms_tools.roms_grid_velocity_averages()`"
            "is zero."
            )
        raise ValueError(err_msg)

    # Handle the case where time dimension is included
    if u_roms.ndim == 3:
        u_roms, v_roms = u_roms[time_idx], v_roms[time_idx]
        u_grid, v_grid = u_grid[time_idx], v_grid[time_idx]

    # Assign lat/lon coordinates (lon_v only used for v-dimension)
    lon_u = ds_result[f"lon_u_gridded_{grid_loc}"].data
    lat_u = ds_result[f"lat_u_gridded_{grid_loc}"].data
    lon_v = ds_result[f"lon_v_gridded_{grid_loc}"].data

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
        label="Gridded .at()",
    )
    ax.quiverkey(
            q_roms,
            X=0.2,
            Y=0.25,
            U=0.5,
            label="ROMS\n0.5 m/s",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 9},
            zorder = 5
        )
    ax.quiverkey(
            q_gridded,
            X=0.1,
            Y=0.05,
            U=0.5,
            label="Gridded\n0.5 m/s",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 9},
            zorder = 5
        )

    ax.set_title(
        f"Staggering: {grid_loc.upper()} Locations",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

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

def plot_velocity_error_scatter(
    ds_result: xr.Dataset, 
    grid_loc: str
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
            },
        "v" : {
                "averaged": ds_result[f"v_roms_{grid_loc}"].data,
                "gridded": ds_result[f"v_gridded_{grid_loc}"].data,
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
        plt.colorbar(
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
        ax.set_title(rf"{coord_name.capitalize()} Max |$\Delta$| = {max_error:4.2e}")
        ax.set_xlabel(f"Averaged {vel_dir}-grid {coord_name}")
        ax.set_ylabel(f"ROMS {grid_loc} grid {coord_name}")

    # tie a bow and save figure
    fig.suptitle(
        f"{grid_loc.upper()} grid location evaluation for \n{vel_dir}"
        "-velocity averaging\n"
    )
    plt.tight_layout()
    fig.savefig(
        f"../graphics/horizontal_interp/{vel_dir}_coord_comparison_{grid_loc}"
        ".png",
        dpi=300
    )#, bbox_inches="tight")
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
        ax.set_title(rf"{coord_name.capitalize()} Max |$\Delta$| = {max_error:4.2e}")
        ax.set_xlabel(f"Gridded {grid_loc}-grid {coord_name}")
        ax.set_ylabel(f"ROMS averaged {grid_loc} grid {coord_name}")

    # tie a bow and save figure
    fig.suptitle(f"{grid_loc.upper()} grid location evaluation for \n{vel_dir}-velocity averaging vs. Gridded\n")
    plt.tight_layout()
    fig.savefig(f"../graphics/horizontal_interp/{vel_dir}_coord_comparison-avg_vs_gridded-at_{grid_loc}.png", dpi=300)
    plt.close(fig)
