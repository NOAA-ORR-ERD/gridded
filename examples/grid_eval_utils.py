from pathlib import Path
import gridded
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cf

# --- Utility scripts for averaging Arakawa-C variables ---

def avg_xi(da: xr.DataArray):
    """Average values along the longitude/xi axis.

    Assumes that ``reorder_roms_dims`` was run first to ensure
    order of (time, depth, eta, xi). Accommodates 2D and 3D grids.

    :param da: The DataArray to average along the xi axis.
    :type da: xr.DataArray
    :returns: The averaged DataArray.
    :rtype: xr.DataArray
    """
    # get name of last dimension (e.g. xi_u or xi_v)
    dim_name = da.dims[-1]
    print(f"averaging across {dim_name}")
    # get the slices
    slice1 = da.isel({dim_name: slice(None, -1)})
    slice2 = da.isel({dim_name: slice(1, None)})
    # stack slices and then take the average along fabricated dimension
    # "window", using skipna to preserve non-nan values
    xi_avg = xr.concat(
        [slice1, slice2], 
        dim = "window"
    ).mean(dim = "window", skipna=True)
    
    return xi_avg.data

def avg_eta(da: xr.DataArray):
    """Average values along the latitude/eta axis.

    Assumes that ``reorder_roms_dims`` was run first to ensure
    order of (time, depth, eta, xi). Accommodates 2D and 3D grids.

    :param da: The DataArray to average along the eta axis.
    :type da: xr.DataArray
    :returns: The averaged DataArray.
    :rtype: xr.DataArray
    """
    # get name of second to last dimension (e.g. eta_u or eta_v)
    dim_name = da.dims[-2]
    print(f"averaging across {dim_name}")
    # get the slices
    slice1 = da.isel({dim_name: slice(None, -1)})
    slice2 = da.isel({dim_name: slice(1, None)})
    # stack slices and then take the average along fabricated dimension
    # "window", using skipna to preserve non-nan values
    eta_avg = xr.concat(
        [slice1, slice2], 
        dim = "window"
    ).mean(dim = "window", skipna=True)
    
    return eta_avg.data 

def avg_4pt(da: xr.DataArray):
    """4-point average across 2x2 cells, resulting in a diagonal shift.

    Assumes that ``reorder_roms_dims`` was run first to ensure
    order of (time, depth, eta, xi). Accommodates 2D and 3D grids.

    :param da: The DataArray to average across 2x2 cells.
    :type da: xr.DataArray
    :returns: The 4-point averaged DataArray.
    :rtype: xr.DataArray
    """
    eta_dim_name = da.dims[-2]
    xi_dim_name = da.dims[-1]
    
    # get the slices
    slice1 = da.isel({
        eta_dim_name: slice(None, -1),
        xi_dim_name: slice(None, -1), # arr[..., :-1, :-1]
        
    })
    slice2 = da.isel({
        eta_dim_name: slice(1, None),
        xi_dim_name: slice(None, -1), #arr[..., 1:, :-1]       
    })
    slice3 = da.isel({
        eta_dim_name: slice(None, -1),
        xi_dim_name: slice(1, None), #arr[..., :-1, 1:]   
    })
    slice4 = da.isel({
        eta_dim_name: slice(1, None),
        xi_dim_name: slice(1, None), #arr[..., 1:, 1:]    
    })

    # average results of slices, skipping NaN values. 
    # Would be equivalent to: 
    # 0.25*(
    #    arr[..., :-1, :-1] + 
    #    arr[..., 1:, :-1] + 
    #    arr[..., :-1, 1:] + 
    #    arr[..., 1:, 1:]
    # )
    # if the above skipped NaN values
    four_way_avg = xr.concat(
        [slice1, slice2, slice3, slice4], 
        dim = "window"
    ).mean(dim = "window", skipna=True)
    
    return four_way_avg.data
    
def reorder_roms_dims(da: xr.DataArray):
    """Reorder ROMS DataArray dimensions to (time, depth, eta, xi)."""
    # Identify time dimension
    t_dim = [d for d in da.dims if "time" in d]
    # Identify depth/vertical dimension
    z_dim = [d for d in da.dims if d.startswith("s_") or "depth" in d or "z" in d]
    # Identify eta (y-axis) dimension
    eta_dim = [d for d in da.dims if "eta" in d]
    # Identify xi (x-axis) dimension
    xi_dim = [d for d in da.dims if "xi" in d]
    # Combine existing dimensions in desired order
    target_order = t_dim + z_dim + eta_dim + xi_dim

    return da.transpose(*target_order)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
# --- Pre-processing ROMS output to create the netcdf used for analysis --- 

def create_comparison_netcdf(
    input_file: Path | str, 
    model_name: str | None = "ROMS"
):
    """Creates a NetCDF file with ROMS-averaged velocities, coordinate-averaged
    locations (to compare with actual in order to verify averaging/trimming method),
    and Gridded-interpolated results at u-, v-, rho-, and psi-grid locations. If
    present, wetdry masks are applied to velocities.

    :param input_file: Path and name of ROMS NetCDF file used both for averaging 
        and for location inputs to Gridded.
    :type input_file: Path | str
    :param model_name: Tag-name to use as folder name for saved output as well as 
        in the file name. ROMS is used as a generic default.
    :type model_name: str | None
    
    :returns: None. A NetCDF file is saved to the directory ``../output/[model_name]``.
    :rtype: None
    """    

    # Load ROMS file
    ds = xr.load_dataset(input_file)

    # Apply wetdry masks to velocitities
    variable_list = [*ds]
    M, L = ds["h"].shape
    nt,_,_ = ds["u"].shape
    dg = 0
    wetdrymask = "wetdry_mask_rho" in variable_list
    if wetdrymask:
        # use masks to isolate computational domain and set all other 
        # values to NaN.  xarray's "skipna" is used manage/ignore NaN values
        for velocity in ["u","v"]:
            ds[velocity] = ds[velocity].where(ds[f"wetdry_mask_{velocity}"] == 1)
    
    # --- Use all time steps for surface level ---
    # variables must by xarray.DataArrays for averaging functions
    roms_u_all = ds["u"]  # Shape: (T, M, L-1)
    roms_v_all = ds["v"]  # Shape: (T, M-1, L)
    lat_u = ds["lat_u"]
    lon_u = ds["lon_u"]
    lat_v = ds["lat_v"]
    lon_v = ds["lon_v"]
    
    # --- Assign u-/v-velocities at all Arakawa-C grid locations ---
    # use averages to calculate velocities at v-/u-, rho, and psi locations
    # v-velocity values and averaged-velocity coordinate locations
    u_roms_dict = {
        "u":   roms_u_all.data,
        "rho": avg_xi(roms_u_all),  # [T, M, L-2]
        "psi": avg_eta(roms_u_all),  # [T, M-1, L-1]
        "v":   avg_4pt(roms_u_all),  # [T, M-1, L-2]
    }
    lat_u_roms_dict = {
        "u":   lat_u.data,
        "rho": avg_xi(lat_u), 
        "psi": avg_eta(lat_u),     
        "v":   avg_4pt(lat_u),
    } 
    lon_u_roms_dict = {
        "u":   lon_u.data,
        "rho": avg_xi(lon_u), 
        "psi": avg_eta(lon_u),  
        "v":   avg_4pt(lon_u), 
    }
    # v-velocity values and averaged-velocity coordinate locations
    v_roms_dict = {
        "v":   roms_v_all.data,
        "rho": avg_eta(roms_v_all),  # [T, M-2, L]
        "psi": avg_xi(roms_v_all),  # [T, M-1, L-1]
        "u":   avg_4pt(roms_v_all),  # [T, M-2, L]
    }
    lat_v_roms_dict = {
        "v":   lat_v.data,
        "rho": avg_eta(lat_v), 
        "psi": avg_xi(lat_v),     
        "u":   avg_4pt(lat_v),
    } 
    lon_v_roms_dict = {
        "v":   lon_v.data,
        "rho": avg_eta(lon_v), 
        "psi": avg_xi(lon_v),    
        "u":   avg_4pt(lon_v), 
    }
    
    # --- Define trimming slices  ---
    # used to trim Gridded output so dimensions and locations 
    # match the averaged grids calculated above.  PSI locations
    # don't need trimming b/c they are already interior locations. 
    u_trim_roms = {
        "rho": (slice(0, M), slice(1, L - 1)),
        "v":   (slice(0, M - 1), slice(1, L - 1)),
    }
    
    v_trim_roms = {
        "rho": (slice(1, M - 1), slice(0, L)),
        "u":   (slice(1, M - 1), slice(0, L)),
    }
    # --- Process Grid Locations Across All Time Steps ---
    # load gridded results
    gridded_ds = gridded.Dataset(str(input_file))
    u_var = gridded_ds.variables["u"]
    v_var = gridded_ds.variables["v"]
    
    # mask velocities
    u_var.data = np.where(ds["wetdry_mask_u"].data==1, u_var.data, np.nan)
    v_var.data = np.where(ds["wetdry_mask_v"].data==1, v_var.data, np.nan)
    
    # Use Gridded to estimate velocities at all four grid locations
    gridded_result = {}
    for grid_loc in ["u", "v", "rho", "psi"]:
    
        print(f"---> Interpolating {grid_loc.upper()} grid using Gridded\n")
    
        # Access previously created dictionary with ROMS results
        u_roms_all = u_roms_dict[grid_loc]
        v_roms_all = v_roms_dict[grid_loc]
        
        # Sub-sample full domains to align with the manually-averaged 
        # locations, where needed.  u/v_trim_roms only has `rho` and 
        #`u`/`v` locations.
        indices_u = u_trim_roms.get(grid_loc, None) # re-name "slices_u"? 
        indices_v = v_trim_roms.get(grid_loc, None)
    
        u_gridded_all = np.full((nt, *u_roms_all[0,...].shape), np.nan)
        v_gridded_all = np.full((nt, *v_roms_all[0,...].shape), np.nan)
        
        # Loop through time and create time series of Gridded results
        for t_idx in range(nt):
           # print(f"{t_idx} of {nt}\n")
            
            # Mask out all locations on land (by setting to NaN)
            masked_lats = ds[f"lat_{grid_loc}"].where(
                ds[f"wetdry_mask_{grid_loc}"][t_idx,...] == 1
            ).data
            masked_lons = ds[f"lon_{grid_loc}"].where(
                ds[f"wetdry_mask_{grid_loc}"][t_idx,...] == 1
            ).data
    
            # Flatten coordinate arrays for Gridded (with z=0 for surface)
            loc_coords = np.column_stack(
                (masked_lons.ravel(), masked_lats.ravel(), 
                 np.full(masked_lons.size, dg))
            )
    
            # Get the time, formatted in a way that Gridded will grok
            t_val = u_var.time.data[t_idx]
     
            # Interpolate u-velocity
            u_sampled_t = u_var.at(
                loc_coords, time=t_val
            ).reshape(masked_lons.shape)          
            
            # Interpolate v-velocity
            v_sampled_t = v_var.at(
                loc_coords, time=t_val
            ).reshape(masked_lons.shape)
      
            # return to using the un-masked lat/lon values 
            # for u-velocities
            if indices_u is not None:
                 temp_u = u_sampled_t[*indices_u]
                 lat_u_sub = ds[f"lat_{grid_loc}"][*indices_u].data
                 lon_u_sub = ds[f"lon_{grid_loc}"][*indices_u].data
            else:
                 temp_u = u_sampled_t
                 lat_u_sub = ds[f"lat_{grid_loc}"].data
                 lon_u_sub = ds[f"lon_{grid_loc}"].data
                
            # now for v-velocities
            if indices_v is not None:
                 temp_v = v_sampled_t[*indices_v]
                 lat_v_sub = ds[f"lat_{grid_loc}"][*indices_v].data
                 lon_v_sub = ds[f"lon_{grid_loc}"][*indices_v].data
            else:
                 temp_v = v_sampled_t
                 lat_v_sub = ds[f"lat_{grid_loc}"].data
                 lon_v_sub = ds[f"lon_{grid_loc}"].data
    
            # Lock in the interpolated and trimmed (where needed) results
            u_gridded_all[t_idx, ...] = temp_u
            v_gridded_all[t_idx, ...] = temp_v
    
            # trim wet-dry masks so we can use them in water-location selection
            mask_u = ds["wetdry_mask_u"].values[
                :nt, : u_gridded_all.shape[1], : u_gridded_all.shape[2]
                ]
            mask_v = ds["wetdry_mask_v"].values[
                :nt, : v_gridded_all.shape[1], : v_gridded_all.shape[2]
                ]
            
            # apply the time-varying wet/dry mask to remove problematic locations
            u_gridded_all[mask_u != 1] = np.nan
            v_gridded_all[mask_v != 1] = np.nan
    
        # Store 3D time-series arrays in xarray Dataset
        gridded_result[grid_loc] = xr.Dataset(
            data_vars={
                f"u_roms_{grid_loc}": (
                    ("time", "eta_u", "xi_u"), u_roms_all
                ),
                f"v_roms_{grid_loc}": (
                    ("time", "eta_v", "xi_v"), v_roms_all
                ),
                f"u_gridded_{grid_loc}": (
                    ("time", "eta_u", "xi_u"), u_gridded_all
                ),
                f"v_gridded_{grid_loc}": (
                    ("time", "eta_v", "xi_v"), v_gridded_all
                ),
                f"u_error_{grid_loc}": (
                    ("time", "eta_u", "xi_u"),
                    np.abs(u_gridded_all - u_roms_all),
                ),
                f"v_error_{grid_loc}": (
                    ("time", "eta_v", "xi_v"),
                    np.abs(v_gridded_all - v_roms_all),
                ),
            },
            coords={
                "time": ds.ocean_time.data[:nt],
                f"lat_u_gridded_{grid_loc}": (("eta_u", "xi_u"), lat_u_sub),
                f"lon_u_gridded_{grid_loc}": (("eta_u", "xi_u"), lon_u_sub),
                f"lat_v_gridded_{grid_loc}": (("eta_v", "xi_v"), lat_v_sub),
                f"lon_v_gridded_{grid_loc}": (("eta_v", "xi_v"), lon_v_sub),
                f"lat_v_roms_avg_{grid_loc}": (
                    ("eta_v", "xi_v"), lat_v_roms_dict[grid_loc].data
                ),
                f"lon_v_roms_avg_{grid_loc}": (
                    ("eta_v", "xi_v"), lon_v_roms_dict[grid_loc].data
                ),
                f"lat_u_roms_avg_{grid_loc}": (
                    ("eta_u", "xi_u"), lat_u_roms_dict[grid_loc].data
                ),
                f"lon_u_roms_avg_{grid_loc}": (
                    ("eta_u", "xi_u"), lon_u_roms_dict[grid_loc].data
                ),
            },
            attrs={
                "grid_location": grid_loc,
                "description": (
                    f"Velocities evaluated across {nt} time steps at "
                    f"{grid_loc} locations",
                )
            },
        )
    
        # save calculations to netcdf
        output_file = Path(
             f"../output/{model_name}/{model_name}_{grid_loc}_"
            "grid_location_wetdry.nc"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        gridded_result[grid_loc].to_netcdf(output_file)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- Plotting functions ---
# plot_velocity_error_scatter   
    # 1:1 scatter plot comparing ROMS-averaged output to Gridded-interpolations.
# plot_coord_error
    # 1:1 scatter plot comparing averaged coordinates to actual coordinates
# map_roms_error: 
    # Straight up maps of the ROMS vs. Gridded error calc.     
# map_roms_error_with_vectors: 
    # There are two of these funtions, and I'm not sure
    # which is the right one to keep (if there is any difference?).  For now, 
    # I just commented one out. This script super-imposes the quiver plots from the 
    # ROMS-averaged velocity components and the Gridded-interpolated values.    
# map_curvi: 
    #plot up metric of grid's curvilinearity   
# map_zonal_cell_width: 
    # plot up estimate of cell width. Assumes eta cell width 
    # is equivalent to lat cell width...which is a flawed assumption but the 
    # best we have for a quick, ballpark estimate

    
# --- Scatter plot of ROMS-averaged velocities vs. Gridded velocities ---
def plot_velocity_error_scatter(
    ds_result: xr.Dataset, grid_loc: str, num_times: int, model_name: str | None = None
) -> None:
    """Scatter plot comparing manually averaged velocity values to gridded .at()
    results, including all output time steps.

    Parameters
    ----------
    ds_result : xr.Dataset
        Dataset containing latitude, longitude, and error variables.
    grid_loc : str
        Grid point location identifier (e.g., 'rho', 'u', 'v', 'psi').
    num_times: int
        Number of time steps include in graphic. 
    """
    velocities = {
        "u": {
            "averaged": ds_result[f"u_roms_{grid_loc}"].data,
            "gridded": ds_result[f"u_gridded_{grid_loc}"].data,
        },
        "v": {
            "averaged": ds_result[f"v_roms_{grid_loc}"].data,
            "gridded": ds_result[f"v_gridded_{grid_loc}"].data,
        },
    }
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (velocity, result) in zip(axs, velocities.items()):
        avg_vals = result["averaged"].ravel()
        gridded_vals = result["gridded"].ravel()
        error = np.abs(avg_vals - gridded_vals)
        max_error = np.nanmax(error)
        median_error = np.nanmedian(error)

        sc = ax.scatter(avg_vals, gridded_vals, c=error, vmin=0, vmax=0.5, s=10)

        # Plot 1:1 reference line
        min_val = min(np.nanmin(avg_vals), np.nanmin(gridded_vals))
        max_val = max(np.nanmax(avg_vals), np.nanmax(gridded_vals))
        ax.plot(
            [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="1:1"
        )

        ax.set_title(
            f"{velocity}-velocity error (all {num_times} output times)\n"
            fr"Error = |{velocity}$_{{avg}}$ - {velocity}$_{{gridded}}$| "
            f"Max Error = {max_error:4.2f} m/s\n"
            f"Median Error = {median_error:4.2f} m/s"
        )
        ax.set_xlabel(f"Averaged ROMS {velocity}-velocity (m/s)")
        ax.set_ylabel(f"Gridded .at() {velocity}-velocity (m/s)")
        plt.colorbar(sc, ax=ax, label=fr"|{velocity}$_{{avg}}$ - {velocity}$_{{gridded}}$| (m/s)")

    fig.suptitle(
        (f"{grid_loc.upper()} Location Velocity Error Evaluation "
         f"(All Time Steps)\n{model_name if model_name is not None else ''}"),
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()

    if model_name is not None: 
        out_dir = Path(f"../graphics/horizontal_interp/{model_name}")
    else:
        out_dir = Path(f"../graphics/horizontal_interp/ROMS")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / f"{grid_loc.upper()}_location_velocity_error_scatter_all_times.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# --- Scatter plot of ROMS-averaged coord locations vs. Gridded locations ---
def plot_coord_error(
    gridded_result: dict,
    grid_loc: str,
    vel_dir: str,
    model_name: str | None = None
) -> None:
    """This function plots averaged ROMS grid locations against gridded grid locations
    for all four arakawa-c grids to ensure that the velocity averaging of ROMS velocity values
    is working by ensuring that the averaging is to the right location and that any error in the 
    velocity comparison is in a true velocity difference rather than a location difference. 

    Parameters
    ----------
    gridded_result : dict
        Dictionary containing Gridded and ROMS-averaged results.
    grid_loc : str
        Grid point location identifier (e.g., 'rho', 'u', 'v').
    vel_dir : str
        Velocity direction identifier ('u' or 'v').
    """

    # create dictionary with information for both coordinate axes
    coords = {
        "latitude": {
            "roms_avg": gridded_result[f"lat_{vel_dir}_roms_avg_{grid_loc}"],
            "gridded": gridded_result[f"lat_{vel_dir}_gridded_{grid_loc}"],
        },
        "longitude": {
            "roms_avg": gridded_result[f"lon_{vel_dir}_roms_avg_{grid_loc}"],
            "gridded": gridded_result[f"lon_{vel_dir}_gridded_{grid_loc}"],
        },
    }

    # setup figure
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # loop through coordinates and plot gridded values vs. ROMS averaged locations
    for ax, (coord_name, values) in zip(axs, coords.items()):
        roms_avg_vals = values["roms_avg"].data.ravel()
        gridded_vals = values["gridded"].data.ravel()
        max_error = np.nanmax(np.abs(roms_avg_vals - gridded_vals))

        # plot scatter
        ax.scatter(roms_avg_vals, gridded_vals)

        # add labels
        ax.set_title(
            fr"{coord_name.capitalize()} Max |$\Delta$| = {max_error:4.2e}"
        )
        ax.set_xlabel(f"Gridded {grid_loc}-grid {coord_name}")
        ax.set_ylabel(f"ROMS averaged {grid_loc} grid {coord_name}")

    # tie a bow and save figure
    fig.suptitle(
        f"{grid_loc.upper()} grid location evaluation for \n{vel_dir}"
        f"-velocity averaging vs. Gridded\n"
        f"{model_name if model_name is not None else ''}"
    )
    plt.tight_layout()
    if model_name is not None:
        output_file = Path(
            f"../graphics/horizontal_interp/{model_name}/{vel_dir}"
            f"_coord-avg_vs_gridded-at_{grid_loc}_{model_name}.png"
        )
    else:
        output_file = Path(
            f"../graphics/horizontal_interp/{vel_dir}_coord"
            f"-avg_vs_gridded-at_{grid_loc}_{model_name}.png"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, bbox_inches="tight", dpi = 300)
    plt.close(fig)

def map_roms_error(
    ds_result: xr.Dataset, 
    vel_dir: str, 
    grid_loc: str,
    model_name: str | None = None
) -> None:
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
    if ds_result[f"{vel_dir}_error_{grid_loc}"].ndim < 3: 
        error = ds_result[f"{vel_dir}_error_{grid_loc}"].data
    else:
        error = ds_result[f"{vel_dir}_error_{grid_loc}"].max(dim = "time").data

    # plot map of error
    fig, ax = plt.subplots(figsize=(8, 6))#, constrained_layout=True)
    pcm = ax.pcolormesh(
        lon, lat, error, vmin=0, vmax=.5
    )

    # add contour where errors >= 0.5
    if np.any(error >= 0.25):
        contour = ax.contour(
            lon,
            lat,
            error,
            levels=[0.5],
            colors="orange",
            linewidths=2,
            zorder = 5,
            label = "> 0.25 m/s"
        )
        ax.clabel(contour, inline=True, fmt="0.25 m/s", fontsize=9)

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
    if model_name is not None:
        output_file = Path(
            f"../graphics/horizontal_interp/{model_name}/"
            f"sfc_{vel_dir}_velocity_at_{grid_loc}_{model_name}.png"
        )
    else:
        output_file = Path(
            f"../graphics/horizontal_interp/"
            f"sfc_{vel_dir}_velocity_at_{grid_loc}_ROMS.png"
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, bbox_inches="tight", dpi=300)


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

    ax.add_feature(cf.LAND, facecolor="lightgray", zorder=2)
    ax.add_feature(cf.COASTLINE, linewidth=1, zorder=3)
    ax.add_feature(cf.BORDERS, linestyle=":", zorder=3)
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
        (f"{grid_loc.upper()} surface velocity comparison "
         f"(t={time_idx})\n{model_name if model_name is not None else ''}"),
        fontsize=14,
        fontweight="bold",
        y=0.98,
     )
    plt.tight_layout()

    # Save output
    if model_name is not None:
        out_path = (
            Path(f"../graphics/horizontal_interp/{model_name}") / 
            f"quiver_comparison_{grid_loc}_{model_name}.png"
        )
    else:
        out_path = (
            Path("../graphics/horizontal_interp") / 
            f"quiver_comparison_{grid_loc}_ROMS.png"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    
# def map_roms_error_with_vectors(
#     ds_result: xr.Dataset,
#     grid_loc: str,
#     time_idx: int = 0,
#     step: int = 2,
#     scale: float = 10.0,
#     model_name: str | None = None,
#     bounds: tuple | None = None
# ) -> None:
#     """Plots velocity vectors with ROMS-averaged (green) and Gridded (purple)
#     using the results from `roms_tools.roms_grid_velocity_averages()`.

#     Parameters
#     ----------
#     gridded_result : dict
#         Dictionary of xarray Datasets keyed by ['u', 'v', 'rho', 'psi'].
#     time_idx : int
#         Time step index to plot if datasets contain a time dimension.
#     step : int
#         Subsampling step size for spatial vectors (e.g., 2 = plot every 2nd vector).
#     scale : float
#         Quiver scaling factor (larger value = smaller arrow length).
#     """
#     # Assign velocities
#     u_roms = ds_result[f"u_roms_{grid_loc}"].data
#     v_roms = ds_result[f"v_roms_{grid_loc}"].data
#     u_grid = ds_result[f"u_gridded_{grid_loc}"].data
#     v_grid = ds_result[f"v_gridded_{grid_loc}"].data

#     if np.nansum(u_grid[:]) == 0:
#         err_msg(
#             "Gridded velocities are all NaN, which can happen if the "
#             "gridded_depth used in `roms_tools.roms_grid_velocity_averages()`"
#             "is zero."
#             )
#         raise ValueError(err_msg)

#     # Handle the case where time dimension is included
#     if u_roms.ndim == 3:
#         u_roms, v_roms = u_roms[time_idx], v_roms[time_idx]
#         u_grid, v_grid = u_grid[time_idx], v_grid[time_idx]

#     # Assign lat/lon coordinates (lon_v only used for v-dimension)
#     lon_u = ds_result[f"lon_u_gridded_{grid_loc}"].data
#     lat_u = ds_result[f"lat_u_gridded_{grid_loc}"].data
#     lon_v = ds_result[f"lon_v_gridded_{grid_loc}"].data

#     # If u and v shapes match (e.g. psi grid), use lon_u directly
#     if lon_u.shape == lon_v.shape:
#         lon_q, lat_q = lon_u, lat_u
#         u_r_sub, v_r_sub = u_roms, v_roms
#         u_g_sub, v_g_sub = u_grid, v_grid
#     else:
#         # Crop to matching interior dimensions if u and v shapes differ
#         # (e.g., on u or v grid)
#         min_eta = min(u_roms.shape[0], v_roms.shape[0])
#         min_xi = min(u_roms.shape[1], v_roms.shape[1])

#         lon_q = lon_u[:min_eta, :min_xi]
#         lat_q = lat_u[:min_eta, :min_xi]

#         # roms
#         u_r_sub = u_roms[:min_eta, :min_xi]
#         v_r_sub = v_roms[:min_eta, :min_xi]
#         # gridded
#         u_g_sub = u_grid[:min_eta, :min_xi]
#         v_g_sub = v_grid[:min_eta, :min_xi]

#     # Apply spatial subsampling (step)
#     sub = (slice(None, None, step), slice(None, None, step))
#     plot_lon = lon_q[sub]
#     plot_lat = lat_q[sub]

#     fig, ax = plt.subplots(
#         figsize=(10, 4),
#         subplot_kw={"projection": ccrs.PlateCarree()},
#         dpi=100
#     )

#     ax.add_feature(cf.LAND, facecolor="lightgray", zorder=2)
#     ax.add_feature(cf.COASTLINE, linewidth=1, zorder=3)
#     ax.add_feature(cf.BORDERS, linestyle=":", zorder=3)
#     # Green ROMS vectors
#     q_roms = ax.quiver(
#         plot_lon, plot_lat,
#         u_r_sub[sub],
#         v_r_sub[sub],
#         color="green",
#         scale=scale,
#         alpha=0.8,
#         transform=ccrs.PlateCarree(),
#         label="ROMS-averaged",
#     )
#     # Purple Gridded vectors
#     q_gridded = ax.quiver(
#         plot_lon, plot_lat,
#         u_g_sub[sub],
#         v_g_sub[sub],
#         color="purple",
#         scale=scale,
#         alpha=0.8,
#         transform=ccrs.PlateCarree(),
#         label="Gridded .at()",
#     )
#     ax.quiverkey(
#             q_roms,
#             X=0.8,
#             Y=0.05,
#             U=0.1,
#             label="0.1 m/s",
#             labelpos="E",
#             coordinates="axes",
#             fontproperties={"size": 9},
#             zorder = 10
#         )
#     ax.quiverkey(
#             q_gridded,
#             X=0.8,
#             Y=0.1,
#             U=0.1,
#             label="0.1 m/s",
#             labelpos="E",
#             coordinates="axes",
#             fontproperties={"size": 9},
#             zorder = 10
#         )

#     if bounds is not None:
#         ax.axis(bounds)

#     ax.set_title(
#         f"Staggering: {grid_loc.upper()} Locations",
#         fontsize=12, fontweight="bold"
#     )
#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")
#     ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

#     fig.suptitle(
#         (f"{grid_loc.upper()} surface velocity comparison (t={time_idx})"
#         f"\n{model_name if model_name is not None else ''}"),
#         fontsize=14,
#         fontweight="bold",
#         y=0.98,
#      )
#     plt.tight_layout()

#     # Save output
#     if model_name is not None:
#         out_path = (
#             Path(f"../graphics/horizontal_interp/{model_name}") /
#             f"quiver_comparison_{grid_loc}_{model_name}.png"
#         )
#     else:
#         out_path = (
#             Path("../graphics/horizontal_interp") /
#             f"quiver_comparison_{grid_loc}_ROMS.png"
#         )
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, dpi=300)
#     plt.close(fig)

def map_curvi(ds: xr.DataArray):

    # get model name from OFS attribute, if exists
    his_base = ds.attrs.get("his_base",None)
    if his_base is not None:
        model_name = his_base.split('.')[0]
    else: 
        model_name = "ROMS"
    
    # get grid angle at all water locations
    ds["angle"] = ds["angle"].where(ds["mask_rho"]==1, np.nan)
    
    # calculate grid angle change in eta direction
    dim_name = ds["angle"].dims[1]
    slice1 = ds["angle"].isel({dim_name: slice(None, -1)})
    slice2 = ds["angle"].isel({dim_name: slice(1, None)})
    d_xi = np.abs(slice1 - slice2)
    
    # calculate grid angle change in xi direction
    dim_name = ds["angle"].dims[0]
    slice1 = ds["angle"].isel({dim_name: slice(None, -1)})
    slice2 = ds["angle"].isel({dim_name: slice(1, None)})
    d_eta = np.abs(slice1 - slice2)
    
    # get the maximum change in grid angle (curvilinearity) 
    # across eta and xi directions
    curvi = xr.full_like(ds['angle'], fill_value=np.nan)
    curvi[:-1,:]=d_eta
    curvi[:,:-1]=xr.ufuncs.maximum(curvi[:,:-1], d_xi)
    curvi = curvi.where(ds["mask_rho"]==1, np.nan)

    fig, ax = plt.subplots(
        figsize = (6, 8), 
        subplot_kw={"projection":ccrs.PlateCarree()},
        dpi = 300
    )
    
    # add shoreline
    land_10m = cf.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='10m',
        facecolor=cf.COLORS['land'],
        edgecolor='grey',
    )
    ax.add_feature(land_10m, zorder=2)
        
    # plot max curvlinearity 
    mesh = ax.pcolormesh(
        ds["lon_rho"].values,
        ds["lat_rho"].values,
        curvi.values,
        cmap = "viridis", vmin = 0, vmax = 0.05,
        transform = ccrs.PlateCarree(),
        zorder = 3
    )
    cbar = plt.colorbar(
        mesh, ax=ax, orientation="vertical", 
        pad=0.03, shrink=0.7
    )
    cbar.set_label(
        "magnitude of rotation change", 
        fontsize = 11
    )
    ax.set_title(
        "Maximum change in grid rotation\n"
        f"across water cells in {model_name} grid.", 
        fontsize = 13
    )

    output_file = Path(
        f"../graphics/horizontal_interp/{model_name}/"
        f"curvilinear_magnitude_{model_name}.png"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)
    
    return curvi

def map_zonal_cell_width(ds:xr.DataArray):

    # get model name from OFS attribute, if exists
    his_base = ds.attrs.get("his_base",None)
    if his_base is not None:
        model_name = his_base.split('.')[0]
    else: 
        model_name = "ROMS"
        
    fig, ax = plt.subplots(
        figsize = (10, 8), 
        subplot_kw={"projection":ccrs.PlateCarree()},
        dpi = 300
    )
    
    # add shoreline
    land_10m = cf.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='10m',
        facecolor=cf.COLORS['land'],
        edgecolor='grey',
    )
    ax.add_feature(land_10m, zorder=2)
        
    # show zonal cell width 
    cell_width = (1/ds["pn"]).where(ds["mask_rho"]==1,np.nan).values
    mesh = ax.pcolormesh(
        ds["lon_rho"].values,
        ds["lat_rho"].values,
        cell_width,
        cmap = "viridis",
        transform = ccrs.PlateCarree(),
        zorder = 3
    )
    cbar = plt.colorbar(
        mesh, ax=ax, orientation="vertical", pad=0.03, shrink=0.7
    )
    cbar.set_label(
        "Grid cell length (meters)", 
        fontsize = 11
    )
    ax.set_title(
        "Zonal grid cell length\n"
        f"{np.nanmin(cell_width):5.2f} - "
        f"{np.nanmax(cell_width):5.2f} meters\n"
        f"Median value: {np.nanmedian(cell_width):5.2f} meters", 
        fontsize = 13
    )

    output_name = Path(
        f"../graphics/horizontal_interp/{model_name}/"
        f"grid_cell_length_{model_name}.png"
    )
    output_name.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(output_name)

