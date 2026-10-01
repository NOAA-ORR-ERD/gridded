import pdb
from pathlib import Path

import numpy as np
import xarray as xr

import gridded


def avg_xi(arr):
    """
    Average values along the longitude/xi axis.

    Computes 2-point averages along the second dimension (columns), reducing the
    x-dimension size by 1.

    :param arr: Array containing field data to average.
    :type arr: numpy.ndarray
    :return: Spatial average along the xi axis.
    :rtype: numpy.ndarray
    """
    return 0.5 * (arr[:, :-1] + arr[:, 1:])

def avg_eta(arr):
    """
    Average values along the latitude/eta axis.

    Computes 2-point averages along the first dimension (rows), reducing the
    y-dimension size by 1.

    :param arr: Array containing field data to average.
    :type arr: numpy.ndarray
    :return: Spatial average along the eta axis.
    :rtype: numpy.ndarray
    """
    return 0.5 * (arr[:-1, :] + arr[1:, :])

def avg_4pt(arr):
    """
    4-point average across 2x2 grid cells, resulting in a diagonal shift.

    Averages neighboring 2x2 grid elements to interpolate between C-grid
    staggering locations (e.g., u to v locations, or vice versa).

    :param arr: 2D array containing grid values to average.
    :type arr: numpy.ndarray
    :return: 4-point averaged array with both dimensions reduced by 1.
    :rtype: numpy.ndarray
    """
    return 0.25 * (
        arr[:-1, :-1] + arr[1:, :-1] +
        arr[:-1, 1:]  + arr[1:, 1:]
    )

def trim_for_v_alignment(M,L):
    """Coordinate trimming to align averaged values with the locations of the 
    grid to which the v-velocities are being averaged. Only `rho` and `u` grids
    require trimming in this case. """
    trimming_rules = {
        "rho": (slice(1, M - 1), slice(0, L)),
        "u":   (slice(1, M - 1), slice(0, L)),
    }
    return trimming_rules

def trim_for_u_alignment(M,L):
    """Coordinate trimming to align averaged values with the locations of the 
    grid to which the v-velocities are being averaged. Only `rho` and `u` grids
    require trimming in this case. """
    trimming_rules = {
        "rho": (slice(0, M), slice(1, L - 1)),
        "v":   (slice(0, M - 1), slice(1, L - 1)),
    }
    return trimming_rules
    
def roms_grid_velocity_averages(
    input_file: str | Path = Path(
        "../../gnome_test_files/gridded_test_files/ROMS_wcofs_May2026_3D.nc"
    ),
    time_index: int = 0,
    depth_index: int | None = None,
    gridded_depth: float = 0.25 # having this as a default may lead to issues
):
    """
    Compute and compare ROMS velocity grid interpolations.

    Calculates Arakawa-C grid-staggering averages (u, v, rho, psi locations)--based
    on ROMS padding--for velocity components and compares these results to `gridded`
    interpolated values.

    :param input_file: Path to the 3D ROMS NetCDF dataset, defaults to 
        Path("../../gnome_test_files/ROMS_wcofs_May2026_3D.nc") from 
        https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/
    :type input_file: str | pathlib.Path, optional
    :param time_index: Time step index to extract from the dataset, defaults to 0.
    :type time_index: int, optional
    :param depth_index: Index to use for depth slice or None.  
        In ROMS, the surface is the Nth (last) level. This input is python indexing, 
        so the level would be the Nth level - 1. If None, then the 
        level defaults to the surface level.
    :type depth_index: int, optional
    :param gridded_depth: Depth (in meters) at which gridded should evaluate a comparison with
        with the ROMS depths.  It's up to the user to make sure that this depth is within
        the layer indexed by depth_index, and this depth cannot be zero.  The default value
        of 0.25 is recommended for the surface level but needs to be verified for a particular 
        application.
    :type depth_index: float, optional
    :return: A tuple containing:
        
        * **gridded_result** (*dict[str, xarray.Dataset]*): Dictionary mapping each
          staggered location (``'u'``, ``'v'``, ``'rho'``, ``'psi'``) to an xarray Dataset
          storing ROMS averaged values, gridded interpolated values, and the absolute 
          difference (error) between the two.
        * **u_trim_roms** (*dict[str, tuple[slice, slice]]*): Indexing slices applied to align
          u-velocity components on rho and v grids.
        * **v_trim_roms** (*dict[str, tuple[slice, slice]]*): Indexing slices applied to align
          v-velocity components on rho and u grids.
    :rtype: tuple[dict[str, xarray.Dataset], dict[str, tuple[slice, slice]], dict[str, tuple[slice, slice]]]
    :raises ValueError: If `depth_index` is less than 1 or is greater than or 
        equal to the total depth dimension `n_z`.
    """
    # --- Load ROMS file without gridded ---
    ds = xr.load_dataset(input_file)
    n_t, n_z, n_eta, n_xi = ds["u"].shape

    # --- Error check inputs ---
    if depth_index is None:
        # defaults to surface
        depth_index = n_z - 1
    elif (depth_index < 1) or (depth_index >= n_z):
        err_msg = (
            "depth_from_sfc_index must be an integer > 0 but less than"
            f" the depth dimension of {n_z}. For ROMS' surface level, "
            f"depth_from_sfc_index = 1 and the grid index used is {n_z} - 1."
        )
        raise ValueError(err_msg)
    else:
        depth_index = n_z - depth_index

    # --- Extract variables --
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

    # define how the roms lat/lon arrays need to be sliced to match the
    # u-averaged locations these will apply to Gridded results as well
    # since they are based on ROMS' grid locations
    u_trim_roms = trim_for_u_alignment(M,L) #{
    #     "rho": (slice(0, M), slice(1, L - 1)),
    #     "v":   (slice(0, M - 1), slice(1, L - 1)),
    # }

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

    # define how the roms lat/lon arrays need to be sliced to match the
    # u-averaged locations these will apply to Gridded results as well
    # since they are based on ROMS' grid locations
    v_trim_roms = trim_for_v_alignment(M,L) #{
    #     "rho": (slice(1, M - 1), slice(0, L)),
    #     "u":   (slice(1, M - 1), slice(0, L)),
    # }


    # --- Load variables with gridded and trim for indexing alignment ---
    gridded_ds = gridded.Dataset(str(input_file))
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
            (full_lons.ravel(),
             full_lats.ravel(),
             gridded_depth * np.ones(full_lons.size) # depth of evaluation
            )
        )

        # interpolate velocities, reshape into 2D array and trim down to
        # match locations of manual interpolation
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
        u_roms = u_roms_dict[grid_loc].data
        v_roms = v_roms_dict[grid_loc].data

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

    # ideally, u_trim_roms and v_trim_roms are in the DataArray...
    # but this solution will have to do for now.
    return gridded_result, u_trim_roms, v_trim_roms
