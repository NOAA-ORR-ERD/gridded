import datetime

import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline

import gridded

# =================================================================
# SETUP
# =================================================================
# ROMS input
model = "../input_files/ROMS_wcofs_May2026_3D.nc"
# Output filename
eta_index, xi_index = 20, 10  # y->eta direction, x->xi direction
output_netcdf = f"ROMS_depthprofiles_eta{eta_index}_xi{xi_index}.nc"
# Details
v_transform = 2
n_depths = 100 # Number of depths in interpolated gridded profile
POLY_DEGREE = 5  # A 3rd-degree cubic spline
s_parameter = 0.5  # Adjust this based on your grid noise scale

# Load dataset and variables
dataset = gridded.Dataset(model)
u_var = dataset.variables["u"]
v_var = dataset.variables["v"]
available_times = u_var.time.data

# Extract spatial coordinates directly from NetCDF
nc = dataset.nc_dataset
lon_u = nc.variables["lon_u"][eta_index, xi_index].item()
lat_u = nc.variables["lat_u"][eta_index, xi_index].item()
lon_v = nc.variables["lon_v"][eta_index, xi_index].item()
lat_v = nc.variables["lat_v"][eta_index, xi_index].item()

# Configure variable-specific depth transforms
u_depth_info = u_var.depth
v_depth_info = v_var.depth
u_depth_info.vtransform = v_transform
v_depth_info.vtransform = v_transform

# Initialize output dictionary logs
depth_profile = {
    "time": [],
    "roms_depth_u": [], "roms_depth_v": [],
    "roms_u": [], "roms_v": [],
    "gridded_depth_u": [], "gridded_depth_v": [],
    "gridded_u": [], "gridded_v": []
}
metrics = {
    "regress_roms_u": [], "regress_roms_v": [],
    "mae_u": [], "mae_v": [],
    "rmse_u": [], "rmse_v": []
}

# =================================================================
# LOOP OVER OUTPUT TIMESTEPS
# =================================================================
for t_idx, t in enumerate(available_times):
    # --- PROCESS U-VELOCITY ---
    u_loc_placeholder = np.array([[lon_u, lat_u, 0.0]])
    try:
        u_transect = u_depth_info.get_depth_profile(
            points=u_loc_placeholder,
            time=t,
            data_shape=u_var.data_shape[1:]
        )
    except Exception as e:
        print(f"Error fetching u-velocity depth profile: {e}")

    # Create depth profile at u-grid locations for gridded interpolation
    max_u_depth = u_transect[0].data.max()
    gridded_depths_u = np.arange(
        1e-3, max_u_depth,
        max_u_depth / n_depths
    )
    u_depth_locations = np.zeros((n_depths, 3))
    u_depth_locations[:, 0] = lon_u
    u_depth_locations[:, 1] = lat_u
    u_depth_locations[:, 2] = gridded_depths_u
    # Use gridded to interpolate u-velocities to gridded depths
    gridded_u = u_var.at(
        points = u_depth_locations,
        time = t
    ).data.flatten()
    # Save ROMS'-original
    raw_roms_u = nc.variables["u"][t_idx, :, eta_index, xi_index].data
    # Store u-velocity values
    depth_profile["time"].append(t)
    depth_profile["roms_depth_u"].append(u_transect[0].data)
    depth_profile["roms_u"].append(raw_roms_u)
    depth_profile["gridded_depth_u"].append(gridded_depths_u)
    depth_profile["gridded_u"].append(gridded_u)

    # --- PROCESS V-VELOCITY ---
    v_loc_placeholder = np.array([[lon_v, lat_v, 0.0]])
    try:
        v_transect = v_depth_info.get_depth_profile(
            points=v_loc_placeholder,
            time=t,
            data_shape=v_var.data_shape[1:]
        )
    except Exception as e:
        print(f"Error fetching V depth profile: {e}")

    max_v_depth = v_transect[0].data.max()
    gridded_depths_v = np.arange(1e-3, max_v_depth, max_v_depth/n_depths)
    # Create depth profile at v-grid locations for gridded interpolation
    v_depth_locations = np.zeros((n_depths, 3))
    v_depth_locations[:, 0] = lon_v
    v_depth_locations[:, 1] = lat_v
    v_depth_locations[:, 2] = gridded_depths_v
    # Use gridded to interpolate v-velocities to gridded depths
    gridded_v = v_var.at(points=v_depth_locations, time=t).data.flatten()
    # Save ROMS'-original
    raw_roms_v = nc.variables["v"][t_idx, :, eta_index, xi_index].data
    # Store v-velocity values
    depth_profile["roms_depth_v"].append(v_transect[0].data)
    depth_profile["roms_v"].append(raw_roms_v)
    depth_profile["gridded_depth_v"].append(gridded_depths_v)
    depth_profile["gridded_v"].append(gridded_v)

    # --- FIT POLYNOMIAL TO ROMS' PROFILES FOR ERROR APPROXIMATION ---
    r_depth = u_transect[0].data

    # # Fit Least-Squares Polynomial Curves directly onto ROMS points
    # coef_u = np.polyfit(r_depth, raw_roms_u, POLY_DEGREE)
    # coef_v = np.polyfit(r_depth, raw_roms_v, POLY_DEGREE)

    # # Regress profiles at the targeted gridded depth levels
    # regress_u = np.polyval(coef_u, gridded_depths)
    # regress_v = np.polyval(coef_v, gridded_depths)

    spline_u = UnivariateSpline(
        np.flip(u_transect[0].data), np.flip(raw_roms_u),
        k = POLY_DEGREE,
        s = s_parameter
    )(gridded_depths_u)
    spline_v = UnivariateSpline(
        np.flip(v_transect[0].data), np.flip(raw_roms_v),
        k = POLY_DEGREE,
        s = s_parameter
    )(gridded_depths_v)

    # Calculate difference between the gridded interpolation output and
    # the ROMS regression baseline
    diff_u = gridded_u - spline_u
    diff_v = gridded_v - spline_v

    metrics["regress_roms_u"].append(spline_u)
    metrics["regress_roms_v"].append(spline_v)
    metrics["mae_u"].append(np.max(np.abs(diff_u)))
    metrics["mae_v"].append(np.max(np.abs(diff_v)))
    metrics["rmse_u"].append(np.sqrt(np.mean(diff_u**2)))
    metrics["rmse_v"].append(np.sqrt(np.mean(diff_v**2)))

# --- BUILD XARRAY DATASET AND EXPORT ---
times = np.array(depth_profile["time"])
gridded_u_depths_axis = np.array(depth_profile["gridded_depth_u"][0])
gridded_v_depths_axis = np.array(depth_profile["gridded_depth_v"][0])
n_s_rho = np.array(depth_profile["roms_u"]).shape[1]
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
ds = xr.Dataset(
    data_vars={
        "roms_depth_u": (["time", "s_rho"], np.array(depth_profile["roms_depth_u"])),
        "roms_depth_v": (["time", "s_rho"], np.array(depth_profile["roms_depth_v"])),
        "roms_u": (["time", "s_rho"], np.array(depth_profile["roms_u"])),
        "roms_v": (["time", "s_rho"], np.array(depth_profile["roms_v"])),
        "gridded_depth_u": (["gridded_u_depth_dim"], gridded_u_depths_axis),
        "gridded_depth_v": (["gridded_v_depth_dim"], gridded_v_depths_axis),
        "gridded_u": (
            ["time", "gridded_depth_dim"],
            np.array(depth_profile["gridded_u"])
        ),
        "gridded_v": (
            ["time", "gridded_depth_dim"],
            np.array(depth_profile["gridded_v"])
        ),

        # Updated Regression variables
        "regression_roms_u": (
            ["time", "gridded_depth_dim"],
            np.array(metrics["regress_roms_u"])
        ),
        "regression_roms_v": (
            ["time", "gridded_depth_dim"],
            np.array(metrics["regress_roms_v"])
        ),
        "mae_u": (["time"], np.array(metrics["mae_u"])),
        "mae_v": (["time"], np.array(metrics["mae_v"])),
        "rmse_u": (["time"], np.array(metrics["rmse_u"])),
        "rmse_v": (["time"], np.array(metrics["rmse_v"])),
    },
    coords={
        "time": times,
        "s_rho": np.arange(n_s_rho),
        "gridded_u_depth_dim": np.arange(len(gridded_u_depths_axis)),
        "gridded_v_depth_dim": np.arange(len(gridded_v_depths_axis))
    },
    attrs={
        "description": (
            "Extracted ROMS vs Gridded profile data and "
            f"regression error metrics at eta={eta_index}, xi={xi_index}"
        ),
        "history": f"Created on {now}"
    }
)

ds.to_netcdf(f"../output/{output_netcdf}")
print("Saved complete profile regression analysis to NetCDF!")
