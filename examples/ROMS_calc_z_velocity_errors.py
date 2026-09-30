import numpy as np
import pandas as pd

import gridded

model = "../input_files/ROMS_wcofs_May2026_3D.nc"
eta_index, xi_index = 29, 8 # y->eta direction, x->xi direction
v_transform = 2

# Use Gridded to load dataset and variables
dataset = gridded.Dataset(model)
u_var = dataset.variables["u"]
v_var = dataset.variables["v"]
available_times = u_var.time.data

# Extract coordinates directly from netCDF
# u lives on (lat_u, lon_u); v lives on (lat_v, lon_v)
nc = dataset.nc_dataset
lon_u = nc.variables["lon_u"][eta_index, xi_index].item()
lat_u = nc.variables["lat_u"][eta_index, xi_index].item()
lon_v = nc.variables["lon_v"][eta_index, xi_index].item()
lat_v = nc.variables["lat_v"][eta_index, xi_index].item()

# Access variable-Specific Depth Objects (Rho-point centers)
u_depth_info = u_var.depth
v_depth_info = v_var.depth
u_depth_info.vtransform = v_transform
v_depth_info.vtransform = v_transform

# Initialize error log dictionaries
u_error_records = []
v_error_records = []

# Loop through time and estimate error for each depth and timestep
for t_idx, t in enumerate(available_times):
    # --- PROCESS U-VELOCITY ---
    # Construct an array for all vertical layers at this horizontal point
    # A temporary placeholder depth (e.g., 0) is used to get the
    # horizontal position
    u_loc_placeholder = np.array([[lon_u, lat_u, 0.0]])
    try: # main branch
        u_transect = u_depth_info.get_depth_profile(
            points = u_loc_placeholder,
            time = t,
            data_shape = u_var.data_shape[1:] #needed for a transect at native depths
        )
    except Exception as e:
        print(f"Error fetching u-velocity depth profile: {e}")

    # Build matrix for gridded (lon, lat, calculated_layer_depth)
    u_depth_locations = np.zeros((u_var.data_shape[1], 3))
    u_depth_locations[:, 0] = lon_u
    u_depth_locations[:, 1] = lat_u
    u_depth_locations[:, 2] = u_transect[0].data # u_native_depths

    # Use gridded's interpolation to get velocities
    gridded_u = u_var.at(points=u_depth_locations, time=t).data.flatten()

    # Pull raw values from the netCDF file at this specific time step and indices
    # ROMS dimensions are typically: (time, s_rho, eta_u, xi_u)
    raw_roms_u = nc.variables["u"][t_idx, :, eta_index, xi_index].data

    # Compute absolute error for u
    u_abs_err = np.abs(gridded_u - raw_roms_u)

    for layer in range(u_var.data_shape[1]):
        u_error_records.append({
            "time": t,
            "layer": layer,
            "depth": u_transect[0].data[layer],
            "roms_val": raw_roms_u[layer],
            "gridded_val": gridded_u[layer].item(),
            "error": u_abs_err[layer].item()
        })

    # --- PROCESS V-VELOCITY ---
    # Construct an array for all vertical layers at this horizontal point
    # A temporary placeholder depth (= 0) is used to get the
    # horizontal position

        # --- PROCESS V-VELOCITY ---
    v_loc_placeholder = np.array([[lon_v, lat_v, 0.0]])
    try: # roms_depth branch
        v_transect = v_depth_info.get_depth_profile(
            points = v_loc_placeholder,
            time = t,
            data_shape = v_var.data_shape[1:] #needed for a transect at native depths
        )
    except Exception as e:
        print(f"Error fetching V depth profile: {e}")

    # Build  matrix for gridded (lon, lat, calculated_layer_depth)
    v_depth_locations = np.zeros((v_var.data_shape[1], 3))
    v_depth_locations[:, 0] = lon_v
    v_depth_locations[:, 1] = lat_v
    v_depth_locations[:, 2] = v_transect[0].data

    # Query gridded's interpolation routine
    gridded_v = v_var.at(points = v_depth_locations, time = t).data.flatten()

    # Get raw values from the netCDF file at this specific time step
    # and indices.  ROMS dimensions are typically:
    # (time, s_rho, eta_v, xi_v)
    raw_roms_v = nc.variables["v"][t_idx, :, eta_index, xi_index].data
    print(raw_roms_v.shape, gridded_v.shape)
    print(np.nanmedian(raw_roms_v[:].data), np.nanmedian(gridded_v[:]))

    # Compute absolute error for v
    v_abs_err = np.abs(gridded_v - raw_roms_v)

    for layer in range(v_var.data_shape[1]):
        v_error_records.append({
            "time": t,
            "layer": layer,
            "depth": v_transect[0].data[layer],
            "roms_val": raw_roms_v[layer],
            "gridded_val": gridded_v[layer].item(),
            "error": v_abs_err[layer].item()
        })

# Create and save DataFrames for plotting
df_u_errors = pd.DataFrame(u_error_records)
df_v_errors = pd.DataFrame(v_error_records)

# Convert DataFrames to multi-indexed structures, then into Xarray Datasets
ds_u = df_u_errors.set_index(['time', 'layer']).to_xarray()
ds_v = df_v_errors.set_index(['time', 'layer']).to_xarray()

# df_u_errors.to_csv(
#     f'../output/ROMS_u_errors_eta{eta_index}_xi{xi_index}.csv',
#     index=False
# )
# df_v_errors.to_csv(
#     f'../output/ROMS_v_errors_eta{eta_index}_xi{xi_index}.csv',
#     index=False
# )

# Save Datasets to NetCDF
ds_u.to_netcdf(f'../output/ROMS_u_errors_eta{eta_index}_xi{xi_index}.nc')
ds_v.to_netcdf(f'../output/ROMS_v_errors_eta{eta_index}_xi{xi_index}.nc')

# Print most likely error across all layers and times
print("Median Absolute Error for u velocity:", np.nanmedian(df_u_errors["error"]))
print("Median Absolute Error for v velocity:", np.nanmedian(df_v_errors["error"]))

#
