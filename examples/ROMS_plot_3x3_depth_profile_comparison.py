import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

# =================================================================
# SETUP
# =================================================================
# Load netcdf created by ROMS_calc_depth_profile.py
eta_index, xi_index = 20, 10
nc_path = f'../output/ROMS_depthprofiles_eta{eta_index}_xi{xi_index}.nc'
ds = xr.open_dataset(nc_path)
times = ds.time.values
gridded_depth_u = ds.gridded_depth_u.values
gridded_depth_v = ds.gridded_depth_v.values
# marker size for line plot markers
ms = 8
# line plot colors: u-velocities (cool), v-velocities (warm)
colors = {
    "roms_u": "#1f4e79",     # Deep Navy Blue
    "gridded_u": "#5b9bd5",  # Light Sky Blue
    "roms_v": "#c65911",     # Deep Burnt Orange
    "gridded_v": "#ffc000"   # Bright Tangerine/Yellow-Orange
}

# =================================================================
# PLOT ROMS PROFILE VS GRIDDED INTERPOLATION (sample of timesteps)
# =================================================================
MAX_TIMESTEPS = 9 #number of time steps to plot
nrows, ncols = 3, 3

fig1, axes = plt.subplots(
    nrows, ncols,
    figsize=(15, 15),
    sharey=True,
    sharex=True
)
axes = axes.flatten()
for i in range(min(MAX_TIMESTEPS, len(times))):
    ax = axes[i]
    t_str = pd.to_datetime(times[i]).strftime("%Y-%m-%d %H:%M")

    # store error metrics
    u_rmse = ds["rmse_u"][i].values
    u_mae  = ds["mae_u"][i].values
    v_rmse = ds["rmse_v"][i].values
    v_mae  = ds["mae_v"][i].values

    # -------------
    # ROMS PROFILES (Dashed-dotted lines, with markers)
    # -------------
    # u-velocity component
    ax.plot(
        ds["roms_u"][i].values, ds["roms_depth_u"][i].values,
        color=colors["roms_u"], alpha=0.5,
        linestyle="-", marker=".", markersize=ms, linewidth=1.5,
        label="ROMS u" if i == 0 else ""
    )
    # u-velocity regression
    ax.plot(
        ds["regression_roms_u"][i].values, gridded_depth_u,
        color=colors["roms_u"],
        linestyle="-", linewidth=1,  alpha=0.5,
        label="ROMS (regress) u" if i == 0 else ""
    )
    # v-velocity component
    ax.plot(
        ds["roms_v"][i].values, ds["roms_depth_v"][i].values,
        color=colors["roms_v"], alpha=0.5,
        linestyle="-", marker=".", markersize=ms, linewidth=1,
        label="ROMS v" if i == 0 else ""
    )
    # v-velocity regression
    ax.plot(
        ds["regression_roms_v"][i].values, gridded_depth_v,
        color=colors["roms_v"],
        linestyle="-", linewidth=1,  alpha=0.5,
        label="ROMS (regress) v" if i == 0 else ""
    )

    # ----------------
    # GRIDDED PROFILES (Solid lines, NO markers)
    # ----------------
    # u-velocity component
    ax.plot(
        ds["gridded_u"][i].values, gridded_depth_u,
        color=colors["gridded_u"], linestyle="-", linewidth=1.8,
        label="Gridded u" if i == 0 else ""
    )
    # v-velocity component
    ax.plot(
        ds["gridded_v"][i].values, gridded_depth_v,
        color=colors["gridded_v"], linestyle="-", linewidth=1.8,
        label="Gridded v" if i == 0 else ""
    )

    # Subplot styling
    ax.set_title(t_str, fontsize=11, weight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.axvline(0, color="black", linestyle="-", alpha=0.4)
    ax.invert_yaxis()  # Surface (0m) at the top

    # Clean outer labels
    if i % ncols == 0:
        ax.set_ylabel("Depth (m)", fontsize=11)
    if i >= (nrows - 1) * ncols:
        ax.set_xlabel("Velocity (m/s)", fontsize=11)

    # # legend in first (for the lazy)
    # if i == 0:
    #     ax.legend(
    #         loc="lower left", #bbox_to_anchor=(0.5, 0.96),
    #         fontsize=11, frameon=True
    #     )

    # Construct the multi-line annotation text string
    u_err_str = (
        f"{u_rmse:.2f}, {u_mae:.2f}"
    )
    v_err_str = (
        f"{v_rmse:.2f}, {v_mae:.2f}"
    )

    # Place text
    ax.text(
        0.03, 0.16, r"RMSE, Max|$\Delta$|",
        transform=ax.transAxes,
        fontsize=14,
        color = "grey",
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7)
    )
    ax.text(
        0.03, 0.10, u_err_str,
        transform=ax.transAxes,
        fontsize=14,
        color = colors["gridded_u"],
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7)
    )
    ax.text(
        0.03, 0.04, v_err_str,
        transform=ax.transAxes,
        fontsize=14,
        color = "goldenrod",
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.7)
    )

# Trim any trailing empty subplots
for j in range(i + 1, len(axes)):
    fig1.delaxes(axes[j])

# Place a clean global legend at the top
fig1.legend(
    loc="upper center", bbox_to_anchor=(0.5, 0.96),
    ncol=4, fontsize=11, frameon=True
)
plt.suptitle(
    "Velocity Depth Profiles by Timestep (ROMS vs Gridded)",
    fontsize=16, weight="bold", y=0.98
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(
    "../graphics/ROMS_vs_Gridded_3x3_profile_comparison.png",
    dpi=300
)
plt.close()
