# Gridded Examples

Here are some examples of how to work with, analyze, and visualize oceanographic model results using `gridded` environment objects.  

`Gridded` handles model output computed on either a regular, matrix-structured, (**i,j,k**)-indexed grid (referred to as a `structured grid`) or an irregular, explicitly-mapped mesh (referred to as an `unstructured grid`). Within structured grids, variables are either all placed at one location within the grid (`collocated`) or at different locations within the grid (`staggered`).  Common examples include:

- Regional Oceanographic Modeling System (ROMS): a structured grid with Arakawa-C staggering in the horizontal.
- Finite Volume Community Model (FVCOM): an unstructied grid with Arakawa-C staggering in the horizontal.
- Nucleus for European Modelling of the Ocean (NEMO): a structured grid with Arakawa-C staggering in the horizontal but a different edge padding (and, hence, horizontal indexing) than ROMS. 

To use `gridded` effectively, users must be aware of differences between models that might affect the portability of `gridded` code.  

Below are links to some examples for working with or plotting commonly used  model architectures. 

---

### General
* **File:** `roms_tools.py`
* **Description:** A set of functions used to manage ROMS model output.  

### Plotting model output from ADCIRC, SELFE and FVCOM 
* **File:** [UGRID_plotting_COMT.ipynb](https://github.com/NOAA-ORR-ERD/gridded/blob/main/examples/UGRID_plotting_COMT.ipynb)
* **Description:** Demonstrates accessing NetCDF files from ADCIRC, SELFE and FVCOM with attributes added or modified virtually using NcML.  Model output is first downloaded and opened from URL and then plotted.

### ROMS Depth Profile Plot
* **File:** `ROMS_calc_depth_profile.py` / `ROMS_calc_depth_profile.py`
[comment]: # `roms_depth_profile.py` / `roms_depth_profile.ipynb`
* **Description:** Extracts and plots vertical velocity profiles from a Regional Ocean Modeling System (ROMS) staggered grid dataset for first 9 timesteps at a specified eta=20, xi=10 location. 
* **Requires:** [ROMS_wcofs_May2026_3D.nc](https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/ROMS_wcofs_May2026_3D.nc)

### FVCOM Depth Profile Plot
* **File:** `FVCOM_plot_1x1_depth_profile_comparison`
[comment]: # `fvcom_depth_profile.py` / `fvcom_depth_profile.ipynb`
* **Description:** Extracts and plots vertical velocity profiles from an unstructured FVCOM staggered grid.
* **Requires:** SFBOFS_3D_07_2026.nc (need to add to repo)
 
### ROMS Depth Level Error Assessment
* **File:** `ROMS_calc_z_velocity_errors.py` / `ROMS_plot_heatmap_errors.py`
[comment]: #`roms_depth_level_error_assessment.py` / `roms_depth_level_error_assessment.ipynb` 
* **Description:** Evaluates `gridded` velocities against `ROMS` velocities at the native u-/v-velocity grid and depth locations using heatmaps of error for u- and v-velocities to quantify and visualize variations.  The test here is to confirm that `gridded` is tracking changing in SSH over time well and that the errors are negligible.
* **Requires:** [ROMS_wcofs_May2026_3D.nc](https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/ROMS_wcofs_May2026_3D.nc)

### ROMS Horizontal Error Assessment
* **File:** `roms_horizontal_error_assessment.py` / `roms_horizontal_error_assessment.ipynb`
* **Description:** Compare ROMS model predictions at grid locations to gridded estimates at the same locations using the absolute error (|u_gridded - u_roms|).  The ROMS prediction values are based on averages adjacent 2-values (rho and psi locations) or 4-values (u and v locations) in order to collocate u- and v-velocitites at the same grid location.  Gridded u- and v-velocities are averaged across 2-values in the x-direction (u-velocity) or y-direction (v-velocity).  Variables averaged in the x-direction are not averaged in the y-direction, and visa versa.  The absolute error reflects the magnitude of the difference between the two representations of velocities.  
* **Requires:** [ROMS_wcofs_May2026_3D.nc](https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/ROMS_wcofs_May2026_3D.nc)

---

## Running the Examples

To run these examples locally, ensure you have `gridded` and its development dependencies installed ([see instructions here](https://github.com/NOAA-ORR-ERD/gridded#installing)).  Some of these examples rely on particular model output NetCDF that will need to be downloaded from [this GNOME testing NetCDF repository](https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/), with the paths to these files tailored to reflect your local user space. 

