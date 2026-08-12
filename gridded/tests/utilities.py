"""
Assorted utilities useful for the tests.


NOTE: the fixtures should be in conftest, so they can be found automatically
"""

import contextlib
import os
from pathlib import Path

import pooch

HERE = Path(__file__).parent
EXAMPLE_DATA = HERE / "example_data"
TEMP_DATA = HERE / "temp_data"
TEST_DATA = HERE / "test_data"

TEST_CDL_FILES = list((TEST_DATA / "cdl").glob("*.cdl"))


# # Files on PYGNOME server -- add them here as needed
data_file_cache = pooch.create(
    # Use a local cache folder for the operating system
    # path=pooch.os_cache("plumbus"),
    path=TEMP_DATA,
    # The remote data is on the pygnome server
    base_url="https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/",
    # version=version,
    # # If this is a development version, get the data from the "main" branch
    # version_dev="main",
    registry={
        "3D_ROMS_example.nc": "sha256:d802d408bf3925dd77ff582bf906b95062eb65161de7b2290fb8d41537a566b6",
        "FVCOM-Erie-OFS-subsetter.nc": "sha256: 96c20ef1f4c463838c86e88baa9eba05aacb2db6fe184dc6d338489c38827567",
        "ROMS-WCOFS-OFS-subsetter.nc": "sha256:04af4479331894ab3abbd789fbfc2e4717c39e9c62123942929775a40406b9e9",
        "SSCOFS.ugrid.nc": "sha256:0dcea2a2fb6ad87c7cce3ebc475fd2f0430616a5019f54f4adf97391e075e939",
        "projected_coords_ugrid.nc": "sha256:019c1469c0583021268dbf1ea3eed97038364a0b7a361bc3f50b6be5f83b1ff2",
    },
)

@contextlib.contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)
