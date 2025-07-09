"""
Assorted utilities useful for the tests.
"""

import contextlib
from pathlib import Path
import glob
import os

import pooch


import urllib.request as urllib_request  # for python 3

import pytest


HERE = Path(__file__).parent
EXAMPLE_DATA = HERE / "example_data"
TEST_DATA = HERE / "test_data"

# # Files on PYGNOME server -- add them here as needed
data_file_cache = pooch.create(
    # Use a local cache folder for the operating system
    # path=pooch.os_cache("plumbus"),
    path=EXAMPLE_DATA,
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
        "projected_coords_ugrid.nc": "sha256:019c1469c0583021268dbf1ea3eed97038364a0b7a361bc3f50b6be5f83b1ff2"
    },
)


def get_test_file_dir():
    """
    returns the test file dir path

    This should be replaced with simple code in the tests ...
    """
    return Path(__file__).parent / 'test_data'


def get_test_cdl_filelist():
    dirpath = os.path.join(get_test_file_dir(), 'cdl')
    return glob.glob(os.path.join(dirpath, '*.cdl'))


# def get_temp_test_file(filename):
#     """
#     returns the path to a temporary test file.

#     If it exists, it will return it directly.

#     If not, it will attempt to download it.

#     If it can't download, it will return None
#     """
#     print("getting temp test file")
#     filepath = os.path.join(os.path.dirname(__file__),
#                             'temp_data',
#                             filename)
#     if os.path.isfile(filepath):
#         print("already there")
#         return filepath
#     else:
#         # attempt to download it
#         print("trying to download")
#         try:
#             get_datafile(filepath)
#         except urllib_request.HTTPError:
#             print("got an error trying to download {}:".format(filepath))
#             return None
#         return None


@pytest.fixture
def two_triangles():
    """
    Returns a simple triangular grid: 4 nodes, two triangles, five edges.

    """
    nodes = [(0.1, 0.1),
             (2.1, 0.1),
             (1.1, 2.1),
             (3.1, 2.1)]

    faces = [(0, 1, 2),
             (1, 3, 2), ]

    edges = [(0, 1),
             (1, 3),
             (3, 2),
             (2, 0),
             (1, 2)]

    return ugrid.UGrid(nodes, faces, edges)


@pytest.fixture
def twenty_one_triangles():
    """
    Returns a basic triangular grid:  21 triangles, a hole, and a tail.

    """
    nodes = [(5, 1),
             (10, 1),
             (3, 3),
             (7, 3),
             (9, 4),
             (12, 4),
             (5, 5),
             (3, 7),
             (5, 7),
             (7, 7),
             (9, 7),
             (11, 7),
             (5, 9),
             (8, 9),
             (11, 9),
             (9, 11),
             (11, 11),
             (7, 13),
             (9, 13),
             (7, 15), ]

    faces = [(0, 1, 3),
             (0, 6, 2),
             (0, 3, 6),
             (1, 4, 3),
             (1, 5, 4),
             (2, 6, 7),
             (6, 8, 7),
             (7, 8, 12),
             (6, 9, 8),
             (8, 9, 12),
             (9, 13, 12),
             (4, 5, 11),
             (4, 11, 10),
             (9, 10, 13),
             (10, 11, 14),
             (10, 14, 13),
             (13, 14, 15),
             (14, 16, 15),
             (15, 16, 18),
             (15, 18, 17),
             (17, 18, 19), ]

    # We may want to use this later to define just the outer boundary.
    boundaries = [(0, 1),
                  (1, 5),
                  (5, 11),
                  (11, 14),
                  (14, 16),
                  (16, 18),
                  (18, 19),
                  (19, 17),
                  (17, 15),
                  (15, 13),
                  (13, 12),
                  (12, 7),
                  (7, 2),
                  (2, 0),
                  (3, 4),
                  (4, 10),
                  (10, 9),
                  (9, 6),
                  (6, 3), ]

    grid = ugrid.UGrid(nodes, faces, boundaries=boundaries)
    grid.build_edges()
    return grid


@contextlib.contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)
