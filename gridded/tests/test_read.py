#!/usr/bin/env python

"""
Tests for testing a UGrid file read.

We really need a **lot** more sample data files...
"""


import os
from pathlib import Path

import numpy as np
import netCDF4

from gridded import Dataset
from gridded.variable import Variable
from gridded.tests.utilities import chdir, get_test_file_dir
from gridded.pyugrid.ugrid import UGrid
from gridded.pysgrid.sgrid import SGrid
# from pyugrid import read_netcdf

HERE = Path(__file__).parent
test_data_dir = HERE / 'test_data'
output_dir = HERE / 'output'
output_dir.mkdir(exist_ok=True)


def test_simple_read():
    """
    Can it be read at all?
    NOTE: passing the file name into the constructor is now deprecated
    """

    ds = Dataset.from_netCDF(test_data_dir / 'UGRIDv0.9_eleven_points.nc')
    assert isinstance(ds, Dataset)
    assert isinstance(ds.grid, UGrid)


def test_read_variables():
    """
    It should get the variables in the:

    UGRIDv0.9_eleven_points.nc file
    """

    ds = Dataset.from_netCDF(test_data_dir / 'UGRIDv0.9_eleven_points.nc')
    varnames = list(ds.variables.keys())
    varnames.sort()
    print("variables are:", varnames)
    assert varnames == ['Mesh2_boundary_count',
                        'Mesh2_boundary_types',
                        'Mesh2_depth',
                        'Mesh2_face_u',
                        'Mesh2_face_v']
    # assert varnames == ['Mesh2_depth', 'Mesh2_face_u', 'Mesh2_face_v']
    for v in ds.variables.values():
        assert isinstance(v, Variable)


def test_read_variable_attributes():
    ds = Dataset(test_data_dir / 'UGRIDv0.9_eleven_points.nc')
    print(ds.variables['Mesh2_depth'].attributes)
    assert (ds.variables['Mesh2_depth'].attributes['standard_name'] ==
            'sea_floor_depth_below_geoid')
    assert ds.variables['Mesh2_depth'].attributes['units'] == 'm'


# def test_read_FVCOM():
#     '''Optional test to make sure that files from  NGOFS are read correctly.'''
#     with chdir(test_data_dir):
#         if os.path.exists('COOPS_NGOFS.nc'):
#             ds = Dataset('COOPS_NGOFS.nc')
#             print("COOPS_NGOFS variables are:", ds.variables.keys())
#             assert isinstance(ds, Dataset)
#             assert isinstance(ds.grid, UGrid)
#             assert isinstance(ds.variables, dict)
#             assert 'u' in ds.variables.keys()
#             assert 'v' in ds.variables.keys()
#         else:
#             print("COOPS_NGOFS.nc could not be found")
#             assert False


# def test_read_TAMU():
#     """
#     Test to see if the TAMU files are read correctly
#     """
#     with chdir(test_data_dir):
#         if os.path.exists('TAMU.nc'):
#             ds = Dataset('TAMU.nc')
#             print("TAMU variables are:", ds.variables.keys())
#             assert isinstance(ds, Dataset)
#             assert isinstance(ds.grid, SGrid)
#             assert isinstance(ds.variables, dict)
#             assert 'water_u' in ds.variables.keys()
#             assert 'water_v' in ds.variables.keys()
#         else:
#             print("TAMU.nc could not be found")
#             assert False


def test_read_variable():
    """
    at least see if one variable can be read :-)

    this is an old pyugrid test ported over.
    """
    expected_depth = [ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    expected_depth_attributes = {'standard_name': 'sea_floor_depth_below_geoid',
                                 'units': 'm',
                                 'positive': 'down',
                                 'coordinates': 'Mesh2_node_y Mesh2_node_x',
                                 'grid': 'Bathymetry_Mesh',
                                 'long_name': 'Bathymetry',
                                 'type': 'data'
                                 }
    ds = Dataset.from_netCDF(test_data_dir / 'UGRIDv0.9_eleven_points_with_depth.nc')

    depth = ds['h']
    assert np.array_equal(depth.data, expected_depth)
    assert depth.attributes == expected_depth_attributes


def test_read_from_nc_dataset():
    """
    Minimal test, but makes sure you can read from an already
    open netCDF4.Dataset.
    """
    with netCDF4.Dataset(test_data_dir / 'UGRIDv0.9_eleven_points_with_depth.nc') as nc:
            ds = Dataset.from_netCDF(nc)
    assert ds.grid.mesh_name == 'Mesh2'
    assert ds.grid.nodes.shape == (11, 2)
    assert ds.grid.faces.shape == (13, 3)


