#!/usr/bin/env python

"""
tests loading a UGRID file with projected coords

This is also a very complete UGRID dataset, with data on nodes, edges, etc...

this test uses a data file auto-downloaded from ORR:

http://gnome.orr.noaa.gov/py_gnome_testdata/

Questions about this data;

there is a node_z variable:

    double mesh2d_node_z(nmesh2d_node) ;
        mesh2d_node_z:mesh = "mesh2d" ;
        mesh2d_node_z:location = "node" ;
        mesh2d_node_z:coordinates = "mesh2d_node_x mesh2d_node_y" ;
        mesh2d_node_z:standard_name = "altitude" ;
        mesh2d_node_z:long_name = "z-coordinate of mesh nodes" ;
        mesh2d_node_z:units = "m" ;
        mesh2d_node_z:grid_mapping = "projected_coordinate_system" ;
        mesh2d_node_z:_FillValue = -999. ;

But also a depth variable:

    double mesh2d_waterdepth(time, nmesh2d_face) ;
        mesh2d_waterdepth:mesh = "mesh2d" ;
        mesh2d_waterdepth:location = "face" ;
        mesh2d_waterdepth:coordinates = "mesh2d_face_x mesh2d_face_y" ;
        mesh2d_waterdepth:cell_methods = "nmesh2d_face: mean" ;
        mesh2d_waterdepth:standard_name = "sea_floor_depth_below_sea_surface" ;
        mesh2d_waterdepth:long_name = "Water depth at pressure points" ;
        mesh2d_waterdepth:units = "m" ;
        mesh2d_waterdepth:grid_mapping = "projected_coordinate_system" ;
        mesh2d_waterdepth:_FillValue = -999. ;

So what does mesh2d_node_z mean?
  This is a 2D grid -- what is the z coord of the node mean?

Also, this:

    double mesh2d_face_x_bnd(nmesh2d_face, max_nmesh2d_face_nodes) ;
        mesh2d_face_x_bnd:units = "m" ;
        mesh2d_face_x_bnd:standard_name = "projection_x_coordinate" ;
        mesh2d_face_x_bnd:long_name = "x-coordinate bounds of 2D mesh face (i.e. corner coordinates)" ;
        mesh2d_face_x_bnd:mesh = "mesh2d" ;
        mesh2d_face_x_bnd:location = "face" ;
        mesh2d_face_x_bnd:_FillValue = -999. ;

This is detected as a face variable by gridded -- but should it be a grid parameter or ??

"""

import pytest

from gridded import Dataset
from gridded.grids import Grid_U

from gridded import VALID_UGRID_LOCATIONS

from .utilities import data_file_cache

# try:
#     data_file = get_temp_test_file("projected_coords_ugrid.nc")
#     if data_file is None:
#         # skip these tests if the data file couldn't be downloaded
#         pytestmark = pytest.mark.skip
# except: # if anything went wrong, skip these.
#     pytestmark = pytest.mark.skip

data_file = data_file_cache.fetch("projected_coords_ugrid.nc")

def test_load():
    """
    The file should load without error
    """
    ds = Dataset.from_netCDF(data_file)

    assert isinstance(ds.grid, Grid_U)

    print(ds.grid.nodes.max(), ds.grid.nodes.min())
    assert ds.grid.nodes.min() > 148_000  # definitely not lat-lon


def test_find_variables():
    """
    Does it find the variables?
    """
    ds = Dataset.from_netCDF(data_file)

    var_names = list(ds.variables.keys())

    print(var_names)

    all_vars =  ['mesh2d_Numlimdt',
                 'mesh2d_czs',
                 'mesh2d_diu',
                 'mesh2d_edge_type',
                 'mesh2d_edge_x_bnd',
                 'mesh2d_edge_y_bnd',
                 'mesh2d_face_x_bnd',
                 'mesh2d_face_y_bnd',
                 'mesh2d_flowelem_ba',
                 'mesh2d_flowelem_bl',
                 'mesh2d_hu',
                 'mesh2d_node_z',
                 'mesh2d_q1',
                 'mesh2d_s0',
                 'mesh2d_s1',
                 'mesh2d_sa1',
                 'mesh2d_taus',
                 'mesh2d_u0',
                 'mesh2d_u1',
                 'mesh2d_ucmag',
                 'mesh2d_ucx',
                 'mesh2d_ucxq',
                 'mesh2d_ucy',
                 'mesh2d_ucyq',
                 'mesh2d_viu',
                 'mesh2d_waterdepth']

    all_vars.sort()
    var_names.sort()

    assert var_names == all_vars

    # check that they all have valid location attributes:
    VALID_LOCATIONS = [loc for loc in VALID_UGRID_LOCATIONS if loc is not None]
    for varname in var_names:
        assert ds[varname].location in VALID_LOCATIONS

