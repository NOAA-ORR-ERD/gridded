#!/usr/bin/env python

"""
tests loading a UGRID file with projected coords

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

from .utilities import get_temp_test_file


try:
    data_file = get_temp_test_file("projected_coords_ugrid.nc")
    if data_file is None:
        # skip these tests if the data file couldn't be downloaded
        pytestmark = pytest.mark.skip
except: # if anything went wrong, skip these.
    pytestmark = pytest.mark.skip


def test_load():
    """
    The file should load without error
    """
    ds = Dataset(data_file)

    assert isinstance(ds.grid, Grid_U)


def test_find_variables():
    """
    does it find the variables?
    """
    ds = Dataset(data_file)

    var_names = list(ds.variables.keys())
    all_vars = ['mesh2d_node_z',
                'mesh2d_Numlimdt',
                'mesh2d_s1',
                'mesh2d_waterdepth',
                'mesh2d_s0',
                'mesh2d_ucx',
                'mesh2d_ucy',
                'mesh2d_ucmag',
                'mesh2d_ucxq',
                'mesh2d_ucyq',
                'mesh2d_taus',
                'mesh2d_czs',
                'mesh2d_sa1',
                'mesh2d_flowelem_ba',
                'mesh2d_flowelem_bl',
                'mesh2d_face_x_bnd',
                'mesh2d_face_y_bnd',
                ]

    for var in all_vars:
        assert var in var_names
    assert len(all_vars) == len(var_names)


@pytest.mark.xfail
@pytest.mark.parametrize('var_name',
                   ['timestep',  # time
                    'mesh2d_edge_type',  # edge
                    'mesh2d_q1',  # edge
                    'mesh2d_viu',  # edge
                    'mesh2d_diu',  # edge
                    'mesh2d_hu',  # edge
                    'mesh2d_u1',  # edge
                    'mesh2d_u0',  # edge
                    ])
def test_missing_variables(var_name):
    """
    these are known not to be found

    but they should be!
    """
    ds = Dataset(data_file)

    var_names = list(ds.variables.keys())

    assert var_name in var_names
