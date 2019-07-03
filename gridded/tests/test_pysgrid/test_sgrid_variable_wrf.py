"""
Test SGrid Variables WRF.

Created on Apr 15, 2015

@author: ayan

"""


from __future__ import (absolute_import, division, print_function)

import pytest

from ..sgrid import SGrid
from ..utils import GridPadding
from ..variables import SGridVariable
from .write_nc_test_files import wrf_sgrid


@pytest.fixture
def sgrid_var_wrf(wrf_sgrid):
    face_padding = [GridPadding(mesh_topology_var=u'grid',
                                face_dim=u'west_east',
                                node_dim=u'west_east_stag',
                                padding=u'none'),
                    GridPadding(mesh_topology_var=u'grid',
                                face_dim=u'south_north',
                                node_dim=u'south_north_stag',
                                padding=u'none')]
    node_dimensions = 'west_east_stag south_north_stag'
    return dict(sgrid=SGrid(face_padding=face_padding,
                            node_dimensions=node_dimensions),
                test_var_1=wrf_sgrid.variables['SNOW'],
                test_var_2=wrf_sgrid.variables['FAKE_U'])


def test_face_location_inference1(sgrid_var_wrf):
    sg_var = SGridVariable.create_variable(sgrid_var_wrf['test_var_1'],
                                           sgrid_var_wrf['sgrid'])
    sg_var_location = sg_var.location
    expected_location = 'face'
    assert sg_var_location == expected_location


def test_edge_location_inference2(sgrid_var_wrf):
    sg_var = SGridVariable.create_variable(sgrid_var_wrf['test_var_2'],
                                           sgrid_var_wrf['sgrid'])
    sg_var_location = sg_var.location
    expected_location = 'edge1'
    assert sg_var_location == expected_location
