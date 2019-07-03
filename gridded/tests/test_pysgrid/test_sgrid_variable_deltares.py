"""
Test SGrid Variables Deltares.

Created on Apr 15, 2015

@author: ayan

"""

from __future__ import (absolute_import, division, print_function)

import pytest

from ..sgrid import SGrid
from ..utils import GridPadding
from ..variables import SGridVariable
from .write_nc_test_files import deltares_sgrid


@pytest.fixture
def sgrid_vars_deltares(deltares_sgrid):
    face_padding = [GridPadding(mesh_topology_var=u'grid',
                                face_dim=u'MMAXZ',
                                node_dim=u'MMAX',
                                padding=u'low'),
                    GridPadding(mesh_topology_var=u'grid',
                                face_dim=u'NMAXZ',
                                node_dim=u'NMAX',
                                padding=u'low')]
    node_dimensions = 'MMAX NMAX'
    return dict(sgrid=SGrid(face_padding=face_padding,
                            node_dimensions=node_dimensions),
                test_var_1=deltares_sgrid.variables['FAKE_W'],
                test_var_2=deltares_sgrid.variables['FAKE_U1'])


def test_face_location_inference_deltares(sgrid_vars_deltares):
    sg_var = SGridVariable.create_variable(sgrid_vars_deltares['test_var_1'],
                                           sgrid_vars_deltares['sgrid'])
    sg_var_location = sg_var.location
    expected_location = 'face'
    assert sg_var_location == expected_location


def test_edge_location_inference_deltares(sgrid_vars_deltares):
    sg_var = SGridVariable.create_variable(sgrid_vars_deltares['test_var_2'],
                                           sgrid_vars_deltares['sgrid'])
    sg_var_location = sg_var.location
    expected_location = 'edge2'
    assert sg_var_location == expected_location
