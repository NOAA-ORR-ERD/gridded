import tempfile
import os
import logging

import netCDF4
import numpy as np

import pytest


test_files = os.path.join(os.path.split(__file__)[0], 'files')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.yield_fixture
def quad_and_triangle():
    """create a quad and triangle grid"""

    # create tempfile in the test directory
    fname = tempfile.mktemp(
        suffix='.nc',
        dir=test_files,
        prefix='tmp_quad_and_triangle'
    )
    logger.debug("creating filename %s", fname)

    # TODO: this file is now created manually, after reading works,
    # also create it
    ds = netCDF4.Dataset(fname, 'w', datamodel='NETCDF4')

    ds.createDimension('nMesh2_node', 5)
    ds.createDimension('nMesh2_edge', 6)
    ds.createDimension('nMesh2_face', 2)
    ds.createDimension('nMaxMesh2_face_nodes', 4)
    ds.createDimension('two', 2)

    def update_with_attributes(obj, attributes):
        """update object with attributes"""
        for key, val in attributes.items():
            setattr(obj, key, val)

    attributes = dict(
        Conventions="UGRID-1.0",
        Title=("2D flexible mesh (mixed triangles,"
               " quadrilaterals, etc.) topology"),
        Institution="Deltares",
        References="fedor.baart@deltares.nl",
        History="created with %s" % (__file__, )
    )
    update_with_attributes(ds, attributes)

    mesh2 = ds.createVariable('Mesh2', 'int32')
    mesh2_attributes = dict(
        cf_role="mesh_topology",
        long_name="Topology data of 2D unstructured mesh",
        topology_dimension=2,
        node_coordinates="Mesh2_node_x Mesh2_node_y",
        face_node_connectivity="Mesh2_face_nodes",
        face_dimension="nMesh2_face",
        # attribute required if variables will be defined on edges
        edge_node_connectivity="Mesh2_edge_nodes",
        edge_dimension="nMesh2_edge",
        # optional attribute (requires edge_node_connectivity)
        edge_coordinates="Mesh2_edge_x Mesh2_edge_y",
        # optional attribute
        face_coordinates="Mesh2_face_x Mesh2_face_y",
        # optional attribute (requires edge_node_connectivity)
        face_edge_connectivity="Mesh2_face_edges",
        # optional attribute
        face_face_connectivity="Mesh2_face_links",
        # optional attribute (requires edge_node_connectivity)
        edge_face_connectivity="Mesh2_edge_face_links"
    )
    update_with_attributes(mesh2, mesh2_attributes)

    mesh2_face_nodes = ds.createVariable(
        'Mesh2_face_nodes', 'int32',
        dimensions=('nMesh2_face', 'nMaxMesh2_face_nodes'),
        fill_value=999999
    )
    mesh2_face_nodes_attrs = dict(
        cf_role="face_node_connectivity",
        long_name="Maps every face to its corner nodes.",
        start_index=0
    )
    update_with_attributes(mesh2_face_nodes, mesh2_face_nodes_attrs)

    mesh2_edge_nodes = ds.createVariable(
        'Mesh2_edge_nodes', 'int32',
        dimensions=('nMesh2_edge', 'two')
    )
    mesh2_edge_nodes_attrs = dict(
        cf_role="edge_node_connectivity",
        long_name="Maps every edge to the two nodes that it connects.",
        start_index=0
    )
    update_with_attributes(mesh2_edge_nodes, mesh2_edge_nodes_attrs)

    # Optional mesh topology variables
    mesh2_face_edges = ds.createVariable(
        'Mesh2_face_edges', 'int32',
        dimensions=('nMesh2_face', 'nMaxMesh2_face_nodes'),
        fill_value=999999
    )
    mesh2_face_edges_attrs = dict(
        cf_role="face_edge_connectivity",
        long_name="Maps every face to its edges.",
        start_index=0
    )
    update_with_attributes(mesh2_face_edges, mesh2_face_edges_attrs)

    mesh2_face_links = ds.createVariable(
        'Mesh2_face_links', 'int32',
        dimensions=('nMesh2_face', 'nMaxMesh2_face_nodes'),
        fill_value=999999)
    mesh2_face_links_attrs = dict(
        cf_role="face_face_connectivity",
        long_name="Indicates which other faces neighbor each face.",
        start_index=0,
        flag_values=-1,
        flag_meanings="out_of_mesh"
    )
    update_with_attributes(mesh2_face_links, mesh2_face_links_attrs)

    mesh2_edge_face_links = ds.createVariable(
        'Mesh2_edge_face_links', 'int32',
        dimensions=('nMesh2_edge', 'two'),
        fill_value=-999
    )
    mesh2_edge_face_links_attrs = dict(
        cf_role="edge_face_connectivity",
        long_name="neighbor faces for edges",
        start_index=0,
        comment="missing neighbor faces are indicated using _FillValue"
    )
    update_with_attributes(mesh2_edge_face_links, mesh2_edge_face_links_attrs)

    # Mesh node coordinates
    mesh2_node_x = ds.createVariable('Mesh2_node_x', 'double',
                                     dimensions=('nMesh2_node', ))
    mesh2_node_x_attrs = dict(
        standard_name="longitude",
        long_name="Longitude of 2D mesh nodes.",
        units="degrees_east"
    )
    update_with_attributes(mesh2_node_x, mesh2_node_x_attrs)

    mesh2_node_y = ds.createVariable('Mesh2_node_y', 'double',
                                     dimensions=('nMesh2_node', ))
    mesh2_node_y_attrs = dict(
        standard_name="latitude",
        long_name="Latitude of 2D mesh nodes.",
        units="degrees_north"
    )
    update_with_attributes(mesh2_node_y, mesh2_node_y_attrs)

    # Optional mesh face and edge coordinate variables
    mesh2_face_x = ds.createVariable('Mesh2_face_x', 'double',
                                     dimensions=('nMesh2_face', ))
    mesh2_face_x_attrs = dict(
        standard_name="longitude",
        long_name="Characteristics longitude of 2D mesh face.",
        units="degrees_east",
        bounds="Mesh2_face_xbnds"
    )
    update_with_attributes(mesh2_face_x, mesh2_face_x_attrs)

    mesh2_face_y = ds.createVariable('Mesh2_face_y', 'double',
                                     dimensions=('nMesh2_face', ))
    mesh2_face_y_attrs = dict(
        standard_name="latitude",
        long_name="Characteristics latitude of 2D mesh face.",
        units="degrees_north",
        bounds="Mesh2_face_ybnds"
    )
    update_with_attributes(mesh2_face_y, mesh2_face_y_attrs)

    mesh2_face_xbnds = ds.createVariable(
        'Mesh2_face_xbnds', 'double',
        dimensions=('nMesh2_face', 'nMaxMesh2_face_nodes'),
        fill_value=9.9692099683868690E36
    )
    mesh2_face_xbnds_attrs = dict(
        standard_name="longitude",
        long_name=("Longitude bounds of 2D mesh face "
                   "(i.e. corner coordinates)."),
        units="degrees_east"
    )
    update_with_attributes(mesh2_face_xbnds, mesh2_face_xbnds_attrs)

    mesh2_face_ybnds = ds.createVariable(
        'Mesh2_face_ybnds', 'double',
        dimensions=('nMesh2_face', 'nMaxMesh2_face_nodes'),
        fill_value=9.9692099683868690E36
    )
    mesh2_face_ybnds_attrs = dict(
        standard_name="latitude",
        long_name="Latitude bounds of 2D mesh face (i.e. corner coordinates).",
        units="degrees_north"
    )
    update_with_attributes(mesh2_face_ybnds, mesh2_face_ybnds_attrs)

    mesh2_edge_x = ds.createVariable(
        'Mesh2_edge_x', 'double',
        dimensions=('nMesh2_edge', )
    )
    mesh2_edge_x_attrs = dict(
        standard_name="longitude",
        long_name=("Characteristic longitude of 2D mesh edge"
                   " (e.g. midpoint of the edge)."),
        units="degrees_east"
    )
    update_with_attributes(mesh2_edge_x, mesh2_edge_x_attrs)

    mesh2_edge_y = ds.createVariable('Mesh2_edge_y', 'double',
                                     dimensions=('nMesh2_edge', ))
    mesh2_edge_y_attrs = dict(
        standard_name="latitude",
        long_name=("Characteristic latitude of 2D mesh edge"
                   " (e.g. midpoint of the edge)."),
        units="degrees_north"
    )
    update_with_attributes(mesh2_edge_y, mesh2_edge_y_attrs)

    '''
    We're working with this grid:
                     4
                    / \
                   /   \
                  5     \
                 /       4
                /         \
               /2     1    \
              /  \          3
             /    \        /
            2      1      /
           /   0    \    3
          /          \  /
         0-----0------1/
    '''

    mesh2_face_nodes[:] = [
        [0, 1, 2, 999999],
        [1, 3, 4, 2]
    ]
    mesh2_edge_nodes[:] = [
        [0, 1],
        [1, 2],
        [2, 0],
        [1, 3],
        [3, 4],
        [4, 2]
    ]
    mesh2_face_edges[:] = [
        [0, 1, 2, 999999],
        [3, 4, 5, 1]
    ]
    mesh2_face_links[:] = [
        [1, -1, -1, -1],
        [0, -1, -1, -1]
    ]
    mesh2_edge_face_links[:] = [
        [0, -999],
        [0, 1],
        [0, -999],
        [1, -999],
        [1, -999],
        [1, -999]
    ]

    mesh2_node_x[:] = [0.0, 1.0, 0.5, 1.5, 1.0]
    mesh2_node_y[:] = [0.0, 0.0, 1.0, 1.0, 2.0]

    mesh2_face_x[:] = [0.5, 1.0]
    mesh2_face_y[:] = [0.5, 1.0]

    mesh2_face_xbnds[:, :] = np.array([
        [0.0, 1.0, 0.5, 9.9692099683868690E36],
        [1.0, 1.5, 1.0, 0.5]
    ], dtype="double")
    mesh2_face_ybnds[:] = [
        [0.0, 0.0, 1.0, 9.9692099683868690E36],
        [0.0, 1.0, 2.0, 1.0]
    ]

    mesh2_edge_x[:] = [0.5, 0.75, 0.25, 1.25, 1.25, 0.75]
    mesh2_edge_y[:] = [0.0, 0.50, 0.50, 0.50, 1.50, 1.50]

    ds.sync()
    yield ds

    ds.close()
    os.remove(fname)
