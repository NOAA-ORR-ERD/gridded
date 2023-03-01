#!/usr/bin/env python

"""
The basic test of the UGrid object.

More specific functionality is other test modules.

"""


import numpy as np

import pytest

from gridded.grids import Grid_U as UGrid
from gridded.pyugrid.ugrid import IND_DT, NODE_DT

# FIXME: Break `test_full_set` into small unittests and check if the grid here
# is the same as `two_triangles`. If so use that.

# Some sample grid data: about the simplest triangle grid possible.
# 4 nodes, two triangles, five edges.

nodes = [(0.1, 0.1),
         (2.1, 0.1),
         (1.1, 2.1),
         (3.1, 2.1)]

node_lon = np.array(nodes)[:, 0]
node_lat = np.array(nodes)[:, 1]


faces = [(0, 1, 2),
         (1, 3, 2)]

edges = [(0, 1),
         (1, 3),
         (3, 2),
         (2, 0),
         (1, 2)]

boundaries = [(0, 1),
              (0, 2),
              (1, 3),
              (2, 3)]


def test_full_set():
    grid = UGrid(nodes=nodes,
                 faces=faces,
                 edges=edges,
                 boundaries=boundaries,
                 )

    # Check the dtype of key objects.
    # Implicitly makes sure they are numpy arrays (or array-like).
    assert grid.num_vertices == 3

    assert grid.nodes.dtype == NODE_DT
    assert grid.faces.dtype == IND_DT
    assert grid.edges.dtype == IND_DT
    assert grid.boundaries.dtype == IND_DT

    # Check shape of grid arrays.
    assert len(grid.nodes.shape) == 2
    assert len(grid.faces.shape) == 2
    assert len(grid.edges.shape) == 2
    assert len(grid.boundaries.shape) == 2

    assert grid.nodes.shape[1] == 2
    assert grid.faces.shape[1] == 3
    assert grid.edges.shape[1] == 2
    assert grid.boundaries.shape[1] == 2


def test_nodes_and_lon_lat():
    grid = UGrid(node_lon=node_lon,
                 node_lat=node_lat,
                 )

    assert np.all(grid.nodes == nodes)


def test_both_nodes_and_lon_lat():
    with pytest.raises(TypeError):
        grid = UGrid(node_lon=node_lon,
                     node_lat=node_lat,
                     nodes=nodes
                     )


def test_both_nodes_and_lon():
    with pytest.raises(TypeError):
        grid = UGrid(node_lon=node_lon,
                     nodes=nodes
                     )

def test_both_nodes_and_lat():
    with pytest.raises(TypeError):
        grid = UGrid(node_lat=node_lat,
                     nodes=nodes
                     )

