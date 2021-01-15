"""Test file for building the face_edge_connectivity"""
import numpy as np

from gridded.pyugrid import ugrid


def test_get_face_edge_orientation():
    faces = [[0, 1, 2], [1, 2, 3]]
    edges = [[0, 1], [1, 2], [0, 2], [3, 2], [3, 1]]
    nodes = [1, 2, 3, 4]
    grid = ugrid.UGrid(
        node_lon=nodes, node_lat=nodes, faces=faces, edges=edges
    )

    ref = [[1, 1, -1], [1, -1, 1]]
    orientation = grid.get_face_edge_orientation()

    assert orientation.tolist() == ref


def test_get_face_edge_orientation_na():
    faces = np.ma.array(
        [[0, 1, 2, 3], [1, 2, 3, -999]],
        mask=[[False, False, False, False], [False, False, False, True]],
    )
    edges = [[0, 1], [1, 2], [2, 3], [0, 3], [1, 3]]
    nodes = [1, 2, 3, 4]
    grid = ugrid.UGrid(
        node_lon=nodes, node_lat=nodes, faces=faces, edges=edges
    )

    ref = [[1, 1, 1, -1], [1, 1, -1, -999]]
    orientation = grid.get_face_edge_orientation()

    assert orientation.filled(-999).tolist() == ref
