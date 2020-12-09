"""Test file for building the face_edge_connectivity"""
import numpy as np

from gridded.pyugrid import ugrid


def test_build_face_edge_connectivity():
    faces = [[0, 1, 2], [1, 2, 3]]
    edges = [[0, 1], [1, 2], [2, 0], [2, 3], [3, 1]]
    nodes = [1, 2, 3, 4]
    grid = ugrid.UGrid(
        node_lon=nodes, node_lat=nodes, faces=faces, edges=edges
    )

    ref = [[2, 0, 1], [4, 1, 3]]
    grid.build_face_edge_connectivity()

    assert grid.face_edge_connectivity.tolist() == ref


def test_build_face_edge_connectivity_na():
    faces = np.ma.array(
        [[0, 1, 2, 3], [1, 2, 3, -999]],
        mask=[[False, False, False, False], [False, False, False, True]],
    )
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    nodes = [1, 2, 3, 4]
    grid = ugrid.UGrid(
        node_lon=nodes, node_lat=nodes, faces=faces, edges=edges
    )

    ref = [[3, 0, 1, 2], [-999, 1, 2, -999]]
    grid.build_face_edge_connectivity()

    assert grid.face_edge_connectivity.filled(-999).tolist() == ref
