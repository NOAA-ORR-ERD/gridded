# coding: utf-8
"""Test file for building the face_edge_connectivity"""
import numpy as np

from gridded.pyugrid import ugrid


def test_create_dual_node_mesh():
    r"""Test for a mesh like

     /|\
    /_|_\
    """
    nodex = [0, 1, 1, 2]
    nodey = [0, 1, 0, 0]
    faces = [[0, 1, 2], [2, 1, 3]]

    edges = [[0, 1], [1, 2], [2, 0], [1, 3], [2, 3]]
    grid = ugrid.UGrid(
        node_lon=nodex, node_lat=nodey, faces=faces, edges=edges
    )

    center1 = len(nodex)
    center2 = center1 + 1
    e01 = center2 + 1
    e02 = e01 + 1
    e12 = e02 + 1
    e23 = e12 + 1
    ref = [
        [4, 6, 0, 7, -999],
        [6, 4, 5, 8, 1],
        [5, 4, 7, 2, 9],
        [8, 5, 9, 3, -999]
    ]

    dual_faces = grid._create_dual_node_mesh()[0]

    assert np.ma.isMA(dual_faces)
    assert dual_faces.filled(-999).tolist() == ref


def test_create_dual_node_mesh_na():
    r"""Test for a mesh like

    |‾|\
    |_|_\
    """
    nodex = [0, 0, 1, 1, 2]
    nodey = [0, 1, 1, 0, 0]
    faces = np.ma.array(
        [[0, 1, 2, 3], [2, 4, 3, -999]],
        mask=[[False, False, False, False], [False, False, False, True]],
    )

    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [4, 3]]
    grid = ugrid.UGrid(
        node_lon=nodex, node_lat=nodey, faces=faces, edges=edges
    )

    ref = [
        [5, 7, 0, 8, -999],
        [7, 5, 9, 1, -999],
        [9, 5, 6, 10, 2],
        [6, 5, 8, 3, 11],
        [10, 6, 11, 4, -999]
    ]

    dual_faces = grid._create_dual_node_mesh()[0]
    assert np.ma.isMA(dual_faces)
    assert dual_faces.filled(-999).tolist() == ref
