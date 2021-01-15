"""Test file for building the face_edge_connectivity"""
import numpy as np

from gridded.pyugrid import ugrid


def test_create_dual_edge_mesh():
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
    ref = [
        [0, center1, 1, -999],
        [1, center1, 2, center2],
        [2, center1, 0, -999],
        [1, center2, 3, -999],
        [3, center2, 2, -999],
    ]

    dual_faces = grid._create_dual_edge_mesh()[0]

    assert np.ma.isMA(dual_faces)
    assert dual_faces.filled(-999).tolist() == ref


def test_create_dual_edge_mesh_na():
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

    center1 = len(nodex)
    center2 = center1 + 1
    ref = [
        [0, center1, 1, -999],
        [1, center1, 2, -999],
        [2, center1, 3, center2],
        [3, center1, 0, -999],
        [2, center2, 4, -999],
        [4, center2, 3, -999],
    ]

    dual_faces = grid._create_dual_edge_mesh()[0]
    assert np.ma.isMA(dual_faces)
    assert dual_faces.filled(-999).tolist() == ref
