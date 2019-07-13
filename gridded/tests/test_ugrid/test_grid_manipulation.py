#!/usr/bin/env python

"""
Testing of various utilities to manipulate the grid.

"""

from __future__ import (absolute_import, division, print_function)

import numpy as np

from utilities import two_triangles, twenty_one_triangles


def test_build_face_face_connectivity_small():
    ugrid = two_triangles()
    ugrid.build_face_face_connectivity()
    face_face = ugrid.face_face_connectivity

    assert np.array_equal(face_face[0], [-1, 1, -1])
    assert np.array_equal(face_face[1], [-1, -1, 0])


def test_build_face_face_connectivity_big():
    ugrid = twenty_one_triangles()
    ugrid.build_face_face_connectivity()
    face_face = ugrid.face_face_connectivity

    assert face_face[0].tolist() == [-1, 3, 2]
    assert face_face[9].tolist() == [8, 10, 7]
    assert face_face[8].tolist() == [-1, 9, 6]
    assert face_face[15].tolist() == [14, 16, 13]
    assert face_face[20].tolist() == [19, -1, -1]


def test_build_edges():
    ugrid = two_triangles()
    ugrid.build_edges()
    edges = ugrid.edges

    edges.sort(axis=0)
    assert np.array_equal(edges, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])


def test_build_face_coordinates():
    grid = two_triangles()
    grid.build_face_coordinates()
    coords = grid.face_coordinates

    assert coords.shape == (2, 2)
    assert np.allclose(coords, [(1.1, 0.76666667),
                                (2.1, 1.43333333)])


def test_build_edge_coordinates():
    grid = two_triangles()
    grid.build_edge_coordinates()
    coords = grid.edge_coordinates

    assert coords.shape == (5, 2)
    assert np.allclose(coords, [[1.1, 0.1],
                                [2.6, 1.1],
                                [2.1, 2.1],
                                [0.6, 1.1],
                                [1.6, 1.1]])


def test_build_boundary_coordinates():
    grid = two_triangles()
    grid.boundaries = [(0, 1), (0, 2), (2, 3), (1, 3)]
    grid.build_boundary_coordinates()
    coords = grid.boundary_coordinates

    assert coords.shape == (4, 2)
    assert np.allclose(coords, [[1.1, 0.1],
                                [0.6, 1.1],
                                [2.1, 2.1],
                                [2.6, 1.1]])


def test_build_boundaries_small():
    ugrid = two_triangles()
    ugrid.build_face_face_connectivity()
    ugrid.build_boundaries()

    boundaries = sorted(ugrid.boundaries.tolist())
    expected_boundaries = [[0, 1], [1, 3], [2, 0], [3, 2]]
    assert boundaries == expected_boundaries


def test_build_boundaries_big():
    ugrid = twenty_one_triangles()
    ugrid.build_face_face_connectivity()
    ugrid.build_boundaries()

    boundaries = sorted(ugrid.boundaries.tolist())
    expected_boundaries = [[0, 1], [1, 5], [2, 0], [3, 6], [4, 3], [5, 11],
                           [6, 9], [7, 2], [9, 10], [10, 4], [11, 14], [12, 7],
                           [13, 12], [14, 16], [15, 13], [16, 18], [17, 15],
                           [18, 19], [19, 17]]
    assert boundaries == expected_boundaries
