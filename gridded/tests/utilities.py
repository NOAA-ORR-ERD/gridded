"""
Assorted utilities useful for the tests.

"""

from __future__ import (absolute_import,
                        division,
                        print_function,
                        unicode_literals)

import os
import contextlib

import pytest

# from pyugrid import ugrid


def get_test_file_dir():
    """
    returns the test file dir path
    """
    test_file_dir = os.path.split(os.path.abspath(__file__))[0]
    test_file_dir = os.path.join(test_file_dir, 'test_data')
    return test_file_dir


@pytest.fixture
def two_triangles():
    """
    Returns a simple triangular grid: 4 nodes, two triangles, five edges.

    """
    nodes = [(0.1, 0.1),
             (2.1, 0.1),
             (1.1, 2.1),
             (3.1, 2.1)]

    faces = [(0, 1, 2),
             (1, 3, 2), ]

    edges = [(0, 1),
             (1, 3),
             (3, 2),
             (2, 0),
             (1, 2)]

    return ugrid.UGrid(nodes, faces, edges)


@pytest.fixture
def twenty_one_triangles():
    """
    Returns a basic triangular grid:  21 triangles, a hole, and a tail.

    """
    nodes = [(5, 1),
             (10, 1),
             (3, 3),
             (7, 3),
             (9, 4),
             (12, 4),
             (5, 5),
             (3, 7),
             (5, 7),
             (7, 7),
             (9, 7),
             (11, 7),
             (5, 9),
             (8, 9),
             (11, 9),
             (9, 11),
             (11, 11),
             (7, 13),
             (9, 13),
             (7, 15), ]

    faces = [(0, 1, 3),
             (0, 6, 2),
             (0, 3, 6),
             (1, 4, 3),
             (1, 5, 4),
             (2, 6, 7),
             (6, 8, 7),
             (7, 8, 12),
             (6, 9, 8),
             (8, 9, 12),
             (9, 13, 12),
             (4, 5, 11),
             (4, 11, 10),
             (9, 10, 13),
             (10, 11, 14),
             (10, 14, 13),
             (13, 14, 15),
             (14, 16, 15),
             (15, 16, 18),
             (15, 18, 17),
             (17, 18, 19), ]

    # We may want to use this later to define just the outer boundary.
    boundaries = [(0, 1),
                  (1, 5),
                  (5, 11),
                  (11, 14),
                  (14, 16),
                  (16, 18),
                  (18, 19),
                  (19, 17),
                  (17, 15),
                  (15, 13),
                  (13, 12),
                  (12, 7),
                  (7, 2),
                  (2, 0),
                  (3, 4),
                  (4, 10),
                  (10, 9),
                  (9, 6),
                  (6, 3), ]

    grid = ugrid.UGrid(nodes, faces, boundaries=boundaries)
    grid.build_edges()
    return grid


@contextlib.contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)
