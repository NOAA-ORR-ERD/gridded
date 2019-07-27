"""
testing verdat read/write capability
"""

import os
import numpy as np
import random

import gridded
from gridded import io

test_filename = os.path.join(os.path.split(__file__)[0], "example_verdat.verdat")
test_filename_no_units = os.path.join(os.path.split(__file__)[0], "example_verdat_no_units.verdat")
test_filename_tiny = os.path.join(os.path.split(__file__)[0], "tiny.verdat")


def test_read():
    """
    at least it does something
    """
    ds = io.load_verdat(test_filename)

    assert isinstance(ds, gridded.Dataset)


def test_read_no_units():
    ds = io.load_verdat(test_filename_no_units)

    assert ds.variables['depth'].units == ""


def test_read_tiny():
    """
    a small example, so we can realy be sure
    """
    ds = io.load_verdat(test_filename_tiny)

    assert isinstance(ds, gridded.Dataset)

    nodes = ds.grid.nodes
    assert len(nodes) == 11
    assert np.array_equal(nodes[:3], [(-62.242001, 12.775000),
                                      (-28.990000, 12.775000),
                                      (-28.990000, 30.645000),
                                      ])

    assert np.array_equal(nodes[-2:], [(-50.821999, 20.202999),
                                       (-34.911236, 29.293791),
                                       ])

    bounds = ds.grid.boundaries
    assert len(bounds) == 9
    print(bounds)
    assert tuple(bounds[0]) == (0, 1)
    assert tuple(bounds[1]) == (1, 2)
    assert tuple(bounds[2]) == (2, 3)
    assert tuple(bounds[3]) == (3, 4)
    assert tuple(bounds[4]) == (4, 0)

    assert tuple(bounds[5]) == (5, 6)
    assert tuple(bounds[6]) == (6, 7)
    assert tuple(bounds[7]) == (7, 8)
    assert tuple(bounds[8]) == (8, 5)

    depths = ds['depth'].data

    assert np.array_equal(depths, [1.0,
                                   1.0,
                                   1.0,
                                   102.0,
                                   1.0,
                                   1.0,
                                   60.0,
                                   1.0,
                                   1.0,
                                   97.0,
                                   1.0,
                                    ])


def test_save_verdat():
    ds = io.load_verdat(test_filename_tiny)

    outfilename = "test_out.verdat"
    io.save_verdat(ds, outfilename)

    assert False


def test_order_boundary_segments():
    """
    tests that we can find the order of the boundary segments
    """
    # bounds from tiny verdat, randomized
    boundaries = np.array([[8, 5], [7, 8], [6, 7], [0, 1], [4, 0], [1, 2], [5, 6], [3, 4], [2, 3]])

    closed_bounds, open_bounds = io.verdat.order_boundary_segments(boundaries)

    assert not open_bounds

    assert len(closed_bounds) == 2

    # check the bounds are exactly correct
    closed_bounds.sort()

    assert sorted(closed_bounds[0]) == [0, 1, 2, 3, 4]
    assert sorted(closed_bounds[1]) == [5, 6, 7, 8]


def test_order_boundary_segments_open():
    """
    tests that we can find the order of the boundary segments

    and it will find an open boundary
    """
    # bounds from tiny verdat, randomized
    boundaries = np.array([[8, 5], [7, 8], [6, 7], [0, 1], [4, 0], [1, 2], [5, 6], [3, 4]])

    closed_bounds, open_bounds = io.verdat.order_boundary_segments(boundaries)

    assert len(closed_bounds) == 1
    assert len(open_bounds) == 1

    # check the bounds are exactly correct
    closed_bounds.sort()
    open_bounds.sort()
    print(closed_bounds)
    print(open_bounds)

    assert sorted(open_bounds[0]) == [0, 1, 2, 3, 4]
    assert sorted(closed_bounds[0]) == [5, 6, 7, 8]


def test_order_boundary_segments_none():
    boundaries = np.array([])
    closed_bounds, open_bounds = io.verdat.order_boundary_segments(boundaries)

    assert len(closed_bounds) == 0
    assert len(open_bounds) == 0




if __name__ == "__main__":
  test_order_boundary_segments_open()




