"""
testing verdat read/write capability
"""

import os
from pathlib import Path
import random

import numpy as np

import pooch

import gridded
from gridded import io

from ..utilities import data_file_cache

DATA_URL = "https://gnome.orr.noaa.gov/py_gnome_testdata/gridded_test_files/"

HERE = Path(__file__).parent
EXAMPLES = HERE / "example_files"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

test_filename = EXAMPLES / "example_verdat.verdat"
test_filename_no_units = EXAMPLES /  "example_verdat_no_units.verdat"
test_filename_tiny = EXAMPLES /  "tiny.verdat"


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
    infilename = EXAMPLES /  "tiny.verdat"
    ds = io.load_verdat(infilename)

    outfilename = OUTPUT / "tiny_out.verdat"

    outfilename.unlink(missing_ok=True)

    io.save_verdat(ds, outfilename)

    assert outfilename.is_file()

    # Check at least a little bit if it's a valid verdat
    orig_contents = open(infilename).readlines()
    contents = open(outfilename).readlines()
    for l1, l2 in zip(orig_contents, contents):
        norm1 = [s.strip() for s in l1.strip().split(",")]
        norm2 = [s.strip() for s in l2.strip().split(",")]
        print()
        print(norm1)
        print(norm2)

        assert norm1 == norm2


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


def test_general_ugrid_to_verdat_no_depth():
    """
    Loads a regular old UGRID netCDF file, and saves it to verdat
    """
    ugrid_file = data_file_cache.fetch("SSCOFS.ugrid.nc")
    ds = gridded.Dataset.from_netCDF(ugrid_file)

    outfile = OUTPUT / "SSCOFS.verdat"
    outfile.unlink(missing_ok=True)

    io.save_verdat(ds, outfile, depth_var=None)

    assert outfile.is_file()

    contents = open(outfile).readlines()

    assert contents[0] == "DOGS \n"


    assert contents[-1] == "190\n"
    assert contents[-2] == "1\n"


if __name__ == "__main__":
  test_order_boundary_segments_open()




