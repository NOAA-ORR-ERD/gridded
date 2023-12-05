"""
amazingly, there were no tests for sgrid itself ..

now there are almost none
"""

from gridded.pysgrid.sgrid import SGrid, load_grid
from .write_nc_test_files import roms_sgrid


def test_eq_same(roms_sgrid):
    grid1 = load_grid(roms_sgrid)
    grid2 = load_grid(roms_sgrid)

    assert grid1 == grid2


def test_eq_diff_type(roms_sgrid):
    """
    Just to make sure it doesn't crash
    """
    grid1 = load_grid(roms_sgrid)

    assert grid1 != "a string"

