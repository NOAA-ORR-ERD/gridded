#!usr/bin/env python

from __future__ import print_function, division, unicode_literals

import gridded
from gridded.pyugrid.ugrid import UGrid


"""
Test code for reading 3dm ascii format
"""


def read_3dm(filename):
    """
    read a 3dm file, and return a gridded.Dataset
    """

    grid = UGrid()
    with open(filename) as infile:
        if infile.readline().strip() != "MESH2D":
            raise ValueError('This does not looke like a "MESH2D" 3dm file')
        key, mesh_name = infile.readline().split()
        if key != "MESHNAME":
            raise ValueError('I expected "MESHNAME" on teh seconds line')
        mesh_name = mesh_name.replace('"', '')

        for line in infile:
            parts = line.split()
            print(parts)


def test_read():
    ds = read_3dm("SanDiego.3dm")

