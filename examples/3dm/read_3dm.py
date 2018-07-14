#!usr/bin/env python

from __future__ import print_function, division, unicode_literals

from gridded import Dataset
from gridded.pyugrid.ugrid import UGrid
from gridded.variable import Variable


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
            raise ValueError('This does not look like a "MESH2D" 3dm file')
        key, mesh_name = infile.readline().split()
        if key != "MESHNAME":
            raise ValueError('I expected "MESHNAME" on the second line')
        mesh_name = mesh_name.replace('"', '')

        # read the connectivity
        faces = []
        nodes = []
        depths = []

        for line in infile:
            # print("processing line", line)
            parts = line.split()
            if parts[0] == "E3T":
                faces.append(E3Tline(line))
            elif parts[0] == "ND":
                node, depth = NDline(line)
                nodes.append(node)
                depths.append(depth)
            else:
                print("not a line I know what to do with")
                # Could it be anything else??
                pass
        grid = UGrid(mesh_name=mesh_name,
                     nodes=nodes,
                     faces=faces)

        depth_var = Variable(name='depth',
                             units='foot',
                             data=depths,
                             grid=grid,
                             dataset=None,
                             varname='depth',
                             attributes=None,
                             )
        ds = Dataset(grid=grid, variables={"depth", depth_var})
        return ds


def E3Tline(line):
    parts = line.split()
    if parts[0] != "E3T":
        raise ValueError("not an E3T line")
    parts = [int(p) for p in parts[1:]]
    face = [p - 1 for p in parts[1:4]]
    return face


def NDline(line):
    parts = line.split()
    if parts[0] != "ND":
        raise ValueError("not an ND line")
    parts = [float(p) for p in parts[1:]]
    node = parts[:2]
    depth = parts[3]
    return (node, depth)


def test_read():
    ds = read_3dm("SanDiego.3dm")
    gr = ds.grid

    print(gr.nodes[:10])
    assert len(gr.nodes) == 9140
    assert len(gr.faces) == 16869
    assert False

