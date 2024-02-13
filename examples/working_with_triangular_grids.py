"""
Example of using gridded UGRID interpolation,
including building the UGRID object from scratch

This example uses the Python triangle library to do the delauney
triangulation

Available on conda-forge:  with conda-forge enabled
"""

from gridded.grids import UGrid
from gridded.plotting.mpl_plotting import plot_ugrid

from matplotlib.tri import Triangulation
from matplotlib.collections import  LineCollection
import matplotlib.pyplot as plt

import triangle


def plot_ugrid(axes, grid, node_numbers=False, edge_numbers=False):
    """
    plot a UGRID in the provided MPL axes
    """

    nodes = grid.nodes
    # plot triangles
    mpl_tri = Triangulation(nodes[:, 0], nodes[:, 1], grid.faces)
    axes.triplot(mpl_tri)

    # plot nodes
    axes.plot(nodes[:, 0], nodes[:, 1], 'o')
    if node_numbers:
        for i, point in enumerate(nodes):
            axes.annotate(f"{i}", point,
                          xytext=(2, 2),
                          textcoords='offset points',
                          bbox = {'facecolor': 'white', 'alpha': 1.0, 'boxstyle': "round,pad=0.0", 'ec': 'white'},
                          )

    # plot boundaries
    if grid.boundaries is not None:
        bounds = grid.boundaries
        lines = []
        for bound in bounds:
            print(bound)
            lines.append([nodes[bound[0]], nodes[bound[1]]])
        print(lines)
        lc = LineCollection(lines, linewidths=3, colors=(1, 0, 0, 1))
        # lc = LineCollection(lines), colors=c, linewidths=2)
        axes.add_collection(lc)



# Sample Grid
# (This is the same as what's in gridded.tests.utilities,
#  but is explicitly here, so that the data structures required are clear

# the nodes are an Nx2 array of (x, y) (lon, lat) points
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


# This defines the outer boundary -- needed for constrained delaunay
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

# triangulate the nodes with py-triangle

# if there are holes in the triangulation
# you need to put a point in each hole.

holes = [[8, 6]]
tris = triangle.triangulate({'vertices': nodes,
                             'segments': boundaries,
                             'holes': holes
                             }, opts='p')


# Create a UGRID with the nodes and faces (triangles)
grid = UGrid(nodes=nodes, faces=tris['triangles'])

# find the boundaries (could have specified from above)
grid.build_boundaries()

fig, axes = plt.subplots(1,)
fig.set_size_inches((6, 6))

plot_ugrid(axes, grid, node_numbers=True)

fig.show()

i = input("enter to quit")

# grid = ugrid.UGrid(nodes, faces, boundaries=boundaries)
# grid.build_edges()





