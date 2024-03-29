"""
Example of using gridded UGRID interpolation,
including building the UGRID object from scratch

This example uses the Python triangle library to do the delauney
triangulation

Available on conda-forge:  with conda-forge enabled
"""

import numpy as np

import gridded
from gridded.grids import UGrid
from gridded.plotting.mpl_plotting import plot_ugrid

from matplotlib.tri import Triangulation
from matplotlib.collections import  LineCollection
import matplotlib.pyplot as plt

import triangle


# if the built in one doesn't work for you.
# requires a recent version of gridded
# you can also customize it here if you want.
# def plot_ugrid(axes,
#                grid,
#                node_numbers=False,
#                face_numbers=False):
#     """
#     plot a UGRID in the provided MPL axes

#     :param axes: an MPL axes object to plot on

#     :param grid: a gridded.Ugrid grid object to plot

#     :param node_numbers=False: If True, plot the node numbers

#     :param face_numbers=False: If True, plot the face numbers

#     """

#     nodes = grid.nodes
#     # plot triangles (faces)
#     mpl_tri = Triangulation(nodes[:, 0], nodes[:, 1], grid.faces)
#     axes.triplot(mpl_tri)
#     if face_numbers:
#         if grid.face_coordinates is None:
#             grid.build_face_coordinates()
#         face_coords = grid.face_coordinates
#         for i, point in enumerate(face_coords):
#             axes.annotate(f"{i}", point,
#                           xytext=(0, 0),
#                           textcoords='offset points',
#                           horizontalalignment='center',
#                           verticalalignment='center',
#                           bbox = {'facecolor': 'white', 'alpha': 1.0, 'boxstyle': "round,pad=0.0", 'ec': 'white'},
#                           )

#     # plot nodes
#     axes.plot(nodes[:, 0], nodes[:, 1], 'o')
#     if node_numbers:
#         for i, point in enumerate(nodes):
#             axes.annotate(f"{i}", point,
#                           xytext=(2, 2),
#                           textcoords='offset points',
#                           bbox = {'facecolor': 'white', 'alpha': 1.0, 'boxstyle': "round,pad=0.0", 'ec': 'white'},
#                           )

#     # plot boundaries
#     if grid.boundaries is not None:
#         bounds = grid.boundaries
#         lines = []
#         for bound in bounds:
#             lines.append([nodes[bound[0]], nodes[bound[1]]])
#         lc = LineCollection(lines, linewidths=3, colors=(1, 0, 0, 1))
#         # lc = LineCollection(lines), colors=c, linewidths=2)
#         axes.add_collection(lc)



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
# in this case, there is one hole

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

plot_ugrid(axes, grid, node_numbers=True, face_numbers=True)

fig.show()

# create a scalar variable "on the nodes"
# made up -- bivariate cosine
min_x, min_y = grid.nodes.min(axis=0)
max_x, max_y = grid.nodes.max(axis=0)


node_data = np.cos(grid.nodes[:,0] / 3 - min_x) + np.cos(grid.nodes[:,1] / 5 - min_y)


node_var = gridded.Variable(name='sample data on nodes',
                            units=None,
                            data=node_data,
                            grid=grid,
                            location='node',
                            )

# Interpolate to a cross section
x = np.linspace(min_x, max_x, 100)
y = np.linspace(min_y, max_y, 100)

points = np.c_[x, y]

# at() always returns a N,m array
val = node_var.at(points)[:,0]

fig, axes = plt.subplots(1,)
fig.set_size_inches((6, 6))

axes.plot(x, val)

fig.show()

# locating what node you want:
node_num = grid.locate_nodes((5, 8))

# data at that node:

node_var.data[node_num]


# for data "on the faces":
# first we need the centroids of the faces -- i.e. the face_coordinates
grid.build_face_coordinates()

# make some fake data
face_data = (np.cos(grid.face_coordinates[:, 0] / 3 - min_x)
             + np.cos(grid.face_coordinates[:, 1] / 5 - min_y))


# create a variable for that data
node_var = gridded.Variable(name="sample data on faces",
                            units=None,
                            data=face_data,
                            grid=grid,
                            location='face'
                            )


# locating what cell you want:
face_num = grid.locate_faces((5,8))

# at() always returns a N,m array
val = node_var.at(points)[:,0]

fig, axes = plt.subplots(1,)
fig.set_size_inches((6, 6))

axes.plot(x, val)

fig.show()





