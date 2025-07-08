"""
Some MPL based plotting utilities for gridded.
"""

import numpy as np
from matplotlib.tri import Triangulation
from matplotlib.collections import  LineCollection



def plot_ugrid(axes, grid, nodes=False, node_numbers=False, face_numbers=False):
    """
    Plot a UGRID in the provided MPL axes.

    Note: this doesn't plot data on the grid, just the grid itself

    :param axes: an MPL axes object to plot on

    :param grid: an gridded UGrid object.

    :param nodes: If True, plot the nodes as dots

    :param node_numbers=False: If True, plot the node numbers

    :param face_numbers=False: If True, plot the face numbers
    """

    nodes_lon, nodes_lat = grid.node_lon, grid.node_lat
    faces = grid.faces

    if faces.shape[0] == 3:
        # swap order for mpl triangulation
        faces = faces.T

    mpl_tri = Triangulation(nodes_lon, nodes_lat, faces)

    axes.triplot(mpl_tri)
    if face_numbers:
        if grid.face_coordinates is None:
            grid.build_face_coordinates()
        face_lon, face_lat = grid.face_coordinates[:,0], grid.face_coordinates[:,1]
        for i, point in enumerate(zip(face_lon, face_lat)):
            axes.annotate(
                f"{i}",
                point,
                xytext=(0, 0),
                textcoords="offset points",
                horizontalalignment="center",
                verticalalignment="center",
                bbox={
                    "facecolor": "white",
                    "alpha": 1.0,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
            )

    # plot nodes
    if nodes:
        axes.plot(nodes_lon, nodes_lat, "o")
    # plot node numbers
    if node_numbers:
        for i, point in enumerate(zip(nodes_lon, nodes_lat)):
            axes.annotate(
                f"{i}",
                point,
                xytext=(2, 2),
                textcoords="offset points",
                bbox={
                    "facecolor": "white",
                    "alpha": 1.0,
                    "boxstyle": "round,pad=0.0",
                    "ec": "white",
                },
            )

    # boundaries -- if they are there.
    if grid.boundaries is not None:
        bounds = grid.boundaries
        lines = []
        for bound in bounds:
            line = (
                (nodes_lon[bound[0]], nodes_lat[bound[0]]),
                (nodes_lon[bound[1]], nodes_lat[bound[1]]),
            )
            lines.append(line)
        lc = LineCollection(lines, linewidths=2, colors=(1, 0, 0, 1))
        axes.add_collection(lc)


def plot_sgrid(axes, grid, nodes=False, rho_points=False, edge_points=False):
    """
    Plot a SGRID in the provided MPL axes.

    Note: this doesn't plot data on the grid, just the grid itself

    :param axes: an MPL axes object to plot on

    :param grid: an gridded.SGrid object.

    :param nodes: If True, plot the nodes as dots

    :param rho_points=False: If True, plot points in the center of the cells
                             (ROMS calls these the rho points)

    :param edge_points=False: If True, plot the points in the center of the edges
                              (where U and V are in ROMS)
    """

    nodes_lon, nodes_lat = np.asarray(grid.node_lon), np.asarray(grid.node_lat)

    # need to set the limits for linecollection
    axes.set_xlim(nodes_lon.min(), nodes_lon.max())
    axes.set_ylim(nodes_lat.min(), nodes_lat.max())

    # plot the grid
    lines = []
    for i in range(nodes_lon.shape[0]):
        line = np.c_[nodes_lon[i, :], nodes_lat[i, :]]
        lines.append(line)
    for j in range(nodes_lon.shape[1]):
        line = np.c_[nodes_lon[:, j], nodes_lat[:, j]]
        lines.append(line)
    lc = LineCollection(lines, linewidths=1, colors=(0, 0, 0, 1))
    axes.add_collection(lc)

    # # plot nodes
    if nodes:
        axes.plot(nodes_lon, nodes_lat, "ok")


    # from ugrid -- needs changes -- maybe (i, j)?
    # if face_numbers:
    #     try:
    #         face_lon, face_lat = (ds[n] for n in mesh_defs["face_coordinates"].split())
    #     except KeyError:
    #         raise ValueError('"face_coordinates" must be defined to plot the face numbers')
    #     for i, point in enumerate(zip(face_lon, face_lat)):
    #         axes.annotate(
    #             f"{i}",
    #             point,
    #             xytext=(0, 0),
    #             textcoords="offset points",
    #             horizontalalignment="center",
    #             verticalalignment="center",
    #             bbox={
    #                 "facecolor": "white",
    #                 "alpha": 1.0,
    #                 "boxstyle": "round,pad=0.0",
    #                 "ec": "white",
    #             },
    #         )

    # # plot node numbers
    # if node_numbers:
    #     for i, point in enumerate(zip(nodes_lon, nodes_lat)):
    #         axes.annotate(
    #             f"{i}",
    #             point,
    #             xytext=(2, 2),
    #             textcoords="offset points",
    #             bbox={
    #                 "facecolor": "white",
    #                 "alpha": 1.0,
    #                 "boxstyle": "round,pad=0.0",
    #                 "ec": "white",
    #             },
    #         )


