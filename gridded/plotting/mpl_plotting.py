"""
Some MPL based plotting utilities for gridded.
"""

from matplotlib.tri import Triangulation
from matplotlib.collections import  LineCollection


def plot_ugrid(axes,
               grid,
               node_numbers=False,
               face_numbers=False):
    """
    plot a UGRID in the provided MPL axes

    :param axes: an MPL axes object to plot on

    :param grid: a gridded.Ugrid grid object to plot

    :param node_numbers=False: If True, plot the node numbers

    :param face_numbers=False: If True, plot the face numbers

    """

    nodes = grid.nodes
    # plot triangles (faces)
    mpl_tri = Triangulation(nodes[:, 0], nodes[:, 1], grid.faces)
    axes.triplot(mpl_tri)
    if face_numbers:
        if grid.face_coordinates is None:
            grid.build_face_coordinates()
        face_coords = grid.face_coordinates
        for i, point in enumerate(face_coords):
            axes.annotate(f"{i}", point,
                          xytext=(0, 0),
                          textcoords='offset points',
                          horizontalalignment='center',
                          verticalalignment='center',
                          bbox = {'facecolor': 'white', 'alpha': 1.0, 'boxstyle': "round,pad=0.0", 'ec': 'white'},
                          )

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
            lines.append([nodes[bound[0]], nodes[bound[1]]])
        lc = LineCollection(lines, linewidths=3, colors=(1, 0, 0, 1))
        # lc = LineCollection(lines), colors=c, linewidths=2)
        axes.add_collection(lc)
