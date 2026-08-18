"""
Matplotlib-based plotting utilities for gridded data.

This module provides interactive plotting tools built on top of Matplotlib
specifically tailored for gridded mesh visualisations.

Functions
---------
fvcom_inspector(nc_file)
    Launches an interactive graphical display of a horizontal FVCOM mesh,
    displaying node and centroid indices of selected faces and saving selected
    node details to a CSV file.

Examples
--------
>>> fvcom_inspector("fvcom_input_file.nc")
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
import matplotlib.pyplot as plt
import shapely.geometry as sgeom

from ..depth import FVCOM_Depth

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
        face_lon, face_lat = grid.face_coordinates[:, 0], grid.face_coordinates[:, 1]
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

# Below code was generated by Gemini with
# review, co-developement, revisions and changes by Rachael Mueller  
class GridGeoGenerator:
    """Helper class to parse and extract geometry from FVCOM grids."""

    DEFAULT_LINE_STYLE = {
        'color': 'black',
        'alpha': 0.5,
        'linewidth': 0.5,
    }

    def __init__(self, filename, crs=None):
        """
        Instantiate by loading a FVCOM Grid Current object from 
        NetCDF file and initializing spatial indexing.

        :param filename: Path to the FVCOM NetCDF file.
        :type filename: str or pathlib.Path
        """
        # Load Gridded Current Object
        self.filename = pathlib.Path(filename)
        self.grid_current = FVCOM_Depth.from_netCDF(str(self.filename))
        self.grid_obj = self.grid_current.grid

        if hasattr(self.grid_obj, 'build_spatial_tree'):
            self.grid_obj.build_spatial_tree()

    def get_triangle_faces(self):
        """
        Extract triangle face connectivity indices from the grid object
        and normalize them to 0-based Python indexing.

        :return: Array of triangle vertex node indices with shape
                 (N_triangles, 3).
        :rtype: numpy.ndarray
        :raises AttributeError: If triangle topology ('faces' or 'nv')
                                cannot be found on the grid object.
        """
        grid = self.grid_obj

        if hasattr(grid, 'faces') and grid.faces is not None:
            faces = np.asarray(grid.faces)
        elif hasattr(grid, 'nv'):
            node_vertices = np.asarray(grid.nv)
            # Transpose array from (3, N_triangles) to (N_triangles, 3)
            faces = (
                node_vertices.T if (
                    node_vertices.shape[0] == 3 and 
                    node_vertices.shape[1] != 3
                ) 
                else node_vertices
            )
            if faces.min() == 1:
                faces = faces - 1
        else:
            raise AttributeError(
                "Could not locate triangle topology ('faces'/'nv')."
            )

        return faces

    def draw_grid_lines(self, ax, appearance=None):
        """
        Draw the unstructured mesh wireframe directly onto Matplotlib
        axes using matplotlib.tri.Triangulation.

        :param ax: Target Matplotlib axes.
        :type ax: matplotlib.axes.Axes
        :param appearance: Line styling kwargs (color, alpha, linewidth).
        :type appearance: dict, optional
        :return: The Matplotlib LineCollection added to the plot.
        """
        faces = self.get_triangle_faces()

        # Normalize longitudes to -180..180
        lons = np.asarray(self.grid_obj.node_lon)
        lons = np.where(lons > 180, lons - 360, lons)
        lats = np.asarray(self.grid_obj.node_lat)

        # Build native Matplotlib triangulation
        triangulation = Triangulation(lons, lats, faces)

        style = appearance or self.DEFAULT_LINE_STYLE
        return ax.triplot(triangulation, **style)

    def gen_grid_lines_U(self):
        """
        Generate Shapely geometries representing unique mesh wireframe
        edges for unstructured triangular grids.

        Extracts element node connectivity, normalizes longitudes to
        the [-180, 180] degree display range, filters out duplicate
        shared edges, and constructs a MultiLineString object.

        :return: MultiLineString object containing all unique mesh
                 edges for plotting.
        :rtype: shapely.geometry.MultiLineString
        """
        faces = self.get_triangle_faces()

        # Enforce longitude display range (-180 to 180 degrees)
        lons = np.asarray(self.grid_obj.node_lon)
        lons = np.where(lons > 180, lons - 360, lons)
        lats = np.asarray(self.grid_obj.node_lat)

        # Extract 3 triangle edges: (0->1), (1->2), (2->0)
        edge_pairs = np.vstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]]
        ])

        # Filter unique edges to prevent double-drawing shared lines
        edge_pairs.sort(axis=1)
        unique_edges = np.unique(edge_pairs, axis=0)

        lines = [
            sgeom.LineString(
                [(lons[p1], lats[p1]), (lons[p2], lats[p2])]
            )
            for p1, p2 in unique_edges
        ]
        return sgeom.MultiLineString(lines)

    def get_max_extent(self, margin=0.05):
        """
        Calculates geographic bounding box [w, e, s, n].

        :param margin: Percent padding buffer around extent.
        :type margin: float
        :return: Bounding extent list [lon_min, lon_max, lat_min, lat_max].
        :rtype: list
        """
        lons = np.where(
            self.grid_obj.node_lon > 180,
            self.grid_obj.node_lon - 360,
            self.grid_obj.node_lon
        )
        lats = self.grid_obj.node_lat

        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()

        lon_margin = (lon_max - lon_min) * margin
        lat_margin = (lat_max - lat_min) * margin

        return [
            lon_min - lon_margin,
            lon_max + lon_margin,
            lat_min - lat_margin,
            lat_max + lat_margin
        ]

    def fvcom_index_query(self, position):
        """
        Finds cell element index (nele) and Fortran node IDs.

        :param position: Clicked (longitude, latitude) tuple.
        :type position: tuple
        :return: Dictionary containing face index, node IDs, and coords.
        :rtype: dict or None
        """
        grid = self.grid_obj
        click_lon, click_lat = position

        # Handle 0..360 degree longitude storage in NetCDF
        search_lon = click_lon
        if grid.node_lon.min() >= 0 and click_lon < 0:
            search_lon = click_lon + 360

        # Query face index via PyGNOME point-in-polygon lookup
        nele_idx = grid.locate_faces(
            (search_lon, click_lat),
            _memo=False,
            _copy=False,
            _hash=False
        )

        # Fallback to triangle search
        if nele_idx is None or np.all(nele_idx == -1):
            if hasattr(grid, 'locate_triangles'):
                nele_idx = grid.locate_triangles(
                    (search_lon, click_lat)
                )

        # Fallback to nearest face center
        if nele_idx is None or np.all(nele_idx == -1):
            if hasattr(grid, 'center_lon') and (
                grid.center_lon is not None
            ):
                center_lons = np.where(
                    grid.center_lon > 180,
                    grid.center_lon - 360,
                    grid.center_lon
                )
                dists = (
                    (center_lons - click_lon) ** 2 +
                    (grid.center_lat - click_lat) ** 2
                )
                nele_idx = np.argmin(dists)
            else:
                return None

        nele = int(np.squeeze(nele_idx))

        # Retrieve 3 Fortran node indices for the selected element
        faces = self.get_triangle_faces()
        nodes = faces[nele]

        # Extract node coordinates and normalize longitudes to -180..180
        raw_lons = grid.node_lon[nodes]
        node_lons = np.where(
            raw_lons > 180, raw_lons - 360, raw_lons
        )
        node_lats = grid.node_lat[nodes]

        return {
            'nele': nele,
            'nodes': nodes,
            'node_lons': node_lons,
            'node_lats': node_lats,
            'click_pos': (click_lon, click_lat)
        }


class PureMatplotlibFVCOMViewer:
    """Interactive Matplotlib GUI viewer for inspecting FVCOM grid."""

    def __init__(self, filename, spill_location=None):
        self.filename = pathlib.Path(filename)
        self.grid_helper = GridGeoGenerator(self.filename)

        # Initialize and output CSV to store results of click query
        self.csv_path = self.filename.parent / (
            f"{self.filename.stem}_selected_cell.csv"
        )
        self.init_csv_file()
        
        # Build figure and standard 2D plot axis
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title(
            f"FVCOM Inspector - {self.filename.name}"
        )

        extent = self.grid_helper.get_max_extent()
        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self.ax.set_aspect('equal')  # Preserve 1:1 geographic aspect ratio

        self.ax.set_xlabel('Longitude (°E)')
        self.ax.set_ylabel('Latitude (°N)')
        self.ax.set_title(
            "Click anywhere inside a grid cell to select",
            fontsize=11,
            fontweight='bold'
        )

        print("Rendering mesh wireframe...")
        self.grid_helper.draw_grid_lines(self.ax)

        if spill_location:
            self.ax.scatter(
                spill_location[0], spill_location[1],
                s=120, color='magenta', marker='*', zorder=6
            )

        self.scatter_overlay = None
        self.triangle_overlay = None
        self.annotation_list = []

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        #self.fig.canvas.mpl_connect("close_event", self.on_window_close)

    def init_csv_file(self):
        """Creates or overwrites the CSV log with headers on start."""
        headers = pd.DataFrame(
            columns=[
                "nele",
                "nele_lat",
                "nele_lon",
                "node1",
                "node1_lat",
                "node1_lon",
                "node2",
                "node2_lat",
                "node2_lon",
                "node3",
                "node3_lat",
                "node3_lon",
            ]
        )
        # clear out old sessions and writes fresh headers once
        headers.to_csv(self.csv_path, mode="w", index=False)
        print(f"[+] Initialized click log: {self.csv_path}")
        
    def on_click(self, event):
        """Handles mouse single clicks on the canvas."""
        if (
            event.button == 1 and
            event.xdata is not None and
            event.ydata is not None
        ):
            click_lon, click_lat = event.xdata, event.ydata

            cell_info = self.grid_helper.fvcom_index_query(
                (click_lon, click_lat)
            )
            if cell_info:
                self.draw_query_result(cell_info)
            else:
                print(
                    f"[-] Click at Lon: {click_lon:.4f}, "
                    f"Lat: {click_lat:.4f} is outside grid."
                )

    def draw_query_result(self, cell_info):
        """Draws selected cell highlights and prints details."""
        # Clear previous highlights
        if self.scatter_overlay:
            self.scatter_overlay.remove()
        if self.triangle_overlay:
            self.triangle_overlay.remove()
        for annotation in self.annotation_list:
            annotation.remove()
        self.annotation_list.clear()

        # Print cell details to terminal
        print("\n" + "=" * 50)
        print(" SELECTED CELL DETAILS")
        print(f"  Face Index (nele) : {cell_info['nele']}")
        print(
            f"  Node Indices      : "
            f"{cell_info['nodes'].tolist()}"
        )
        print(
            f"  Clicked Coordinate: Lon = "
            f"{cell_info['click_pos'][0]:.5f}, "
            f"Lat = {cell_info['click_pos'][1]:.5f}"
        )
        print("=" * 50)

        node_x = cell_info['node_lons']
        node_y = cell_info['node_lats']
        click_x, click_y = cell_info['click_pos']

        # Highlight cell triangle patch
        triangle_coords = np.column_stack([node_x, node_y])
        self.triangle_overlay = plt.Polygon(
            triangle_coords,
            facecolor='mediumaquamarine',
            alpha=0.6,
            edgecolor='teal',
            lw=1.5,
            zorder=4
        )
        self.ax.add_patch(self.triangle_overlay)

        # Highlight corner nodes
        self.scatter_overlay = self.ax.scatter(
            node_x, node_y, s=80, color='olive', marker='o', zorder=3
        )

        # Calculate cell centroid
        center_x, center_y = np.mean(node_x), np.mean(node_y)

        # Create information for storing to file the query results
        # Extract 3 node IDs
        node1, node2, node3 = cell_info['nodes'].tolist()
        # Build DataFrame row
        click_info = pd.DataFrame(
            [
                {
                    'nele': cell_info['nele'],
                    'nele_lat': center_y,
                    'nele_lon': center_x,
                    'node1': node1,
                    'node1_lat': node_y[0],
                    'node1_lon': node_x[0],
                    'node2': node2,
                    'node2_lat': node_y[1],
                    'node2_lon': node_x[1],
                    'node3': node3,
                    'node3_lat': node_y[2],
                    'node3_lon': node_x[2],
                }
            ]
        )
        # Save or append to CSV file
        click_info.to_csv(
            self.csv_path, header = False, mode = "a", index=False)
        print(f"[+] Saved cell info to {self.csv_path}")

        # Draw node badges
        for i in range(3):
            # Calculate outward direction vector
            dir_x = node_x[i] - center_x
            dir_y = node_y[i] - center_y
            norm = np.hypot(dir_x, dir_y) + 1e-6

            # Push 25 points outward along vector
            offset_x = (dir_x / norm) * 25
            offset_y = (dir_y / norm) * 25

            badge = self.ax.annotate(
                f"Node: {cell_info['nodes'][i]}",
                (node_x[i], node_y[i]),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                fontweight='bold',
                bbox={
                    "facecolor": "lemonchiffon",
                    "alpha": 0.9,
                    "pad": 2.0,
                    "boxstyle": "round, pad=.5",
                    "edgecolor":"olive"
                },
                zorder = 5
            )
            self.annotation_list.append(badge)

        # Draw cell summary callout box
        summary_text = (
            f"Face (nele): {cell_info['nele']}\n"
            f"Nodes: {cell_info['nodes'].tolist()}"
        )
        callout = self.ax.annotate(
            summary_text,
            (click_x, click_y),
            xytext=(40, -40),
            textcoords="offset points",
            fontsize=9,
            fontweight='bold',
            bbox={
                "facecolor": "teal",
                "alpha": 0.5,
                "pad": 1,
                "boxstyle": "round",
                "edgecolor":"darkcyan"
            },
            zorder = 5,
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc3,rad=.2",
                color="teal",
                lw=1.5,
                zorder=4
            )
        )
        self.annotation_list.append(callout)

        # Marker at center of face
        self.ax.scatter(
            center_x, center_y,
            s=20,
            color="teal",
            edgecolors="teal",
            zorder=5
        )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()


def fvcom_inspector(filename, spill_location=None, backend=None):
    """
    Launches interactive FVCOM grid inspector GUI.

    :param filename: Path to FVCOM NetCDF grid file.
    :param spill_location: Optional (lon, lat) tuple.
    :param backend: Optional explicit Matplotlib backend for the 
           underlying engine responsible for taking plot elements 
           (lines, shapes, text) and rendering them onto an actual 
           plot destination. 'TkAgg'(Tkinter) is Python's is Python's
           built-in GUI library.  Other options include: QtAgg, WXAgg, 
           and MacOSX. 
    """
    if backend:
        matplotlib.use(backend)
    elif matplotlib.get_backend().lower() == 'agg':
        # Use a specific rendering backend to draw graphics and handle user input.
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            pass

    viewer = PureMatplotlibFVCOMViewer(
        filename, spill_location=spill_location
    )
    viewer.show()
