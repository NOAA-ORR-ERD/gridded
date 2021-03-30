#!/usr/bin/env python

"""
ugrid classes

set of classes for working with unstructured model grids

The "ugrid" class is the base class: it stores everything in memory

It can read from and write to netcdf files in the UGRID format.

It may be able to reference a netcdf file at some point, rather than storing
directly in memory.

NOTE: only tested for triangular and quad mesh grids at the moment.

"""

from __future__ import (absolute_import, division, print_function)

import hashlib
from itertools import chain
from collections import OrderedDict

import numpy as np
import netCDF4

import gridded.pyugrid.read_netcdf as read_netcdf
from gridded.pyugrid.util import point_in_tri

from gridded.utilities import get_writable_dataset

# from gridded.pyugrid.uvar import UVar

# __all__ = ['UGrid',
#            'UVar']


# datatype used for indexes -- might want to change for 64 bit some day.
IND_DT = np.int32
NODE_DT = np.float64  # datatype used for node coordinates.


class UGrid(object):
    """
    A basic class to hold an unstructured grid as defined in the UGrid convention.

    The internal structure mirrors the netcdf data standard.
    """

    def __init__(self,
                 nodes=None,
                 node_lon=None,
                 node_lat=None,
                 faces=None,
                 edges=None,
                 boundaries=None,
                 face_face_connectivity=None,
                 face_edge_connectivity=None,
                 edge_coordinates=None,
                 face_coordinates=None,
                 boundary_coordinates=None,
                 data=None,
                 mesh_name="mesh",
                 ):
        """
        ugrid class -- holds, saves, etc. an unstructured grid

        :param nodes=None : the coordinates of the nodes
        :type nodes: (NX2) array of floats

        :param faces=None : the faces of the grid. Indexes for the nodes array.
        :type faces: (NX3) array of integers

        :param edges=None : the edges of the grid. Indexes for the nodes array.
        :type edges: (NX2) array of integers

        :param boundaries=None: specification of the boundaries are usually a
                                subset of edges where boundary condition
                                information, etc is stored.
                                (NX2) integer array of indexes for the nodes
                                array.
        :type boundaries: numpy array of integers

        :param face_face_connectivity=None: connectivity arrays.
        :param face_edge_connectivity=None: connectivity arrays.

        :param edge_coordinates=None: representative coordinate of the edges.
        :param face_coordinates=None: representative coordinate of the faces.
        :param boundary_coordinates=None: representative coordinate of the
                                          boundaries.

        :param edge_coordinates=None: representative coordinate of the edges
        :type edge_coordinates: (NX2) array of floats

        :param face_coordinates=None: representative coordinate of the faces
                                      (NX2) float array
        :type face_coordinates: (NX2) array of floats


        :param boundary_coordinates=None: representative coordinate of the
                                          boundaries
        :type boundary_coordinates: (NX2) array of floats


        :param data = None: associated variables
        :type data: dict of UVar objects

        :param mesh_name = "mesh": optional name for the mesh
        :type mesh_name: string

        Often this is too much data to pass in as literals -- so usually
        specialized constructors will be used instead (load from file, etc).

        The index variables faces can be a masked array. The mask is
        used for so called flexible meshes. Flexible meshes contain
        cells with varying number of nodes per face.  See the flexible
        mesh section in the convention for further details.
        """

        if ((nodes is not None) and
            ((node_lon is not None) or
             (node_lat is not None))):
            raise TypeError("You need to provide a single nodes array "
                            "or node_lon and node_lat")
        if nodes is None:
            if node_lon is not None and node_lat is not None:
                nodes = np.ma.column_stack((node_lon, node_lat))
        self.nodes = nodes
        self.faces = faces
        self.edges = edges
        self.boundaries = boundaries

        self.face_face_connectivity = face_face_connectivity
        self.face_edge_connectivity = face_edge_connectivity

        self.edge_coordinates = edge_coordinates
        self.face_coordinates = face_coordinates
        self.boundary_coordinates = boundary_coordinates

        self.mesh_name = mesh_name

        # # the data associated with the grid
        # # should be a dict of UVar objects
        # self._data = {}  # The data associated with the grid.
        # if data is not None:
        #     for dataset in data.values():
        #         self.add_data(dataset)

        # A kdtree is used to locate nodes.
        # It will be created if/when it is needed.
        self._kdtree = None
        self._cell_tree = None
        self._ind_memo_dict = OrderedDict()
        self._alpha_memo_dict = OrderedDict()

    @classmethod
    def from_ncfile(klass, nc_url, mesh_name=None):  # , load_data=False):
        """
        create a UGrid object from a netcdf file name (or opendap url)

        :param nc_url: the filename or OpenDap url you want to load

        :param mesh_name=None: the name of the mesh you want. If None, then
                               you'll get the only mesh in the file. If there
                               is more than one mesh in the file, a ValueError
                               Will be raised
        """
        grid = klass()
        read_netcdf.load_grid_from_ncfilename(nc_url, grid, mesh_name)  # , load_data)
        return grid

    @classmethod
    def from_nc_dataset(klass, nc, mesh_name=None):  # , load_data=False):
        """
        create a UGrid object from a netcdf file (or opendap url)

        :param nc: An already open Dataset object
        :type nc: netCDF4.DataSet

        :param mesh_name=None: the name of the mesh you want. If None, then
                               you'll get the only mesh in the file. If there
                               is more than one mesh in the file, a ValueError
                               Will be raised

        # :param load_data=False: flag to indicate whether you want to load the
        #                         associated data or not.  The mesh will be
        #                         loaded in any case.  If False, only the mesh
        #                         will be loaded.  If True, then all the data
        #                         associated with the mesh will be loaded.
        #                         This could be huge!

        # :type load_data: boolean

        """
        grid = klass()
        read_netcdf.load_grid_from_nc_dataset(nc, grid, mesh_name)  # , load_data)
        return grid

    @property
    def info(self):
        """
        summary of information about the grid
        """
        msg = ["UGrid object:"]

        msg.append("Number of nodes: %i" % len(self.nodes))
        msg.append("Number of faces: %i with %i vertices per face" %
                   (len(self.faces), self.num_vertices))
        if self.boundaries is not None:
            msg.append("Number of boundaries: %i" % len(self.boundaries))

        # if self._data:
        #     msg.append("Variables: " + ", ".join([str(v) for v in self._data.keys()]))
        return "\n".join(msg)

    def check_consistent(self):
        """
        Check if the various data is consistent: the edges and faces reference
        existing nodes, etc.

        """
        raise NotImplementedError

    @property
    def num_vertices(self):
        """
        Maximum number of vertices in a face.

        """
        if self._faces is None:
            return None
        else:
            return self._faces.shape[1]

    @property
    def nodes(self):
        return self._nodes

    @property
    def node_lon(self):
        return self._nodes[:, 0]

    @property
    def node_lat(self):
        return self._nodes[:, 1]

    @nodes.setter
    def nodes(self, nodes_coords):
        # Room here to do consistency checking, etc.
        # For now -- simply make sure it's a numpy array.
        if nodes_coords is None:
            self.nodes = np.zeros((0, 2), dtype=NODE_DT)
        else:
            self._nodes = np.asanyarray(nodes_coords, dtype=NODE_DT)

    @nodes.deleter
    def nodes(self):
        # If there are no nodes, there can't be anything else.
        self._nodes = np.zeros((0, 2), dtype=NODE_DT)
        self._edges = None
        self._faces = None
        self._boundaries = None

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces_indexes):
        # Room here to do consistency checking, etc.
        # For now -- simply make sure it's a numpy array.
        if faces_indexes is not None:
            self._faces = np.asanyarray(faces_indexes, dtype=IND_DT)
        else:
            self._faces = None
            # Other things are no longer valid.
            self._face_face_connectivity = None
            self._face_edge_connectivity = None

    @faces.deleter
    def faces(self):
        self._faces = None
        self._faces = None
        # Other things are no longer valid.
        self._face_face_connectivity = None
        self._face_edge_connectivity = None
        self.edge_coordinates = None

    @property
    def edges(self):
        return self._edges

    @edges.setter
    def edges(self, edges_indexes):
        # Room here to do consistency checking, etc.
        # For now -- simply make sure it's a numpy array.
        if edges_indexes is not None:
            self._edges = np.asanyarray(edges_indexes, dtype=IND_DT)
        else:
            self._edges = None
            self._face_edge_connectivity = None

    @edges.deleter
    def edges(self):
        self._edges = None
        self._face_edge_connectivity = None
        self.edge_coordinates = None

    @property
    def boundaries(self):
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries_indexes):
        # Room here to do consistency checking, etc.
        # For now -- simply make sure it's a numpy array.
        if boundaries_indexes is not None:
            self._boundaries = np.asanyarray(boundaries_indexes, dtype=IND_DT)
        else:
            self._boundaries = None

    @boundaries.deleter
    def boundaries(self):
        self._boundaries = None
        self.boundary_coordinates = None

    @property
    def face_face_connectivity(self):
        return self._face_face_connectivity

    @face_face_connectivity.setter
    def face_face_connectivity(self, face_face_connectivity):
        # Add more checking?
        if face_face_connectivity is not None:
            face_face_connectivity = np.asanyarray(face_face_connectivity,
                                                   dtype=IND_DT)
            if face_face_connectivity.shape != (len(self.faces),
                                                self.num_vertices):
                msg = ("face_face_connectivity must be size "
                       "(num_faces, {})").format
                raise ValueError(msg(self.num_vertices))
        self._face_face_connectivity = face_face_connectivity

    @face_face_connectivity.deleter
    def face_face_connectivity(self):
        self._face_face_connectivity = None

    @property
    def face_edge_connectivity(self):
        return self._face_edge_connectivity

    @face_edge_connectivity.setter
    def face_edge_connectivity(self, face_edge_connectivity):
        # Add more checking?
        if face_edge_connectivity is not None:
            face_edge_connectivity = np.asanyarray(face_edge_connectivity,
                                                   dtype=IND_DT)
            if face_edge_connectivity.shape != (len(self.faces),
                                                self.num_vertices):
                msg = ("face_face_connectivity must be size "
                       "(num_face, {})").format
                raise ValueError(msg(self.num_vertices))
        self._face_edge_connectivity = face_edge_connectivity

    @face_edge_connectivity.deleter
    def face_edge_connectivity(self):
        self._face_edge_connectivity = None

    # @property
    # def data(self):
    #     """
    #     dict of data associated with the data arrays
    #     You can't set this -- must use UGrid.add_data().

    #     """
    #     return self._data

    def infer_location(self, data):
        """
        :param data:

        :returns: 'nodes' if data will fit to the nodes,
                  'faces' if the data will fit to the faces,
                  'boundaries' if the data will fit the boundaries.
                  None otherwise.

        If data is a netcdf variable, the "location" attribute is checked.
        """
        # We should never be calling infer_locations if it was already defined
        # try:
        #     loc = data.location
        #     if loc == "face":
        #         # FIXME: should we check the array size in this case?
        #         return "face"
        # except AttributeError:
        #     pass # try checking array size
        # # fixme: should use UGRID compliant nc_attributes if possible
        try:
            size = data.shape[-1]
        except IndexError:
            return None  # Variable has a size-zero data array
        if size == self.nodes.shape[0]:
            return 'node'
        if self.faces is not None and size == self.faces.shape[0]:
            return 'face'
        if self.boundaries is not None and size == self.boundaries.shape[0]:
            return 'boundary'
        return None

    # def add_data(self, uvar):
    #     """
    #     Add a UVar to the data dict

    #     :param uvar: the UVar object to add.
    #                  Its name will be the key in the data dict.
    #     :type uvar: a ugrid.UVar object

    #     Some sanity checking is done to make sure array sizes are correct.

    #     """
    #     # Size check:
    #     if uvar.location == 'node':
    #         if self.nodes is None:
    #             raise ValueError("adding data to nodes "
    #                              "but nodes are None")
    #         if len(uvar.data) != len(self.nodes):
    #             raise ValueError("length of data array must match "
    #                              "the number of nodes")
    #     elif uvar.location == 'edge':
    #         if self.edges is None:
    #             raise ValueError("adding data to edges "
    #                              "but edges are None")
    #         if len(uvar.data) != len(self.edges):
    #             raise ValueError("length of data array must match "
    #                              "the number of edges")
    #     elif uvar.location == 'face':
    #         if self.faces is None:
    #             raise ValueError("adding data to faces "
    #                              "but faces are None")
    #         if len(uvar.data) != len(self.faces):
    #             raise ValueError("length of data array must match "
    #                              "the number of faces")
    #     elif uvar.location == 'boundary':
    #         if self.boundaries is None:
    #             raise ValueError("adding data to boundaries "
    #                              "but boundaries are None")
    #         if len(uvar.data) != len(self.boundaries):
    #             raise ValueError("length of data array must match "
    #                              "the number of boundaries")
    #     else:
    #         msg = "Can't add data associated with '{}'".format
    #         raise ValueError(msg(uvar.location))
    #     self._data[uvar.name] = uvar

    # def find_uvars(self, standard_name, location=None):
    #     """
    #     Find all :class:`.UVar` objects  that match the specified standard name

    #     :param str standard_name: the standard name attribute.
    #                               Based on the UGRID conventions.

    #     :keyword location: optional attribute location to narrow the returned
    #                        :py:class:`UVar` objects
    #                        (one of 'node', 'edge', 'face', or 'boundary').

    #     :return: set of matching :py:class:`UVar` objects

    #     """
    #     found = set()
    #     for ds in self._data.values():
    #         if not ds.attributes or 'standard_name' not in ds.attributes:
    #             continue
    #         if ds.attributes['standard_name'] == standard_name:
    #             if location is not None and ds.location != location:
    #                 continue
    #             found.add(ds)
    #     return found

    def locate_nodes(self, points):
        """
        Returns the index of the closest nodes to the input locations.

        :param points: the lons/lats of locations you want the nodes
                       closest to.
        :type point: a (N, 2) ndarray of points
                     (or something that can be converted).

        :returns: the index of the closest node.

        """
        if self._kdtree is None:
            self._build_kdtree()

        node_inds = self._kdtree.query(points, k=1)[1]
        return node_inds

    def _build_kdtree(self):
        # Only import if it's used.
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("The scipy package is required to use "
                              "UGrid.locate_nodes\n"
                              " -- nearest neighbor interpolation")

        self._kdtree = cKDTree(self.nodes)

    def _hash_of_pts(self, points):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return hashlib.sha1(points.tobytes()).hexdigest()

    def _add_memo(self, points, item, D, _copy=False, _hash=None):
        """
        :param points: List of points to be hashed.
        :param item: Result of computation to be stored.
        :param location: Name of grid on which computation was done.
        :param D: Dict that will store hash -> item mapping.
        :param _hash: If hash is already computed it may be passed in here.
        """
        if _copy:
            item = item.copy()
        item.setflags(write=False)
        if _hash is None:
            _hash = self._hash_of_pts(points)
        if D is not None:
            D[_hash] = item
            if len(D.keys()) > 6:
                D.popitem(last=False)
            D[_hash].setflags(write=False)

    def _get_memoed(self, points, D, _copy=False, _hash=None):
        if _hash is None:
            _hash = self._hash_of_pts(points)
        if (D is not None and _hash in D):
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def locate_faces(self, points, method='celltree', _copy=False, _memo=True, _hash=None):
        """
        Returns the face indices, one per point.

        Points that are not in the mesh will have an index of -1

        If a single point is passed in, a single index will be returned
        If a sequence of points is passed in an array of indexes will be returned.

        :param points:  The points that you want to locate -- (lon, lat). If the shape of point
                        is 1D, function will return a scalar index. If it is 2D, it will return
                        a 1D array of indices
        :type point: array-like containing one or more points: shape (2,) for one point, shape (N, 2)
                     for more than one point.

        :param method='celltree': method to use. Options are 'celltree', 'simple'.
                                  for 'celltree' the celltree2d pacakge must be installed:
                                  https://github.com/NOAA-ORR-ERD/cell_tree2d/
                                  'simple' is very, very slow for large grids.
        :type simple: str

        This version utilizes the CellTree data structure.

        """
        points = np.asarray(points, dtype=np.float64)
        just_one = (points.ndim == 1)
        points.shape = (-1, 2)

        if _memo:
            if _hash is None:
                _hash = self._hash_of_pts(points)
            result = self._get_memoed(points, self._ind_memo_dict, _copy, _hash)
            if result is not None:
                return result

        if method == 'celltree':
            try:
                import cell_tree2d
            except ImportError:
                raise ImportError("the cell_tree2d package must be installed to use the celltree search:\n"
                                  "https://github.com/NOAA-ORR-ERD/cell_tree2d/")
            if self._cell_tree is None:
                self.build_celltree()
            indices = self._cell_tree.locate(points)
        elif method == 'simple':
            indices = np.zeros((points.shape[0]), dtype=IND_DT)
            for n, point in enumerate(points):
                for i, face in enumerate(self._faces):
                    f = self._nodes[face]
                    if point_in_tri(f, point):
                        indices[n] = i
                        break
                    else:
                        indices[n] = -1
        else:
            raise ValueError('"method" must be one of: "celltree", "simple"')

        if _memo:
            self._add_memo(points, indices, self._ind_memo_dict, _copy, _hash)

        if just_one:
            return indices[0]
        else:
            return indices

    def build_celltree(self):
        """
        Tries to build the celltree for the current UGrid. Will fail if nodes
        or faces is not defined.
        """
        from cell_tree2d import CellTree
        if self.nodes is None or self.faces is None:
            raise ValueError(
                "Nodes and faces must be defined in order to create and use CellTree")
        self._cell_tree = CellTree(self.nodes, self.faces)

    def interpolation_alphas(self, points, indices=None, _copy=False, _memo=True, _hash=None):
        """
        Given an array of points, this function will return the bilinear interpolation alphas
        for each of the three nodes of the face that the point is located in. If the point is
        not located on the grid, the alphas are set to 0
        :param points: Nx2 numpy array of lat/lon coordinates

        :param indices: If the face indices of the points is already known, it can be passed
                        in to save repeating the effort.

        :return: Nx3 numpy array of interpolation factors

        TODO: mask the indices that aren't on the grid properly.
        """

        if _memo:
            if _hash is None:
                _hash = self._hash_of_pts(points)
            result = self._get_memoed(points, self._alpha_memo_dict, _copy, _hash)
            if result is not None:
                return result

        if indices is None:
            indices = self.locate_faces(points, 'celltree', _copy, _memo, _hash)
        node_positions = self.nodes[self.faces[indices]]

        (lon1, lon2, lon3) = node_positions[:, :, 0].T
        (lat1, lat2, lat3) = node_positions[:, :, 1].T

        reflats = points[:, 1]
        reflons = points[:, 0]

        denoms = (
            (lat3 - lat1) * (lon2 - lon1) - (lon3 - lon1) * (lat2 - lat1))
        # alphas should all add up to 1
        alpha1s = (reflats - lat3) * (lon3 - lon2) - \
            (reflons - lon3) * (lat3 - lat2)
        alpha2s = (reflons - lon1) * (lat3 - lat1) - \
            (reflats - lat1) * (lon3 - lon1)
        alpha3s = (reflats - lat1) * (lon2 - lon1) - \
            (reflons - lon1) * (lat2 - lat1)
        alphas = np.column_stack(
            (alpha1s / denoms, alpha2s / denoms, alpha3s / denoms))
        alphas[indices == -1] *= 0

        if _memo:
            self._add_memo(points, alphas, self._alpha_memo_dict, _copy, _hash)

        return alphas

    def interpolate_var_to_points(self,
                                  points,
                                  variable,
                                  location=None,
                                  fill_value=0,
                                  indices=None,
                                  alphas=None,
                                  slices=None,
                                  _copy=False,
                                  _memo=True,
                                  _hash=None):
        """
        Interpolates a variable on one of the grids to an array of points.

        :param points: Nx2 Array of lon/lat coordinates to be interpolated to.

        :param variable: Array-like of values to associate at location on grid
                         (node, center, edge1, edge2). This may be more than a
                         2-dimensional array, but you must pass 'slices' kwarg
                         with appropriate slice collection to reduce it to
                         2 dimensions.

        :param location: One of ('node', 'center', 'edge1', 'edge2') 'edge1' is
                         conventionally associated with the 'vertical' edges and
                         likewise 'edge2' with the 'horizontal'

        :param fill_value: If masked values are encountered in interpolation, this
                           value takes the place of the masked value

        :param indices: If computed already, array of Nx2 cell indices can be passed
                        in to increase speed.
        :param alphas: If computed already, array of alphas can be passed in to increase
                       speed.


        With a numpy array:

        sgrid.interpolate_var_to_points(points, sgrid.u[time_idx, depth_idx])

        With a raw netCDF Variable:

        sgrid.interpolate_var_to_points(points, nc.variables['u'], slices=[time_idx, depth_idx])

        If you have pre-computed information, you can pass it in to avoid unnecessary
        computation and increase performance.

        - ind = # precomputed indices of points

        - alphas = # precomputed alphas (useful if interpolating to the same points frequently)
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        # location should be already known by the variable
        if hasattr(variable, 'location'):
            location = variable.location
        # But if it's not, then it can be inferred
        # (for compatibility with old code)
        if location is None:
            location = self.infer_location(variable)
            variable.location = location
        if location is None:
            raise ValueError("Data is incompatible with grid nodes or faces")

        if slices is not None:
            if len(slices) == 1:
                slices = slices[0]
            variable = variable[slices]

        _hash = self._hash_of_pts(points)

        inds = self.locate_faces(points, 'celltree', _copy, _memo, _hash)
        if location == 'face':
            vals = variable[inds]
            vals[inds == -1] = vals[inds == -1] * 0
            return vals
#             raise NotImplementedError("Currently does not support interpolation of a "
#                                       "variable defined on the faces")
        if location == 'node':
            pos_alphas = self.interpolation_alphas(points, inds, _copy, _memo, _hash)
            vals = variable[self.faces[inds]]
            vals[inds == -1] = vals[inds == -1] * 0
            return np.sum(vals * pos_alphas, axis=1)
        return None

    interpolate = interpolate_var_to_points

    def build_face_face_connectivity(self):
        """
        Builds the face_face_connectivity array: giving the neighbors of each cell.

        Note: arbitrary order and CW vs CCW may not be consistent.
        """

        num_vertices = self.num_vertices
        num_faces = self.faces.shape[0]
        face_face = np.zeros((num_faces, num_vertices), dtype=IND_DT)
        face_face += -1  # Fill with -1.

        # Loop through all the faces to find the matching edges:
        edges = {}  # dict to store the edges.
        for i, face in enumerate(self.faces):
            # Loop through edges of the cell:
            for j in range(num_vertices):
                if j < self.num_vertices - 1:
                    edge = (face[j], face[j + 1])
                else:
                    edge = (face[-1], face[0])
                if edge[0] > edge[1]:  # Sort the node numbers.
                    edge = (edge[1], edge[0])
                # see if it is already in there
                prev_edge = edges.pop(edge, None)
                if prev_edge is not None:
                    face_num, edge_num = prev_edge
                    face_face[i, j] = face_num
                    face_face[face_num, edge_num] = i
                else:
                    edges[edge] = (i, j)  # face num, edge_num.
        self._face_face_connectivity = face_face

    def get_lines(self):
        if self.edges is None:
            self.build_edges()
        return self.nodes[self.edges]

    def build_edges(self):
        """
        Builds the edges array: all the edges defined by the faces

        This will replace the existing edge array, if there is one.

        NOTE: arbitrary order -- should the order be preserved?
        """

        num_vertices = self.num_vertices
        if self.faces is None:
            # No faces means no edges
            self._edges = None
            return
        num_faces = self.faces.shape[0]
        face_face = np.zeros((num_faces, num_vertices), dtype=IND_DT)
        face_face += -1  # Fill with -1.

        # Loop through all the faces to find all the edges:
        edges = set()  # Use a set so no duplicates.
        for i, face in enumerate(self.faces):
            # Loop through edges:
            for j in range(num_vertices):
                edge = (face[j - 1], face[j])
                if edge[0] > edge[1]:  # Flip them
                    edge = (edge[1], edge[0])
                edges.add(edge)
        self._edges = np.array(list(edges), dtype=IND_DT)

    def build_boundaries(self):
        """
        Builds the boundary segments from the cell array.

        It is assumed that -1 means no neighbor, which indicates a boundary

        This will over-write the existing boundaries array if there is one.

        This is a not-very-smart just loop through all the faces method.

        """
        boundaries = []
        for i, face in enumerate(self.face_face_connectivity):
            for j, neighbor in enumerate(face):
                if neighbor == -1:
                    if j == self.num_vertices - 1:
                        bound = (self.faces[i, -1], self.faces[i, 0])
                    else:
                        bound = (self.faces[i, j], self.faces[i, j + 1])
                    boundaries.append(bound)
        self.boundaries = boundaries

    def build_face_edge_connectivity(self):
        """
        Builds the face-edge connectivity array

        Not implemented yet.

        """

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("The scipy package is required to use "
                              "UGrid.locatbuild_face_edge_connectivity")

        faces = self.faces
        edges = self.edges.copy()
        face_edges = np.dstack([faces, np.roll(faces, 1, 1)])
        if np.ma.isMA(faces) and np.ndim(faces.mask):
            face_edges.mask = np.dstack([
                faces.mask, np.roll(faces.mask, 1, 1)
            ])
        face_edges.sort(axis=-1)
        edges.sort(axis=-1)

        tree = cKDTree(edges)

        face_edge_2d = face_edges.reshape((-1, 2))

        if np.ma.isMA(faces) and faces.mask.any():
            mask = face_edge_2d.mask.any(-1)
            connectivity = np.ma.ones(
                len(face_edge_2d), dtype=face_edge_2d.dtype,
            )
            connectivity.mask = mask
            connectivity[~mask] = tree.query(face_edge_2d[~mask])[1]
        else:
            connectivity = tree.query(face_edge_2d)[1]
        self.face_edge_connectivity = np.roll(
            connectivity.reshape(faces.shape), -1, -1
        )

    def build_face_coordinates(self):
        """
        Builds the face_coordinates array, using the average of the
        nodes defining each face.

        Note that you may want a different definition of the face
        coordinates than this computes, but this is here to have
        an easy default.

        This will write-over an existing face_coordinates array.

        Useful if you want this in the output file.

        """
        self.face_coordinates = self.nodes[self.faces].mean(axis=1)

    def build_edge_coordinates(self):
        """
        Builds the edge_coordinates array, using the average of the
        nodes defining each edge.

        Note that you may want a different definition of the edge
        coordinates than this computes, but this is here to have
        an easy default.


        This will write-over an existing edge_coordinates array

        Useful if you want this in the output file

        """
        self.edge_coordinates = self.nodes[self.edges].mean(axis=1)

    def build_boundary_coordinates(self):
        """
        Builds the boundary_coordinates array, using the average of the
        nodes defining each boundary segment.

        Note that you may want a different definition of the boundary
        coordinates than this computes, but this is here to have
        an easy default.

        This will write-over an existing face_coordinates array

        Useful if you want this in the output file

        """
        self.boundary_coordinates = self.nodes[self.boundaries].mean(axis=1)

    def save_as_netcdf(self, filename, format='netcdf4'):
        """
        save the dataset to a file

        :param filename: full path to file to save to.

        :param format: format to save -- 'netcdf3' or 'netcdf4'
                       are the only options at this point.
        """
        self.save(filename, format='netcdf4')


    def save(self, filepath, format='netcdf4', variables={}):
        """
        Save the ugrid object as a netcdf file.

        :param filepath: path to file you want o save to.  An existing one
                         will be clobbered if it already exists.

        :param variables: dict of gridded.Variable objects to save to file

        Follows the convention established by the netcdf UGRID working group:

        http://ugrid-conventions.github.io/ugrid-conventions

        NOTE: Variables are saved here, because different conventions do it
              differently.
        """
        format_options = ('netcdf3', 'netcdf4')
        if format not in format_options:
            raise ValueError("format: {} not supported. Options are: {}".format(format, format_options))

        mesh_name = self.mesh_name

        nclocal = get_writable_dataset(filepath)

        nclocal.createDimension(mesh_name + "_num_node", len(self.nodes))
        if self._edges is not None:
            nclocal.createDimension(
                mesh_name + "_num_edge", len(self._edges))
        if self._boundaries is not None:
            nclocal.createDimension(mesh_name + "_num_boundary",
                                    len(self._boundaries))
        if self._faces is not None:
            nclocal.createDimension(
                mesh_name + "_num_face", len(self._faces))
            nclocal.createDimension(mesh_name + "_num_vertices",
                                    self._faces.shape[1])
        nclocal.createDimension("two", 2)

        # mesh topology
        mesh = nclocal.createVariable(mesh_name, IND_DT, (),)
        mesh.cf_role = "mesh_topology"
        mesh.long_name = "Topology data of 2D unstructured mesh"
        mesh.topology_dimension = 2
        mesh.node_coordinates = "{0}_node_lon {0}_node_lat".format(mesh_name)  # noqa

        if self._edges is not None:
            # Attribute required if variables will be defined on edges.
            mesh.edge_node_connectivity = mesh_name + "_edge_nodes"
            if self.edge_coordinates is not None:
                # Optional attribute (requires edge_node_connectivity).
                coord = "{0}_edge_lon {0}_edge_lat".format
                mesh.edge_coordinates = coord(mesh_name)
        if self._faces is not None:
            mesh.face_node_connectivity = mesh_name + "_face_nodes"
            if self.face_coordinates is not None:
                # Optional attribute.
                coord = "{0}_face_lon {0}_face_lat".format
                mesh.face_coordinates = coord(mesh_name)
        if self.face_edge_connectivity is not None:
            # Optional attribute (requires edge_node_connectivity).
            mesh.face_edge_connectivity = mesh_name + "_face_edges"
        if self.face_face_connectivity is not None:
            # Optional attribute.
            mesh.face_face_connectivity = mesh_name + "_face_links"
        if self._boundaries is not None:
            mesh.boundary_node_connectivity = mesh_name + "_boundary_nodes"

        # FIXME: This could be re-factored to be more generic, rather than
        # separate for each type of data see the coordinates example below.
        if self._faces is not None:
            nc_create_var = nclocal.createVariable
            face_nodes = nc_create_var(mesh_name + "_face_nodes", IND_DT,
                                       (mesh_name + '_num_face',
                                        mesh_name + '_num_vertices'),)
            face_nodes[:] = self.faces

            face_nodes.cf_role = "face_node_connectivity"
            face_nodes.long_name = ("Maps every triangular face to "
                                    "its three corner nodes.")
            face_nodes.start_index = IND_DT(0)

        if self._edges is not None:
            nc_create_var = nclocal.createVariable
            edge_nodes = nc_create_var(mesh_name + "_edge_nodes", IND_DT,
                                       (mesh_name + '_num_edge', 'two'),)
            edge_nodes[:] = self.edges

            edge_nodes.cf_role = "edge_node_connectivity"
            edge_nodes.long_name = ("Maps every edge to the two "
                                    "nodes that it connects.")
            edge_nodes.start_index = IND_DT(0)

        if self._boundaries is not None:
            nc_create_var = nclocal.createVariable
            boundary_nodes = nc_create_var(mesh_name + "_boundary_nodes",
                                           IND_DT,
                                           (mesh_name + '_num_boundary',
                                            'two'),)
            boundary_nodes[:] = self.boundaries

            boundary_nodes.cf_role = "boundary_node_connectivity"
            boundary_nodes.long_name = ("Maps every boundary segment to "
                                        "the two nodes that it connects.")
            boundary_nodes.start_index = IND_DT(0)

        # Optional "coordinate variables."
        for location in ['face', 'edge', 'boundary']:
            loc = "{0}_coordinates".format(location)
            if getattr(self, loc) is not None:
                for axis, ind in [('lat', 1), ('lon', 0)]:
                    nc_create_var = nclocal.createVariable
                    name = "{0}_{1}_{2}".format(mesh_name, location, axis)
                    dimensions = "{0}_num_{1}".format(mesh_name, location)
                    var = nc_create_var(name, NODE_DT,
                                        dimensions=(dimensions),)
                    loc = "{0}_coordinates".format(location)
                    var[:] = getattr(self, loc)[:, ind]
                    # Attributes of the variable.
                    var.standard_name = ("longitude" if axis == 'lon'
                                         else 'latitude')
                    var.units = ("degrees_east" if axis == 'lon'
                                 else 'degrees_north')
                    name = "Characteristics {0} of 2D mesh {1}".format
                    var.long_name = name(var.standard_name, location)

        # The node data.
        node_lon = nclocal.createVariable(mesh_name + '_node_lon',
                                          self._nodes.dtype,
                                          (mesh_name + '_num_node',),
                                          chunksizes=(len(self.nodes),),
                                          # zlib=False,
                                          # complevel=0,
                                          )
        node_lon[:] = self.nodes[:, 0]
        node_lon.standard_name = "longitude"
        node_lon.long_name = "Longitude of 2D mesh nodes."
        node_lon.units = "degrees_east"

        node_lat = nclocal.createVariable(mesh_name + '_node_lat',
                                          self._nodes.dtype,
                                          (mesh_name + '_num_node',),
                                          chunksizes=(len(self.nodes),),
                                          # zlib=False,
                                          # complevel=0,
                                          )
        node_lat[:] = self.nodes[:, 1]
        node_lat.standard_name = "latitude"
        node_lat.long_name = "Latitude of 2D mesh nodes."
        node_lat.units = "degrees_north"

        self._save_variables(nclocal, variables)

        nclocal.sync()
        return nclocal

    def _save_variables(self, nclocal, variables):
        """
        Save the Variables
        """
        mesh_name = self.mesh_name
        for name, var in variables.items():
            if var.location == 'node':
                shape = (mesh_name + '_num_node',)
                coordinates = "{0}_node_lon {0}_node_lat".format(mesh_name)
                chunksizes = (len(self.nodes),)
            elif var.location == 'face':
                shape = (mesh_name + '_num_face',)
                coord = "{0}_face_lon {0}_face_lat".format
                coordinates = (coord(mesh_name) if self.face_coordinates
                               is not None else None)
                chunksizes = (len(self.faces),)
            elif var.location == 'edge':
                shape = (mesh_name + '_num_edge',)
                coord = "{0}_edge_lon {0}_edge_lat".format
                coordinates = (coord(mesh_name) if self.edge_coordinates
                               is not None else None)
                chunksizes = (len(self.edges),)
            elif var.location == 'boundary':
                shape = (mesh_name + '_num_boundary',)
                coord = "{0}_boundary_lon {0}_boundary_lat".format
                bcoord = self.boundary_coordinates
                coordinates = (coord(mesh_name) if bcoord
                               is not None else None)
                chunksizes = (len(self.boundaries),)
            else:
                raise ValueError("I don't know how to save a variable located on: {}".format(var.location))
            print("Saving:", var)
            print("name is:", var.name)
            print("var data is:", var.data)
            print("var data shape is:", var.data.shape)
            data_var = nclocal.createVariable(var.name,
                                              var.data.dtype,
                                              shape,
                                              chunksizes=chunksizes,
                                              # zlib=False,
                                              # complevel=0,
                                              )
            print("new dat var shape:", shape)
            data_var[:] = var.data[:]
            # Add the standard attributes:
            data_var.location = var.location
            data_var.mesh = mesh_name
            if coordinates is not None:
                data_var.coordinates = coordinates
            # Add the extra attributes.
            for att_name, att_value in var.attributes.items():
                setattr(data_var, att_name, att_value)


class Mesh(dict):
    REQUIRED = ('cf_role', 'topology_dimension', 'node_coordinates')
    def __init__(self):
        super().__init__()
        self['cf_role'] = 'mesh_topology'
        self['data_model'] = 'NETCDF4'


        # populate required keys and fill with None
        self.update(dict.fromkeys(self.REQUIRED, None))
        self['dimension'] = {}
        self['variables'] = {}

    @classmethod
    def from_nc(cls, ncfile):
        rv = cls()

        rv['data_model'] = ncfile.data_model

        # find the first mesh variable
        # What to do when error condition exists?
        mesh = ncfile.get_variables_by_attributes(cf_role="mesh_topology")[0]
        rv['name'] = mesh.name
        
        # Ensure that the required attributes get copied (or raise an error)
        for k in set(chain(rv.REQUIRED, mesh.ncattrs())):
            rv[k] = getattr(mesh, k)

        rv['dimensions'] = dimensions = {}
        rv['variables'] = variables = {}

        for x in ncfile.dimensions.values():
            dimensions[x.name] = Dimension.from_ncdimension(x)

        for x in ncfile.variables.values():
            variables[x.name] = Variable.from_ncvariable(x)

        return rv

class Dimension(dict):
    @classmethod
    def from_ncdimension(cls, dimension):
        assert isinstance(dimension, netCDF4.Dimension)
        rv = cls()
        rv['name'] = dimension.name
        rv['size'] = dimension.size
        return rv

class Variable(dict):
    @classmethod
    def from_ncvariable(cls, variable):
        assert isinstance(variable, netCDF4.Variable)
        rv = cls()
        rv['name'] = variable.name
        rv['dimension'] = variable.dimension
        rv['dtype'] = variable.dtype
        rv['attrs'] = dict((v, getattr(variable, v)) for v in variable.ncattrs())
        return rv
