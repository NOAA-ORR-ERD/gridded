#!/usr/bin/env python

"""
class for ungridded data

This class represents data on arbitrary points
-- it provides the same API as a Grid type, even tough there is no grid.

It can provide data access through interpolation however
"""

from __future__ import (absolute_import, division, print_function)

# import hashlib
# from collections import OrderedDict

# import numpy as np

# import gridded.pyugrid.read_netcdf as read_netcdf
# from gridded.pyugrid.util import point_in_tri

from gridded.utilities import get_writable_dataset

#from gridded.pyugrid.uvar import UVar

# __all__ = ['UGrid',
#            'UVar']


# datatype used for indexes -- might want to change for 64 bit some day.
IND_DT = np.int32
NODE_DT = np.float64  # datatype used for node coordinates.


class NoGrid(object):
    """
    A basic class to hold ungridded data

    The internal structure mirrors the netcdf CF as best as can be.
    """

    def __init__(self,
                 nodes=None,
                 node_lon=None,
                 node_lat=None,
                 ):
        """
        NoGrid class -- holds, saves, etc. ungridded data

        :param nodes=None : the coordinates of the nodes
        :type nodes: (NX2) array of floats

        :param nodes=None : the coordinates of the nodes
        :type nodes: (NX2) array of floats

        :param node_lon=None: latitudes of nodes
        :param node_lat=None: longitudes of nodes

        The nodes can be passed in as either a single (NX2) (lon, lat) array
        or as two separate arrays of lon and lat

        Often this is too much data to pass in as literals -- so usually
        specialized constructors will be used instead (load from file, etc).
        """

        if ((nodes is not None) and
            ((node_lon is not None) or
             (node_lat is not None))):
            raise TypeError("You need to provide either a single nodes array "
                            "or node_lon and node_lat array")
        if nodes is None:
            if node_lon is not None and node_lat is not None:
                nodes = np.ma.column_stack((node_lon, node_lat))

        self.nodes = nodes

        # A kdtree is used to locate nodes.
        # It will be created if/when it is needed.
        self._kdtree = None

        self._ind_memo_dict = OrderedDict()
        self._alpha_memo_dict = OrderedDict()

    @classmethod
    def from_ncfile(klass, nc_url):  # , load_data=False):
        """
        create a NoGrid object from a netcdf file name (or opendap url)

        :param nc_url: the filename or OpenDap url you want to load

        """
        grid = klass()
        read_netcdf.load_grid_from_ncfilename(nc_url, grid, mesh_name)  # , load_data)
        return grid

    @classmethod
    def from_nc_dataset(klass, nc, mesh_name=None):  # , load_data=False):
        """
        create a UGrid object from an netcdf4 dataset

        :param nc: An already open Dataset object
        :type nc: netCDF4.DataSet
        """
        grid = klass()
        read_netcdf.load_grid_from_nc_dataset(nc, grid, mesh_name)  # , load_data)
        return grid

    @property
    def info(self):
        """
        summary of information about the grid
        """
        return "NoGrid object:\nNumber of nodes: %i".format(len(self.nodes))

    @property
    def nodes(self):
        return self._nodes

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

    @property
    def node_lon(self):
        return self._nodes[:, 0]

    @property
    def node_lat(self):
        return self._nodes[:, 1]

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
                              "NoGrid.locate_nodes\n"
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

#     def interpolate_var_to_points(self,
#                                   points,
#                                   variable,
#                                   location=None,
#                                   fill_value=0,
#                                   indices=None,
#                                   alphas=None,
#                                   slices=None,
#                                   _copy=False,
#                                   _memo=True,
#                                   _hash=None):
#         """
#         Interpolates a variable on one of the grids to an array of points.
#         :param points: Nx2 Array of lon/lat coordinates to be interpolated to.

#         :param variable: Array-like of values to associate at location on grid (node, center, edge1, edge2).
#         This may be more than a 2 dimensional array, but you must pass 'slices' kwarg with appropriate
#         slice collection to reduce it to 2 dimensions.

#         :param location: One of ('node', 'center', 'edge1', 'edge2') 'edge1' is conventionally associated with the
#         'vertical' edges and likewise 'edge2' with the 'horizontal'

#         :param fill_value: If masked values are encountered in interpolation, this value takes the place of the masked value

#         :param indices: If computed already, array of Nx2 cell indices can be passed in to increase speed. # noqa
#         :param alphas: If computed already, array of alphas can be passed in to increase speed. # noqa


#         - With a numpy array:
#         sgrid.interpolate_var_to_points(points, sgrid.u[time_idx, depth_idx])
#         - With a raw netCDF Variable:
#         sgrid.interpolate_var_to_points(points, nc.variables['u'], slices=[time_idx, depth_idx])

#         If you have pre-computed information, you can pass it in to avoid unnecessary
#         computation and increase performance.
#         - ind = # precomputed indices of points
#         - alphas = # precomputed alphas (useful if interpolating to the same points frequently)

#         sgrid.interpolate_var_to_points(points, sgrid.u, indices=ind, alphas=alphas,
#         slices=[time_idx, depth_idx])

#         """
#         points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
#         # location should be already known by the variable
#         if hasattr(variable, 'location'):
#             location = variable.location
#         # But if it's not, then it can be infered
#         # (for compatibilty with old code)
#         if location is None:
#             location = self.infer_location(variable)
#             variable.location = location
#         if location is None:
#             raise ValueError("Data is incompatible with grid nodes or faces")

#         if slices is not None:
#             if len(slices) == 1:
#                 slices = slices[0]
#             variable = variable[slices]

#         _hash = self._hash_of_pts(points)

#         inds = self.locate_faces(points, 'celltree', _copy, _memo, _hash)
#         if location == 'face':
#             vals = variable[inds]
#             vals[inds == -1] = vals[inds == -1] * 0
#             return vals
# #             raise NotImplementedError("Currently does not support interpolation of a "
# #                                       "variable defined on the faces")
#         if location == 'node':
#             pos_alphas = self.interpolation_alphas(points, inds, _copy, _memo, _hash)
#             vals = variable[self.faces[inds]]
#             vals[inds == -1] = vals[inds == -1] * 0
#             return np.sum(vals * pos_alphas, axis=1)
#         return None

#     interpolate = interpolate_var_to_points

    # def build_face_face_connectivity(self):
    #     """
    #     Builds the face_face_connectivity array: giving the neighbors of each cell.

    #     Note: arbitrary order and CW vs CCW may not be consistent.
    #     """

    #     num_vertices = self.num_vertices
    #     num_faces = self.faces.shape[0]
    #     face_face = np.zeros((num_faces, num_vertices), dtype=IND_DT)
    #     face_face += -1  # Fill with -1.

    #     # Loop through all the faces to find the matching edges:
    #     edges = {}  # dict to store the edges.
    #     for i, face in enumerate(self.faces):
    #         # Loop through edges of the cell:
    #         for j in range(num_vertices):
    #             if j < self.num_vertices - 1:
    #                 edge = (face[j], face[j + 1])
    #             else:
    #                 edge = (face[-1], face[0])
    #             if edge[0] > edge[1]:  # Sort the node numbers.
    #                 edge = (edge[1], edge[0])
    #             # see if it is already in there
    #             prev_edge = edges.pop(edge, None)
    #             if prev_edge is not None:
    #                 face_num, edge_num = prev_edge
    #                 face_face[i, j] = face_num
    #                 face_face[face_num, edge_num] = i
    #             else:
    #                 edges[edge] = (i, j)  # face num, edge_num.
    #     self._face_face_connectivity = face_face

    # def get_lines(self):
    #     if self.edges is None:
    #         self.build_edges()
    #     return self.nodes[self.edges]

    # def build_edges(self):
    #     """
    #     Builds the edges array: all the edges defined by the faces

    #     This will replace the existing edge array, if there is one.

    #     NOTE: arbitrary order -- should the order be preserved?
    #     """

    #     num_vertices = self.num_vertices
    #     if self.faces is None:
    #         # No faces means no edges
    #         self._edges = None
    #         return
    #     num_faces = self.faces.shape[0]
    #     face_face = np.zeros((num_faces, num_vertices), dtype=IND_DT)
    #     face_face += -1  # Fill with -1.

    #     # Loop through all the faces to find all the edges:
    #     edges = set()  # Use a set so no duplicates.
    #     for i, face in enumerate(self.faces):
    #         # Loop through edges:
    #         for j in range(num_vertices):
    #             edge = (face[j - 1], face[j])
    #             if edge[0] > edge[1]:  # Flip them
    #                 edge = (edge[1], edge[0])
    #             edges.add(edge)
    #     self._edges = np.array(list(edges), dtype=IND_DT)

    # def build_boundaries(self):
    #     """
    #     Builds the boundary segments from the cell array.

    #     It is assumed that -1 means no neighbor, which indicates a boundary

    #     This will over-write the existing boundaries array if there is one.

    #     This is a not-very-smart just loop through all the faces method.

    #     """
    #     boundaries = []
    #     for i, face in enumerate(self.face_face_connectivity):
    #         for j, neighbor in enumerate(face):
    #             if neighbor == -1:
    #                 if j == self.num_vertices - 1:
    #                     bound = (self.faces[i, -1], self.faces[i, 0])
    #                 else:
    #                     bound = (self.faces[i, j], self.faces[i, j + 1])
    #                 boundaries.append(bound)
    #     self.boundaries = boundaries

    # def build_face_edge_connectivity(self):
    #     """
    #     Builds the face-edge connectivity array

    #     Not implemented yet.

    #     """
    #     raise NotImplementedError

    # def build_face_coordinates(self):
    #     """
    #     Builds the face_coordinates array, using the average of the
    #     nodes defining each face.

    #     Note that you may want a different definition of the face
    #     coordinates than this computes, but this is here to have
    #     an easy default.

    #     This will write-over an existing face_coordinates array.

    #     Useful if you want this in the output file.

    #     """
    #     face_coordinates = np.zeros((len(self.faces), 2), dtype=NODE_DT)
    #     # FIXME: there has got to be a way to vectorize this.
    #     for i, face in enumerate(self.faces):
    #         coords = self.nodes[face]
    #         face_coordinates[i] = coords.mean(axis=0)
    #     self.face_coordinates = face_coordinates

    # def build_edge_coordinates(self):
    #     """
    #     Builds the face_coordinates array, using the average of the
    #     nodes defining each edge.

    #     Note that you may want a different definition of the edge
    #     coordinates than this computes, but this is here to have
    #     an easy default.


    #     This will write-over an existing face_coordinates array

    #     Useful if you want this in the output file

    #     """
    #     edge_coordinates = np.zeros((len(self.edges), 2), dtype=NODE_DT)
    #     # FIXME: there has got to be a way to vectorize this.
    #     for i, edge in enumerate(self.edges):
    #         coords = self.nodes[edge]
    #         edge_coordinates[i] = coords.mean(axis=0)
    #     self.edge_coordinates = edge_coordinates

    # def build_boundary_coordinates(self):
    #     """
    #     Builds the boundary_coordinates array, using the average of the
    #     nodes defining each boundary segment.

    #     Note that you may want a different definition of the boundary
    #     coordinates than this computes, but this is here to have
    #     an easy default.

    #     This will write-over an existing face_coordinates array

    #     Useful if you want this in the output file

    #     """
    #     boundary_coordinates = np.zeros((len(self.boundaries), 2),
    #                                     dtype=NODE_DT)
    #     # FXIME: there has got to be a way to vectorize this.
    #     for i, bound in enumerate(self.boundaries):
    #         coords = self.nodes[bound]
    #         boundary_coordinates[i] = coords.mean(axis=0)
    #     self.boundary_coordinates = boundary_coordinates


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
        Save the NoGrid object in a netcdf file.

        :param filepath: path to file you want o save to.  An existing one
                         will be clobbered if it already exists.
        """
        format_options = ('netcdf3', 'netcdf4')
        if format not in format_options:
            raise ValueError("format: {} not supported. Options are: {}".format(format, format_options))

        nclocal = get_writable_dataset(filepath)

        nclocal.createDimension(mesh_name + "_num_node", len(self.nodes))
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
        node_lon.long_name = "Longitude of nodes."
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
        node_lat.long_name = "Latitude of nodes."
        node_lat.units = "degrees_north"

        nclocal.sync()
        return nclocal

    # def _save_variables(self, nclocal, variables):
    #     """
    #     Save the Variables
    #     """
    #     mesh_name = self.mesh_name
    #     for name, var in variables.items():
    #         if var.location == 'node':
    #             shape = (mesh_name + '_num_node',)
    #             coordinates = "{0}_node_lon {0}_node_lat".format(mesh_name)
    #             chunksizes = (len(self.nodes),)
    #         elif var.location == 'face':
    #             shape = (mesh_name + '_num_face',)
    #             coord = "{0}_face_lon {0}_face_lat".format
    #             coordinates = (coord(mesh_name) if self.face_coordinates
    #                            is not None else None)
    #             chunksizes = (len(self.faces),)
    #         elif var.location == 'edge':
    #             shape = (mesh_name + '_num_edge',)
    #             coord = "{0}_edge_lon {0}_edge_lat".format
    #             coordinates = (coord(mesh_name) if self.edge_coordinates
    #                            is not None else None)
    #             chunksizes = (len(self.edges),)
    #         elif var.location == 'boundary':
    #             shape = (mesh_name + '_num_boundary',)
    #             coord = "{0}_boundary_lon {0}_boundary_lat".format
    #             bcoord = self.boundary_coordinates
    #             coordinates = (coord(mesh_name) if bcoord
    #                            is not None else None)
    #             chunksizes = (len(self.boundaries),)
    #         else:
    #             raise ValueError("I don't know how to save a variable located on: {}".format(var.location))
    #         print("Saving:", var)
    #         print("name is:", var.name)
    #         print("var data is:", var.data)
    #         print("var data shape is:", var.data.shape)
    #         data_var = nclocal.createVariable(var.name,
    #                                           var.data.dtype,
    #                                           shape,
    #                                           chunksizes=chunksizes,
    #                                           # zlib=False,
    #                                           # complevel=0,
    #                                           )
    #         print("new dat var shape:", shape)
    #         data_var[:] = var.data[:]
    #         # Add the standard attributes:
    #         data_var.location = var.location
    #         data_var.mesh = mesh_name
    #         if coordinates is not None:
    #             data_var.coordinates = coordinates
    #         # Add the extra attributes.
    #         for att_name, att_value in var.attributes.items():
    #             setattr(data_var, att_name, att_value)


