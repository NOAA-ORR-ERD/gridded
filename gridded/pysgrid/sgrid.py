'''
Created on Apr 20, 2015

@author: ayan
'''


from netCDF4 import Dataset
import numpy as np
import hashlib
import warnings
from collections import OrderedDict

from gridded.pysgrid.read_netcdf import NetCDFDataset, parse_padding, find_grid_topology_var
from gridded.pysgrid.utils import calculate_angle_from_true_east, pair_arrays, GridPadding
from gridded.pysgrid.variables import SGridVariable
from gridded.utilities import gen_celltree_mask_from_center_mask

node_alternate_names = ['node','nodes', 'psi', 'vertex','vertices', 'point','points']
center_alternate_names = ['center','centers','face','faces','cell','cells']
edge1_alternate_names = ['edge1','u']
edge2_alternate_names = ['edge2','v']

class SGrid(object):

    padding_slices = {'both': (1, -1),
                      'none': (None, None),
                      'low': (1, None),
                      'high': (None, 1)
                      }

    topology_dimension = 2

    def __init__(self,
                 node_lon=None,
                 node_lat=None,
                 node_mask=None,
                 center_lon=None,
                 center_lat=None,
                 center_mask=None,
                 edge1_lon=None,
                 edge1_lat=None,
                 edge1_mask=None,
                 edge2_lon=None,
                 edge2_lat=None,
                 edge2_mask=None,
                 edges=None,
                 node_padding=None,
                 edge1_padding=None,
                 edge2_padding=None,
                 grid_topology_var=None,
                 variables=None,
                 grid_variables=None,
                 dimensions=None,
                 node_dimensions=None,
                 node_coordinates=None,
                 edge1_coordinates=None,
                 edge2_coordinates=None,
                 angles=None,
                 edge1_dimensions=None,
                 edge2_dimensions=None,
                 faces=None,
                 face_padding=None,
                 face_coordinates=None,
                 face_dimensions=None,
                 vertical_padding=None,
                 vertical_dimensions=None,
                 tree=None,    #Fixme: should this be initilizable here?
                 use_masked_boundary=False,
                 grid_topology=None,
                 masked_interpolant_behavior='zero',
                 *args,
                 **kwargs):

        '''
        :param node_lon: Longitude of the nodes
        :param node_lat: Latitude of the nodes
        :param node_mask: Mask of the nodes
        :param center_lon: Longitude of the cell centers
        :param center_lat: Latitude of the cell centers
        :param center_mask: Mask of the cell centers
        :param edge1_lon: Longitude of the edge1 (u) points
        :param edge1_lat: Latitude of the edge1 (u) points
        :param edge1_mask: Mask of the edge1 (u) points
        :param edge2_lon: Longitude of the edge2 (v) points
        :param edge2_lat: Latitude of the edge2 (v) points
        :param edge2_mask: Mask of the edge2 (v) points
        :param edges: Edges of the grid
        :param node_padding: Padding of the nodes. 2-tuple of strings, one per dimension.
        Valid values = 'low', 'high', 'both', 'none', or None
        'none' or None means no padding.    eg ('none', None) as slicing would be [:,:]
        'low' means the low end is padded.  eg ('low', None) as slicing would be [1:,:]
        'high' means the high endis padded. eg ('none', 'high') as slicing would be [:, :-1]
        'both' means both ends are padded.  eg ('both', 'both') as slicing would be [1:-1, 1:-1]
        :param face_padding: See node_padding. AKA center_padding
        :param edge1_padding: See node_padding
        :param edge2_padding: See node_padding
        :param grid_topology_var: The variable that contains the grid topology
        
        :param use_masked_boundary: When creating the BVH of the cells, we only use unmasked cells.
        This parameter controls whether a cell needs to have ALL masked nodes to be considered masked.
        If True, a cell is NOT masked if ANY of its nodes are unmasked. This can create a 'fuller' boundary
        of cells, but may cause major issues if masked nodes have junk values
        
        :param masked_interpolant_behavior: When interpolating a masked value, this parameter controls
        what happens to it. See docstring for compute_interpolant for more information. This attribute
        is used as the 'masked_behavior' parameter in compute_interpolant if not provided directly.
        '''

        self.node_lon = node_lon
        self.node_lat = node_lat
        self.node_mask = node_mask
        self.center_lon = center_lon
        self.center_lat = center_lat
        self.center_mask = center_mask
        self.edge1_lon = edge1_lon
        self.edge1_lat = edge1_lat
        self.edge1_mask = edge1_mask
        self.edge2_lon = edge2_lon
        self.edge2_lat = edge2_lat
        self.edge2_mask = edge2_mask
        self.edges = edges  # Fixme: is this needed?
        self.node_padding = node_padding
        self.edge1_padding = edge1_padding
        self.edge2_padding = edge2_padding
        self.grid_topology_var = grid_topology_var
        self.variables = variables
        self.grid_variables = grid_variables
        self.dimensions = dimensions
        self.node_dimensions = node_dimensions
        self.node_coordinates = node_coordinates
        self.edge1_coordinates = edge1_coordinates
        self.edge2_coordinates = edge2_coordinates
        self.angles = angles
        self.edge1_dimensions = edge1_dimensions
        self.edge2_dimensions = edge2_dimensions
        self.faces = faces
        self.face_padding = face_padding
        self.face_coordinates = face_coordinates
        self.face_dimensions = face_dimensions
        self.vertical_padding = vertical_padding
        self.vertical_dimensions = vertical_dimensions
        self.tree = tree
        self.use_masked_boundary = use_masked_boundary
        self._l_coeffs = None
        self._m_coeffs = None

        # used for nearest neighbor interpolation
        self._kd_trees = {}
        self._cell_tree = None

        self._log_ind_memo_dict = OrderedDict()
        self._cell_ind_memo_dict = OrderedDict()
        self._cell_tree_mask = None
        self.grid_topology = grid_topology
        self.masked_interpolant_behavior = masked_interpolant_behavior

    def __eq__(self, other):
        # Fixme: should this even exist
        #        keeping it because it's used in a test.
        if self is other:
            return True
        # maybe too strict?
        if self.__class__ is not other.__class__:
            return False
        # there is way too much to really check here ...
        # but it would a heck of a coincidence ...
        for attr in ('node_lon',
                     'node_lat',
                     'node_mask',
                     'center_lon',
                     'center_lat',
                     'center_mask',
                     'faces',
                     'face_padding',
                     ):
            if not np.array_equal(getattr(self, attr), getattr(other, attr)):
                return False
        return True


    @classmethod
    def load_grid(cls, nc):
        if isinstance(nc, Dataset):
            pass
        else:
            nc = Dataset(nc, 'r')
        topology_var = find_grid_topology_var(nc)
        sa = SGridAttributes(nc, cls.topology_dimension, topology_var)
        dimensions = sa.get_dimensions()
        node_dimensions, node_coordinates = sa.get_node_coordinates()
        grid_topology_var = sa.get_topology_var()
        edge1_dimensions, edge1_padding = sa.get_attr_dimension('edge1_dimensions')  # noqa
        edge2_dimensions, edge2_padding = sa.get_attr_dimension('edge2_dimensions')  # noqa
        edge1_coordinates = sa.get_attr_coordinates('edge1_coordinates')
        edge2_coordinates = sa.get_attr_coordinates('edge2_coordinates')
        angles = sa.get_angles()
        vertical_dimensions, vertical_padding = sa.get_attr_dimension('vertical_dimensions')  # noqa
        node_lon, node_lat = sa.get_cell_node_lat_lon()
        center_lon, center_lat = sa.get_cell_center_lat_lon()
        edge1_lon, edge1_lat = sa.get_cell_edge1_lat_lon()
        edge2_lon, edge2_lat = sa.get_cell_edge2_lat_lon()
        face_dimensions, face_padding = sa.get_attr_dimension('face_dimensions')  # noqa
        face_coordinates = sa.get_attr_coordinates('face_coordinates')
        node_mask, center_mask, edge1_mask, edge2_mask = sa.get_masks(node_lon,
                                                                      center_lon,
                                                                      edge1_lon,
                                                                      edge2_lon)
        sgrid = cls(angles=angles,
                    node_lon=node_lon,
                    node_lat=node_lat,
                    node_mask=node_mask,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    center_mask=center_mask,
                    edge1_lon=edge1_lon,
                    edge1_lat=edge1_lat,
                    edge1_mask=edge1_mask,
                    edge2_lon=edge2_lon,
                    edge2_lat=edge2_lat,
                    edge2_mask=edge2_mask,
                    dimensions=dimensions,
                    edge1_coordinates=edge1_coordinates,
                    edge1_dimensions=edge1_dimensions,
                    edge1_padding=edge1_padding,
                    edge2_coordinates=edge2_coordinates,
                    edge2_dimensions=edge2_dimensions,
                    edge2_padding=edge2_padding,
                    edges=None,
                    face_coordinates=face_coordinates,
                    face_dimensions=face_dimensions,
                    face_padding=face_padding,
                    faces=None,
                    grid_topology_var=grid_topology_var,
                    grid_variables=None,
                    node_coordinates=node_coordinates,
                    node_dimensions=node_dimensions,
                    node_padding=None,
                    variables=None,
                    vertical_dimensions=vertical_dimensions,
                    vertical_padding=vertical_padding)
        sa.get_variable_attributes(sgrid)
        return sgrid

    @property
    def info(self):
        """
        Summary of information about the grid

        This needs to be implimented -- see UGrid for example
        """
        names = ", ".join([name for name, at in vars(self).items()
                           if not name.startswith("_") if at is not None])

        msg = ("SGrid object with defined:\n"
               "    {}".format(names))

        return msg

    def get_all_face_padding(self):
        if self.face_padding is not None:
            all_face_padding = self.face_padding
        else:
            all_face_padding = []
        return all_face_padding

    def get_all_edge_padding(self):
        all_edge_padding = []
        if self._edge1_padding is not None:
            all_edge_padding += self._edge1_padding
        if self._edge2_padding is not None:
            all_edge_padding += self._edge2_padding
        return all_edge_padding

    def all_padding(self):
        all_padding = self.get_all_face_padding() + self.get_all_edge_padding()
        if self.vertical_padding is not None:
            all_padding += self.vertical_padding
        return all_padding

    def save_as_netcdf(self, filepath):
        """
        save the grid as a netcdf file

        :param filepath: path to the file to be created and saved to
        """
        with Dataset(filepath, 'w') as nclocal:
            grid_vars = self._save_common_components(nclocal)
            # Add attributes to the grid_topology variable.
            grid_vars.face_dimensions = self.face_dimensions
            if self.vertical_dimensions is not None:
                grid_vars.vertical_dimensions = self.vertical_dimensions
            if self.face_coordinates is not None:
                grid_vars.face_coordinates = ' '.join(self.face_coordinates)

    @property
    def non_grid_variables(self):
        non_grid_variables = [variable for variable in self.variables if
                              variable not in self.grid_variables]
        return non_grid_variables

    @property
    def nodes(self):
        return np.stack((self.node_lon, self.node_lat), axis=-1)

    @property
    def centers(self):
        return np.stack((self.center_lon, self.center_lat), axis=-1)

    @property
    def node_padding(self):
        if hasattr(self, '_node_padding') and self._node_padding:
            return self._node_padding
        else:
            return (None, None)

    @node_padding.setter
    def node_padding(self, val):
        self._node_padding = val

    @property
    def center_padding(self):
        if hasattr(self, '_center_padding') and self._center_padding:
            return self._center_padding
        elif hasattr(self, 'center_lon') and self.center_lon is not None:
            face_shape = self.center_lon.shape
            node_shape = self.node_lon.shape
            diff = np.array(face_shape) - node_shape
            rv = []
            for dim in (0,1):
                rv.append(('low', 'both', 'none')[diff[dim]])
                if rv[-1] == 'low':
                    warnings.warn('Assuming low padding for faces')
            return tuple(rv)
        else:
            return (None, None)

    @center_padding.setter
    def center_padding(self, val):
        self._center_padding = val

    @property
    def edge1_padding(self):
        if hasattr(self, '_edge1_padding') and self._edge1_padding:
            if isinstance(self._edge1_padding[0], GridPadding):
                return (self._edge1_padding[0].padding, None)
            else:
                return self._edge1_padding
        else:
            return (self.center_padding[0], None)

    @edge1_padding.setter
    def edge1_padding(self, val):
        self._edge1_padding = val

    @property
    def edge2_padding(self):
        if hasattr(self, '_edge2_padding') and self._edge2_padding:
            if isinstance(self._edge2_padding[0], GridPadding):
                return (None, self._edge2_padding[0].padding)
            else:
                return self._edge2_padding
        else:
            return (None, self.center_padding[1])

    @edge2_padding.setter
    def edge2_padding(self, val):
        self._edge2_padding = val

    def infer_location(self, variable):
        """
        Assuming default is psi grid, check variable dimensions to determine which grid
        it is on.
        """
        shape = None
        try:
            shape = np.array(variable.shape)
        except:
            return None  # Variable has no shape attribute!
        if len(variable.shape) < 2:
            return None
        difference = (shape[-2:] - self.node_lon.shape).tolist()
        if (difference == [1, 1] or difference == [-1, -1]) and self.center_lon is not None:
            location = 'center'
        elif difference == [1, 0] and self.edge1_lon is not None:
            location = 'edge1'
        elif difference == [0, 1] and self.edge2_lon is not None:
            location = 'edge2'
        elif difference == [0, 0] and self.node_lon is not None:
            location = 'node'
        else:
            location = None
        return location

    def _save_common_components(self, nc_file):
        grid_var = self.grid_topology_var
        # Create dimensions.
        for grid_dim in self.dimensions:
            dim_name, dim_size = grid_dim
            nc_file.createDimension(dim_name, dim_size)
        # Create variables.
        center_lon, center_lat = self.face_coordinates
        center_lon_obj = getattr(self, center_lon)
        center_lat_obj = getattr(self, center_lat)
        center_lon = nc_file.createVariable(center_lon_obj.variable,
                                            center_lon_obj.dtype,
                                            center_lon_obj.dimensions)
        center_lat = nc_file.createVariable(center_lat_obj.variable,
                                            center_lat_obj.dtype,
                                            center_lat_obj.dimensions)
        center_lon[:] = self.center_lon[:]
        center_lat[:] = self.center_lat[:]
        try:
            node_lon, node_lat = self.node_coordinates
        except TypeError:
            pass
        else:
            node_lon_obj = getattr(self, node_lon)
            grid_node_lon = nc_file.createVariable(node_lon_obj.variable,
                                                   node_lon_obj.dtype,
                                                   node_lon_obj.dimensions)
            node_lat_obj = getattr(self, node_lat)
            grid_node_lat = nc_file.createVariable(node_lat_obj.variable,
                                                   node_lat_obj.dtype,
                                                   node_lat_obj.dimensions)
            grid_node_lon[:] = self.node_lon[:]
            grid_node_lat[:] = self.node_lat[:]
        grid_var_obj = getattr(self, grid_var)
        grid_vars = nc_file.createVariable(grid_var_obj.variable,
                                           grid_var_obj.dtype)
        grid_vars.cf_role = 'grid_topology'
        grid_vars.topology_dimension = self.topology_dimension
        grid_vars.node_dimensions = self.node_dimensions
        if self.edge1_dimensions is not None:
            grid_vars.edge1_dimensions = self.edge1_dimensions
        if self.edge2_dimensions is not None:
            grid_vars.edge2_dimensions = self.edge2_dimensions
        if self.node_coordinates is not None:
            grid_vars.node_coordinates = ' '.join(self.node_coordinates)
        if self.edge1_coordinates is not None:
            grid_vars.edge1_coordinates = ' '.join(self.edge1_coordinates)
        if self.edge2_coordinates is not None:
            grid_vars.edge2_coordinates = ' '.join(self.edge2_coordinates)
        if hasattr(self, 'angle'):
            angle_obj = getattr(self, 'angle', None)
            grid_angle = nc_file.createVariable(angle_obj.variable,
                                                angle_obj.dtype,
                                                angle_obj.dimensions
                                                )
            if self.angles is not None:
                grid_angle[:] = self.angles[:]
        for dataset_variable in self.variables:
            dataset_var_obj = getattr(self, dataset_variable)
            try:
                dataset_grid_var = nc_file.createVariable(
                    dataset_var_obj.variable,
                    dataset_var_obj.dtype,
                    dataset_var_obj.dimensions
                )
            except RuntimeError:
                continue
            else:
                axes = []
                if dataset_var_obj.grid is not None:
                    dataset_grid_var.grid = grid_var
                if dataset_var_obj.standard_name is not None:
                    dataset_grid_var.standard_name = dataset_var_obj.standard_name  # noqa
                if dataset_var_obj.coordinates is not None:
                    dataset_grid_var.coordinates = ' '.join(dataset_var_obj.coordinates)  # noqa
                if dataset_var_obj.x_axis is not None:
                    x_axis = 'X: {0}'.format(dataset_var_obj.x_axis)
                    axes.append(x_axis)
                if dataset_var_obj.y_axis is not None:
                    y_axis = 'Y: {0}'.format(dataset_var_obj.y_axis)
                    axes.append(y_axis)
                if dataset_var_obj.z_axis is not None:
                    z_axis = 'Z: {0}'.format(dataset_var_obj.z_axis)
                    axes.append(z_axis)
                if axes:
                    dataset_grid_var.axes = ' '.join(axes)
        return grid_vars

    def _get_geo_mask(self, name):
        if name == 'node':
            return self.node_mask
        elif name == 'center':
            return self.center_mask
        elif name == 'edge1':
            return self.edge1_mask
        elif name == 'edge2':
            return self.edge2_mask
        else:
            raise ValueError('Invalid grid name {0}'.format(name))

    def _get_grid_vars(self, name):
        if name == 'node':
            return (self.node_lon, self.node_lat)
        elif name == 'center':
            return (self.center_lon, self.center_lat)
        elif name == 'edge1':
            return (self.edge1_lon, self.edge1_lat)
        elif name == 'edge2':
            return (self.edge2_lon, self.edge2_lat)
        else:
            raise ValueError('Invalid grid name {0}'.format(name))

    def _hash_of_pts(self, points):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return hashlib.sha1(points.tobytes()).hexdigest()

    def _add_memo(self, points, item, D, _copy=False, _hash=None):
        """
        :param points: List of points to be hashed.
        :param item: Result of computation to be stored.
        :param D: Dict that will store hash -> item mapping.
        :param _hash: If hash is already computed it may be passed in here.
        """
        if _copy:
            item = item.copy()
        item.setflags(write=False)
        if _hash is None:
            _hash = self._hash_of_pts(points)
        if D is not None and len(D) > 6:
            D.popitem(last=False)
        D[_hash] = item
        D[_hash].setflags(write=False)

    def _get_memoed(self, points, D, _copy=False, _hash=None):
        if _hash is None:
            _hash = self._hash_of_pts(points)
        if (D is not None and _hash in D):
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def _compute_transform_coeffs(self):
        """
        https://www.particleincell.com/2012/quad-interpolation/

        This computes the and b coefficients of the equations
        x = a1 + a2*l + a3*m + a4*l*m
        y = b1 + b2*l + b3*m + b4*l*m

        The results are memoized per grid since their geometry is different, and
        is not expected to change over the lifetime of the object.
        """
        lon, lat = self.node_lon, self.node_lat
        l_coeffs = self._l_coeffs = np.zeros((lon[0:-1, 0:-1].shape + (4,)), dtype=np.float64)
        m_coeffs = self._m_coeffs = self._l_coeffs.copy('C')

        indices = np.stack(np.indices(lon[0:-1, 0:-1].shape), axis=-1).reshape(-1, 2)
        polyx = self.get_variable_by_index(lon, indices)
        polyy = self.get_variable_by_index(lat, indices)
        # for every cell
        A = np.array(([1, 0, 0, 0],
                      [1, 0, 1, 0],
                      [1, 1, 1, 1],
                      [1, 1, 0, 0],
                      ))
        # A = np.array(([1, 0, 0, 0],
        #               [1, 1, 0, 0],
        #               [1, 1, 1, 1],
        #               [1, 0, 1, 0],
        #               ))
        # polyx = np.matrix(polyx)
        # polyy = np.matrix(polyy)
        AI = np.linalg.inv(A)
        a = np.dot(AI, polyx.T).T
        b = np.dot(AI, polyy.T).T

        self._l_coeffs = np.asarray(a).reshape(l_coeffs.shape)
        self._m_coeffs = np.asarray(b).reshape(m_coeffs.shape)

    def get_efficient_slice(self,
                            points=None,
                            indices=None,
                            location=None,
                            _memo=False,
                            _copy=False,
                            _hash=None):
        """
        Computes the minimum 2D slice that captures all the provided points/indices
        within.
        :param points: Nx2 array of longitude/latitude. (Optional)
        :param indices: Nx2 array of logical cell indices (Optional, but required if points omitted)
        :param location: 'center', 'edge1', 'edge2','node'
        """
        if indices is None:
            indices = self.locate_faces(points, _memo, _copy, _hash)
        xmin = indices[:, 0].astype('uint32').min()
        ymin = indices[:, 1].astype('uint32').min()
        xmax = indices[:, 0].astype('uint32').max() + 1
        ymax = indices[:, 1].astype('uint32').max() + 1

        if location in edge1_alternate_names:
            xmax += 1
        elif location in edge2_alternate_names:
            ymax += 1
        elif location in node_alternate_names:
            xmax += 1
            ymax += 1
        elif location in center_alternate_names:
            pass
        else:
            raise ValueError('location not recognized')

        x_slice = slice(xmin, xmax)
        y_slice = slice(ymin, ymax)
        return (x_slice, y_slice)

    def locate_faces(self,
                     points,
                     _memo=False,
                     _copy=False,
                     _hash=None,
                     use_mask=True):
        """
        Given a list of points, returns a list of x, y indices of the cell
        that contains each respective point

        Points that are not on the node grid will have an index of -1

        If a single point is passed in, a single index will be returned.
        If a sequence of points is passed in an array of indexes will be returned.

        :param points:  The points that you want to locate -- (lon, lat). If the shape of point
                        is 1D, function will return a scalar index. If it is 2D, it will return
                        a 1D array of indices.
        :type points: array-like containing one or more points: shape (2,) for one point,
                      shape (N, 2) for more than one point.

        :param grid: The grid on which you want to locate the points
        :type grid: Name of the grid ('node', 'center', 'edge1', 'edge2)

        This version utilizes the CellTree data structure.

        """
        points = np.asarray(points, dtype=np.float64)
        just_one = (points.ndim == 1)
        points = points.reshape(-1, 2)

        if _memo:
            if _hash is None:
                _hash = self._hash_of_pts(points)
            result = self._get_memoed(points, self._cell_ind_memo_dict, _copy, _hash)
            if result is not None:
                return result

        if self._cell_tree is None:
            self.build_celltree(use_mask=use_mask)
        tree = self._cell_tree[0]
        rev_arrs = None
        if self._cell_tree_mask is not None:
            rev_arrs = self._cell_tree_mask[1]
        indices = tree.locate(points)
        if rev_arrs is not None:
            indices = rev_arrs[indices]
        lon, lat = self.node_lon, self.node_lat
        x = indices % (lat.shape[1] - 1)
        y = indices // (lat.shape[1] - 1)
        ind = np.column_stack((y, x))

        ind[ind[:, 0] == -1] = [-1, -1]
        if just_one:
            res = ind[0]
            return res
        else:
            res = np.ma.masked_less(ind, 0)
            if _memo:
                self._add_memo(points, res, self._cell_ind_memo_dict, _copy, _hash)
            return res
        
    def index_of(self,
                 points,
                _memo=False,
                _copy=False,
                _hash=None,
                use_mask=True):
        return self.locate_faces(points=points,
                                 _memo=_memo,
                                 _copy=_copy,
                                 _hash=_hash,
                                 use_mask=use_mask)

    def locate_nearest(self,
                       points,
                       grid,
                       _memo=False,
                       _copy=False,
                       _hash=None):
        points = np.asarray(points, dtype=np.float64)
        points = points.reshape(-1, 2)

        if self._kd_trees[grid] is None:
            self.build_kdtree(grid)
        tree = self._kd_trees[grid]
        lin_indices = np.array(tree.query(points))[1].astype(np.int32)
        lon, lat = self._get_grid_vars(grid)
        ind = np.unravel_index(lin_indices, shape=lon.shape)
        ind = np.array(ind).T
        return ind

    def apply_padding_to_idxs(self,
                              idxs,
                              padding=('none','none')):
        '''
        Given a list of indexes, increment each dimension to compensate for padding.
        Input indexes are assumed to be cell indexes
        '''
        for dim, typ in enumerate(padding):
            if typ == 'none' or  typ == 'high' or typ is None:
                continue
            elif typ == 'both' or typ == 'low':
                idxs[:,dim] += 1
            else:
                raise ValueError('unrecognized padding type in dimension {0}: {1}'.format(dim, typ))
        return idxs

    def get_padding_by_location(self, location):
        d = [(node_alternate_names, 'node_padding'),
         (center_alternate_names, 'center_padding'),
         (edge1_alternate_names, 'edge1_padding'),
         (edge2_alternate_names, 'edge2_padding')]
        for namelist, propname in d:
            if location in namelist:
                rv = getattr(self, propname)
                assert rv is not None
                return rv

    def get_padding_slices(self,
                           padding=('none','none')):
        '''
        Given a pair of padding types, return a numpy slice object you can use directly on
        data or lon/lat variables
        '''
        lo_offsets = [0,0]
        hi_offsets = [0,0]
        for dim, typ in enumerate(padding):
            if typ == 'none' or typ is None:
                continue
            elif typ == 'high':
                hi_offsets[dim] -= 1
            elif typ == 'low':
                lo_offsets[dim] += 1
            elif typ == 'both':
                hi_offsets[dim] -= 1
                lo_offsets[dim] += 1
            else:
                hi_offsets[dim] = None
                lo_offsets[dim] = 0

        lo_offsets = [l if l != 0 else None for l in lo_offsets]
        hi_offsets = [h if h != 0 else None for h in hi_offsets]
        return (np.s_[lo_offsets[0]:hi_offsets[0], lo_offsets[1]:hi_offsets[1]])

    def get_variable_by_index(self, var, index):
        """
        index = index arr of quads (maskedarray only)
        var = ndarray/ma.array
        returns ndarray/ma.array

        ordering is idx, idx+[0,1], idx+[1,1], idx+[1,0]
        masked values from var remain masked

        Function to get the node values of a given face index.
        Emulates the 'self.grid.nodes[self.grid.nodes.faces[index]]'
        paradigm of unstructured grids.
        """

        var = var[:]

        if isinstance(var, np.ma.MaskedArray) and isinstance(index, np.ma.MaskedArray):
            rv = np.ma.empty((index.shape[0], 4), dtype=np.float64)
            if index.mask is not np.bool_():  # because False is not False. Thanks numpy
                rv.mask = np.zeros_like(rv, dtype=bool)
                rv.mask[:] = index.mask[:, 0][:, np.newaxis]
            rv.harden_mask()
        else:
            rv = np.zeros((index.shape[0], 4), dtype=np.float64)

        raw = np.ravel_multi_index(index.T, var.shape, mode='clip')
        rv[:, 0] = np.take(var, raw)
        raw += np.array(var.shape[1], dtype=np.int32)
        rv[:, 1] = np.take(var, raw)
        raw += 1
        rv[:, 2] = np.take(var, raw)
        raw -= np.array(var.shape[1], dtype=np.int32)
        rv[:, 3] = np.take(var, raw)
        return rv

    def get_variable_at_index(self, var, index):
        '''
        Given a list of 2D indices, return the value of var at each index
        '''
        var = var[:]
        rv = np.ma.zeros((index.shape[0], 1), dtype=np.float64)
        mask = np.ma.zeros((index.shape[0], 1), dtype=bool)
        raw = np.ravel_multi_index(index.T, var.shape, mode='clip')
        if (np.ma.is_masked(index)):
            raw.mask = index.mask[:,0] #need to remask raw because ravel_multi_index wipes out index's mask
        rv[:, 0] = np.take(var, raw)
        if var.mask is False:
            mask[:, 0] = np.take(var.mask, raw)
        return np.ma.array(rv, mask=mask)

    def build_kdtree(self, grid='node'):
        """Builds the kdtree for the specified grid"""
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("The scipy package is required to use "
                              "SGrid.locate_nearest\n"
                              " -- nearest neighbor interpolation")
        lon, lat = self._get_grid_vars(grid)
        if lon is None or lat is None:
            raise ValueError("{0}_lon and {0}_lat must be defined in order to "
                             "create and use KDTree for this grid".format(grid))
        lin_points = np.column_stack((lon.ravel(), lat.ravel()))
        self._kd_trees[grid] = cKDTree(lin_points, leafsize=4)

    def build_celltree(self, use_mask=True):
        """
        Builds the celltree across the grid defined by nodes (self.node_lon, self.node_lat)
        If center masking is provided in self.center_mask, it will remove masked cells, and
        take precedence over any node masking for celltree insertion.

        If node masking is provided in self.node_mask and self.center_mask is not provided,
        it will remove masked nodes from the grid, which also removes all adjacent cells

        :param use_mask: If False, ignores all masks and builds the celltree over the raw
                         arrays. Does nothing if self.node_mask or self.center_mask are not
                         present
        """

        try:
            from cell_tree2d import CellTree
        except ImportError:
            raise ImportError("the cell_tree2d package must be installed to use the "
                              "celltree search:\n"
                              "https://github.com/NOAA-ORR-ERD/cell_tree2d/")

        lon, lat = self.node_lon, self.node_lat
        if lon is None or lat is None:
            raise ValueError("node_lon and node_lat must be defined in order to create and "
                             "use CellTree for this grid")

        if (use_mask and
            ((self.node_mask is not None and self.node_mask is not False) or
            (self.center_mask is not None and self.center_mask is not False))):
            if np.any(self.center_mask):
                cell_mask = gen_celltree_mask_from_center_mask(self.center_mask, self.get_padding_slices(self.center_padding))
            else:
                pass

            lin_faces = np.empty(shape=(lon[1::,1::].size,4))
            if lin_faces.shape[0] != cell_mask.size:
                raise ValueError("Could not match mask and faces array length. If padding is in use, please set self.center_padding")

            lon = np.ma.MaskedArray(lon[:].copy())
            lat = np.ma.MaskedArray(lat[:].copy())
            #Water cells grab all nodes that belong to them
            node_mask = np.zeros_like(lon, dtype=np.bool_)
            node_mask[:-1,:-1] += ~cell_mask
            node_mask[:-1,1:] += ~cell_mask
            node_mask[1:,1:] += ~cell_mask
            node_mask[1:,:-1] += ~cell_mask
            node_mask = ~node_mask
            lon.mask = node_mask
            lat.mask = node_mask
            masked_faces_idxs = np.zeros_like(node_mask, dtype=np.int32)
            masked_faces_idxs[node_mask] = -1
            tmp = np.where(~ node_mask.ravel())[0]
            masked_faces_idxs[~node_mask] = np.arange(0,len(tmp))
            lin_faces = np.full(shape=(lon[0:-1,0:-1].size,4), fill_value=-1, dtype=np.int32)
            lin_faces[:,0] = np.ravel(masked_faces_idxs[0:-1, 0:-1])
            lin_faces[:,1] = np.ravel(masked_faces_idxs[0:-1, 1:])
            lin_faces[:,2] = np.ravel(masked_faces_idxs[1:, 1:])
            lin_faces[:,3] = np.ravel(masked_faces_idxs[1:, 0:-1])

            lin_faces[cell_mask.reshape(-1)] = [-1,-1,-1,-1]
            lin_faces = np.ma.masked_less(lin_faces, 0).compressed().reshape(-1,4)
            #need to make a reversal_array. This is an array of the same length
            #as the unmasked nodes that contains the 'true' LINEAR index of the
            #unmasked node. When CellTree gives back an index, it's 'true'
            #index is discovered using this array
            reversal_array = np.where(~cell_mask.reshape(-1))[0].astype(np.int32)
            #append a -1 to preserve -1 entries when back-translating the indices
            reversal_array = np.concatenate((reversal_array, np.array([-1,])))
            self._cell_tree_mask = (node_mask, reversal_array)
        else:
            self._cell_tree_mask = None
            y_size = lon.shape[0]
            x_size = lon.shape[1]
            lin_faces = np.array([np.array([[x, x + 1, x + x_size + 1, x + x_size]
                                            for x in range(0, x_size - 1, 1)]) + y * x_size
                                            for y in range    (0, y_size - 1)])
        lin_faces = np.ascontiguousarray(lin_faces.reshape(-1, 4).astype(np.int32))

        if isinstance(lon, np.ma.MaskedArray) and lon.mask is not False and use_mask:
            lin_nodes = np.ascontiguousarray(np.column_stack((np.ma.compressed(lon[:]),np.ma.compressed(lat[:]))).reshape(-1, 2).astype(np.float64))
        else:
            lin_nodes = np.ascontiguousarray(np.stack((lon, lat), axis=-1).reshape(-1, 2).astype(np.float64))

        self._cell_tree = (CellTree(lin_nodes, lin_faces), lin_nodes, lin_faces)


    def nearest_var_to_points(self,
                              points,
                              variable,
                              indices=None,
                              grid=None,
                              alphas=None,
                              mask=None,
                              slices=None,
                              _memo=False,
                              slice_grid=True,
                              _hash=None,
                              _copy=False):
        if grid is None:
            grid = self.infer_location(variable)
        if indices is None:
            # ind has to be writable
            indices = self.locate_nearest(points, grid, _memo, _copy, _hash)
        [yslice, xslice] = self.get_efficient_slice(points, indices, grid, _memo, _copy, _hash)
        if slices is not None:
            slices = slices + (yslice,)
            slices = slices + (xslice,)
        else:
            slices = (yslice, xslice)

        if self.infer_location(variable) is not None:
            variable = variable[slices]
        if len(variable.shape) > 2:
            raise ValueError("Variable has too many dimensions to \
            associate with grid. Please specify slices.")

        ind = indices.copy() - [yslice.start, xslice.start]
        result = self.get_variable_at_index(variable, ind)
        return result

    def mirror_mask_values(self, values):
        '''
        :param values: The values to be mirrored. Must be a 2D array of 2nd dimension length 2 or 4
        
        :type values array of arrays
        
        NOTE: This function is not currently implemented for interpolation situations of length 4
        '''
        if values.shape[1] == 4:
            raise ValueError('Mirror behavior is not implemented for more than two values')
        mm_values = np.where(values.mask, -values[:,::-1], values)
        return mm_values

    def compute_interpolant(self, values, alphas, mask_behavior=None, check_alphas=True):
        '''
        Interpolation of a value field upon this grid to an interpolant point is
        accomplished by determining the relevant values, and the alphas (proportions)
        that each value contributes. This function computes the interpolated value
        while also applying any relevant masking behavior specified.
        
        mask_behavior can be one of 'mask', 'zero', 'mirror'. 
        If not provided (None) it will default to the self.masked_interpolant_behavior attribute
        (default 'zero')
        
        'mask' will cause interpolation to return a masked value if any of the values
        are masked
        'zero' will replace any masked values with zero before interpolation
        'mirror' will replace any masked values with the negative inverse of the
        other value (eg values of [10, masked] will become [10, -10])
        
        NOTE: This function does NOT mask NaN or Inf values. It is assumed that
        the values have already been masked if necessary.
        NOTE: 'mirror' is not currently implemented for interpolation situations
        with more than two values (such as bilinear interpolation (4 values)).
        NOTE: If you have masked alpha values, these results will still be masked. Masked
        alpha values in this context generally mean the original point was outside the grid.
        
        :param values: The values to be interpolated
        :param alphas: The proportions of each value to be used in the interpolation
        :param mask_behavior: The behavior to be used when masked values are encountered
        :param check_alphas: If True, produces a warning if the sum of alphas is not close to 1 (tol 0.001)
        
        :type values: array of arrays
        :type alphas: array of arrays
        :type mask_behavior: string. One of 'mask', 'zero', 'mirror'
        :type check_alphas: boolean
        
        :return: np.ndarray or np.ma.MaskedArray
        '''
        values = np.asanyarray(values)
        alphas = np.asanyarray(alphas)
        assert np.all(values.shape == alphas.shape) #values and alphas must be the same length
        if check_alphas and not np.isclose(np.sum(alphas, axis=-1), 1, atol=0.001).all():
            warnings.warn('Alphas do not sum to 1. Results may be unexpected.')
        
        if mask_behavior is None:
            mask_behavior = self.masked_interpolant_behavior
        
        if mask_behavior == 'mirror' and len(values) > 2:
            raise ValueError('Mirror behavior is not implemented for more than two values')
        
        if mask_behavior == 'zero':
            filled_values = np.ma.filled(values, 0)
        elif mask_behavior == 'mirror':
            filled_values = self.mirror_mask_values(values)
        else:
            filled_values = values
        
        result = np.sum(filled_values * alphas, axis=-1)
        if hasattr(result, 'mask') or hasattr(filled_values, 'mask'):
            if len(filled_values.shape) == 1 and filled_values.mask.any():
                result = np.ma.masked
            else:
                result.mask = np.any(values.mask, axis=-1)
        return result


    def interpolate_var_to_points(self,
                                  points,
                                  variable,
                                  location=None,
                                  fill_value=0,
                                  indices=None,
                                  alphas=None,
                                  padding=None,
                                  slices=None,
                                  unmask=False,
                                  _memo=False,
                                  _hash=None,
                                  _copy=False):
        """
        Interpolates a variable on one of the grids to an array of points.
        :param points: Nx2 Array of lon/lat coordinates to be interpolated to.

        :param variable: Array-like of values to associate at location on grid
                         (node, center, edge1, edge2). This may be more than a
                         2 dimensional array, but you must pass 'slices' kwarg
                         with appropriate slice collection to reduce it to 2 dimensions.

        :param location: One of ('node', 'center', 'edge1', 'edge2', 'face').
                         'edge1' is conventionally associated with the 'vertical' edges
                         and likewise 'edge2' with the 'horizontal'. Determines type of
                         interpolation, see below for details

        :param fill_value: If masked values are encountered in interpolation, this value
                           takes the place of the masked value

        :param indices: If computed already, array of Nx2 cell indices can be passed in
                        to increase speed.

        :param alphas: If computed already, array of alphas can be passed in to increase
                       speed.
                       
        :param unmask: If true, unmask results, using fill value for masked values.

        Depending on the location specified, different interpolation will be used.

        For 'center', no interpolation

        For 'edge1' or 'edge2', interpolation is linear, edge to edge across the cell

        For 'node', interpolation is bilinear from the four nodes of each cell

        The variable specified may be any array-like.
        - With a numpy array:

        sgrid.interpolate_var_to_points(points, sgrid.u[time_idx, depth_idx])
        - With a raw netCDF Variable:

        sgrid.interpolate_var_to_points(points, nc.variables['u'], slices=[time_idx, depth_idx])

        If you have pre-computed information, you can pass it in to avoid unnecessary
        computation and increase performance.

        - ind = # precomputed indices of points. This may be a masked array

        - alphas = # precomputed alphas (useful if interpolating to the same points frequently)

        sgrid.interpolate_var_to_points(points, sgrid.u, indices=ind, alphas=alphas,
        slices=[time_idx, depth_idx])

        """
        # eventually should remove next line once celltree can support it
        points = points.reshape(-1, 2)

        ind = indices
        breakpoint()
        if hash is None:
            _hash = self._hash_of_pts(points)

        if location is None:
            location = self.infer_location(variable)
            warnings.warn('No location provided. Assuming data is on {0}'.format(location))

        if ind is None:
            # ind has to be writable
            ind = self.locate_faces(points, _memo, _copy, _hash)
            if (ind.mask).all():
                return np.ma.masked_all((points.shape[0],1))

        if self._l_coeffs is None:
            self._compute_transform_coeffs()

        logical_coords = self.geo_to_logical(points, indices=ind)

        if alphas is None:
            #Better name for this would be per_cell_logical_offset
            alphas = per_cell_log_offset = logical_coords - ind

        if padding is None:
            padding = self.get_padding_by_location(location)

        #Setup done. Determine slicing and zero-align indices and slice variable

        idxs = self.apply_padding_to_idxs(ind.copy(), padding=padding)
        [xslice, yslice] = self.get_efficient_slice(indices=idxs, location=location, _memo=_memo, _copy=_copy, _hash=_hash)
        if slices is not None:
            slices = slices + (xslice,)
            slices = slices + (yslice,)
        else:
            slices = (xslice, yslice)
        zero_aligned_idxs = idxs.copy() - [xslice.start, yslice.start]
        var = variable[slices]
        if len(var.shape) > 2:
            raise ValueError("Variable has too many dimensions to \
            associate with grid. Please specify slices.")
        if not isinstance(var, np.ma.MaskedArray):
            #this is because MFDataset isn't always returning a masked array, the same as pre netCDF 1.4 behavior
            #Until they fix this, we need to ensure it gets masked.
            var = np.ma.MaskedArray(var, mask=False)

        if location in center_alternate_names:
            #No interpolation across the cell
            result = self.get_variable_at_index(var, zero_aligned_idxs)
            if unmask:
                result = result.filled(fill_value)
            return result

        elif location in edge1_alternate_names:
            #interpolate as a uniform gradient from 'left side' to 'right side'
            center_idxs = self.apply_padding_to_idxs(ind.copy(), padding=self.get_padding_by_location('center'))
            if self.center_mask is None:
                cm = np.zeros((self.node_lon.shape[0] - 1, self.node_lon.shape[1] - 1)).astype(np.bool_)
                cm = np.ma.MaskedArray(cm, mask=False)
            else:
                cm = gen_celltree_mask_from_center_mask(self.center_mask, np.s_[:])
                cm = np.ma.MaskedArray(cm, mask=False)

            u2_offset = [0, 1]
            #get the alpha for the relevant dimension. The value of alpha is the
            #logical distance from the RIGHT. AKA, the proportion for value2
            #per_cell_log_offset[:,0]
            #because per_cell_log_offset has the correct shape for upcoming calculations
            #discard the other dimension and use it to store the complement alpha value
            alpha_dim_idx = 0
            alpha = per_cell_log_offset[:, ::-1] #swap the value we want into alpha2
            alpha[:, 0] = 1 - alpha[:, 1] #compute alpha1
            breakpoint()

            u1 = self.get_variable_at_index(var, zero_aligned_idxs)
            m1 = np.logical_xor(self.get_variable_at_index(cm, center_idxs), self.get_variable_at_index(cm, center_idxs - u2_offset))
            u1.mask = np.logical_or(u1.mask, m1)
            
            u2 = self.get_variable_at_index(var, zero_aligned_idxs + u2_offset)
            m2 = np.logical_xor(self.get_variable_at_index(cm, center_idxs), self.get_variable_at_index(cm, center_idxs + u2_offset))
            u2.mask = np.logical_or(u2.mask, m2)

            result = self.compute_interpolant(np.ma.concatenate((u1, u2), axis=-1), alpha)

        elif location in edge2_alternate_names:
            #interpolate as a uniform gradient from 'bottom' to 'top'
            center_idxs = self.apply_padding_to_idxs(ind.copy(), padding=self.get_padding_by_location('center'))
            if self.center_mask is None:
                cm = np.zeros((self.node_lon.shape[0] - 1, self.node_lon.shape[1] - 1)).astype(np.bool_)
                cm = np.ma.MaskedArray(cm, mask=False)
            else:
                cm = gen_celltree_mask_from_center_mask(self.center_mask, np.s_[:])
                cm = np.ma.MaskedArray(cm, mask=False)

            v2_offset = [1, 0]
            #get the alpha for the relevant dimension. The value of alpha is the
            #logical distance from the RIGHT. AKA, the proportion for value2
            #per_cell_log_offset[:,1]
            #because per_cell_log_offset has the correct shape for upcoming calculations
            #discard the other dimension and use it to store the complement alpha value
            alpha_dim_idx = 1
            alpha = per_cell_log_offset #no swap needed, alpha2 is already in the correct place
            alpha[:, 0] = 1 - alpha[:, 1] #compute alpha1
            breakpoint()

            v1 = self.get_variable_at_index(var, zero_aligned_idxs)
            m1 = np.logical_xor(self.get_variable_at_index(cm, center_idxs), self.get_variable_at_index(cm, center_idxs - v2_offset))
            v1.mask = np.logical_or(v1.mask, m1)
            
            v2 = self.get_variable_at_index(var, zero_aligned_idxs + v2_offset)
            m2 = np.logical_xor(self.get_variable_at_index(cm, center_idxs), self.get_variable_at_index(cm, center_idxs + v2_offset))
            v2.mask = np.logical_or(v2.mask, m2)

            result = self.compute_interpolant(np.ma.concatenate((v1, v2), axis=-1), alpha)

        elif location in node_alternate_names:
            l = per_cell_log_offset[:, 0]
            m = per_cell_log_offset[:, 1]

            #Each corner alpha is the ratio Area_opposite/Area_total
            #Since Area_total is unit square (1), each corner is simply Area_opposite
            aa = 1 - l - m + l * m
            ab = m - l * m
            ac = l * m
            ad = l - l * m
            
            assert np.allclose(aa + ab + ac + ad, 1)

            alphas = np.stack((aa, ab, ac, ad), axis=-1)
            vals = self.get_variable_by_index(var, zero_aligned_idxs)
            vals *= alphas
            result = np.sum(vals, axis=1)
        else:
            raise ValueError('invalid location name')

        if unmask:
            result = result.filled(fill_value)
        return result

    interpolate = interpolate_var_to_points

    def geo_to_logical(self,
                       points,
                       indices=None,
                       _memo=False,
                       _copy=False,
                       _hash=None):
        """
        Given a list of lon/lat points, converts them to l/m coordinates in
        logical cell space.
        """
        if _memo:
            if _hash is None:
                _hash = self._hash_of_pts(points)
            result = self._get_memoed(points, self._log_ind_memo_dict, _copy, _hash)
            if result is not None:
                return result

        if self._l_coeffs is None:
            self._compute_transform_coeffs()

        if indices is None:
            indices = self.locate_faces(points,
                                        _memo=_memo,
                                        _copy=_copy,
                                        _hash=_hash)

        a = self._l_coeffs[indices[:, 0], indices[:, 1]]
        b = self._m_coeffs[indices[:, 0], indices[:, 1]]
        (l, m) = self.x_to_l(points[:, 0], points[:, 1], a, b)

        result = indices.copy() + np.stack((l, m), axis=-1)

        if _memo:
            self._add_memo(points, result, self._log_ind_memo_dict, _copy, _hash)

        return result

    @staticmethod
    def x_to_l(x, y, a, b):
        """
        Params:
        x: x coordinate of point
        y: y coordinate of point
        a: x coefficients
        b: y coefficients

        Returns:
        (l,m) - coordinate in logical space to use for interpolation

        Eqns:
        m = (-bb +- sqrt(bb^2 - 4*aa*cc))/(2*aa)
        l = (l-a1 - a3*m)/(a2 + a4*m)
        """

        def quad_eqn(l, m, t, aa, bb, cc):
            """
            solves the following eqns for m and l
            m = (-bb +- sqrt(bb^2 - 4*aa*cc))/(2*aa)
            l = (l-a1 - a3*m)/(a2 + a4*m)
            """
            if len(aa) == 0:
                return
            k = bb * bb - 4 * aa * cc
            k = np.ma.masked_less(k, 0)

            det = np.ma.sqrt(k)
            m1 = (-bb - det) / (2 * aa)
            l1 = (x[t] - a[0][t] - a[2][t] *
                    m1) / (a[1][t] + a[3][t] * m1)

            m2 = (-bb + det) / (2 * aa)
            l2 = (x[t] - a[0][t] - a[2][t] *
                    m2) / (a[1][t] + a[3][t] * m2)

            t1 = np.logical_or(l1 < 0, l1 > 1)
            t2 = np.logical_or(m1 < 0, m1 > 1)
            t3 = np.logical_or(t1, t2)

            m[t] = np.choose(t3, (m1, m2))
            l[t] = np.choose(t3, (l1, l2))

        a = a.T
        b = b.T
        aa = a[3] * b[2] - a[2] * b[3]
        bb = a[3] * b[0] - a[0] * b[3] + a[1] * \
            b[2] - a[2] * b[1] + x * b[3] - y * a[3]
        cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]
        m = np.zeros(bb.shape)
        l = np.zeros(bb.shape)

        t = aa[:] == 0

        # Attempts to solve the simpler linear case first.
        with np.errstate(invalid='ignore'):
            m[t] = -cc[t] / bb[t]
            l[t] = (x[t] - a[0][t] - a[2][t] * m[t]) / (a[1][t] + a[3][t] * m[t])
        # now solve the quadratic cases
        quad_eqn(l, m, ~t, aa[~t], bb[~t], cc[~t])

        return (l, m)


class SGridAttributes(object):
    """
    Class containing methods to help with getting the
    attributes for either SGrid.

    """

    def __init__(self, nc, topology_dim, topology_variable):
        self.nc = nc
        self.ncd = NetCDFDataset(self.nc)
        self.topology_dim = topology_dim
        self.topology_variable = topology_variable
        self.topology_var = self.nc.variables[self.topology_variable]

    def get_dimensions(self):
        ds_dims = self.nc.dimensions
        grid_dims = [(ds_dim, len(ds_dims[ds_dim])) for ds_dim in ds_dims]
        return grid_dims

    def get_topology_var(self):
        grid_topology_var = find_grid_topology_var(self.nc)
        return grid_topology_var

    def get_attr_dimension(self, attr_name):
        try:
            attr_dim = getattr(self.topology_var, attr_name)
        except AttributeError:
            attr_dim = None
            attr_padding = None
        else:
            attr_dim_padding = parse_padding(attr_dim, self.topology_variable)
            attr_padding = attr_dim_padding
        return attr_dim, attr_padding

    def get_attr_coordinates(self, attr_name):
        try:
            attr_coordinates_raw = getattr(self.topology_var, attr_name)
        except AttributeError:
            location_name = attr_name.split('_')[0]
            attr_coordinates = self.ncd.find_coordinates_by_location(location_name, self.topology_dim)  # noqa
        else:
            attr_coordinates_val = attr_coordinates_raw.split(' ')
            attr_coordinates = tuple(attr_coordinates_val)
        return attr_coordinates

    def get_node_coordinates(self):
        node_dims = self.topology_var.node_dimensions
        node_dimensions = node_dims
        try:
            node_coordinates = self.topology_var.node_coordinates
        except AttributeError:
            grid_cell_node_vars = self.ncd.find_node_coordinates(node_dimensions)  # noqa
            node_coordinates = grid_cell_node_vars
        else:
            node_coordinate_val = node_coordinates.split(' ')
            node_coordinates = tuple(node_coordinate_val)
        return node_dimensions, node_coordinates

    def get_variable_attributes(self, sgrid):
        dataset_variables = []
        grid_variables = []
        nc_variables = self.nc.variables
        for nc_variable in nc_variables:
            nc_var = nc_variables[nc_variable]
            sgrid_var = SGridVariable.create_variable(nc_var, sgrid)
            setattr(sgrid, sgrid_var.variable, sgrid_var)
            dataset_variables.append(nc_var.name)
            if hasattr(nc_var, 'grid'):
                grid_variables.append(nc_var.name)
        sgrid.variables = dataset_variables
        sgrid.grid_variables = grid_variables

    def get_angles(self):
        angles = self.nc.variables.get('angle')
        if not angles:
            # FIXME: Get rid of pair_arrays.
            center_lon, center_lat = self.get_cell_center_lat_lon()
            cell_centers = pair_arrays(center_lon, center_lat)
            centers_start = cell_centers[..., :-1, :]
            centers_end = cell_centers[..., 1:, :]
            angles = calculate_angle_from_true_east(centers_start, centers_end)
        return angles

    def get_cell_center_lat_lon(self):
        try:
            grid_cell_center_lon_var, grid_cell_center_lat_var = self.get_attr_coordinates('face_coordinates')  # noqa
        except TypeError:
            center_lat, center_lon = None, None
        else:
            center_lat = self.nc[grid_cell_center_lat_var][:]
            center_lon = self.nc[grid_cell_center_lon_var][:]
        return center_lon, center_lat

    def get_cell_node_lat_lon(self):
        try:
            node_lon_var, node_lat_var = self.get_node_coordinates()[1]
        except TypeError:
            node_lon, node_lat = None, None
        else:
            node_lat = self.nc[node_lat_var][:]
            node_lon = self.nc[node_lon_var][:]
        return node_lon, node_lat

    def get_cell_edge1_lat_lon(self):
        try:
            edge1_lon_var, edge1_lat_var = self.get_attr_coordinates('edge1_coordinates')
        except:
            edge1_lon, edge1_lat = None, None
        else:
            edge1_lon = self.nc[edge1_lon_var][:]
            edge1_lat = self.nc[edge1_lat_var][:]
        return edge1_lon, edge1_lat

    def get_cell_edge2_lat_lon(self):
        try:
            edge2_lon_var, edge2_lat_var = self.get_attr_coordinates('edge2_coordinates')
        except TypeError:
            edge2_lon, edge2_lat = None, None
        else:
            edge2_lon = self.nc[edge2_lon_var][:]
            edge2_lat = self.nc[edge2_lat_var][:]
        return edge2_lon, edge2_lat

    def get_masks(self, node, center, edge1, edge2):
        node_shape = node.shape if node is not None and node.shape else None
        center_shape = center.shape if center is not None and center.shape else None
        edge1_shape = edge1.shape if edge1 is not None and edge1.shape else None
        edge2_shape = edge2.shape if edge2 is not None and edge2.shape else None
        mask_candidates = [var.name for var in self.nc.variables.values() if 'mask' in var.name or (hasattr(var, 'long_name') and 'mask' in var.long_name)]
        node_mask = center_mask = edge1_mask = edge2_mask = None
        for mc in mask_candidates:
            if node_shape and self.nc.variables[mc].shape == node_shape and node_mask is None:
                node_mask = self.nc.variables[mc]
            if center_shape and self.nc.variables[mc].shape == center_shape and center_mask is None:
                center_mask = self.nc.variables[mc]
            if edge1_shape and self.nc.variables[mc].shape == edge1_shape and edge1_mask is None:
                edge1_mask = self.nc.variables[mc]
            if edge2_shape and self.nc.variables[mc].shape == edge2_shape and edge2_mask is None:
                edge2_mask = self.nc.variables[mc]

        return node_mask, center_mask, edge1_mask, edge2_mask



def load_grid(nc):
    """
    Get a SGrid object from a netCDF4.Dataset or file/URL.

    :param str or netCDF4.Dataset nc: a netCDF4 Dataset or URL/filepath
                                       to the netCDF file
    :return: SGrid object
    :rtype: sgrid.SGrid

    """
    if isinstance(nc, Dataset):
        pass
    else:
        nc = Dataset(nc, 'r')

    return SGrid.load_grid(nc)
