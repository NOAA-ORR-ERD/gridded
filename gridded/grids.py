
from gridded.pysgrid.sgrid import SGrid
from gridded.pyugrid.ugrid import UGrid
import numpy as np

from gridded.utilities import get_dataset, parse_filename_dataset_args


class GridBase(object):
    '''
    Base object for grids to share common behavior
    '''
    _def_count = 0

    def __init__(self,
                 filename=None,
                 *args,
                 **kwargs):
        """
        Init common to all Grid types. This initializer will take all the kwargs of both
        pyugrid.UGrid and pysgrid.SGrid. See their documentation for details

        :param filename: Name of the file this grid was constructed from, if available.
        """
        if 'name' in kwargs:
            self.name = kwargs['name']
        else:
            self.name = self.__class__.__name__ + '_' + str(type(self)._def_count)
        self.filename = filename
        type(self)._def_count += 1

        super(GridBase, self).__init__(**kwargs)

    @classmethod
    def from_netCDF(cls, *args, **kwargs):
        kwargs['grid_type'] = cls
        return Grid.from_netCDF(*args, **kwargs)

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None,):
        '''
        This function is the top level 'search for attributes' function. If there are any
        common attributes to all potential grid types, they will be sought here.

        This function returns a dict, which maps an attribute name to a netCDF4
        Variable or numpy array object extracted from the dataset. When called from
        Grid_U or Grid_S, this function should provide all the kwargs needed to
        create a valid instance.
        '''
        gf_vars = dataset.variables if dataset is not None else get_dataset(filename).variables
        gf_vars = dict([(k.lower(), v) for k, v in gf_vars.items()])
        init_args = {}
        gt = {}
        init_args['filename'] = filename
        node_attrs = ['node_lon', 'node_lat']
        node_coord_names = [['node_lon', 'node_lat'],
                            ['lon', 'lat'],
                            ['lon_psi', 'lat_psi'],
                            ['longitude', 'latitude']]
        composite_node_names = ['nodes', 'node']
        if grid_topology is None:
            for n1, n2 in node_coord_names:
                if n1 in gf_vars and n2 in gf_vars:
                    init_args[node_attrs[0]] = gf_vars[n1][:]
                    init_args[node_attrs[1]] = gf_vars[n2][:]
                    gt[node_attrs[0]] = n1
                    gt[node_attrs[1]] = n2
                    break
            if node_attrs[0] not in init_args:
                for n in composite_node_names:
                    if n in gf_vars:
                        v = gf_vars[n][:].reshape(-1, 2)
                        init_args[node_attrs[0]] = v[:, 0]
                        init_args[node_attrs[1]] = v[:, 1]
                        gt['node_coordinates'] = n
                        break
            if node_attrs[0] not in init_args:
                raise ValueError('Unable to find node coordinates.')
        else:
            for n, v in grid_topology.items():
                if n in node_attrs:
                    init_args[n] = gf_vars[v][:]
                if n in composite_node_names:
                    v = gf_vars[n][:].reshape(-1, 2)
                    init_args[node_attrs[0]] = v[:, 0]
                    init_args[node_attrs[1]] = v[:, 1]
        return init_args, gt

    @property
    def shape(self):
        return self.node_lon.shape

    def _write_grid_to_file(self, pth):
        self.save_as_netcdf(pth)

    def import_variable(self, variable, location='node'):
        """
        Takes a Variable or VectorVariable and interpolates the data onto this grid.
        You may pass a location ('nodes', 'faces', 'edge1', 'edge2) and the
        variable will be interpolated there if possible
        If no location is passed, the variable will be interpolated to the
        nodes of this grid. If the Variable's grid and this grid are the same, this
        function will return the Variable unchanged.

        If this grid covers area that the source grid does not, all values
        in this area will be masked. If regridding from cell centers to the nodes,
        The values of any border point not within will be equal to the value at the
        center of the border cell.
        """

        raise NotImplementedError("GridBase cannot interpolate variables to itself")


class Grid_U(GridBase, UGrid):

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None):

        gf_vars = dataset.variables if dataset is not None else get_dataset(filename).variables
        gf_vars = dict([(k.lower(), v) for k, v in gf_vars.items()])
        # Get superset attributes
        init_args, gt = super(Grid_U, cls)._find_required_grid_attrs(filename=filename,
                                                                     dataset=dataset,
                                                                     grid_topology=grid_topology)

        face_attrs = ['faces']
        face_var_names = ['faces', 'tris', 'nv', 'ele', 'nele']
        if grid_topology is None:
            for n in face_var_names:
                if n in gf_vars:
                    init_args[face_attrs[0]] = gf_vars[n][:]
                    gt[face_attrs[0]] = n
                    break
            if face_attrs[0] not in init_args:
                raise ValueError('Unable to find face connectivity array.')

        else:
            for n, v in grid_topology.items():
                if n in face_attrs:
                    init_args[n] = gf_vars[v][:]
                    break
        # fixme: This is assuming that the array will be in Fortran order and index
        #        from 1, or in C order and index from 0
        #        Those are actually independent concepts!
        if init_args['faces'].shape[0] == 3:
            init_args['faces'] = np.ascontiguousarray(np.array(init_args['faces']).T - 1)

        return init_args, gt

    # @classmethod
    # def gen_from_quads(cls, nodes):
    #     # Fixme: this looks incomplete -- used anywhere?
    #     if not len(nodes.shape) == 3:
    #         raise ValueError('Nodes of a quad grid must be 2 dimensional')
    #     lin_nodes = None
    #     if isinstance(nodes, np.ma.MaskedArray):
    #         lin_nodes = nodes.reshape(-1, 2)[nodes]


class Grid_S(GridBase, SGrid):

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None):

        # THESE ARE ACTUALLY ALL OPTIONAL. This should be migrated when optional attributes
        #   are dealt with
        # Get superset attributes
        # get_datset is not defined -- this must not be used
        gf_vars = dataset.variables if dataset is not None else get_dataset(filename).variables
        gf_vars = dict([(k.lower(), v) for k, v in gf_vars.items()])
        init_args, gt = super(Grid_S, cls)._find_required_grid_attrs(filename,
                                                                     dataset=dataset,
                                                                     grid_topology=grid_topology)

        center_attrs = ['center_lon', 'center_lat']
        edge1_attrs = ['edge1_lon', 'edge1_lat']
        edge2_attrs = ['edge2_lon', 'edge2_lat']
        node_mask = 'node_mask'
        center_mask = 'center_mask'
        edge1_mask = 'edge1_mask'
        edge2_mask = 'edge2_mask'

        center_coord_names = [['center_lon', 'center_lat'], ['lon_rho', 'lat_rho'], ['lonc', 'latc']]
        edge1_coord_names = [['edge1_lon', 'edge1_lat'], ['lon_u', 'lat_u']]
        edge2_coord_names = [['edge2_lon', 'edge2_lat'], ['lon_v', 'lat_v']]
        node_mask_names = ['mask_psi']
        center_mask_names = ['mask_rho']
        edge1_mask_names = ['mask_u']
        edge2_mask_names = ['mask_v']

        if grid_topology is None:
            for attr, names, maskattr, maskname in zip(
                    (center_attrs, edge1_attrs, edge2_attrs),
                    (center_coord_names, edge1_coord_names, edge2_coord_names),
                    (center_mask, edge1_mask, edge2_mask),
                    (center_mask_names, edge1_mask_names, edge2_mask_names)):
                for n1, n2 in names:
                    if n1 in gf_vars and n2 in gf_vars:
                        mask = False
                        # for n in maskname:
                        #     if n in gf_vars:
                        #         mask = gen_mask(gf_vars[n])
                        a1 = gf_vars[n1][:]
                        a2 = gf_vars[n2][:]
                        init_args[attr[0]] = a1
                        init_args[attr[1]] = a2
                        if maskname[0] in gf_vars:
                            init_args[maskattr] = gf_vars[maskname[0]]
                            gt[maskattr] = maskname[0]
                        gt[attr[0]] = n1
                        gt[attr[1]] = n2
                        break
            if 'node_lon' in init_args and 'node_lat' in init_args:
                mask = False  # fixme -- is mask used??
                for name in node_mask_names:
                    if name in gf_vars:
                        init_args[node_mask] = gf_vars[name]
                gt[node_mask] = name

        else:
            for n, v in grid_topology.items():
                if n in center_attrs + edge1_attrs + edge2_attrs and v in gf_vars:
                    init_args[n] = gf_vars[v][:]
        return init_args, gt


class Grid_R(GridBase):
    """
    Rectangular Grid

    lon and lat of the nodes are vectors
    """

    def __init__(self,
                 node_lon=None,
                 node_lat=None,
                 grid_topology=None,
                 node_dimensions=None,
                 node_coordinates=None,
                 *args,
                 **kwargs):
        """
        :param node_lon=None: vector of the node longitudes
        :param node_lat=None: vector of the node latitudes
        :param grid_topology=None: ????
        :param node_dimensions=None: (should only be required for netcdf)
        :param node_coordinates=None:  ?????
        """
        self.node_lon = node_lon
        self.node_lat = node_lat
        self.grid_topology = grid_topology
        if self.grid_topology is not None:
            self.dimensions = [grid_topology['node_lat'], grid_topology['node_lon']]
        else:
            self.dimensions = ['lat', 'lon']
        self.node_dimensions = node_dimensions
        self.node_coordinates = node_coordinates

        super(Grid_R, self).__init__(*args, **kwargs)

    @classmethod
    def _find_required_grid_attrs(cls, filename, dataset=None, grid_topology=None):

        # THESE ARE ACTUALLY ALL OPTIONAL. This should be migrated when optional attributes
        # are dealt with
        # Get superset attributes
        gf_vars = dataset.variables if dataset is not None else get_dataset(filename).variables
        gf_vars = dict([(k.lower(), v) for k, v in gf_vars.items()])
        init_args, gt = super(Grid_R, cls)._find_required_grid_attrs(filename,
                                                                     dataset=dataset,
                                                                     grid_topology=grid_topology)

        # Grid_R only needs node_lon and node_lat.
        # However, they must be a specific shape (1D)
        node_lon = init_args['node_lon']
        node_lat = init_args['node_lat']
        if len(node_lon.shape) != 1:
            raise ValueError('Too many dimensions in node_lon. '
                             'Must be 1D, was {0}D'.format(len(node_lon.shape)))
        if len(node_lat.shape) != 1:
            raise ValueError('Too many dimensions in node_lat. '
                             'Must be 1D, was {0}D'.format(len(node_lat.shape)))
        return init_args, gt

    @property
    def nodes(self):
        return np.stack((np.meshgrid(self.node_lon, self.node_lat)), axis=-1)

    @property
    def center_lon(self):
        return (self.node_lon[0:-1] + self.node_lon[1:]) / 2

    @property
    def center_lat(self):
        return (self.node_lat[0:-1] + self.node_lat[1:]) / 2

    @property
    def centers(self):
        return np.stack((np.meshgrid(self.center_lon, self.center_lat)), axis=-1)

    def locate_faces(self,
                     points):
        """
        Returns the node grid indices, one per point.

        Points that are not on the node grid will have an index of -1

        If a single point is passed in, a single index will be returned.
        If a sequence of points is passed in an array of indexes will be returned.

        :param points:  The points that you want to locate -- (lon, lat). If the shape of point
                        is 1D, function will return a scalar index. If it is 2D, it will return
                        a 1D array of indices.
        :type points: array-like containing one or more points: shape (2,) for one point,
                      shape (N, 2) for more than one point.
        """
        points = np.asarray(points, dtype=np.float64)
        just_one = (points.ndim == 1)
        points = points.reshape(-1, 2)
        lons = points[:, 0]
        lats = points[:, 1]
        lon_idxs = np.digitize(lons, self.node_lon) - 1
        for i, n in enumerate(lon_idxs):
            if n == len(self.node_lon) - 1:
                lon_idxs[i] = -1
#             if n == 0 and not lons[i] < self.node_lon.max() and not lons[i] >= self.node_lon.min():
#                 lon_idxs[i] = -1
        lat_idxs = np.digitize(lats, self.node_lat) - 1
        for i, n in enumerate(lat_idxs):
            if n == len(self.node_lat) - 1:
                lat_idxs[i] = -1
#             if n == 0 and not lats[i] < self.node_lat.max() and not lats[i] >= self.node_lat.min():
#                 lat_idxs[i] = -1
        idxs = np.column_stack((lon_idxs, lat_idxs))
        idxs[:, 0] = np.where(idxs[:, 1] == -1, -1, idxs[:, 0])
        idxs[:, 1] = np.where(idxs[:, 0] == -1, -1, idxs[:, 1])
        if just_one:
            res = idxs[0]
            return res
        else:
            return idxs

    def lonlat_to_yx(self, variable):
        '''
        The RegualarGridInterpolator needs to have its two dimensions x and y be associated
        correctly with lon and lat (or vice versa). The order depends on the orientation in
        the variable

        if the variable provided does not have a dimensions attribute,
        it will use the dimensions arg
        '''
        retval = (self.node_lat, self.node_lon)
        var_shape = variable.shape[-2::]
        grid_shape = (len(self.node_lat), len(self.node_lon))
        if (hasattr(variable, 'dimensions')):
            if not all([k in self.dimensions for k in variable.dimensions[-2:]]):
                raise ValueError('Dimension provided by variable is not compatible \
                                 with this Grid_R object. Provided: {0} \
                                 self.dimensions: {1}'.format(variable.dimensions,
                                                              self.dimensions))
            var_dims = variable.dimensions[-2:]  # assume the last two are the lon/lat x/y
        else:
            var_dims = self.dimensions
        # self.dimensions is always [y(lat), x(lon)],
        # so if var.dimensions is [lon, lat] we need
        # to reverse x/y association
        if not all([dlen in grid_shape for dlen in var_shape]):
            raise ValueError('Incompatible dimensions. '
                             'Variable: {0}, Grid_R: {1}'.format(variable.shape, grid_shape))

        if hasattr(variable, 'dimensions'):
            if var_dims[0] == self.dimensions[1]:  # case 2, dims provided, dims swapped
                retval = retval[::-1]
            # else: case 1, no change
        else:
            if var_shape[0] == var_shape[1]:  # case 4, dims not provided, dim length same
                raise ValueError('Provided square variable with no dimensions attribute')
            # case 3, dims not provided, dim length different
            if var_shape[0] == len(self.node_lon):
                retval = retval[::-1]

        return retval

    def interpolate_var_to_points(self,
                                  points,
                                  variable,
                                  method='linear',
                                  indices=None,
                                  slices=None,
                                  mask=None,
                                  **kwargs):
        try:
            from scipy.interpolate import RegularGridInterpolator
        except ImportError:
            raise ImportError("The scipy package is required to use "
                              "Grid_R.interpolate_var_to_points\n"
                              " -- interpolating a regular grid")
        points = np.asarray(points, dtype=np.float64)
        just_one = (points.ndim == 1)
        points = points.reshape(-1, 2)
        points_y = points[:, 1] #lat
        points_x = points[:, 0] #lon
        y, x = self.lonlat_to_yx(variable)
        
        if slices is not None:
            variable = variable[slices]
            if np.ma.isMA(variable):
                variable = variable.filled(0)  # eventually should use Variable fill value
        
        # idx_y = np.digitize(points_y, y) - 1
        # idx_x = np.digitize(points_x, x) - 1
        
        # v1 = variable[idx_y, idx_x]
        # v2 = variable[idx_y, idx_x + 1]
        # v3 = variable[idx_y + 1, idx_x]
        # v4 = variable[idx_y + 1, idx_x + 1]
        
        # ay = 1- (y[idx_y + 1] - points_y) / (y[idx_y + 1] - y[idx_y])
        # ax = 1- (x[idx_x + 1] - points_x) / (x[idx_x + 1] - x[idx_x])
        
        # vx1 = v1 + ax * (v2 - v1)
        # vx2 = v3 + ax * (v4 - v3)
        
        # iv = vx1 + ay * (vx2 - vx1)
        # if just_one:
        #     return iv[0]
        # else:
        #     return iv
        
        interp_func = RegularGridInterpolator((y, x),
                                              variable,
                                              method=method,
                                              bounds_error=False,
                                              fill_value=0)
        if y is self.node_lon:
            vals = interp_func(points, method=method)
        else:
            vals = interp_func(points[:, ::-1], method=method)
            
        breakpoint()
        if just_one:
            return vals[0]
        else:
            return vals

    def infer_location(self, variable):
        """
        fixme: should first look for "location" attribute.

        But now we are checking variable dimensions to which part
        of the grid it is on.
        """
        if hasattr(variable, 'location'):
            return variable.location
        
        x_len = len(self.node_lon)
        y_len = len(self.node_lat)
        node_shape = np.array((y_len, x_len))
        var_shape = np.array(variable.shape[-2:])
        if np.isclose(node_shape[::-1], var_shape, atol=1).all():
            node_shape = node_shape[::-1] # reverse the dimensions
        difference = (var_shape - node_shape).tolist()
        if (difference == [1, 1] or difference == [-1, -1]) and self.center_lon is not None:
            return 'center'
        elif difference == [1, 0] and self.edge1_lon is not None:
            return 'edge1'
        elif difference == [0, 1] and self.edge2_lon is not None:
            return 'edge2'
        elif difference == [0, 0] and self.node_lon is not None:
            return 'node'
        else:
            return None


class Grid(object):
    '''
    Factory class that generates grid objects. Also handles common
    loading and parsing operations
    '''

    def __init__(self):
        '''
        Init common to all Grid types. This constructor will take all the kwargs of both
        pyugrid.UGrid and pysgrid.SGrid. See their documentation for details

        :param filename: Name of the file this grid was constructed from, if available.
        '''
        raise NotImplementedError("Grid is not meant to be instantiated. "
                                  "Please use the from_netCDF function. "
                                  "or initialize the type of grid you want directly")

    @staticmethod
    def _load_grid(filename, grid_type, dataset=None):
        '''
        Redirect to grid-specific loading routine.
        '''
        if issubclass(grid_type, UGrid):
            return grid_type.from_ncfile(filename=filename, dataset=dataset)
        elif issubclass(grid_type, SGrid):
            ds = get_dataset(filename, dataset)
            g = grid_type.load_grid(ds)
            g.filename = filename
            return g
        else:
            return grid_type.from_ncfile(filename)
        pass

    @staticmethod
    def from_netCDF(filename=None,
                    dataset=None,
                    data_file=None,
                    grid_file=None,
                    grid_type=None,
                    grid_topology=None,
                    _default_types=(('ugrid', Grid_U),
                                    ('sgrid', Grid_S),
                                    ('rgrid', Grid_R)),
                    *args,
                    **kwargs):
        '''
        :param filename: File containing a grid

        :param dataset: Takes precedence over filename, if provided.

        :param grid_type: Must be provided if Dataset does not have a 'grid_type' attribute,
                          or valid topology variable

        :param grid_topology: A dictionary mapping of grid attribute to variable name.
                              Takes precedence over discovered attributes

        :param kwargs: All kwargs to SGrid, UGrid, or RGrid are valid, and take precedence
                       over all.

        :returns: Instance of Grid_U, Grid_S, or Grid_R
        '''
        df, dg = parse_filename_dataset_args(filename=filename,
                                             dataset=dataset,
                                             data_file=data_file,
                                             grid_file=grid_file)

        cls = grid_type
        if (grid_type is None or
                isinstance(grid_type, str) or
                not issubclass(grid_type, GridBase)):
            cls = Grid._get_grid_type(dg, grid_type, grid_topology, _default_types)

        # if grid_topology is passed in, don't look for the variable
        if not grid_topology:
            compliant = Grid._find_topology_var(None, dg)
        else:
            compliant = None

        if compliant is not None:
            c = Grid._load_grid(filename, cls, dg)
            c.grid_topology = compliant.__dict__
        else:
            init_args, gt = cls._find_required_grid_attrs(filename,
                                                          dataset=dg,
                                                          grid_topology=grid_topology)
            c = cls(grid_topology=gt, **init_args)
        return c

    @staticmethod
    def _get_grid_type(dataset,
                       grid_type=None,
                       grid_topology=None,
                       _default_types=(('ugrid', Grid_U),
                                       ('sgrid', Grid_S),
                                       ('rgrid', Grid_R))):
        # fixme: this logic should probably be deferred to
        #        the grid type code -- that is, ask each grid
        #        type if this dataset is its type.
        #
        #        It also should be refactored to start with the standards
        #        and maybe have a pedantic mode where it won't load non-standard
        #        files

        if _default_types is None:
            _default_types = dict()
        else:
            _default_types = dict(_default_types)

        Grid_U = _default_types.get('ugrid', None)
        Grid_S = _default_types.get('sgrid', None)
        Grid_R = _default_types.get('rgrid', None)

        sgrid_names = ['sgrid', 'pygrid_s', 'staggered', 'curvilinear', 'roms']
        ugrid_names = ['ugrid', 'pygrid_u', 'triangular', 'unstructured']
        rgrid_names = ['rgrid', 'regular', 'rectangular', 'rectilinear']
        if grid_type is not None:
            if grid_type.lower() in sgrid_names:
                return Grid_S
            elif grid_type.lower() in ugrid_names:
                return Grid_U
            elif grid_type.lower() in rgrid_names:
                return Grid_R
            else:
                raise ValueError('Specified grid_type not recognized/supported')
        if grid_topology is not None:
            if ('faces' in grid_topology.keys() or
                    grid_topology.get('grid_type', 'notype').lower() in ugrid_names):
                return Grid_U
            elif grid_topology.get('grid_type', 'notype').lower() in rgrid_names:
                return Grid_R
            else:
                return Grid_S
        else:
            # no topology, so search dataset for grid_type variable
            if (hasattr(dataset, 'grid_type') and
                    dataset.grid_type in sgrid_names + ugrid_names):
                if dataset.grid_type.lower() in ugrid_names:
                    return Grid_U
                elif dataset.grid_type.lower() in rgrid_names:
                    return Grid_R
                else:
                    return Grid_S
            else:
                # TODO: Determine an effective decision tree for picking if
                #       a topology variable is present
                # no grid type explicitly specified. is a topology variable present?
                topology = Grid._find_topology_var(None, dataset=dataset)

                if topology is not None:
                    if (hasattr(topology, 'node_coordinates') and
                            not hasattr(topology, 'node_dimensions')):
                        return Grid_U
                    else:
                        return Grid_S
                else:
                    # no topology variable either, so generate and try again.
                    # if no defaults are found, _gen_topology will raise an error
                    try:
                        u_init_args, u_gf_vars = Grid_U._find_required_grid_attrs(None, dataset)
                        return Grid_U
                    except ValueError:
                        try:
                            r_init_args, r_gf_vars = Grid_R._find_required_grid_attrs(None, dataset)
                            return Grid_R
                        except ValueError:
                            try:
                                s_init_args, s_gf_vars = Grid_S._find_required_grid_attrs(None, dataset)
                            except ValueError:
                                raise ValueError("Can not figure out what type of grid this is. "
                                                 "Try specifying the grid_topology attributes "
                                                 "or specifying the grid type")
                            return Grid_S

    @staticmethod
    def _find_topology_var(filename,
                           dataset=None):
        gf = get_dataset(filename, dataset)
        gts = []
        for k, v in gf.variables.items():
            if hasattr(v, 'cf_role') and 'topology' in v.cf_role:
                gts.append(v)
#         gts = gf.get_variables_by_attributes(cf_role=lambda t: t is not None and 'topology' in t)
        if len(gts) != 0:
            return gts[0]
        else:
            return None
