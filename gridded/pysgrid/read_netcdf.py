'''
Created on Mar 19, 2015

@author: ayan
'''


import re

from gridded.pysgrid.lookup import X_COORDINATES, Y_COORDINATES
from gridded.pysgrid.utils import GridPadding


def find_grid_topology_var(nc):
    """
    Get the variable from a netCDF dataset
    that have a `cf_role `attribute of 'grid_topology' and
    `topology_dimension` of 2.

    :params nc: netCDF dataset
    :type nc: netCDF4.Dataset
    :return: variable name that contain grid topologies
    :rtype: string

    """
    grid_topology = nc.get_variables_by_attributes(cf_role='grid_topology')

    if not grid_topology:
        raise ValueError('Could not find the `grid_topology` variable.')
    if len(grid_topology) > 1:
        msg = 'Expected 1 `grid_topology` variable.  Got {0}'.format
        raise ValueError(msg(len(grid_topology)))

    grid_topology = grid_topology[0]
    topology_dimension = int(grid_topology.topology_dimension) if hasattr(grid_topology, 'topology_dimension') else None  # noqa
    
    if topology_dimension != 2:
        msg = ('Only 2 dimensions topology are supported.'
               'Got {0}'.format)
        raise ValueError(msg(topology_dimension))

    return grid_topology.name


def parse_padding(padding_str, mesh_topology_var):
    """
    Use regex expressions to break apart an
    attribute string containing padding types
    for each variable with a cf_role of
    'grid_topology'.

    Padding information is returned within a named tuple
    for each node dimension of an edge, face, or vertical
    dimension. The named tuples have the following attributes:
    mesh_topology_var, dim_name, dim_var, and padding.
    Padding information is returned as a list
    of these named tuples.

    :param str padding_str: string containing padding types from
                            a netCDF attribute.
    :return: named tuples with padding information.
    :rtype: list.

    """
    p = re.compile(r'([a-zA-Z0-9_]+:) ([a-zA-Z0-9_]+) (\(padding: [a-zA-Z]+\))')
    padding_matches = p.findall(padding_str)
    padding_type_list = []
    for padding_match in padding_matches:
        raw_dim, raw_sub_dim, raw_padding_var = padding_match
        dim = raw_dim.split(':')[0]
        sub_dim = raw_sub_dim
        # Remove parentheses. (That is why regular expressions are bad!
        # You need a commend to explain what is going on!!)
        cleaned_padding_var = re.sub(r'[\(\)]', '', raw_padding_var)
        # Get the padding value and remove spaces.
        padding_type = cleaned_padding_var.split(':')[1].strip()
        grid_padding = GridPadding(mesh_topology_var=mesh_topology_var,
                                   face_dim=dim,
                                   node_dim=sub_dim,
                                   padding=padding_type
                                   )
        padding_type_list.append(grid_padding)
    if len(padding_type_list) > 0:
        final_padding_types = padding_type_list
    else:
        final_padding_types = None
        msg = ('The netCDF file appears to have conform to SGRID conventions, '
               'but padding values cannot be found.')
        raise ValueError(msg)
    return final_padding_types


def parse_axes(axes_attr):
    p = re.compile('([a-zA-Z]: [a-zA-Z_]+)')
    matches = p.findall(axes_attr)
    x_axis = None
    y_axis = None
    z_axis = None
    for match in matches:
        axis_split = match.split(':')
        axis_name = axis_split[0].strip()
        axis_coordinate = axis_split[1].strip()
        if axis_name.lower() == 'x':
            x_axis = axis_coordinate
        elif axis_name.lower() == 'y':
            y_axis = axis_coordinate
        elif axis_name.lower() == 'z':
            z_axis = axis_coordinate
    return x_axis, y_axis, z_axis


def parse_vector_axis(variable_standard_name):
    p = re.compile('[a-z_]+_[xyz]_[a-z_]+')
    match = p.match(variable_standard_name)
    if match is not None:
        direction_pattern = re.compile('_[xyz]_')
        direction_substr = direction_pattern.search(match.string).group()
        vector_direction = direction_substr.replace('_', '').upper()
    else:
        vector_direction = None
    return vector_direction


class NetCDFDataset(object):

    def __init__(self, nc):
        self.nc = nc
        # in case a user as a version netcdf C library < 4.1.2
        try:
            self._filepath = nc.filepath()
        except ValueError:
            self._filepath = None
        self.sgrid_compliant_file()

    def find_node_coordinates(self, node_dimensions):
        """
        Find the variables for the grid
        cell vertices.

        """
        nc_vars = self.nc.variables
        node_dims = node_dimensions.split(' ')
        node_dim_set = set(node_dims)
        x_node_coordinate = None
        y_node_coordinate = None
        for nc_var in nc_vars.keys():
            nc_var_obj = nc_vars[nc_var]
            nc_var_dims = nc_var_obj.dimensions
            nc_var_dim_set = set(nc_var_dims)
            name_lower = nc_var_obj.name.lower()
            try:
                standard_name_lower = nc_var_obj.standard_name.lower()
            except AttributeError:
                standard_name_lower = ''
            if nc_var_dim_set == node_dim_set:
                if (any(x in name_lower for x in X_COORDINATES) or
                   any(x in standard_name_lower for x in X_COORDINATES)):
                    x_node_coordinate = nc_var
                elif (any(y in name_lower for y in Y_COORDINATES) or
                      any(y in standard_name_lower for y in Y_COORDINATES)):
                    y_node_coordinate = nc_var
            if x_node_coordinate is not None and y_node_coordinate is not None:
                # Exit the loop once both x and y coordinates are found.
                break
        if x_node_coordinate is not None and y_node_coordinate is not None:
            return x_node_coordinate, y_node_coordinate
        else:
            return None

    def find_variables_by_attr(self, **kwargs):
        nc_vars = self.nc.variables
        matches = []
        keys = kwargs.keys()
        for nc_var in nc_vars.keys():
            nc_var_obj = nc_vars[nc_var]
            nc_var_attrs = dir(nc_var_obj)  # All object attributes.
            # Check to see if the requested attributes are in the
            # variable object if not, don't bother with it.
            if set(keys).issubset(nc_var_attrs):
                attr_tracking = {}
                for key in keys:
                    nc_var_attr_value = getattr(nc_var_obj, key)
                    attr_tracking[key] = nc_var_attr_value
                if attr_tracking == kwargs:
                    matches.append(nc_var)
        return matches

    def find_coordinates_by_location(self, location_str, topology_dim):
        """
        Find a grid coordinates variables with a location attribute equal
        to location_str. This method can be used to infer edge, face, or
        volume coordinates from the location attribute of a variable.

        Location is a required attribute per SGRID conventions.

        :param str location_str: the location value to search for
        :param int topology_dim: the topology dimension of the grid

        """
        nc_vars = self.nc.variables
        vars_with_location = self.find_variables_by_attr(location=location_str)
        x_coordinate = None
        y_coordinate = None
        z_coordinate = None
        for var_with_location in vars_with_location:
            location_var = nc_vars[var_with_location]
            location_var_dims = location_var.dimensions
            try:
                location_var_coordinates = location_var.coordinates
            except AttributeError:
                # Run through this if a location attributed is defined,
                # but not coordinates.
                potential_coordinates = []
                for nc_var in nc_vars.keys():
                    nc_var_obj = nc_vars[nc_var]
                    nc_var_dim_set = set(nc_var_obj.dimensions)
                    if (nc_var_dim_set.issubset(location_var_dims) and
                       nc_var != var_with_location and
                       len(nc_var_dim_set) > 0):
                        potential_coordinates.append(nc_var_obj)
                for potential_coordinate in potential_coordinates:
                    pc_name = potential_coordinate.name
                    try:
                        pc_std_name = potential_coordinate.standard_name
                    except AttributeError:
                        pc_std_name = ''
                    if (any(x in pc_name.lower() for x in X_COORDINATES) or
                       any(x in pc_std_name.lower() for x in X_COORDINATES)):
                        x_coordinate = pc_name
                    elif (any(y in pc_name.lower() for y in Y_COORDINATES) or
                          any(y in pc_std_name.lower() for y in Y_COORDINATES)):  # noqa
                        y_coordinate = pc_name
                    else:
                        z_coordinate = pc_name  # this might not always work...
            else:
                lvc_split = location_var_coordinates.strip().split(' ')
                for lvc in lvc_split:
                    var_coord = nc_vars[lvc]
                    try:
                        var_coord_standard_name = var_coord.standard_name
                    except AttributeError:
                        var_coord_standard_name = ''
                    try:
                        var_coord_desc = var_coord.description
                    except AttributeError:
                        var_coord_desc = ''
                    if ('lon' in var_coord.name.lower() or
                        'longitude' in var_coord_standard_name.lower() or
                       'longitude' in var_coord_desc.lower()):
                        x_coordinate = lvc
                    elif ('lat' in var_coord.name.lower() or
                          'latitude' in var_coord_standard_name.lower() or
                          'latitude' in var_coord_desc.lower()):
                        y_coordinate = lvc
                if len(lvc_split) == 3:
                    z_coordinate = lvc_split[-1]
                break
        if topology_dim == 2:
            coordinates = (x_coordinate, y_coordinate)
        else:
            coordinates = (x_coordinate, y_coordinate, z_coordinate)
        if all(coordinates):
            coordinate_result = coordinates
        else:
            coordinate_result = None
        return coordinate_result

    def sgrid_compliant_file(self):
        """
        Determine whether a dataset is
        SGRID compliant.

        :return: True if dataset is compliant, raise an exception if it is not
        :rtype: bool

        """
        try:
            find_grid_topology_var(self.nc)
        except ValueError as e:
            raise e

        return True
