#!/usr/bin/env python

"""
Utilities to help with grid io

NOTE: this isn't used yet, but should be useful for loading non
UGRID-compliant files.

"""

from __future__ import (absolute_import, division, print_function)

import netCDF4
import numpy as np

from ..ugrid import UGrid


def load_from_varnames(filename, names_mapping, attribute_check=None):
    """
    Load a UGrid from a netcdf file where the roles are defined by the
    names of the variables.

    :param filename: names of the file to load (or OPeNDAP URL).

    :param names_mapping: dict that maps the variable names to UGrid components

    :param attribute_check=None: list of global attributes that are expected
    :type attribute_check: list of tuples to check. Example:
                           [('grid_type','triangular'),] will check if the
                           grid_type attribute is set to "triangular"

    The names_mapping dict has to contain at least: 'nodes_lon', 'nodes_lat'

    Optionally (and mostly required), it can contain: face_face_connectivity',
    'face_coordinates_lon', 'face_coordinates_lat', and 'faces'

    """
    ug = UGrid()
    attribute_check = {} if attribute_check is None else attribute_check

    nc = netCDF4.Dataset(filename)

    # Check for the specified attributes.
    for name, value in attribute_check:
        if nc.getncattr(name).lower() != value:
            raise ValueError('This does not appear to be a valid file:\n'
                             'It does not have the "{}"="{}"'
                             'global attribute set'.format(name, value))

    # Nodes.
    lon = nc.variables[names_mapping['nodes_lon']]
    lat = nc.variables[names_mapping['nodes_lat']]

    num_nodes = lon.size
    ug.nodes = np.zeros((num_nodes, 2), dtype=lon.dtype)
    ug.nodes[:, 0] = lon[:]
    ug.nodes[:, 1] = lat[:]

    # Faces.
    faces = nc.variables[names_mapping['faces']]
    # FIXME: This logic assumes there are more than three triangles.
    if faces.shape[0] <= faces.shape[1]:
        # Fortran order.
        faces = faces[:].T
    else:
        faces = faces[:]

    # One-indexed?
    if faces.min() == 1:
        one_indexed = True
    else:
        one_indexed = False

    if one_indexed:
        faces -= 1
    ug.faces = faces

    # Connectivity (optional).
    if 'face_face_connectivity' in names_mapping:
        face_face_connectivity = nc.variables[names_mapping['face_face_connectivity']]  # noqa
        # FIXME: This logic assumes there are more than three triangles.
        if face_face_connectivity.shape[0] <= face_face_connectivity.shape[1]:
            # Fortran order.
            face_face_connectivity = face_face_connectivity[:].T
        else:
            face_face_connectivity = face_face_connectivity[:]
        if one_indexed:
            face_face_connectivity -= 1
        ug.face_face_connectivity = face_face_connectivity

    # Center (optional).
    if ('face_coordinates_lon' in names_mapping and
       'face_coordinates_lat' in names_mapping):

        ug.face_coordinates = np.zeros((len(ug.faces), 2), dtype=lon.dtype)
        ug.face_coordinates[:, 0] = nc.variables[names_mapping['face_coordinates_lon']][:]  # noqa
        ug.face_coordinates[:, 1] = nc.variables[names_mapping['face_coordinates_lat']][:]  # noqa

    # Boundaries (optional).
    if 'boundaries' in names_mapping:
        # FIXME: this one is weird and non-conforming!
        # Ignoring the second two fields. What are they?
        boundaries = nc.variables[names_mapping['boundaries']][:, :2]
        if one_indexed:
            boundaries -= 1
        ug.boundaries = boundaries

    return ug
