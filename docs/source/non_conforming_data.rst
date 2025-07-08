###########################
Loading Non Conforming Data
###########################

``gridded`` is designed to conform to the data model desribed by the following standards:

CF: http://cfconventions.org/

UGRID: http://ugrid-conventions.github.io/ugrid-conventions/

SGRID: http://sgrid.github.io/sgrid/

If your data files conform to those convensions, the ``gridded`` should load them automatically::

    gridded.Dataset("the_name_of_the_file")

However, if your files do not conform to those convensions, ``gridded`` may not be able to figure out what to do, or may do the wrong thing. You have to provide it with more information.

If your files conform to the underlying data model
==================================================

Many times, the data in your files (in particular netcdf) conform to the underlying data model, but does not have the full metadata to describe the relationship between the variables. In this case, you can provide a mapping of netcdf variable names to the role in the grid::

    names_mapping = {'nodes_lon': 'lon',
                     'nodes_lat': 'lat',
                     'faces': 'nbe',
                     }

    dataset = gridded.Dataset(ncfile="COOPS_SFBOFS.nc",
          grid_topology=names_mapping
          )


Unstructured Grids
------------------

(See the UGRID standard for more detailed explanation)

The core "parts" of the grid are:

Minimum Required (You can make a UGrid that has no actual mesh, but it's not very useful) ::

  nodes  (Nx2 array of lon, lat coordinates)

or ::

  nodes_lat
  nodes_lon

UGrid internally requires a single Nx2 array of coordinates for the nodes. It will concatenate separate arrays for you when constructing the grid object.

Usually Required (You can make a UGrid that has no actual mesh, but it's not very useful)::

  faces

The faces define the grid itself, in terms of the nodes. "face" is a 2D cell.

Optional: (a number of these can be constructed for you by ``gridded``)::

  edges
  boundaries
  face_face_connectivity
  face_edge_connectivity
  edge_coordinates
  face_coordinates
  boundary_coordinates


Curvilinear Grids
-----------------

(See the SGRID standard for more detailed explanation)

The core "parts" of the grid are:

Minimum Required
................

Nodes of the grid -- MxNx2 arrays:
::

    node_lon
    node_lat

Optional
........

If some elements in the node arrays are invalid, the mask::

    node_mask


If there are data on cell centers::

    center_lon
    center_lat
    center_mask

If there are data on the cell edges::

    edge1_lon
    edge1_lat
    edge1_mask
    edge2_lon
    edge2_lat
    edge2_mask

For staggered grids, if there is padding: "none", high", "low", "both"

    node_padding
    edge1_padding
    edge2_padding
    face_padding

If your files do not conform to the underlying data model
=========================================================

``gridded`` Datasets can be initialized entirely with direct data in anything that can be "turned in to" a numpy array: lists, numpy arrays, etc.

So your files can be in a totally different file format (text, etc), or be in netcdf in a form that is incompatible with the standards, you can pre-process the input data, and construct the parts of the ``gridded.Dataset``:

Here is an example of an unstructured grid:

(Complete example in the Examples dir: ``gridded/Examples/load_arbitrary_ugrid.py``)

A complete ``gridded.Dataset`` has:

* A Grid object

Optionally:

* Variables containing data on that grid.

* A Time object if the data are time dependent

* A Depth object, if the data are 3-d


The Grid
--------

.. code-block:: python

    nc = netCDF4.Dataset("SanDiego.nc")

    # need to convert to zero-indexing
    nodes = nc.variables['nodes'][:] - 1
    faces = nc.variables['E3T'][:, :3] - 1

    # make the grid
    # gridded.grids.Grid_U
    grid = gridded.grids.Grid_U(nodes=nodes,
                                faces=faces,
                                )

    # make the time object (handles time interpolation, etc)
    times_var = nc.variables['times'][:]

    # Time axis needs to be a list of datetime objects.
    # If the meta data are not there in the netcdf file, you have to do it by hand.
    start = datetime(2019, 1, 1, 12)
    times = [start + timedelta(seconds=val) for val in times_var]

    # This isn't a compliant file, so this will not work.
    # time_obj = gridded.time.Time.from_netCDF(dataset=nc,
    #                                          varname='times')

    time_obj = gridded.time.Time(data=times,
                                 filename=None,
                                 varname=None,
                                 tz_offset=None,
                                 origin=None,
                                 displacement=timedelta(seconds=0),)

    # make the variables
    depth = nc.variables['Depth']

    depth_var = gridded.variable.Variable(name=None,
                                          units="meters",
                                          time=None,
                                          data=depth,
                                          grid=grid,
                                          depth=None,
                                          data_file=nc,
                                          grid_file=nc,
                                          dataset=None,
                                          varname=None, # huh??
                                          fill_value=0,
                                          attributes=None)

    # global attributes
    attrs = {key: nc.getncattr(key) for key in nc.ncattrs()}

    # now make a dataset out of it all:
    ds = gridded.Dataset(ncfile=None,
                         grid=grid,
                         variables={'Depth': depth_var},
                         attributes=attrs)














