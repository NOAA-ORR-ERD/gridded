``gridded API``
===============

The ``gridded`` API is built around a few core object types. In the general case, model results are delivered in a file or collection of files that contain a definition of the grid and data (or variables) that are associated with that grid. Depending on use-case, it can make sense to work with the entire collection of data and grid, or to work with just a couple of particular variables. The ``gridded`` object model is designed to allow either use case.

``gridded.Dataset``
-------------------

The :class:`gridded.gridded.Dataset` object represents the entire gridded dataset. It more-or-less maps to what is usually found in a CF-compliant netcdf file output from a model run.

A :class:``gridded.gridded.Dataset`` contains a few core pieces of information:

 - Attributes of the whole dataset
 - A Grid object -- this is a duck-typed grid object that could be any of the supported grid types.
 - A dict of :class:`gridded.variable.Variable` objects -- names as the key
 - Assorted utilities for accessing, saving, and loading the data.

Grid objects
------------

Grid objects represent the grid itself, and contain functionality to work with the grid that requires specific knowledge of grid types, such as interpolation searching, etc.

The :class:`gridded.grids.GridBase` object provides shared functionality that all Grids require. It does not (but probably should) specify the full Grid API.

Variable objects
----------------

:class:`gridded.variable.Variable` contains the data associated with a particular quantity in the ``Dataset``. It maps more-or-less to a netcdf Variable, with a name, attributes, and an array that holds the actual data. It also holds a reference to the Grid, Time, and Depth objects for the Grid that it is defined on, so that it can use those object to perform interpolation, etc.

The idea is that a Variable represents a continuous field of some property in 4-D space (lat, lon, depth, time).


Time object
-----------

:class:`gridded.time.Time` object represent the time axis of the dataset, providing interpolation in time for Variables.


Depth object
------------

:class:`gridded.depth.Depth` objects represent the depth axis of the data set, it provides an abstraction around various vertical coordinate schemes, and provides interpolation in depth for Variables.



Design Principles
=================

Grid Independence
-----------------

The primary goal of ``gridded`` is for an end-user to be able to do data analysis and visualization without needing to understand the intricacies of the grid structure, and ideally to no have to even know what grid a particular dataset or variable is using.

For example one should be able to plot a time series of a parameter, like sea surface temperature, with exactly the same code regardless of the underlying grid:

.. code-block:: python

  ds = gridded.Dataset(path_to_file)
  sst = ds.get_variables_by_attributes(standard_name='sea_surface_temperature')
  time_series = sst.at(lat, lon, times)

The user can now plot the time series for a given location and times, without having to know anything about the grid structure.


Duck Typing
-----------

These are for the most part "duck typed", rather than strict subclassing. Though there are base classes that provide shared functionality.

We are trying to be clear about the "public" vs "private" API by using leading underscores for methods and attributes not intended for external use.


Lazy loading / data arrays
--------------------------

Many of the datasets users need to work with can be quite large. As a result it is impractical to load entire datasets into memory at once. ``gridded`` for the most part shifts the burden of handling lazy loading to external libraries, and does this by keeping data stored in a "numpy array-like" objects. Users can use pure numpy arrays, or any object that "acts" like a numpy array. This should allow ``gridded`` to work with netcdf variables, hdf5 arrays, dask arrays, etc.

In practice, there is no clear definition of "array-like", so ``gridded`` has defined its own definition, based on features we know we need. But it is assumed that nd indexing behaves that same as numpy arrays -- as there is no way to easily confirm that.

To support that, we support converting and testing for array-like with:

``gridded.utils.asarraylike()``

and

``gridded.utils.isarraylike()``


Reference
=========

.. toctree::
   :maxdepth: 4

   gridded
