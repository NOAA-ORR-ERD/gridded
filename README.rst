.. image:: https://travis-ci.org/NOAA-ORR-ERD/gridded.svg?branch=master
    :target: https://travis-ci.org/NOAA-ORR-ERD/gridded

###########
``gridded``
###########

A single API for accessing / working with gridded model results on multiple grid types

`Documentation <https://noaa-orr-erd.github.io/gridded/index.html>`_

Goal
====

The goal of this package is to present a single way to work with results from ANY model -- regardless of what type of grid it was computed on. In particular:


* Regular Structured Grids (`CF Conventions <http://cfconventions.org/>`_), with API embedded in `Iris <http://scitools.org.uk/iris/>`_ and to some degree in `xarray <https://github.com/pydata/xarray>`_

* Unstructured Grids (CF + `UGRID Conventions <https://github.com/ugrid-conventions/ugrid-conventions/blob/master/README.md>`_), with nascent API in `pyugrid <https://github.com/pyugrid/pyugrid>`_

* Staggered Grids (CF + `SGRID Conventions <https://publicwiki.deltares.nl/display/NETCDF/Deltares+proposal+for+Staggered+Grid+data+model>`_) with nascent API in `pysgrid <https://github.com/sgrid/pysgrid>`_


Why gridded?
============

``gridded`` has been developed because a number of us need to work with multiple model types, and have found ourselves writing a lot of custom code for each type. In particular, inter-comparison of results is an ugly process. To preserve the integrity of the results, it's best to NOT interpolate on to a common grid. ``gridded`` lets one work with multiple model types with the same API, while preserving the native grid as much as possible.

Other solutions have (so far) built assumptions about the underlying grid type into the code and API, making it difficult to adapt to other grid types. Nevertheless, ``gridded`` hopes to learn from the the fabulous work done by other packages, such as:

Iris: http://scitools.org.uk/iris/  and xarray: https://github.com/pydata/xarray

Data standards
==============

``gridded`` seeks to support data standards such as:

* The CF Conventions: http://cfconventions.org/

* UGRID Conventions: http://ugrid-conventions.github.io/ugrid-conventions/

* SGRID Conventions: http://sgrid.github.io/sgrid/

``gridded`` also provided APIs for reading results that do not conform to the conventions, allowing one to work with non-confirming datasets with the same API, as well as providing tools to convert non-confirming files to conforming files.


Installing
==========

``gridded`` itself is pure python and easy to install from source or packages. However, it does rely on a number of complex compiled dependencies, notable netCDF4 and celltree2d.

For easiest results, install the dependencies from conda-forge:

https://anaconda.org/conda-forge

And then install ``gridded`` itself from source or from the conda package.

Dependencies are listed in ``conda_requirements.txt``:

``conda install --file conda_requirements.txt``

If you want to develop, test or work with the examples, you will need the development requirements as well:

``conda install --file conda_requirements_dev.txt``





