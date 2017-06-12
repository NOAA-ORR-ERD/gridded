###########
``gridded``
###########

A single API for accessing / working with gridded model results on multiple grid types


Goal
====

The goal of this pacakge is to present a single way to work with restuls from ANY model -- regardless of what type of grid it was computed on. IN particular:


* Regular structured Grids ([CF Conventions](http://cfconventions.org/)), with API embedded in [Iris](http://scitools.org.uk/iris/) and to some degree in [Xray](https://github.com/xray/xray)

* Unstructured Grids (CF + [UGRID Conventions](https://github.com/ugrid-conventions/ugrid-conventions/blob/master/README.md)), with nascent API in [pyugrid](https://github.com/pyugrid/pyugrid)

* Staggered Grids (CF + [SGRID Conventions](https://publicwiki.deltares.nl/display/NETCDF/Deltares+proposal+for+Staggered+Grid+data+model)) with nascent API in [pysgrid](https://github.com/sgrid/pysgrid)

Why gridded?
============

``gridded`` has been developed because a number of us need to work with multiple model types, and have found ourselves writing a lot of custom code for each type. In particular, intercomparison of results is an ugly process. To preserve the integrity of the results, it's best to NOT interpolate on to a common grid. ``gridded`` lets one work with multiple model types with the same API, while preserving the native grid as much as possible.

Other solutions have (so far) built assumptions about the underlying grid type into the code and API, making it difficult to adapt to other grid types. Nevertheless, ``gridded`` hopes to lwear from the the fabulous work done by other packages, such as:

Iris: http://scitools.org.uk/iris/  and Xray: https://github.com/xray/xray


Data standards
==============

``gridded`` seeks to support data standards such as:

* The CF Conventions: http://cfconventions.org/

* UGRID Conventions: https://github.com/ugrid-conventions/ugrid-conventions

* SGRID Conventions: https://publicwiki.deltares.nl/display/NETCDF/Deltares+proposal+for+Staggered+Grid+data+model

``gridded`` also provided APIs for reading results that do not conform to the conventions, allowing one to work with non-confirming datasets with the same API, as well as providing tools to convert non-confirming files to conforming files.
