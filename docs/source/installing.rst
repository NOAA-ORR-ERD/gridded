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

