Installing
==========

``gridded`` itself is pure python and easy to install from source or packages. However, it does rely on a number of complex compiled dependencies, notable netCDF4 and celltree2d.

pip
---

As of yet, there is no package on PyPi, so a plain ``pip install gridded`` Will not work.

However, if you are managing your packages with pip, then you can use pip to install from source or gitHub:

From source
...........

Either make a clone of the gitHub repo or download a source tarball and unpack it.

Then, in the ``gridded`` dir:

``pip install ./``

or, if you want to contribute to the code, you can install in editable mode:

``pip install -e ./``

conda
-----

The conda system provides a way to manage complex compiled packages:

https://conda.io/docs/intro.html

You can install a base conda system with either the Anaconda distribution, or miniconda:

https://www.continuum.io/downloads

https://conda.io/miniconda.html

conda-forge
...........

``gridded`` has some dependencies that are not supported in the default conda channel. These are all support on the conda-forge channel:

https://anaconda.org/conda-forge

You can add the channel to your system with:

``conda config --add channels conda-forge``

Dependencies are listed in ``conda_requirements.txt``, and can be installed with:

``conda install --file conda_requirements.txt``

If you want to develop, test or work with the examples, you will need the development requirements as well:

``conda install --file conda_requirements_dev.txt``

Environments
............

There are also a few "environment files" that will directly build a conda environment for you:

``conda env create -f py2_environment.yml``

For python2, and:

``py3_environment.yml``

For python 3

