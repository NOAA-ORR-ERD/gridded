"""
tests of Variable object

Variable objects are mostly tested implicitly in other tests,
but good to have a few explicitly for the Variable object
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import netCDF4

from .utilities import get_test_file_dir

from gridded import Variable

test_dir = get_test_file_dir()
sample_sgrid_file = os.path.join(test_dir, 'arakawa_c_test_grid.nc')


def test_create_from_netcdf_dataset():
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = Variable.from_netCDF(dataset=ds,
                               varname='u',
                               )
    print(var.info)

    assert var.data_file == 'arakawa_c_test_grid.nc'
    assert var.data.shape == (1,12,11)

