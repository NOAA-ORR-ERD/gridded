"""
tests of Variable object

Variable objects are mostly tested implicitly in other tests,
but good to have a few explicitly for the Variable object
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import netCDF4
import numpy as np

from .utilities import get_test_file_dir

from gridded import Variable, VectorVariable

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

def test_Variable_api_at_function():
    '''
    Test to ensure that the .at() function can accept a list, array, or separate lon/lat array,
    and that the output format is the same in all cases
    '''
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = Variable.from_netCDF(dataset=ds,
                               varname='u',
                               )
    
    p1 = [(1.5,35.5),(2.5, 35.5),(2.5, 35.5),(2.5, 35.5),(2.5, 35.5)]
    p2 = np.array(p1)
    p3 = p2.T

    t = var.time.min_time

    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(lons=p3[0], lats=p3[1], time=t, _mem=False)

    assert np.all(np.logical_and(r1 == r2, r2 == r3))

def test_Variable_api_at_function_edge_cases():
    '''
    Test for edge cases of input point shape (eg, 2x2) as well as single values
    '''
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = Variable.from_netCDF(dataset=ds,
                               varname='u',
                               )
    p1 = (1.5,35.5)
    p2 = (1.5,35.5, 1) #expected: r2 == r1
    p3 = [(1.5,35.5), (35.5,1.5)] # expected: [[1],[masked]]
    p4 = [(1.5,35.5, 1), (35.5,1.5, 1)] # expected: [[1],[masked]]
    
    t = var.time.min_time
    
    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(p3, t, _mem=False)
    r4 = var.at(lons=1.5, lats=35.5, time=t, _mem=False)

    assert np.all(r1 == np.array([[1,],[1,]]))
    assert np.all(r3 == np.ma.MaskedArray([[1,],[0,]], mask=[False, True]))
    assert np.all(r1 == r2)
    assert r4.shape == (1,1)

def test_VectorVariable_api_at_function():
    '''
    Test to ensure that the .at() function can accept a list, array, or separate lon/lat array,
    and that the output format is the same in all cases
    '''
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = VectorVariable.from_netCDF(dataset=ds,
                               varnames=['u', 'v'],
                               )
    
    p1 = [(1.5,35.5),(2.5, 35.5),(2.5, 35.5),(2.5, 35.5),(2.5, 35.5)]
    p2 = np.array(p1)
    p3 = p2.T

    t = var.time.min_time

    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(lons=p3[0], lats=p3[1], time=t, _mem=False)

    assert np.all(np.logical_and(r1 == r2, r2 == r3))

def test_VectorVariable_api_at_function_edge_cases():
    '''
    Test for edge cases of input point shape (eg, 2x2) as well as single values
    '''
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = VectorVariable.from_netCDF(dataset=ds,
                               varnames=['u', 'v'],
                               )
    p1 = (1.5,35.5)
    p2 = (1.5,35.5, 1) #expected: r2 == r1
    p3 = [(1.5,35.5), (35.5,1.5)] # expected: [[1],[masked]]
    p4 = [(1.5,35.5, 1), (35.5,1.5, 1)] # expected: [[1],[masked]]
    
    t = var.time.min_time
    
    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(p3, t, _mem=False)
    r4 = var.at(lons=1.5, lats=35.5, time=t, _mem=False)

    assert np.all(r1 == np.array([[1,0],[1,0]]))
    assert np.all(r3 == np.ma.MaskedArray([[1,0],[0,-1]], mask=[[False, False], [True,True]]))
    assert np.all(r1 == r2)
    assert r4.shape == (1,2)