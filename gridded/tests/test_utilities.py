#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals


import numpy as np
import netCDF4 as nc
from gridded import utilities
from gridded.tests.test_depth import get_s_depth

def test_gen_mask():
    mask = np.array(([True, True , True, True],
                     [True, False, False, True],
                     [True, False, False, True],
                     [True, True, True, True]))

    m = utilities.gen_mask(mask, add_boundary=False)

    assert np.all(m == mask)

    m2 = utilities.gen_mask(mask, add_boundary=True)

    expected_mask = np.array(([True, False, False, True],
                              [False, False, False, False],
                              [False, False, False, False],
                              [True, False, False, True]))

    assert np.all(m2 == expected_mask)

    testds = nc.Dataset('foo', mode='w', diskless=True)
    testds.createDimension('x', 4)
    testds.createDimension('y', 4)
    testds.createVariable('mask', 'b', dimensions=('y', 'x'))
    testds['mask'][:] = mask

    m3 = utilities.gen_mask(testds['mask'], add_boundary=True)

    assert np.all(m3 == expected_mask)

    testds['mask'][:] = ~mask
    testds['mask'].flag_values = [0,1]
    testds['mask'].flag_meanings = ['land', 'water']

    m4 = utilities.gen_mask(testds['mask'], add_boundary=True)

    assert np.all(m4 == expected_mask)

    testds['mask'][:,2] = [0,2,2,0]
    testds['mask'].flag_values = [0,1,2]
    testds['mask'].flag_meanings = ['land', 'water', 'water2']

    m5 = utilities.gen_mask(testds['mask'], add_boundary=True)

    assert np.all(m5 == expected_mask)

    #because sometimes it's a damn string
    testds['mask'].flag_meanings = 'land water water2'

    m5 = utilities.gen_mask(testds['mask'], add_boundary=True)

    assert np.all(m5 == expected_mask)
    testds.close()



def test_reorganize_spatial_data():
    #1-dimensional data
    sample_1 = [1,2,3]
    sample_2 = [(1,),(2,),(3,)]
    sample_3 = [(1,2,3),]
    a1 = utilities._reorganize_spatial_data(sample_1)
    a2 = utilities._reorganize_spatial_data(sample_2)
    a3 = utilities._reorganize_spatial_data(sample_3)
    assert np.all(a1 == a2)
    assert np.all(a3 == np.array([(1,2,3),]))
    assert np.all(a2 == a3)

    #impossible cases
    sample_1 = np.array([[1,2,3],[4,5,6]])
    sample_2 = np.array([[1,2],[3,4],[5,6]])
    a1 = utilities._reorganize_spatial_data(sample_1)
    a2 = utilities._reorganize_spatial_data(sample_2)
    assert np.all(sample_2 == a2)
    assert np.all(sample_1 == a1)

    #dim 0 > dim 1
    sample_1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    a1 = utilities._reorganize_spatial_data(sample_1)
    assert np.all(a1 == sample_1)

    #dim 1 > dim 0
    sample_2 = [[1,4,7,10],[2,5,8,11],[3,6,9,12]]
    a2 = utilities._reorganize_spatial_data(sample_2)
    assert np.all(a2 == sample_1)


def test_spatial_data_metadata():
    pts_1 = [1,2,3]
    pts_2 = [(1,),(2,),(3,)]
    pts_3 = np.array([[1,2,3],[4,5,6]])
    pts_4 = [[1,4,7,10],[2,5,8,11],[3,6,9,12]]

    res_1 = np.array([[1,],])
    res_2 = np.array([[1,2,3,4,5,6],])
    res_3 = np.array([[1,2,3,4,5,6],
                      [2,3,4,5,6,7]])
    res_4 = np.array([[1,2,3,4,5,6],
                      [7,8,9,10,11,12],
                      [13,14,15,16,17,18],
                      [19,20,21,22,23,24]])
    res_5 = np.array([[1,],
                      [2,],
                      [3,],
                      [4,]])

    a1 = utilities._align_results_to_spatial_data(res_1, pts_1)
    assert np.all(a1 == res_1)
    a2 = utilities._align_results_to_spatial_data(res_2, pts_2)
    assert np.all(a2 == res_2)
    a3 = utilities._align_results_to_spatial_data(res_3, pts_3)
    assert np.all(a3 == res_3)

    a4 = utilities._align_results_to_spatial_data(res_4, pts_4)
    a5 = utilities._align_results_to_spatial_data(res_5, pts_4)
    assert np.all(a4.T == res_4)
    assert np.all(a5.T == res_5)


def test_regrid_variable_TDStoS(get_s_depth):
    # Time is present
    # Depth is present
    # Grid_S to Grid_S
    from gridded.variable import Variable
    from gridded.time import Time
    from gridded.grids import Grid_S
    sd = get_s_depth
    grid = sd.grid
    n_levels = sd.num_w_levels
    data = np.ones((1,
                    n_levels, grid.node_lon.shape[0],
                    grid.node_lon.shape[1]))
    for l in range(0, n_levels):
        data[0, l] *= l

    v1 = Variable(name='v1',
                  grid=grid,
                  data=data,
                  depth=sd,
                  time=Time.constant_time())

    g2 = Grid_S(node_lon=(grid.node_lon[0:-1, 0:-1] + grid.node_lon[1:, 1:]) / 2,
                node_lat=(grid.node_lat[0:-1, 0:-1] + grid.node_lat[1:, 1:]) / 2)

    v2 = utilities.regrid_variable(g2, v1)
    # time should be unchanged
    assert v2.time is v1.time
    # number of depth levels should remain unchanged
    assert len(v2.depth) == len(v1.depth)
    sz = v1.data.shape[-1]
    # data shape should retain the same time/depth dimensions as the original
    # except in xy
    assert v2.data.shape == (v1.data.shape[0], v1.data.shape[1], sz - 1, sz - 1)


def test_regrid_variable_StoS(get_s_depth):
    # Time is not present
    # Depth is not present
    # Grid_S to Grid_S
    from gridded.variable import Variable
    from gridded.time import Time
    from gridded.grids import Grid_S
    sd = get_s_depth
    grid = sd.grid
    data = np.ones((grid.node_lon.shape[0], grid.node_lon.shape[1]))

    v1 = Variable(name='v1',
                  grid=grid,
                  data=data,
                  depth=None,
                  time=None)

    g2 = Grid_S(node_lon=(grid.node_lon[0:-1, 0:-1] + grid.node_lon[1:, 1:]) / 2,
                node_lat=(grid.node_lat[0:-1, 0:-1] + grid.node_lat[1:, 1:]) / 2)

    v2 = utilities.regrid_variable(g2, v1)
    # time should be unchanged
    assert v2.time is v1.time
    # depth should be None
    assert v2.depth is v1.depth is None
    sz = v1.data.shape[-1]
    # data shape should retain the same time/depth dimensions as the original
    # except in xy
    assert v2.data.shape[-2::] == (sz - 1, sz - 1)


class DummyArrayLike(object):
    """
    Class that will look like an array to this function, even
    though it won't work!

    Just for tests. All it does is add a few expected attributes

    This will need to be updated when the function is changed.

    """
    must_have = ['dtype', 'shape', 'ndim', '__len__', '__getitem__', '__getattribute__']

    # pretty kludgy way to do this..
    def __new__(cls):
        print ("in new"), cls
        obj = object.__new__(cls)
        for attr in cls.must_have:
            setattr(obj, attr, None)
        return obj


def test_dummy_array_like():
    dum = DummyArrayLike()
    print(dum)
    print(dum.dtype)
    for attr in DummyArrayLike.must_have:
        assert hasattr(dum, attr)


def test_asarraylike_list():
    """
    Passing in a list should return a np.ndarray.

    """
    lst = [1, 2, 3, 4]
    result = utilities.asarraylike(lst)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, lst)


def test_asarraylike_array():
    """
    Passing in a list should return a np.ndarray.

    """
    arr = np.array([1, 2, 3, 4])
    result = utilities.asarraylike(arr)

    assert result is arr


def test_as_test_asarraylike_dummy():
    dum = DummyArrayLike()
    result = utilities.asarraylike(dum)
    assert result is dum
