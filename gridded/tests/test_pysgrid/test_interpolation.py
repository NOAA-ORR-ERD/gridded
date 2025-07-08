"""
Created on Feb 17, 2016

@author: jay.hennen
"""

import numpy as np
import pytest

from gridded.pysgrid.sgrid import SGrid
from gridded.pysgrid.utils import points_in_polys


node_lon = np.array(([1, 3, 5], [1, 3, 5], [1, 3, 5]))
node_lat = np.array(([1, 1, 1], [3, 3, 3], [5, 5, 5]))
edge2_lon = np.array(([0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6]))
edge2_lat = np.array(([1, 1, 1, 1], [3, 3, 3, 3], [5, 5, 5, 5]))
edge1_lon = np.array(([1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]))
edge1_lat = np.array(([0, 0, 0], [2, 2, 2], [4, 4, 4], [6, 6, 6]))
center_lon = np.array(([0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6], [0, 2, 4, 6]))
center_lat = np.array(([0, 0, 0, 0], [2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6]))

sgrid = SGrid(node_lon=node_lon,
              node_lat=node_lat,
              edge1_lon=edge1_lon,
              edge1_lat=edge1_lat,
              edge2_lon=edge2_lon,
              edge2_lat=edge2_lat,
              center_lon=center_lon,
              center_lat=center_lat)

c_var = np.array(([0, 0, 0, 0], [0, 1, 2, 0], [0, 2, 1, 0], [0, 0, 0, 0]))
e2_var = np.array(([1, 0, 0, 1], [0, 1, 2, 0], [0, 0, 0, 0]))
e1_var = np.array(([1, 1, 0], [0, 1, 0], [0, 2, 0], [1, 1, 0]))
n_var = np.array(([0, 1, 0], [1, 0, 1], [0, 1, 0]))

ptsx, ptsy = np.mgrid[0:6:600j, 0:6:600j]
pts = np.stack((ptsx, ptsy), axis=-1)


def test_locate_faces():
    diagonal = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]
    ind_ans = np.ma.masked_equal([[-1, -1],
                                  [0, 0],
                                  [0, 0],
                                  [1, 1],
                                  [1, 1],
                                  [-1, -1],
                                  [-1, -1],
                                  ], -1)
    indices = sgrid.locate_faces(diagonal, 'node')
    assert ((indices.data == ind_ans.data).all() and
            (indices.mask == ind_ans.mask).all())


def test_points_in_polys():
    points = np.array(
        [[0, 0],
         [1, 0],
         [2, 0],
         [0, 1],
         [0, 2],
         [1, 2],
         [2, 2],
         [2, 1]])
    polygon = np.array(([0, 0], [2, 0], [2, 2], [0, 2])).reshape(1, 4, 2)
    pinp = np.array([points_in_polys(point.reshape(1, 2), polygon)
                     for point in points]).reshape(-1)
    answer = sgrid.locate_faces(points + 1, 'node') == [0, 0]
    answer = np.logical_and(answer[:, 0], answer[:, 1])
    res = (answer == pinp).all()
    assert(res)


def test_points_in_polys2():
    rectangle = np.array(([0, 0],
                          [2, 0],
                          [2, 2],
                          [0, 2])).reshape(1, 4, 2)
    boundaries = np.array([[0, 0],
                           [1, 0],
                           [2, 0],
                           [0, 1],
                           [0, 2],
                           [1, 2],
                           [2, 2],
                           [2, 1]])
    pinp = np.array([points_in_polys(point.reshape(1, 2), rectangle)
                     for point in boundaries]).reshape(-1)
    answer = sgrid.locate_faces(boundaries + 1) == [0, 0]
    answer = np.logical_and(answer[:, 0], answer[:, 1])
    assert (answer == pinp).all()


def test_nearest_neighbor():
    sgrid.build_kdtree()

    diagonal = [[0, 0], [1, 1], [1, 2], [1, 2.1], [2.1, 2.1], [5, 5], [6, 6]]
    inds = sgrid.locate_nearest(diagonal, 'node')
    ind_ans = np.array([[0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 0],
                        [1, 1],
                        [2, 2],
                        [2, 2]], dtype=np.int64)

    assert np.all(inds == ind_ans)

def test_mirror_mask_values():
    values = np.ma.MaskedArray(data=[[9999, 7],      [9999, -11],      [150, 9999]],
                                 mask=[[True, False], [True, False],     [False, True]])
    mm_values = sgrid.mirror_mask_values(values)
    assert np.all(mm_values == [[-7, 7], [11, -11], [150, -150]])
    assert isinstance(mm_values, np.ndarray)

def test_compute_interpolant():
    
    with pytest.warns(UserWarning, match='Alphas do not sum to 1'):
        values = [-10, 10]
        alphas = [0.75, 0.252]
        interp = sgrid.compute_interpolant(values, alphas)
    
    #basic test, linear arrays    
    values = [-10, 10]
    alphas = [0.75, 0.25]
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert interp == -5
    
    #basic test, linear masked arrays    
    values = np.ma.MaskedArray(data=[-10, 10], mask=[True, False])
    alphas = [0.75, 0.25]
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert interp == 2.5
    values = np.ma.MaskedArray(data=[-6, 6,9000,6], mask=[True, False, True, False])
    alphas = [0.25,0.25,0.25,0.25]
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='mask')
    assert interp is np.ma.masked
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert interp == 3.0
    
    #basic test, vector arrays
    values = [[-10, 10], [10, -10]]
    alphas = [[0.75, 0.25], [0.5, 0.5]]
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert np.all(interp == [-5, 0])
    assert isinstance(interp, np.ndarray)
    values = [[1, 2, 3, 4], [10, -10, -10, 10]]
    alphas = [[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0 , 0]]
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert np.all(interp == [2.5, 0])
    assert isinstance(interp, np.ndarray)
    
    #nan alphas/values test
    values = [[np.nan, np.nan], [10, -10], [150, 200]]
    alphas = [[0.75, 0.25], [0.5, 0.5], [np.nan, np.nan]]
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert np.isnan(interp[0])
    assert np.isnan(interp[2])
    assert isinstance(interp, np.ndarray)
    
    #masked values test, 'zero' mask_behavior (both points masked)
    values = np.ma.MaskedArray(data=[[-10, 10],      [10, -10],      [150, 200]],
                               mask=[[False, False], [False, False], [True, True]])
    alphas = np.ma.MaskedArray(data=[[0.75, 0.25],   [0.5, 0.5],     [0.5, 0.5]],
                               mask=[[False, False], [False, False], [False, False]])
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert np.all(interp == [-5, 0, 0])
    assert isinstance(interp, np.ma.MaskedArray)
    
    #masked values test, 'mask' mask_behavior (both point masked)
    values = np.ma.MaskedArray(data=[[-10, 10],      [10, -10],      [150, 200]],
                               mask=[[False, False], [False, False], [True, True]])
    alphas = np.ma.MaskedArray(data=[[0.75, 0.25],   [0.5, 0.5],     [0.5, 0.5]],
                               mask=[[False, False], [False, False], [False, False]])
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='mask')
    assert np.all(interp == [-5, 0, np.ma.masked])
    assert isinstance(interp, np.ma.MaskedArray)
    
    #masked values test, 'zero' mask_behavior (one point masked)
    values = np.ma.MaskedArray(data=[[-10, 10],      [10, -10],      [150, 200]],
                               mask=[[False, False], [False, False], [True, False]])
    alphas = np.ma.MaskedArray(data=[[0.75, 0.25],   [0.5, 0.5],     [0.5, 0.5]],
                               mask=[[False, False], [False, False], [False, False]])
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='zero')
    assert np.all(interp == [-5, 0, 100])
    assert isinstance(interp, np.ma.MaskedArray)
    
    #masked values test, 'mask' mask_behavior (one point masked)
    values = np.ma.MaskedArray(data=[[-10, 10],      [10, -10],      [150, 200]],
                               mask=[[False, False], [False, False], [True, False]])
    alphas = np.ma.MaskedArray(data=[[0.75, 0.25],   [0.5, 0.5],     [0.5, 0.5]],
                               mask=[[False, False], [False, False], [False, False]])
    interp = sgrid.compute_interpolant(values, alphas, mask_behavior='mask')
    assert np.all(interp == [-5, 0, np.ma.masked])
    assert isinstance(interp, np.ma.MaskedArray)