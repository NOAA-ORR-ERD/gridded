"""
Created on Feb 17, 2016

@author: jay.hennen
"""

import numpy as np

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


def test_interpolation_alphas():
    points = np.array(([2, 2], [2, 4], [4, 2], [4, 4]))
    alphas_c = sgrid.interpolation_alphas(points, grid='center')
    alphas_e1 = sgrid.interpolation_alphas(points, grid='edge1')
    alphas_e2 = sgrid.interpolation_alphas(points, grid='edge2')
    alphas_n = sgrid.interpolation_alphas(points, grid='node')

    answer_c = np.array([[1., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 0., 0., 0.]])
    answer_e1 = np.array([[0.5, 0., 0., 0.5],
                          [0.5, 0., 0., 0.5],
                          [0.5, 0., 0., 0.5],
                          [0.5, 0., 0., 0.5]])
    answer_e2 = np.array([[0.5, 0.5, 0., 0.],
                          [0.5, 0.5, 0., 0.],
                          [0.5, 0.5, 0., 0.],
                          [0.5, 0.5, 0., 0.]])
    answer_n = np.array([[0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25],
                         [0.25, 0.25, 0.25, 0.25]])

    assert(np.all(alphas_c == answer_c))
    assert(np.all(alphas_n == answer_n))
    assert(np.all(alphas_e1 == answer_e1))
    assert(np.all(alphas_e2 == answer_e2))


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
    answer = sgrid.locate_faces(boundaries + 1, 'node') == [0, 0]
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
