#!/usr/bin/env python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

from gridded.time import Time


def test_init():
    """
    can one even be initialized?
    """
    t = Time()


## this needs a big data file -- could use some refactoring
# class TestTime:
#     time_var = circular_3D['time']
#     time_arr = nc.num2date(time_var[:], units=time_var.units)
#
#     def test_construction(self):
#
#         t1 = Time(TestTime.time_var)
#         assert all(TestTime.time_arr == t1.time)
#
#         t2 = Time(TestTime.time_arr)
#         assert all(TestTime.time_arr == t2.time)
#
#         t = Time(TestTime.time_var, tz_offset=dt.timedelta(hours=1))
#         print TestTime.time_arr
#         print t.time
#         print TestTime.time_arr[0] + dt.timedelta(hours=1)
#         assert t.time[0] == (TestTime.time_arr[0] + dt.timedelta(hours=1))
#
#         t = Time(TestTime.time_arr.copy(), tz_offset=dt.timedelta(hours=1))
#         assert t.time[0] == TestTime.time_arr[0] + dt.timedelta(hours=1)
#
#     def test_extrapolation(self):
#         ts = Time(TestTime.time_var)
#         before = TestTime.time_arr[0] - dt.timedelta(hours=1)
#         after = TestTime.time_arr[-1] + dt.timedelta(hours=1)
#         assert ts.index_of(before, True) == 0
#         assert ts.index_of(after, True) == 11
#         assert ts.index_of(ts.time[-1], True) == 10
#         assert ts.index_of(ts.time[0], True) == 0
#         with pytest.raises(ValueError):
#             ts.index_of(before, False)
#         with pytest.raises(ValueError):
#             ts.index_of(after, False)
#         assert ts.index_of(ts.time[-1], True) == 10
#         assert ts.index_of(ts.time[0], True) == 0


# class TestS_Depth_T1:
#
#     def test_construction(self):
#
#         test_grid = PyGrid_S(node_lon=np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]),
#                             node_lat=np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))
#
#         u = np.zeros((3, 4, 4), dtype=np.float64)
#         u[0, :, :] = 0
#         u[1, :, :] = 1
#         u[2, :, :] = 2
#
#         w = np.zeros((4, 4, 4), dtype=np.float64)
#         w[0, :, :] = 0
#         w[1, :, :] = 1
#         w[2, :, :] = 2
#         w[3, :, :] = 3
#
#         bathy_data = -np.array([[1, 1, 1, 1],
#                                [1, 2, 2, 1],
#                                [1, 2, 2, 1],
#                                [1, 1, 1, 1]], dtype=np.float64)
#
#         Cs_w = np.array([1.0, 0.6667, 0.3333, 0.0])
#         s_w = np.array([1.0, 0.6667, 0.3333, 0.0])
#         Cs_r = np.array([0.8333, 0.5, 0.1667])
#         s_rho = np.array([0.8333, 0.5, 0.1667])
#         hc = np.array([1])
#
#         b = Bathymetry(name='bathymetry', data=bathy_data, grid=test_grid, time=None)
#
#         dep = S_Depth_T1(bathymetry=b, terms=dict(zip(S_Depth_T1.default_terms[0], [Cs_w, s_w, hc, Cs_r, s_rho])), dataset='dummy')
#         assert dep is not None
#
#         corners = np.array([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0]], dtype=np.float64)
#         res, alph = dep.interpolation_alphas(corners, w.shape)
#         assert res is None  # all particles on surface
#         assert alph is None  # all particles on surface
#         res, alph = dep.interpolation_alphas(corners, u.shape)
#         assert res is None  # all particles on surface
#         assert alph is None  # all particles on surface
#
#         pts2 = corners + (0, 0, 2)
#         res = dep.interpolation_alphas(pts2, w.shape)
#         assert all(res[0] == 0)  # all particles underground
#         assert np.allclose(res[1], -2.0)  # all particles underground
#         res = dep.interpolation_alphas(pts2, u.shape)
#         assert all(res[0] == 0)  # all particles underground
#         assert np.allclose(res[1], -2.0)  # all particles underground
#
#         layers = np.array([[0.5, 0.5, .251], [1.5, 1.5, 1.0], [2.5, 2.5, 1.25]])
#         res, alph = dep.interpolation_alphas(layers, w.shape)
#         print res
#         print alph
#         assert all(res == [3, 2, 1])
#         assert np.allclose(alph, np.array([0.397539, 0.5, 0]))
