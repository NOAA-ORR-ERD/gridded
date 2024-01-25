#!/usr/bin/env python

import copy
from datetime import datetime, timedelta

import numpy as np

from gridded.time import Time, TimeSeriesError, OutOfTimeRangeError

import pytest

SAMPLE_TIMESERIES = []
start = datetime(2023, 11, 28, 12)
dt = timedelta(minutes=15)
SAMPLE_TIMESERIES = [(start + i * dt) for i in range(10)]
STS = SAMPLE_TIMESERIES


def test_init():
    """
    can one even be initialized with no data?
    """
    t = Time()


def test_init_with_timeseries():
    t = Time(SAMPLE_TIMESERIES)

    # an implementation detail -- kind of
    assert isinstance(t.data, np.ndarray)


def test_origin():
    origin = datetime(1984, 1, 1, 6)
    t = Time(SAMPLE_TIMESERIES, origin=origin)

    assert t.data[0] == origin
    assert t.data[-1] == origin + (SAMPLE_TIMESERIES[-1] - SAMPLE_TIMESERIES[0])


def test_displacement():
    disp = -timedelta(days=35, hours=12)
    t = Time(SAMPLE_TIMESERIES, displacement=disp)

    assert t.data[0] == SAMPLE_TIMESERIES[0] + disp
    assert t.data[-1] == SAMPLE_TIMESERIES[-1] + disp


def test_tz_offset():
    offset = -8
    offset = timedelta(hours=offset)
    t = Time(SAMPLE_TIMESERIES, tz_offset=offset)

    assert t.data[0] == SAMPLE_TIMESERIES[0] + offset
    assert t.data[-1] == SAMPLE_TIMESERIES[-1] + offset


def test_tz_offset_hours():
    offset = -8
    t = Time(SAMPLE_TIMESERIES, tz_offset=offset)

    offset = timedelta(hours=offset)
    assert t.data[0] == SAMPLE_TIMESERIES[0] + offset
    assert t.data[-1] == SAMPLE_TIMESERIES[-1] + offset


def test_get_time_array():
    t = Time(SAMPLE_TIMESERIES)

    ta = t.get_time_array()

    assert ta is not t.data
    assert np.array_equal(ta, SAMPLE_TIMESERIES)


def test_info():
    # just to make sure it's not broken
    t = Time(SAMPLE_TIMESERIES)

    info = t.info

    print(info)

    assert True


def test_iter():
    t = Time(SAMPLE_TIMESERIES)
    data2 = list(t)

    assert data2 == SAMPLE_TIMESERIES


def test_reset_data():
    t = Time()
    t.data = SAMPLE_TIMESERIES

    assert np.array_equal(t.data, SAMPLE_TIMESERIES)
    # an implementation detail, but want to make sure
    assert isinstance(t.data, np.ndarray)


def test_eq():
    t1 = Time(data=copy.copy(SAMPLE_TIMESERIES))
    t2 = Time(data=copy.copy(SAMPLE_TIMESERIES))

    assert t1 == t2


def test_eq_diff_length():
    data2 = copy.copy(SAMPLE_TIMESERIES)
    del data2[-1]
    t1 = Time(data=copy.copy(SAMPLE_TIMESERIES))
    t2 = Time(data=data2)

    assert t1 != t2


def test_eq_diff_values():
    data2 = copy.copy(SAMPLE_TIMESERIES)
    data2[4] = data2[4] + timedelta(minutes=5)
    t1 = Time(data=copy.copy(SAMPLE_TIMESERIES))
    t2 = Time(data=data2)

    assert t1 != t2


def test_eq_diff_one_constant():
    data2 = copy.copy(SAMPLE_TIMESERIES)
    t1 = Time(data=data2)
    t2 = Time(data=[data2[2]])

    assert t1 != t2


def test_eq_constant_time():
    t1 = Time.constant_time()
    t2 = Time.constant_time()

    assert t1 == t2


def test_eq_different_type():
    """
    A Time object is never equal to any other type
    """
    t = Time(SAMPLE_TIMESERIES)

    assert not t == 'a string'
    assert not "a string" == t


def test_out_of_order():
    data2 = copy.copy(SAMPLE_TIMESERIES)
    data2.insert(1, data2[4])

    with pytest.raises(TimeSeriesError):
        Time(data2)


def test_decending():
    data2 = copy.copy(SAMPLE_TIMESERIES)
    data2.reverse()

    with pytest.raises(TimeSeriesError):
        Time(data2)


def test_duplicate():
    data2 = SAMPLE_TIMESERIES[:5] + SAMPLE_TIMESERIES[4:]

    with pytest.raises(TimeSeriesError):
        Time(data2)


def test_min_max():
    t = Time(SAMPLE_TIMESERIES)

    assert t.min_time == SAMPLE_TIMESERIES[0]
    assert t.max_time == SAMPLE_TIMESERIES[-1]


def test_time_in_bounds():
    t = Time(SAMPLE_TIMESERIES)

    assert t.time_in_bounds(SAMPLE_TIMESERIES[0])
    assert t.time_in_bounds(SAMPLE_TIMESERIES[2] + timedelta(minutes=10))
    assert t.time_in_bounds(SAMPLE_TIMESERIES[-1])

    assert not t.time_in_bounds(SAMPLE_TIMESERIES[0] - timedelta(seconds=1))
    assert not t.time_in_bounds(SAMPLE_TIMESERIES[-1] + timedelta(seconds=1))


def test_valid_time():
    t = Time(SAMPLE_TIMESERIES)

    assert t.valid_time(SAMPLE_TIMESERIES[0]) is None
    assert t.valid_time(SAMPLE_TIMESERIES[2] + timedelta(minutes=10)) is None
    assert t.valid_time(SAMPLE_TIMESERIES[-1]) is None

    with pytest.raises(OutOfTimeRangeError):
        t.valid_time(SAMPLE_TIMESERIES[0] - timedelta(seconds=1))
    with pytest.raises(OutOfTimeRangeError):
        t.valid_time(SAMPLE_TIMESERIES[-1] + timedelta(seconds=1))


@pytest.mark.parametrize("dt, expected",
                         [(STS[3], 1.0),  # exact time
                          (STS[4] + (STS[5] - STS[4]) / 2, 0.5),  # in the middle
                          (STS[4] + (STS[5] - STS[4]) / 4, 0.25),  # in the middle
                          (STS[0], 0.0),  # at the beginning
                          (STS[-1], 1.0),  # at the end
                          ])
def test_interp_alpha(dt, expected):
    t = Time(SAMPLE_TIMESERIES)

    print(dt)
    alpha = t.interp_alpha(dt)

    assert alpha == expected


@pytest.mark.parametrize("dt, expected",
                         [(STS[0] - timedelta(seconds=1), 0.0),  # a little before
                          (STS[-1] + timedelta(seconds=1), 1.0),  # a little after
                          ])
def test_interp_alpha_outside(dt, expected):
    t = Time(SAMPLE_TIMESERIES)

    print(dt)
    with pytest.raises(OutOfTimeRangeError):
        alpha = t.interp_alpha(dt)

    alpha = t.interp_alpha(dt, extrapolate=True)
    assert alpha == expected


#@pytest.mark.xfail
@pytest.mark.parametrize("shift, expected",
                         [(-timedelta(days=365), 1.0),  # before
                          (timedelta(days=365), 1.0),  # after
                          (timedelta(0), 1.0),  # on the nose
                          ])
def test_interp_alpha_constant_time(shift, expected):
    """
    What should the constant time Time object give for alphas?

    Turns out it should never be called.

    So is a bit of over testing, but there you go.

    -- see gridded.Variable.at -- this won't get called for constant time object.
    """
    t = Time.constant_time()

    print(t.data)
    print(t.min_time)
    print(t.max_time)

    with pytest.raises(TimeSeriesError):
        alpha = t.interp_alpha(t.data[0] + shift)

# def test_index_of_contant_time():
#     pass


## this needs a big data file -- could use some refactoring
## It would be good to test the netcdf stuff.
## there must be a sample file with time in it somewhere??
## However -- this is testing a lot that doesn't need a data file.

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
