"""
tests of Variable object

Variable objects are mostly tested implicitly in other tests,
but good to have a few explicitly for the Variable object
"""

import datetime

import netCDF4
import numpy as np
import pytest

from gridded import Variable, VectorVariable
from gridded.grids import Grid_S
from gridded.time import OutOfTimeRangeError, Time

from .utilities import TEST_DATA

sample_sgrid_file = TEST_DATA / "arakawa_c_test_grid.nc"


def test_create_from_netcdf_dataset():
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = Variable.from_netCDF(
        dataset=ds,
        varname="u",
    )
    print(var.info)

    assert var.data.shape == (1, 12, 11)


def test_Variable_api_at_function():
    """
    Test to ensure that the .at() function can accept a list, array, or separate lon/lat array,
    and that the output format is the same in all cases
    """
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = Variable.from_netCDF(
        dataset=ds,
        varname="u",
    )

    p1 = [(1.5, 35.5), (2.5, 35.5), (2.5, 35.5), (2.5, 35.5), (2.5, 35.5)]
    p2 = np.array(p1)
    p3 = p2.T

    t = var.time.min_time

    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(lons=p3[0], lats=p3[1], time=t, _mem=False)

    assert np.all(np.logical_and(r1 == r2, r2 == r3))


def test_Variable_api_at_function_edge_cases():
    """
    Test for edge cases of input point shape (eg, 2x2) as well as single values
    """
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = Variable.from_netCDF(
        dataset=ds,
        varname="u",
    )
    p1 = (1.5, 35.5)
    p2 = (1.5, 35.5, 1)  # expected: r2 == r1
    p3 = [(1.5, 35.5), (35.5, 1.5)]  # expected: [[1],[masked]]
#    p4 = [(1.5, 35.5, 1), (35.5, 1.5, 1)]  # expected: [[1],[masked]]

    t = var.time.min_time

    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(p3, t, _mem=False)
    r4 = var.at(lons=1.5, lats=35.5, time=t, _mem=False)

    assert np.all(
        r1
        == np.array(
            [
                [
                    1,
                ],
                [
                    1,
                ],
            ]
        )
    )
    assert np.all(
        r3
        == np.ma.MaskedArray(
            [
                [
                    1,
                ],
                [
                    0,
                ],
            ],
            mask=[False, True],
        )
    )
    assert np.all(r1 == r2)
    assert r4.shape == (1, 1)


def test_VectorVariable_api_at_function():
    """
    Test to ensure that the .at() function can accept a list, array, or separate lon/lat array,
    and that the output format is the same in all cases
    """
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = VectorVariable.from_netCDF(
        dataset=ds,
        varnames=["u", "v"],
    )

    p1 = [(1.5, 35.5), (2.5, 35.5), (2.5, 35.5), (2.5, 35.5), (2.5, 35.5)]
    p2 = np.array(p1)
    p3 = p2.T

    t = var.time.min_time

    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(lons=p3[0], lats=p3[1], time=t, _mem=False)

    assert np.all(np.logical_and(r1 == r2, r2 == r3))


def test_VectorVariable_api_at_function_edge_cases():
    """
    Test for edge cases of input point shape (eg, 2x2) as well as single values
    """
    ds = netCDF4.Dataset(sample_sgrid_file)

    var = VectorVariable.from_netCDF(
        dataset=ds,
        varnames=["u", "v"],
    )
    p1 = (1.5, 35.5)
    p2 = (1.5, 35.5, 1)  # expected: r2 == r1
    p3 = [(1.5, 35.5), (35.5, 1.5)]  # expected: [[1],[masked]]
    # p4 = [(1.5, 35.5, 1), (35.5, 1.5, 1)]  # expected: [[1],[masked]]

    t = var.time.min_time

    r1 = var.at(p1, t, _mem=False)
    r2 = var.at(p2, t, _mem=False)
    r3 = var.at(p3, t, _mem=False)
    r4 = var.at(lons=1.5, lats=35.5, time=t, _mem=False)

    assert np.all(r1 == np.array([[1, 0], [1, 0]]))
    assert np.all(r3 == np.ma.MaskedArray([[1, 0], [0, -1]], mask=[[False, False], [True, True]]))
    assert np.all(r1 == r2)
    assert r4.shape == (1, 2)


def make_time_series_variable(name="u"):
    """
    A Variable on a plain 4x4 grid with three time steps, so that a query
    outside the time range is out of range rather than constant in time.
    """
    node_lat, node_lon = np.mgrid[0:4, 0:4]
    grid = Grid_S(node_lon=node_lon, node_lat=node_lat)
    times = np.array([datetime.datetime(2020, 1, 1) + datetime.timedelta(hours=h) for h in range(3)])
    data = np.stack([np.full((4, 4), float(step)) for step in range(3)])
    return Variable(name=name, grid=grid, time=Time(data=times), data=data)


def test_Variable_at_memo_respects_extrapolate():
    """
    A memoized result must not be handed back to a call that asked for
    different extrapolation behaviour.
    """
    var = make_time_series_variable()

    p = [(1.5, 1.5)]
    t = var.time.max_time + datetime.timedelta(days=1)

    assert var.at(p, t, extrapolate=True) == 2.0
    with pytest.raises(OutOfTimeRangeError):
        var.at(p, t, extrapolate=False)


def test_Variable_at_memo_respects_unmask():
    """
    A memoized filled result must not be handed back to a call that asked for
    a masked array.
    """
    ds = netCDF4.Dataset(sample_sgrid_file)
    var = Variable.from_netCDF(dataset=ds, varname="u")

    p = [(35.5, 1.5)]  # off the grid, so the result is masked
    t = var.time.min_time

    filled = var.at(p, t, unmask=True)
    assert not np.ma.is_masked(filled)

    masked = var.at(p, t, unmask=False)
    assert np.ma.is_masked(masked)


def test_Variable_at_memo_reused_for_identical_calls():
    """
    The options taking part in the key must not stop identical calls from
    being served out of the memo.
    """
    ds = netCDF4.Dataset(sample_sgrid_file)
    var = Variable.from_netCDF(dataset=ds, varname="u")

    p = [(1.5, 35.5)]
    t = var.time.min_time

    r1 = var.at(p, t)
    assert len(var._result_memo) == 1
    r2 = var.at(p, t)
    assert len(var._result_memo) == 1
    assert np.all(r1 == r2)


def test_Variable_at_memo_respects_boundary_conditions():
    """
    The boundary conditions change what ``at`` returns for points outside the
    depth interval, so they have to take part in the key.
    """
    ds = netCDF4.Dataset(sample_sgrid_file)
    var = Variable.from_netCDF(dataset=ds, varname="u")

    _hash = var._get_hash(np.array([[1.5, 35.5, 0.0]]), var.time.min_time)
    keys = {
        var._result_key(_hash, False, False, {}),
        var._result_key(_hash, False, False, {"surface_boundary_condition": "mask"}),
        var._result_key(_hash, False, False, {"bottom_boundary_condition": "extrapolate"}),
    }
    assert len(keys) == 3


def test_VectorVariable_at_memo_respects_extrapolate():
    """
    As for Variable: the vector memo must not answer a call that asked for
    different extrapolation behaviour.
    """
    var = VectorVariable(
        name="vel",
        variables=[make_time_series_variable("u"), make_time_series_variable("v")],
    )

    p = [(1.5, 1.5)]
    t = var.time.max_time + datetime.timedelta(days=1)

    assert np.all(var.at(p, t, extrapolate=True) == 2.0)
    with pytest.raises(OutOfTimeRangeError):
        var.at(p, t, extrapolate=False)


def test_VectorVariable_at_memo_respects_unmask():
    """
    As for Variable: a filled result must not be handed back to a call that
    asked for a masked array.
    """
    ds = netCDF4.Dataset(sample_sgrid_file)
    var = VectorVariable.from_netCDF(dataset=ds, varnames=["u", "v"])

    p = [(35.5, 1.5)]  # off the grid, so the result is masked
    t = var.time.min_time

    filled = var.at(p, t, unmask=True)
    assert not np.ma.is_masked(filled)

    masked = var.at(p, t, unmask=False)
    assert np.ma.is_masked(masked)
