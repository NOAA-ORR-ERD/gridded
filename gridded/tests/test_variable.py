"""
tests of Variable object

Variable objects are mostly tested implicitly in other tests,
but good to have a few explicitly for the Variable object
"""

import os
import datetime

import netCDF4
import numpy as np

from gridded import Variable, VectorVariable

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

    r1 = var.at(p1, t)
    r2 = var.at(p2, t)
    r3 = var.at(p3, t)

    assert np.all(np.logical_and(r1 == r2, r2 == r3))

def test_Variable_constant():
    v = Variable.constant(5, name="test")
    assert v.data == 5
    result = v.at((1, 2), 0)
    assert result.shape == (1, 1)
    assert result == 5
    result = v.at([(1, 2),(3, 4)], 0)
    assert result.shape == (2, 1)
    assert np.all(result == 5)

def test_Variable_timeseries():
    v = Variable(data=np.array([1, 2, 3]), name="test", time=[datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2), datetime.datetime(2020, 1, 3)])
    assert v.data.shape == (3,)
    assert v.time.min_time == datetime.datetime(2020, 1, 1)
    assert len(v.dimension_ordering) == 1
    assert v.dimension_ordering[0] == "time"
    assert v.at((0,1), v.time.data[1]) == 2
    assert v.at((0,1), v.time.data[0] + datetime.timedelta(days=0.5)) == 1.5


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
    p4 = [(1.5, 35.5, 1), (35.5, 1.5, 1)]  # expected: [[1],[masked]]

    t = var.time.min_time

    r1 = var.at(p1, t)
    r2 = var.at(p2, t)
    r3 = var.at(p3, t)
    r4 = var.at([1.5, 35.5], t)

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

    r1 = var.at(p1, t)
    r2 = var.at(p2, t)
    r3 = var.at(p3, t)

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
    p4 = [(1.5, 35.5, 1), (35.5, 1.5, 1)]  # expected: [[1],[masked]]

    t = var.time.min_time

    r1 = var.at(p1, t)
    r2 = var.at(p2, t)
    r3 = var.at(p3, t)
    r4 = var.at([1.5, 35.5], t)

    assert np.all(r1 == np.array([[1, 0], [1, 0]]))
    assert np.all(r3 == np.ma.MaskedArray([[1, 0], [0, -1]], mask=[[False, False], [True, True]]))
    assert np.all(r1 == r2)
    assert r4.shape == (1, 2)
