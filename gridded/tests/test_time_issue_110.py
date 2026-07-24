from datetime import datetime, timedelta

import numpy as np

from gridded.time import Time, parse_time_offset


def test_new_tz_offset_does_not_mutate_input_array():
    arr = np.array([datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 6)])
    original = arr.copy()

    Time(arr, tz_offset=0, new_tz_offset=5)

    assert np.array_equal(arr, original)


def test_time_from_time_does_not_share_data():
    arr = np.array([datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 6)])
    t1 = Time(arr, tz_offset=0)
    t2 = Time(t1, tz_offset=0)

    t2.tz_offset = 3

    assert t1.data[0] == datetime(2024, 1, 1, 0)


def test_constant_time_returns_isolated_instances():
    c1 = Time.constant_time()
    before = c1.data.copy()

    c2 = Time.constant_time()
    c2.displacement = timedelta(hours=72)
    c3 = Time.constant_time()

    assert np.array_equal(c1.data, before)
    assert np.array_equal(c3.data, before)


def test_parse_time_offset_without_since(caplog):
    caplog.clear()

    offset, name = parse_time_offset("days")

    assert offset is None
    assert name is None
    assert any("No 'since' in time units string" in rec[2] for rec in caplog.record_tuples)
