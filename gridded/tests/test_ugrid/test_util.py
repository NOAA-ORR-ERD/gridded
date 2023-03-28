"""
Tests for the utilities module.

"""


import numpy as np
from gridded.pyugrid import util


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
    result = util.asarraylike(lst)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, lst)


def test_asarraylike_array():
    """
    Passing in a list should return a np.ndarray.

    """
    arr = np.array([1, 2, 3, 4])
    result = util.asarraylike(arr)

    assert result is arr


def test_as_test_asarraylike_dummy():
    dum = DummyArrayLike()
    result = util.asarraylike(dum)
    assert result is dum
