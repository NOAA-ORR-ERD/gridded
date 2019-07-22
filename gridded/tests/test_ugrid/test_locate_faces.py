import numpy as np
import pytest
from .utilities import twenty_one_triangles


# used to parametrize tests for both methods
try:
    import cell_tree2d  # noqa: ignore=F401
    methods = ['simple', 'celltree']
except ImportError:
    # no cell tree -- only test simple
    methods = ['simple']


@pytest.mark.parametrize("method", methods)
def test_single(method, twenty_one_triangles):
    ugrid = twenty_one_triangles
    face = ugrid.locate_faces((4, 6.5), method)
    assert face == 6


@pytest.mark.parametrize("method", methods)
def test_multi(method, twenty_one_triangles):
    ugrid = twenty_one_triangles
    face = ugrid.locate_faces(np.array(((4, 6.5), (7, 2))), method)
    assert (face == np.array((6, 0))).all()


@pytest.mark.parametrize("method", methods)
def test_oob(method, twenty_one_triangles):
    ugrid = twenty_one_triangles
    face = ugrid.locate_faces((0, 0), method)
    assert face == -1
    face = 0
    face = ugrid.locate_faces(np.array(((0, 0),)), method)
    assert np.array_equal(face, np.array((-1, )))
