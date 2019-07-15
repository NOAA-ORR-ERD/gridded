
import pytest

import gridded.pyugrid.ugrid as ugrid

# test stuff
from .write_nc_test_files import quad_and_triangle # noqa: ignore=F401


def test_read_flexible_mesh(quad_and_triangle):  # noqa: ignore=F811

    """
    Test if we get back a mesh from a flexible mesh
    """
    grid = ugrid.UGrid.from_nc_dataset(quad_and_triangle)
    assert grid.mesh_name == 'Mesh2'

@pytest.mark.skipif(True, reason="just broken")
def test_read_flexible_mesh_mask(quad_and_triangle):  # noqa: ignore=F811

    """
    Test if we get back a masked array from a flexible mesh (for faces and edges)
    """
    grid = ugrid.UGrid.from_nc_dataset(quad_and_triangle)
    assert grid.mesh_name == 'Mesh2'
    assert hasattr(grid.faces, 'mask'), "expected masked faces"
    assert not hasattr(grid.edges, 'mask'), "expected unmasked edges"


def test_read_flexible_mesh_nodes_per_face(quad_and_triangle): # noqa: ignore=F811
    """
    Test if we the grid contains both triangles and quads
    """
    grid = ugrid.UGrid.from_nc_dataset(quad_and_triangle)
    n_nodes_per_face = (~grid.faces.mask).sum(axis=1)
    assert set(n_nodes_per_face) == set([3, 4]), 'expected triangles and quads'
