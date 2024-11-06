"""
Quick and dirty tests of the plotting code.

All this really does it make sure it doesn't barf out

But you can look at the results to see if it makes sense
"""

from pathlib import Path

import gridded

import pytest

try:
    import matplotlib.pyplot as plt  # noqa

    from gridded.plotting.mpl_plotting import plot_ugrid ,plot_sgrid

except ImportError:
    pytestmark = pytest.mark.skip(reason="matplotlib is not installed")


EXAMPLE_DATA = Path(__file__).parent.parent / "test_data"
OUTPUT_DIR = Path(__file__).parent / "plotting_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_plot_ugrid_no_numbers():
    gds = gridded.Dataset(EXAMPLE_DATA / "UGRIDv0.9_eleven_points.nc")

    fig, axis = plt.subplots()

    plot_ugrid(axis, gds.grid, nodes=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_no_numbers")


def test_plot_ugrid_face_numbers():
    grid = gridded.Dataset(EXAMPLE_DATA / "UGRIDv0.9_eleven_points.nc").grid

    fig, axis = plt.subplots()

    plot_ugrid(axis, grid, face_numbers=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_face_numbers")


def test_plot_ugrid_node_numbers():
    grid = gridded.Dataset(EXAMPLE_DATA / "UGRIDv0.9_eleven_points.nc").grid

    fig, axis = plt.subplots()

    plot_ugrid(axis, grid, nodes=True, node_numbers=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_node_numbers")

def test_plot_ugrid_FVCOM():

    grid = gridded.Dataset(EXAMPLE_DATA / "tri_grid_example-FVCOM.nc").grid

    fig, axis = plt.subplots()

    plot_ugrid(axis, grid) #, nodes=True, node_numbers=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_FVCOM")

def test_plot_ugrid_bounds():

    grid = gridded.Dataset(EXAMPLE_DATA / "small_ugrid_zero_based.nc").grid

    fig, axis = plt.subplots()

    plot_ugrid(axis, grid) #, nodes=True, node_numbers=True)

    fig.savefig(OUTPUT_DIR / "ugrid_plot_with_bounds.png")

#############
# SGRID tests
#############

def test_plot_sgrid_no_nodes():
    grid = gridded.Dataset(EXAMPLE_DATA / "WCOFS_subset.nc").grid

    fig, axis = plt.subplots()

    plot_sgrid(axis, grid, nodes=False)

    fig.savefig(OUTPUT_DIR / "sgrid_no_nodes")

def test_plot_sgrid_and_nodes():
    grid = gridded.Dataset(EXAMPLE_DATA / "staggered_sine_channel.nc").grid

    fig, axis = plt.subplots()

    plot_sgrid(axis, grid, nodes=True)

    fig.savefig(OUTPUT_DIR / "sgrid_nodes")

