"""
tests for NoGrid class
"""

import pathlib
import numpy as np

import pytest

from gridded.nogrid.nogrid import NoGrid

DATA_DIR = pathlib.Path(__file__).parent / "data"

print(DATA_DIR)


@pytest.fixture
def sample_data():
    data = np.loadtxt(DATA_DIR / "velocities.csv", delimiter=",", skiprows=1)
    lat = data[:, 0]
    lon = data[:, 1]
    speed = data[:, 3]
    dir = data[:, 4]

    return {'lat': lat,
            'lon': lon,
            'speed': speed,
            'dir': dir,
            }


def test_create_1(sample_data):
    grid = NoGrid(node_lat = sample_data['lat'],
                  node_lon = sample_data['lon'],
                  )

    assert grid.nodes.shape == (72, 2)






