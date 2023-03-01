#!/usr/bin/env python
"""
Script to test GNOME with plume element type
 - weibull droplet size distribution

Simple map and simple current mover

Rise velocity and vertical diffusion

This is simply making a point source with a given distribution of droplet sizes

"""
import gridded
import netCDF4
import pytest
import numpy as np

from pathlib import Path
HERE = Path(__file__).parent

test_file = HERE/'cropped_test.nc'

u_data = []
u_interpolation = []
points = ()

time_step = 1
lat_node = 158
lon_node = 120

def test_gridded():
   print('gemrge;lrgkel;gegkelge', test_file)
   ds = gridded.Dataset(str(test_file))

@pytest.fixture
def netcdf_example():

    #nc = netCDF4.Dataset(test_file)
    ds = gridded.Dataset(str(test_file))
    #depth = gridded.depth.Depth.from_netCDF(filename=str(test_file))
    #time = gridded.time.Time.from_netCDF(filename=str(test_file), dataset=nc, datavar=nc['u'])

    return ds #, depth, time

# @pytest.mark.parametrize("index", (5, 10))
# def test_layer_interpolation(netcdf_example, index):

    # points = ((ds.grid.node_lon[lon_node], ds.grid.node_lat[lat_node], depth.depth_levels[index]), )

    # assert ds.variables['u'].at(points=points,time=time.data[time_step])[0] == ds.variables['u'].data[time_step,index,lat_node,lon_node]

# def test_layer_below_bottom_layer(netcdf_example):
    # ds = netcdf_example
    # points = ((ds.grid.node_lon[lon_node], ds.grid.node_lat[lat_node], depth.depth_levels[-1]+100.), )

    # assert ds.variables['u'].at(points=points,time=time.data[time_step])[0] == 0.0

# def test_layer_both_surface_subsurface(netcdf_example):
    # points = ((ds.grid.node_lon[lon_node], ds.grid.node_lat[lat_node], depth.depth_levels[0]),
             # (ds.grid.node_lon[lon_node], ds.grid.node_lat[lat_node], depth.depth_levels[10]),
             # (ds.grid.node_lon[lon_node], ds.grid.node_lat[lat_node], depth.depth_levels[20]),
             # )

    # assert ds.variables['u'].at(points=points,time=time.data[time_step])[0] == ds.variables['u'].data[time_step,0,lat_node,lon_node]
    # assert ds.variables['u'].at(points=points,time=time.data[time_step])[1] == ds.variables['u'].data[time_step,10,lat_node,lon_node]
    # assert ds.variables['u'].at(points=points,time=time.data[time_step])[2] == ds.variables['u'].data[time_step,20,lat_node,lon_node]


