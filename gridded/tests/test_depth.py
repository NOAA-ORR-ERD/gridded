#!/usr/bin/env python

import os
import sys
import datetime
import pytest
import numpy as np
import netCDF4 as nc
import gridded

from gridded.variable import Variable, VectorVariable
from gridded.tests.utilities import get_test_file_dir, get_test_cdl_filelist
from gridded.grids import Grid_S
from gridded.time import Time
from gridded.utilities import search_dataset_for_variables_by_varname

from gridded.depth import S_Depth, L_Depth

test_dir = get_test_file_dir()
cdl_files = get_test_cdl_filelist()

def valid_depth_test_dataset(ds):
    '''
    A filter to be applied to the loaded .cdl that excludes the file if it's almost definitely
    not going to work. A file must contain at least 4D variable for it to pass this test
    '''
    for k, v in ds.variables.items():
        if len(v.dimensions) == 4:
            return True
    return False

def test_from_netCDF():
    for fn in cdl_files:
        ds = nc.Dataset.fromcdl(fn, ncfilename=fn+'.nc')
        if not valid_depth_test_dataset(ds):
            continue
        d = S_Depth.from_netCDF(dataset=ds)
    


@pytest.fixture(scope="module")
def get_s_depth():
    """
    This is setup for a ROMS S-level Depth that is on a square grid with a center
    mound. Control vars: sz=xy size, center_el=height of the mound in meters,
    d0=general depth in meters, sig=steepness of mound, nz=number of z levels.
    """
    sz = 40
    center_el = 10
    d0 = 20
    sig = 0.75
    nz = 11
    node_lat, node_lon = np.mgrid[0:sz, 0:sz]
    b_data = np.empty((sz, sz))
    for x in range(0, sz):
        for y in range(0, sz):
            b_data[x, y] = d0 - center_el * np.exp(
                -0.1
                * (
                    (x - (sz / 2)) ** 2 / 2.0 * ((sig) ** 2)
                    + (y - (sz / 2)) ** 2
                    / 2.0
                    * ((sig) ** 2)
                )
            )
    z_data = np.empty((3, sz, sz))
    for t in range(0, 3):
        for x in range(0, sz):
            for y in range(0, sz):
                z_data[t, x, y] = (t - 1.0) / 2.0
    g = Grid_S(node_lon=node_lon, node_lat=node_lat)
    bathy = Variable(name="bathy", grid=g, data=b_data)
    t_data = np.array(
        [
            Time.constant_time().data[0]
            + datetime.timedelta(minutes=10 * d)
            for d in range(0, 3)
        ]
    )
    zeta = Variable(
        name="zeta",
        time=Time(data=t_data),
        grid=g,
        data=z_data,
    )

    s_w = np.linspace(-1, 0, nz)
    s_rho = (s_w[0:-1] + s_w[1:]) / 2
    # equidistant layers, no stretching
    Cs_w = np.linspace(-1, 0, nz)
    Cs_w = 1 - 1 / np.exp(2 * Cs_w)
    Cs_w /= -Cs_w[0]
    Cs_r = (Cs_w[0:-1] + Cs_w[1:]) / 2
    hc = np.array([0])

    sd = S_Depth(
        time=zeta.time,
        grid=zeta.grid,
        bathymetry=bathy,
        zeta=zeta,
        terms={
            "s_w": s_w,
            "s_rho": s_rho,
            "Cs_w": Cs_w,
            "Cs_r": Cs_r,
            "hc": hc,
        },
    )
    return sd


@pytest.fixture(scope="module")
def get_l_depth():
    """
    This sets up a HYCOM level depth where surface is index 0 and bottom is last index
    """

    depth_levels = np.array(([0, 1, 2, 4, 6, 10]))
    ld = L_Depth(
        surface_index=0,
        bottom_index=len(depth_levels) - 1,
        terms={"depth_levels": depth_levels},
    )
    return ld


class Test_S_Depth(object):
    def test_construction(self, get_s_depth):
        assert get_s_depth is not None

    def test_structure(self, get_s_depth):
        sd = get_s_depth
        sd.Cs_w = np.linspace(-1, 0, len(sd.Cs_w))
        sd.Cs_r = (sd.Cs_w[0:-1] + sd.Cs_w[1:]) / 2

        sz = sd.bathymetry.data.shape[1]
        levels = sd.get_section(sd.time.data[1], "w")[
            :, :, :
        ]  # 3D cross section
        edge = np.linspace(20, 0, len(sd.Cs_w))
        center = np.linspace(10, 0, len(sd.Cs_w))
        assert np.allclose(levels[:, 0, 0], edge)
        assert np.allclose(
            levels[:, int(sz / 2), int(sz / 2)], center
        )

    def test_interpolation_alphas_bottom(self, get_s_depth):
        '''
        We will focus on the center mound to see if the correct alphas and
        indices are returned for various points nearby.
        interpolation_alphas(self, points, time, data_shape, _hash=None, extrapolate=False):
        '''
        sd = get_s_depth
        # query center mound (20,20) at 3 depths. 1st point is 0.1m off the seafloor and 10% of
        # the distance to the next s-layer (meaning, expected alpha returned is 10%)
        # 2nd point is directly on the seafloor (should register as 'in bounds' and 0 alpha)
        # 3rd point is 0.1m underground, and should indicate with -2 alpha
        points = np.array([[20,20, 9.9],[20,20,10.0], [20,20,10.1]])
        ts = sd.time.data[1]
        assert sd.bottom_boundary_condition == 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        expected_idx = np.ma.array(np.array([0,0,-1]), mask = [False, False, True])
        expected_alpha = np.ma.array(np.array([0.1, 0, -1]), mask = [False, False, True])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))
        
        sd.bottom_boundary_condition == 'extrapolate'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        expected_idx = np.ma.array(np.array([0,0,-1]), mask = [False, False, False])
        expected_alpha = np.ma.array(np.array([0.1, 0, 1]), mask = [False, False, False])

    def test_interpolation_alphas_surface(self, get_s_depth):
        sd = get_s_depth
        points = np.array([[20,20, 0],[20,20,0.1], [20,20,-0.1]])
        ts = sd.time.data[1]
        assert sd.surface_boundary_condition == 'extrapolate'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface.
        expected_idx = np.ma.array(np.array([10,9,10]), mask = [False, False, False])
        expected_alpha = np.ma.array(np.array([0, 0.9, 0]), mask = [False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))
        
        sd.surface_boundary_condition == 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface.
        expected_idx = np.ma.array(np.array([-1,9,-1]), mask = [True, False, True])
        expected_alpha = np.ma.array(np.array([0, 0.9, 0]), mask = [True, False, True])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))

        sd.surface_boundary_condition == 'extrapolate'
        # switch to timestep with -0.5m zeta
        ts = sd.time.data[0]
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        expected_idx = np.ma.array(np.array([10,10,10]), mask = [False, False, False])
        expected_alpha = np.ma.array(np.array([0, 0, 0]), mask = [False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))

        sd.surface_boundary_condition = 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_w_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface.
        expected_idx = np.ma.array(np.array([-1,-1,-1]), mask = [True, True, True])
        expected_alpha = np.ma.array(np.array([0, 0.9, 0]), mask = [True, True, True])
        assert np.all(idx.mask == expected_idx.mask)
        assert np.all(alphas.mask == expected_alpha.mask)

@pytest.fixture(scope="module")
def get_database_nc():
    """
    This sets up netcdf dataset for interpolation test
    """
    L_depth_file = os.path.join(test_dir, 'test_L_Depth.nc')

    ncfile = nc.Dataset(L_depth_file)
    ds = gridded.Dataset(L_depth_file)
    depth = gridded.depth.Depth.from_netCDF(filename=L_depth_file)
    time = gridded.time.Time.from_netCDF(filename=L_depth_file, datavar=ncfile['u'])

    return time, depth, ds

class Test_L_Depth(object):
    def test_construction(self, get_l_depth):

        assert get_l_depth is not None

    def test_interpolation_alphas_all_surface(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([0, 0, 0], [1, 1, 0], [4, 9, 0]))
        idxs, alphas = ld.interpolation_alphas(points)

        assert idxs is None
        assert alphas is None

    def test_interpolation_alphas_1_surface(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([0, 0, 0],
                           [1, 1, 0.5],
                           [0, 0, 1],
                          ))
        idxs, alphas = ld.interpolation_alphas(points)

        assert np.all(idxs == np.array([1, 1, 1]))
        assert np.all(np.isclose(alphas, np.array([0, 0.5, 1])))

    def test_interpolation_alphas_above_surface(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([0, 0, -1], [0, 0, 10]))
        idxs, alphas = ld.interpolation_alphas(points)

        assert np.all(idxs == np.array([-1, 5]))
        assert np.all(alphas == np.array([-3, 1]))

    def test_interpolation_alphas_below_grid(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([0, 0, 20], [0, 0, 4.25]))
        idxs, alphas = ld.interpolation_alphas(points)

        assert np.all(idxs == np.array([-1, 4]))
        assert np.all(alphas == np.array([-2, 0.125]))

    def test_interpolation_alphas_full(self, get_l_depth):
        ld = get_l_depth
        points = np.array(([1, 2, 0], [3, 4, 5.5], [3, 4, 15.5], [3, 4, -10], [3, 4, 10]))
        idxs, alphas = ld.interpolation_alphas(points)

        assert np.all(idxs == np.array([1, 4, -1, -1, 5]))
        assert np.all(alphas == np.array([0, 0.75, -2, -3, 1]))

    @pytest.mark.parametrize("index", (0, 5, 10, 20))
    def test_vertical_interpolation_within_grid(self, index, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], depth.depth_levels[index]), )

        assert ds.variables['u'].at(points=points,time=time.data[0])[0] \
               == ds.variables['u'].data[0,index,1,1]

    def test_vertical_interpolation_onsurface(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], 0.0), )

        assert ds.variables['u'].at(points=points,time=time.data[0])[0] \
               == ds.variables['u'].data[0,0,1,1]

    def test_vertical_interpolation_belowgrid(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], depth.depth_levels[-1]+100.), )

        assert np.isnan(ds.variables['u'].at(points=points,time=time.data[0])[0])

    def test_vertical_interpolation_abovesurface(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], -10.), )

        assert np.isnan(ds.variables['u'].at(points=points,time=time.data[0])[0])

    def test_vertical_interpolation_full(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], -10.), (ds.grid.node_lon[1], ds.grid.node_lat[1], 0.), (ds.grid.node_lon[1], ds.grid.node_lat[1], depth.depth_levels[3]))

        assert np.isnan(ds.variables['u'].at(points=points,time=time.data[0])[0])
        assert ds.variables['u'].at(points=points,time=time.data[0])[1] == ds.variables['u'].data[0,0,1,1]
        assert ds.variables['u'].at(points=points,time=time.data[0])[2] == ds.variables['u'].data[0,3,1,1]