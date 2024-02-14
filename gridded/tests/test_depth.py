#!/usr/bin/env python

import os
import sys
import datetime
import pytest
import numpy as np
import netCDF4 as nc
import gridded
from math import sqrt

from gridded.variable import Variable, VectorVariable
from gridded.tests.utilities import get_test_file_dir, get_test_cdl_filelist
from gridded.grids import Grid_S
from gridded.time import Time
from gridded.utilities import search_dataset_for_variables_by_varname

from gridded.depth import S_Depth, ROMS_Depth, FVCOM_Depth, L_Depth

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
        d = S_Depth.from_netCDF(data_file=ds, grid_file=ds)

@pytest.fixture(scope="module")
def get_fvcom_depth():
    ds = nc.Dataset(os.path.join(test_dir, 'UGRIDv0.9_eleven_points_with_depth.nc'))
    return FVCOM_Depth.from_netCDF(data_file=ds, grid_file=ds)


@pytest.fixture(scope="function")
def get_roms_depth():
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
    
    #bathymetry data
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
    
    #zeta data        
    z_data = np.empty((3, sz, sz))
    for t in range(0, 3):
        for x in range(0, sz):
            for y in range(0, sz):
                z_data[t, x, y] = (t - 1.0) / 2.0
    g = Grid_S(node_lon=node_lon, node_lat=node_lat)
    bathy = Variable(name="bathy", grid=g, data=b_data)
    
    #time data
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
    Cs_r = (Cs_w[0:-1] + Cs_w[1:]) / 2
    hc = np.array([0,])

    sd = ROMS_Depth(
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
    ld1 = L_Depth(
        surface_index=0,
        bottom_index=len(depth_levels) - 1,
        terms={"depth_levels": depth_levels},
    )
    ld2 = L_Depth(
        surface_index=len(depth_levels) - 1,
        bottom_index=0,
        terms={"depth_levels": depth_levels[::-1]},
    )
    return ld1, ld2

class Test_S_Depth(object):
    def test_apply_boundary_conditions(self):
        #Test 1 surf 0 bottom 9 surf extrapolate bottom mask
        depths = [-0.1, 0, 0.1, 1, 6, 8, 8.9, 9, 9.1] #for human reference
        #v = v1 - (1-a)*(v1 - v0)
        indices = np.ma.masked_array(data=np.array([-1, 0, 0, 1, 3, 8, 8, 9, 9]), mask = np.zeros((9), dtype=bool))
        indices.mask[4] = True
        alphas = np.ma.masked_array(data=np.ones((indices.shape))*np.nan, mask=np.zeros((indices.shape), dtype=bool))
        alphas.mask[4] = True
        surface_index = 0
        bottom_index = 9
        surface_boundary_condition = 'extrapolate'
        bottom_boundary_condition = 'mask'
        expected_indices = indices.copy()
        expected_indices.mask[-2:] = True # [False, False, False, False, True, False, False, True, True]
        
        expected_alphas = alphas.copy()
        expected_alphas[:] = [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 10000, 10000]
        expected_alphas.mask = alphas.mask.copy()
        expected_alphas.mask[-2:] = True # [False, False, False, False, True, False, False, True, True]
        #10K doesnt actually matter, just an alternative to nan since they should be masked
        sd = S_Depth(terms={'s_w':np.arange(0,-1, 10)})
        idx, alp, oob_mask = sd._apply_boundary_conditions(indices, alphas, surface_index, bottom_index, surface_boundary_condition, bottom_boundary_condition)
        assert np.all(idx == expected_indices)
        assert np.all(idx.mask == expected_indices.mask)
        assert alp[0] == expected_alphas[0]
        assert alp[4] is np.ma.masked
        assert alp[-2] is np.ma.masked
        assert alp[-1] is np.ma.masked
        assert np.all(alp.mask == expected_alphas.mask)
        
        #Test 2 surf 9 bottom 0 surf extrapolate bottom mask
        depths = [-1, 0, 0.1, 1, 6, 8, 8.9, 9, 9.1] #for human reference
        #v = v1 - (1-a)*(v1 - v0)
        indices = np.ma.masked_array(data=np.array([9, 8, 8, 7, 3, 1, 0, 0, -1]), mask = np.zeros((9), dtype=bool))
        indices.mask[4] = True
        alphas = np.ma.masked_array(data=np.ones((indices.shape))*np.nan, mask=np.zeros((indices.shape), dtype=bool))
        alphas.mask[4] = True
        surface_index = 9
        bottom_index = 0
        surface_boundary_condition = 'extrapolate'
        bottom_boundary_condition = 'mask'
        expected_indices = indices.copy()
        expected_indices.mask[-1] = True # [True, True, False, False, True, False, False, False, False]
        
        expected_alphas = alphas.copy()
        expected_alphas[:] = [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1]
        expected_alphas.mask = alphas.mask.copy()
        expected_alphas.mask[-1] = True # [True, True, False, False, True, False, False, False, False]
        #10K doesnt actually matter, just an alternative to nan since they should be masked
        
        sd = S_Depth(terms={'s_w':np.arange(-1,0, 10)})
        idx, alp, oob_mask = sd._apply_boundary_conditions(indices, alphas, surface_index, bottom_index, surface_boundary_condition, bottom_boundary_condition)
        assert np.all(idx == expected_indices)
        assert np.all(idx.mask == expected_indices.mask)
        assert alp[0] == expected_alphas[0]
        assert alp[4] is np.ma.masked
        assert alp[-1] is np.ma.masked
        assert np.all(alp.mask == expected_alphas.mask)
        
        #Test 3 surf 9 bottom 0 surf extrapolate bottom extrapolate
        depths = [-1, 0, 0.1, 1, 6, 8, 8.9, 9, 9.1] #for human reference
        #v = v1 - (1-a)*(v1 - v0)
        indices = np.ma.masked_array(data=np.array([9, 8, 8, 7, 3, 1, 0, 0, -1]), mask = np.zeros((9), dtype=bool))
        indices.mask[4] = True
        alphas = np.ma.masked_array(data=np.ones((indices.shape))*np.nan, mask=np.zeros((indices.shape), dtype=bool))
        alphas.mask[4] = True
        surface_index = 9
        bottom_index = 0
        surface_boundary_condition = 'extrapolate'
        bottom_boundary_condition = 'extrapolate'
        expected_indices = indices.copy()
        
        expected_alphas = alphas.copy()
        expected_alphas[:] = [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1]
        expected_alphas.mask = alphas.mask.copy()
        #10K doesnt actually matter, just an alternative to nan since they should be masked
        
        sd = S_Depth(terms={'s_w':np.arange(-1,0, 10)})
        idx, alp, oob_mask = sd._apply_boundary_conditions(indices, alphas, surface_index, bottom_index, surface_boundary_condition, bottom_boundary_condition)
        assert np.all(idx == expected_indices)
        assert np.all(idx.mask == expected_indices.mask)
        assert alp[0] == 0
        assert alp[4] is np.ma.masked
        assert alp[-1] == 1
        assert np.all(alp.mask == expected_alphas.mask)
        
    def test_mask_surface(self):
        #Test 3 surf 9 bottom 0 surf extrapolate bottom extrapolate
        depths = [-1, 0, 0.1, 1, 6, 8, 8.9, 9, 9.1] #for human reference
        #v = v1 - (1-a)*(v1 - v0)
        indices = np.ma.masked_array(data=np.array([9, 8, 8, 7, 3, 1, 0, 0, -1]), mask = np.zeros((9), dtype=bool))
        indices.mask[4] = True
        alphas = np.ma.masked_array(data=np.ones((indices.shape))*np.nan, mask=np.zeros((indices.shape), dtype=bool))
        alphas.mask[4] = True
        surface_index = 9
        bottom_index = 0
        surface_boundary_condition = 'mask'
        bottom_boundary_condition = 'mask'
        expected_indices = indices.copy()
        expected_indices.mask[0] = True
        expected_indices.mask[-1] = True
        
        expected_alphas = alphas.copy()
        expected_alphas[:] = [0, 1, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 1]
        expected_alphas.mask = alphas.mask.copy()
        expected_alphas.mask[0] = True
        expected_alphas.mask[-1] = True
        
        sd = S_Depth(terms={'s_w':np.arange(-1,0, 10)})
        idx, alp, oob_mask = sd._apply_boundary_conditions(indices, alphas, surface_index, bottom_index, surface_boundary_condition, bottom_boundary_condition)
        assert np.all(idx == expected_indices)
        assert np.all(idx.mask == expected_indices.mask)
        assert alp[0] is np.ma.masked
        assert alp[4] is np.ma.masked
        assert alp[-1] is np.ma.masked
        assert np.all(alp.mask == expected_alphas.mask)
            

class Test_ROMS_Depth(object):
    def test_construction(self, get_roms_depth):
        assert get_roms_depth is not None
        assert get_roms_depth.num_levels == 11

    def test_interpolation_alphas(self, get_roms_depth):
        '''
        We will focus on the center mound to see if the correct alphas and
        indices are returned for various points nearby.
        interpolation_alphas(self, points, time, data_shape, _hash=None, extrapolate=False):
        
        With ROMS, the bottom level is index 0 and the top level is last index (in this case, 10)
        # query center mound (20,20) at 3 depths. 1st point is 0.1m off the seafloor and 10% of
        # the distance to the next s-layer (meaning, expected alpha returned is 10%)
        # this is an index of 0 and an alpha of 0.1
        # 2nd point is directly on the seafloor (should register as 'in bounds' index 0 and 0 alpha)
        # 3rd point is 0.1m underground, and should indicate with masked value.
        # 4th point is off grid, masked index and alpha expected.
        # 5th point is 0.1m ABOVE surface. Expected index of 10 and alpha of 0
        # 6th point is directly at surface. Expected index of 10 and alpha of 0
        #7th point is 0.1m below surface. Expected index of 9 and alpha of 0.9
        '''
        sd = get_roms_depth
        points = np.array([[20,20, 9.9],[20,20,10.0], [20,20,10.1], [-1, -1, 5], [20, 20, -0.1], [20, 20, 0], [20,20, 0.1]])
        ts = sd.time.data[1]
        assert sd.default_bottom_boundary_condition == 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        expected_idx = np.ma.array(np.array([0, -1000, -1000, -1000, 10, 9, 9]), mask = [False, True, True, True, False, False, False])
        expected_alpha = np.ma.array(np.array([0.1, -1000, -1000, -1000, 0, 1, 0.9]), mask = [False, True, True, True, False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))
        
        sd.default_bottom_boundary_condition == 'extrapolate'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        expected_idx = np.ma.array(np.array([0, -1000,-1000, -1000, 10, 9, 9]), mask = [False, False, True, True, False, False, False])
        expected_alpha = np.ma.array(np.array([0.1, -1000, -1000, -1000, 0, 1, 0.9]), mask = [False, False, True, True, False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))

    def test_interpolation_alphas_reversed(self, get_roms_depth):
        '''
        Same as the previous test, but with the depth system reversed as it may be in FVCOM
        
        # query center mound (20,20) at 3 depths. 1st point is 0.1m off the seafloor and 10% of
        # the distance to the next s-layer (meaning, expected alpha returned is 10%)
        # 1st point is 0.1 off the seafloor this is an index of 9 and an alpha of 0.9
        # 2nd point is directly on the seafloor (should register as 'in bounds' index 10 and 0 alpha)
        # 3rd point is 0.1m underground, and should indicate with masked value.
        # 4th point is off grid, masked index and alpha expected.
        # 5th point is 0.1m ABOVE surface. Expected index of 0 and alpha of 0
        # 6th point is directly at surface. Expected index of 0 and alpha of 0
        #7th point is 0.1m below surface. Expected index of 1 and alpha of 0.1
        '''
        sd = get_roms_depth
        sd.Cs_r = sd.Cs_r[::-1]
        sd.Cs_w = sd.Cs_w[::-1]
        sd.hc = sd.hc[::-1]
        sd.s_rho = sd.s_rho[::-1]
        sd.s_w = sd.s_w[::-1]
        
        points = np.array([[20,20, 9.9],[20,20,10.0], [20,20,10.1], [-1, -1, 5], [20, 20, -0.1], [20, 20, 0], [20,20, 0.1]])
        ts = sd.time.data[1]
        assert sd.default_bottom_boundary_condition == 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        expected_idx = np.ma.array(np.array([9, -1000, -1000, -1000, -1, 0, 0]), mask = [False, True, True, True, False, False, False])
        expected_alpha = np.ma.array(np.array([0.9, -1000, -1000, -1000, 1, 0, 0.1]), mask = [False, True, True, True, False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))
        
        sd.default_bottom_boundary_condition == 'extrapolate'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        expected_idx = np.ma.array(np.array([0,0,-1]), mask = [False, False, False])
        expected_alpha = np.ma.array(np.array([0.1, 0, 1]), mask = [False, False, False])
        
        expected_idx = np.ma.array(np.array([9, 9, -1000, -1000, -1, 0, 0]), mask = [False, False, True, True, False, False, False])
        expected_alpha = np.ma.array(np.array([0.9, 0, -1000, -1000, 1, 0, 0.1]), mask = [False, False, True, True, False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))

    def test_interpolation_alphas_surface(self, get_roms_depth):
        sd = get_roms_depth
        points = np.array([[20,20, -0.1],[20,20,0], [20,20,0.1]])
        ts = sd.time.data[1]
        assert sd.default_surface_boundary_condition == 'extrapolate'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface.
        expected_idx = np.ma.array(np.array([10,9,9]), mask = [False, False, False])
        expected_alpha = np.ma.array(np.array([0, 1, 0.9]), mask = [False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))
        
        sd.default_surface_boundary_condition = 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface.
        expected_idx = np.ma.array(np.array([-1,9,9]), mask = [True, False, False])
        expected_alpha = np.ma.array(np.array([0, 1, 0.9]), mask = [True, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))

        sd.default_surface_boundary_condition = 'extrapolate'
        # switch to timestep with -0.5m zeta
        ts = sd.time.data[0]
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        expected_idx = np.ma.array(np.array([10,10,10]), mask = [False, False, False])
        expected_alpha = np.ma.array(np.array([0, 0, 0]), mask = [False, False, False])
        assert np.all(idx == expected_idx)
        assert np.all(np.isclose(alphas, expected_alpha))

        sd.default_surface_boundary_condition = 'mask'
        idx, alphas = sd.interpolation_alphas(points, ts, [sd.num_levels,])
        # only the element 0.1m deep should register with an index and alpha since
        # it is the only element below surface.
        expected_idx = np.ma.array(np.array([-1,-1,-1]), mask = [True, True, True])
        expected_alpha = np.ma.array(np.array([0, 0.9, 0]), mask = [True, True, True])
        assert np.all(idx.mask == expected_idx.mask)
        assert np.all(alphas.mask == expected_alpha.mask)
        
class Test_FVCOM_Depth(object):
    def test_construction(self, get_fvcom_depth):
        assert get_fvcom_depth is not None
        
    def test_get_transect(self, get_fvcom_depth):
        dp = get_fvcom_depth
        tris = dp.grid.nodes.take(dp.grid.faces, axis=0)
        centroids = np.mean(tris, axis=1)
        transects = dp.get_transect(centroids, dp.zeta.time.min_time)
        
        #because we use the centroids, the values should the average of the 3 nodes
        #Not intending to test interpolation here there's a separate test for that
        
        expected_transects = np.mean((dp.siglev * dp.bathymetry.data[:]).take(dp.grid.faces, axis=1), axis=2)
        expected_transects = expected_transects.T * -1
        assert np.all(np.isclose(transects, expected_transects))

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
        ld1, ld2 = get_l_depth
        points = np.array(([0, 0, 0], [1, 1, 0], [4, 9, 0]))
        idxs, alphas = ld1.interpolation_alphas(points)
        expected_idxs = np.array([0,0,0])
        expected_alphas = np.array([0,0,0])

        assert np.all(idxs == expected_idxs)
        assert np.all(alphas == expected_alphas)
        
        idxs, alphas = ld2.interpolation_alphas(points)
        expected_idxs = np.array([5,5,5])
        expected_alphas = np.array([0,0,0])

    def test_interpolation_alphas_1_surface(self, get_l_depth):
        ld1, ld2 = get_l_depth
        points = np.array(([0, 0, 0],
                           [1, 1, 0.4],
                           [0, 0, 1],
                          ))
        idxs, alphas = ld1.interpolation_alphas(points)

        assert np.all(idxs == np.array([0, 0, 1]))
        assert np.all(np.isclose(alphas, np.array([0, 0.4, 0])))
        
        idxs, alphas = ld2.interpolation_alphas(points)
        
        assert np.all(idxs == np.array([5, 4, 4]))
        assert np.all(np.isclose(alphas, np.array([0, 0.6, 0])))

    def test_interpolation_alphas_above_surface(self, get_l_depth):
        ld1, ld2 = get_l_depth
        points = np.array(([0, 0, -1], [0, 0, 4.5])) #one above, one below depth interval
        idxs, alphas = ld1.interpolation_alphas(points)

        assert np.all(idxs == np.array([-1, 3]))
        assert np.all(alphas == np.array([1, 0.25]))
        
        idxs, alphas = ld2.interpolation_alphas(points)

        assert np.all(idxs == np.array([5, 1]))
        assert np.all(alphas == np.array([0, 0.75]))
        

    def test_interpolation_alphas_below_grid(self, get_l_depth):
        ld1, ld2 = get_l_depth
        points = np.array(([0, 0, 20], [0, 0, 4.5]))
        idxs, alphas = ld1.interpolation_alphas(points)

        assert np.all(idxs == np.array([5, 3]))
        assert np.all(alphas == np.array([0, 0.25]))
        
        idxs, alphas = ld2.interpolation_alphas(points)
        assert np.all(idxs == np.array([-1, 1]))
        assert np.all(alphas == np.array([1, 0.75]))

    def test_interpolation_alphas_full(self, get_l_depth):
        ld1, ld2 = get_l_depth
        points = np.array(([1, 2, 0], [3, 4, 5.5], [3, 4, 15.5], [3, 4, -10], [3, 4, 10]))
        idxs, alphas = ld1.interpolation_alphas(points)

        assert np.all(idxs == np.array([0, 3, 5, -1, 5]))
        assert np.all(alphas == np.array([0, 0.75, 0, 1, 0]))

    @pytest.mark.parametrize("index", (0, 5, 10, 20))
    def test_vertical_interpolation_within_grid(self, index, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], depth.depth_levels[index]), )
        rtv = ds.variables['u'].at(points=points,time=time.data[0])[0]

        assert rtv == ds.variables['u'].data[0,index,1,1]

    def test_vertical_interpolation_onsurface(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], 0.0), )
        rtv = ds.variables['u'].at(points=points,time=time.data[0])[0]

        assert rtv == ds.variables['u'].data[0,0,1,1]

    def test_vertical_interpolation_belowgrid(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], depth.depth_levels[-1]+100.), )
        rtv = ds.variables['u'].at(points=points,time=time.data[0])[0]

        assert np.isnan(rtv)

    def test_vertical_interpolation_abovesurface(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], -10.), )
        rtv = ds.variables['u'].at(points=points,time=time.data[0])[0]

        assert rtv == ds.variables['u'].data[0,0,1,1]

    def test_vertical_interpolation_full(self, get_database_nc):
        time, depth, ds = get_database_nc
        points = ((ds.grid.node_lon[1], ds.grid.node_lat[1], -10.), (ds.grid.node_lon[1], ds.grid.node_lat[1], 0.), (ds.grid.node_lon[1], ds.grid.node_lat[1], depth.depth_levels[3]))
        rtv = ds.variables['u'].at(points=points,time=time.data[0])

        assert rtv[0] == ds.variables['u'].data[0,0,1,1]
        assert rtv[1] == ds.variables['u'].data[0,0,1,1]
        assert rtv[2] == ds.variables['u'].data[0,3,1,1]
