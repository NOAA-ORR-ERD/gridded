import netCDF4 as nc
import numpy as np
import shutil


def fvcom_eleven_points_with_depth():
    '''Use UGRIDv0.9_eleven_points.nc as a starting point. Add sigma depth coordinates and provide better values for the bathymetry'''
    fn = './UGRIDv0.9_eleven_points.nc'
    shutil.copy(fn, './UGRIDv0.9_eleven_points_with_depth.nc')
    
    df = nc.Dataset('./UGRIDv0.9_eleven_points_with_depth.nc', mode = 'a')
    df.createDimension('siglev', 5)
    df.createDimension('siglay', 4)
    df.createDimension('t', 1)
    
    df.createVariable('time', 'f8', ('t',))
    df.createVariable('siglay', 'f8', ('siglay', 'nMesh2_node'))
    df.createVariable('siglev', 'f8', ('siglev', 'nMesh2_node'))
    df.createVariable('siglay_center', 'f8', ('siglay', 'nMesh2_face'))
    df.createVariable('siglev_center', 'f8', ('siglev', 'nMesh2_face'))
    df.createVariable('h', 'f8', ('nMesh2_node',))
    df.createVariable('h_center', 'f8', ('nMesh2_face',))
    df.createVariable('zeta', 'f8', ('t', 'nMesh2_node',))
    
    df['time'].units = 'days since 1970-01-01 00:00:00'
    df['time'].calendar = 'gregorian'
    df['time'].long_name = 'time'
    df['time'].time_zone = 'UTC'
    df['time'].format = 'defined reference date'
    df['time'][0] = 19636.541
    
    siglay_attrs = {'long_name': 'Sigma Layers',
                    'standard_name': 'ocean_sigma_coordinate',
                    'positive': 'up',
                    'valid_min': -1,
                    'valid_max': 0,
                    'formula_terms': 'sigma: siglay eta: zeta depth: h'}
    df['siglay'].setncatts(siglay_attrs)
    df['siglay_center'].setncatts(siglay_attrs)
    df['siglay_center'].formula_terms = 'sigma: siglay_center eta: zeta_center depth: h_center'
    
    siglev_attrs = {'long_name': 'Sigma Levels',
                    'standard_name': 'ocean_sigma_coordinate',
                    'positive': 'up',
                    'valid_min': -1,
                    'valid_max': 0,
                    'formula_terms': 'sigma: siglay eta: zeta depth: h'}    
    df['siglev'].setncatts(siglev_attrs)
    df['siglev_center'].setncatts(siglev_attrs)
    df['siglev_center'].formula_terms = 'sigma: siglev_center eta: zeta_center depth: h_center'
    bathy_attrs = {
        'long_name': 'Bathymetry',
        'standard_name': 'sea_floor_depth_below_geoid',
        'units': 'm',
        'positive': 'down',
        'grid': 'Bathymetry_Mesh',
        'coordinates': 'Mesh2_node_y Mesh2_node_x',
        'type': 'data'
    }
    df['h'].setncatts(bathy_attrs)
    df['h_center'].setncatts(bathy_attrs)
    df['h_center'].grid = 'Bathymetry_Mesh2'
    df['h_center'].coordinates = 'Mesh2_face_y Mesh2_face_x'
    df['h_center'].grid_location = 'center'
    df['zeta'][:] = np.zeros(df['zeta'].shape)
    
    zeta_attrs = {
        'long_name': 'Water Surface Elevation',
        'units': 'meters',
        'positive': 'up',
        'standard_name': 'sea_surface_height_above_geoid',
        'grid': 'Bathymetry_Mesh',
        'coordinates': 'time Mesh2_node_y Mesh2_node_x',
        'type': 'data',
        'location': 'node'
    }
    df['zeta'].setncatts(zeta_attrs)
    
    df['h'][:] = np.linspace(0, 10, df['h'].shape[0]).reshape(df['h'].shape)
    df['h_center'][:] = np.ones(df['h_center'].shape) * 10
    df['Mesh2_depth'][:] = df['h'][:]
    df['siglev'][:] = np.linspace(0, -1, 5).T[:,np.newaxis]
    df['siglay'][:] = np.linspace(0, -1, 9)[1::2].T[:, np.newaxis]
    
    
    df.close()
    

def make_all():
    """
    make all the sample files
    """

    fvcom_eleven_points_with_depth()
    # semi_circular_single_grid()


if __name__ == "__main__":
    make_all()
    # fname = semi_circular_single_grid()
    # print("making:", fname)