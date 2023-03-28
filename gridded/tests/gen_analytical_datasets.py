
import numpy as np
import netCDF4 as nc4
import os
from datetime import datetime, timedelta



from gridded import Variable, Grid
from gridded.grids import Grid_S

def gen_vortex_3D(filename=None):
    x, y = np.mgrid[-30:30:61j, -30:30:61j]
    y = np.ascontiguousarray(y.T)
    x = np.ascontiguousarray(x.T)
    x_size = 61
    y_size = 61
    g = Grid_S(node_lon=x,
               node_lat=y)
    g.build_celltree()
    lin_nodes = g._cell_tree[1]
    lin_faces = np.array([np.array([([lx, lx + x_size + 1, lx + 1], [lx, lx + x_size, lx + x_size + 1]) for lx in range(0, x_size - 1, 1)]) + ly * x_size for ly in range(0, y_size - 1)])
    lin_faces = lin_faces.reshape(-1, 3)
    # y += np.sin(x) / 1
    # x += np.sin(x) / 5

    t0 = datetime(2001, 1, 1, 0, 0)
    tarr = [t0 + timedelta(hours=i) for i in range(0, 11)]
    angs = -np.arctan2(y, x)
    mag = np.sqrt(x ** 2 + y ** 2)
    vx = np.cos(angs) * mag
    vy = np.sin(angs) * mag
    vx = vx[np.newaxis, :] * 20
    vy = vy[np.newaxis, :] * 20
    vw = -0.001

    d_scale = [1, 0.5, 0, -0.5, -1]
    t_scale = np.linspace(0, 1, 11)

    tvx = np.array([vx * t for t in t_scale]).squeeze()
    tvy = np.array([vy * t for t in t_scale]).squeeze()

    dvx = np.array([vx * s for s in d_scale]).squeeze()
    dvy = np.array([vy * s for s in d_scale]).squeeze()

    tdvx = np.array([dvx * t for t in t_scale]).squeeze()
    tdvy = np.array([dvy * t for t in t_scale]).squeeze()

    lin_vx = vx.reshape(-1)
    lin_vy = vy.reshape(-1)

    lin_tvx = np.array([lin_vx * t for t in t_scale])
    lin_tvy = np.array([lin_vy * t for t in t_scale])

    lin_dvx = np.array([lin_vx * s for s in d_scale])
    lin_dvy = np.array([lin_vy * s for s in d_scale])

    lin_tdvx = np.array([lin_dvx * t for t in t_scale])
    lin_tdvy = np.array([lin_dvy * t for t in t_scale])

    ds = None
    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)

        ds.createDimension('y', y.shape[0])
        ds.createDimension('x', x.shape[1])
        ds.createDimension('time', len(tarr))
        ds.createDimension('depth', len(d_scale))
        ds.createVariable('x', 'f8', dimensions=('x', 'y'))
        ds['x'][:] = x
        ds.createVariable('y', 'f8', dimensions=('x', 'y'))
        ds['y'][:] = y
        ds.createVariable('time', 'f8', dimensions=('time'))
        ds['time'][:] = nc4.date2num(tarr, 'hours since {0}'.format(t0))
        ds['time'].setncattr('units', 'hours since {0}'.format(t0))
        ds.createVariable('vx', 'f8', dimensions=('x', 'y'))
        ds.createVariable('vy', 'f8', dimensions=('x', 'y'))
        ds['vx'][:] = vx
        ds['vy'][:] = vy
        ds.createVariable('tvx', 'f8', dimensions=('time', 'x', 'y'))
        ds.createVariable('tvy', 'f8', dimensions=('time', 'x', 'y'))
        ds['tvx'][:] = tvx
        ds['tvy'][:] = tvy
        ds.createVariable('dvx', 'f8', dimensions=('depth', 'x', 'y'))
        ds.createVariable('dvy', 'f8', dimensions=('depth', 'x', 'y'))
        ds['dvx'][:] = dvx
        ds['dvy'][:] = dvy
        ds.createVariable('tdvx', 'f8', dimensions=('time', 'depth', 'x', 'y'))
        ds.createVariable('tdvy', 'f8', dimensions=('time', 'depth', 'x', 'y'))
        ds['tdvx'][:] = tdvx
        ds['tdvy'][:] = tdvy
        for v in ds.variables:
            if 'v' in v:
                ds[v].units = 'm/s'

        ds.createDimension('nv', lin_nodes.shape[0])
        ds.createDimension('nele', lin_faces.shape[0])
        ds.createDimension('two', 2)
        ds.createDimension('three', 3)
        ds.createVariable('nodes', 'f8', dimensions=('nv', 'two'))
        ds.createVariable('faces', 'f8', dimensions=('nele', 'three'))
        ds.createVariable('lin_vx', 'f8', dimensions=('nv'))
        ds.createVariable('lin_vy', 'f8', dimensions=('nv'))
        ds.createVariable('lin_tvx', 'f8', dimensions=('time', 'nv'))
        ds.createVariable('lin_tvy', 'f8', dimensions=('time', 'nv'))
        ds.createVariable('lin_dvx', 'f8', dimensions=('depth', 'nv'))
        ds.createVariable('lin_dvy', 'f8', dimensions=('depth', 'nv'))
        ds.createVariable('lin_tdvx', 'f8', dimensions=('time', 'depth', 'nv'))
        ds.createVariable('lin_tdvy', 'f8', dimensions=('time', 'depth', 'nv'))
        for k, v in {'nodes': lin_nodes,
                     'faces': lin_faces,
                     'lin_vx': lin_vx,
                     'lin_vy': lin_vy,
                     'lin_tvx': lin_tvx,
                     'lin_tvy': lin_tvy,
                     'lin_dvx': lin_dvx,
                     'lin_dvy': lin_dvy,
                     'lin_tdvx': lin_tdvx,
                     'lin_tdvy': lin_tdvy
                     }.items():
            ds[k][:] = v
            if 'lin' in k:
                ds[k].units = 'm/s'
        Grid._get_grid_type(ds, grid_topology={'node_lon': 'x', 'node_lat': 'y'})
        Grid._get_grid_type(ds)
        ds.setncattr('grid_type', 'sgrid')
    if ds is not None:
        # Need to test the dataset...
        sgt = {'node_lon': 'x', 'node_lat': 'y'}
        sg = Grid.from_netCDF(dataset=ds, grid_topology=sgt, grid_type='sgrid')

        ugt = {'nodes': 'nodes', 'faces': 'faces'}
#         ug = PyGrid_U(nodes=ds['nodes'][:], faces=ds['faces'][:])

        ds.close()
    return {'sgrid': (x, y),
            'sgrid_vel': (dvx, dvy),
            'sgrid_depth_vel': (tdvx, tdvy),
            'ugrid': (lin_nodes, lin_faces),
            'ugrid_depth_vel': (lin_tdvx, lin_tdvy)}


def gen_sinusoid(filename=None):
    y, x = np.mgrid[-1:1:5j, 0:(6 * np.pi):25j]
    y = y + np.sin(x / 2)
    Z = np.zeros_like(x)
    # abs(np.sin(x / 2)) +
    vx = np.ones_like(x)
    vy = np.cos(x / 2) / 2
    vz = np.zeros_like(x)
#     ax.quiver(x, y, vx, vy, color='darkblue', pivot='tail', angles='xy', scale=1.5, scale_units='xy', width=0.0025)
#     ax.plot(x[2], y[2])
    mask_rho = np.zeros_like(x, dtype=bool)
    mask_rho[0,:] = True
    mask_rho[-1,:] = True
    rho = {'r_grid': (x, y, Z),
           'r_vel': (vx, vy, vz)}

    yc, xc = np.mgrid[-0.75:0.75: 4j, 0.377:18.493:24j]
    yc = yc + np.sin(xc / 2)
    zc = np.zeros_like(xc) + 0.025
    vxc = np.ones_like(xc)
    vyc = np.cos(xc / 2) / 2
    vzc = np.zeros_like(xc)
    mask_psi = np.zeros_like(xc, dtype=bool)
    mask_psi[0,:] = True
    mask_psi[-1,:] = True
    psi = {'p_grid': (xc, yc, zc),
           'p_vel': (vxc, vyc, vzc)}

    yu, xu = np.mgrid[-1:1:5j, 0.377:18.493:24j]
    yu = yu + np.sin(xu / 2)
    zu = np.zeros_like(xu) + 0.05
    vxu = np.ones_like(xu) * 2
    vyu = np.zeros_like(xu)
    vzu = np.zeros_like(xu)
    mask_u = np.zeros_like(xu, dtype=bool)
    mask_u[0,:] = True
    mask_u[-1,:] = True
    u = {'u_grid': (xu, yu, zu),
         'u_vel': (vzu, vxu, vzu)}

    yv, xv = np.mgrid[-0.75:0.75: 4j, 0:18.87:25j]
    yv = yv + np.sin(xv / 2)
    zv = np.zeros_like(xv) + 0.075
    vxv = np.zeros_like(xv)
    vyv = np.cos(xv / 2) / 2
    vzv = np.zeros_like(xv)
    mask_v = np.zeros_like(xv, dtype=bool)
    mask_v[0,:] = True
    mask_v[-1,:] = True
    v = {'v_grid': (xv, yv, zv),
         'v_vel': (vyv, vzv, vzv)}

    angle = np.cos(x / 2) / 2

    ds = None
    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)
        for k, v in {'eta_psi': 24,
                     'xi_psi': 4,
                     'eta_rho': 25,
                     'xi_rho':5}.items():
            ds.createDimension(k, v)
        for k, v in {'lon_rho': ('xi_rho', 'eta_rho', x),
                     'lat_rho': ('xi_rho', 'eta_rho', y),
                     'lon_psi': ('xi_psi', 'eta_psi', xc),
                     'lat_psi': ('xi_psi', 'eta_psi', yc),
                     'lat_u': ('xi_rho', 'eta_psi', xu),
                     'lon_u': ('xi_rho', 'eta_psi', yu),
                     'lat_v': ('xi_psi', 'eta_rho', xv),
                     'lon_v': ('xi_psi', 'eta_rho', yv),
                     }.items():
            ds.createVariable(k, 'f8', dimensions=v[0:2])
            ds[k][:] = v[2]
        for k, v in {'u_rho': ('xi_rho', 'eta_rho', vx),
                     'v_rho': ('xi_rho', 'eta_rho', vy),
                     'u_psi': ('xi_psi', 'eta_psi', vxc),
                     'v_psi': ('xi_psi', 'eta_psi', vyc),
                     'u': ('xi_rho', 'eta_psi', vxu),
                     'v': ('xi_psi', 'eta_rho', vyv)}.items():
            ds.createVariable(k, 'f8', dimensions=v[0:2])
            ds[k][:] = v[2]
            ds[k].units = 'm/s'

        for k, v in {'mask_rho': ('xi_rho', 'eta_rho', mask_rho),
                     'mask_psi': ('xi_psi', 'eta_psi', mask_psi),
                     'mask_u':('xi_rho', 'eta_psi', mask_u),
                     'mask_v':('xi_psi', 'eta_rho', mask_v)}.items():
            ds.createVariable(k, 'b', dimensions=v[0:2])
            ds[k][:] = v[2]
        ds.grid_type = 'sgrid'
        ds.createVariable('angle', 'f8', dimensions=('xi_rho', 'eta_rho'))
        ds['angle'][:] = angle
    if ds is not None:
        # Need to test the dataset...
        sg = Grid.from_netCDF(dataset=ds)
        ds.close()



def gen_ring(filename=None):
    import matplotlib.tri as tri
    import math

    n_angles = 36
    n_radii = 6
    min_radius = 0.25
    radii = np.linspace(min_radius, 1, n_radii)

    angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += math.pi / n_angles
    print(angles.shape)

    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    z = (np.cos(radii) * np.cos(angles * 3.0)).flatten()

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)
    # Mask off unwanted triangles.
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid * xmid + ymid * ymid < min_radius * min_radius, 1, 0)
    triang.set_mask(mask)
    faces = triang.get_masked_triangles()

    vx = (np.ones_like(radii) * np.cos(angles + np.pi)).flatten()
    vy = (np.ones_like(radii) * np.sin(angles)).flatten()

    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)
        ds.createDimension('nv', x.shape[0])
        ds.createDimension('nele', faces.shape[0])
        ds.createDimension('three', 3)
        for k, v in {'node_lon': ('nv', x),
                     'node_lat': ('nv', y),
                     'faces': ('nele', 'three', faces),
                     'u': ('nv', vx),
                     'v': ('nv', vy)}.items():
            ds.createVariable(k, 'f8', dimensions=v[0:-1])
            ds[k][:] = v[-1]
            ds[k].units = 'm/s'

        ds.close()


def gen_arakawa_c_test_grid(filename=None): #added 9/2019 by Jay Hennen
    #Initialization for a 4 cell wide east-west channel that has two one cell wide 'inlets'
    #on the north side and two one cell wide outlets. The north and south side have one outlet each. Also included are
    #various boundary conditions set at the corners of the inlets and outlets. The u/v velocities on this grid
    #have been manually tuned to conserve mass across cell boundaries. The mask information and naming style are 
    #similar to what NOAA GOODS ROMS output looks like circa 9/2019

    #TODO Add depth dimensions (needs s_rho, s_w, h, zeta, Cs_r, Cs_w, and w)

    lat_rho, lon_rho = np.mgrid[30.5:41.5:12j, 0.5:11.5:12j]

    lat_psi = (lat_rho[0:-1, 0:-1] + lat_rho[1:, 1:]) /2
    lon_psi = (lon_rho[0:-1, 0:-1] + lon_rho[1:, 1:]) /2

    lat_v = (lat_rho[0:-1,:] + lat_rho[1:,:]) /2
    lon_v = (lon_rho[0:-1,:] + lon_rho[1:,:]) /2

    lat_u = (lat_rho[:, 0:-1] + lat_rho[:, 1:]) /2
    lon_u = (lon_rho[:, 0:-1] + lon_rho[:, 1:]) /2

    mask_rho = np.zeros_like(lat_rho, dtype=np.uint8) #0 = land, 1 = water
    mask_rho[3:7,:] = 1
    mask_rho[6:,2] = 1
    mask_rho[6:,5] = 1
    mask_rho[7:10,9] = 1
    mask_rho[9,7:9] = 1
    mask_rho[1:3,9] = 1
    mask_rho[1,7:9] = 1

    mask_psi = np.zeros_like(lat_psi, dtype=np.uint8) # 0 = land, 1 = water (free-slip?), 2 = no-slip

    #This mask is what will be replacing the current gridded.utilities.gen_mask
    #and be used as the cell tree mask ONLY
    #TODO: Find a more efficient implementation, if reasonable!
    mask_p_from_r = mask_psi.copy()
    cellmask = mask_rho[1:-1,1:-1]
    for i, r in enumerate(cellmask):
        mask_p_from_r[i,1::] += r
        mask_p_from_r[i,:-1:] += r
        mask_p_from_r[i+1,1::] += r
        mask_p_from_r[i+1,:-1:] += r

    mask_psi[2,:] = 2
    mask_psi[6,:] = 2
    mask_psi[6,4:6] = 1
    mask_psi[3:6,:] = 1
    mask_psi[6:,1:3] = 2
    mask_psi[7:,4:6] = 2
    mask_psi[7:10,8:10] = 2
    mask_psi[8:10,7] = 2
    mask_psi[0:2,7:10] = 2
    mask_psi[0:2,6] = 1
    mask_psi[1,8] = 1
    mask_psi[[6,8],8] = 1

    #mask_psi[3,4] = 2 #No-slip in middle of channel!

    # 0 = land, 1 = water
    mask_u = np.zeros_like(lat_u, dtype=np.uint8)
    mask_u[3:7,:] = 1
    mask_u[9,7:9] = 1
    mask_u[1,6:9] = 1

    # 0 = land, 1 = water
    mask_v = np.zeros_like(lat_v, dtype=np.uint8)
    mask_v[3:6,:] = 1
    mask_v[6:,2] = 1
    mask_v[6:,5] = 1
    mask_v[6:9,9] = 1
    mask_v[1:3,9] = 1

    #Below contains demo u/v initialization

    u = np.ma.MaskedArray(np.zeros_like(lon_u), mask = ~(mask_u == 1))
    u.data[3:7,0:2] = 1.0
    u.data[3:7,2] = [1.4375, 1.375,1.125, 1.0625][::-1]
    u.data[3:7,3:5] = [1.25,1.25]
    u.data[3:7,5] = [1.7, 1.625, 1.375, 1.325][::-1]
    u.data[3:7,6:9] = [1.5, 1.5, 1.5]
    u.data[3:7,9:11] = 1.0
    u.data[9,7:9] = -1.0 #Outlet 1
    u.data[1,6:9] = -1.0 #Outlet 2

    v = np.ma.MaskedArray(np.zeros_like(lon_v), mask = ~(mask_v == 1))
    v.data[3:6,:] = 0 #Main Channel
    v.data[6:11,2] = -1 #Inlet #1
    v.data[5,2] = -0.5625
    v.data[4,2] = -0.1875
    v.data[3,2] = -0.0625
    v.data[5,3] = -0.1875
    v.data[4,3] = -0.3125
    v.data[3,3] = -0.0625
    v.data[6:11,5] = -1 #Inlet #2
    v.data[5,5] = -0.5625 
    v.data[4,5] = -0.1875
    v.data[3,5] = -0.0625
    v.data[5,6] = -0.1875
    v.data[4,6] = -0.3125
    v.data[3,6] = -0.0625
    v.data[5,9] = 0.5 #Outlet #1
    v.data[3,9] = -0.5 #Outlet #2
    v.data[6:9,9] = 1
    v.data[1:3,9] = -1
    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)
        for k, val in {'xi_rho': 12,
                     'xi_u': 11,
                     'xi_v': 12,
                     'xi_psi': 11,
                     'eta_rho': 12,
                     'eta_u': 12,
                     'eta_v': 11,
                     'eta_psi': 11,
                     'ocean_time': None}.items():
            ds.createDimension(k, val)
        for k, val in {'lon_rho': {'dimensions':('eta_rho', 'xi_rho'),
                                 'data':lon_rho,
                                 'long_name':'longitude of RHO-points',
                                 'units':'degrees_east',
                                 'standard_name': 'longitude',
                                 'field':'lon_rho, scalar'},
                     'lat_rho': {'dimensions':('eta_rho', 'xi_rho'),
                                 'data':lat_rho,
                                 'long_name':'latitude of RHO-points',
                                 'units':'degrees_north',
                                 'standard_name': 'latitude',
                                 'field':'lat_rho, scalar'},
                     'lon_psi': {'dimensions':('eta_psi', 'xi_psi'),
                                 'data':lon_psi,
                                 'long_name':'longitude of PSI-points',
                                 'units':'degrees_east',
                                 'standard_name': 'longitude',
                                 'field':'lon_psi, scalar'},
                     'lat_psi': {'dimensions':('eta_psi', 'xi_psi'),
                                'data':lat_psi,
                                'long_name':'latitude of PSI-points',
                                'units':'degrees_north',
                                'standard_name': 'latitude',
                                'field':'lat_psi, scalar'},
                     'lon_u': {'dimensions':('eta_psi', 'xi_rho'),
                                'data':lon_u,
                                'long_name':'longitude of U-points',
                                'units':'degrees_east',
                                'standard_name': 'longitude',
                                'field':'lon_u, scalar'},
                     'lat_u': {'dimensions':('eta_psi', 'xi_rho'),
                                'data':lat_u,
                                'long_name':'latitude of U-points',
                                'units':'degrees_north',
                                'standard_name': 'latitude',
                                'field':'lat_u, scalar'},
                     'lon_v': {'dimensions':('eta_rho', 'xi_psi'),
                                'data':lon_v,
                                'long_name':'longitude of V-points',
                                'units':'degrees_east',
                                'standard_name': 'longitude',
                                'field':'lon_v, scalar'},
                     'lat_v': {'dimensions':('eta_rho', 'xi_psi'),
                                'data':lat_v,
                                'long_name':'latitude of V-points',
                                'units':'degrees_north',
                                'standard_name': 'latitude',
                                'field':'lat_v, scalar'},
                     'u': {'dimensions':('ocean_time', 'eta_u', 'xi_u'),
                            'data': u[np.newaxis],
                            'long_name': 'u-momentum component',
                            'units': 'meter second-1',
                            'time': 'ocean_time',
                            'grid': 'grid',
                            'location': 'edge1',
                            'coordinates': 'lon_u lat_u ocean_time',
                            'field': 'u-velocity, scalar, series'},
                     'v': {'dimensions':('ocean_time', 'eta_v', 'xi_v'),
                            'data': v[np.newaxis],
                            'long_name': 'v-momentum component',
                            'units': 'meter second-1',
                            'time': 'ocean_time',
                            'grid': 'grid',
                            'location': 'edge2',
                            'coordinates': 'lon_v lat_v ocean_time',
                            'field': 'v-velocity, scalar, series'},
                     'mask_rho': {'dimensions':('eta_rho', 'xi_rho'),
                                  'data': mask_rho,
                                  'long_name': 'mask on RHO-points',
                                  'flag_values': np.array([0, 1]),
                                  'flag_meanings': 'land water',
                                  'grid': 'grid',
                                  'location': 'face',
                                  'coordinates': 'lon_rho lat_rho'},
                     'mask_psi': {'dimensions':('eta_psi', 'xi_psi'),
                                  'data': mask_psi, #In real data, there is also '2', which denotes 'no-slip'
                                  'long_name': 'mask on psi-points',
                                  'flag_values': np.array([0, 1]), #Is this the real life? Is this just fantasy?
                                  'flag_meanings': 'land water', #Caught in a landslide, no escape from reality
                                  'grid': 'grid', 
                                  'location': 'node', 
                                  'coordinates': 'lon_psi lat_psi'},
                     'mask_u': {'dimensions':('eta_u', 'xi_u'),
                                  'data': mask_u,
                                  'long_name': 'mask on U-points',
                                  'flag_values': np.array([0, 1]),
                                  'flag_meanings': 'land water',
                                  'grid': 'grid',
                                  'location': 'edge1',
                                  'coordinates': 'lon_u lat_u'},
                     'mask_v': {'dimensions':('eta_v', 'xi_v'),
                                  'data': mask_v,
                                  'long_name': 'mask on V-points',
                                  'flag_values': np.array([0, 1]),
                                  'flag_meanings': 'land water',
                                  'grid': 'grid',
                                  'location': 'edge2',
                                  'coordinates': 'lon_v lat_v'},
                     'ocean_time': {'dimensions':('ocean_time'),
                                    'data': np.array([65520000.,]),
                                    'long_name': 'time since initialization',
                                    'units': 'seconds since 2016-01-01 00:00:00',
                                    'calendar': 'gregorian_proleptic',
                                    'field': 'time, scalar, series'},
                     'grid': {'dimensions':(),
                              'data':1,
                              'cf_role': 'grid_topology',
                              'topology_dimension': 2,
                              'node_dimensions': 'xi_psi eta_psi',
                              'face_dimensions': 'xi_rho: xi_psi (padding: both) eta_rho: eta_psi (padding: both)',
                              'edge1_dimensions': 'xi_u: xi_psi eta_u: eta_psi (padding: both)',
                              'edge2_dimensions': 'xi_v: xi_psi (padding: both) eta_v: eta_psi',
                              'node_coordinates': 'lon_psi lat_psi',
                              'face_coordinates': 'lon_rho lat_rho',
                              'edge1_coordinates': 'lon_u lat_u',
                              'edge2_coordinates': 'lon_v lat_v',
                              'vertical_dimensions': 's_rho: s_w (padding: none)',
                              }
                     }.items():
            var = ds.createVariable(k, 'f8', dimensions=val.pop('dimensions'))
            data = val.pop('data')
            var[:] = data
            var.setncatts(val)
    if ds is not None:
        ds.close()

def gen_all(path=None):
    filenames = ['arakawa_c_test_grid.nc', 'staggered_sine_channel.nc', '3D_circular.nc', 'tri_ring.nc']
    if path is not None:
        filenames = [os.path.join(path, fn) for fn in filenames]
    for fn, func in zip(filenames, (gen_arakawa_c_test_grid, gen_sinusoid, gen_vortex_3D, gen_ring)):
        func(fn)

if __name__ == '__main__':
    gen_arakawa_c_test_grid('arakawa_c_test_grid.nc')
    gen_sinusoid('staggered_sine_channel.nc')
    gen_vortex_3D('3D_circular.nc')
    gen_ring('tri_ring.nc')
