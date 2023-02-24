"""
Created on Apr 7, 2015

@author: ayan

"""


from gridded.pysgrid.read_netcdf import NetCDFDataset, find_grid_topology_var
from .write_nc_test_files import roms_sgrid, wrf_sgrid


"""
Test NetCDF Dataset With Nodes.

"""


def test_finding_node_variables(roms_sgrid):
    nc_ds = NetCDFDataset(roms_sgrid)
    result = nc_ds.find_node_coordinates('xi_psi eta_psi')
    expected = ('lon_psi', 'lat_psi')
    assert result == expected


def test_find_face_coordinates_by_location(roms_sgrid):
    nc_ds = NetCDFDataset(roms_sgrid)
    result = nc_ds.find_coordinates_by_location('face', 2)
    expected = ('lon_rho', 'lat_rho')
    assert result == expected


def test_find_edge_coordinates_by_location(roms_sgrid):
    nc_ds = NetCDFDataset(roms_sgrid)
    result = nc_ds.find_coordinates_by_location('edge1', 2)
    expected = ('lon_u', 'lat_u')
    assert result == expected


def test_find_grid_topology(roms_sgrid):
    result = find_grid_topology_var(roms_sgrid)
    expected = 'grid'
    assert result == expected


def test_find_variables_by_standard_name(roms_sgrid):
    nc_ds = NetCDFDataset(roms_sgrid)
    result = nc_ds.find_variables_by_attr(standard_name='time')
    expected = ['time']
    assert result == expected


def test_find_variables_by_standard_name_none(roms_sgrid):
    nc_ds = NetCDFDataset(roms_sgrid)
    result = nc_ds.find_variables_by_attr(standard_name='some standard_name')
    assert result == []


def test_sgrid_compliant_check(roms_sgrid):
    nc_ds = NetCDFDataset(roms_sgrid)
    result = nc_ds.sgrid_compliant_file()
    assert result


"""
Test NetCDF Dataset Without Nodes.

"""


def test_node_coordinates(wrf_sgrid):
    nc_ds = NetCDFDataset(wrf_sgrid)
    node_coordinates = nc_ds.find_node_coordinates('west_east_stag south_north_stag')  # noqa
    assert node_coordinates is None


def test_find_variable_by_attr(wrf_sgrid):
    nc_ds = NetCDFDataset(wrf_sgrid)
    result = nc_ds.find_variables_by_attr(cf_role='grid_topology',
                                          topology_dimension=2)
    expected = ['grid']
    assert result == expected


def test_find_variable_by_nonexistant_attr(wrf_sgrid):
    nc_ds = NetCDFDataset(wrf_sgrid)
    result = nc_ds.find_variables_by_attr(bird='tufted titmouse')
    assert result == []
