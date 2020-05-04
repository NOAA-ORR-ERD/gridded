"""
An example of how to load a non-compliant unstructured grid file

NOTE: This example works for loading a gridded dataset from
      an arbitrary text file. You should be able to work with
      it once loaded.

But ASaving it back out as a conforming file is broken:

We need a "proper" way to save a full dataset. Currently the code can save a UGRId, but the rest of teh
dataset info is lost. i.e. it can't find use the time variable. or it's coordinates.

This example uses a data file available in this gitHub repo:

https://github.com/erdc/AdhModel

You should be able to directly download it from:

https://github.com/erdc/AdhModel/blob/master/tests/test_files/SanDiego/SanDiego.nc
"""

from datetime import datetime, timedelta
import gridded
import netCDF4

with netCDF4.Dataset("SanDiego.nc") as nc:

    # need to convert to zero-indexing
    nodes = nc.variables['nodes'][:] - 1
    faces = nc.variables['E3T'][:, :3] - 1

    # make the grid
    # gridded.grids.Grid_U
    grid = gridded.grids.Grid_U(nodes=nodes,
                                faces=faces,
                                )

    # make the time object (handles time interpolation, etc)
    times_var = nc.variables['times'][:]

    # Time axis needs to be a list of datetime objects.
    # If the meta data are not there in the netcdf file, you have to do it by hand.
    start = datetime(2019, 1, 1, 12)
    times = [start + timedelta(seconds=val) for val in times_var]

    # This isn't a compliant file, so this will not work.
    # time_obj = gridded.time.Time.from_netCDF(dataset=nc,
    #                                          varname='times')

    time_obj = gridded.time.Time(data=times,
                                 filename=None,
                                 varname=None,
                                 tz_offset=None,
                                 origin=None,
                                 displacement=timedelta(seconds=0),)

    # make the variables
    depth = nc.variables['Depth']



    depth_var = gridded.variable.Variable(name='depth',
                                          units="meters",
                                          data=depth,
                                          data_file=nc,
                                          grid_file=nc,
                                          fill_value=0,
                                          location='node',
                                          attributes=None,
                                          )

    # global attributes
    attrs = {key: nc.getncattr(key) for key in nc.ncattrs()}

    # now make a dataset out of it all:
    ds = gridded.Dataset(ncfile=None,
                         grid=grid,
                         variables={'Depth': depth_var},
                         attributes=attrs
                         )

    ## now learn a bit about it:

    # What is its grid type?
    print("The dataset Grid is:", type(ds.grid))

    print("It has these variables:", list(ds.variables.keys()))

    print('You can access the variable with indexing: ds["Depth"]')
    Depth = ds["Depth"]

    print(Depth)

    print('you can access the Variables data directly:')
    print(Depth.data)

    # saving disabled -- not working for now.
    # # Now save it out as a conforming netcdf file:
    # ds.save("SanDiego_ugrid.nc", format="netcdf4") # only netcdf4 is supported for now

