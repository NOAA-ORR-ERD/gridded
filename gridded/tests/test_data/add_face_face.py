"""
Add a mesh_links variable to

"small_ugrid_zero_based.nc"

This was used to add a missing variable to the example file

The script was kept around as an example for future use.

"""

import numpy as np
import netCDF4

data = np.array([[-1,  3,  2],
       [ 2,  5, -1],
       [ 0, -1,  1],
       [ 4, -1,  0],
       [-1, 11,  3],
       [ 1,  6, -1],
       [ 8,  7,  5],
       [ 6,  9, -1],
       [-1,  9,  6],
       [ 8, 10,  7],
       [13, -1,  9],
       [ 4, -1, 12],
       [11, 14, -1],
       [-1, 15, 10],
       [12, -1, 15],
       [14, 16, 13],
       [15, 17, -1],
       [-1, 18, 16],
       [17, -1, 19],
       [18, 20, -1],
       [19, -1, -1]], dtype=np.int32)

ds = netCDF4.Dataset("small_ugrid_zero_based.nc", mode='a')

mesh = ds.variables['mesh']

var = ds.createVariable("mesh_face_links",
                        data.dtype,
                        dimensions=('mesh_num_face', 'mesh_num_vertices'),
                        )
var.setncattr('cf_role', "face_face_connectivity")
var.setncattr('long_name', "Maps each triangular face to its neighbors")
var[:] = data

# should it have coordinates??

# used ugrid.build_face_face_connectivity to make.




# ds.close()
