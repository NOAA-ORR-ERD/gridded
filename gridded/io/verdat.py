"""
functions for reading/writting "verdat" files

verdat ("vertex data") is a text file format for bathymetry grids for
triangular mesh models.

It is used by NOAA's Emergegency Response Division for its CATS model

It is limited to storing points with associated depths, and grid boudnaries
(including islands), but that's about it.
"""
import numpy as np
import gridded


def load_verdat(filename):

    grid = gridded.grids.Grid_U()

    # read the file
    with open(filename) as infile:
        header = infile.readline().strip()
        try:
            units = header.split()[1].lower()
        except IndexError:
            units = ""

        # read the points:
        lons, lats, depths = [], [], []
        for line in infile:
            ind, lon, lat, depth = [float(x) for x in line.split(",")]
            if ind == 0.0 and lon == 0.0 and lat == 0.0 and depth == 0.0:
                break
            lons.append(lon)
            lats.append(lat)
            depths.append(depth)
        # print(lons)
        # print(lats)
        # print(depths)
        #read the boundaries:
        line = infile.readline().strip()
        try:
            num_bounds = int(line)
        except ValueError:
            if line == "":
                num_bounds = 0
            else:
                raise ValueError("something wrong with file after the end of the points\n"
                                 "(The line after the line with all zeros should be the\n"
                                 "number of boundaries)")
        bounds = []
        start_point = 0
        for _ in range(num_bounds):
            end_point = int(infile.readline().strip())
            bound = []
            for i in range(start_point, end_point-1):
                bound.append((i, i + 1))
            bound.append(((i + 1), start_point))
            start_point = end_point
            bounds.extend(bound)


    nodes = np.c_[lons, lats]

    print("nodes:", nodes)
    print("bounds:", bounds)

    grid = gridded.grids.Grid_U(nodes=nodes,
                                boundaries=bounds)

    depth_var = gridded.variable.Variable(name="depth",
                                          units=units.lower(),
                                          data=depths,
                                          location='node',
                                          )
    ds = gridded.Dataset(grid=grid,
                         variables={'depth': depth_var},
                         )

    return ds




def save_verdat(ds, filename, depth_var="depth"):
    """
    Saves and approriate dataset as a verdat file

    :param ds: The gridded.Dataset you want to save

    :param filename: name (full or relative path) of the file to save

    :param depth_var="depth": name of the variable iwth the depths in it.

    The dataset must:

    * Have a UGrid grid
    * Have a variable for the depth

    If it has boundaries, they will be used. Otherwise,
    it will create them from the grid.
    """

    depth = ds[depth_var]
    with open(filename, 'w') as outfile:
        outfile.write("DOGS ")
        if depth.units:
            outfile.write(depth.units.upper())
        outfile.write("\n")
        for i, ((lon, lat), d) in enumerate(zip(ds.grid.nodes, depth.data)):
            outfile.write("{0:d},{1:.6f},{1:.6f}{1:.3f}".format(i + 1,
                                                                lon,
                                                                lat,
                                                                d))











