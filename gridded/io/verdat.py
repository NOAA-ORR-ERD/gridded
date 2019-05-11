"""
functions for reading/writting "verdat" files

verdat ("vertex data") is a text file format for bathymetry grids for
triangular mesh models.

It is used by NOAA's Emergegency Response Division for its CATS model

It is limited to storing points with associated depths, and grid boudnaries
(including islands), but that's about it.
"""

import gridded
import gridded.pyugrid.ugrid


def dataset_from_verdat(filename):

    # create an empty UGrid object
    grid = gridded.pyugrid.ugrid.UGrid()

    # read the file
    with open(filename) as infile:
        header = infile.readline().strip()
        try:
            units = header.split()[1].lower()
        except IndexError:
            units = ""
        print("units:", units)

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
                raise ValueError("somethign wrong with file after the end of the points\n"
                                 "(The line after the line with all zeros should be the\n"
                                 "number of boundaries)")
        bounds = []
        for _ in range(num_bounds):
            start_point = 0
            end_point = int(infile.readline().strip())
            bound = []
            for i in range(start_point, end_point):
                print(bound, bounds)
                bound.append((i, i+1))
            bounds = bounds.append(bound)

        print(bounds)









