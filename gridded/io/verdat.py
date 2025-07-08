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

# verdat only supports FEET or METERS
FEET = ("foot", "ft", "feet")
METER = ("meter", "m", "meters", "metre")
UNITS_MAP = {u: "FEET" for u in FEET}
UNITS_MAP.update({u: "METERS" for u in METER})



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
        # read the boundaries:
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
    Saves an appropriate dataset as a verdat file

    :param ds: The gridded.Dataset you want to save

    :param filename: name (full or relative path) of the file to save

    :param depth_var="depth": name of the variable with the depths in it.
                     if depth is None, all depths will be set to 1

    The dataset must: Have a UGrid grid

    If it has boundaries, they will be used. Otherwise,
    it will create them from the grid.
    """
    nodes = ds.grid.nodes
    if depth_var is None:
        depth = np.ones((nodes.shape[0],), dtype=np.float32)
        depth_units = ""
    else:
        depth = ds[depth_var]
        depth_units = UNITS_MAP[depth.units.strip().lower()]
    f_string = "{0:4d}, {1:10.6f}, {2:10.6f}, {3:8.3f}\n"

    with open(filename, 'w') as outfile:
        outfile.write("DOGS")
        outfile.write(f" {depth_units}\n")

        depth = depth.data
        # write out the boundaries first
        if ds.grid.boundaries is None:
            ds.grid.build_boundaries()
        bounds, open_bounds = order_boundary_segments(ds.grid.boundaries)
        points_written = []
        i = 1
        for bound in bounds:
            for p in bound:
                lon = nodes[p, 0]
                lat = nodes[p, 1]
                d = depth[p]
                outfile.write(f_string.format(i,
                                              lon,
                                              lat,
                                              d))
                points_written.append(p)
                i += 1
        # write the field points.
        points_written.sort()
        for j in range(len(nodes)):
            if j not in points_written:
                outfile.write(f_string.format(i,
                                              nodes[j, 0],
                                              nodes[j, 1],
                                              depth[j]))
                i += 1
        outfile.write(f_string.format(0, 0, 0, 0))
        outfile.write("{:d}\n".format(len(bounds)))
        i = 0
        for bound in bounds:
            i += len(bound)
            outfile.write("{:d}\n".format(i))


def order_boundary_segments(bound_segs):
    """
    verdat requires that the boundary segments all be in order

    This code re-orders the segments as required
    """
    # make a list so they can be removed as processed
    bound_segs = bound_segs.tolist()
    # sort just in case the point numbers are close
    # to each other and reverse so that we can work from the
    # back
    bound_segs.sort(reverse=True)

    # There can be zero or more boundaries
    closed_bounds = []
    open_bounds = []
    # start with the first boundary segment:
    while bound_segs:
        seg = bound_segs.pop()
        first_p, second_p = seg
        bound = [first_p, second_p]
        # find a connecting segment
        done = False
        while not done:
            for i in range(len(bound_segs) - 1, -1, -1):
                p0, p1 = bound_segs[i]
                if p0 == bound[-1]:
                    bound.append(p1)
                    bound_segs.pop(i)
                elif p1 == bound[-1]:
                    bound.append(p0)
                    bound_segs.pop(i)
                elif p0 == bound[0]:
                    bound.insert(p1)
                    bound_segs.pop(i)
                elif p1 == bound[0]:
                    bound.insert(0, p0)
                    bound_segs.pop(i)
                else:
                    continue
                if bound[0] == bound[-1]:  # closed the bound
                    bound.pop()  # take the duplicate point off
                    closed_bounds.append(bound)
                    done = True
                    break
                else:
                    done = False
                    break
            else:  # didn't find any more -- not closed
                # didn't get closed
                open_bounds.append(bound)
                done = True
    return closed_bounds, open_bounds


def make_outer_first(bounds, nodes):
    """
    figures out which boundary is the outer boundary,
    and puts it first in the list
    """
    # note: code for this is in the NOAA ERD ood_utils package.
    raise NotImplementedError("this code needs to be written")
    try:
        import geometry_utils
    except ImportError:
        print("writing verdat requires the geometry_utils module:\n"
              "github.com/NOAA-ORR-ERD/geometry_utils")

    #Assume the first bound is the outer one to start
    outer = bounds[0]
    for bound in bounds[1:]:
        pass


def set_winding_order(bounds, nodes, order="clockwise"):
    raise NotImplementedError("This code needs to be written")












