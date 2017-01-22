
class Depth(object):

    def __init__(self,
                 surface_index=-1):
        self.surface_index = surface_index
        self.bottom_index = surface_index

    @classmethod
    def from_netCDF(cls,
                    surface_index=-1):
        return cls(surface_index)

    def interpolation_alphas(self, points, data_shape, _hash=None):
        return (None, None)