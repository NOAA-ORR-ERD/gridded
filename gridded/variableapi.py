
from collections.abc import Iterable
import hashlib
import numpy as np
from gridded.grids import Grid, Grid_R, Grid_S, Grid_U
from gridded.utilities import (
    _align_results_to_spatial_data,
    _reorganize_spatial_data,
    asarraylike,
    get_dataset,
    parse_filename_dataset_args,
    search_dataset_for_variables_by_varname,
)
class VariableAPI(object):
    
    memoization_enabled = False # class-level flag to enable/disable memoization for all VariableAPI instances
    
    @property
    def data(self):
        '''
        Abstract property that must be implemented by subclasses.
        
        The only requirement is that the data property returns something with a shape
        that is consistent with the underlying data being represented.
        '''
        
        raise NotImplementedError("VariableAPI is an abstract base class. Subclasses must implement the .data property")

    @property
    def dimension_ordering(self):
        """
        Returns a list that describes the dimensions of the property's data.
        If a dimension_ordering is assigned, it will continue to use that.
        If no dimension_ordering is set, then a default ordering will be generated
        based on the object properties and data shape.

        For example, if the data has 4 dimensions and is represented by a
        Grid_S (structured grid), and the Variable has a depth and time assigned,
        then the assumed ordering is ['time','depth','lon','lat']

        If the data has 3 dimensions, self.grid is a Grid_S, and self.time is None,
        then the ordering is ['depth','lon','lat']
        If the data has 3 dimensions, self.grid is a Grid_U, the ordering is
        ['time','depth','ele']
        """
        if not hasattr(self, "_order"):
            self._order = None
        if self._order is not None:
            return self._order
        else:
            order = []
            if self.time is not None:
                order.append("time")
            if self.depth is not None:
                order.append("depth")
            if self.grid is not None:
                if isinstance(self.grid, (Grid_S, Grid_R)):
                    order.extend(["lon", "lat"])
                else:
                    order.append("ele")
            return order

    @dimension_ordering.setter
    def dimension_ordering(self, order):
        self._order = order
    
    def at(self, p, t, extrapolate=False, unmask=False, _hash=None):
        """
        Retrieve or interpolate the value of the data to positions P at times T

        :param p: Cartesian coordinates to be queried (P).
            Lon, Lat required, Depth (Z) is optional
            Coordinates must be organized as a 2D array or list,
            one coordinate per row.

            Failure to provide point data in this format may cause
            unexpected behavior. Not providing a Z value when a depth
            dimension is present will cause an IndexError.
            
            Any iterable passed into numpy.array() that results in a ndarray
            of shape (N, 2) or (N, 3) should work. (eg list of tuples, list of list)

                [[Lon1, Lat1, Z1],
                [Lon2, Lat2, Z2],
                [Lon3, Lat3, Z3],
                ...]

        :type p: Nx3 array of double


        :param t: The time(s) at which to query these points (T)
            Providing a single time will squeeze the time dimension out of the result, returning a NxD array instead of NxTxD.
            If NxTxD is desired, provide a list of times, even if it is a single time in the list.
        :type t: scalar (single) or iterable of datetime.datetime objects
        
        :param extrapolate: Turns extrapolation on or off globally for this call. How extrapolation in any 
            particular dimension is handled is determined by the subcomponent objects (Time, Depth, Grid)
        :type extrapolate: boolean (default False)
        
        :param unmask: If True, returns a filled numpy.ndarray instead using self.fill_value
        :type unmask: boolean (default False)
                
        :param _hash: optional pre-computed hash for memoization. If not provided, it will be computed from the points and time.
        :type _hash: str
        
        :return: returns a NxTxD array of interpolated values.
        :rtype: numpy.ma.MaskedArray or numpy.ndarray
        """
        ps, ts, _hash = self._prepare_at(p, t, _hash=_hash)
        out = self._compute_at(ps, ts, extrapolate=extrapolate, unmask=unmask, _hash=_hash)
        retval = self._post_compute_at(out, p, t, unmask=unmask, _hash=_hash)
        return retval
    
    interpolate = at  # common request
    
    def _prepare_at(self, p, t, _hash=None, _mem=True):
        """
        First stage of the .at function. Handles points and time normalization
        and hash generation for memoization.     
        
        """
        ps = _reorganize_spatial_data(p)
        if isinstance(t, (str, bytes)):
            raise TypeError("times cannot be a string or bytes")
        if not isinstance(t, Iterable):
            ts = [t]
        else:
            ts = t

        if _hash is None and self.memoization_enabled:
            _hash = self._get_hash(p, t)

        return ps, ts, _hash
    
    def _compute_at(self, ps, ts, extrapolate=False, unmask=False, _hash=None):
        """
        Computation core of the .at function. All arguments should already be prepared for internal use.
        
        Returns the interpolated values at the points and times specified.
        If memoization is enabled, return of memoized results are done here.
        
        Return value must be a NxTxD array of interpolated values.
        Return value must be a numpy.ma.MaskedArray.
        """
        raise NotImplementedError("VariableAPI is an abstract base class. Subclasses must implement the ._compute_at method")

    def _post_compute_at(self, out, p, t, unmask=False, _hash=None):
        """
        Post computation step of the .at function.
        Handles fill values, unmasking, and memoization of the result.
        NOTE that unit conversion of computed results must happen *after* memoization
        as the requested units for any given .at call are not part of the memoization hash.
        """
        
        #shape of incoming value is expected to be (N, T, D) where N = number of points, T = number of times, D = dimensionality of the variable (eg 1 for scalar, 2+ for vector)
        
        if not isinstance(t, Iterable):
            out = out.squeeze(axis=1)  # squeeze out the time dimension if a single time was provided
        if isinstance(out, np.ma.MaskedArray) and hasattr(self, "fill_value") and self.fill_value is not None:
            np.ma.set_fill_value(out, self.fill_value)
        if unmask:
            out = np.ma.filled(out)

        if self.memoization_enabled:
            #memoize with original points and times, not the reorganized points and times
            self._memoize_result(p, t, out, self._result_memo, _hash=_hash)
        return out

    def _get_hash(self, points, times, extrapolate=False):
        """
        Returns a SHA1 hash of the array of points passed in
        """
        return (hashlib.sha1(points.tobytes()).hexdigest(), hashlib.sha1(str(times).encode("utf-8")).hexdigest(), bool(extrapolate))

    def _memoize_result(self, points, times, result, D, _copy=False, _hash=None, extrapolate=False):
        if _copy:
            result = result.copy()
        result.setflags(write=False)
        if _hash is None:
            _hash = self._get_hash(points, times)
        if D is not None and len(D) > 4:
            D.popitem(last=False)
        D[_hash] = result
        D[_hash].setflags(write=False)

    def _get_memoed(self, points, times, D, _copy=False, _hash=None):
        if _hash is None:
            _hash = self._get_hash(points, times)
        if D is not None and _hash in D:
            return D[_hash].copy() if _copy else D[_hash]
        else:
            return None

    def _clear_memo(self):
        if self._result_memo is not None:
            self._result_memo.clear()
