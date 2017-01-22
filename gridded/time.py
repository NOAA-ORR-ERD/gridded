#!/usr/binenv python

# py2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta

from .utilities import get_dataset


class Time(object):
    _const_time = None

    def __init__(self,
                 time=None,
                 filename=None,
                 varname=None,
                 tz_offset=None,
                 offset=None,
                 **kwargs):
        '''
        Representation of a time axis. Provides interpolation alphas and indexing.

        :param time: Ascending list of times to use
        :param tz_offset: offset to compensate for time zone shifts
        :type time: netCDF4.Variable or [] of datetime.datetime
        :type tz_offset: datetime.timedelta

        '''
        if isinstance(time, (nc4.Variable, nc4._netCDF4._Variable)):
            self.time = nc4.num2date(time[:], units=time.units)
        elif time is None:
            self.time = [datetime.now()]
        else:
            self.time = time

        self.filename = filename
        self.varname = varname

#         if self.filename is None:
#             self.filename = self.id + '_time.txt'

        if tz_offset is not None:
            self.time += tz_offset

        if not self._timeseries_is_ascending(self.time):
            raise ValueError("Time sequence is not ascending")
        if self._has_duplicates(self.time):
            raise ValueError("Time sequence has duplicate entries")

        self.name = time.name if hasattr(time, 'name') else None

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    varname=None,
                    datavar=None,
                    tz_offset=None,
                    **kwargs):
        if dataset is None:
            dataset = get_dataset(filename)
        if datavar is not None:
            if hasattr(datavar, 'time') and datavar.time in dataset.dimensions.keys():
                varname = datavar.time
            else:
                varname = datavar.dimensions[0] if 'time' in datavar.dimensions[0] else None
                if varname is None:
                    return None
        time = cls(time=dataset[varname],
                   filename=filename,
                   varname=varname,
                   tz_offset=tz_offset,
                   **kwargs
                       )
        return time

    @staticmethod
    def constant_time():
        if Time._const_time is None:
            Time._const_time = Time([datetime.now()])
        return Time._const_time

    @property
    def data(self):
        if self.filename is None:
            return self.time
        else:
            return None

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        return self.time.__iter__()

    def __eq__(self, other):
        r = self.time == other.time
        return all(r) if hasattr(r, '__len__') else r

    def __ne__(self, other):
        return not self.__eq__(other)

    def _timeseries_is_ascending(self, ts):
        return all(np.sort(ts) == ts)

    def _has_duplicates(self, time):
        return len(np.unique(time)) != len(time) and len(time) != 1

    @property
    def min_time(self):
        '''
        First time in series

        :rtype: datetime.datetime
        '''
        return self.time[0]

    @property
    def max_time(self):
        '''
        Last time in series

        :rtype: datetime.datetime
        '''
        return self.time[-1]

    def get_time_array(self):
        return self.time[:]

    def time_in_bounds(self, time):
        '''
        Checks if time provided is within the bounds represented by this object.

        :param time: time to be queried
        :type time: datetime.datetime
        :rtype: boolean
        '''
        return not time < self.min_time or time > self.max_time

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise ValueError('time specified ({0}) is not within the bounds of the time ({1} to {2})'.format(
                time.strftime('%c'), self.min_time.strftime('%c'), self.max_time.strftime('%c')))

    def index_of(self, time, extrapolate=False):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.

        :param time: Time to be queried
        :param extrapolate:
        :type time: datetime.datetime
        :type extrapolate: boolean
        :return: index of first time before specified time
        :rtype: integer
        '''
        if not (extrapolate or len(self.time) == 1):
            self.valid_time(time)
        index = np.searchsorted(self.time, time)
        return index

    def interp_alpha(self, time, extrapolate=False):
        '''
        Returns interpolation alpha for the specified time

        :param time: Time to be queried
        :param extrapolate:
        :type time: datetime.datetime
        :type extrapolate: boolean
        :return: interpolation alpha
        :rtype: double (0 <= r <= 1)
        '''
        if not len(self.time) == 1 or not extrapolate:
            self.valid_time(time)
        i0 = self.index_of(time, extrapolate)
        if i0 > len(self.time) - 1:
            return 1
        if i0 == 0:
            return 0
        t0 = self.time[i0 - 1]
        t1 = self.time[i0]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()

