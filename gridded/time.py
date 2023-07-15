#!/usr/binenv python

from textwrap import dedent
import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta

from gridded.utilities import get_dataset


class Time(object):
    _const_time = None

    def __init__(self,
                 data=(datetime.now(),),
                 filename=None,
                 varname=None,
                 tz_offset=None,
                 origin=None,
                 displacement=timedelta(seconds=0),
                 *args,
                 **kwargs):
        '''
        Representation of a time axis. Provides interpolation alphas and indexing.

        :param time: Ascending list of times to use
        :param tz_offset: offset to compensate for time zone shifts
        :param origin: shifts the time interval to begin at the time specified
        :param displacement: displacement to apply to the time data. Allows shifting entire time interval into future or past
        :type time: netCDF4.Variable or [] of datetime.datetime
        :type tz_offset: datetime.timedelta
        :type origin: datetime.timedelta
        :type displacement: datetime.timedelta
        '''

        if isinstance(data, (nc4.Variable, nc4._netCDF4._Variable)):
            if (hasattr(nc4, 'num2pydate')):
                self.data = nc4.num2pydate(data[:], units=data.units)
            else:
                self.data = nc4.num2date(data[:], units=data.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        elif data is None:
            self.data = np.array([datetime.now()])
        else:
            self.data = np.asarray(data)

        if origin is not None:
            diff = self.data[0] - origin
            self.data -= diff

        self.data += displacement

        self.filename = filename
        self.varname = varname

#         if self.filename is None:
#             self.filename = self.id + '_time.txt'

        if tz_offset is not None:
            self.data += tz_offset

        if not self._timeseries_is_ascending(self.data):
            raise ValueError("Time sequence is not ascending")
        if self._has_duplicates(self.data):
            raise ValueError("Time sequence has duplicate entries")

        super(Time, self).__init__(*args, **kwargs)
    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    varname=None,
                    datavar=None,
                    tz_offset=None,
                    **kwargs):
        """
        construct a Time object from a netcdf file

        :param filename=None: name of netcddf file

        :param dataset=None: netcdf dataset object (one or the other)

        :param varname=None: name of the netcdf variable

        :param datavar=None: Either the time variable name, or
                             A netcdf variable that needs a Time object.
                             It will try to find the time variable that
                             corresponds to the passed in variable.

        :param tz_offset=None: offset to adjust for timezone, in hours.

        """
        breakpoint()
        if dataset is None:
            dataset = get_dataset(filename)
        if datavar is not None:
            if 'time' in datavar.attrs and datavar.attrs['time'] in dataset.dims.keys():
                varname = datavar.attrs['time']
            else:
                varname = datavar.dims[0] if 'time' in datavar.dims[0] else None
                if varname is None:
                    return cls.constant_time()
        time = cls(data=dataset[varname],
                   filename=filename,
                   varname=varname,
                   tz_offset=tz_offset,
                   **kwargs
                       )
        return time

    @classmethod
    def constant_time(cls):
        if cls._const_time is None:
            cls._const_time = cls(np.array([datetime.now()]))
        return cls._const_time

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        if isinstance(d, self.__class__) or d.__class__ in self.__class__.__mro__:
            d = d.data
        self._data = d

    @property
    def info(self):
        """
        Provides info about this Time object

        """
        msg = """
              Time object:
                filename: {}
                varname: {}
                first timestep: {}
                final timestep: {}
                number of timesteps: {}
              """.format(self.filename,
                         self.varname,
                         self.min_time,
                         self.max_time,
                         len(self.data),
                         )

        return dedent(msg)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def __eq__(self, other):
        r = self.data == other.data
        return all(r) if hasattr(r, '__len__') else r

    def __ne__(self, other):
        return not self.__eq__(other)

    def _timeseries_is_ascending(self, ts):
        return np.all(np.sort(ts) == ts)

    def _has_duplicates(self, time):
        return len(np.unique(time)) != len(time) and len(time) != 1

    @property
    def min_time(self):
        '''
        First time in series

        :rtype: datetime.datetime
        '''
        return self.data[0]

    @property
    def max_time(self):
        '''
        Last time in series

        :rtype: datetime.datetime
        '''
        return self.data[-1]

    def get_time_array(self):
        return self.data[:]

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
        if not (extrapolate or len(self.data) == 1):
            self.valid_time(time)
        index = np.searchsorted(self.data, time)
        if len(self.data) == 1:
            index = 0
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
        if not len(self.data) == 1 or not extrapolate:
            self.valid_time(time)
        i0 = self.index_of(time, extrapolate)
        if i0 > len(self.data) - 1:
            return 1
        if i0 == 0:
            return 0
        t0 = self.data[i0 - 1]
        t1 = self.data[i0]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()
