#!/usr/binenv python

from textwrap import dedent
import netCDF4 as nc4
import numpy as np

from datetime import datetime, timedelta

from gridded.utilities import get_dataset


class OutOfTimeRangeError(ValueError):
    """
    Used for when data is asked for outside of the range of times supported by a Time object.
    """
    pass


class TimeSeriesError(ValueError):
    """
    Used for when data is asked for outside of the range of times supported by a Time object.
    """
    pass

def offset_as_iso_string(offset_hours):
    """
    returns the offset as an isostring:

    -8:00

    3:30

    etc ...
    """
    if offset_hours is None:
        return ""
    else:
        sign = "-" if offset_hours <0 else "+"
        hours = int(abs(offset_hours))
        minutes = int((abs(offset_hours) - hours) * 60)
        return f"{sign}{hours:0>2}:{minutes:0>2}"


def parse_time_offset(unit_str):
    """
    find the time offset from a CF-style time units string.

    Follows the ISO format(s):

    ('UTC', 'days since 2024-1-1T00:00:00Z'),
    ('UTC-0', 'days since 2024-1-1T00:00:00+00:00'),
    ('naive', 'days since 2024-1-1T00:00:00'),
    ('offset-7', 'days since 2024-1-1T00:00:00-7:00')
    """
    import dateutil
    t_string = unit_str.split('since')[1]
    dt = dateutil.parser.parse(t_string)
    offset = dt.utcoffset()
    if offset is None:
        offset_hours = None
        name = None
    else:
        offset_hours = offset.total_seconds() / 3600
        name = dt.tzname()
        if name is None:
            name = offset_as_iso_string(offset_hours)
    return offset_hours, name


class Time:

    # Used to make a singleton with the constant_time class method.
    #  question: why not a ContantTime Class?
    _const_time = None

    def __init__(self,
                 data=(datetime.now(),),
                 filename=None,
                 varname=None,
                 tz_offset=None,
                 origin=None,
                 displacement=None,
                 new_tz_offset=None,
                 *args,
                 **kwargs):
        '''
        Representation of a time axis. Provides interpolation alphas and indexing.

        :param time: Ascending list of times to use
        :type time: netCDF4.Variable or Sequence of `datetime.datetime`

        :param data_: timezone of the data. If not provided, tz_offset is used.
        :type data_tz: `datetime.timedelta` or float or integer hours

        :param time_offset: offset to compensate for time zone shifts
        :type time_offset: `datetime.timedelta` or float or integer hours

        :param origin: shifts the time interval to begin at the time specified
        :type origin: `datetime.datetime`

        :param displacement: displacement to apply to the time data.
               Allows shifting entire time interval into future or past
        :type displacement: `datetime.timedelta`
        '''
        if isinstance(data, Time):
            self.data = data.data
        elif data is None:
            self.data = np.array([datetime.now()])
        else:
            self.data = np.asarray(data)
        
        # Quick check to ensure data is 'datetime-like' enough
        # to be compatible with timedelta operations
        # fixme: add a try:except around this to raise a meaningful error
        self.data += timedelta(seconds=0)

        if origin is not None:
            diff = self.data[0] - origin
            self.data -= diff


        self.filename = filename
        self.varname = varname

        if isinstance(tz_offset, (float, int)):
            tz_offset = timedelta(hours=tz_offset)

        # set the private attribute directly, because using the property
        # can cause the data to be shifted twice in case of loading from a serialization of this object
        self._tz_offset = tz_offset

        if new_tz_offset is not None:
            self.tz_offset = new_tz_offset

        if displacement is not None:
            self.displacement = displacement

        if not self._timeseries_is_ascending(self.data):
            raise TimeSeriesError("Time sequence is not ascending")
        if self._has_duplicates(self.data):
            raise TimeSeriesError("Time sequence has duplicate entries")
        super(Time, self).__init__(*args, **kwargs)

    @classmethod
    def locate_time_var_from_var(cls, datavar):
        if hasattr(datavar, 'time') and datavar.time in datavar._grp.dimensions.keys():
            varname = datavar.time
        else:
            varname = datavar.dimensions[0] if 'time' in datavar.dimensions[0] else None
        
        return varname
        

    @classmethod
    def from_netCDF(cls,
                    filename=None,
                    dataset=None,
                    varname=None,
                    datavar=None,
                    tz_offset=None,
                    new_tz_offset=None,
                    **kwargs):
        """
        construct a Time object from a netcdf file.

        You can specify the time variable name, or another variable,
        for which you want the corresponding times.

        :param filename=None: name of netcdf file

        :param dataset=None: netcdf dataset object (one or the other)

        :param varname=None: Name of the time variable.

        :param datavar=None: A netcdf variable or name of netcdf variable
                             for which you want the corresponding time
                             object.
                             It will try to find the time variable that
                             corresponds to the passed in variable.

        :param tz_offset=None: Timezone offset the data are in, in hours.
                               If None: offset will be read from file, if present.
                               If not in file, UTC will be assumed
                               If 'Naive', then no offset will be assigned.

        :param new_tz_offset=None: New Timezone offset desired in hours.
                                   Data will be shifted to the new offset.
                                   If tz_offset is set to Naive, then this will fail.
                                   (it can't be changed without knowing what it was to begin with)
        """
        if varname is None and datavar is None:
            raise TypeError('you must pass in either a varname or a datavar')
        if dataset is None:
            dataset = get_dataset(filename)

        if varname is None and datavar is not None:
            if isinstance(datavar, str):
                datavar = dataset.variables[datavar]
            varname = cls.locate_time_var_from_var(datavar)
            # fixme: This seems risky -- better to raise and deal with it elsewhere.
            if varname is None:
                return cls.constant_time()
        if isinstance(varname, str):
            tvar = dataset.variables[varname]
        # figure out the timezone_offset
        name = ""  # this should be passed in ...
        if isinstance(tz_offset, str) and tz_offset.lower() == 'naive':
            tz_offset = None
        elif tz_offset is None:
            # look in the time units attribute:
            tz_offset, name = parse_time_offset(tvar.units)

        tdata = nc4.num2date(tvar[:], units=tvar.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
        # Fixme: use the name and pass it through?
        time = cls(data=tdata,
                   filename=filename,
                   varname=varname,
                   tz_offset=tz_offset,
                   new_tz_offset=new_tz_offset,
                   **kwargs
                   )
        return time

    @classmethod
    def constant_time(cls):
        """
        Returns a Time object that represents no change in time

        In practice, that's a Time object with a single datetime
        """
        # this caches a single instance of a constant time object
        # in the class, and always returns the same one (a singleton)
        if cls._const_time is None:
            cls._const_time = cls(np.array([datetime.now()]))
        return cls._const_time

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        # If it's passed in a Time object, it gets its data object
        # Fixme: Why? this seems like more magic than necessary
        if isinstance(data, self.__class__) or data.__class__ in self.__class__.__mro__:
            data = data.data
        # add check for valid datetime list?
        self._data = np.asarray(data)

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
        return iter(self.data)

    def __eq__(self, other):
        # r = self.data == other.data
        # return all(r) if hasattr(r, '__len__') else r
        if not isinstance(other, self.__class__):
            return False
        return np.array_equal(self.data, other.data) and self.tz_offset == other.tz_offset

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

    @property
    def tz_offset(self):
        '''
        Timezone offset of the time series

        :rtype: datetime.timedelta
        '''
        if not hasattr(self, '_tz_offset'):
            self._tz_offset = None
            return None
        else:
            return self._tz_offset

    @tz_offset.setter
    def tz_offset(self, offset):
        '''
        Set the timezone offset of the time series. Replaces the current offset by
        reverting the current offset and applying the new offset.

        :param offset: offset to adjust for timezone, in hours.
        :type offset: float, integer, or datetime.timedelta
        '''
        if not hasattr(self, '_tz_offset'):
            self._tz_offset = None
        if isinstance(offset, (float, int)):
            offset = timedelta(hours=offset)
        if offset is None:
            self._tz_offset = None
            return
        
        if self._tz_offset is not None:
            # undo previous offset
            self.data -= self._tz_offset
        
        self.data += offset
        self._tz_offset = offset

    @property
    def displacement(self):
        if not hasattr (self, '_displacement'):
            self._displacement = None
        return self._displacement
    
    @displacement.setter
    def displacement(self, displacement):
        if not hasattr(self, '_displacement') or self._displacement is None:
            if displacement is None:
                self._displacement = None
                return
            if isinstance(displacement, (float, int)):
                displacement = timedelta(hours=displacement)
            self.data += displacement
            self._displacement = displacement
        else:
            # why not jsut make  this an init thing, then??
            raise AttributeError('displacement is settable only once')
        
    
    def get_time_array(self):
        """
        returns a copy of the internal data array
        """
        return self.data.copy()

    def time_in_bounds(self, time):
        '''
        Checks if time provided is within the bounds represented by this object.

        :param time: time to be queried
        :type time: `datetime.datetime`
        :rtype: bool
        '''
        return not ((time < self.min_time) or (time > self.max_time))

    def valid_time(self, time):
        if time < self.min_time or time > self.max_time:
            raise OutOfTimeRangeError('time specified ({0}) is not within the bounds of '
                                      'the time ({1} to {2})'.format(time.strftime('%c'),
                                                                 self.min_time.strftime('%c'),
                                                                 self.max_time.strftime('%c')))

    def index_of(self, time, extrapolate=False):
        '''
        Returns the index of the provided time with respect to the time intervals in the file.

        :param time: Time to be queried
        :type time: `datetime.datetime`

        :param extrapolate: whether to allow extrapolation:
                            i.e. will not raise if outside the bounds of the time data.
        :type extrapolate: bool

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

        This is the weighting to give the index before
        -- 1-alpha would be the weight to the index after.


        :param time: Time to be queried
        :type time: `datetime.datetime`

        :param extrapolate: if True, 0.0 (before) or 1.0 (after) is returned.
                            if False, a ValueError is raised if outside the time series.
        :type extrapolate: bool

        :return: interpolation alpha
        :rtype: float (0 <= r <= 1)
        '''
        if (not extrapolate) and (not len(self.data) == 1):
            self.valid_time(time)
        i0 = self.index_of(time, extrapolate)
        if i0 > len(self.data) - 1:
            return 1.0
        if i0 == 0:
            return 0.0
        t0 = self.data[i0 - 1]
        t1 = self.data[i0]
        return (time - t0).total_seconds() / (t1 - t0).total_seconds()
