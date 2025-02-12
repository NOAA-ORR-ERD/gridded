"""
generate a few netcdf files that have time in various forms.

NOTE: this was run, and the results saved -- but figured I
might as well save it in git in case we want to make more / different
"""

from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc4
from pathlib import Path

HERE = Path(__file__).parent

epoch = datetime(2024, 1, 1)
dt = timedelta(minutes=15)

timeseries = [epoch + i * dt for i in range(72 * 4)]

# print(timeseries)

units = [('UTC', 'days since 2024-1-1T00:00:00Z'),
         ('UTC-0', 'days since 2024-1-1T00:00:00+00:00'),
         ('UTC-UTC', 'days since 2024-1-1T00:00:00 UTC'),
         ('naive', 'days since 2024-1-1T00:00:00'),
         ('offset-7', 'days since 2024-1-1T00:00:00-7:00'),
         ('bad_tzo', 'days since 2024-1-1T00:00:00 U C'),
         ]

for name, unit in units:
    with nc4.Dataset(HERE / f'just_time_{name}.nc', 'w') as ds:
        ds.createDimension('time', size=len(timeseries))
        time = ds.createVariable('time', np.float64, dimensions=('time',))
        time.units = unit
        time.standard_name = 'time'
        vals = nc4.date2num(timeseries, unit)
        time[:] = vals

