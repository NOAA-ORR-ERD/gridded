#!/usr/bin/env python

import pytest
import os

from gridded import Grid


def test_init():
    """ tests you can intitize a basic datset"""
    G = Grid.from_netCDF(os.path.join('test_data', 'staggered_sine_channel.nc'))
    print G.node_lon

if __name__ == '__main__':
    test_init()
    print 'success'
