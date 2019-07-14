"""
Created on Apr 7, 2015

@author: ayan

"""

from __future__ import (absolute_import, division, print_function)

import pytest

from gridded.pysgrid.sgrid import load_grid

from .write_nc_test_files import non_compliant_sgrid


def test_exception_raised(non_compliant_sgrid):
    with pytest.raises(ValueError):
        load_grid(non_compliant_sgrid)
