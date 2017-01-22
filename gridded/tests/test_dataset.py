#!/usr/bin/env python

import pytest

from ..gridded import Dataset


def test_init():
    """ tests you can intitize a basic datset"""
    D = Dataset()
