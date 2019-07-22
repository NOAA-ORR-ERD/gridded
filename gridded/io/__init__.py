"""
subpackge for custom input / ouput code


At some point, it would be nice to have gridded have a
registry of file readers, so you could point Dataset()
at any file, and it would loop through them and try to
read them, and so find whicherver one worked.one

But for now, special functions for loading custom file
types can go here.


"""

from .verdat import load_verdat

