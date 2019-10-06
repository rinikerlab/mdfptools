"""
mdfptools
python implementation of molecular dynamics fingerprints as delineated in https://pubs.acs.org/doi/10.1021/acs.jcim.6b00778
"""

# Make Python 2 and 3 imports work the same
# Safe to remove with Python 3-only code
from __future__ import absolute_import

# Add imports here
# from .mdfptools import *

from . import Composer, Parameteriser, Extractor, Simulator, utils

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
