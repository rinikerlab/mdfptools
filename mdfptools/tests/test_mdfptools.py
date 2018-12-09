"""
Unit and regression test for the mdfptools package.
"""

# Import package, test suite, and other packages as needed
import mdfptools
import pytest
import sys

def test_mdfptools_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "mdfptools" in sys.modules
