# Copyright SciTools contributors
#
# This file is part of SciTools/ncdata and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for :mod:`ncdata`."""

import os
from pathlib import Path
from typing import AnyStr


# Note : `_env_bin_path` and the ncgen/ncdump path constants are shared across
# the tests package, so they live here rather than in an individual test module.
def _env_bin_path(exe_name: AnyStr | None = None):
    """
    Return a Path object for (an executable in) the environment bin directory.

    Parameters
    ----------
    exe_name : str
        If set, the name of an executable to append to the path.

    Returns
    -------
    exe_path : Path
        A path to the bin directory, or an executable file within it.

    Notes
    -----
    For use in tests which spawn commands which should call executables within
    the Python environment, since many IDEs (Eclipse, PyCharm) don't
    automatically include this location in $PATH (as opposed to $PYTHONPATH).
    """
    exe_path = Path(os.__file__)
    exe_path = (exe_path / "../../../bin").resolve()
    if exe_name is not None:
        exe_path = exe_path / exe_name
    return exe_path


NCGEN_PATHSTR = str(_env_bin_path("ncgen"))
NCDUMP_PATHSTR = str(_env_bin_path("ncdump"))

testdata_dir = Path(__file__).parent / "testdata"


class MonitoredArray:
    """
    An array wrapper for monitoring dask deferred accesses.

    Wraps a real array, and can be read (indexed), enabling it to be wrapped with
    dask.array_from_array.  It then records the read operations performed on it.
    """

    def __init__(self, data):
        """Create a MonitoredArray, containing a numpy array."""
        self.dtype = data.dtype
        self.shape = data.shape
        self.ndim = data.ndim
        self._data = data
        self._accesses = []

    def __getitem__(self, keys):
        """Fetch indexed data section."""
        self._accesses.append(keys)
        return self._data[keys]


def version_tuple(version_string):
    """
    Make a comparable tuple from a package version string.

    Allows only the final segment to have non-numeric parts.
    Returns a tuple of the parts.
    """
    parts = version_string.split(".")
    last_part = parts[-1]
    while len(last_part) and not last_part[-1].isdigit():
        last_part = last_part[:-1]
    if len(last_part):
        parts[-1] = last_part
    else:
        parts = parts[:-1]
    if any(str(int(part)) != part for part in parts):
        raise ValueError(f"invalid numerics in version string: {version_string!r}")
    return tuple(int(part) for part in parts)
