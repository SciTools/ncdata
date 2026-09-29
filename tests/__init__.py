"""Tests for :mod:`ncdata`."""

from pathlib import Path

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
    """Make a comparable tuple from a package version string.

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
        raise ValueError(
            f"invalid numerics in version string: {version_string!r}"
        )
    return tuple(int(part) for part in parts)
