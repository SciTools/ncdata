"""
Tests for :func:`ncdata.iris.from_iris`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_iris`.  For example, different datatypes,
the use of groups, passing of data arrays and attributes.

This module only tests some specific API and behaviours of the top-level function, not
covered by the generic 'roundtrip' testcases.
"""

from unittest.mock import patch

import dask.array as da
from iris.coords import DimCoord
from iris.cube import Cube
import numpy as np
import pytest

from ncdata.iris import from_iris

from tests import MonitoredArray


def sample_cube(data_array=None):
    """
    Create a basic test-cube.

    It always has a first dimension of 3, and a matching "time" dim-coord.
    You can pass in a different data array with more dims, but first dim must be 3.
    """
    if data_array is None:
        data_array = np.arange(3.0)
    time_co = DimCoord(
        [0.0, 1, 2], standard_name="time", units="days since 2001-01-01"
    )
    cube = Cube(data_array, standard_name="air_temperature", units="K")
    cube.add_dim_coord(time_co, 0)
    return cube


def test_lazy_nocompute():
    """Check that translation from iris converts lazy data without fetching it."""
    real_array = np.arange(6.0).reshape((3, 2))
    monitored_array = MonitoredArray(real_array)
    dask_array = da.from_array(monitored_array, chunks=(3, 1), meta=np.ndarray)
    cube = sample_cube(dask_array)

    _ = from_iris(cube)

    # check that the wrapped data array has not yet been read (i.e. no compute)
    assert len(monitored_array._accesses) == 0
    # .. and the cube itself has still not been realised
    assert cube.has_lazy_data()
    # check sameness
    assert np.all(cube.data == real_array)
    # The original data *has* now been accessed, in 2 chunks ..
    # NOTE: order of access is not guaranteed, hence 'sorted'.
    assert sorted(monitored_array._accesses) == [
        (slice(0, 3, None), slice(0, 1, None)),
        (slice(0, 3, None), slice(1, 2, None)),
    ]


def test_real_nocopy():
    """Check that translation from iris converts real data without copying it."""
    real_array = np.arange(6.0)
    monitored_array = MonitoredArray(real_array)

    # We must use a slight patch to prevent Cube creation converting our array to a
    # MaskedArray.
    # This ensures we don't copy, so that "cube.data is monitored_array".
    with patch("iris.cube.ma.isMaskedArray", return_value=True):
        cube = Cube(monitored_array, var_name="x")
        assert cube.data is monitored_array

    ncdata = from_iris(cube)

    # check that conversion has *NOT* fetched the data
    # (except for a 0-length test access required by dask.array.from_array)
    assert monitored_array._accesses == [(slice(0, 0),)]

    # check for values equality, then re-check that the data *has* now been fetched.
    var = ncdata.variables["x"]
    monitored_array._accesses = []
    assert np.all(var.data.compute() == real_array)
    assert monitored_array._accesses == [(slice(0, 6),)]


def test_multiple_cubes():
    """Check that we can pass multiple cubes to the conversion."""
    cube1 = sample_cube()
    cube2 = cube1.copy()
    cube2.rename("cube2")

    # convert 2 cubes at once
    ncdata = from_iris([cube1, cube2])

    # the result should now have 3 data variables: time, air_temperature and 'cube2'
    # N.B. this means the two cubes share a 'time' variable, as it should be.
    assert list(ncdata.variables.keys()) == [
        "air_temperature",
        "time",
        "cube2",
    ]


@pytest.mark.skip("'from_iris' not yet working with unlimited dimensions.")
def test_kwargs_unlimited():
    """Check iris-save kwargs control : make the time dimension unlimited."""
    cube = sample_cube()
    kwargs = dict(unlimited_dimensions=["time"])

    ncdata = from_iris(cube, **kwargs)

    assert ncdata.dimensions["time"].unlimited is True


def test_kwargs_fill_value():
    """Check iris-save kwargs control : save with a user fill-value."""
    data = np.ma.masked_array([0.0, 1.0, 2.0], mask=[False, True, False])
    cube = sample_cube(data)

    # Save with afil-value keyword.
    kwargs = dict(fill_value=-999.0)
    ncdata = from_iris(cube, **kwargs)

    # Check that the resulting variable has the expected attribute + (raw) data values
    ncdata_var = ncdata.variables["air_temperature"]
    assert ncdata_var.avals["_FillValue"] == -999.0
    assert np.all(ncdata_var.data == [0.0, -999, 2])


def test_kwargs_packing():
    """Check iris-save kwargs control : scale+offset packing into a smaller dtype."""
    data = np.array([0.0, 1.0, 2.0])
    cube = sample_cube(data)

    # Save with packing controls in a 'packing' keyword.
    kwargs = dict(
        packing=dict(dtype=np.int16, scale_factor=0.1, add_offset=-5.3)
    )
    ncdata = from_iris(cube, **kwargs)

    # Check that the resulting variable has the expected dtype and (raw) data values
    var_data = ncdata.variables["air_temperature"].data
    assert var_data.dtype == np.dtype(np.int16)
    assert isinstance(var_data, da.Array)
    real_data = var_data.compute()
    assert np.all(real_data == [53, 63, 73])
