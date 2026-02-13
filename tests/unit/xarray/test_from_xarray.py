"""
Tests for :func:`ncdata.xarray.from_xarray`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_xarray`.  For example, different datatypes,
the use of dimensions, passing of data arrays and attributes.

This module only tests some specific API and behaviours of the top-level function, not
covered by the generic 'roundtrip' testcases.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from ncdata.xarray import from_xarray

from tests import MonitoredArray
from tests.data_testcase_schemas import make_testcase_dataset


def file_and_xarray_from_spec(
    filepath: Path, test_spec: dict, **xr_kwargs
) -> xr.Dataset:
    """
    Make a testcase from a 'make_testcase_dataset' type spec.

    Save to a netcdf file at the given path, re-load as xarray and return that.
    """
    make_testcase_dataset(filepath, test_spec)
    ds = xr.open_dataset(filepath, **xr_kwargs)
    return ds


def test_lazy_nocompute(tmp_path):
    """
    Check that from_xarray converts lazy variable data without fetching.
    """
    # Create a simple xarray dataset from a spec
    test_spec = {
        "dims": [dict(name="x", size=3)],
        "vars": [
            dict(
                name="x", dims=["x"], dtype=np.float32, data=np.arange(3.0)
            ),  # dimcoord
            dict(name="var_x", dims=["x"], dtype=np.float32, data=[1.0, 3, 7]),
        ],
    }
    original_path = tmp_path / "input.nc"
    xrds = file_and_xarray_from_spec(original_path, test_spec)

    # Replace data with a "monitored" array in a Dask wrapper
    # -- so we can check if+when it is fetched (i.e. computed)
    real_numpy_data = np.arange(3.0)
    monitored_array = MonitoredArray(real_numpy_data)
    lazy_data = da.from_array(monitored_array, chunks=(2,), meta=np.ndarray)
    var = xrds.variables["var_x"]
    var.data = lazy_data

    # Make the call
    _ = from_xarray(xrds)

    # Check that the underlying real data has *not* been read ..
    assert len(monitored_array._accesses) == 0
    # .. then compute, and check again
    assert np.all(var.data.compute() == real_numpy_data)
    # NOTE: order of access is not guaranteed, hence 'sorted'.
    assert sorted(monitored_array._accesses) == [
        (slice(0, 2),),
        (slice(2, 3),),
    ]


def test_real_nocopy(tmp_path):
    """
    Check that from_xarray converts real variable data directly without copying.
    """
    # Create a simple xarray dataset from a spec
    test_spec = {
        "dims": [dict(name="x", size=3)],
        "vars": [
            dict(
                name="x", dims=["x"], dtype=np.float32, data=np.arange(3.0)
            ),  # dimcoord
            dict(name="var_x", dims=["x"], dtype=np.float32, data=[1.0, 3, 7]),
        ],
    }
    original_path = tmp_path / "input.nc"
    xrds = file_and_xarray_from_spec(original_path, test_spec)

    # Replace data with a known local real array,
    var = xrds.variables["var_x"]
    real_data = np.arange(3.0)
    var.data = real_data

    ncdata = from_xarray(xrds)

    # Check that the ncdata variable array is the *SAME* array as the original
    # NOTE: this does *NOT* work for integers, which xarray converts to floats
    assert ncdata.variables["var_x"].data is real_data


@pytest.mark.parametrize("unlim_dims", ["none", "x"])
def test_kwargs__unlimited_dims(tmp_path, unlim_dims):
    """Check xarray-save kwargs control : make the "x" dimension unlimited."""
    test_spec = {
        "dims": [dict(name="x", size=3)],
        "vars": [
            dict(name="var_x", dims=["x"], dtype=np.float32, data=[1.0, 3, 7])
        ],
    }
    original_path = tmp_path / "input.nc"
    xrds = file_and_xarray_from_spec(original_path, test_spec)

    kwargs = {
        "none": {},
        "x": dict(unlimited_dims="x"),
    }[unlim_dims]
    ncdata = from_xarray(xrds, **kwargs)

    assert ncdata.dimensions["x"].unlimited == (unlim_dims == "x")


def test_kwargs__encoding__packing(tmp_path):
    """Repeat the above, checking use of the 'encoding' kwarg."""
    test_spec = {
        "dims": [dict(name="x", size=3)],
        "vars": [
            dict(
                name="var_x", dims=["x"], dtype=np.float32, data=[1.0, 2, 3.3]
            )
        ],
    }
    original_path = tmp_path / "input.nc"
    xrds = file_and_xarray_from_spec(original_path, test_spec)

    kwargs = {
        "encoding": {
            "var_x": {
                "dtype": np.int16,
                "scale_factor": 0.1,
                "add_offset": -5.3,
            }
        }
    }
    ncdata = from_xarray(xrds, **kwargs)

    var = ncdata.variables["var_x"]
    assert var.dtype == np.int16
    assert np.all(var.data == [63, 73, 86])
