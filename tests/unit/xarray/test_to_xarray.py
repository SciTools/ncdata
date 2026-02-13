"""
Tests for :func:`ncdata.xarray.to_xarray`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_xarray`.  For example, different datatypes,
the assignment of dimensions and attributes.

This module only tests some specific API and behaviours of the top-level function, not
covered by the generic 'roundtrip' testcases.
"""

import dask.array as da
import numpy as np
import pytest

from ncdata import NcData, NcDimension, NcVariable
from ncdata.xarray import to_xarray

from tests import MonitoredArray


def test_lazy_nocompute():
    """Check that to_xarray transfers lazy variable data without computing it."""
    real_numpy_data = np.array([1.23, 4, 7.3, 4.1], dtype=np.float32)
    monitored_array = MonitoredArray(real_numpy_data)
    dask_array = da.from_array(monitored_array, chunks=(2,), meta=np.ndarray)
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(name="var_x", dimensions=["x"], data=dask_array)
        ],
    )

    # convert to xarray.Dataset
    xrds = to_xarray(ncdata)

    # Check that no data was read during conversion.
    assert monitored_array._accesses == []
    # fetch + check values, then check access record again
    data = xrds.variables["var_x"].data
    assert np.all(data.compute() == real_numpy_data)
    # should have fetched in 2 chunks
    # NOTE: order of access is not guaranteed, hence 'sorted'.
    assert sorted(monitored_array._accesses) == [
        (slice(0, 2),),
        (slice(2, 4),),
    ]


def test_real_nocopy():
    """Check that to_xarray transfers real variable data directly without copying."""
    real_numpy_data = np.array([1.23, 4, 7.3, 4.1], dtype=np.float32)
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(name="var_x", dimensions=["x"], data=real_numpy_data)
        ],
    )

    # convert to xarray.Dataset
    xrds = to_xarray(ncdata)

    # Check that the data content is  the *SAME ARRAY*
    xr_data = xrds.variables["var_x"]._data
    # There may be a wrapper object, for some versions of xarray,
    #  so test by modifying the original + check that the result tracks it.
    real_numpy_data[-1] = 777
    assert np.all(xr_data == real_numpy_data)


@pytest.mark.parametrize("scaleandoffset", ["WITHscaling", "NOscaling"])
def test_kwargs__scaleandoffset(scaleandoffset):
    """Check the operation of kwargs (= xarray load controls)."""

    # Make a test NcData dataset with a scaled-and-offset variable.
    raw_int_data = np.array([66, 75, 99], dtype=np.int16)
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(
                name="var_x",
                dimensions=["x"],
                attributes={
                    "scale_factor": np.array(0.1, dtype=np.float32),
                    "add_offset": np.array(-5.3, dtype=np.float32),
                },
                data=raw_int_data,
            )
        ],
    )

    # convert to Xarray
    if scaleandoffset == "NOscaling":
        # Set a kwarg to turn OFF the normal mask+scale operation
        load_kwargs = dict(mask_and_scale=False)
        expected_data = raw_int_data
    else:
        assert scaleandoffset == "WITHscaling"
        load_kwargs = {}
        # in "normal" default operation, the "var_x" data gets scaled to these values
        expected_data = np.array([1.3, 2.2, 4.6], dtype=np.float32)

    # Make the call
    xrds = to_xarray(ncdata, **load_kwargs)

    # check the resulting Xarray variable data
    xr_data = xrds["var_x"].data
    assert np.allclose(xr_data, expected_data)
