"""
Test ncdata.xarray by checking roundtrips for standard testcases.

Testcases start as netcdf files.
(1) check equivalence of cubes : xarray.load(file) VS xarray.load(ncdata(file))
(2) check equivalence of files : xarray -> file VS xarray->ncdata->file
"""

import numpy as np
import pytest
import xarray
from ncdata.netcdf4 import from_nc4, to_nc4
from ncdata.threadlock_sharing import lockshare_context
from ncdata.utils import dataset_differences
from ncdata.xarray import from_xarray, to_xarray

from tests.data_testcase_schemas import (
    BAD_LOADSAVE_TESTCASES,
    session_testdir,
    standard_testcase,
)

# Avoid complaints that imported fixtures are "unused"
# TODO: declare fixtures in usual way in pytest config?
standard_testcase, session_testdir


# _FIX_LOCKS = True
_FIX_LOCKS = False


@pytest.fixture(scope="session")
def use_xarraylock():
    if _FIX_LOCKS:
        with lockshare_context(xarray=True):
            yield
    else:
        yield


def check_load_equivalence(ds1: xarray.Dataset, ds2: xarray.Dataset):
    """
    Check that datasets differ only in "expected" ways.

    The key differences are due to coordinates remaining lazy in loading via ncdata, but
    having real data in a "normal" load.  This also affects which coords have indexes,
    but we are not checking that here anyway.
    """

    def check_attrs_equivalent(attrs1, attrs2):
        # Because dict-eq does not work when values can be arrays (!)
        okay = set(attrs1.keys()) == set(attrs2.keys())
        if okay:
            for attr in attrs1:
                okay = np.all(attrs1[attr] == attrs2[attr])
                if not okay:
                    break
        assert okay

    def check_vars_equivalent(v1, v2):
        check_attrs_equivalent(v1.attrs, v2.attrs)
        assert v1.dims == v2.dims
        assert v1.dtype == v2.dtype
        if v1.dtype.kind not in ("iufM"):
            # Nonnumeric cases are relatively simple
            result = np.all(v1.data == v2.data)
        else:
            # Numeric cases must allow for NaNs, which don't compare
            d1, d2 = v1.data, v2.data
            if d1.ndim == 0:
                # awkward special case where indexing operations otherwise fail
                d1, d2 = [a.reshape((a.size,)) for a in (d1, d2)]
            data_diff = d1 - d2
            # Account for NaN -or "NaT" for time types
            data_diff = data_diff[np.logical_not(np.isnan(data_diff))]
            # Note: not entirely happy with exact equality, but the time types make this
            if data_diff.dtype.kind == "f":
                # Slight tolerance on floats
                result = np.allclose(data_diff, 0)
            else:
                # Exact equality - including time types, which allclose can't handle.
                result = np.all(data_diff == 0)
        if hasattr(result, "compute"):
            result = result.compute()
        assert result

    check_attrs_equivalent(ds1.attrs, ds2.attrs)
    assert ds1.dims == ds2.dims
    assert list(ds1.variables) == list(ds2.variables)
    for varname in ds1.variables:
        check_vars_equivalent(ds1.variables[varname], ds2.variables[varname])


def test_load_direct_vs_viancdata(standard_testcase, use_xarraylock, tmp_path):
    source_filepath = standard_testcase.filepath
    ncdata = from_nc4(source_filepath)

    excluded_testcases = BAD_LOADSAVE_TESTCASES["xarray"]["load"]
    if any(key in standard_testcase.name for key in excluded_testcases):
        pytest.skip("excluded testcase (xarray cannot load)")

    # Load the testcase with Xarray.
    xr_ds = xarray.open_dataset(source_filepath, chunks=-1)

    # Load same, via ncdata
    xr_ncdata_ds = to_xarray(ncdata)

    check_load_equivalence(xr_ds, xr_ncdata_ds)


def test_save_direct_vs_viancdata(standard_testcase, tmp_path):
    source_filepath = standard_testcase.filepath

    excluded_testcases = BAD_LOADSAVE_TESTCASES["xarray"]["load"]
    excluded_testcases.extend(BAD_LOADSAVE_TESTCASES["xarray"]["save"])
    if any(key in standard_testcase.name for key in excluded_testcases):
        pytest.skip("excluded testcase")

    # Load the testcase into xarray.
    xrds = xarray.load_dataset(source_filepath, chunks=-1)

    # Re-save from Xarray
    temp_direct_savepath = tmp_path / "temp_save_xarray.nc"
    xrds.to_netcdf(temp_direct_savepath, engine="netcdf4")
    # Save same, via ncdata
    temp_ncdata_savepath = tmp_path / "temp_save_xarray_via_ncdata.nc"
    ncds_fromxr = from_xarray(xrds)
    to_nc4(ncds_fromxr, temp_ncdata_savepath)

    # Check equivalence
    results = dataset_differences(
        temp_direct_savepath,
        temp_ncdata_savepath,
        check_dims_order=False,
        check_dims_unlimited=False,  # TODO: remove this when we fix it
        suppress_warnings=True,
    )
    assert results == []
