"""
Test ncdata.xarray by checking roundtrips for standard testcases.

Testcases start as netcdf files.
(1) check equivalence of cubes : xarray.load(file) VS xarray.load(ncdata(file))
(2) check equivalence of files : xarray -> file VS xarray->ncdata->file
"""

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

    # Treat as OK if it passes xarray comparison
    assert xr_ds.identical(xr_ncdata_ds)


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
