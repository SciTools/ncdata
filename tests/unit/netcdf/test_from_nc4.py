"""
Tests for :func:`ncdata.netcdf.from_nc4`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_netcdf`.  For example, different datatypes,
the use of groups, passing of data arrays and attributes.

This module only tests some specific API of the top-level access function, not covered
by the generic 'roundtrip' testcases.
"""

from pathlib import Path

import dask.array as da
import dask.config
import netCDF4 as nc
import numpy as np
import pytest

from ncdata import NcData, NcDimension, NcVariable
from ncdata.netcdf4 import from_nc4
from ncdata.utils import dataset_differences

from tests.data_testcase_schemas import make_testcase_dataset


def file_and_ncdata_from_spec(filepath: Path, test_spec: dict) -> NcData:
    """
    Make a testcase from a 'make_testcase_dataset' type spec.

    Save to a netcdf file at the given path, re-load as ncdata and return that.
    """
    make_testcase_dataset(filepath, test_spec)
    return from_nc4(filepath)


_sourcetype_opts = ["str", "path", "file", "group"]


@pytest.mark.parametrize("sourcetype", _sourcetype_opts)
def test_target_types(sourcetype, tmp_path):
    """Check the various ways of specifying the input data."""
    # This testcase is a rather complicated, but we need to test with groups, and we
    # may as well also test for variables which map dimensions from multiple levels.
    # In effect, this is also exercising tricky bits of 'dataset_differences' !!
    test_spec = {
        "dims": [dict(name="xdim", size=3)],
        "vars": [
            dict(name="x", dims=["xdim"], dtype=np.float32, data=[1.23, 2, 9])
        ],
        "groups": [
            {
                "dims": [dict(name="ydim", size=2)],
                "name": "inner_group",
                "vars": [
                    dict(
                        name="y",
                        dims=["xdim", "ydim"],
                        dtype=int,
                        data=[[77, 2], [13, 1], [19, 3]],
                    )
                ],
            }
        ],
    }
    original_path = tmp_path / "input.nc"
    make_testcase_dataset(filepath=original_path, spec=test_spec)

    if sourcetype == "path":
        source = original_path
    elif sourcetype == "str":
        source = str(original_path)
    elif sourcetype in ("file", "group"):
        source = nc.Dataset(original_path)
        if sourcetype == "group":
            source = source.groups["inner_group"]
    else:
        raise ValueError(
            f"unexpected test param : {sourcetype} not in {_sourcetype_opts}"
        )

    # Read the testcase from the generated netCDF file.
    ncdata = from_nc4(source)

    # Construct an NcData which *ought* to match the test specification.
    ncdata_expected = NcData(
        dimensions=[NcDimension(name="xdim", size=3)],
        variables=[
            NcVariable(
                name="x",
                dimensions=("xdim",),
                dtype=np.float32,
                data=[1.23, 2, 9],
            )
        ],
        groups=[
            NcData(
                name="inner_group",
                dimensions=[NcDimension(name="ydim", size=2)],
                variables=[
                    NcVariable(
                        name="y",
                        dimensions=("xdim", "ydim"),
                        dtype=np.int64,
                        data=[[77, 2], [13, 1], [19, 3]],
                    )
                ],
            )
        ],
    )
    if sourcetype == "group":
        ncdata_expected = ncdata_expected.groups["inner_group"]

    diffs = dataset_differences(ncdata, ncdata_expected)
    assert diffs == []


class TestVarStrs:
    def test_load_vlenstrs_basic(self, tmp_path):
        varstr_test_spec = {
            "dims": [dict(name="x", size=3)],
            "vars": [
                dict(
                    name="var_0",
                    dims=["x"],
                    dtype=str,
                    data=np.array(["one", "two", "three"], dtype="U10"),
                ),
            ],
        }
        filepath = tmp_path / "testinput_basic.nc"
        ncds = file_and_ncdata_from_spec(filepath, varstr_test_spec)
        var = ncds.variables["var_0"]
        assert var.dtype == "O"
        data = var.data
        assert isinstance(data, da.Array)
        assert data.shape == (3,)
        assert data.dtype == "O"
        values = data.compute()
        assert values.shape == (3,)
        assert values.dtype == "O"
        expect = np.array(["one", "two", "three"], dtype="O")
        assert np.all(values == expect)

    def test_load_large_chunks(self, tmp_path, mocker):
        # NB the input string array to create the file, via make_testcase_dataset, is
        #  NOT an object-array, as that's not how netCDF4 creates a 'str' type variable.
        nparray_onestr = np.array(["this"], dtype="U10").reshape((1, 1))
        darr_onestr = da.from_array(nparray_onestr, chunks=1)
        # expand to get a big array of (100 x 100) strings.
        big_string_array, _ = da.broadcast_arrays(
            darr_onestr, da.zeros((100, 100))
        )
        varstr_test_spec = {
            "dims": [dict(name="x", size=100), dict(name="y", size=100)],
            "vars": [
                dict(
                    name="var_1",
                    dims=["y", "x"],
                    dtype=str,
                    data=big_string_array,
                ),
            ],
        }
        filepath = tmp_path / "testinput_largearr.nc"
        with dask.config.set({"array.chunk-size": "4000b"}):
            ncds = file_and_ncdata_from_spec(filepath, varstr_test_spec)

        var = ncds.variables["var_1"]
        assert var.dtype == "O"
        data = var.data
        assert isinstance(data, da.Array)
        assert data.shape == (100, 100)
        assert data.chunksize == (1, 8)
