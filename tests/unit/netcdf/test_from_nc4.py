"""
Tests for :func:`ncdata.netcdf.from_nc4`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_netcdf`.  For example, different datatypes,
the use of groups, passing of data arrays and attributes.

This module only tests some specific API of the top-level access function, not covered
by the generic 'roundtrip' testcases.
"""

from pathlib import Path

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
