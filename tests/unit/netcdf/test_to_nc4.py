"""
Tests for :func:`ncdata.netcdf.to_nc4`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_netcdf`.  For example, different datatypes,
the use of groups, passing of data arrays and attributes.

This module tests specific API properties of the top-level access function not covered
by the generic 'roundtrip' testcases.  This includes error cases.
"""

from pathlib import Path
from typing import List

import netCDF4 as nc
import numpy as np
import pytest

from ncdata import NcData
from ncdata.netcdf4 import from_nc4, to_nc4
from ncdata.utils import dataset_differences

from tests.data_testcase_schemas import make_testcase_dataset


def file_and_ncdata_from_spec(filepath: Path, test_spec: dict) -> NcData:
    """
    Make a testcase from a 'make_testcase_dataset' type spec.

    Save to netcdf at the given path, re-load as ncdata and return that.
    """
    make_testcase_dataset(filepath, test_spec)
    return from_nc4(filepath)


_targettype_opts = ["str", "path", "file"]


@pytest.mark.parametrize("targettype", _targettype_opts)
def test_target_types(targettype, tmp_path):
    """Check the various ways of specifying the output location."""
    # Testcase is minimal spec with just one scalar variable.
    test_spec = {
        "vars": [dict(name="x", dims=[], dtype=np.float32, data=[1.23])]
    }
    original_path = tmp_path / "input.nc"
    ncdata = file_and_ncdata_from_spec(original_path, test_spec)

    target_path = tmp_path / "output.nc"
    if targettype == "path":
        target = target_path
    elif targettype == "str":
        target = str(target_path)
    elif targettype == "file":
        target = nc.Dataset(target_path, "w")
    else:
        raise ValueError(
            f"unexpected test param : {targettype} not in {_targettype_opts}"
        )

    to_nc4(ncdata, target)
    if targettype == "file":
        target.close()

    assert target_path.exists()
    assert dataset_differences(target_path, original_path) == []


def fetch_nc_var(nc_file: nc.Dataset, var_path: str or List[str]):
    """
    Return a variable instance from an open dataset for inspection.

    E.G. ``nc_var = fetch_nc_var(varname)``
    E.G. ``nc_var = fetch_nc_var([groupname, subgroupname, varname])``

    Returns
    -------
    nc.Variable
    """
    nc = nc_file
    if isinstance(var_path, str):
        var_path = [var_path]
    while len(var_path) > 1:
        groupname, var_path = var_path[0], var_path[1:]
        nc = nc.groups[groupname]
    return nc.variables[var_path[0]]


def test_var_kwargs(tmp_path):
    """
    Check the function of the "var_kwargs" argument.

    NOTE: effectively, also tests that you *can* control chunking and compression.
    """
    # Create a test-spec including same-named vars in different groups and subgroups.
    # N.B. we don't bother to specify data for them
    test_spec = {
        "dims": [dict(name="x", size=4)],
        "vars": [
            dict(name="var_1", dims=["x"], dtype=np.float32),
        ],
        "groups": [
            {
                "name": "subgroup_1",
                "vars": [
                    dict(name="var_1", dims=["x"], dtype=np.float32),
                ],
                "groups": [
                    {
                        "name": "sub_subgroup_1_2",
                        "vars": [
                            dict(name="var_xx", dims=["x"], dtype=np.float32)
                        ],
                    },
                ],
            },
        ],
    }
    input_path = tmp_path / "input.nc"
    ncdata = file_and_ncdata_from_spec(input_path, test_spec)

    # Save the testcase ncdata to an output file, with var-specific controls
    output_path = tmp_path / "output.nc"
    # Set up controls to modify e.g. chunking and compression of some variables.
    var_kwargs = {
        "var_1": dict(compression="zlib", complevel=3),
        "/subgroup_1": {
            "var_1": dict(contiguous=True),
            "/sub_subgroup_1_2": {"var_xx": dict(chunksizes=(2,))},
        },
    }
    to_nc4(ncdata, output_path, var_kwargs=var_kwargs)

    # re-open the saved file and check specific var properties.
    saved_ds = nc.Dataset(output_path)

    var_1 = fetch_nc_var(saved_ds, "var_1")
    assert var_1.filters()["zlib"] is True
    assert var_1.filters()["complevel"] == 3
    assert var_1.chunking() == [4]

    var_2 = fetch_nc_var(saved_ds, ["subgroup_1", "var_1"])
    assert var_2.filters()["zlib"] is False
    assert var_2.chunking() == "contiguous"

    var_3 = fetch_nc_var(
        saved_ds, ["subgroup_1", "sub_subgroup_1_2", "var_xx"]
    )
    assert var_3.filters()["zlib"] is False
    assert var_3.chunking() == [2]


def test_var_kwargs__bad_kwarg(tmp_path):
    """Check that unsuitable var_kwargs entries are errored."""
    # Testcase is a minimal spec with just one scalar variable.
    test_spec = {
        "vars": [dict(name="x", dims=[], dtype=np.float32, data=[1.23])]
    }
    original_path = tmp_path / "input.nc"
    ncdata = file_and_ncdata_from_spec(original_path, test_spec)

    output_path = tmp_path / "output.nc"
    var_kwargs = {"x": dict(fill_value=999)}
    expected_msg_regex = (
        "additional `var_kwargs` for variable.*"
        r"included key\(s\) \['fill_value'\],"
    )
    with pytest.raises(ValueError, match=expected_msg_regex):
        to_nc4(ncdata, output_path, var_kwargs=var_kwargs)
