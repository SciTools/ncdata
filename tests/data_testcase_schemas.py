"""
Define a set of "standard" test cases, built as actual netcdf files.

The main product is a pytest fixture `standard_testcase`, which is parametrised over
testcase names, and returns info on the testcase, its defining spec and a filepath it
can be loaded from.

These testcases also include various pre-existing testfiles, which are NOT built from
specs.  This enables us to perform various translation tests on standard testfiles from
the Iris and Xarray test suites.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import iris.tests
import netCDF4 as nc
import numpy as np
import pytest


def data_types():
    """
    Produce a sequence of valid netCDF4 datatypes.

    All possible datatypes for variable data or attributes.
    Not yet supporting variable or user-defined (structured) types.

    Results are strings for all valid numeric dtypes, plus 'string'.
    The strings are our choice, chosen to suit inclusion in pytest parameter naming.
    """
    # unsigned int types
    for n in (1, 2, 4, 8):
        yield f"u{n}"
    # signed int types
    for n in (1, 2, 4, 8):
        yield f"i{n}"
    # float types
    for n in (4, 8):
        yield f"f{n}"
    # .. and strings
    # NB we use 'string' instead of 'S1', as strings always need handling differently.
    yield "string"


# Confirm that the above list of types matches those of netCDF4, with specific fixes
# Get the dtype list from the netCDF default fill-values.
_nc_dtypes = set(nc.default_fillvals.keys())
# Remove the numpy-only complex types
_nc_dtypes = set(
    typename for typename in _nc_dtypes if np.dtype(typename).kind != "c"
)
# Also replace 'S1' with our own 'string' type marker
_nc_dtypes.remove("S1")
_nc_dtypes.add("string")
# This should match the list of dtypes which we support (and test against)
assert set(data_types()) == _nc_dtypes

# Suitable test values for each attribute/data type.
_INT_Approx_2x31 = int(2**31 - 1)
_INT_Approx_2x32 = int(2**32 - 2)
_INT_Approx_2x63 = int(2**63 - 3)
_INT_Approx_2x64 = int(2**64 - 4)

_Datatype_Sample_Values = {
    "u1": np.array([3, 250], dtype="u1"),
    "i1": np.array([-120, 120], dtype="i1"),
    "u2": np.array([4, 65001], dtype="u2"),
    "i2": np.array([-30001, 30002], dtype="i2"),
    "u4": np.array([5, _INT_Approx_2x32], dtype="u4"),
    "i4": np.array([-_INT_Approx_2x31, _INT_Approx_2x31], dtype="i4"),
    "u8": np.array([6, _INT_Approx_2x64], dtype="u8"),
    "i8": np.array([-_INT_Approx_2x63, _INT_Approx_2x63], dtype="i8"),
    "f4": np.array([1.23e-9, -34.0e12], dtype="f4"),
    "f8": np.array([1.23e-34, -34.0e52], dtype="f8"),
    "string": ["one", "three"],
}
# Just confirm that the list of sample-value types matches 'data_types'.
assert set(_Datatype_Sample_Values.keys()) == set(data_types())


def _write_nc4_dataset(
    spec: dict,
    ds: nc.Dataset,  # or an inner Group
    parent_spec: dict = None,
):
    """
    Create a netcdf test file from a 'testcase spec'.

    Inner routine for ``make_testcase_dataset``.
    Separated for recursion and to keep dataset open/close in separate wrapper routine.
    """

    def objmap(objs):
        """Create a map from a list of dim-specs, indexing by their names."""
        return {obj["name"]: obj for obj in objs}

    # Get sub-components of the spec.
    # NB most elements are lists of dicts which all have a 'name' item ..
    var_dims = objmap(spec.get("dims", []))
    var_specs = objmap(spec.get("vars", []))
    group_specs = objmap(spec.get("groups", []))
    # .. but attr specs are a simple dict
    attr_specs = spec.get("attrs", {})

    # Assign actual file attrs (group or global)
    for name, value in attr_specs.items():
        ds.setncattr(name, value)

    # Create actual file dims
    for dim_name, dim in var_dims.items():
        # Dims are a simple list, containing names as a dict entry
        size, unlimited = dim["size"], dim.get("unlimited", False)
        if unlimited:
            size = 0
        ds.createDimension(dim_name, size)

    # Create and configure actual file variables
    for var_name, var_spec in var_specs.items():
        var_dims = var_spec["dims"]  # this is just a list of names
        data = var_spec.get("data", None)
        if data is not None:
            data = np.array(data)
            dtype = data.dtype
        else:
            dtype = np.dtype(var_spec["dtype"])
        attrs = var_spec.get("attrs", {})
        fill_value = attrs.pop("_FillValue", None)

        # Create the actual data variable
        nc_var = ds.createVariable(
            varname=var_name,
            dimensions=var_dims,
            datatype=dtype,
            fill_value=fill_value,
        )

        # Add attributes
        for name, value in attrs.items():
            nc_var.setncattr(name, value)

        # Add data, provided or default-constructed
        if data is None:
            shape = nc_var.shape
            n_points = int(np.prod(shape))
            if nc_var.dtype.kind == "S":
                data = np.array(
                    "abcdefghijklmnopqrstuvwxyz"[:n_points], dtype="S1"
                )
            else:
                data = np.arange(1, n_points + 1)
            data = data.astype(dtype)
            i_miss = var_spec.get("missing_inds", [])
            if i_miss:
                data = np.ma.masked_array(data)
                data[i_miss] = np.ma.masked
            data = data.reshape(shape)

        # provided data is raw values (not scaled)
        nc_var.set_auto_maskandscale(False)
        nc_var[:] = data

    # Finally, recurse over sub-groups
    for group_name, group_spec in group_specs.items():
        nc_group = ds.createGroup(group_name)
        _write_nc4_dataset(
            spec=group_spec,
            ds=nc_group,
            parent_spec=parent_spec,
        )


def make_testcase_dataset(filepath, spec):
    """
    Convert a test dataset 'spec' into an actual netcdf file.

    A generic routine interpreting "specs" provided as dictionaries.
    This is rather frustratingly similar to ncdata.to_nc4, but it needs to remain
    separate as we use it to test that code (!)

    specs are just a structure of dicts and lists...
    group_spec = {
        'name': str
        'dims': [ *{'name':str, size:int [, 'unlim':True]} ]
        'attrs': {'name':str, 'value':array}
        'vars': [ *{
                        'name':str,
                        'dims':list[str],
                        [, 'attrs': {'name':str, 'value':array}]
                        [, 'dtype':dtype]
                        [, 'data':array]
                        [, 'missing_inds': list(int)]  # NB **flat** indexes
                    }
                ]
        'groups': list of group_spec
    }

    From this we populate dims + vars, all filling data content according to certain
    default rules (or the provided 'data' item).
    """
    ds = nc.Dataset(filepath, "w")
    try:
        _write_nc4_dataset(spec, ds)
    finally:
        ds.close()


_minimal_variable_test_spec = {
    "vars": [dict(name="var_0", dims=[], dtype=int)]
}


_simple_test_spec = {
    "dims": [dict(name="x", size=3), dict(name="y", size=2)],
    "attrs": {"ga1": 2.3, "gas2": "this"},
    "vars": [dict(name="vx", dims=["x"], dtype=np.int32)],
    "groups": [
        {
            "name": "g_inner",
            "dims": [dict(name="dim_i1", size=4)],
            "vars": [
                dict(
                    name="vix1",
                    dims=["dim_i1", "y"],
                    dtype="u1",
                    missing_inds=[2, 5],
                    attrs={"_FillValue": 100},
                )
            ],
            "attrs": {"ia1": 777},
        }
    ],
}

_scaleoffset_test_spec = {
    "dims": [dict(name="x", size=3)],
    "vars": [
        dict(
            name="vx",
            dims=["x"],
            data=np.array([2, 7, 10], dtype=np.int16),
            attrs={
                "scale_factor": np.array(0.01, dtype=np.float32),
                "add_offset": 12.34,
            },
        )
    ],
}

_masked_floats_test_spec = {
    "dims": [dict(name="x", size=3)],
    "vars": [
        dict(
            name="vx",
            dims=["x"],
            data=np.ma.array([2, 7, 10], mask=[0, 1, 0], dtype=np.float32),
        )
    ],
}

_masked_withnans_test_spec = {
    "dims": [dict(name="x", size=4)],
    "vars": [
        dict(
            name="vx",
            dims=["x"],
            data=np.ma.array(
                [2.0, 7.0, 10, np.nan], mask=[0, 1, 0, 0], dtype=np.float32
            ),
        )
    ],
}

_masked_ints_test_spec = {
    "dims": [dict(name="x", size=3)],
    "vars": [
        dict(
            name="vx",
            dims=["x"],
            data=np.ma.array([2, 7, 10], mask=[0, 1, 0], dtype=np.int16),
        )
    ],
}

_masked_scaled_ints_test_spec = {
    "dims": [dict(name="x", size=3)],
    "vars": [
        dict(
            name="vx",
            dims=["x"],
            data=np.ma.array([2, 7, 10], mask=[0, 1, 0], dtype=np.int16),
            attrs={
                "scale_factor": np.array(0.01, dtype=np.float32),
                "add_offset": 12.34,
            },
        )
    ],
}

# Define a sequence of standard testfile specs, with suitable param-names.
_STANDARD_TESTCASES: Dict[str, Union[Path, dict]] = {}


# A decorator for spec-generating routines.
# It automatically **calls** the wrapped function, and adds all the results into the
# global "_Standard_Testcases" dictionary.
def standard_testcases_func(func):
    """
    Include the results of the wrapped function in the 'standard testcases'.

    A decorator for spec-generating routines.  It automatically **calls** the wrapped
    function, and adds the results into the global "_Standard_Testcases" dictionary.
    """
    _STANDARD_TESTCASES.update(func())
    return func


@standard_testcases_func
def _define_simple_testcases():
    testcases = {
        "ds_Empty": {},
        "ds_Minimal": _minimal_variable_test_spec,
        "ds_Basic": _simple_test_spec,
        "ds_testdata1": (
            Path(__file__).parent
            / "testdata"
            / "toa_brightness_temperature.nc"
        ),
        "ds_scaleoffset": _scaleoffset_test_spec,
        "ds_masked_floats": _masked_floats_test_spec,
        "ds_masked_ints": _masked_ints_test_spec,
        "ds_masked_withnans": _masked_withnans_test_spec,
        "ds_masked_scaled_ints": _masked_scaled_ints_test_spec,
    }
    return testcases


ADD_IRIS_FILES = True
# ADD_IRIS_FILES = False


@standard_testcases_func
def _define_iris_testdata_testcases():
    testcases = {}
    if ADD_IRIS_FILES:
        # Add all the netcdf files from the Iris testdata paths
        _testdirpath = Path(iris.tests.get_data_path("NetCDF"))
        _netcdf_testfile_paths = _testdirpath.rglob("**/*.nc")

        # optional exclusions for useful speedup in test debugging.
        # EXCLUDES = [
        #     "_unstructured_",
        #     "_volcello_",
        #     "_GEMS_CO2",
        #     "_ORCA2__votemper",
        # ]
        EXCLUDES = []
        for filepath in _netcdf_testfile_paths:
            param_name = str(filepath)
            # remove unwanted path elements
            param_name = param_name.replace(str(_testdirpath), "")
            if param_name.endswith(".nc"):
                param_name = param_name[:-3]
            # replace path-separators and other awkward chars with dunder
            for char in ("/", ".", "-"):
                param_name = param_name.replace(char, "__")
            # TEMPORARY: skip unstructured ones, just for now, as it makes the run faster
            if not any(key in param_name for key in EXCLUDES):
                # if "small_theta_colpex" in param_name:
                testcases[f"testdata__{param_name}"] = filepath

    return testcases


ADD_UNIT_TESTS = True
# ADD_UNIT_TESTS = False


@standard_testcases_func
def _define_unit_singleitem_testcases():
    testcases = {}
    if ADD_UNIT_TESTS:
        # Add selected targeted test datasets.

        # dataset with a single attribute
        testcases["ds__singleattr"] = {"attrs": {"attr1": 1}}

        # dataset with a single variable
        testcases["ds__singlevar"] = {
            "vars": [dict(name="vx", dims=[], dtype=np.int32)]
        }

        # dataset with a single variable
        testcases["ds__dimonly"] = {
            "dims": [dict(name="x", size=2)],
        }

    return testcases


@standard_testcases_func
def _define_unit_dtype_testcases():
    testcases = {}
    if ADD_UNIT_TESTS:
        # dataset with attrs and vars of all possible types
        # TODO: .. and missing-data testcases ???
        for dtype_name in data_types():
            dtype = "S1" if dtype_name == "string" else dtype_name
            if dtype_name == "string":
                # Not working for now
                # TODO: fix !
                # continue
                pass

            testcases[f"ds__dtype__{dtype_name}"] = {
                "attrs": {
                    f"tstatt_type__{dtype_name}__single": _Datatype_Sample_Values[
                        dtype_name
                    ][
                        0
                    ],
                    f"tstatt_type__{dtype_name}__multi": _Datatype_Sample_Values[
                        dtype_name
                    ],
                },
                "vars": [dict(name="vx", dims=[], dtype=dtype)],
            }

        testcases["ds__stringvar__singlepoint"] = {
            "dims": [dict(name="strlen", size=3)],
            "vars": [
                dict(
                    name="vx",
                    dims=["strlen"],
                    dtype="S1",
                    data=np.array("abc", dtype="S1"),
                )
            ],
        }

        testcases["ds__stringvar__multipoint"] = {
            "dims": [
                dict(name="x", size=2),
                dict(name="strlen", size=3),
            ],
            "vars": [
                dict(
                    name="vx",
                    dims=["x", "strlen"],
                    dtype="S1",
                    data=np.array([list("abc"), list("def")], dtype="S1"),
                )
            ],
        }

    return testcases


@pytest.fixture(scope="session")
def session_testdir(tmp_path_factory):
    """Provide a common temporary-files directory path."""
    tmp_dir = tmp_path_factory.mktemp("standard_schema_testfiles")
    return tmp_dir


@dataclass
class TestcaseSchema:
    """The type of information object returned by the "standard testcase" fixture."""

    name: str = ""
    spec: dict = None
    filepath: Path = None


@pytest.fixture(params=list(_STANDARD_TESTCASES.keys()))
def standard_testcase(request, session_testdir):
    """
    Provide a set of "standard" dataset testcases.

    A fixture returning a parameterised sequence of TestCaseSchema objects.

    Some of these are based on a 'testcase spec', from which it builds an actual netcdf
    testfile : these files are created in a temporary directory provided by pytest
    ("tmp_path_factory"), are are cached so they only get built once per session.
    Other testcases are just pre-existing test files.

    For each it returns a TestcaseSchema tuple (parameter-name, spec, filepath).
    For those not based on a spec, 'spec' is None.
    """
    name = request.param
    spec = _STANDARD_TESTCASES[name]
    if isinstance(spec, dict):
        # Build a temporary testfile from the spec, and pass that out.
        filepath = session_testdir / f"sampledata_{name}.nc"
        if not filepath.exists():
            # Cache testcase files so we create only once per session.
            make_testcase_dataset(str(filepath), spec)
    else:
        # Otherwise 'spec' is a pre-existing test file: pass (filepath, spec=None)
        filepath = spec
        spec = None

    return TestcaseSchema(name=name, spec=spec, filepath=filepath)


# Some testcases that are known not to load or save correctly, due to limitations or
# errors in the data handling packages
BAD_LOADSAVE_TESTCASES = {
    "iris": {
        # We think Iris can load ~anything (maybe returning nothing)
        "load": [],
        # Iris can't save data with no data-variables.
        "save": ["ds_Empty", "ds__singleattr", "ds__dimonly"],
    },
    "xarray": {
        # We think Xarray can load ~anything (maybe returning nothing)
        "load": [
            # .. except a few specific bounds variables generate a peculiar error
            # """
            #   xarray.core.variable.MissingDimensionsError: 'time_bnd' has more than
            #   1-dimension and the same name as one of its dimensions
            #   ('time', 'time_bnd'). xarray disallows such variables because they
            #   conflict with the coordinates used to label dimensions.
            # """
            "small_rotPole_precipitation",
            "small_FC_167",
            # And this one, which has recently acquired a problem?
            "test_monotonic_coordinate",
        ],
        # Xarray can save ~anything
        "save": [r"test_monotonic_coordinate"],
    },
}
