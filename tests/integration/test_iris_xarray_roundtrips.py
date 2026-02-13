"""
Test ncdata.iris by checking roundtrips for standard testcases.

Testcases start as netcdf files.
(1) check equivalence of cubes : iris.load(file) VS iris.load(ncdata(file))
(2) check equivalence of files : iris -> file VS iris->ncdata->file
"""

from subprocess import check_output

import dask.array as da
import iris
import netCDF4
import numpy as np
import pytest
import xarray

from ncdata.iris import from_iris
from ncdata.iris_xarray import cubes_to_xarray
from ncdata.netcdf4 import from_nc4
from ncdata.threadlock_sharing import lockshare_context
from ncdata.utils import dataset_differences
from ncdata.xarray import from_xarray

from tests.data_testcase_schemas import (
    BAD_LOADSAVE_TESTCASES,
    session_testdir,
    standard_testcase,
)
from tests.integration.equivalence_testing_utils import (
    adjust_chunks,
    cubes_equal__corrected,
    namesort_cubes,
    nanmask_cube,
    prune_cube_varproperties,
    remove_cube_nounits,
    set_tiny_chunks,
)

# Avoid complaints that imported fixtures are "unused"
standard_testcase, session_testdir, adjust_chunks

# import iris.fileformats.netcdf._thread_safe_nc as ifnt

_FIX_LOCKS = True
# _FIX_LOCKS = False
if _FIX_LOCKS:

    @pytest.fixture(scope="session")
    def use_irislock():
        with lockshare_context(iris=True, xarray=True):
            yield


# _TINY_CHUNKS = True
_TINY_CHUNKS = False
set_tiny_chunks(_TINY_CHUNKS)


def test_roundtrip_ixi(standard_testcase, use_irislock, adjust_chunks):
    source_filepath = standard_testcase.filepath

    # Skip cases where there are no data variables for Iris to load.
    exclude_case_keys = (
        # unloadable by iris
        BAD_LOADSAVE_TESTCASES["iris"]["load"]
        # unsaveable by iris (= can't convert to xarray)
        + BAD_LOADSAVE_TESTCASES["iris"]["save"]
        # unloadable by xarray
        + BAD_LOADSAVE_TESTCASES["xarray"]["load"]
        # TODO: remaining unresolved problems ...
        + [
            # string dimension problem
            "ds__dtype__string",
            # outstanding dims-mismatch problems.
            "testing__small_theta_colpex",
            # coordinate attributes on mesh coordinate variables
            "testdata____unstructured_grid__data_C4",
            "testdata____ugrid__21_triangle_example",
            # Problem with units on time bounds
            "label_and_climate__small_FC_167",
            # Broken UGRID files now won't load in Iris >= 3.10
            "unstructured_grid__mesh_C12",
            "_unstructured_grid__theta_nodal_xios",
        ]
    )
    if any(key in standard_testcase.name for key in exclude_case_keys):
        pytest.skip("excluded testcase")

    # _Debug = True
    _Debug = False
    if _Debug:
        print(f"\ntestcase: {standard_testcase.name}")
        print("spec =")
        print(standard_testcase.spec)
        print("\nncdump =")
        txt = check_output([f"ncdump {source_filepath}"], shell=True).decode()
        print(txt)

    # Load the testcase with Iris.
    iris_cubes = iris.load(source_filepath)

    # Remove any attributes that match properties of a netCDF4.Variable
    # Since, due to an Iris bug, these cannot be saved to a file variable.
    # N.B. a bit subtle, as mostly doesn't apply to
    _PRUNE_CFVAR_PROPNAMES = True
    # _PRUNE_CFVAR_PROPNAMES = False
    if _PRUNE_CFVAR_PROPNAMES:
        removed_props = prune_cube_varproperties(iris_cubes)
        assert removed_props == set() or removed_props == {
            "name",
        }

    # Remove any 'no_units' units, since these do not save correctly
    # see : https://github.com/SciTools/iris/issues/5368
    remove_cube_nounits(iris_cubes)

    # Unfortunately, cube order is not guaranteed to be stable.
    iris_cubes = namesort_cubes(iris_cubes)

    # Convert to xarray, and back again.
    ds = cubes_to_xarray(iris_cubes)
    ncds_fromxr = from_xarray(ds)
    from ncdata.iris import to_iris

    iris_xr_cubes = to_iris(ncds_fromxr)

    # Unfortunately, cube order is not guaranteed to be stable.
    iris_xr_cubes = namesort_cubes(iris_xr_cubes)

    # Because Iris awkwardly special-cases the units of variables with flag_values etc,
    # we need to also ignore 'no_units' in the re-loaded data.
    remove_cube_nounits(iris_xr_cubes)

    # N.B. Conventions are not preserved from original, since Iris re-writes them.
    # So for now, just remove them all
    for cube in iris_cubes + iris_xr_cubes:
        cube.attributes.pop("Conventions", None)

    # correct for xarray handling of time units.
    # not entirely clear yet where this happens,
    #  - probably in the loading phase, which therefore can maybe be controlled ??
    _UNIFY_TIME_UNITS = True
    # _UNIFY_TIME_UNITS = False
    if _UNIFY_TIME_UNITS:
        for iris_cube, iris_xr_cube in zip(iris_cubes, iris_xr_cubes):
            for xr_coord in iris_xr_cube.coords():
                iris_coords = iris_cube.coords(xr_coord.name())
                if len(iris_coords) != 1:
                    # Coords don't match, which is nasty!
                    # Just skip out + let the test fail
                    break
                (iris_coord,) = iris_coords
                # Detecting differently constructed time units is awkward,
                # because you can have unit1==unit2, but still str(unit1) != str(unit2)
                xr_ut, iris_ut = (co.units for co in (xr_coord, iris_coord))
                xr_utstr, iris_utstr = (str(ut) for ut in (xr_ut, iris_ut))
                if (
                    iris_coord == xr_coord
                    and iris_coord.units == xr_coord.units
                    and xr_utstr != iris_utstr
                ):
                    # Fix "equal" units to have identical **string representations**.
                    xr_coord.units = iris_ut

    iris_cubes = [nanmask_cube(cube) for cube in iris_cubes]
    iris_xr_cubes = [nanmask_cube(cube) for cube in iris_xr_cubes]

    results = [
        (c1.name(), cubes_equal__corrected(c1, c2))
        for c1, c2 in zip(iris_cubes, iris_xr_cubes)
    ]
    expected = [(cube.name(), True) for cube in iris_cubes]
    result = results == expected

    if not result:
        # FOR NOW: compare with experimental ncdata comparison.
        # I know this is a bit circular, but it is useful for debugging, for now ...
        result = dataset_differences(
            from_iris(iris_cubes), from_iris(iris_xr_cubes)
        )
        assert result == []


# N.B. FOR NOW skip this test entirely.
# There are lots of problems here, mostly caused by dimension naming.
# It's not currently clear if we *can* find a simple view of what "ought" to work here,
# or whether we should be attempting to test this at all.
@pytest.mark.skip("Roundtrip testing xarray-iris-xarray : currently disabled.")
def test_roundtrip_xix(
    standard_testcase, use_irislock, adjust_chunks, tmp_path
):
    source_filepath = standard_testcase.filepath

    # Skip some cases
    excluded_casename_keys = [
        # these won't load (either Iris or Xarray)
        "ds_Empty",
        "ds__singleattr",
        "ds__dimonly",
        # these ones have a type of bounds variable that Xarray can't handle :
        #     xarray.core.variable.MissingDimensionsError: 'time_bnd' has more than
        #     1-dimension and the same name as one of its dimensions
        #     ('time', 'time_bnd').
        #     xarray disallows such variables because they conflict with the
        #     coordinates used to label dimensions.
        # E.G.
        #   (dims include 'time', 'time_bnds')
        #   float time(time) ;
        #       time:bounds = 'time_bnds'
        #   float time_bnds(time, time_bnds) ;
        "label_and_climate__small_FC_167",
        "rotated__xyt__small_rotPole_precipitation",
        # This one fails to load in xarray, for somewhat unclear reasons
        #     NotImplementedError: Can not use auto rechunking with object dtype.
        #     We are unable to estimate the size in bytes of object data
        "unstructured_grid__lfric_surface_mean",
        # Iris loses the name of the unstructured dimension, causing multiple problems
        "unstructured_grid__data_C4",
    ]
    if any(key in standard_testcase.name for key in excluded_casename_keys):
        pytest.skip("excluded testcase")

    # _Debug = True
    _Debug = False
    if _Debug:
        print(f"\ntestcase: {standard_testcase.name}")
        print("spec =")
        print(standard_testcase.spec)
        print("\nncdump =")
        txt = check_output([f"ncdump {source_filepath}"], shell=True).decode()
        print(txt)

    # Load the testcase with Xarray.
    xrds = xarray.open_dataset(source_filepath, chunks="auto")

    # Convert to iris cubes, and back again.

    # TODO: reinstate un-staged xarray-->iris conversion.
    # but for now, pull out intermediate stage for debugging
    from ncdata.iris import to_iris

    ncds = from_xarray(xrds)
    iris_cubes = to_iris(ncds)

    # iris_cubes = cubes_from_xarray(xrds)
    xrds_iris = cubes_to_xarray(iris_cubes)

    # # Check equivalence
    # result = xrds_iris.identical(xrds)
    #
    # if not result:
    #     # FOR NOW: compare with experimental ncdata comparison.
    #     # I know this is a bit circular, but it is useful for debugging, for now ...

    # Check equivalence (a less old but still obsolete way, via "from_xarray")
    # ncds_xr = from_xarray(xrds)
    # ncds_xr_iris = from_xarray(xrds_iris)

    # Check equivalence of Xarray datasets : going via file
    # This is a useful stopgap as it is flexible + diffs are explainable
    # TODO: replace by fixes **to xarray datasets**, and ds.identical()
    fp_xr = tmp_path / "tmp_xr.nc"
    fp_xr_iris = tmp_path / "tmp_xr_iris.nc"
    xrds.to_netcdf(fp_xr)
    xrds_iris.to_netcdf(fp_xr_iris)
    ncds_xr = from_nc4(fp_xr)
    ncds_xr_iris = from_nc4(fp_xr_iris)

    # Sanitise results in various ways to avoid common encoding disagreements
    # (for now, at least)
    for ds in (ncds_xr, ncds_xr_iris):
        ds.avals.pop("Conventions", None)
        for var in ds.variables.values():
            if var.name == "data":
                pass
            if "grid_mapping_name" in var.avals:
                # Fix datatypes of grid-mapping variables.
                # Iris creates all these as floats, but int is more common in inputs.
                FIXED_GRIDMAPPING_DTYPE = np.dtype("i4")
                var.dtype = FIXED_GRIDMAPPING_DTYPE
                var.data = var.data.astype(FIXED_GRIDMAPPING_DTYPE)
                # Remove any coordinates of grid-mapping variables : Xarray adds these.
                var.avals.pop("coordinates", None)
            fv = var.avals.pop("_FillValue", None)
            if fv is None:
                dt = var.data.dtype
                nn = f"{dt.kind}{dt.itemsize}"
                fv = netCDF4.default_fillvals[nn]
            else:
                fv = fv.as_python_value()
            data = da.ma.getdata(var.data)
            mask = da.ma.getmaskarray(var.data)
            if data.dtype.kind == "f":
                mask |= da.isnan(data)
            mask |= data == fv
            data = da.ma.masked_array(data, mask=mask)
            var.data = data
            if "calendar" in var.avals:
                if var.avals["calendar"] == "gregorian":
                    # the calendar name 'gregorian' is now deprecated, so Iris replaces it.
                    var.avals["calendar"] = "standard"

    result = dataset_differences(
        ncds_xr, ncds_xr_iris
    )  # , check_var_data=False)
    assert result == []

    # TODO:  check equivalence, in Xarray terms
    # xr_result = xrds_iris.equals(xrds)
    # ncd_result = dataset_differences(
    #     ncds_xr, ncds_xr_iris
    # )  # , check_var_data=False)
    # print("\nDATASET COMPARE RESULTS:\n" + "\n".join(ncd_result))
    # assert (xr_result, ncd_result) == (True, [])
