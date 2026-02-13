"""
Test ncdata.iris by checking roundtrips for standard testcases.

Testcases start as netcdf files.
(1) check equivalence of cubes : iris.load(file) VS iris.load(ncdata(file))
(2) check equivalence of files : iris -> file VS iris->ncdata->file
"""

from subprocess import check_output

import iris
import pytest

from ncdata.netcdf4 import from_nc4, to_nc4
from ncdata.utils import dataset_differences

from tests.data_testcase_schemas import session_testdir, standard_testcase
from tests.integration.equivalence_testing_utils import (
    adjust_chunks,
    cubes_equal__corrected,
    set_tiny_chunks,
)

# Avoid complaints that imported fixtures are "unused"
standard_testcase, session_testdir, adjust_chunks

from ncdata.iris import from_iris, to_iris
from ncdata.threadlock_sharing import lockshare_context

# import iris.fileformats.netcdf._thread_safe_nc as ifnt

_FIX_LOCKS = True
# _FIX_LOCKS = False


@pytest.fixture(scope="session")
def use_irislock():
    if _FIX_LOCKS:
        with lockshare_context(iris=True):
            yield
    else:
        yield


# _USE_TINY_CHUNKS = True
_USE_TINY_CHUNKS = False
set_tiny_chunks(_USE_TINY_CHUNKS)


def test_load_direct_vs_viancdata(
    standard_testcase, use_irislock, adjust_chunks
):
    specific_excludes = [
        # This one has latitude points which exceed the valid_min/max attributes
        # The netcdf-variable-like transform in Nc4VariableLike._data_array don't
        # yet account for this.
        # TODO: fix in Nc4VariableLike, when we are sure of the interpretation
        "label_and_climate__small_FC_167_mon_19601101",
        # Some of the legacy UGRID unstructured files have incorrect encodings
        # which are currently causing loading errors in Iris since UGRID loading
        # became an always-on thing in v3.11
        "unstructured_grid__theta_nodal_xios",
        "ugrid__21_triangle_example",
    ]
    if any(
        name_fragment in standard_testcase.name
        for name_fragment in specific_excludes
    ):
        pytest.skip("excluded testcase")

    source_filepath = standard_testcase.filepath
    ncdata = from_nc4(source_filepath)

    # _Debug = True
    _Debug = False
    if _Debug:
        print(f"\ntestcase: {standard_testcase.name}")
        print("spec =")
        print(standard_testcase.spec)
        print("\nncdata =")
        print(ncdata)
        print("\nncdump =")
        txt = check_output([f"ncdump {source_filepath}"], shell=True).decode()
        print(txt)

    # Load the testcase with Iris.
    iris_cubes = iris.load(source_filepath)
    # Load same, via ncdata
    iris_ncdata_cubes = to_iris(ncdata)
    # Unfortunately, cube order is not guaranteed to be stable.
    iris_cubes, iris_ncdata_cubes = (
        sorted(cubes, key=lambda cube: cube.name())
        for cubes in (iris_cubes, iris_ncdata_cubes)
    )

    results = [
        (c1.name(), cubes_equal__corrected(c1, c2))
        for c1, c2 in zip(iris_cubes, iris_ncdata_cubes)
    ]
    expected = [(cube.name(), True) for cube in iris_cubes]
    result = results == expected

    if not result:
        # FOR NOW: compare with experimental ncdata comparison.
        # I know this is a bit circular, but it is useful for debugging, for now ...
        result = dataset_differences(
            from_iris(iris_cubes), from_iris(iris_ncdata_cubes)
        )
        assert result == []

    # assert iris_cubes == iris_ncdata_cubes
    assert result


def test_save_direct_vs_viancdata(standard_testcase, tmp_path):
    specific_excludes = [
        # Generally not-working for saves
        "ds_Empty",
        "ds__singleattr",
        "ds__dimonly",
        # Some of the legacy UGRID unstructured files have incorrect encodings
        # which are currently causing loading errors in Iris since UGRID loading
        # became an always-on thing in v3.11
        "unstructured_grid__theta_nodal_xios",
        "unstructured_grid__mesh_C12",
        "ugrid__21_triangle_example",
    ]
    if any(
        name_fragment in standard_testcase.name
        for name_fragment in specific_excludes
    ):
        pytest.skip("excluded testcase")

    source_filepath = standard_testcase.filepath
    ncdata = from_nc4(source_filepath)

    # Load the testcase into Iris.
    iris_cubes = iris.load(source_filepath)

    # Re-save from iris
    temp_iris_savepath = tmp_path / "temp_save_iris.nc"
    iris.save(iris_cubes, temp_iris_savepath)
    # Save same, via ncdata
    temp_ncdata_savepath = tmp_path / "temp_save_iris_via_ncdata.nc"
    ncdata_ex_iris = from_iris(iris_cubes)
    to_nc4(ncdata_ex_iris, temp_ncdata_savepath)

    # _Debug = True
    _Debug = False
    if _Debug:
        print(f"\ntestcase: {standard_testcase.name}")
        print("spec =")
        print(standard_testcase.spec)
        print("\nncdata =")
        print(ncdata)
        print("\nncdump ORIGINAL TESTCASE SOURCEFILE =")
        txt = check_output([f"ncdump {source_filepath}"], shell=True).decode()
        print(txt)
        print("\nncdump DIRECT FROM IRIS =")
        txt = check_output(
            [f"ncdump {temp_iris_savepath}"], shell=True
        ).decode()
        print(txt)
        print("\nncdump VIA NCDATA =")
        txt = check_output(
            [f"ncdump {temp_ncdata_savepath}"], shell=True
        ).decode()
        print(txt)

    # Check equivalence
    results = dataset_differences(temp_iris_savepath, temp_ncdata_savepath)
    assert results == []
