"""
Tests for :mod:`tests.unit.netcdf._compare_nc_files`
Split in two files ...
  * HERE: "additional" tests cover subsidiary routines and the main
    API usage modes.
  * ( ALSO: "mainfunctions" (q.v.) cover the core functionality
    -- which elements are compared and what errors this constructs. )
"""

import shutil
import warnings

import netCDF4 as nc
import numpy as np
import pytest

from ncdata.utils._compare_nc_datasets import (
    _attribute_differences,
    _namelist_differences,
    dataset_differences,
)

from tests.test_samplecode_cdlgen_comparablecdl import ncgen_from_cdl

_simple_cdl = """
netcdf test {
dimensions:
    x = 5 ;
variables:
    float x(x) ;
    int y(x) ;

data:
  x = 0.12, 1.23, 2.34, _, _ ;
  y = 1, 2, 3, _, _ ;
}
"""


class Test_namelist_differences:
    # Test subsidiary routine for checking a list of names
    def test_empty(self):
        errs = _namelist_differences([], [], "named-elements")
        assert errs == []

    def test_same(self):
        tst = ["a", "b"]
        errs = _namelist_differences(tst, tst, "named-elements")
        assert errs == []

    def test_diff(self):
        errs = _namelist_differences(["a"], [], "named-elements")
        assert errs == ["named-elements do not match: ['a'] != []"]

    def test_difforder(self):
        errs = _namelist_differences(["a", "b"], ["b", "a"], "named-elements")
        assert errs == [
            "named-elements do not match: ['a', 'b'] != ['b', 'a']"
        ]

    def test_difforder_tolerant_warns(self):
        with pytest.warns(
            UserWarning, match="Ignoring: named-elements do not match"
        ):
            errs = _namelist_differences(
                ["a", "b"],
                ["b", "a"],
                "named-elements",
                order_strict=False,
            )
        assert errs == []

    def test_difforder_tolerant_nowarn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            errs = _namelist_differences(
                ["a", "b"],
                ["b", "a"],
                "named-elements",
                order_strict=False,
                suppress_warnings=True,
            )
        assert errs == []


class Test_attribute_differences:
    def test_compare_attributes_namelists(self, mocker):
        # Check that it calls the generic _namelist_differences routine, passing all the
        # correct controls
        # NB make the compared object mimic nc Variables, not NcData
        attrnames_1 = ["a", "b"]
        attrnames_2 = ["c", "d"]
        obj1 = mocker.Mock(
            spec=nc.Variable, ncattrs=mocker.Mock(return_value=attrnames_1)
        )
        obj2 = mocker.Mock(
            spec=nc.Variable, ncattrs=mocker.Mock(return_value=attrnames_2)
        )
        elemname = "<elem_types>"
        order = mocker.sentinel.attrs_order
        suppress = mocker.sentinel.suppress_warnings
        tgt = "ncdata.utils._compare_nc_datasets._namelist_differences"
        patch_tgt = mocker.patch(tgt)
        _attribute_differences(
            obj1=obj1,
            obj2=obj2,
            elemname=elemname,
            attrs_order=order,
            suppress_warnings=suppress,
        )
        (one_call,) = patch_tgt.call_args_list
        assert one_call == mocker.call(
            attrnames_1,
            attrnames_2,
            "<elem_types> attribute lists",
            order_strict=order,
            suppress_warnings=suppress,
        )

    class Nc4ObjectWithAttrsMimic:
        def __init__(self, **attrs):
            self._attrs = attrs or {}

        def ncattrs(self):
            return self._attrs.keys()

        def getncattr(self, item):
            # Mimic how netCDF4 returns attribute values
            #  numeric arrays --> numpy ndarray
            #  numeric scalars --> Python int / float
            #  strings --> strings
            #  char arrays --> list of string
            # AND IN ALL CASES : [x] --> x
            value = np.array(self._attrs[item], ndmin=1)
            is_scalar = value.size == 1
            if value.dtype.kind in ("S", "U"):
                # strings are returned in lists, not arrays
                value = [str(x) for x in value]
            if is_scalar:
                # single values are always returned as scalars
                value = value[0]
            return value

    def test_compare_attributes_empty(self):
        # Test two objects with no attributes
        obj1 = self.Nc4ObjectWithAttrsMimic()
        obj2 = self.Nc4ObjectWithAttrsMimic()
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__allok(self):
        # Objects with matching attributes
        obj1 = self.Nc4ObjectWithAttrsMimic(a=1, b=2)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=1, b=2)
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__scalar_arrayof1(self):
        # Objects with matching attributes
        obj1 = self.Nc4ObjectWithAttrsMimic(a=1, b=2)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=1, b=[2])
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__data_mismatch(self):
        # Attributes of different value (but matching dtype)
        obj1 = self.Nc4ObjectWithAttrsMimic(a=1, b=2, c=3)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=1, b=-77, c=3)
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            '<object attributes> "b" attribute values differ : 2 != -77'
        ]

    def test_compare_attributes_values__dtype_mismatch__length(self):
        # Attributes of different dtypes, even though values ==
        obj1 = self.Nc4ObjectWithAttrsMimic(a=np.float32(0))
        obj2 = self.Nc4ObjectWithAttrsMimic(a=np.float64(0))
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute datatypes differ : '
                "dtype('float32') != dtype('float64')"
            )
        ]

    def test_compare_attributes_values__dtype_mismatch__signed_unsigned(self):
        # Attributes of different dtypes, even though values ==
        obj1 = self.Nc4ObjectWithAttrsMimic(a=np.uint32(0))
        obj2 = self.Nc4ObjectWithAttrsMimic(a=np.int32(0))
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute datatypes differ : '
                "dtype('uint32') != dtype('int32')"
            )
        ]

    def test_compare_attributes_values__dtype_mismatch__float_int(self):
        # Attributes of different dtypes, even though values ==
        obj1 = self.Nc4ObjectWithAttrsMimic(a=np.float32(0))
        obj2 = self.Nc4ObjectWithAttrsMimic(a=np.int32(0))
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute datatypes differ : '
                "dtype('float32') != dtype('int32')"
            )
        ]

    def test_compare_attributes_values__dtype_mismatch__numeric_string(self):
        # Attributes of different dtypes, even though values ==
        obj1 = self.Nc4ObjectWithAttrsMimic(a=np.float32(0))
        obj2 = self.Nc4ObjectWithAttrsMimic(a="this")
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute datatypes differ : '
                "dtype('float32') != <class 'str'>"
            )
        ]

    def test_compare_attributes_values__dtype_and_data_mismatch(self):
        # Attributes of different dtypes, but values !=
        obj1 = self.Nc4ObjectWithAttrsMimic(a=np.float32(0))
        obj2 = self.Nc4ObjectWithAttrsMimic(a=np.float64(1))
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            '<object attributes> "a" attribute datatypes differ : '
            "dtype('float32') != dtype('float64')"
        ]

    def test_compare_attributes_values__data_arrays_match(self):
        # Attributes of different value (but matching dtype)
        array = np.arange(3.0)
        obj1 = self.Nc4ObjectWithAttrsMimic(a=array)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=array)
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__data_arrays_dtype_mismatch(self):
        # Attributes of different value (but matching dtype)
        array = np.arange(3, dtype="f4")
        obj1 = self.Nc4ObjectWithAttrsMimic(a=array)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=array.astype("f8"))
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute datatypes differ : '
                "dtype('float32') != dtype('float64')"
            )
        ]

    def test_compare_attributes_values__data_arrays_shape_mismatch(self):
        # Attributes of different value (but matching dtype)
        array = np.arange(3)
        obj1 = self.Nc4ObjectWithAttrsMimic(a=array)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=array[:-1])
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute values differ : '
                "[0, 1, 2] != [0, 1]"
            )
        ]

    def test_compare_attributes_values__data_arrays_value_mismatch(self):
        # Attributes of different value (but matching dtype)
        array1 = np.array([1, 2, 3])
        array2 = np.array([1, 2, 777])
        obj1 = self.Nc4ObjectWithAttrsMimic(a=array1)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=array2)
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute values differ : '
                "[1, 2, 3] != [1, 2, 777]"
            )
        ]

    def test_compare_attributes_values__data_arrays_nans_match(self):
        # Attributes of different value (but matching dtype)
        array = np.array([1, np.nan, 3])
        obj1 = self.Nc4ObjectWithAttrsMimic(a=array)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=array)
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__data_arrays_nans_mismatch(self):
        # Attributes of different value (but matching dtype)
        array1 = np.array([1.0, 2.0, 3.0])
        array2 = np.array([1.0, np.nan, 3.0])
        obj1 = self.Nc4ObjectWithAttrsMimic(a=array1)
        obj2 = self.Nc4ObjectWithAttrsMimic(a=array2)
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            (
                '<object attributes> "a" attribute values differ : '
                "[1.0, 2.0, 3.0] != [1.0, nan, 3.0]"
            )
        ]

    def test_compare_attributes_values__string_nonstring(self):
        # Attributes of string and non-string types, since we handle that differently
        obj1 = self.Nc4ObjectWithAttrsMimic(a=1)
        obj2 = self.Nc4ObjectWithAttrsMimic(a="1")
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            '<object attributes> "a" attribute datatypes differ : '
            "dtype('int64') != <class 'str'>"
        ]

    def test_compare_attributes_values__string_match(self):
        # Attributes of string type (since netCDF4 returns char attributes as string)
        obj1 = self.Nc4ObjectWithAttrsMimic(S="this")
        obj2 = self.Nc4ObjectWithAttrsMimic(S="this")
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__string_mismatch(self):
        # Attributes of string type (since netCDF4 returns char attributes as string)
        obj1 = self.Nc4ObjectWithAttrsMimic(S="this")
        obj2 = self.Nc4ObjectWithAttrsMimic(S="that")
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            "<object attributes> \"S\" attribute values differ : 'this' != 'that'"
        ]

    def test_compare_attributes_values__string_array_match(self):
        # Attributes of string type (since netCDF4 returns char attributes as string)
        obj1 = self.Nc4ObjectWithAttrsMimic(S=["a", "b"])
        obj2 = self.Nc4ObjectWithAttrsMimic(S=["a", "b"])
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []

    def test_compare_attributes_values__string_array_mismatch(self):
        # Attributes of string type (since netCDF4 returns char attributes as string)
        obj1 = self.Nc4ObjectWithAttrsMimic(S=["a", "b"])
        obj2 = self.Nc4ObjectWithAttrsMimic(S=["a", "c"])
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == [
            '<object attributes> "S" attribute values differ : '
            "['a', 'b'] != ['a', 'c']"
        ]

    def test_compare_attributes__ncdata_string_scalar_array(self):
        # Attributes of string type (since netCDF4 returns char attributes as string)
        from ncdata import NcAttribute, NcData

        obj1 = NcData(attributes=[NcAttribute("x", ["string"])])
        obj2 = NcData(attributes=[NcAttribute("x", "string")])
        errs = _attribute_differences(obj1, obj2, "<object attributes>")
        assert errs == []


@pytest.fixture(autouse=True, scope="module")
def temp_ncfiles_dir(tmp_path_factory):
    tmp_dirpath = tmp_path_factory.mktemp("samencfiles")
    return tmp_dirpath


@pytest.fixture
def samefiles_filesonly(temp_ncfiles_dir):
    file1_nc = temp_ncfiles_dir / "tmp1.nc"
    file1_cdl = temp_ncfiles_dir / "tmp1.cdl"
    file2_nc = temp_ncfiles_dir / "tmp2.nc"
    file2_cdl = temp_ncfiles_dir / "tmp2.cdl"
    ncgen_from_cdl(cdl_str=_simple_cdl, nc_path=file1_nc, cdl_path=file1_cdl)
    ncgen_from_cdl(cdl_str=_simple_cdl, nc_path=file2_nc, cdl_path=file2_cdl)
    return file1_nc, file2_nc


@pytest.fixture(params=["InputsFile", "InputsNcdata"])
def sourcetype(request):
    return request.param


@pytest.fixture
def samefiles_bothtypes(samefiles_filesonly, sourcetype):
    source1, source2 = samefiles_filesonly
    if sourcetype == "InputsNcdata":
        from ncdata.netcdf4 import from_nc4

        source1, source2 = [from_nc4(src) for src in (source1, source2)]
    return source1, source2


class Test_compare_nc_files__api:
    def test_identical(self, samefiles_bothtypes):
        source1, source2 = samefiles_bothtypes
        result = dataset_differences(source1, source2)
        assert result == []

    def test_identical_stringpaths(self, samefiles_filesonly):
        source1, source2 = samefiles_filesonly
        result = dataset_differences(str(source1), str(source2))
        assert result == []

    def test_identical_datasets(self, samefiles_filesonly, sourcetype):
        source1, source2 = samefiles_filesonly
        ds1, ds2 = None, None
        try:
            ds1 = nc.Dataset(source1)
            ds2 = nc.Dataset(source2)
            result = dataset_differences(ds1, ds2)
            assert result == []
        finally:
            for ds in (ds1, ds2):
                if ds:
                    ds.close()

    def test_small_difference(
        self, samefiles_bothtypes, temp_ncfiles_dir, sourcetype
    ):
        source1, source2 = samefiles_bothtypes

        if sourcetype == "InputsFile":
            # Replace source2 with a modified, renamed file.
            source2 = temp_ncfiles_dir / "smalldiff.nc"
            shutil.copy(source1, source2)
            ds = nc.Dataset(source2, "r+")
            ds.extra_global_attr = 1
            ds.close()
        else:
            # Source1/2 are NcData : just modify source2
            source2.avals["extra_global_attr"] = 1

        result = dataset_differences(source1, source2)
        assert result == [
            "Dataset attribute lists do not match: [] != ['extra_global_attr']"
        ]

    def test_vardata_difference(
        self, samefiles_bothtypes, temp_ncfiles_dir, sourcetype
    ):
        # Temporary test for specific problem encountered with masked data differences.
        source1, source2 = samefiles_bothtypes
        ds = None
        try:
            if sourcetype == "InputsFile":
                # Make a modified, renamed file for the comparison.
                source2 = temp_ncfiles_dir / "vardiff.nc"
                shutil.copy(source1, source2)
                ds = nc.Dataset(source2, "r+")
                tgtx = ds.variables["x"]
                tgty = ds.variables["y"]
            else:
                # Source1/2 are NcData : just modify source2
                tgtx = source2.variables["x"].data
                tgty = source2.variables["y"].data

            tgtx[-1] = 101.23
            tgtx[0] = 102.34
            tgty[2:5] = [201, 202, 203]

        finally:
            if ds is not None:
                ds.close()

        result = dataset_differences(source1, source2)
        # N.B. ncdata comparison bypasses the masked+scaled view of data, hence the
        # message differs.  Could fix this?
        mask1 = "masked" if sourcetype == "InputsFile" else "9.96921e+36"
        mask2 = "masked" if sourcetype == "InputsFile" else "-2147483647"
        assert result == [
            (
                'Dataset variable "x" data contents differ, at 2 points: '
                f"@INDICES[(0,), (4,)] : LHS=[0.12, {mask1}], RHS=[102.34, 101.23]"
            ),
            (
                'Dataset variable "y" data contents differ, at 3 points: '
                "@INDICES[(2,), (3,), ...] : "
                f"LHS=[3, {mask2}, ...], RHS=[201, 202, ...]"
            ),
        ]
