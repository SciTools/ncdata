"""Tests for class :class:`ncdata.NcVariable`.

There is almost no actual behaviour, but we can test some constructor behaviours,
such as valid calling options and possibilities for data and dtype.
"""

import dask.array as da
import numpy as np
import pytest

from ncdata import NcVariable
from ncdata.utils import variable_differences


class Test_NcVariable__init__:
    def test_minimum_args(self):
        # We can create a variable with no args.
        name = "varname"
        sample = NcVariable("varname")
        # No data, no dtype.  Variables don't have 'shape' anyway
        assert sample.name is name
        assert sample.dimensions == ()
        assert sample.data is None
        assert sample.dtype is None
        assert sample.avals == {}

    def test_never_nameless(self):
        # Can't create a variable without a name.
        msg = "required positional argument: 'name'"
        with pytest.raises(TypeError, match=msg):
            _ = NcVariable()

    def test_allargs(self):
        # Constructor slots are not type-checked, if the right kind of arg. There is no enforced type checking, put whatever you like in the "slots".
        name = "data_name"
        # NOTE: for *no-copy* construction, args must be NameMap of correct item_type
        dims = [f"dimname_{i}" for i in range(3)]
        # note: variable dims and shape **don't have to agree**.
        data = np.array((3.0, 2, 1))
        attrs = {f"attr_{i}": i for i in range(6)}
        dtype = "f8"  # match the provided data for correct matching
        group = ""
        sample1 = NcVariable(
            name=name,
            dimensions=dims,
            data=data,
            dtype=data.dtype,
            attributes=attrs,
            group=group,
        )
        # Nothing is copied : all contents are simply references to the inputs
        assert sample1.name == name
        assert sample1.dimensions == tuple(dims)
        assert sample1.data is data
        assert sample1.dtype == np.dtype(dtype)
        assert sample1.avals == attrs
        assert sample1.group is group
        # Also check construction with arguments alone (no keywords).
        sample2 = NcVariable(name, dims, data, dtype, attrs, group)
        # result is new object, but all contents are references identical to the other
        assert sample2 is not sample1
        for name in (
            "name",
            "dimensions",
            "data",
            "dtype",
            "attributes",
            "group",
        ):
            assert np.all(getattr(sample2, name) == getattr(sample1, name))

    @pytest.mark.parametrize("withdata_or_not", ["withdata", "nodata"])
    def test_valid_dtype__with_and_without_data(self, withdata_or_not):
        test_dtype = "i2"
        with_data = withdata_or_not == "withdata"
        data = np.arange(2.0) if with_data else None
        if with_data:
            # with provided data, dtype is *ignored*
            var = NcVariable("x", data=data, dtype=test_dtype)
            assert var.dtype == data.dtype
        else:
            # with no data, provided dtype is respected
            var = NcVariable("x", data=data, dtype=test_dtype)
            assert var.dtype == np.dtype(test_dtype)

    @pytest.mark.parametrize("withdata_or_not", ["withdata", "nodata"])
    def test_invalid_dtype__with_and_without_data(self, withdata_or_not):
        test_dtype_nonfunctional = "invalid_dtype"
        with_data = withdata_or_not == "withdata"
        data = np.arange(2.0) if with_data else None
        if with_data:
            # with provided data. invalid dtype is still ignored
            var = NcVariable("x", data=data, dtype=test_dtype_nonfunctional)
            assert var.dtype == data.dtype
        else:
            # with no data, invalid dtype causes error
            with pytest.raises(TypeError, match="data type.*not understood"):
                _ = NcVariable("x", data=data, dtype=test_dtype_nonfunctional)

    @pytest.mark.parametrize(
        # All of these cases should deliver data wrapped as np.asanyarray()
        "data",
        [
            12,
            3.4,
            "a",
            "bcd",
            [1, 2],
            [(3.4, 4), (5, 6)],
            ["xxx", "yyy", "zzz"],
        ],
    )
    def test_data__pythontypes(self, data):
        var = NcVariable("x", data=data, dtype="ineffective")
        expect_array = np.asanyarray(data)
        assert isinstance(var.data, np.ndarray)
        assert var.dtype == expect_array.dtype
        assert np.all(var.data == expect_array)
        if isinstance(data, np.ndarray):
            # Actual array data is stored directly
            assert var.data is data

    def test_data__listsofstrings(self):
        # handling of a ragged string-length is a specific case worth exploring
        data = [["one", "two", "three"], ["4", "5", "6"]]
        var = NcVariable("x", data=data)
        assert isinstance(var.data, np.ndarray)
        assert var.data.shape == (2, 3)
        assert var.dtype == var.data.dtype == np.dtype("U5")

    @pytest.mark.parametrize(
        "dtype", ["i4", "i8", "f4", "f8", "u1", "i1", "U4"]
    )
    def test_data__numpy(self, dtype):
        elem = "abc" if dtype.startswith("U") else 12
        dtype = np.dtype(dtype)
        arr = np.zeros((2, 3), dtype=dtype)
        arr[:] = elem
        # The variable should preserve the original of a contained numpy array, and adopt
        # it's dtype as it's own
        var = NcVariable("x", data=arr)
        assert var.data is arr  # not copied or re-formed
        assert var.dtype == dtype

    @pytest.mark.parametrize("dtype", ["i4", "i8", "f4", "f8", "u1", "i1"])
    def test_data__numpy_scalar(self, dtype):
        # As above, but for scalars.
        dtype = np.dtype(dtype)
        value = np.array(1.0, dtype=dtype)
        var = NcVariable("x", data=value)
        assert var.data is value
        assert var.dtype == dtype

    def test_data__dask(self):
        # Check that lazy data is also contained directly, without adjustment or
        # modification
        arr = da.ones((2, 3))
        var = NcVariable("x", data=arr)
        assert var.data is arr


class TestNcVariable__str__repr:
    def test_nameonly(self):
        var = NcVariable("varname")
        result = str(var)
        expected = "<NcVariable(<no-dtype>): varname()>"
        assert result == expected

    def test_dimsnoargs(self):
        var = NcVariable("var_w_dims", ["dim0", "dim1"])
        result = str(var)
        expected = "<NcVariable(<no-dtype>): var_w_dims(dim0, dim1)>"
        assert result == expected

    def test_oneattr(self):
        var = NcVariable("var_w_attrs", attributes={"a1": 1})
        result = str(var)
        expected = "\n".join(
            [
                "<NcVariable(<no-dtype>): var_w_attrs()",
                "    var_w_attrs:a1 = 1",
                ">",
            ]
        )
        assert result == expected

    def test_multiattrs(self):
        var = NcVariable(
            "var_multi",
            attributes={"a1": 1, "a2": ["one", "three"]},
        )
        result = str(var)
        expected = "\n".join(
            [
                "<NcVariable(<no-dtype>): var_multi()",
                "    var_multi:a1 = 1",
                "    var_multi:a2 = ['one', 'three']",
                ">",
            ]
        )
        assert result == expected

    def test_str(self):
        var = NcVariable(
            "var",
            dimensions=("x", "y"),
            data=np.ones((2, 3), dtype=np.float32),
        )
        result = str(var)
        expected = "<NcVariable(float32): var(x, y)>"
        assert result == expected

    def test_repr(self):
        var = NcVariable(
            "var",
            dimensions=("x", "y"),
            attributes={"a1": 1},
        )
        result = repr(var)
        import sys

        if sys.platform == "win32":
            objrep = f"0x{id(var):016X}"
        else:
            objrep = f"0x{id(var):012x}"
        expected = f"<ncdata._core.NcVariable object at {objrep}>"
        assert result == expected


class Test_NcVariable_copy:
    @staticmethod
    def check_var_iscopy(trial, reference):
        # capture expected copy equivalence check.
        # Which is : equal but distinct, containing same data array
        assert trial is not reference
        assert not variable_differences(trial, reference)
        assert trial.data is reference.data

    def test_minimal_nodata(self):
        var = NcVariable(name="x")
        result = var.copy()
        assert var.data is None
        self.check_var_iscopy(result, var)

    def test_populated(self):
        var = NcVariable(
            name="x",
            dimensions=(
                "y",
                "x",
            ),
            attributes={
                "a": 1,
                "v": np.array([1, 2, 3]),
                "s": "some characters",
            },
            data=np.array([[1.0, 2, 3], [11, 12, 13]]),
        )
        result = var.copy()
        self.check_var_iscopy(result, var)


class Test_NcVariable_eq:
    def test_equal(self, mocker):
        # Check that == calls variable_differences
        var1, var2 = [NcVariable(name=name) for name in ("x", "y")]
        called = mocker.patch("ncdata.utils.variable_differences")
        var1 == var2
        assert called.call_args_list == [mocker.call(var1, var2)]

    def test_self_equal(self, mocker):
        var = NcVariable(name="x")
        called = mocker.patch("ncdata.utils.variable_differences")
        assert var == var
        assert called.call_args_list == []

    def test_badtype_nonequal(self, mocker):
        var = NcVariable(name="x")
        called = mocker.patch("ncdata.utils.variable_differences")
        assert var != 1
        assert called.call_args_list == []
