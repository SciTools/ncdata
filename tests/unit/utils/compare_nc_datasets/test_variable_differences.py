import dask.array as da
import numpy as np
import pytest

from ncdata import NcVariable
from ncdata.utils import variable_differences

_DEBUG_RESULTS = True
# _DEBUG_RESULTS = True


def check(results, expected):
    if _DEBUG_RESULTS:
        print("\nResult messages:")
        for msg in results:
            print("  ", msg)
    assert results == expected


class TestSimpleProperties:
    @pytest.fixture(autouse=True)
    def _vars_data(self):
        self.var1, self.var2 = [
            NcVariable("v1", ("y", "x"), data=np.zeros((2, 3)))
            for _ in range(2)
        ]

    def test_var_names(self):
        self.var2.name = "q"

        errs = variable_differences(self.var1, self.var2)
        expected = ['Variable "v1 / q" names differ : ' "'v1' != 'q'"]
        check(errs, expected)

    def test_var_dims__reorder(self):
        # N.B. here we check behaviour of the DIMENSIONS order control, but this does
        # not apply to dimensions order in a variable,which is *always* significant.
        self.var2.dimensions = self.var2.dimensions[::-1]
        # N.B. the data shape doesn't now correspond, but that won't matter as, with
        #  mismatched dimensions, the data won't be checked.

        errs = variable_differences(self.var1, self.var2)

        expected = [
            'Variable "v1" dimensions differ : ' "('y', 'x') != ('x', 'y')"
        ]
        check(errs, expected)

    def test_var_dims__extra_or_missing(self):
        # N.B. here we check for DIMENSIONS order check control.
        self.var2.dimensions = self.var2.dimensions[:-1]
        # N.B. the data shape doesn't now correspond, but that won't matter as, with
        #  mismatched dimensions, the data won't be checked.

        errs = variable_differences(self.var1, self.var2)

        expected = ["Variable \"v1\" dimensions differ : ('y', 'x') != ('y',)"]
        check(errs, expected)


class TestDtypes:
    # Note: testing variable comparison via the 'main' public API instead of
    #  via 'variable_differences'.  This makes sense because it is only called
    #  in one way, from one place.
    @pytest.fixture(autouse=True)
    def _vars_data(self):
        self.var1, self.var2 = [
            NcVariable("v1", ("x"), data=np.zeros(3)) for _ in range(2)
        ]

    def test_numbers_v_strings(self):
        # Set a different dtype
        # NB this is different from the actual data array, but that doesn't
        #  matter, as it won't attempt to compare strings with numbers
        self.var2.dtype = np.dtype("S5")

        # Test the comparison
        errs = variable_differences(self.var1, self.var2)
        expected = [
            'Variable "v1" datatypes differ : '
            "dtype('float64') != dtype('S5')"
        ]
        check(errs, expected)

    @pytest.mark.parametrize("equaldata", [False, True])
    def test_ints_v_floats(self, equaldata):
        # In this case, there is also a data comparison to check.
        v1 = self.var2

        new_dtype = np.dtype(np.int32)
        v1.data = v1.data.astype(new_dtype)
        if not equaldata:
            v1.data.flat[0] += 1
        v1.dtype = new_dtype

        # Test the comparison
        errs = variable_differences(self.var1, self.var2)

        expected = [
            'Variable "v1" datatypes differ : '
            "dtype('float64') != dtype('int32')"
        ]
        if not equaldata:
            expected.append(
                'Variable "v1" data contents differ, at 1 points: '
                "@INDICES[(0,)] : LHS=[0.0], RHS=[1]"
            )
        check(errs, expected)

    @pytest.mark.parametrize("equaldata", [False, True])
    def test_wordlengths(self, equaldata):
        # Test floats with wordlength difference -- assume ints are the same
        # In this case, there is also a data comparison to check.
        v1 = self.var2

        new_dtype = np.dtype(np.float32)
        v1.data = v1.data.astype(new_dtype)
        if not equaldata:
            v1.data.flat[0] += 1
        v1.dtype = new_dtype

        # Test the comparison
        errs = variable_differences(self.var1, self.var2)

        expected = [
            'Variable "v1" datatypes differ : '
            "dtype('float64') != dtype('float32')"
        ]
        if not equaldata:
            expected.append(
                'Variable "v1" data contents differ, at 1 points: '
                "@INDICES[(0,)] : LHS=[0.0], RHS=[1.0]"
            )
        check(errs, expected)

    @pytest.mark.parametrize("equaldata", [False, True])
    def test_signed_unsigned(self, equaldata):
        # Test floats with wordlength difference -- assume ints are the same
        # In this case, there is also a data comparison to check.
        new_dtype = np.dtype(np.int64)
        v0 = self.var1
        v0.data = v0.data.astype(new_dtype)
        v0.dtype = new_dtype

        new_dtype = np.dtype(np.uint64)
        v1 = self.var2
        v1.data = v1.data.astype(new_dtype)
        if not equaldata:
            v1.data.flat[0] += 1
        v1.dtype = new_dtype

        # Test the comparison
        errs = variable_differences(self.var1, self.var2)

        expected = [
            'Variable "v1" datatypes differ : '
            "dtype('int64') != dtype('uint64')"
        ]
        if not equaldata:
            expected.append(
                'Variable "v1" data contents differ, at 1 points: '
                "@INDICES[(0,)] : LHS=[0], RHS=[1]"
            )
        check(errs, expected)

    @pytest.mark.parametrize("given", ["nodata", "data", "dtype"])
    def test_nodata_nodtype(self, given):
        # Check that we can correctly compare a variable with NO specified data or dtype,
        #  with one that may have either.
        # N.B. this omits comparing 2 variables with dtype only. See following test,
        #  'test_nodata_withdtype` for that.
        v1 = NcVariable("x")

        kwargs = {}
        if given == "data":
            kwargs["data"] = [1, 2]
            expected = [
                'Variable "x" shapes differ : None != (2,)',
                "Variable \"x\" datatypes differ : None != dtype('int64')",
            ]
        elif given == "dtype":
            kwargs["dtype"] = np.float32
            expected = [
                "Variable \"x\" datatypes differ : None != dtype('float32')"
            ]
        elif given == "nodata":
            expected = []
        else:
            raise ValueError(f"unrecognised 'given' param : {given!s}")

        v2 = NcVariable("x", **kwargs)
        errs = variable_differences(v1, v2)
        check(errs, expected)

    @pytest.mark.parametrize("equality", ["same", "different"])
    def test_nodata_withdtype(self, equality):
        # Check that we can correctly compare variables which have dtype but no data.
        # N.B. the other possibilities are all covered in the "nodata_nodtype" test.
        dtype = np.int16
        v1 = NcVariable("x", dtype=dtype)
        expected = []
        if equality == "different":
            dtype = np.float16
            expected = [
                "Variable \"x\" datatypes differ : dtype('int16') != dtype('float16')"
            ]

        v2 = NcVariable("x", dtype=dtype)
        errs = variable_differences(v1, v2)
        check(errs, expected)

    @pytest.mark.parametrize(
        "differs",
        ["shapes", "shapes_nodata", "dims", "dims_nodata", "dims_both_nodata"],
    )
    def test_differences_ignore_datacheck(self, differs):
        # Check that we can safely compare variables with different shapes or dims,
        #  even with non-broadcastable content, as this should prevent the data
        #  equivalence testing.
        # Effectively, this is also testing "safe_varshape"
        dims1 = ["x", "y"]
        dims2 = dims1
        shape1 = (2, 3)
        shape2 = shape1
        nodata = differs.endswith("nodata")
        if differs.startswith("dims"):
            dims2 = ["x"]
            expected = [
                "Variable \"x\" dimensions differ : ('x', 'y') != ('x',)"
            ]
            if nodata:
                # In this case, either one or both have no data.
                shape1 = None
                both_nodata = "both" in differs
                if both_nodata:
                    shape2 = None
                else:
                    # Only one has data --> triggers a shape warning too.
                    expected.append(
                        'Variable "x" shapes differ : None != (2, 3)'
                    )
        elif differs.startswith("shapes"):
            if nodata:
                # In this case one has data (and therefore a shape), and the other not.
                shape2 = None
                expected = ['Variable "x" shapes differ : (2, 3) != None']
            else:
                shape2 = (3, 4)
                expected = ['Variable "x" shapes differ : (2, 3) != (3, 4)']
        else:
            raise ValueError(f"unrecognised 'differs' param : {differs!s}")

        data1 = np.ones(shape1) if shape1 is not None else None
        data2 = np.ones(shape2) if shape2 is not None else None
        var1 = NcVariable("x", dimensions=dims1, data=data1)
        var2 = NcVariable("x", dimensions=dims2, data=data2)

        errs = variable_differences(var1, var2)
        check(errs, expected)


class TestDataCheck__controls:
    # Note: testing variable comparison via the 'main' public API instead of
    #  via 'variable_differences'.  This makes sense because it is only called
    #  in one way, from one place.
    @pytest.fixture(autouse=True)
    def _vars_data(self):
        self.var1, self.var2 = [
            NcVariable("v1", ("x"), data=np.arange(6.0).reshape((2, 3)))
            for _ in range(2)
        ]

    def test_no_values_check(self):
        self.var2.data += 1
        errs = variable_differences(self.var1, self.var2, check_var_data=False)
        check(errs, [])

    def test_print_bad_nprint(self):
        msg = "'show_n_diffs' must be >=1 : got 0."
        with pytest.raises(ValueError, match=msg):
            variable_differences(
                self.var1, self.var2, show_n_first_different=0
            )

    @pytest.mark.parametrize("ndiffs", [1, 2, 3])
    def test_ndiffs(self, ndiffs):
        self.var2.data.flat[1 : ndiffs + 1] += 1
        errs = variable_differences(self.var1, self.var2)
        detail = {
            1: "[(0, 1)] : LHS=[1.0], RHS=[2.0]",
            2: "[(0, 1), (0, 2)] : LHS=[1.0, 2.0], RHS=[2.0, 3.0]",
            3: (
                "[(0, 1), (0, 2), ...] : "
                "LHS=[1.0, 2.0, ...], RHS=[2.0, 3.0, ...]"
            ),
        }[ndiffs]
        expected = [
            f'Variable "v1" data contents differ, at {ndiffs} points: '
            f"@INDICES{detail}"
        ]
        check(errs, expected)

    @pytest.mark.parametrize("nprint", [1, 2, 3])
    def test_show_n_first_different(self, nprint):
        self.var2.data.flat[1:3] += 1
        errs = variable_differences(
            self.var1, self.var2, show_n_first_different=nprint
        )
        detail = {
            1: "[(0, 1), ...] : LHS=[1.0, ...], RHS=[2.0, ...]",
            2: "[(0, 1), (0, 2)] : LHS=[1.0, 2.0], RHS=[2.0, 3.0]",
            3: "[(0, 1), (0, 2)] : LHS=[1.0, 2.0], RHS=[2.0, 3.0]",
        }[nprint]
        expected = [
            f'Variable "v1" data contents differ, at 2 points: '
            f"@INDICES{detail}"
        ]
        check(errs, expected)


class TestDataCheck__difference_reports:
    # Note: testing variable comparison via the 'main' public API instead of
    #  via 'variable_differences'.  This makes sense because it is only called
    #  in one way, from one place.
    @pytest.fixture(autouse=True)
    def _vars_data(self):
        self.var1, self.var2 = [
            NcVariable("v1", ("x"), data=np.arange(4.0)) for _ in range(2)
        ]

    @pytest.mark.parametrize("datavalues", ["same", "different"])
    @pytest.mark.parametrize("masks", ["onemasked", "bothmasked"])
    def test_masked(self, datavalues, masks):
        different = datavalues == "different"
        bothmasked = masks == "bothmasked"
        testvar = self.var2
        testvar.data = np.ma.masked_array(testvar.data)
        if different:
            testvar.data[1:2] += 1
        testvar.data[1:2] = np.ma.masked
        if bothmasked:
            self.var1.data = np.ma.masked_array(self.var1.data)
            self.var1.data[1:2] = np.ma.masked
        errs = variable_differences(self.var1, self.var2)
        if bothmasked:
            expected = []
        else:
            expected = [
                'Variable "v1" data contents differ, at 1 points: '
                "@INDICES[(1,)] : LHS=[1.0], RHS=[masked]"
            ]
        check(errs, expected)

    @pytest.mark.parametrize("nans", ["onenans", "bothnans"])
    def test_nans(self, nans):
        bothnans = nans == "bothnans"
        self.var2.data[1:2] = np.nan
        if bothnans:
            self.var1.data[1:2] = np.nan
        errs = variable_differences(self.var1, self.var2)
        if bothnans:
            expected = []
        else:
            expected = [
                'Variable "v1" data contents differ, at 1 points: '
                "@INDICES[(1,)] : LHS=[1.0], RHS=[nan]"
            ]
        check(errs, expected)

    def test_scalar(self):
        # Check how a difference of scalar arrays is reported
        for value, var in enumerate([self.var1, self.var2]):
            var.dimensions = ()
            var.data = np.array(value, dtype=var.dtype)
        errs = variable_differences(self.var1, self.var2)
        expected = [
            'Variable "v1" data contents differ, at 1 points: '
            "@INDICES[(0,)] : LHS=[0.0], RHS=[1.0]"
        ]
        check(errs, expected)

    @pytest.mark.parametrize(
        "argtypes", ["real_real", "real_lazy", "lazy_lazy"]
    )
    def test_real_and_lazy(self, argtypes):
        type1, type2 = argtypes[:4], argtypes[-4:]
        # fix the testvar to create a difference
        self.var2.data[1:2] += 1
        # setup vars with lazy/real data arrays
        for arraytype, var in zip([type1, type2], [self.var1, self.var2]):
            if arraytype == "lazy":
                var.data = da.from_array(var.data, chunks=-1)
        # compare + check results
        errs = variable_differences(self.var1, self.var2)
        # N.B. the result should be the same in all cases
        expected = [
            'Variable "v1" data contents differ, at 1 points: '
            "@INDICES[(1,)] : LHS=[1.0], RHS=[2.0]"
        ]
        check(errs, expected)

    @pytest.mark.parametrize(
        "ndiffs", [0, 1, 2], ids=["no_diffs", "one_diff", "two_diffs"]
    )
    def test_string_data(self, ndiffs):
        # FOR NOW test only with character arrays, encoded as expected ("S1" dtype)
        strings = ["one", "three", "", "seventeen"]
        str_len = max(len(x) for x in strings)
        chararray = np.zeros((4, str_len), dtype="S1")
        for ind, el in enumerate(strings):
            chararray[ind, 0 : len(el)] = list(el)
        self.var1, self.var2 = [
            NcVariable("vx", ("x"), data=chararray.copy()) for ind in range(2)
        ]

        if ndiffs > 0:
            self.var2.data[1, 1] = "X"  # modify one character
        if ndiffs > 1:
            self.var2.data[3, 3:] = ""  # (also) cut short this string

        # compare + check results
        errs = variable_differences(self.var1, self.var2)

        expected = []
        if ndiffs == 1:
            expected = [
                'Variable "vx" data contents differ, at 1 points: '
                "@INDICES[(1, 1)] : LHS=[b'h'], RHS=[b'X']"
            ]
        elif ndiffs == 2:
            expected = [
                'Variable "vx" data contents differ, at 7 points: '
                "@INDICES[(1, 1), (3, 3), ...] : "
                "LHS=[b'h', b'e', ...], RHS=[b'X', b'', ...]"
            ]
        check(errs, expected)
