"""Tests for class :class:`ncdata.NcData`.

There is almost no behaviour, but we can test some constructor usages.
"""

import pytest

from ncdata import NcData, NcDimension, NcVariable


class Test_NcData__init__:
    def test_noargs(self):
        sample = NcData()
        assert sample.name is None
        assert sample.dimensions == {}
        assert sample.variables == {}
        assert sample.groups == {}
        assert sample.avals == {}

    def test_allargs(self):
        name = "data_name"
        dims = [NcDimension(f"dimname_{i}", i) for i in range(3)]
        vars = [NcVariable(f"varname_{i}", (), int) for i in range(5)]
        attrs = {f"attrname_{i}": i for i in range(6)}
        grps = [NcData(f"groupname_{i}") for i in range(3)]
        sample1 = NcData(
            name=name,
            dimensions=dims,
            variables=vars,
            attributes=attrs,
            groups=grps,
        )
        # Nothing is copied : all contents are simply references to the inputs
        assert sample1.name is name
        assert list(sample1.dimensions.values()) == dims
        assert sample1.dimensions.item_type == NcDimension
        assert list(sample1.variables.values()) == vars
        assert sample1.variables.item_type == NcVariable
        assert list(sample1.groups.values()) == grps
        assert sample1.groups.item_type == NcData
        assert sample1.avals == attrs

        # Also check construction with arguments alone (no keywords).
        sample2 = NcData(name, dims, vars, attrs, grps)
        # result is a new object, but contents are references identical to the other
        assert sample2 is not sample1
        for name in ("dimensions", "variables", "attributes", "groups"):
            assert getattr(sample2, name) == getattr(sample1, name)


# Note: str() and repr() of NcData are too complex to test unit-wise.
# See integration tests for some sample results.


class Test_NcData_copy:
    # We only need to check that this calls "ncdata_copy", which is tested elsewhere.
    def test(self, mocker):
        mock_copied_ncdata = mocker.sentinel.copied_result
        mock_copycall = mocker.Mock(return_value=mock_copied_ncdata)
        mocker.patch("ncdata.utils.ncdata_copy", mock_copycall)
        ncdata = NcData()
        result = ncdata.copy()
        assert mock_copycall.call_args_list == [mocker.call(ncdata)]
        assert result == mock_copied_ncdata


class Test_NcData_eq:
    # check that == calls dataset_differences
    def test(self, mocker):
        ncdata1, ncdata2 = [NcData(name) for name in ("data1", "data2")]
        called = mocker.patch("ncdata.utils.dataset_differences")
        ncdata1 == ncdata2
        assert called.call_args_list == [mocker.call(ncdata1, ncdata2)]

    def test_self_equal(self, mocker):
        ds = NcData()
        called = mocker.patch("ncdata.utils.dataset_differences")
        assert ds == ds
        assert called.call_args_list == []

    def test_badtype_nonequal(self, mocker):
        ds = NcData()
        called = mocker.patch("ncdata.utils.dataset_differences")
        assert ds != 1
        assert called.call_args_list == []


class Test_NcVariable_slicer:
    # check that .slicer makes a slice.
    def test(self, mocker):
        ncdata1 = NcData()

        dim_args = (1, 2, 3)  # N.B. not actually acceptable for a real usage
        mock_return = mocker.sentinel.retval
        called = mocker.patch("ncdata.utils.Slicer", return_value=mock_return)
        result = ncdata1.slicer(*dim_args)

        assert called.call_args_list == [mocker.call(ncdata1, *dim_args)]
        assert result == mock_return


class Test_NcVariable_getitem:
    # check that data[*keys] calls data.slicer[*keys]
    @pytest.mark.parametrize(
        "multiple_indices", [False, True], ids=["oneIndex", "multiIndex"]
    )
    def test(self, mocker, multiple_indices):
        from ncdata.utils import Slicer

        ncdata1 = NcData()

        mock_slicer = mocker.MagicMock(spec=Slicer)
        called = mocker.patch(
            "ncdata._core.NcData.slicer", return_value=mock_slicer
        )
        if multiple_indices:
            # Index with multiple keys, which should be passed as a tuple
            # Slightly awkward code, because the "[*keys]" syntax is only valid from
            #  Python >=3.11
            result = ncdata1[1, 2, 3]
            dim_keys = (1, 2, 3)
        else:
            # Index with a single key: should be passed as-is
            dim_keys = slice(0, 3)
            result = ncdata1[dim_keys]

        assert called.call_args_list == [mocker.call()]
        assert mock_slicer.__getitem__.call_args_list == [
            mocker.call(dim_keys)
        ]
        assert result == mock_slicer[dim_keys]
