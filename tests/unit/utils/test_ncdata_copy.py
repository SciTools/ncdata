"""Tests for :func:`ncdata.utils.ncdata_copy`.

This is generic utility function version of the copy operation.
"""

import numpy as np
import pytest

from ncdata import NameMap, NcAttribute, NcData, NcDimension, NcVariable
from ncdata.utils import dataset_differences, ncdata_copy


def _ncdata_duplicate_object(d1: NcData, d2: NcData):
    """Find and return the first known duplicate objects between two NcData."""
    dup = None
    for var1, var2 in zip(d1.variables.values(), d2.variables.values()):
        if var1 is var2:
            dup = var1
            break
    if not dup:
        for dim1, dim2 in zip(d1.dimensions.values(), d2.dimensions.values()):
            if dim1 is dim2:
                dup = dim1
                break
    if not dup:
        for attr1, attr2 in zip(d1.avals.values(), d2.avals.values()):
            if attr1 is attr2:
                dup = attr1
                break
    if not dup:
        for grp1, grp2 in zip(d1.groups.values(), d2.groups.values()):
            if grp1 is grp2:
                dup = grp1
                break
    return dup


def differences_or_duplicated_objects(original: NcData, duplicate: NcData):
    # Return difference messages or duplicate objects between two NcData
    results = dataset_differences(original, duplicate)
    if not results:
        results = _ncdata_duplicate_object(original, duplicate)
    return results


class Test:
    def test_empty(self):
        sample = NcData()
        result = ncdata_copy(sample)
        assert not differences_or_duplicated_objects(sample, result)

    @pytest.fixture()
    def sample(self):
        attrs = NameMap.from_items(
            [NcAttribute("q", 3)], item_type=NcAttribute
        )
        dims = NameMap.from_items([NcDimension("x", 3)], item_type=NcDimension)
        data_array = np.array([1, 2, 3])
        var = NcVariable(
            name="a", dimensions=("x"), data=data_array, attributes=attrs
        )
        sample = NcData(
            dimensions=dims,
            variables=[var],
            attributes=attrs,
            groups=[
                NcData("g1", dimensions=dims, variables=[var]),
                NcData("g2", dimensions=dims, variables=[var]),
            ],
        )
        assert sample.variables["a"].data is data_array
        assert sample.groups["g1"].variables["a"].data is data_array
        assert sample.groups["g2"].variables["a"].data is data_array
        return sample

    def test_general(self, sample):
        result = ncdata_copy(sample)
        assert not differences_or_duplicated_objects(sample, result)

    def test_sample_variable_data(self, sample):
        # Check that data arrays are *not* copied
        result = ncdata_copy(sample)

        data_arr = sample.variables["a"].data
        assert result.variables["a"].data is data_arr
        assert result.groups["g1"].variables["a"].data is data_arr
        assert result.groups["g2"].variables["a"].data is data_arr

    def test_sample_attribute_arraydata(self, sample):
        # Check that attributes arrays *are* copied
        arr1 = np.array([9.1, 7, 4])
        sample.set_attrval("extra", arr1)
        sva = sample.variables["a"]
        sva.set_attrval("xx2", arr1)

        result = ncdata_copy(sample)
        rva = result.variables["a"]

        assert (
            result.attributes["extra"].value
            is not sample.attributes["extra"].value
        ) and np.all(
            result.attributes["extra"].value
            == sample.attributes["extra"].value
        )

        assert (
            rva.attributes["xx2"].value is not sva.attributes["xx2"].value
        ) and np.all(
            rva.attributes["xx2"].value == sva.attributes["xx2"].value
        )
