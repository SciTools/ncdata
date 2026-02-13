"""
Tests for :mod:`ncdata.utils.save_errors`
"""

import re

import numpy as np
import pytest

from ncdata import NcData, NcDimension, NcVariable
from ncdata.utils import save_errors

from tests.unit.core.test_NcAttribute import attrvalue, datatype, structuretype

_ = datatype, structuretype


def _basic_testdata():
    ncdata = NcData(
        name="test_ds",
        dimensions=[
            NcDimension(
                "xxx", 2
            ),  # unused first dim, so rename doesn't break variable
            NcDimension("x", 3),
        ],
        variables=[
            NcVariable(
                name="vx1",
                dimensions=("x"),
                data=[1, 2, 3],
                attributes={"xx": 1},
            )
        ],
        groups=[NcData("inner")],
        attributes={"x": 1},
    )
    return ncdata


do_debug = True


# do_debug = False
def debug_errors(errors):
    if do_debug and errors:
        print("\n\nERROR RESULTS:")
        for msg in errors:
            print("  ", msg)


class TestSaveErrors_Okay:
    def test_noerrors_empty(self):
        ncdata = NcData()
        assert save_errors(ncdata) == []

    def test_noerrors_basic(self):
        ncdata = _basic_testdata()
        assert save_errors(ncdata) == []


class TestSaveErrors_Names:
    @pytest.fixture(params=["attribute", "dimension", "variable", "group"])
    def component(self, request):
        return request.param

    @staticmethod
    def component_propname(component: str):
        """Return the property name corresponding to the component type."""
        if component == "attribute":
            # Access the NameMap of NcAttributes (not the .attributes values map)
            propname = "attributes"
        else:
            propname = component + "s"
        return propname

    @pytest.mark.parametrize(
        "badnametype", ["None", "empty", "number", "object"]
    )
    def test_bad_name_type(self, badnametype, component):
        bad_name = {
            "None": None,
            "empty": "",
            "number": 3,
            "object": ("x", 2),
        }[badnametype]
        ncdata = _basic_testdata()
        elements = getattr(ncdata, self.component_propname(component))
        element = list(elements.values())[0]
        name = element.name
        elements.pop(name)
        element.name = bad_name
        elements.add(element)

        errors = save_errors(ncdata)
        debug_errors(errors)

        assert len(errors) == 1
        assert re.search(f"{component}.* invalid netCDF name", errors[0])

    def test_bad_name_string(self, component):
        # Basically, only "/" is banned, at present
        ncdata = _basic_testdata()
        elements = getattr(ncdata, component + "s")
        component_name = list(elements.keys())[0]
        elements.rename(component_name, "qq/q")

        errors = save_errors(ncdata)
        debug_errors(errors)

        assert len(errors) == 1
        assert re.search(f"{component}.* invalid netCDF name", errors[0])

    def test_key_name_mismatch(self, component):
        ncdata = _basic_testdata()
        elements = getattr(ncdata, self.component_propname(component))
        key, element = list(elements.items())[0]
        element.name = "qqq"

        errors = save_errors(ncdata)
        debug_errors(errors)

        assert len(errors) == 1
        msg = (
            f"{component} element {key!r} has a different element.name : 'qqq'"
        )
        assert re.search(msg, errors[0])


class TestSaveErrors_Attributes:
    def test_valid_datatypes(self, datatype, structuretype):
        # Check that all expected types + structures of attribute are accepted
        if "none" in datatype or "custom" in datatype:
            # These produce "unsaveable datatype" errors.
            pytest.skip("invalid dtype fails")
        value = attrvalue(datatype, structuretype)
        ncdata = NcData(attributes={"x": value})
        errors = save_errors(ncdata)
        assert errors == []

    @pytest.mark.parametrize(
        "context",
        [
            "named",
            "unnamed",
            "group_in_named",
            "group_in_unnamed",
            "group_of_group",
        ],
    )
    def test_bad_dataset_attribute(self, context):
        # NOTE: using this to test all the Dataset/Group naming constructions
        ncdata = _basic_testdata()
        ncdata.avals["q"] = None
        if "group" in context:
            ncdata = NcData(name="top", groups=[ncdata])
            if context == "group_of_group":
                ncdata.name = "middle"
                ncdata = NcData(name="top", groups=[ncdata])
        if "unnamed" in context:
            # Remove name of top-level dataset (only -- others parts must have names !)
            ncdata.name = None

        errors = save_errors(ncdata)
        debug_errors(errors)

        expected_id_str = {
            "named": "Dataset('test_ds')",
            "unnamed": "Dataset",
            "group_in_named": "Group 'top/test_ds'",
            "group_in_unnamed": "Group '/test_ds'",
            "group_of_group": "Group 'top/middle/test_ds'",
        }[context]
        msg = (
            f"{expected_id_str} attribute 'q' has a value which cannot "
            "be saved to netcdf : array(None, dtype=object) ::dtype=object."
        )
        assert errors == [msg]

    @pytest.mark.parametrize(
        "context",
        ["dataset", "group_in_named", "group_in_unnamed", "group_of_group"],
    )
    def test_bad_variable_attribute(self, context):
        ncdata = _basic_testdata()
        ncdata.variables["vx1"].set_attrval("q", None)
        if "group" in context:
            ncdata = NcData(name="top", groups=[ncdata])
            if context == "group_of_group":
                # push it down another level
                ncdata.name = "middle"
                ncdata = NcData("top", groups=[ncdata])
        if "unnamed" in context:
            ncdata.name = None
        print(ncdata)
        expected_id_str = {
            "dataset": "Variable 'vx1'",
            "group_in_named": "Variable 'top/test_ds/vx1'",
            "group_in_unnamed": "Variable '/test_ds/vx1'",
            "group_of_group": "Variable 'top/middle/test_ds/vx1'",
        }[context]

        errors = save_errors(ncdata)
        debug_errors(errors)

        msg = (
            f"{expected_id_str} attribute 'q' has a value which cannot be saved "
            "to netcdf : array(None, dtype=object) ::dtype=object."
        )
        assert errors == [msg]


class TestSaveErrors_Variables:
    def test_missing_data(self):
        ncdata = _basic_testdata()
        var = list(ncdata.variables.values())[0]
        var.data = None

        errors = save_errors(ncdata)

        msg = "Variable 'vx1' has no data array."
        assert errors == [msg]

    def test_invalid_dtype(self):
        ncdata = _basic_testdata()
        var = list(ncdata.variables.values())[0]
        arr = np.array([None for _ in var.data])
        var.data = arr
        var.dtype = arr.dtype

        errors = save_errors(ncdata)

        msg = "Variable 'vx1' has a dtype which cannot be saved to netcdf : dtype('O')."
        assert errors == [msg]

    def test_shape_mismatch(self):
        ncdata = _basic_testdata()
        var = list(ncdata.variables.values())[0]
        var.data = np.zeros((1, 2))

        errors = save_errors(ncdata)

        msg = (
            "Variable 'vx1' data shape = (1, 2), "
            "does not match that of its dimensions = (3,)."
        )
        assert errors == [msg]

    @pytest.mark.parametrize("extradims", ["v", "vw"])
    def test_missing_dims(self, extradims):
        ncdata = _basic_testdata()
        var = list(ncdata.variables.values())[0]
        extradims = list(char for char in extradims)
        var.dimensions = var.dimensions + tuple(extradims)

        errors = save_errors(ncdata)

        msg = (
            "Variable 'vx1' references dimensions which are not found in the "
            f"enclosing dataset : {extradims!r}"
        )
        assert errors == [msg]
