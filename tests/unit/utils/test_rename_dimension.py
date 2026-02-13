"""Tests for :func:`ncdata.utils.rename_dimension`."""

import numpy as np
import pytest

from ncdata import NcData, NcDimension, NcVariable
from ncdata.utils import rename_dimension, save_errors


def make_saveable(
    ncdata: NcData, _outer_dims: dict[str, NcDimension] | None = None
):
    """Add missing dimensions + data, to make a sample NcData save-able.

    Create any missing variable dimensions (length==2).
    Add any missing variable data arrays.

    N.B. this might actually be a useful utility one day??
    """
    if _outer_dims is None:
        _outer_dims = {}

    outer_dim_names = [dim.name for dim in _outer_dims.values()]

    def getdim(dimname):
        """Fetch a known dimension by name.

        Check own, then "outer" ones, to correctly implement *scope masking*.
        """
        if dimname in ncdata.dimensions:
            # These ones must take precedence!
            result = ncdata.dimensions[dimname]
        else:
            # If not here, it should be in the 'outer' dims (else error).
            result = _outer_dims[dimname]
        return result

    for var in ncdata.variables.values():
        # Where variables reference dims which don't exist, create them (length=2).
        for dimname in var.dimensions:
            # Note: list of dims we check is *dynamic* (since we may add them)
            if dimname not in outer_dim_names + list(ncdata.dimensions.keys()):
                ncdata.dimensions.add(NcDimension(dimname, 2))

        # Where variables have no data, add some.
        if var.data is None:
            shape = tuple([getdim(dimname).size for dimname in var.dimensions])
            var.data = np.zeros(shape)

    # recurse through groups.
    all_dims = _outer_dims.copy()
    all_dims.update(ncdata.dimensions)
    for grp in ncdata.groups.values():
        make_saveable(grp, _outer_dims=all_dims)


class TestRenameDimension:
    """Tests for :func:`ncdata.utils.rename_dimension`."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ncdata = NcData(
            dimensions=[
                NcDimension("y", 2),
                NcDimension("x", 3),
            ],
            variables=[
                NcVariable("vx", ["x"], data=[0, 1, 2]),
                NcVariable("vy", ["y"], data=[11, 12]),
                NcVariable("vyx", ["y", "x"], data=np.zeros((2, 3))),
            ],
        )

    def test_basic(self):
        ncdata = self.ncdata
        xdim = ncdata.dimensions["x"]
        rename_dimension(ncdata, "x", "zz")
        assert ncdata.dimensions["zz"] is xdim
        assert ncdata.variables["vx"].dimensions == ("zz",)
        assert ncdata.variables["vy"].dimensions == ("y",)
        assert ncdata.variables["vyx"].dimensions == ("y", "zz")
        # Check that the result is still save-able.
        assert save_errors(ncdata) == []

    def test_name_collision_fail(self):
        ncdata = self.ncdata
        msg = "Cannot rename dimension 'x' to 'y', because a 'y' dimension already exists."
        with pytest.raises(ValueError, match=msg):
            rename_dimension(ncdata, "x", "y")

    @pytest.mark.parametrize(
        "innergroup", [False, True], ids=["maingroup", "innergroup"]
    )
    def test_name_collision_ingroup_fail(self, innergroup):
        ncdata = self.ncdata
        grp = NcData(name="inner", dimensions=[NcDimension("z", 2)])
        msg = "Cannot rename dimension 'x' to 'z', because a 'z' dimension already exists"
        if innergroup:
            grp = NcData(name="main", groups=[grp])
            msg += ' in group "/main/inner".'
        else:
            msg += ' in group "/inner".'
        ncdata.groups.add(grp)
        with pytest.raises(ValueError, match=msg):
            rename_dimension(ncdata, "x", "z")

    def test_name_ingroup_masked_nofail(self):
        ncdata = self.ncdata
        ncdata.groups.add(
            NcData(
                name="inner",
                dimensions=[NcDimension("x", 2), NcDimension("z", 2)],
            )
        )
        # No error in this case, because the "z" is inside a group with its own "x".
        rename_dimension(ncdata, "x", "z")

    @pytest.fixture()
    def group_example(self, setup):
        ncdata = self.ncdata.copy()
        ncdata.groups.addall(
            [
                NcData(
                    "a",
                    variables=[
                        NcVariable("ax", ["x"]),
                        NcVariable("aqxr", ["q", "x", "r"]),
                    ],
                ),
                NcData(
                    "b",
                    dimensions=[NcDimension("x", 20)],
                    variables=[NcVariable("bx", ["x"])],
                ),
            ]
        )
        yield ncdata

    def test_groups(self, group_example):
        ncdata = group_example
        rename_dimension(ncdata, "x", "zz")
        assert ncdata.groups["a"].variables["ax"].dimensions == ("zz",)
        assert ncdata.groups["a"].variables["aqxr"].dimensions == (
            "q",
            "zz",
            "r",
        )
        # This one doesn't get renamed: it is in a "scope hole" because the group
        #  defines its own "x" dimension, which takes precedence.
        assert ncdata.groups["b"].variables["bx"].dimensions == ("x",)

    def test_saveable(self, group_example):
        # Construct a complex example, make it saveable, and check that a renamed
        #  version is still saveable.
        ncdata = group_example.copy()

        make_saveable(ncdata)
        assert save_errors(ncdata) == []

        # now rename and try again.
        rename_dimension(ncdata, "x", "zz")
        assert save_errors(ncdata) == []
