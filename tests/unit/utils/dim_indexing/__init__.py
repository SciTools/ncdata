"""Unit tests for :mod:`ncdata.utils._dim_indexing`."""

import numpy as np
import pytest

from ncdata import NcData, NcDimension, NcVariable


def make_dims_testdata(x, y):
    """
    Generate a test dataset that we can slice in various ways.

    If 'x' or 'y' are lists (~arrays) then make an X/Y dimension with these values,
    otherwise if they are a scalar value then not, mimicking numpy indexing away of a
    dimension.

    Also add a separate Z dimension, plus variables mapping every combination of
    X,Y,Z and no dimension.

    Also include some Groups and Attributes to ensure preservation of all structure and
    metadata.
    """
    data = NcData(
        dimensions=[NcDimension("Z", 2)],
        variables=[
            NcVariable("var_0", [], data=np.array(1.2)),
            NcVariable(
                "var_Z",
                ["Z"],
                data=np.linspace(1.3, 2.3, 2),
                attributes={"role": "Z dim var"},
            ),
        ],
        attributes={"global": "common_value"},
    )

    x, y = [np.asarray(key) for key in (x, y)]

    data.variables.add(
        NcVariable(
            "yvals",
            ["Y"] if y.ndim else [],
            data=2.2 * np.ones(y.shape),
            attributes={"role": "y values", "units": "ft"},
        )
    )
    if y.ndim:
        data.dimensions.add(NcDimension("Y", len(y)))
        ydims = ["Y"]
    else:
        ydims = []

    data.variables.add(
        NcVariable("Y", ydims, data=y, attributes={"role": "Y dim var"})
    )

    data.variables.add(
        NcVariable(
            "xvals",
            ["X"] if x.ndim else [],
            data=3.3 * np.ones(x.shape),
            attributes={"role": "x values", "units": "m"},
        )
    )
    if x.ndim:
        data.dimensions.add(NcDimension("X", len(x)))
        xdims = ["X"]
    else:
        xdims = []

    data.variables.add(
        NcVariable("X", xdims, data=x, attributes={"role": "X dim var"})
    )
    xydims = ["Z"] + (["Y"] if y.ndim else []) + (["X"] if x.ndim else [])
    shape = [2] + ([len(y)] if y.ndim else []) + ([len(x)] if x.ndim else [])
    data.variables.add(
        NcVariable(
            "zyx_vals",
            xydims,
            data=np.ones(shape),
            attributes={"role": "data"},
        )
    )

    return data


def make_dims_testdata_1d(x):
    """Create 1-D testdata."""
    data = make_dims_testdata(x=x, y=1)
    # remove 'Z' dimension altogether.
    del data.dimensions["Z"]
    # prune away any variables referring to 'Y' or 'Z'.
    for varname, var in list(data.variables.items()):
        if any(dim in var.dimensions for dim in ("Y", "Z")):
            data.variables.pop(varname)
    return data


class Slicekeys:  # noqa: D101
    def __getitem__(self, keys):  # noqa: D105
        return keys


_SLICEKEYS = Slicekeys()
_NUL = _SLICEKEYS[:]

# Define one-dimensional testcases, ultimately as the 'x_slices' fixture.
# NB these are used to test both 'index_by_dimensions' and 'Slicer'.
_1D_X_SLICE_OPTS = {
    "empty": _NUL,
    "single": 1,
    "range": _SLICEKEYS[2:4],
    "picked": [0, 1, 3],
    "openL": _SLICEKEYS[:3],
    "openR": _SLICEKEYS[2:],
    "minus": -2,
    "top2": _SLICEKEYS[-2:],
}
_1D_OPT_NAMES = list(_1D_X_SLICE_OPTS.keys())
_1D_OPT_VALUES = list(_1D_X_SLICE_OPTS.values())


@pytest.fixture(params=_1D_OPT_VALUES, ids=_1D_OPT_NAMES)
def x_slices(request):
    """Fixture yielding 1-D testcases."""
    return request.param


# Define two-dimensional testcases, ultimately as the 'xy_slices' fixture.
# NB these are used to test both 'index_by_dimensions' and 'Slicer'.
_2D_XY_SLICE_OPTS = {
    "empty": (_NUL, _NUL),
    "singleX": (1, _NUL),
    "singleY": (_NUL, 2),
    "rangeX": (_SLICEKEYS[2:4], _NUL),
    "rangeY": (_NUL, _SLICEKEYS[1:3]),
    "picked": ([0, 1, 3], _NUL),
    "openL": (_SLICEKEYS[:3], _NUL),
    "openR": (_SLICEKEYS[2:], _NUL),
    "dual": (_SLICEKEYS[1:3], _SLICEKEYS[2:]),
    "minus": (2, -2),
    "top2": (2, _SLICEKEYS[-2:]),
}
_2D_OPT_NAMES = list(_2D_XY_SLICE_OPTS.keys())
_2D_OPT_VALUES = list(_2D_XY_SLICE_OPTS.values())


@pytest.fixture(params=_2D_OPT_VALUES, ids=_2D_OPT_NAMES)
def xy_slices(request):
    """Fixture yielding 2-D testcases."""
    return request.param
