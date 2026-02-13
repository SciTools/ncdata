"""Tests for class :class:`ncdata.utils.Slicer`.

This is the "indirect" approach, with a Slicer object.
"""

import numpy as np
import pytest

from ncdata.utils import Slicer, dataset_differences

from . import (  # noqa: F401
    make_dims_testdata,
    make_dims_testdata_1d,
    x_slices,
    xy_slices,
)

#
# These tests apply the same test-cases as "test_index_by_dimensions".
#


def test_1d_index(x_slices):  # noqa: F811
    x_base = np.arange(5.0)
    data = make_dims_testdata_1d(x_base)

    print(x_slices)
    result = Slicer(data, "X")[x_slices]

    # Make an expected result from an array indexed with *numpy*, to check equivalence.
    expect_x = x_base[x_slices]
    expect_data = make_dims_testdata_1d(expect_x)

    assert dataset_differences(result, expect_data) == []


def test_2d_index(xy_slices):  # noqa: F811
    x_inds, y_inds = xy_slices
    x_base, y_base = np.arange(5.0), np.arange(4.0)

    data = make_dims_testdata(x=x_base, y=y_base)

    print(xy_slices)
    result = Slicer(data, "X", "Y")[x_inds, y_inds]

    expect_x = x_base[x_inds]
    expect_y = y_base[y_inds]
    expect_data = make_dims_testdata(expect_x, expect_y)

    assert dataset_differences(result, expect_data) == []


class TestSlicerDims:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.testdata = make_dims_testdata([1, 2], [3, 4, 5])

    def test_nodims(self):
        # Check indexing with no names dims.
        slicer1 = Slicer(self.testdata)
        assert slicer1.dim_names == ("Z", "Y", "X")

    def test_fewerdims(self):
        slicer1 = Slicer(self.testdata, "X", "Y")
        assert slicer1[1] == slicer1[1, :]

    def test_toomanydims_fail(self):
        # Check that too many dims causes an error.
        msg = (
            r"Too many index keys \(3\), for the indexing dimensions: "
            r"\('X', 'Y'\)\."
        )
        with pytest.raises(ValueError, match=msg):
            Slicer(self.testdata, "X", "Y")[0, 0, 0]
