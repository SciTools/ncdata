"""Tests for class :class:`ncdata.utils.index_by_dimension`.

This is the more direct indexer approach.
"""

import numpy as np
import pytest

from ncdata.utils import dataset_differences, index_by_dimensions

from . import (  # noqa: F401
    make_dims_testdata,
    make_dims_testdata_1d,
    x_slices,
    xy_slices,
)


def test_1d_index(x_slices):  # noqa: F811
    x_base = np.arange(5.0)
    data = make_dims_testdata_1d(x_base)
    result = index_by_dimensions(data, X=x_slices)

    # Make an expected result from an array indexed with *numpy*, to check equivalence.
    expect_x = x_base[x_slices]
    expect_data = make_dims_testdata_1d(expect_x)

    assert dataset_differences(result, expect_data) == []


def test_2d_index(xy_slices):  # noqa: F811
    x_inds, y_inds = xy_slices
    x_base, y_base = np.arange(5.0), np.arange(4.0)

    data = make_dims_testdata(x=x_base, y=y_base)
    result = index_by_dimensions(data, X=x_inds, Y=y_inds)

    expect_x = x_base[x_inds]
    expect_y = y_base[y_inds]
    expect_data = make_dims_testdata(expect_x, expect_y)

    assert dataset_differences(result, expect_data) == []


class TestKwargs:
    @pytest.fixture(autouse=True)
    def sample_3d(self):
        self.sample = make_dims_testdata(np.arange(5), np.arange(4))

    def test_bad_dimname_fail(self):
        msg = "Dimension 'Q' is not present in 'ncdata'."
        with pytest.raises(ValueError, match=msg):
            index_by_dimensions(self.sample, Q=7)


class TestIndexTypes:
    @pytest.fixture(autouse=True)
    def sample_3d(self):
        self.sample = make_dims_testdata(np.arange(5), np.arange(4))

    def test_slices(self):
        res1 = index_by_dimensions(
            self.sample, Z=slice(0, 1), Y=slice(None, 2), X=slice(1, None, 2)
        )
        assert np.array_equal(
            res1.variables["zyx_vals"].data,
            self.sample.variables["zyx_vals"].data[0:1, :2, 1::2],
        )

    def test_integer_array(self):
        res1 = index_by_dimensions(self.sample, X=[1, 3, 2, 1])
        assert np.array_equal(
            res1.variables["zyx_vals"].data,
            self.sample.variables["zyx_vals"].data[:, :, [1, 3, 2, 1]],
        )

    def test_boolean_array(self):
        res1 = index_by_dimensions(
            self.sample, X=[True, False, True, False, False]
        )
        res2 = index_by_dimensions(self.sample, X=[0, 2])
        assert dataset_differences(res1, res2) == []

    def test_multidim_fail(self):
        msg = "Key for dimension 'Y' is multi-dimensional: .* not supported."
        with pytest.raises(ValueError, match=msg):
            index_by_dimensions(self.sample, Y=[[1, 2], [3, 4]])

    def test_ellipsis_fail(self):
        msg = "Key for dimension 'Z' is Ellipsis.* not supported."
        with pytest.raises(ValueError, match=msg):
            index_by_dimensions(self.sample, Z=...)

    @pytest.mark.parametrize(
        "key", [np.newaxis, None], ids=["newaxis", "None"]
    )
    def test_newaxis_fail(self, key):
        msg = "Key for dimension 'Y' is np.newaxis / None.* not supported."
        with pytest.raises(ValueError, match=msg):
            index_by_dimensions(self.sample, Y=key)
