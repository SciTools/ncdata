"""Tests for class :class:`ncdata.NcDimension`."""

import numpy as np
import pytest

from ncdata import NcDimension


class Test_NcDimension__init__:
    def test_simple(self):
        name = "dimname"
        # Use an array object for test size, just so we can check "is".
        # N.B. assumes the constructor copies the reference - which could change.
        size = np.array(4)
        sample = NcDimension(name, size)
        # No data, no dtype.  Variables don't have 'shape' anyway
        assert sample.name is name
        assert sample.size is size

    @pytest.mark.parametrize("size", [0, 2])
    @pytest.mark.parametrize(
        "unlim",
        [0, 1, False, True, "x"],
        ids=["unlim_0", "unlim_1", "unlim_F", "unlim_T", "unlim_x"],
    )
    def test_unlimited(self, size, unlim):
        name = "dimname"
        sample = NcDimension(name, size, unlimited=unlim)
        expected_unlim = bool(unlim) or size == 0
        assert isinstance(sample.unlimited, bool)
        assert sample.unlimited == expected_unlim
        # assert sample.isunlimited() == sample.unlimited


class Test_NcDimension__str_repr:
    @pytest.fixture(params=(0, 23), autouse=True)
    def size(self, request):
        return request.param

    @pytest.fixture(params=("fixed", "unlimited"), autouse=True)
    def unlim_type(self, request):
        return request.param

    def test_repr_sized(self, size, unlim_type):
        kwargs = {}
        unlimited = unlim_type == "unlimited"
        if unlimited:
            kwargs["unlimited"] = True
        sample = NcDimension("this", size, **kwargs)
        result = repr(sample)
        expected = f"NcDimension('this', {size})"
        if size == 0 or unlimited:
            expected = expected[:-1] + ", unlimited=True)"
        assert result == expected

    def test_str_repr_same(self, size, unlim_type):
        # In all cases, str == repr
        sample = NcDimension("this", size)
        result = str(sample)
        assert result == repr(sample)


class Test_NcDimension_copy:
    @pytest.mark.parametrize("size", [0, 2])
    @pytest.mark.parametrize("unlim", [False, True])
    def test(self, size, unlim):
        sample = NcDimension("this", size, unlimited=unlim)
        result = sample.copy()
        assert result is not sample
        assert result.name == sample.name
        assert result.size == sample.size
        assert result.unlimited == sample.unlimited


class Test_NcDimension_eq:
    @pytest.fixture(params=["isunlimited", "notunlimited"])
    def unlimited(self, request):
        return request.param == "isunlimited"

    @pytest.fixture(params=[0, 3])
    def size(self, request):
        return request.param

    @pytest.fixture()
    def refdim(self, unlimited, size):
        return NcDimension(name="ref_name", size=size, unlimited=unlimited)

    def test_eq(self, refdim, size, unlimited):
        thisdim = NcDimension("ref_name", size=size, unlimited=unlimited)
        assert thisdim == refdim

    def test_noneq_name(self, refdim, size, unlimited):
        thisdim = NcDimension("other_name", size=size, unlimited=unlimited)
        assert thisdim != refdim

    def test_noneq_size(self, refdim, size, unlimited):
        if unlimited:
            pytest.skip("unsupported case")
        thisdim = NcDimension("ref_name", size=7)
        assert thisdim != refdim

    def test_noneq_unlim(self, refdim, size, unlimited):
        if size == 0:
            pytest.skip("unsupported case")
        thisdim = NcDimension("ref_name", size=size, unlimited=not unlimited)
        assert thisdim != refdim
