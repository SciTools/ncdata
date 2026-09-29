"""
Tests for the test utility :func:`tests.version_majorminor`.
"""

import pytest

from tests import version_tuple


def test_empty():
    assert version_tuple("") == ()


def test_basic():
    assert version_tuple("1.2.3") == (1, 2, 3)


def test_one():
    assert version_tuple("1") == (1,)


@pytest.mark.parametrize("tail", ["alpha", ".alpha", "."])
@pytest.mark.parametrize("base", ["1", "1.2"])
def test_valid_tails(tail, base):
    assert version_tuple(base + tail) == version_tuple(base)


@pytest.mark.parametrize("version", ["a1.", "1a.2", "a1.2", "1.2a."])
def test_invalid_nonnumerics(version):
    with pytest.raises(ValueError, match="invalid"):
        version_tuple(version)


def test_interlength_compares():
    assert version_tuple("2") > version_tuple("1.99")
    assert version_tuple("1.2.3") > version_tuple("1.1")
    assert version_tuple("1.2.3") < version_tuple("1.3")
