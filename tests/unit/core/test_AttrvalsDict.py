"""Tests for class :class:`ncdata._core.AttrvalsDict`."""

from copy import deepcopy

import numpy as np
import pytest

from ncdata import NcAttribute, NcVariable


class MixinEquivTest:
    """A helper for checking equivalence to the operation on ".attributes"."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.var = NcVariable("x", attributes={"a": 123, "b": "two"})
        self.vars_a_b = [self.var.copy() for _ in "ab"]


_testitem = {
    "int": 3,
    "ilist": [3, 4],
    "iarray": np.array([7, -3]),
    "iscalar": np.array([21]),
    "float": 7.2,
    "flist": [3.13, -2.7],
    "farray": np.array([4.129, 5.3]),
    "fscalar": np.array([12.9]),
    "string": "<this>",
    "object": {},
    "attrobj": NcAttribute("q", 3.12),
}

_alltype_opts = list(_testitem.keys())


@pytest.fixture(params=_alltype_opts)
def itemtype(request):
    yield request.param


@pytest.fixture(params=_alltype_opts)
def value_oftype(request):
    yield _testitem[request.param]


class Test_setitem(MixinEquivTest):
    """Testing equivalence to ".attributes" """

    def test_types(self, value_oftype):
        a, b = self.vars_a_b
        value = value_oftype

        # Test op
        a.avals["name"] = value

        # Equivalent op (working on .attributes)
        if isinstance(value_oftype, NcAttribute):
            # special case !
            b.set_attrval("name", value.as_python_value())
        else:
            b.set_attrval("name", value)

        # Check same results
        av, bv = [v.attributes["name"].value for v in (a, b)]
        assert isinstance(av, np.ndarray)
        assert isinstance(bv, np.ndarray)
        assert np.all(av == bv)

    def test_preexist(self):
        # Check same result when updating an existing attribute
        a, b = self.vars_a_b
        original_attribute_object = a.attributes["a"]
        for v in (a, b):
            v.avals["a"] = 7.2
        assert a.attributes == b.attributes
        assert a.attributes["a"] is original_attribute_object
        assert b.attributes["a"] is not original_attribute_object


class Test_getitems(MixinEquivTest):
    """Testing equivalence to ".attributes" """

    def test_getitem(self, value_oftype):
        v = self.var
        # Test op
        result = v.avals["a"]
        expected = v.attributes["a"].as_python_value()
        assert type(result) == type(expected)  # noqa: E721
        assert np.all(result == expected)

    def test_get_isthere(self):
        v = self.var
        assert v.avals.get("a", None) == 123

    def test_get_notthere(self):
        v = self.var
        assert v.avals.get("qq", "<my-default>") == "<my-default>"


class Test_pops(MixinEquivTest):
    def test_pop_isthere(self):
        a, b = self.vars_a_b
        ra = a.attributes.pop("a", "<noattr>")
        rb = b.avals.pop("a", "<novalue>")
        assert a.attributes == b.attributes
        assert ra == NcAttribute("a", 123)
        assert rb == 123

    def test_pop_notthere(self):
        a, b = self.vars_a_b
        ra = a.attributes.pop("q", "<noattr>")
        rb = b.avals.pop("q", "<novalue>")
        assert a.attributes == b.attributes
        assert ra == "<noattr>"
        assert rb == "<novalue>"

    def test_popitem(self):
        v = self.var
        assert list(v.attributes.keys()) == ["a", "b"]
        v.avals.popitem()
        assert list(v.attributes.keys()) == ["b"]
        v.avals.popitem()
        assert list(v.attributes.keys()) == []


class Test_rename(MixinEquivTest):
    def test_rename_basic(self):
        v = self.var
        original_a_attribute = v.attributes["a"]
        v.avals.rename("a", "z")
        assert list(v.attributes.keys()) == ["z", "b"]
        assert v.attributes["z"] is original_a_attribute
        assert v.attributes["z"].as_python_value() == 123

    def test_rename_nochange(self):
        v = self.var
        before = deepcopy(v.attributes)
        v.avals.rename("a", "a")
        after = deepcopy(v.attributes)
        assert list(v.attributes.keys()) == ["a", "b"]
        assert after == before

    def test_rename_nonexist(self):
        v = self.var
        with pytest.raises(KeyError):
            v.avals.rename("OTHER", "a")

    def test_rename_sameorder(self):
        v = self.var
        original_a_attribute = v.attributes["a"]
        v.avals.rename("a", "zzz")
        assert list(v.attributes.keys()) == ["zzz", "b"]
        assert v.attributes["zzz"] is original_a_attribute


class Test_otherops(MixinEquivTest):
    def test_del(self):
        a, b = self.vars_a_b
        del a.attributes["a"]
        del b.avals["a"]
        assert a.attributes == b.attributes

    def test_update(self):
        v = self.var
        # renaming : name of NcAttribute supplied as a value is lost
        v.avals.update(
            {"q": 49.3, "ZZ": NcAttribute("<lost>", [1, 2]), "a": 17}
        )
        # ordering : "q" and "ZZ" are added after, but "a" stays where it was
        assert list(v.attributes.keys()) == ["a", "b", "q", "ZZ"]
        assert v.attributes["a"] == NcAttribute("a", 17)
        assert v.attributes["q"] == NcAttribute("q", 49.3)
        assert v.attributes["ZZ"] == NcAttribute("ZZ", [1, 2])

    def test_clear(self):
        v = self.var
        assert v.attributes != {}
        v.avals.clear()
        assert v.attributes == {}

    def test_copy(self):
        v = self.var
        av = v.avals
        av_copy = av.copy()
        # check that: the attributes map is independent..
        before = deepcopy(av)
        av.rename("a", "z")
        assert av_copy == before  # NB uses dict.__eq__
        # ..but the contents are not
        av["z"] = "new-value"
        assert av_copy["a"] == "new-value"
