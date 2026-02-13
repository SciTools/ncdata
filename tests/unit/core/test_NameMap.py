"""
Tests for class :class:`ncdata.NameMap`.
"""

from copy import deepcopy

import pytest

from ncdata import NameMap, NcAttribute


class NamedItem:
    """A minimal 'named type', from which you can construct a NameMap."""

    def __init__(self, name):
        self.name = name


class OtherNamedItem:
    """A distinct, identical but unrelated named type, for type testing."""

    def __init__(self, name):
        self.name = name


_test_items = [NamedItem(name) for name in "abcd"]


def sample_namemap(item_type=None):
    # Make a deep copy of the items (which themselves are mutable).
    items = deepcopy(_test_items)
    # Return an all-new map each time, to avoid any problems with in-place operations
    return NameMap.from_items(items, item_type=item_type)


@pytest.fixture(ids=["notytpe", "withtype"], params=[None, NamedItem])
def item_type(request):
    return request.param


@pytest.fixture
def item_type_and_kwargs(item_type):
    kwargs = {}
    if item_type is not None:
        kwargs["item_type"] = item_type
    return item_type, kwargs


class TestDictlikeInit:
    def test_empty(self, item_type_and_kwargs):
        item_type, kwargs = item_type_and_kwargs
        namemap = NameMap(**kwargs)
        assert isinstance(namemap, NameMap)
        assert namemap.item_type == item_type

    def test_item_pairs(self, item_type_and_kwargs):
        item_type, kwargs = item_type_and_kwargs
        pairs = [(item.name, item) for item in _test_items]
        namemap = NameMap(pairs, **kwargs)
        assert isinstance(namemap, NameMap)
        assert namemap.item_type == item_type
        assert len(namemap) == len(_test_items)
        assert list(namemap.keys()) == [item.name for item in _test_items]
        assert all(namemap[key].name == key for key in namemap.keys())

    def test_nameless_items(self):
        """Check we can construct like a "normal" dictionary when item_type is None."""
        pairs = [(item.name, item) for item in _test_items]
        pairs.append(("xxx", "odd-string-value"))
        namemap = NameMap(pairs)
        assert namemap.item_type is None
        assert len(namemap) == len(pairs)
        assert namemap["xxx"] == "odd-string-value"


class TestFromItems:
    def test_empty(self, item_type_and_kwargs):
        item_type, kwargs = item_type_and_kwargs
        namemap = NameMap.from_items([], **kwargs)
        assert isinstance(namemap, NameMap)
        assert len(namemap) == 0
        assert namemap.item_type == item_type

    def test_list_arg(self, item_type_and_kwargs):
        item_type, kwargs = item_type_and_kwargs
        namemap = NameMap.from_items(_test_items, **kwargs)
        assert isinstance(namemap, NameMap)
        assert namemap.item_type == item_type
        assert len(namemap) == len(_test_items)
        assert list(namemap.keys()) == [item.name for item in _test_items]
        assert all(namemap[key].name == key for key in namemap.keys())

    def test_map_arg(self, item_type_and_kwargs):
        item_type, kwargs = item_type_and_kwargs
        # Construct a dictionary, with *arbitrary* keys (they are ignored).
        arg = {count: item for count, item in enumerate(_test_items)}
        namemap = NameMap.from_items(arg, **kwargs)
        assert isinstance(namemap, NameMap)
        assert namemap.item_type == item_type
        assert len(namemap) == len(_test_items)
        assert list(namemap.keys()) == [item.name for item in _test_items]
        assert all(namemap[key].name == key for key in namemap.keys())

    def test_items_typemismatch(self, item_type_and_kwargs):
        item_type, kwargs = item_type_and_kwargs
        items = list(_test_items)
        items[2:2] = [OtherNamedItem("qq")]
        if item_type is None:
            # succeeds
            NameMap.from_items(items, **kwargs)
        else:
            # fails
            msg = r"Item expected to be of type.*\bNamedItem\b"
            with pytest.raises(TypeError, match=msg):
                NameMap.from_items(items, **kwargs)

    def test_namemap_arg__bothtyped_nocopy(self):
        arg = sample_namemap(item_type=NamedItem)
        namemap = NameMap.from_items(arg, item_type=arg.item_type)
        assert namemap is arg

    def test_namemap_arg__bothuntyped_nocopy(self):
        arg = sample_namemap(item_type=None)
        namemap = NameMap.from_items(arg)
        assert namemap is arg

    def test_namemap_arg__typed2untyped(self):
        arg = sample_namemap(item_type=NamedItem)
        namemap = NameMap.from_items(arg)
        assert namemap is not arg
        assert namemap.item_type is None
        assert namemap == arg

    def test_namemap_arg__untyped2typed(self):
        arg = sample_namemap(item_type=None)
        namemap = NameMap.from_items(arg, item_type=NamedItem)
        assert namemap is not arg
        assert namemap.item_type is NamedItem
        assert namemap == arg

    def test_namemap_arg__bad_typeconvert__fail(self):
        arg = sample_namemap()
        msg = r"Item expected to be of type .*\bOtherNamedItem\b"
        with pytest.raises(TypeError, match=msg):
            NameMap.from_items(arg, item_type=OtherNamedItem)


class TestAttributesFromItems:
    """
    Extra constructor checks *specifically* for attributes.

    Since they are treated differently in order to support the
    "attributes={'x':1, 'y':2}" constructor style.
    """

    @pytest.fixture(
        params=[None, NcAttribute, NamedItem],
        ids=["none", "attrs", "nonattrs"],
    )
    def target_itemtype(self, request):
        return request.param

    @pytest.fixture(params=[False, True], ids=["single", "multiple"])
    def multiple(self, request):
        return request.param

    def test_attributes__map(self, target_itemtype, multiple):
        # Create from classic map {name: NcAttr(name, value)}
        arg = {"x": NcAttribute("x", 1)}
        if multiple:
            arg["y"] = NcAttribute("y", 2)

        if target_itemtype == NamedItem:
            msg = "Item expected to be of type.*NamedItem.* got NcAttribute"
            with pytest.raises(TypeError, match=msg):
                NameMap.from_items(arg, item_type=target_itemtype)
        else:
            namemap = NameMap.from_items(arg, item_type=target_itemtype)
            assert namemap.item_type == target_itemtype
            # Note: this asserts that the contents are the *original* uncopied
            # NcAttribute objects, since we don't support == on NcAttributes
            assert namemap == arg

    def test_attributes__list(self, target_itemtype, multiple):
        # Create from a list [*NcAttr(name, value)]
        arg = [NcAttribute("x", 1), NcAttribute("y", 2)]
        if not multiple:
            arg = arg[:1]

        if target_itemtype == NamedItem:
            msg = "Item expected to be of type.*NamedItem.* got NcAttribute"
            with pytest.raises(TypeError, match=msg):
                NameMap.from_items(arg, item_type=target_itemtype)
        else:
            namemap = NameMap.from_items(arg, item_type=target_itemtype)
            assert namemap.item_type == target_itemtype
            assert list(namemap.keys()) == [attr.name for attr in arg]
            # Again, content is the original objects
            assert list(namemap.values()) == arg

    def test_attributes__namevaluemap(self, target_itemtype, multiple):
        # Create from a newstyle map {name: value}
        arg = {"x": 1}
        if multiple:
            arg["y"] = 2
        if target_itemtype != NcAttribute:
            if target_itemtype is None:
                msg = "Item has no '.name' property"
            else:
                # target_itemtype == NamedItem
                msg = "Item expected to be of type.*NamedItem"
            with pytest.raises(TypeError, match=msg):
                NameMap.from_items(arg, item_type=target_itemtype)
        else:
            namemap = NameMap.from_items(arg, item_type=target_itemtype)
            assert namemap.item_type == target_itemtype
            assert list(namemap.keys()) == list(arg.keys())
            # Note: a bit of a fuss because we don't have == for NcAttributes
            vals = list(namemap.values())
            assert all(isinstance(el, NcAttribute) for el in vals)
            vals = [val.value for val in vals]
            assert vals == list(arg.values())

    @pytest.mark.parametrize(
        "arg", [[], {}, None], ids=["list", "map", "none"]
    )
    def test_attributes_empty(self, arg):
        # Just check correct construction from empty args.
        namemap = NameMap.from_items(arg, item_type=NcAttribute)
        assert namemap == {} and namemap is not arg


class Test_copy:
    def test_copy(self, item_type):
        source = sample_namemap(item_type=item_type)
        result = source.copy()
        assert result is not source
        assert result == source
        assert result.item_type == item_type
        # NOTE: not a *deep* copy : contains the same item objects
        assert all(
            e1 is e2 for e1, e2 in zip(result.values(), source.values())
        )


class Test_add:
    def test_sametyped_item(self, item_type):
        source = sample_namemap(item_type=item_type)
        testmap = source.copy()
        testmap.add(NamedItem("qq"))
        assert set(testmap) == set(["qq"]) | set(source)

    def test_differenttype_item(self, item_type):
        testmap = sample_namemap(item_type=item_type)
        newitem = OtherNamedItem("qq")
        if item_type is None:
            testmap.add(newitem)
            assert newitem in list(testmap.values())
        else:
            msg = r"Item expected to be of type .*\bNamedItem\b"
            with pytest.raises(TypeError, match=msg):
                testmap.add(newitem)

    def test_overwrite(self):
        source = sample_namemap()
        testmap = source.copy()
        # Use an added item with the same same as an existing one.
        newitem = NamedItem("b")
        testmap.add(newitem)
        # The resulting keys, and their order, should be the same.
        assert list(testmap.keys()) == list(source.keys())
        assert testmap["b"] is newitem


class Test_addall:
    def test_basic(self):
        source = sample_namemap()
        testmap = source.copy()
        new_items = [NamedItem("x"), NamedItem("y")]
        testmap.addall(new_items)
        assert list(testmap.keys()) == ["a", "b", "c", "d", "x", "y"]
        assert set(testmap.values()) == set(source.values()) | set(new_items)

    def test_overwrites(self):
        source = sample_namemap()
        testmap = source.copy()
        new_items = [NamedItem("x"), NamedItem("b")]
        testmap.addall(new_items)
        assert list(testmap.keys()) == ["a", "b", "c", "d", "x"]
        assert set(testmap.values()) == (
            (set(source.values()) - set([source["b"]])) | set(new_items)
        )


class Test_rename:
    def test_basic(self):
        source = sample_namemap()
        testmap = source.copy()
        target = testmap["b"]

        testmap.rename("b", "q")

        assert target.name == "q"
        assert list(testmap.keys()) == ["a", "q", "c", "d"]
        assert list(testmap.values()) == list(source.values())
        # NOTE: target is not copied (no deepcopy) : it changed in the source also
        assert source["b"] is target
        assert source["b"].name == "q"

    def test_overwrite(self):
        source = sample_namemap()
        testmap = source.copy()
        target = testmap["b"]

        testmap.rename("b", "c")

        assert target.name == "c"
        assert list(testmap.keys()) == ["a", "c", "d"]
