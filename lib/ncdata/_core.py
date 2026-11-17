# Copyright SciTools contributors
#
# This file is part of SciTools/ncdata and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""
Our main classes for representing netCDF data.

These structures support groups, variables, attributes.
Both data and attributes are stored as numpy-compatible array-like values (though this
may include dask.array.Array), and hence their types are modelled as np.dtype's.

Current limitations :
(1) we are *not* supporting user-defined or variable-length types.
(2) there is no built-in consistency checking -- e.g. for correct length or existence
of dimensions referenced by variables.

"""

from collections.abc import Iterable, Mapping, MutableMapping
from functools import wraps
from typing import (
    Any,
    Optional,
    Union,
)

import numpy
import numpy as np


class NameMap(dict):
    """
    A specialised dictionary type for data manipulation convenience.

    All values (aka 'content items') are expected to have a ".name" property which is a
    string, and we aim to ensure that "value.name == key" for each key, value pair.

    This "item key relation" is *not* rigorously enforced, but we provide convenience
    methods which make use of it and help to maintain it :
    See :meth:`NameMap.add`, :meth:`NameMap.addall` :meth:`NameMap.from_items` and
    :meth:`NameMap.rename`.

    ..
    """

    def __init__(self, *args, item_type=None, **kwargs):
        """
        Create a NameMap with dict-style constructor behaviour.

        For example, from key/value pairs.

        Notes
        -----
        A keyword-only 'item_type' arg sets the 'item_type' property.

        Creation using :meth:`NameMap.from_items` is generally more convenient.
        """
        super().__init__(*args, **kwargs)
        #: expected type of all content items (if not None)
        self.item_type = item_type

    def copy(self) -> "NameMap":
        """Produce a new NameMap with same content and item_type."""
        # NOTE: dict.copy() produces a dict, and will not duplicate 'item_type'.
        return NameMap.from_items(self.values(), item_type=self.item_type)

    def add(self, item):
        """
        Enter a content item under its '.name'.

        If the NameMap has a non-None 'item_type', the added item is type checked.
        """
        if self.item_type is not None:
            if not isinstance(item, self.item_type):
                raise TypeError(
                    f'Item expected to be of type {self.item_type} : got {item}."'
                )
        if not hasattr(item, "name"):
            raise TypeError(f"Item has no '.name' property : {item}.")
        self[item.name] = item

    def addall(self, items):
        """Add a number of content items with self.add()."""
        for item in items:
            self.add(item)

    def rename(self, name: str, new_name: str):
        """
        Rename a content item.

        Parameters
        ----------
        name
            name of an existing item.  If not, a KeyError will occur.

        new_name
            new name for the selected item.  Both the container key and its ".name"
            will be changed.

        Notes
        -----
        The order of items is preserved.

        If "new_name == name", has no effect.  Otherwise, if `new_name` already exists,
        the old item of that name is removed, but the renamed item remains in its
        original order place.

        Examples
        --------
        >>> mymap = NameMap.from_items([NcAttribute(x, x.upper()) for x in "abcd"])
        >>> mymap
        {'a': NcAttribute('a', 'A'), 'b': NcAttribute('b', 'B'), 'c': NcAttribute('c', 'C'), 'd': NcAttribute('d', 'D')}
        >>> mymap.rename('b', 'qqq')
        >>> mymap
        {'a': NcAttribute('a', 'A'), 'qqq': NcAttribute('qqq', 'B'), 'c': NcAttribute('c', 'C'), 'd': NcAttribute('d', 'D')}
        >>> mymap.rename('a', 'c')
        >>> mymap
        {'c': NcAttribute('c', 'A'), 'qqq': NcAttribute('qqq', 'B'), 'd': NcAttribute('d', 'D')}
        """  # noqa: E501
        if new_name == name:
            # skip this case, to avoid removing the item because it matches 'new_name'.
            pass
        else:
            # Since keys are immutable, we can't change a key in-place. So to preserve
            # item order within the container, we extract all items + re-insert them.
            # Get all items in original order, except an existing item of the new name
            items = [item for key, item in self.items() if key != new_name]
            # rename the selected item object (meaning it no longer matches its key)
            self[name].name = new_name
            # clear content and re-insert items in the original order.
            self.clear()
            self.addall(items)

    @classmethod
    def from_items(
        cls, arg: Iterable | Mapping, item_type=None
    ) -> "NameMap":
        """
        Convert an iterable or mapping of items to a NameMap.

        Parameters
        ----------
        arg
            an iterable or mapping of 'content items'.

        item_type
            if not None, we expect all contents to be of this type.

        Returns
        -------
        map
            a NameMap with the given 'item_type'

        Notes
        -----
        All content items must have a ".name" property.  If 'item_type' is not None,
        all items must be of the given type.

        If 'arg' is an iterable, its contents are added.

        If 'arg' is a mapping, it normally must have (key == arg[key].name)
        for all keys.  As a special case, only if `item_type` ==
        :class:`NcAttribute`, a plain name: value map can be provided, which
        is converted to name: NcAttribute(name, value).

        If 'arg'' is a NameMap of the same 'item_type' (including None), then 'arg'
        is returned unchanged as the result.

        If the input is a NameMap of a different 'item_type', it is converted to a new
        NameMap of the required 'item_type' (assuming contents match the requirement).
        """
        if isinstance(arg, cls):
            if arg.item_type == item_type:
                # If input is of required type, this is a no-copy operation.
                result = arg
            else:
                # Replace with a map of of the required type.
                result = cls(item_type=item_type)
                result.addall(arg.values())
        else:
            # Start with an empty map of the required type.
            result = cls(item_type=item_type)
            if arg is not None:
                # We expect either another type of dictionary, or a list of items.
                if isinstance(arg, Mapping):
                    if (
                        item_type == NcAttribute
                        and len(arg) > 0
                        and not isinstance(
                            next(iter(arg.values())), NcAttribute
                        )
                    ):
                        # for attributes only, also allow simple name=value map
                        # for which, convert each value to an NcAttribute
                        arg = {
                            name: NcAttribute(name, value)
                            for name, value in arg.items()
                        }
                    # existing mapping of NameMap type
                    # ignore mapping keys, and set [name]=item.name for each value.
                    result.addall(arg.values())
                elif isinstance(arg, Iterable):
                    result.addall(arg)
                else:
                    msg = (
                        f"Argument must be an iterable or mapping: got {arg}."
                    )
                    raise TypeError(msg)

        return result


# A convenient name alias for :meth:`NameMap.from_items`.
as_namemap = NameMap.from_items


class AttrvalsDict(MutableMapping):
    """
    A container for the attribute values of an NcData or NcVariable object.

    Its values are actually stored in the ._attributes of the parent object, which is a
    :class:`NameMap` of :class:`NcAttribute` objects.
    This means that all attribute values are cast to/from valid NcAttribute values,
    i.e. numpy arrays of selected dtypes and <=1 dimensions.

    This allows "data.attributes" to provide the easy-access form {name: value},
    while "data._attributes" is the NameMap {name: NcAttribute}.

    """

    def __init__(self, attrs: NameMap):
        self._attrs = attrs

    def __getitem__(self, key):
        return self._attrs[key].as_python_value()

    def __setitem__(self, key, value):
        # Special case : assigning an NcAttribute object is equivalent to assigning
        #  its value (which forces rename).
        # This is supposed to avoid potential confusion (!)
        if isinstance(value, NcAttribute):
            value = value.value

        if key in self._attrs:
            # Update the existing NcAttribute.
            self._attrs[key].value = value
        else:
            # Create an NcAttribute.
            self._attrs[key] = NcAttribute(key, value)

    #
    # Provide the minimal set of methods required for a MutableMapping
    #
    def __len__(self):
        return len(self._attrs)

    def __iter__(self):
        return iter(self._attrs)

    def __delitem__(self, key):
        self._attrs.__delitem__(key)

    #
    # Other useful methods
    #
    def __str__(self):
        """
        Provide a string representation.

        A convenient representation, somewhat consistent with how we print the
        attributes in the printout of an NcData / NcVariable.
        """
        text = "AttrvalsDict{"
        if not self._attrs:
            text += "}"
        else:
            lines = [
                f"{attr.name!s}: {attr._print_value()}"
                for attr in self._attrs.values()
            ]
            if len(lines) == 1:
                text += lines[0] + "}"
            else:
                lines = ["\n    " + line for line in lines]
                text += "".join(lines)
                text += "\n}"

        return text

    # Also really nice-to-have : the "rename" operation, as implemented by NameMap
    @wraps(NameMap.rename)
    # N.B. this is *not* missing a docstring, actually..
    def rename(self, name: str, new_name: str):
        self._attrs.rename(name, new_name)

    def copy(self, deepcopy=True):
        """
        Make a copy.

        Returns
        -------
            AttrvalsDict

        The copy is 'shallow', in that it contains a NameMap with the *same*
        NcAttributes as the original. These are themselves mutable, so the copy will
        change if the original content attributes are modified, and vice versa.

        N.B. for a truly independent 'deep' copy, use `copy.deepcopy`.
        """
        return self.__class__(self._attrs.copy())


class _AttributeAccessMixin:
    """
    A mixin with attribute access conveniences for NcData and NcVariable.

    This assists in assigning and extracting attributes as python values.
    See :meth:`NcAttribute.get_python_value()` for how different types are handled.
    """

    @property
    def avals(self):
        r"""
        A convenience view of an NcData or NcVariable's attribute values.

        The ``.avals`` property acts as a more convenient **view** onto the
        ``.attributes``, providing a simple "{name: value}" map.

        ( Meanwhile, the attributes are actually **stored** in the
        ``.attributes`` property, but this is a "{name: NcAttribute}" map ).

        The values of the ``.avals`` map are the values of the ``.attributes``, but
        converted to convenience forms, as described in
        :meth:`NcAttribute.as_python_value`.

        See also: :ref:`attributes_and_avals`.
        """
        if not hasattr(self, "_attrmap"):
            self._attrmap = AttrvalsDict(self.attributes)
        return self._attrmap

    def get_attrval(self, name: str) -> Any:
        """
        Get the Python value of a named attribute in self.attributes.

        If no such attribute exists, returns None.

        .. warning::
            This legacy method is deprecated, and will be removed in a future release.

        """
        return self.avals.get(name, None)

    def set_attrval(self, name: str, value) -> "NcAttribute":
        """
        Set the Python value of a named attribute in self.attributes.

        .. warning::
            This legacy method is deprecated, and will be removed in a future release.

        """
        self.avals[name] = value
        attr = self.avals[name]
        return attr


#
# A relatively simple and naive representation of netCDF data.
#


def _addlines_indent(text, indent=""):
    # Routine to indent each line within a newline-joined string.
    return [indent + line for line in text.split("\n")]


# common indent spacing
_STANDARD_INDENT = " " * 4


class NcData(_AttributeAccessMixin):
    """
    An object representing a netcdf group- or dataset-level container.

    Containing dimensions, variables, attributes and sub-groups.
    """

    def __init__(
        self,
        name: str | None = None,
        dimensions: Mapping | Iterable = None,
        variables: Mapping | Iterable = None,
        attributes: Mapping | Iterable = None,
        groups: Mapping | Iterable = None,
    ):
        #: a group/dataset name (optional)
        self.name: str = name
        #: group/dataset dimensions
        self.dimensions: dict[str, NcDimension] = as_namemap(
            dimensions, NcDimension
        )
        #: group/dataset variables
        self.variables: dict[str, NcVariable] = as_namemap(
            variables, NcVariable
        )
        #: group/dataset global attributes
        self.attributes: dict[str, NcAttribute] = as_namemap(
            attributes, NcAttribute
        )
        #: sub-groups
        self.groups: dict[str, NcData] = as_namemap(groups, NcData)

    def _print_content(self) -> str:
        """
        Construct a string printout.

        NcData classes all define '_print_content' (though they have no common base
        class, so it isn't technically an abstract method).
        This "NcData._print_content()" is called recursively for groups.
        """
        global _STANDARD_INDENT
        # Define a header line (always a separate line).
        noname = "<'no-name'>"
        lines = [f"<NcData: {self.name or noname}"]

        # Add internal sections in order, indenting everything.
        for eltype in ("dimensions", "variables", "groups", "attributes"):
            els = getattr(self, eltype)
            if len(els):
                if eltype == "attributes":
                    # Attributes a bit different: #1 add 'globol' to section title.
                    lines += [f"{_STANDARD_INDENT}global attributes:"]
                    # NOTE: #2 show like variable attributes, but *no parent name*.
                    attrs_lines = [
                        f":{attr._print_content()}"
                        for attr in self.attributes.values()
                    ]
                    lines += _addlines_indent(
                        "\n".join(attrs_lines), _STANDARD_INDENT * 2
                    )
                else:
                    lines += [f"{_STANDARD_INDENT}{eltype}:"]
                    for el in els.values():
                        lines += _addlines_indent(
                            el._print_content(), _STANDARD_INDENT * 2
                        )
                lines.append("")

        # Strip off final blank lines (tidier for Groups as well as main dataset).
        while len(lines[-1]) == 0:
            lines = lines[:-1]

        # Add closing line.
        lines += [">"]
        # Join with linefeeds for a simple string result.
        return "\n".join(lines)

    def __str__(self):
        return self._print_content()

    # NOTE: for 'repr', an interpretable literal string is too complex.
    # So just retain the default "object" address-based representation.

    def copy(self):
        """
        Copy self.

        This duplicates structure with all-new ncdata core objects, but does not
        duplicate variable data arrays.  See :func:`ncdata.utils.ncdata_copy`.
        """
        from ncdata.utils import ncdata_copy

        return ncdata_copy(self)

    # Provide a slicing interface, by just linking to ncdata.utils._dim_indexing code.
    def slicer(self, *dim_names):
        """
        Make a :class:`~ncdata.utils.Slicer` object to index the data.

        This creates a slicer which can then be indexed to sub-index the data.
        See: :ref:`howto_slice`

        Parameters
        ----------
        dim_names: list(str)
            Names of dimensions to slice.

        Returns
        -------
        :class:`~ncdata.utils.Slicer`

        Examples
        --------
        .. testsetup::
            >>> from ncdata._core import NcData, NcDimension
            >>> ncdata = NcData(dimensions=[NcDimension('x', 4), NcDimension('y', 5)])

        >>> subregion = ncdata.slicer('x', 'y')[3, 2:4]
        """
        from ncdata.utils import Slicer

        return Slicer(self, *dim_names)

    def __getitem__(self, indices):
        # N.B. Python [,,] syntax ensures there is only ever one argument,
        #  either a single key or a tuple :
        #    : [x] --> keys == x
        #    : [x, y, ...] --> keys == (x, y, ...)
        return self.slicer().__getitem__(indices)

    # Define equality in terms of dataset comparison utility
    def __eq__(self, other):
        if id(other) == id(self):
            result = True
        elif not isinstance(other, NcData):
            result = False
        else:
            from ncdata.utils import dataset_differences

            result = dataset_differences(self, other) == []
        return result


class NcDimension:
    """
    An object representing a netcdf dimension.

    Associates a name with a length, and also an 'unlimited' flag.
    """

    def __init__(
        self, name: str, size: int, unlimited: bool | None = None
    ):
        #: dimension name
        self.name: str = name
        #: dimension size (current size, if unlimited)
        self.size: int = size  # N.B. we retain the 'zero size means unlimited'
        if size == 0:
            unlimited = True
        else:
            unlimited = bool(unlimited)
        #: whether dimension is unlimited
        self.unlimited: bool = unlimited

    def _print_content(self) -> str:
        str_unlim = "  **UNLIMITED**" if self.unlimited else ""
        return f"{self.name} = {self.size}{str_unlim}"

    def __repr__(self):
        str_unlim = ", unlimited=True" if self.unlimited else ""
        return f"NcDimension({self.name!r}, {self.size}{str_unlim})"

    def __str__(self):
        return repr(self)

    def copy(self):
        """Copy self."""
        return NcDimension(self.name, size=self.size, unlimited=self.unlimited)

    def __eq__(self, other):
        """Support simply equality testing."""
        return (
            self.name == other.name
            and self.size == other.size
            and self.unlimited == other.unlimited
        )


class NcVariable(_AttributeAccessMixin):
    """
    An object representing a netcdf variable.

    With dimensions, dtype and data, and attributes.

    'data' may be None, but if not is expected to be an array : either numpy (real) or
    Dask (lazy).

    The 'dtype' will presumably match the data, if any.

    It has no 'shape' property, in practice this might be inferred from either the
    data or dimensions.  If the dims are empty, it is a scalar.

    A variable makes no effort to ensure that its dimensions, dtype + data are
    consistent.  This is to be managed by the creator.
    """

    def __init__(
        self,
        name: str,
        dimensions: Iterable[str] = (),
        # NOTE: flake8 objects to type checking against an unimported package, even in
        # a quoted reference when it's not otherwise needed (and we don't want to
        # import it).
        # TODO: remove the flake8 annotation if this gets fixed.
        data: Optional[
            Union[np.ndarray, "dask.array.array"]  # noqa: F821
        ] = None,
        dtype: np.dtype = None,
        attributes: Mapping | Iterable = None,
        group: "NcData" = None,
    ):
        """
        Create a variable.

        The 'dtype' arg is relevant only when no data is provided :
        If 'data' is provided, it is converted to an array if needed, and its dtype
        replaces any provided 'dtype'.
        """
        #: variable name
        self.name: str = name
        #: variable dimension names (a list of strings, *not* a dict of objects)
        self.dimensions: tuple[str] = tuple(dimensions)
        if data is not None:
            if not hasattr(data, "dtype"):
                data = np.asanyarray(data)
            dtype = data.dtype
        #: variable datatype, as a numpy :class:`numpy.dtype`
        if dtype is not None and not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        self.dtype: numpy.dtype = dtype
        #: variable data (an array-like, typically a dask or numpy array)
        self.data = data  # Supports lazy, and normally provides a dtype
        #: variable attributes
        self.attributes: NameMap = as_namemap(attributes, NcAttribute)
        #: parent group
        self.group: NcData | None = group

    # # Provide some array-like readonly properties reflected from the data.
    # @property
    # def dtype(self):
    #     return self.data.dtype
    #
    # @property
    # def shape(self):
    #     return self.data.shape

    def _print_content(self):
        global _STANDARD_INDENT
        dimstr = ", ".join(self.dimensions)
        typestr = str(self.dtype) if self.dtype else "<no-dtype>"
        hdr = f"<NcVariable({typestr}): {self.name}({dimstr})"
        if not self.attributes:
            hdr += ">"
            lines = [hdr]
        else:
            lines = [hdr]
            attrs_lines = [
                f"{self.name}:{attr._print_content()}"
                for attr in self.attributes.values()
            ]
            lines += _addlines_indent("\n".join(attrs_lines), _STANDARD_INDENT)
            lines += [">"]
        return "\n".join(lines)

    def __str__(self):
        return self._print_content()

    # NOTE: as for NcData, an interpretable 'repr' string is too complex.
    # So just retain the default "object" address-based representation.

    def copy(self):
        """
        Copy self.

        Does not duplicate arrays in data content.
        See :func:`ncdata.utils.ncdata_copy`.
        """
        from ncdata.utils._copy import _attributes_copy

        var = NcVariable(
            name=self.name,
            dimensions=self.dimensions,
            dtype=self.dtype,
            data=self.data,
            attributes=_attributes_copy(self.attributes),
            group=self.group,
        )
        return var

    # Define equality in terms of variable comparison utility
    def __eq__(self, other):
        if id(other) == id(self):
            result = True
        elif not isinstance(other, NcVariable):
            result = False
        else:
            from ncdata.utils import variable_differences

            result = variable_differences(self, other) == []
        return result


class NcAttribute:
    """
    An object representing a netcdf variable, group or dataset attribute.

    Associates a name to a value which is always a numpy scalar or 1-D array, of an
    allowed dtype.  See :ref:`attribute-dtypes`.

    In an actual netcdf dataset, a "scalar" is actually just an array of length 1.
    """

    def __init__(self, name: str, value):
        #: attribute name
        self.name: str = name
        # Attribute values are arraylike, have dtype
        #: attribute value, constrained to a suitable numpy array object.
        self.value: np.ndarray = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not hasattr(value, "dtype"):
            value = np.asanyarray(value)
        if value.ndim > 1:
            raise ValueError(
                "Attribute value should only be 0- or 1-dimensional."
            )
        self._value = value

    def as_python_value(self):
        """
        Return the attribute value, converted for convenient use in Python code.

        An array of one element returns as a single scalar value, since this is how
        attributes behave in actual netcdf data.

        * Single numeric values are returned as numpy scalars, i.e. integers or floats.
          The advantage of this is that they retain an exact dtype,
          e.g. np.int16 or np.float32, but are still mostly interchangeable with Python
          ints and floats.

        * A single string value is returned as a Python string object.
          See :ref:`attribute-dtypes`.

        * Multiple numeric values are returned as a 1-D numpy array (which is mostly
          interchangeable with a list).

        * Multiple strings are returned as a list of strings.

          .. Note::

                However, an attribute *in an actual NetCDF file* cannot currently
                contain multiple strings, due to shortcomings in the netCDF
                implementation: they will be concatenated into a single string.

        """
        result = self.value
        # Attributes are either scalar or 1-D
        if result.ndim >= 2:
            raise ValueError(
                "Attribute value should only be 0- or 1-dimensional."
            )

        if result.shape == (1,):
            # Reduce the dimension, but *always* return an array scalar.
            # This is useful for consistency of handling non-numpy content
            result = result[0]
            result = np.asanyarray(result)

        if result.dtype.kind in ("U", "S"):
            if result.ndim == 0:
                result = str(result)
            elif result.ndim == 1:
                result = [str(x) for x in result]

        # TODO: is this a possibiliity ?
        # if isinstance(result, bytes):
        #     result = result.decode()
        return result

    def _print_value(self):
        value = self.as_python_value()

        # Convert numpy non-string scalars to simple Python values, in string output.
        if getattr(value, "shape", None) in ((0,), (1,), ()):
            op = {"i": int, "u": int, "f": float}.get(value.dtype.kind)
            if op:
                value = op(value.flatten()[0])

        return repr(value)

    def _print_content(self):
        return f"{self.name} = {self._print_value()}"

    def __repr__(self):
        return f"NcAttribute({self.name!r}, {self._print_value()})"

    def __str__(self):
        return repr(self)

    def copy(self):
        """Copy self, including any array value content."""
        return NcAttribute(self.name, self.value.copy())

    def __eq__(self, other):
        """Support simple equality testing."""
        if not isinstance(other, NcAttribute):
            result = NotImplemented
        else:
            result = self.name == other.name
            if result:
                v1 = self.value
                v2 = other.value
                result = (
                    v1.shape == v2.shape
                    and v1.dtype == v2.dtype
                    and np.all(v1 == v2)
                )
        return result
