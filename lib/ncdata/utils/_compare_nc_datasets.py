"""
Utility for comparing 2 netcdf datasets.

Works with file-specs, netCDF4.Datasets *or* NcData.

For purposes of testing ncdata.netcdf4 behaviour.
TODO: one day might be public ?
"""

from pathlib import Path
from typing import AnyStr, List, Union
from warnings import warn

import netCDF4
import netCDF4 as nc
import numpy as np

from ncdata import NcData, NcVariable


def dataset_differences(
    dataset_or_path_1: Union[Path, AnyStr, nc.Dataset, NcData],
    dataset_or_path_2: Union[Path, AnyStr, nc.Dataset, NcData],
    check_names: bool = False,
    check_dims_order: bool = True,
    check_dims_unlimited: bool = True,
    check_vars_order: bool = True,
    check_attrs_order: bool = True,
    check_groups_order: bool = True,
    check_var_data: bool = True,
    show_n_first_different: int = 2,
    suppress_warnings: bool = False,
) -> List[str]:
    r"""
    Compare two netcdf datasets.

    Accepts paths, pathstrings, open :class:`netCDF4.Dataset`\s or
    :class:`~ncdata.NcData` objects.
    File paths are opened with the :mod:`netCDF4` module.

    See: :ref:`equality_testing`

    Parameters
    ----------
    dataset_or_path_1 : str or Path or netCDF4.Dataset or NcData
        First dataset to compare : either an open :class:`netCDF4.Dataset`, a path to
        open one, or an :class:`~ncdata.NcData` object.

    dataset_or_path_2 : str or Path or netCDF4.Dataset or NcData
        Second dataset to compare : either an open :class:`netCDF4.Dataset`, a path to
        open one, or an :class:`~ncdata.NcData` object.

    check_dims_order : bool, default True
        If False, no error results from the same dimensions appearing in a different
        order.  However, unless `suppress_warnings` is True, the error string is issued
        as a warning.

    check_vars_order : bool, default True
        If False, no error results from the same variables appearing in a different
        order. However unless `suppress_warnings` is True, the error string is issued
        as a warning.

    check_attrs_order : bool, default True
        If False, no error results from the same attributes appearing in a different
        order.  However unless `suppress_warnings` is True, the error string is issued
        as a warning.

    check_groups_order : bool, default True
        If False, no error results from the same groups appearing in a different order.
        However unless `suppress_warnings` is True, the error string is issued as a
        warning.

    check_names : bool, default False
        Whether to warn if the names of the top-level datasets are different

    check_dims_unlimited : bool, default True
        Whether to compare the 'unlimited' status of dimensions

    check_var_data : bool, default True
        If True, all variable data is also checked for equality.
        If False, only dtype and shape are compared.
        NOTE: comparison of arrays is done in-memory, so could be highly inefficient
        for large variable data.

    show_n_first_different : int, default 2
        Number of value differences to display.

    suppress_warnings : bool, default False
        When False (the default), report changes in content order as Warnings.
        When True, ignore changes in ordering.
        See also : :ref:`container-ordering`.

    Returns
    -------
    errs : list of str
        A list of "error" strings, describing differences between the inputs.
        If empty, no differences were found.

    Examples
    --------
    .. doctest::

        >>> data = NcData(
        ...    name="a",
        ...    variables=[NcVariable("b", data=[1, 2, 3, 4])],
        ...    attributes={"a1": 4}
        ... )
        >>> data2 = data.copy()
        >>> data2.avals.update({"a1":3, "v":7})
        >>> data2.variables["b"].data = np.array([1, 7, 3, 99])  # must be an array!
        >>> print('\n'.join(dataset_differences(data, data2)))
        Dataset attribute lists do not match: ['a1'] != ['a1', 'v']
        Dataset "a1" attribute values differ : 4 != 3
        Dataset variable "b" data contents differ, at 2 points: @INDICES[(1,), (3,)] : LHS=[2, 4], RHS=[7, 99]

    See Also
    --------
    :func:`~ncdata.utils.variable_differences`
    """
    ds1_was_path = not hasattr(dataset_or_path_1, "variables")
    ds2_was_path = not hasattr(dataset_or_path_2, "variables")
    ds1, ds2 = None, None
    try:
        # convert path-likes to netCDF4.Dataset
        if ds1_was_path:
            ds1 = nc.Dataset(dataset_or_path_1)
        else:
            ds1 = dataset_or_path_1

        if ds2_was_path:
            ds2 = nc.Dataset(dataset_or_path_2)
        else:
            ds2 = dataset_or_path_2

        # NOTE: Both ds1 and ds2 are now *either* NcData *or* netCDF4.Dataset
        #  _isncdata() will be used to distinguish.

        errs = _group_differences(
            ds1,
            ds2,
            group_id_string="Dataset",
            dims_order=check_dims_order,
            vars_order=check_vars_order,
            attrs_order=check_attrs_order,
            groups_order=check_groups_order,
            data_equality=check_var_data,
            suppress_warnings=suppress_warnings,
            check_names=check_names,
            check_unlimited=check_dims_unlimited,
            show_n_diffs=show_n_first_different,
        )
    finally:
        if ds1_was_path and ds1:
            ds1.close()
        if ds2_was_path and ds2:
            ds2.close()

    return errs


def _namelist_differences(
    l1, l2, elemname, order_strict=True, suppress_warnings=False
):
    errs = []
    msg = f"{elemname} do not match: {list(l1)} != {list(l2)}"
    ok = l1 == l2
    ok_except_order = ok
    if not ok:
        ok_except_order = sorted(l1) == sorted(l2)

    if not ok:
        if not ok_except_order or order_strict:
            errs.append(msg)
        elif ok_except_order and not suppress_warnings:
            warn("(Ignoring: " + msg + " )", category=UserWarning)
    return errs


def _isncdata(obj):
    """
    Distinguish NcData objects from similar netCDF4 ones.

    A crude test, used to support comparisons on either type of data.
    """
    return hasattr(obj, "_print_content")


def _attribute_arrays_eq(a1, a2):
    """
    Test equality of array values in attributes.

    Assumes values (attributes) are presented as numpy arrays (not lazy).
    Matches any NaNs.
    Does *NOT* handle masked data -- which does not occur in attributes.
    """
    result = True
    result &= a1.shape == a2.shape
    result &= a1.dtype == a2.dtype
    if result:
        if a1.dtype.kind in ("S", "U", "b"):
            result = np.all(a1 == a2)
        else:
            # array_equal handles possible NaN cases
            result = np.array_equal(a1, a2, equal_nan=True)
    return result


def _array_element_str(x):
    """Make a string representation of a numpy array element (scalar).

    Does *not* rely on numpy array printing.
    Instead converts to an equivalent Python object, and takes str(that).
    Hopefully delivers independence of numpy version (a lesson learned the hard way
    way in Iris development !)
    """
    if not isinstance(x, np.ndarray) or not hasattr(x.dtype, "kind"):
        result = str(x)
    elif np.ma.is_masked(x):
        result = "masked"
    else:
        kind = x.dtype.kind
        if kind in "iu":
            result = int(x)
        elif kind == "f":
            result = float(x)
        else:
            # Strings, and possibly other things.
            # Not totally clear what other things might occur here.
            result = str(x)
        result = str(result)
    return result


def _attribute_str(x):
    """Make a string representing an attribute value.

    Like the above, not depending on numpy array printing.
    """
    if isinstance(x, str):
        result = f"'{x}'"
    elif not isinstance(x, np.ndarray):
        result = str(x)
    elif x.ndim < 1:
        result = _array_element_str(x)
    else:
        els = [_array_element_str(el) for el in x]
        result = f"[{', '.join(els)}]"
    return result


def _attribute_differences(
    obj1,
    obj2,
    elemname,
    attrs_order=True,
    suppress_warnings=False,
    force_first_attrnames=None,
) -> List[str]:
    """
    Compare attribute name lists.

    Return a list of error messages.
    """
    attrnames, attrnames2 = [
        list(obj.avals.keys()) if _isncdata(obj) else list(obj.ncattrs())
        for obj in (obj1, obj2)
    ]
    if attrs_order and force_first_attrnames:
        # In order to ignore the order of appearance of *specific* attributes, move
        # all those ones to the front in a known order.
        def fix_orders(attrlist):
            for name in force_first_attrnames[::-1]:
                if name in attrlist:
                    attrlist = [name] + [n for n in attrlist if n != name]
            return attrlist

        attrnames = fix_orders(attrnames)
        attrnames2 = fix_orders(attrnames2)

    errs = _namelist_differences(
        attrnames,
        attrnames2,
        f"{elemname} attribute lists",
        order_strict=attrs_order,
        suppress_warnings=suppress_warnings,
    )

    # Compare the attributes themselves (dtypes and values)
    for attrname in attrnames:
        if attrname not in attrnames2:
            # Only compare attributes existing on both inputs.
            continue

        attr, attr2 = [
            (
                obj.avals[attrname]
                if _isncdata(obj)
                else obj.getncattr(attrname)
            )
            for obj in (obj1, obj2)
        ]

        # TODO: this still doesn't work well for strings : for those, we should ignore
        #  exact "type" (including length), and just compare the content.
        # TODO: get a good testcase going to check this behaviour
        dtype, dtype2 = [
            # Get x.dtype, or fallback on type(x) -- basically, for strings.
            getattr(attr, "dtype", type(attr))
            for attr in (attr, attr2)
        ]
        if all(
            isinstance(dt, np.dtype) and dt.kind in "SUb"
            for dt in (dtype, dtype2)
        ):
            dtype = dtype2 = "string"
        if dtype != dtype2:
            msg = (
                f'{elemname} "{attrname}" attribute datatypes differ : '
                f"{dtype!r} != {dtype2!r}"
            )
            errs.append(msg)
        else:
            # If datatypes match (only then), compare values
            # Cast attrs, which might be strings, to arrays for comparison
            arr, arr2 = [np.asarray(attr) for attr in (attr, attr2)]
            if not _attribute_arrays_eq(arr, arr2):
                # N.B. special comparison to handle strings and NaNs
                msg = (
                    f'{elemname} "{attrname}" attribute values differ : '
                    f"{_attribute_str(attr)} != {_attribute_str(attr2)}"
                )
                errs.append(msg)
    return errs


def variable_differences(
    v1: NcVariable,
    v2: NcVariable,
    check_attrs_order: bool = True,
    check_var_data: bool = True,
    show_n_first_different: int = 2,
    suppress_warnings: bool = False,
    _group_id_string: str = None,
) -> List[str]:
    r"""
    Compare variables.

    See: :ref:`equality_testing`

    Parameters
    ----------
    v1, v2 : NcVariable
        variables to compare
    check_attrs_order : bool, default True
        If False, no error results from the same contents in a different order,
        however unless `suppress_warnings` is True, the error string is issued as a warning.
    check_var_data : bool, default True
        If True, all variable data is also checked for equality.
        If False, only dtype and shape are compared.
        NOTE: comparison of large arrays is done in-memory, so may be highly inefficient.
    show_n_first_different: int, default 2
        Number of value differences to display.
    suppress_warnings : bool, default False
        When False (the default), report changes in content order as Warnings.
        When True, ignore changes in ordering entirely.
    _group_id_string : str
        (internal use only)

    Returns
    -------
    errs : list of str
        A list of "error" strings, describing differences between the inputs.
        If empty, no differences were found.

    See Also
    --------
    :func:`~ncdata.utils.dataset_differences`
    """
    errs = []

    show_n_first_different = int(show_n_first_different)
    if show_n_first_different < 1:
        msg = f"'show_n_diffs' must be >=1 : got {show_n_first_different!r}."
        raise ValueError(msg)

    if v1.name == v2.name:
        varname = v1.name
    else:
        varname = f"{v1.name} / {v2.name}"

    if _group_id_string:
        var_id_string = f'{_group_id_string} variable "{varname}"'
    else:
        var_id_string = f'Variable "{varname}"'

    if v1.name != v2.name:
        msg = f"{var_id_string} names differ : {v1.name!r} != {v2.name!r}"
        errs.append(msg)

    # dimensions
    dims, dims2 = [v.dimensions for v in (v1, v2)]
    if dims != dims2:
        msg = f"{var_id_string} dimensions differ : {dims!r} != {dims2!r}"
        errs.append(msg)

    # attributes
    errs += _attribute_differences(
        v1,
        v2,
        var_id_string,
        attrs_order=check_attrs_order,
        suppress_warnings=suppress_warnings,
        force_first_attrnames=[
            "_FillValue"
        ],  # for some reason, this doesn't always list consistently
    )

    # shapes
    def safe_varshape(var):
        if _isncdata(var):
            # NcVariable passed
            if var.data is None:
                # Allow for NcVariable.data to be empty
                shape = None
            else:
                shape = var.data.shape
        else:
            # netCDF4.Variable passed
            shape = var.shape
        return shape

    shape, shape2 = [safe_varshape(v) for v in (v1, v2)]
    if shape != shape2:
        msg = f"{var_id_string} shapes differ : {shape!r} != {shape2!r}"
        errs.append(msg)

    # dtypes
    dtype, dtype2 = [v.dtype if _isncdata(v) else v.datatype for v in (v1, v2)]
    if dtype != dtype2:
        msg = f"{var_id_string} datatypes differ : {dtype!r} != {dtype2!r}"
        errs.append(msg)

    # data values
    def _is_strtype(dt):
        if dt is None:
            result = False
        else:
            result = dt.kind in "SUb"
        return result

    is_str, is_str2 = (_is_strtype(dt) for dt in (dtype, dtype2))
    # TODO: is this correct check to allow compare between different dtypes?
    if (
        check_var_data
        and dims == dims2
        and shape == shape2
        and is_str == is_str2
    ):
        # NOTE: no attempt to use laziness here.  Could be improved.
        def getdata(var):
            if _isncdata(var):
                data = var.data
                if hasattr(data, "compute"):
                    data = data.compute()
            else:
                # expect var to be an actual netCDF4.Variable
                # (check for obscure property NOT provided by mimics)
                assert hasattr(var, "use_nc_get_vars")
                data = var[:]

            if data is None:
                # Empty variables still "sort of" work.
                data = np.array((), dtype=float)

            # Return 0D as 1D, as this makes results simpler to interpret.
            if data.ndim == 0:
                data = data.flatten()
                assert data.shape == (1,)
            return data

        data, data2 = (getdata(v) for v in (v1, v2))
        flatdata, flatdata2 = (
            np.asanyarray(arr).flatten() for arr in (data, data2)
        )

        # For simpler checking, use flat versions
        flat_diff_inds = (
            []
        )  # NB *don't* make this an array, it causes problems

        # Work out whether string : N.B. array type does not ALWAYS match the
        # variable type, because apparently the scalar content of a *masked* scalar
        # string variable has a numeric type (!! yuck !!)
        is_string_data = flatdata.dtype.kind in ("S", "U")
        if is_string_data:
            safe_fill_const = ""
        else:
            safe_fill_const = np.zeros((1,), dtype=flatdata.dtype)[0]

        # Where data is masked, count mask mismatches and skip those points
        if any(np.ma.is_masked(arr) for arr in (data, data2)):
            mask, mask2 = (
                np.ma.getmaskarray(array) for array in (flatdata, flatdata2)
            )
            flat_diff_inds = list(np.where(mask != mask2)[0])
            # Replace all masked points to exclude them from unmasked-point checks.
            either_masked = mask | mask2
            flatdata[either_masked] = safe_fill_const
            flatdata2[either_masked] = safe_fill_const

        # Where data has NANs, count mismatches and skip (as for masked)
        if not is_string_data:
            isnans, isnans2 = (np.isnan(arr) for arr in (flatdata, flatdata2))
            if np.any(isnans) or np.any(isnans2):
                nandiffs = np.where(isnans != isnans2)[0]
                if nandiffs.size > 0:
                    flat_diff_inds += list(nandiffs)
                anynans = isnans | isnans2
                flatdata[anynans] = safe_fill_const
                flatdata2[anynans] = safe_fill_const

        flat_diff_inds += list(np.where(flatdata != flatdata2)[0])
        # Order the nonmatching indices :  We report just the first few ...
        flat_diff_inds = sorted(flat_diff_inds)
        n_diffs = len(flat_diff_inds)
        if n_diffs:
            msg = (
                f"{var_id_string} data contents differ, at {n_diffs} points: "
            )
            ellps = ", ..." if n_diffs > show_n_first_different else ""
            diffinds = flat_diff_inds[:show_n_first_different]
            diffinds = [
                np.unravel_index(ind, shape=data.shape) for ind in diffinds
            ]
            diffinds_str = ", ".join(
                str(tuple([int(ind) for ind in x])) for x in diffinds
            )
            inds_str = f"[{diffinds_str}{ellps}]"
            points_lhs_str = ", ".join(
                _array_element_str(data[ind]) for ind in diffinds
            )
            points_rhs_str = ", ".join(
                _array_element_str(data2[ind]) for ind in diffinds
            )
            points_lhs_str = f"[{points_lhs_str}{ellps}]"
            points_rhs_str = f"[{points_rhs_str}{ellps}]"
            msg += (
                f"@INDICES{inds_str}"
                f" : LHS={points_lhs_str}, RHS={points_rhs_str}"
            )
            errs.append(msg)
    return errs


def _group_differences(
    g1: Union[netCDF4.Dataset, netCDF4.Group],
    g2: Union[netCDF4.Dataset, netCDF4.Group],
    group_id_string: str,
    dims_order: bool = True,
    vars_order: bool = True,
    attrs_order: bool = True,
    groups_order: bool = True,
    data_equality: bool = True,
    suppress_warnings: bool = False,
    check_names: bool = False,
    check_unlimited: bool = True,
    show_n_diffs: int = 2,
) -> List[str]:
    """
    Inner routine to compare either whole datasets or subgroups.

    Returns a list of error strings.
    """
    errs = []

    if check_names:
        if g1.name != g2.name:
            errs.append(
                f"Datasets have different names: {g1.name!r} != {g2.name!r}."
            )
    # Compare lists of dimension names
    dimnames, dimnames2 = [list(grp.dimensions.keys()) for grp in (g1, g2)]
    errs += _namelist_differences(
        dimnames,
        dimnames2,
        f"{group_id_string} dimension lists",
        order_strict=dims_order,
        suppress_warnings=suppress_warnings,
    )

    # Compare the dimensions themselves
    for dimname in dimnames:
        if dimname not in dimnames2:
            continue
        d1, d2 = [grp.dimensions[dimname] for grp in (g1, g2)]
        dimlen, dimlen2 = [dim.size for dim in (d1, d2)]
        if dimlen != dimlen2:
            msg = (
                f'{group_id_string} "{dimname}" dimensions '
                f"have different sizes: {dimlen} != {dimlen2}"
            )
            errs.append(msg)

        if check_unlimited:
            unlim1, unlim2 = [
                dim.unlimited if _isncdata(dim) else dim.isunlimited()
                for dim in (d1, d2)
            ]
            if unlim1 != unlim2:
                msg = (
                    f'{group_id_string} "{dimname}" dimension '
                    f'has different "unlimited" status : {unlim1} != {unlim2}'
                )
                errs.append(msg)

    # Compare file attributes
    errs += _attribute_differences(
        g1,
        g2,
        group_id_string,
        attrs_order=attrs_order,
        suppress_warnings=suppress_warnings,
    )

    # Compare lists of variables
    varnames, varnames2 = [list(grp.variables.keys()) for grp in (g1, g2)]
    errs += _namelist_differences(
        varnames,
        varnames2,
        f"{group_id_string} variable lists",
        order_strict=vars_order,
        suppress_warnings=suppress_warnings,
    )

    # Compare the variables themselves
    for varname in varnames:
        if varname not in varnames2:
            continue
        v1, v2 = [grp.variables[varname] for grp in (g1, g2)]
        errs += variable_differences(
            v1,
            v2,
            check_attrs_order=attrs_order,
            check_var_data=data_equality,
            show_n_first_different=show_n_diffs,
            suppress_warnings=suppress_warnings,
            _group_id_string=group_id_string,
        )

    # Finally, recurse over groups
    grpnames, grpnames2 = [list(grp.groups.keys()) for grp in (g1, g2)]
    errs += _namelist_differences(
        grpnames,
        grpnames2,
        f"{group_id_string} subgroup lists",
        order_strict=groups_order,
        suppress_warnings=suppress_warnings,
    )
    for grpname in grpnames:
        if grpname not in grpnames2:
            continue
        grp1, grp2 = [grp.groups[grpname] for grp in (g1, g2)]
        errs += _group_differences(
            grp1,
            grp2,
            group_id_string=f"{group_id_string}/{grpname}",
            dims_order=dims_order,
            vars_order=vars_order,
            attrs_order=attrs_order,
            groups_order=groups_order,
            data_equality=data_equality,
            check_unlimited=check_unlimited,
            show_n_diffs=show_n_diffs,
        )
    return errs
