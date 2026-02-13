"""User utility routines for ncdata."""

from typing import Dict, List, Union

import netCDF4 as nc
import numpy as np

from ncdata import NcData, NcVariable


def _name_is_valid(name) -> bool:
    result = True
    if not isinstance(name, str) or not name:
        # Catches non-string (e.g. None, 0, ..) and empty string
        result = False
    else:
        # The name rules for netCDF are not fully clear, but seem *extremely* liberal.
        # It seems that "/" is not allowed, and that's about it
        # So *allow* whitespace, backslash, initial digit, initial underscore ...
        if "/" in name:
            result = False
    return result


def _name_errors(element_container, id_string):
    """Check that all elements in the container have valid and consistent names."""
    errors = []
    for name, element in element_container.items():
        if element.name != name:
            errors.append(
                f"{id_string} element {name!r} has a different element.name : "
                f"{element.name!r}."
            )
        if not _name_is_valid(name):
            errors.append(
                f"{id_string}s has an element with an invalid netCDF name : "
                f"{name!r}"
            )
    return errors


_NETCDF_VALID_DTYPES = [np.dtype(key) for key in nc.default_fillvals.keys()]


def _valid_attr_dtype(dtype):
    # For attributes, we currently accept any kind of string dtype
    # We should probably rationalise this, but for now they are converted by netCDF4
    return dtype.kind in "SU" or dtype in _NETCDF_VALID_DTYPES


def _invalid_attr_errors(
    element: Union[NcData, NcVariable], name_prefix: str
) -> List[str]:
    errors = []
    for attr in element.attributes.values():
        dtype = attr.value.dtype
        if not _valid_attr_dtype(dtype):
            errors.append(
                f"{name_prefix} attribute {attr.name!r} has a value which cannot be "
                f"saved to netcdf : {attr.value!r} ::dtype={dtype}."
            )
    return errors


def _variable_errors(
    var: NcVariable, var_prefix: str, known_dimensions: Dict[str, int]
) -> List[str]:
    errors = []
    if var.data is None:
        errors.append(f"{var_prefix} has no data array.")
    else:
        if var.dtype not in _NETCDF_VALID_DTYPES:
            errors.append(
                f"{var_prefix} has a dtype which cannot be saved to netcdf : "
                f"{var.dtype!r}."
            )

        unknown_dimensions = [
            dim for dim in var.dimensions if dim not in known_dimensions
        ]
        if unknown_dimensions:
            errors.append(
                f"{var_prefix} references dimensions which are not found in the "
                f"enclosing dataset : {unknown_dimensions!r}"
            )
        else:
            dims_shape = tuple(known_dimensions[dim] for dim in var.dimensions)
            if var.data.shape != dims_shape:
                errors.append(
                    f"{var_prefix} data shape = {var.data.shape}, does not match that "
                    f"of its dimensions = {dims_shape}."
                )

    # Warn about any unsaveable variable attributes
    errors += _invalid_attr_errors(var, var_prefix)
    return errors


def _save_errors_inner(
    ncdata: NcData,
    enclosing_dimensions: Dict[str, int] = None,
    group_path: str = None,
) -> List[str]:
    """
    Scan dataset, with context allowing operation over inner groups.

    Parameters
    ----------
    ncdata
        data to check

    enclosing_dimensions
        A mapping {name:length} of dimensions existing in the enclosing dataset,
        within which 'ncdata' is a group

    group_path
        The group name or path of ncdata (including its name), when 'ncdata' is a
        group within an enclosing dataset

    Returns
    -------
    errors
        A list of strings describing problems with the dataset
    """
    # Construct a name prefix for naming dataset/group attributes
    if group_path is None:
        group_path = ""
        ncdata_identity_prefix = "Dataset"
        if ncdata.name:
            ncdata_identity_prefix += f"({ncdata.name!r})"
    else:
        ncdata_identity_prefix = f"Group {group_path!r}"

    if enclosing_dimensions is None:
        enclosing_dimensions = {}

    # Add local definitions to the map of available dimensions
    # (N.B. inner name duplicates simply replace those from the caller).
    known_dimensions = enclosing_dimensions.copy()  # don't the passed arg
    known_dimensions.update(
        {name: dimension.size for name, dimension in ncdata.dimensions.items()}
    )

    # Collect the various detected errors
    errors = []

    # Check that all named containers use only valid names
    for component in ("dimension", "variable", "attribute", "group"):
        errors += _name_errors(
            getattr(ncdata, component + "s"),  # N.B. pluralise here
            id_string=f"{ncdata_identity_prefix} {component}",
        )

    # List all the variable errors
    path_context = group_path
    if path_context:
        path_context += "/"
    for var in ncdata.variables.values():
        var_prefix = f"Variable '{path_context}{var.name}'"
        errors += _variable_errors(var, var_prefix, known_dimensions)

    # Warn about unsaveable dataset/group attributes
    errors += _invalid_attr_errors(ncdata, ncdata_identity_prefix)

    # Recurse over inner groups
    if ncdata.groups:
        if not group_path:
            # prefix inner group paths with the dataset name, if any
            group_path = ncdata.name or ""
        for group in ncdata.groups.values():
            errors.extend(
                _save_errors_inner(
                    group,
                    enclosing_dimensions=known_dimensions,
                    group_path=group_path + f"/{group.name}",
                )
            )

    return errors


def save_errors(ncdata: NcData) -> List[str]:
    """
    Scan a dataset for consistency and completeness.

    See: :ref:`correctness-checks`

    Describe any aspects of this dataset which would prevent it from saving (cause an
    error).
    If there are any such problems, then an attempt to save the ncdata to a netcdf file
    will fail.  If there are none, then a save should succeed.

    Parameters
    ----------
    ncdata
        data to check

    Returns
    -------
    errors
        A list of strings, error messages describing problems with the dataset.
        If no errors, returns an empty list.

    Notes
    -----
    The checks made are roughly the following:

    **(1)** check names in all components (dimensions, variables, attributes and groups):

    * all names are valid netcdf names
    * all element names match their key in the component,
      i.e. ``component[key].name == key``

    **(2)** check that all attribute values have netcdf-compatible dtypes.

    * ( E.G. no object or compound (recarray) dtypes )

    **(3)** check that, for all contained variables:

    * its dimensions are all present in the enclosing dataset
    * it has an attached data array, of a netcdf-compatible dtype
    * the shape of its data matches the lengths of its dimensions
    """
    return _save_errors_inner(ncdata)
