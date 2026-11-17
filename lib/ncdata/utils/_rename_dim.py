"""Utility to rename dimensions."""

from ncdata import NcData


def _rename_dims_in_vars(ncdata: NcData, name_from: str, name_to: str) -> None:
    """Rename a dimension in all contained variables which reference it."""
    for var in ncdata.variables.values():
        if name_from in var.dimensions:
            var.dimensions = tuple(
                [
                    name_to if name == name_from else name
                    for name in var.dimensions
                ]
            )

    # Also rename in all sub-groups, except where the dimension is redefined
    #  (i.e. inside a "scope hole").
    for grp in ncdata.groups.values():
        if name_from not in grp.dimensions:
            _rename_dims_in_vars(grp, name_from, name_to)


def _check_name_collisions(ncdata, name_from, name_to, group_path=""):
    if name_to in ncdata.dimensions:
        inner = f' in group "{group_path}"' if group_path else ""
        msg = (
            f"Cannot rename dimension {name_from!r} to {name_to!r}, "
            f"because a {name_to!r} dimension already exists{inner}."
        )
        raise ValueError(msg)

    for group in ncdata.groups.values():
        if name_from in group.dimensions:
            # Skip this group as its 'name_from' dim makes it a "scope hole".
            continue
        inner_path = group_path + "/" + group.name
        _check_name_collisions(
            group, name_from, name_to, group_path=inner_path
        )


def rename_dimension(ncdata: NcData, name_from: str, name_to: str) -> None:
    """
    Rename a dimension of an :class:`~ncdata.NcData`.

    This function calls ``ncdata.dimensions.rename``, but then it *also* renames the
    dimension in all the variables which reference it, including those in sub-groups.

    See: :ref:`operations_rename`

    Parameters
    ----------
    ncdata : NcData
        data with a top-level dimension to rename.

    name_from: str
        existing name of dimension to rename.

    name_to: str
        new name of dimension.

    Notes
    -----
    * The operation is in-place. To produce a *new* :class:`~ncdata.NcData` with the
      renamed dimension, create a copy first with :meth:`~ncdata.NcData.copy`.

    * Unlike a simple :meth:`~ncdata.NameMap.rename`, this checks whether a dimension
      of the new name already exists, and if so raises an error.

    """
    _check_name_collisions(ncdata, name_from, name_to)
    ncdata.dimensions.rename(name_from, name_to)
    _rename_dims_in_vars(ncdata, name_from, name_to)
