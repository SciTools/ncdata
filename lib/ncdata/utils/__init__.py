# Copyright SciTools contributors
#
# This file is part of SciTools/ncdata and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""General user utility functions."""

from ._compare_nc_datasets import dataset_differences, variable_differences
from ._copy import ncdata_copy
from ._dim_indexing import Slicer, index_by_dimensions
from ._rename_dim import rename_dimension
from ._save_errors import save_errors

__all__ = [  # noqa: RUF022
    "rename_dimension",
    "dataset_differences",
    "variable_differences",
    "index_by_dimensions",
    "Slicer",
    "save_errors",
    "ncdata_copy",
]
