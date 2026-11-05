# Copyright SciTools contributors
#
# This file is part of SciTools/ncdata and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""
An abstract representation of Netcdf data with groups, variables + attributes.

As provided by the NetCDF "Common Data Model" :
https://docs.unidata.ucar.edu/netcdf-java/5.3/userguide/common_data_model_overview.html

It is also provided with read/write conversion interfaces to Xarray, Iris and netCDF4,
thus acting as an efficient exchange channel between any of those forms.

"""

# N.B. this file excluded from isort, as we want a specific class order for the docs

from ._core import NameMap, NcAttribute, NcData, NcDimension, NcVariable
from ._version import __version__

# N.B. avoiding sorting this, the order is convenient for understanding.
__all__ = [  # noqa: RUF022
    "NcData",
    "NcDimension",
    "NcVariable",
    "NcAttribute",
    "NameMap",
    "__version__",
]
