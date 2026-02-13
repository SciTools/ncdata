.. _interface_support:

Support for Interfaces and Dependencies
=======================================

Python Compatibility
--------------------
NcData currently supports Python >= 3.10, and is tested against 3.10, 3.11, 3.12 and 3.13.

NetCDF4 Compatibility
---------------------
NcData supports netCDF4-python package versions >= 1.6.5,
which implies the NetCDF4 library version >= 4.9.

The `Continuous Integration testing on GitHub`_ tests against the latest version compatible with Iris and Xarray.

Datatypes
^^^^^^^^^
Ncdata supports all the regular datatypes of netcdf, but *not* the
variable-length and user-defined datatypes.
Please see : :ref:`data-types`.


Data Scaling and Masking
^^^^^^^^^^^^^^^^^^^^^^^^
Ncdata does not implement scaling and offset within variable data arrays :  The ".data"
array has the actual variable dtype, and the "scale_factor" and
"add_offset" attributes are treated like any other attribute.

Likewise, Ncdata does not use masking within its variable data arrays, so that variable
data arrays contain "raw" data, which include any "fill" values -- i.e. at any missing
data points you will have a "fill" value rather than a masked point.

The use of "scale_factor", "add_offset" and "_FillValue" attributes are standard
conventions described in the NetCDF documentation itself, and implemented by NetCDF
library software including the Python netCDF4 library.  To ignore these default
interpretations, ncdata has to actually turn these features "off".  The rationale for
this, however, is that the low-level unprocessed data content, equivalent to actual
file storage, may be more likely to form a stable common basis of equivalence, particularly
between different system architectures.


.. _file-storage:

File storage control
^^^^^^^^^^^^^^^^^^^^
The :func:`ncdata.netcdf4.to_nc4` cannot control compression or storage options
provided by :meth:`netCDF4.Dataset.createVariable`, which means you can't
control the data compression and translation facilities of the NetCDF file
library.
If required, you should use :mod:`iris` or :mod:`xarray` for this, i.e. use
:meth:`xarray.Dataset.to_netcdf` or :func:`iris.save` instead of
:func:`ncdata.netcdf4.to_nc4`, as these provide more special options for controlling
netcdf file creation.

File-specific storage aspects, such as chunking, data-paths or compression
strategies, are not recorded in the core objects.  However, array representations in
variable and attribute data (notably dask lazy arrays) may hold such information.

The concept of "unlimited" dimensions is also, you might think, outside the abstract
model of NetCDF data and not of concern to Ncdata .  However, in fact this concept is
present as a core property of dimensions in the classic NetCDF data model (see
"Dimension" in the `NetCDF Classic Data Model`_), so that is why it **is** an essential
property of an NcDimension also.


Dask chunking control
^^^^^^^^^^^^^^^^^^^^^
Loading from netcdf files generates  variables whose data arrays are all Dask
lazy arrays.  These are created with the "chunks='auto'" setting.

However there is a simple per-dimension chunking control available on loading.
See :func:`ncdata.netcdf4.from_nc4`.


Xarray Compatibility
--------------------
NcData is tested with xarray >= 2023.7.0

The `Continuous Integration testing on GitHub`_ test against the latest
released version of Xarray compatible with the other requirements
(i.e., usually, the latest release version).

The "encoding" controls are lost in converting from Xarray to Ncdata.
However, an "encoding" attribute which exists on a variable **will** be passed
to the 'encoding' control in Xarray, so this can be used to control data
storage formats.

The NcVariable.data arrays behave as if netCDF4.Variable arrays, hence they
contain raw data which Xarray may convert from integers to floating-point, or
masked data to NaNs.

.. warning::
    In conversion to xarray with :func:`~ncdata.xarray.to_xarray`
    dataset encodings are not reproduced, most notably
    **any "unlimited_dims" control is lost**.  But, this is effectively a bug,
    which may be fixed later.
    See : `issue#66 <https://github.com/SciTools/ncdata/issues/66>`_


Iris Compatibility
------------------
The `Continuous Integration testing on GitHub`_ tests against the
latest release of Iris.

Ncdata is compatible with iris >= v3.7.0
see : `support added in v3.7.0 <https://scitools-iris.readthedocs.io/en/stable/whatsnew/3.7.html#internal>`_

.. warning::

    In conversion from iris cubes with :func:`ncdata.iris.from_iris`
    use of an `unlimited_dims` key currently causes an exception
    See : `issue#43 <https://github.com/SciTools/ncdata/issues/43>`_


.. _Continuous Integration testing on GitHub: https://github.com/SciTools/ncdata/blob/main/.github/workflows/ci-tests.yml
.. _NetCDF Classic Data Model: https://docs.unidata.ucar.edu/netcdf-c/current/netcdf_data_model.html#classic_model
