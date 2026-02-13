.. _known-issues:

Outstanding Issues
==================

Known Problems
--------------
To be fixed

* in conversion from iris cubes
  with `from_iris <https://ncdata.readthedocs.io/en/latest/details/api/ncdata.iris.html#ncdata.iris.from_iris>`_

   * use of an `unlimited_dims` key currently causes an exception

   * `issue#43 <https://github.com/SciTools/ncdata/issues/43>`_

* in conversion to xarray
  with `to_xarray <https://ncdata.readthedocs.io/en/latest/details/api/ncdata.xarray.html#ncdata.xarray.to_xarray>`_

   * dataset encodings are not reproduced

   * most notably, **the "unlimited_dims" control is missing**

   * `issue#66 <https://github.com/SciTools/ncdata/issues/66>`_

* in conversion to/from netCDF4 files

   * netCDF4 performs automatic encoding/decoding of byte data to characters, triggered
     by the existence of an ``_Encoding`` attribute on a character type variable.
     Ncdata does not currently account for this, and may fail to read/write correctly.


.. _todo:

Incomplete Documentation
^^^^^^^^^^^^^^^^^^^^^^^^
(PLACEHOLDER: documentation is incomplete, please fix me !)


Identified Design Limitations
-----------------------------

Features unsupported
^^^^^^^^^^^^^^^^^^^^
There are no current plans to address these, but could be considered in future

* user-defined datatypes are not supported

    * i.e. as introduced in NetCDF4

    * notably, includes compound and variable-length types

    * ..and especially **variable-length strings in variables**.
      see : :ref:`string-and-character-data`, :ref:`data-types`


Features planned
^^^^^^^^^^^^^^^^
Features not yet implemented, but intended for future releases

   * groups (not yet fully supported ?)

   * file output chunking control

