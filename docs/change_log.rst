.. _change_log:

Versions and Change Notes
=========================

.. _development_status:

Project Development Status
--------------------------
We intend to follow `PEP 440 <https://peps.python.org/pep-0440/>`_,
or (older) `SemVer <https://semver.org/>`_ versioning principles.
This means the version string has the basic form **"major.minor.bugfix[special-types]"**.

Current release version is at : |version|

This is a complete implementation, with functional operational of all public APIs.
The code is however still experimental, and APIs are not stable
(hence no major version yet).

.. _change_notes:

Change Notes
------------
Summary of key features by release number.

.. towncrier release notes start

ncdata 0.3.2 (2025-12-01)
~~~~~~~~~~~~~~~~~~~~~~~~~
A bugfix release, just to allow working with Python 3.10.

Bug Fixes
^^^^^^^^^

- Made indexing work with Python v3.10. (`ISSUE#183 <https://github.com/SciTools/ncdata/pull/183>`_)


Developer and Internal changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Support more Python versions: now allowing >=3.10 and no longer limiting <3.14.
  CI testing now against 3.10 / 3.11 / 3.12 / 3.13.
  (`ISSUE#186 <https://github.com/SciTools/ncdata/pull/186>`_)



v0.3.1 (2025-11-06)
~~~~~~~~~~~~~~~~~~~~~~~~~
A minor release to replace v0.3.0, fixing some test errors introduced by the latest
xarray (`2025.10.1 <https://github.com/pydata/xarray/releases/tag/v2025.10.1>`_).

.. note::
    **Note on Python v3.14**

    At present (Nov 2025), a number of dependencies do not work
    with Python 3.14 -- notably Iris.

    These will probably be fixed soon, but for now we aren't testing ncdata with Python 3.14.

    Core operations do *appear* to function with Python 3.14, but correct operation can't be
    guaranteed until we specifically adopt it.


Documentation changes
^^^^^^^^^^^^^^^^^^^^^

- Document how to create a developer installation. (`ISSUE#174 <https://github.com/SciTools/ncdata/pull/174>`_)


Developer and Internal changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Fix xarray 2025.09.1 problem. (`ISSUE#173 <https://github.com/SciTools/ncdata/pull/173>`_)
- Test against given Python versions : currently 3.12 and 3.13.
  **Not** python 3.14, for now, due to emerging problems with dependencies (notably Iris). (`ISSUE#175 <https://github.com/SciTools/ncdata/pull/175>`_)


v0.3.0
~~~~~~
Added handy utilities; made attribute access easier; reworked documentation.

Features
^^^^^^^^

- Added the ability to extract a sub-region by indexing/slicing over dimensions.
  The :class:`ncdata.NcData` objects can be indexed with the ``[]`` operation, or over
  specifed dimensions with the :meth:`~ncdata.NcData.slicer` method.
  This is based on the new :meth:`~ncdata.utils.index_by_dimensions()` utility method
  and :class:`~ncdata.utils.Slicer` class.
  See: :ref:`utils_indexing` (`ISSUE#68 <https://github.com/SciTools/ncdata/pull/68>`_)
- Added the :func:`~ncdata.utils.rename_dimension` utility.
  This provides a "safe" dimension rename, which also replaces
  the name in all variables which use it. (`ISSUE#87 <https://github.com/SciTools/ncdata/pull/87>`_)
- Added the ".avals" property as an easier way of managing attributes:
  This provides a simple "name: value" map, bypassing the NcAttribute objects and converting values to and from simple Python equivalents.
  This effectively replaces the older 'set_attrval' and 'get_attrval', which will eventually be removed.
  See: :ref:`attributes_and_avals` (`ISSUE#117 <https://github.com/SciTools/ncdata/pull/117>`_)
- Make :meth:`~ncdata.iris.to_iris` use the full iris load processing,
  instead of :meth:`iris.fileformats.netcdf.loader.load_cubes`.
  This means you can use load controls such as callbacks and constraints. (`ISSUE#131 <https://github.com/SciTools/ncdata/pull/131>`_)
- Provide exact == and !=  for datasets and variables, by just calling the difference utilities.
  This can be inefficient, but is simple to understand and generally useful.
  See: :ref:`equality_testing` (`ISSUE#166 <https://github.com/SciTools/ncdata/pull/166>`_)


Documentation changes
^^^^^^^^^^^^^^^^^^^^^

- Added a `userguide page <userdocs/user_guide/utilities.html>`_ summarising all the utility features in :mod:`ncdata.utils`. (`ISSUE#161 <https://github.com/SciTools/ncdata/pull/161>`_)
- Made all docs examples into doctests; add doctest CI action. (`ISSUE#136 <https://github.com/SciTools/ncdata/pull/136>`_)


Bug Fixes
^^^^^^^^^

- Fixed a bug in dataset comparison, where variables with missing or unbroadcastable data arrays could cause errors rather than generating difference messages. (`ISSUE#153 <https://github.com/SciTools/ncdata/pull/153>`_)


Developer and Internal changes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Switch to towncrier for whats-new management. (`ISSUE#116 <https://github.com/SciTools/ncdata/pull/116>`_)
- Added regular linkcheck gha. (`ISSUE#123 <https://github.com/SciTools/ncdata/pull/123>`_)
- @valeriupredoi added test for Zarr conversion to Iris cubes. (`ISSUE#145 <https://github.com/SciTools/ncdata/pull/145>`_)


v0.2.0
~~~~~~
Overhauled data manipulation APIs.  Expanded and improved documentation.

* `@pp-mo`_ Reviewed, corrected, reorganised and expanded all documentation.
  Added description section on core classes and operations, and how-to snippets.
  (`PR#109 <https://github.com/SciTools/ncdata/pull/109>`_).

* `@pp-mo`_ Unpin Numpy to support versions >= 2.0
  (`PR#112 <https://github.com/SciTools/ncdata/pull/112>`_).

* `@pp-mo`_ Added crude dimension-based load chunking control.
  (`PR#108 <https://github.com/SciTools/ncdata/pull/108>`_).

* `@pp-mo`_ Support equality testing (==) of dimensions and attributes.
  (`PR#107 <https://github.com/SciTools/ncdata/pull/107>`_).

* `@pp-mo`_ Enforce that NcAttribute.value is always an 0- or 1-D array.
  (`PR#106 <https://github.com/SciTools/ncdata/pull/106>`_).

* `@pp-mo`_ Support copy as utility, and as core classes copy() methods.
  (`PR#98 <https://github.com/SciTools/ncdata/pull/98>`_).

* `@pp-mo`_ Support a simple {name: value} map for attributes in data constructors.
  (`PR#71 <https://github.com/SciTools/ncdata/pull/71>`_).

* `@pp-mo`_ Make dataset comparison routines a public utility.
  (`PR#70 <https://github.com/SciTools/ncdata/pull/70>`_).

* `@pp-mo`_ initial Sphinx documentation
  (`PR#76 <https://github.com/SciTools/ncdata/pull/76>`_).

* `@trexfeathers`_ added a Logo
  (`PR#75 <https://github.com/SciTools/ncdata/pull/75>`_).

* `@pp-mo`_ added Save errors util
  (`PR#64 <https://github.com/SciTools/ncdata/pull/64>`_).


v0.1.1
~~~~~~
Small tweaks + bug fixes.
**Note:** `PR#62 <https://github.com/SciTools/ncdata/pull/62>`_, and 
`PR#59 <https://github.com/SciTools/ncdata/pull/59>`_ are important fixes to
achieve intended performance goals,
i.e. moving arbitrarily large data via Dask without running out of memory.

`v0.1.1 on GitHub <https://github.com/SciTools/ncdata/releases/tag/v0.1.1>`_

* Stop non-numpy attribute values from breaking attribute printout.
  `PR#63 <https://github.com/SciTools/ncdata/pull/63>`_

* Stop ``ncdata.iris.from_iris()`` consuming full data memory for each variable.
  `PR#62 <https://github.com/SciTools/ncdata/pull/62>`_

* Provide convenience APIs for ncdata component dictionaries and attribute values.
  `PR#61 <https://github.com/SciTools/ncdata/pull/61>`_

* Use dask ``chunks="auto"`` in ``ncdata.netcdf4.from_nc4()``.
  `PR#59 <https://github.com/SciTools/ncdata/pull/59>`_


v0.1.0
~~~~~~
First release

`v0.1.0 on GitHub <https://github.com/SciTools/ncdata/releases/tag/v0.1.0>`_

.. _@trexfeathers: https://github.com/trexfeathers
.. _@pp-mo: https://github.com/trexfeathers