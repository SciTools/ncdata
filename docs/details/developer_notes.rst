Developer Notes
===============
Maintenance and manual processes


Change Log Maintenance
----------------------
Change log entries are now managed with `towncrier <https://towncrier.readthedocs.io/en/stable/>`_.

A new change-note fragment file should be included in each PR, but is normally created
with a ``towncrier`` command-line command:

* shortly, with ``towncrier create --content "mynotes..." <ISSUE-num>.<category>.rst``

  ... or, for longer content, use ``towncrier create --edit``.

* Here, "<category>" is one of:

    * "feat": user features
    * "doc": documentation changes
    * "bug": bug fixes
    * "dev": general developer-relevant changes
    * "misc": miscellaneous

  (For reference, these categories are configured in ``pyproject.toml``).

* the fragment files are stored in ``docs/changelog_fragments``.

* N.B. for this to work well, every change should be identified with a matching github issue.
  If there are multiple associated PRs, they should all be linked to the issue.

.. _developer_install:

Developer Installation
----------------------
For an editable installation, make a Python environment containing at least **numpy,
netCDF4, dask and pip**.  It is also highly recommended to get
`towncrier <https://github.com/twisted/towncrier>`_ and
`pre-commit <https://github.com/pre-commit/pre-commit>`_.
(and enable pre-commit with ``$ pre-commit install``).

Then, cd to your checkout, and ``$ pip install -e .``.
This should result in an editable development installation.


Documentation build
-------------------

For a full docs-build:

* The most useful way is simply ``$ cd docs`` and ``$ make html-keeplog``.
   * Note: the plainer ``$ make html`` is the same, but "-keeplog", in addition, preserves the
     changelog fragments **and** reverts the change_log.rst after the html build:
     This stops you accidentally including a "built" changelog when making further commits.
* The ``docs/Makefile`` wipes the API docs and invokes sphinx-apidoc for a full rebuild
* It also calls towncrier to clear out the changelog fragments + update ``docs/change_log.rst``.
* ( *assuming "-keeplog"*: fragments and change_notes.rst are then reverted, undoing the towncrier build ).
* the result is then available at ``docs/_build/html/index.html``.

.. note::

    * the above is just for **local testing**, when required.
    * For PRs (and releases), we also provide *automatic* builds on GitHub,
      via ReadTheDocs_.


Release actions
---------------

#. Update the :ref:`change-log page <change_log>` in the details section

    #. start with ``$ towncrier build``

    #. ensure all major changes + PRs are referenced in the :ref:`change_notes` section.

    #. update the "latest version" stated in the :ref:`development_status` section

#. Cut a release on GitHub

    * this triggers a new docs version on ReadTheDocs_.

#. Build the distribution

    #. if needed, get `build <https://github.com/pypa/build>`_

    #. run ``$ python -m build``

#. Push to PyPI

    #. if needed, get `twine <https://github.com/pypa/twine>`_

    #. run

        * ``$ python -m twine upload --repository testpypi dist/*``
        * this uploads to TestPyPI

    #. create a new env with test dependencies

       * ``$ conda create -n ncdtmp python=3.13 iris xarray filelock requests pytest pip``
       * ( N.B. 'filelock' and 'requests' are *test dependencies* of iris )

    #. install the new package with

       * ``$ pip install --index-url https://test.pypi.org/simple/ ncdata``
       * ..and run tests

    #. if that checks OK,

        * **remove** ``--repository testpypi`` **and repeat** "upload" step (2)
        * --> uploads to "real" PyPI

    #. repeat "pip install" step (4)

       * but **removing** the ``--index-url``
       * ..to check that ``pip install ncdata`` now finds the new version

#. Update conda to source the new version from PyPI

    #. create a PR on the `ncdata feedstock <https://github.com/conda-forge/ncdata-feedstock>`_
    #. update :

        * `version number <https://github.com/conda-forge/ncdata-feedstock/blob/3f6b35cbdffd2ee894821500f76f2b0b66f55939/recipe/meta.yaml#L2>`_
        * `SHA <https://github.com/conda-forge/ncdata-feedstock/blob/3f6b35cbdffd2ee894821500f76f2b0b66f55939/recipe/meta.yaml#L10>`_
        * Notes:

            * the `PyPI reference <https://github.com/conda-forge/ncdata-feedstock/blob/3f6b35cbdffd2ee894821500f76f2b0b66f55939/recipe/meta.yaml#L9>`_
              will normally look after itself
            * also at this point

               * make any required changes to `dependencies <https://github.com/conda-forge/ncdata-feedstock/blob/3f6b35cbdffd2ee894821500f76f2b0b66f55939/recipe/meta.yaml#L17-L29>`_
               * ..but normally, **no** changes will be required

    #. get PR merged

       * wait a few hours..
       * check that the new version appears in the output of ``$ conda search ncdata``


.. _ReadTheDocs: https://app.readthedocs.org/projects/ncdata/
