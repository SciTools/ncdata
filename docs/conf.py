# noqa
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from ncdata._version import version_tuple

# -- Project information -----------------------------------------------------

project = "ncdata"
copyright = "2023, SciTools"
author = "SciTools"

# The complete version, including alpha/beta/rc tags
version_parts = [str(part) for part in version_tuple]
release = ".".join(version_parts)
# A version string omitting extra segments, e.g. "v0.2.0.dev3"
version = ".".join(version_parts[:4])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
]

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "iris": ("https://scitools-iris.readthedocs.io/en/latest/", None),
    # Can't make this work ??
    # "netCDF4": ("https://github.com/Unidata/netcdf4-python", None),
}

from pathlib import Path

docsdir_pth = Path(__name__).parent.absolute()
print("docsdir import path:", docsdir_pth)
ncdata_pth = (docsdir_pth.parent / "lib").absolute()
print("ncdata import path:", ncdata_pth)

import sys

sys.path.append(str(ncdata_pth))
print("PATH:")
print("\n".join(p for p in sys.path))

# Autodoc config..
autopackage_name = [
    "ncdata",
    "ncdata.iris_xarray",
    "ncdata.netcdf4",
    "ncdata.xarray",
    "ncdata.dataset_like",
]
# api generation configuration
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autodoc_inherited_members = True

autodoc_default_options = {
    "member-order": "bysource",
    "inherited-members": True,
    "class-signature": "separated",
    "autodoc_typehints": "description",
    "autoclass_content": "class",
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "changelog_fragments",
    "details/api/modules.rst",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_logo = "_static/ncdata_logo.png"
html_favicon = "_static/ncdata_logo.png"
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- copybutton extension -----------------------------------------------------
# See https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"


# Various scheme control settings.
# See https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/layout.html

html_sidebars = {
    # Set content of main (lhs) sidebar to show an "ncdata on github" link.
    # This also ensures that the sidebar is not empty even on the main page,
    #  so the ReadTheDocs version switcher is always shown.
    "**": ["repo", "sidebar-nav-bs"]
}

html_theme_options = {
    "logo": {
        "text": f"ncdata v{version}",
    },
    # NOTE: not controlling navigation or TOC depth, as 'auto' is better?
    "navigation_with_keys": False,
    "show_prev_next": False,
    # Control top nav bar : default shows just the links (and a search field?)
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    # Control rhs sidebar : default shows edit/source buttons (let's not).
    "secondary_sidebar_items": ["page-toc"],
}

html_context = {
    # Possibly needed for pydata_theme?
    "github_repo": "ncdata",
    "github_user": "SciTools",
    "github_version": "main",
    "doc_path": "docs",
    # Default light/dark mode.
    # Info: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/light-dark.html#configure-default-theme-mode
    "default_mode": "auto",
}
