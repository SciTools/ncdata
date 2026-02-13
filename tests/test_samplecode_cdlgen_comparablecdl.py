"""
PoC code for netcdf file CDL-based testing mechanisms.

File generation from CDL (with ncgen), and CDL comparison (with ncdump).

The status and usage of this are yet to be determined.
"""

import os
from pathlib import Path
import subprocess
from typing import AnyStr, List, Optional

import pytest

# Note : `_env_bin_path` and `ncgen_from_cdl` are taken from Iris test code.
# Code here duplicated from Iris v3.4.1
# https://github.com/SciTools/iris/blob/v3.4.1/lib/iris/tests/stock/netcdf.py#L18-L63


def _env_bin_path(exe_name: AnyStr = None):
    """
    Return a Path object for (an executable in) the environment bin directory.

    Parameters
    ----------
    exe_name : str
        If set, the name of an executable to append to the path.

    Returns
    -------
    exe_path : Path
        A path to the bin directory, or an executable file within it.

    Notes
    -----
    For use in tests which spawn commands which should call executables within
    the Python environment, since many IDEs (Eclipse, PyCharm) don't
    automatically include this location in $PATH (as opposed to $PYTHONPATH).
    """
    exe_path = Path(os.__file__)
    exe_path = (exe_path / "../../../bin").resolve()
    if exe_name is not None:
        exe_path = exe_path / exe_name
    return exe_path


NCGEN_PATHSTR = str(_env_bin_path("ncgen"))
NCDUMP_PATHSTR = str(_env_bin_path("ncdump"))


def ncgen_from_cdl(
    cdl_str: Optional[str], cdl_path: Optional[str], nc_path: str
):
    """
    Generate a test netcdf file from cdl.

    Source is CDL in either a string or a file.
    If given a string, will either save a CDL file, or pass text directly.
    A netcdf output file is always created, at the given path.

    Parameters
    ----------
    cdl_str : str or None
        String containing a CDL description of a netcdf file.
        If None, 'cdl_path' must be an existing file.
    cdl_path : str or None
        Path of temporary text file where cdl_str is written.
        If None, 'cdl_str' must be given, and is piped direct to ncgen.
    nc_path : str
        Path of temporary netcdf file where converted result is put.

    Notes
    -----
    For legacy reasons, the path args are 'str's not 'Path's.

    """
    if cdl_str and cdl_path:
        with open(cdl_path, "w") as f_out:
            f_out.write(cdl_str)
    if cdl_path:
        # Create netcdf from stored CDL file.
        call_args = [NCGEN_PATHSTR, "-k3", "-o", nc_path, cdl_path]
        call_kwargs = {}
    else:
        # No CDL file : pipe 'cdl_str' directly into the ncgen program.
        if not cdl_str:
            raise ValueError("Must provide either 'cdl_str' or 'cdl_path'.")
        call_args = [NCGEN_PATHSTR, "-k3", "-o", nc_path]
        call_kwargs = dict(input=cdl_str, encoding="ascii")

    subprocess.run(call_args, check=True, **call_kwargs)


# CDL to create a reference file with "all" features included.
_base_cdl = """
netcdf everything {
dimensions:
    x = 2 ;
    y = 3 ;
    strlen = 5 ;
variables:
    int x(x) ;
        x:name = "var_x" ;
    int var_2d(x, y) ;
    uint var_u8(x) ;
    float var_f4(x) ;
    double var_f8(x) ;
    char var_str(x, strlen) ;
    int other(x) ;
        other:attr_int = 1 ;
        other:attr_float = 2.f ;
        other:attr_double = 2. ;
        other:attr_string = "this" ;
    int masked_int(y) ;
        masked_int:_FillValue = -3 ;
    int masked_float(y) ;
        masked_float:_FillValue = -4 ;

// global attributes:
        :global_attr_1 = "one" ;
        :global_attr_2 = 2 ;

group: grp_1 {
    dimensions:
        y = 7 ;
    variables:
        int parent_dim(x) ;
        int own_dim(y) ;
} // group grp_1

group: grp_2 {
    variables:
        int grp2_x(x) ;
} // group grp_2
}
"""


def comparable_cdl(text: str) -> List[str]:
    """
    Convert a CDL string to a list of stripped lines, with certain problematic things
    removed.  The resulting list of strings should be comparable between identical
    netcdf files.

    """
    lines = text.split("\n")
    lines = [line.strip(" \t\n") for line in lines]
    lines = [
        line
        for line in lines
        if (
            len(line)
            # Exclude global pseudo-attribute (some versions of ncdump)
            and ":_NCProperties =" not in line
        )
    ]
    # Also exclude the first line, which includes the filename
    hdr, lines = lines[0], lines[1:]
    assert hdr.startswith("netcdf") and hdr[-1] == "{"
    return lines


@pytest.fixture(scope="module")
def testdata_cdl(tmp_path_factory):
    tmpdir_path = tmp_path_factory.mktemp("cdltests")
    cdl_path = tmpdir_path / "tmp_nccompare_test.cdl"
    nc_path = tmpdir_path / "tmp_nccompare_test.nc"
    ncgen_from_cdl(cdl_str=_base_cdl, cdl_path=cdl_path, nc_path=nc_path)
    bytes = subprocess.check_output([NCDUMP_PATHSTR, "-h", nc_path])
    cdl_regen_text = bytes.decode()
    return cdl_regen_text


def test_ncgen_from_cdl(testdata_cdl):
    # Integration test for 'ncgen_from_cdl' (! tests-of-tests !)
    def text_lines_nowhitespace(text):
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        lines = [
            "".join(char for char in line.strip() if not char.isspace())
            for line in lines
        ]
        lines = [line for line in lines if line]
        return lines

    lines_result = text_lines_nowhitespace(testdata_cdl)
    lines_original = text_lines_nowhitespace(_base_cdl)

    # Full original lines definitely do NOT match, because dataset name =filename
    assert lines_result != lines_original

    # Lines beyond 1st may STILL not match because of _NCProperties
    # ... however, skipping that line (and first line), they *should* match
    lines_result = [line for line in lines_result if "_NCProp" not in line]
    assert lines_result[1:] == lines_original[1:]


def test_comparable_cdl(testdata_cdl):
    # Integration test for 'comparable_cdl' (! tests-of-tests !)
    cdl_lines = comparable_cdl(_base_cdl)
    dump_lines = comparable_cdl(testdata_cdl)
    assert cdl_lines == dump_lines
