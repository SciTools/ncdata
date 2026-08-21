import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import run_doctests
from run_doctests import list_filepaths_recursive, list_modules_recursive


def make_dirs_and_files(pattern, basepath):
    for dirname, pyfiles in pattern.items():
        dirpath = basepath / dirname.replace(".", "/")
        if not dirpath.exists():
            dirpath.mkdir()
        for content in pyfiles:
            with open(dirpath / content, "w") as mainfile:
                # Just create an empty top-level file
                pass


@pytest.fixture
def tempmodules(tmp_path):
    """Create  temporary discoverable modules."""
    basename = "tmp_test_module"
    patt = {
        basename: ["__init__.py", "s0.py", "s1.py"],
        basename + ".sm1": ["__init__.py", "s1.py", "s2.py", "_ps1.py"],
        basename + ".sm1.ssm1": ["__init__.py", "s3.py", "s4.py"],
        basename + ".sm1.ssm2": ["__init__.py", "s4.py", "s5.py", "_ps2.py"],
        basename + ".notamodule": ["xx1.py", "xx2.py"],
        basename + ".sm2": ["__init__.py", "s6.py"],
    }
    make_dirs_and_files(patt, tmp_path)
    try:
        sys.path.append(str(tmp_path))
        yield "tmp_test_module"
    finally:
        sys.path.remove(str(tmp_path))


@pytest.fixture
def tempsources(tmp_path):
    """Create temporary sourcefiles."""
    patt = {
        "maindir": ["s0.rst", "s1.rst", "ignore.this"],
        "maindir/subdir1": ["s1.rst", "s2.rst", "_px1.rst"],
        "maindir/subdir1/subsubdir1": ["s3.rst", "s4.rst"],
        "maindir/subdir1/subsubdir2": ["s4.rst", "s5.rst", "_px2.rst"],
        "maindir/subdir2": ["s6.rst"],
    }
    make_dirs_and_files(patt, tmp_path)
    return str(tmp_path / "maindir")  # targets must be str, not Path


@pytest.fixture
def badsources(tempsources):
    # Same as tempsources, but put a failing doctest in the first source file
    filepath = Path(tempsources) / "s0.rst"
    with open(filepath, "wt") as f_out:
        f_out.write(">>> 1\n0\n")
    return tempsources


class TestListModules:
    def test_import(self, tempmodules):
        """Check that 'tempmodules' puts the test module on the import path."""
        import tmp_test_module

        assert "<module 'tmp_test_module' from" in str(tmp_test_module)

    def test_nomatch(self):
        result = list_modules_recursive("not.exists")
        assert result == ["not.exists"]

    def test_recurse(self, tempmodules):
        result = list_modules_recursive(tempmodules)
        assert result == [
            "tmp_test_module",
            "tmp_test_module.s0",
            "tmp_test_module.s1",
            "tmp_test_module.sm1",
            "tmp_test_module.sm1._ps1",
            "tmp_test_module.sm1.s1",
            "tmp_test_module.sm1.s2",
            "tmp_test_module.sm1.ssm1",
            "tmp_test_module.sm1.ssm1.s3",
            "tmp_test_module.sm1.ssm1.s4",
            "tmp_test_module.sm1.ssm2",
            "tmp_test_module.sm1.ssm2._ps2",
            "tmp_test_module.sm1.ssm2.s4",
            "tmp_test_module.sm1.ssm2.s5",
            "tmp_test_module.sm2",
            "tmp_test_module.sm2.s6",
        ]

    def test_recurse_noprivate(self, tempmodules):
        result = list_modules_recursive(tempmodules, include_private=False)
        assert result == [
            "tmp_test_module",
            "tmp_test_module.s0",
            "tmp_test_module.s1",
            "tmp_test_module.sm1",
            # 'tmp_test_module.sm1._ps1',
            "tmp_test_module.sm1.s1",
            "tmp_test_module.sm1.s2",
            "tmp_test_module.sm1.ssm1",
            "tmp_test_module.sm1.ssm1.s3",
            "tmp_test_module.sm1.ssm1.s4",
            "tmp_test_module.sm1.ssm2",
            # 'tmp_test_module.sm1.ssm2._ps2',
            "tmp_test_module.sm1.ssm2.s4",
            "tmp_test_module.sm1.ssm2.s5",
            "tmp_test_module.sm2",
            "tmp_test_module.sm2.s6",
        ]


class TestListSources:
    def test_nonexist(self, tempsources):
        result = list_filepaths_recursive("none")
        assert result == [Path("none")]

    def test_toponly(self, tempsources, tmp_path):
        result = list_filepaths_recursive(tempsources)
        assert result == []

    def test_recurse_all(self, tempsources, tmp_path):
        result = list_filepaths_recursive(tempsources + "/**/*")
        assert result == [
            tmp_path / pathstr
            for pathstr in [
                "maindir/s0.rst",
                "maindir/s1.rst",
                "maindir/ignore.this",
                "maindir/subdir1/s1.rst",
                "maindir/subdir1/s2.rst",
                "maindir/subdir1/_px1.rst",
                "maindir/subdir2/s6.rst",
                "maindir/subdir1/subsubdir1/s3.rst",
                "maindir/subdir1/subsubdir1/s4.rst",
                "maindir/subdir1/subsubdir2/s4.rst",
                "maindir/subdir1/subsubdir2/s5.rst",
                "maindir/subdir1/subsubdir2/_px2.rst",
            ]
        ]

    def test_recurse_rsts(self, tempsources, tmp_path):
        result = list_filepaths_recursive(tempsources + "/**/*.rst")
        assert result == [
            tmp_path / pathstr
            for pathstr in [
                "maindir/s0.rst",
                "maindir/s1.rst",
                # 'maindir/ignore.this',
                "maindir/subdir1/s1.rst",
                "maindir/subdir1/s2.rst",
                "maindir/subdir1/_px1.rst",
                "maindir/subdir2/s6.rst",
                "maindir/subdir1/subsubdir1/s3.rst",
                "maindir/subdir1/subsubdir1/s4.rst",
                "maindir/subdir1/subsubdir2/s4.rst",
                "maindir/subdir1/subsubdir2/s5.rst",
                "maindir/subdir1/subsubdir2/_px2.rst",
            ]
        ]

    def test_recurse_namematch_1(self, tempsources, tmp_path):
        """Search for '*s*.rst'."""
        result = list_filepaths_recursive(
            tempsources + "/**/*s*.rst",
        )
        assert result == [
            tmp_path / pathstr
            for pathstr in [
                "maindir/s0.rst",
                "maindir/s1.rst",
                "maindir/subdir1/s1.rst",
                "maindir/subdir1/s2.rst",
                "maindir/subdir2/s6.rst",
                "maindir/subdir1/subsubdir1/s3.rst",
                "maindir/subdir1/subsubdir1/s4.rst",
                "maindir/subdir1/subsubdir2/s4.rst",
                "maindir/subdir1/subsubdir2/s5.rst",
            ]
        ]

    def test_recurse_namematch_2(self, tempsources, tmp_path):
        """Search for '*1.rst'."""
        result = list_filepaths_recursive(
            tempsources + "/**/*1.rst",
        )
        assert result == [
            tmp_path / pathstr
            for pathstr in [
                "maindir/s1.rst",
                "maindir/subdir1/s1.rst",
                "maindir/subdir1/_px1.rst",
            ]
        ]

    def test_recurse_match_nonexist_glob(self, tempsources, tmp_path):
        result = list_filepaths_recursive(tempsources + "/*pqr*")
        assert result == []

    def test_recurse_match_nonexist_noglob(self, tempsources, tmp_path):
        search_path = tempsources + "subdir1/non.exist"
        # As not actually a search, returns the given path.
        result = list_filepaths_recursive(search_path)
        assert result == [Path(search_path)]


_env_path = (Path(os.__file__) / "../../../bin/python").resolve()
assert _env_path.exists()
_PYTHON_PATHSTR = str(_env_path)

_doctests_path = Path(run_doctests.__file__).resolve()
_DOCTESTS_PATH = str(_doctests_path)

_RE_ANY_NONBLANK = re.compile(r".*\S.*")


def runmain(*args, expect_rc: int | None = 0) -> list[str]:
    arglist = [_PYTHON_PATHSTR, _DOCTESTS_PATH] + list(args)
    call_data = subprocess.run(arglist, capture_output=True)
    rc = call_data.returncode
    if expect_rc is not None:
        assert rc == expect_rc
    lines = call_data.stdout.decode("ascii").split("\n")
    # for simplicity, remove blank lines.
    lines = [line for line in lines if _RE_ANY_NONBLANK.match(line)]
    return lines


class TestCliSources:
    def test_nopaths_help(self, tempsources):
        result = runmain()
        # Choose some sample lines to show it has output help text.
        test_lines = [
            "Run doctests in docs files, or docstrings in packages.",
            "Notes:",
            "* N.B. use ** to include subdirectories",
        ]
        for test_line in test_lines:
            assert test_line in [line.strip() for line in result]

    def test_multipath(self, tempsources):
        result = runmain("this", "that", "other", "--dryrun")
        assert result == [
            "-----",
            "doctest.testfile: this",
            "-----",
            "doctest.testfile: that",
            "-----",
            "doctest.testfile: other",
            "=====",
            "run_doctest: FINAL REPORT",
            "(DRY RUN: no actual tests)",
            "    paths tested    = 0",
            "    tests completed = 0",
            "    errors          = 0",
            "OK.",
        ]

    def test_basic_error(self, badsources):
        result = runmain(badsources + "/*.rst", "-v", expect_rc=1)
        test_lines = f"""
            paths_are_modules=False, recurse_modules=False, include_private_modules=True, exclude_fragments=[], doctest_kwargs={{'module_relative': False, 'optionflags': 12}}, verbose=True, dry_run=False, stop_on_failure=False
            0/1 OK, 1/1 FAILED in path: {badsources}/s0.rst
            0/0 OK in path: {badsources}/s1.rst
            run_doctest: FINAL REPORT
                paths tested    = 2
                tests completed = 1
                errors          = 1

            FAILED.
        """
        result = "\n".join(result)
        for line in test_lines.split("\n"):
            assert line.strip() in result

    def test_stop_on_fail(self, badsources):
        result = runmain(badsources + "/*.*t", "-vf", expect_rc=1)
        result_fullstr = "\n".join(result)
        assert not "s1.rst" in result_fullstr
        assert (
            f"0/1 OK, 1/1 FAILED in path: {badsources}/s0.rst"
            in result_fullstr
        )
        test_lines = [
            "(FAIL FAST: stopped at first path with errors)",
            "    paths tested    = 1",
            "    tests completed = 1",
            "    errors          = 1",
            "FAILED.",
        ]
        assert result[-5:] == test_lines

    def test_options_passing(self):
        result = runmain(
            "any/*.rst",
            "--dryrun",
            "--verbose",
            "-o",
            "module_relative=false, junk= 3, unknown=this",
        )
        expected = [
            "RUNNING run_doctest(paths=['any/*.rst'], paths_are_modules=False, "
            "recurse_modules=False, include_private_modules=True, exclude_fragments=[], "
            "doctest_kwargs={'module_relative': False, 'junk': 3, 'unknown': 'this', "
            "'optionflags': 12}, verbose=True, dry_run=True, stop_on_failure=False)",
            "=====",
            "run_doctest: FINAL REPORT",
            "(DRY RUN: no actual tests)",
            "    paths tested    = 0",
            "    tests completed = 0",
            "    errors          = 0",
            "OK.",
        ]
        assert result == expected

    def test_options_bad(self, tempsources):
        result = runmain(
            tempsources + "/*.rst",
            "-o",
            "module_relative=false, junk= 3, unknown=this",
            expect_rc=1,
        )
        test_lines = [
            f"ERROR occurred at PosixPath('{tempsources}/s0.rst'):"
            " testfile() got an unexpected keyword argument 'junk'",
            "    paths tested    = 0",
            "    tests completed = 0",
            "    errors          = 2",
        ]
        for line in test_lines:
            assert line in result

    def test_multi_exclude(self, tempsources):
        result = runmain(
            tempsources + "/**/*.rst",
            "-d",
            "-e",
            "s1",
            "-e",
            "subdir2",
        )
        find_str = "doctest.testfile"
        hits = [line for line in result if find_str in line]
        name_prefix = find_str + f": {tempsources}/"
        assert hits == [
            name_prefix + name
            for name in [
                "s0.rst",
                # "s1.rst",
                # "subdir1/s1.rst",
                "subdir1/s2.rst",
                "subdir1/_px1.rst",
                # "subdir2/s6.rst",
                "subdir1/subsubdir1/s3.rst",
                "subdir1/subsubdir1/s4.rst",
                # "subdir1/subsubdir2/s4.rst",
                # "subdir1/subsubdir2/s5.rst",
                # "subdir1/subsubdir2/_px2.rst",
            ]
        ]


class TestCliModules:
    @pytest.mark.parametrize("publiconly", [False, True])
    def test_modules_publiconly(self, publiconly):
        """Check that the '-p' option is passed down."""
        args = [
            "argparse",
            "-mdv",
        ]  # NB list module: don't recurse or run actual tests
        if publiconly:
            args += ["-p"]
        result = runmain(*args)
        print("\n".join(result))
        expect = f"include_private_modules={not publiconly}"
        assert expect in result[0]

    @pytest.mark.parametrize("recurse", [False, True])
    def test_modules_recurse(self, recurse):
        """Check that the '-r' option is passed down."""
        # Test this with a stdlib module which has (just a few) submodules
        args = ["json", "-mdv"]  # NB list modules, don't actually run tests
        if recurse:
            args += ["-r"]
        result = runmain(*args)
        print("\n".join(result))
        expect = f"recurse_modules={recurse}"
        assert expect in result[0]
        n_mods = sum("doctest.testmod:" in line for line in result)
        if recurse:
            assert n_mods > 1
        else:
            assert n_mods == 1

    @pytest.mark.parametrize("exclude", ["noexclude", "exclude"])
    def test_realrun_submodules(self, exclude):
        # Tested via CLI because the exclude logic is now common to modules+sourcefiles
        do_exclude = exclude == "exclude"
        args = ["-mrd", "curses"]
        if do_exclude:
            args += ["-e", "as"]
        result = runmain(*args)
        expected = [
            "-----",
            "doctest.testmod: curses",
        ]
        if not do_exclude:
            expected += [
                "-----",
                "doctest.testmod: curses.ascii",
                "-----",
                "doctest.testmod: curses.has_key",
            ]
        expected += [
            "-----",
            "doctest.testmod: curses.panel",
            "-----",
            "doctest.testmod: curses.textpad",
            "=====",
            "run_doctest: FINAL REPORT",
            "(DRY RUN: no actual tests)",
            "    paths tested    = 0",
            "    tests completed = 0",
            "    errors          = 0",
            "OK.",
        ]
        assert result == expected
