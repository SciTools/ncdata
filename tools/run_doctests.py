#!/usr/bin/env python3
import argparse
import doctest
import importlib
import traceback
from pathlib import Path
import pkgutil
import sys
import warnings


def list_modules_recursive(
    module_importname: str, include_private: bool = True,
    exclude_matches: list[str] = []
):
    module_names = [module_importname]
    # Identify module from its import path (no import -> fail back to caller)
    try:
        error = None
        module = importlib.import_module(module_importname)
    except Exception as exc:
        print(f"\n\nIMPORT FAILED: {module_importname}\n")
        error = exc

    if error is None:
        # Add sub-modules to the list
        # Get the filepath of the module base directory
        module_filepath = Path(module.__file__)
        if module_filepath.name == "__init__.py":
            search_filepath = str(module_filepath.parent)
            for _, name, ispkg in pkgutil.iter_modules([search_filepath]):
                if name.startswith("_") and not include_private:
                    continue

                submodule_name = module_importname + "." + name
                if any(match in submodule_name for match in exclude_matches):
                    continue

                module_names.append(submodule_name)
                if ispkg:
                    module_names.extend(
                        list_modules_recursive(
                            submodule_name, include_private=include_private
                        )
                    )

    # I don't know why there are duplicates, but there can be.
    result = []
    for name in module_names:
        if name not in result:
            # For some reason, some things get listed twice.
            result.append(name)

    return result


def process_options(opt_str: str) -> dict[str, str]:
    """Convert the "-o/--options" arg into a **kwargs for the doctest function call."""
    # Remove all spaces (think they are never needed).
    opt_str = opt_str.replace(" ", "")
    # Split on commas, and split each one on "=" expecting a simple name=val form
    opts_dict = {}
    if opt_str:  # N.B. to avoid unexpected behaviour: "".split() --> [""]
        for setting_str in opt_str.split(","):
            try:
                name, val = setting_str.split("=")

                # Detect + translate numberic and boolean values.
                bool_vals = {"true": True, "false": False}
                if val.isdigit():
                    val = int(val)
                elif val.lower() in bool_vals:
                    val = bool_vals[val.lower()]

            except ValueError:
                msg = f"Invalid option setting {setting_str!r}, expected 'name=value' only."
                raise ValueError(msg)

            opts_dict[name] = val

    return opts_dict


def run_doctest_paths(
    paths: list[str],
    paths_are_modules:bool = False,
    recurse_modules: bool = False,
    include_private_modules: bool = False,
    exclude_matches: list[str] = [],
    option_kwargs: dict = {},
    verbose: bool = False,
    dry_run: bool = False,
    stop_on_failure: bool = False,
):
    n_total_fails, n_total_tests, n_paths_tested = 0, 0, 0
    if verbose:
        print(
            "RUNNING run_doctest("
            f"paths={paths!r}"
            f", paths_are_modules={paths_are_modules!r}"
            f", recurse_modules={recurse_modules!r}"
            f", include_private_modules={include_private_modules!r}"
            f", exclude_matches={exclude_matches!r}"
            f", option_kwargs={option_kwargs!r}"
            f", verbose={verbose!r}"
            f", dry_run={dry_run!r}"
            f", stop_on_failure={stop_on_failure!r}"
            ")"
        )
    if dry_run:
        verbose = True

    warnings.simplefilter("ignore")

    if paths_are_modules:
        doctest_function = doctest.testmod
        if recurse_modules:
            module_paths = []
            for path in paths:
                module_paths += list_modules_recursive(
                    path, include_private=include_private_modules,
                    exclude_matches=exclude_matches
                )
            paths = module_paths

    else:  # paths are filepaths
        doctest_function = doctest.testfile
        # Fix options : TODO this is not very clever, think of something better??
        if not "module_relative" in option_kwargs:
            option_kwargs["module_relative"] = False
        if not "verbose" in option_kwargs:
            option_kwargs["verbose"] = False
        if not "optionflags" in option_kwargs:
            default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
            option_kwargs["optionflags"] = default_flags

    for path in paths:
        if verbose:
            print(f"\n-----\ndoctest.{doctest_function.__name__}: {path!r}")
        if not dry_run:
            op_fail = None
            if paths_are_modules:
                try:
                    arg = importlib.import_module(path)
                except Exception as exc:
                    op_fail = exc
            else:
                arg = path

            if op_fail is None:
                try:
                    n_fails, n_tests = doctest_function(arg, **option_kwargs)
                    n_total_fails += n_fails
                    n_total_tests += n_tests
                    n_paths_tested += 1
                    if n_fails:
                        print(f"\nERRORS from doctests in path: {arg}\n")
                except Exception as exc:
                    op_fail = exc

            if op_fail is not None:
                n_total_fails += 1
                print(f"\n\nERROR occurred at {path!r}: {op_fail}\n")
                if isinstance(op_fail, doctest.UnexpectedException):
                    # This is what happens with "-o raise_on_error=True", which is the
                    #  Python call equivalent of "-o FAIL_FAST" in the doctest CLI.
                    print(f"Doctest caught exception: {op_fail}")
                    traceback.print_exception(*op_fail.exc_info)

            if n_total_fails > 0 and stop_on_failure:
                break

    if verbose or n_total_fails > 0:
        # Print a final report
        msgs = ["", "=====", "run_doctest: FINAL REPORT"]
        if dry_run:
            msgs += ["(DRY RUN: no actual tests)"]
        elif stop_on_failure and n_total_fails > 0:
            msgs += ["(FAIL FAST: stopped at first target with errors)"]

        msgs += [
            f"    paths tested    = {n_paths_tested}",
            f"    tests completed = {n_total_tests}",
            f"    errors          = {n_total_fails}",
            ""
        ]
        if n_total_fails > 0:
            msgs += ["FAILED."]
        else:
            msgs += ["OK."]

        print('\n'.join(msgs))

    return n_total_fails


_parser = argparse.ArgumentParser(
    prog="run_doctests",
    description="Runs doctests in docs files, or docstrings in packages.",
)
_parser.add_argument(
    "-m", "--module", action="store_true",
    help="Paths are module paths (xx.yy.zz), instead of filepaths."
)
_parser.add_argument(
    "-r",
    "--recursive",
    action="store_true",
    help="If set, include submodules (only applies with -m).",
)
_parser.add_argument(
    "-p",
    "--publiconly",
    action="store_true",
    help="If set, exclude private modules (only applies with -m and -r)",
)
_parser.add_argument(
    "-e",
    "--exclude",
    action="append",
    help="Match fragments of paths to exclude.",
)
_parser.add_argument(
    "-o",
    "--options",
    nargs="?",
    help=(
        "doctest function kwargs (string)"
        ", e.g. \"report=False, raise_on_error=True, optionflags=8\"."
    ),
    type=str,
    default="",
)
_parser.add_argument(
    "-v", "--verbose", action="store_true", help="Show details of each action."
)
_parser.add_argument(
    "-d",
    "--dryrun",
    action="store_true",
    help="Only print the names of modules/files which *would* be tested.",
)
_parser.add_argument(
    "-f",
    "--stop-on-fail",
    action="store_true",
    help="If set, stop at the first path with an error (else continue to test all).",
)
_parser.add_argument(
    "paths",
    nargs="*",
    help="docs filepaths, or module paths (not both).",
    type=str,
    default=[],
)


def parserargs_as_kwargs(args):
    return dict(
        paths=args.paths,
        paths_are_modules=args.module,
        recurse_modules=args.recursive,
        include_private_modules=not args.publiconly,
        exclude_matches=args.exclude or [],
        option_kwargs=process_options(args.options),
        verbose=args.verbose,
        dry_run=args.dryrun,
        stop_on_failure=args.stop_on_fail,
    )


if __name__ == "__main__":
    args = _parser.parse_args(sys.argv[1:])
    kwargs = parserargs_as_kwargs(args)
    n_errs = run_doctest_paths(**kwargs)
    exit(1 if n_errs > 0 else 0)
