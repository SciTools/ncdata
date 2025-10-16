#!/usr/bin/env python3
import argparse
import doctest
import importlib
from pathlib import Path
import pkgutil
import sys

def list_modules_recursive(module_importname: str, include_private: bool = False):
    module_names = [module_importname]
    # Identify module from its import path (no import -> fail back to caller)
    module = importlib.import_module(module_importname)
    # Get the filepath of the module base directory
    module_filepath = str(Path(module.__file__).parent)
    for _, name, ispkg in pkgutil.iter_modules([module_filepath]):
        if not name.startswith("_") or include_private:
            submodule_name = module_importname + "." + name
            module_names.append(submodule_name)
            if ispkg:
                module_names.extend(list_modules_recursive(
                    submodule_name, include_private=include_private
                ))
    # I don't know why there are duplicates, but there can be.
    result = []
    for name in module_names:
        if name not in result:
            # For some reason, some things get listed twice.
            result.append(name)

    return result


def process_options(opt_str:str) -> dict[str, str]:
    # First collapse spaces around equals signs."
    while " =" in opt_str:
        opt_str = opt_str.replace(" =", "=")
    while "= " in opt_str:
        opt_str = opt_str.replace("= ", "=")
    # Collapse (remaining) duplicate spaces
    while "  " in opt_str:
        opt_str = opt_str.replace("  ", " ")
    # Split on spaces, and split each one on "=" expecting a simple name=val form
    opts_dict = {}
    if opt_str:  # N.B. to avoid unexpected behaviour from "".split()
        for setting_str in opt_str.split(" "):
            try:
                name, val = setting_str.split("=")

                # Translate certain things (but do not exec!!)
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
    opts_str:str,
    verbose:bool = False,
    dry_run:bool=False,
    do_all:bool=False
):
    if verbose:
        print(
            "RUNNING run_doctest("
            f"paths={paths!r}"
            f", opts_str={opts_str!r}"
            f", verbose={verbose!r}"
            f", dry_run={dry_run!r}"
            f", do_all={do_all!r}"
            ")"
        )
    if dry_run:
        verbose = True
    opts_kwargs = process_options(opts_str)
    finished = False
    try:
        module_paths = []
        for path in paths:
            module_paths += list_modules_recursive(path, include_private=do_all)
        for path in module_paths:
            if verbose:
                print(f"\ndoctest.testmod: {path!r}")
            if not dry_run:
                module = importlib.import_module(path)
                doctest.testmod(module, **opts_kwargs)
        finished = True
    except (ImportError, ModuleNotFoundError, TypeError):
        # TODO: this list is "awkward" !
        pass

    if not finished:
        # Module search failed : treat paths as (documentation) filepaths instead
        # Fix options : TODO this is not very clever, think of something better??
        if not "module_relative" in opts_kwargs:
            opts_kwargs["module_relative"] = False
        if not "optionflags" in opts_kwargs:
            default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
            opts_kwargs["optionflags"] = default_flags
        for path in paths:
            if verbose:
                print(f"\ndoctest.testfile: {path!r}")
            if not dry_run:
                doctest.testfile(path, **opts_kwargs)


_parser = argparse.ArgumentParser(
    prog="run_doctests",
    description="Runs doctests in docs files, or docstrings in packages."
)
_parser.add_argument(
    "-o", "--options", nargs="?",
    help="doctest options settings (as a string).",
    type=str, default=""
)
_parser.add_argument(
    "-v", "--verbose", action="store_true",
    help="Show actions."
)
_parser.add_argument(
    "-d", "--dryrun", action="store_true",
    help="Only print the names of modules/files which *would* be tested."
)
_parser.add_argument(
    "-a", "--all", action="store_true",
    help="If set, include private files/modules "
)
_parser.add_argument(
    "paths", nargs="*",
    help="docs filepaths, or module paths (not both).",
    type=str, default=[]
)

def parserargs_as_kwargs(args):
    return dict(
        paths=args.paths,
        opts_str=args.options,
        verbose=args.verbose,
        dry_run=args.dryrun,
        do_all=args.all
    )


if __name__ == '__main__':
    args = _parser.parse_args(sys.argv[1:])
    kwargs = parserargs_as_kwargs(args)
    run_doctest_paths(**kwargs)
