from doctest import ELLIPSIS as ELLIPSIS_FLAG
from run_doctests import run_doctest_paths, _parser, parserargs_as_kwargs

# tstargs = ['ncdata', '-da', '--options', 'verbose=1']

tstargs = [
    "/home/users/patrick.peglar/git/ncdata/docs/userdocs/user_guide/howtos.rst",
    "--options",
    "verbose=1",
]


args = _parser.parse_args(tstargs)
kwargs = parserargs_as_kwargs(args)
# if not "options" in kwargs:
#     kwargs["options"] = "ELLIPSIS=1"
run_doctest_paths(**kwargs)

#
# Currently good:
# $ tools/run_doctests.py docs/userdocs/getting_started/introduction.rst -o "optionflags=8"
# $ tools/run_doctests.py docs/userdocs/getting_started/*.rst -vo "optionflags=8"
