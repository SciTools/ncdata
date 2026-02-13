"""
A proof-of-concept example workflow for :func:`ncdata.iris.from_iris`.

Check that conversion succeeds and print the resulting dataset.
"""

import iris

from ncdata.iris import from_iris

from tests import testdata_dir


def example_ncdata_from_iris():
    """Demonstrate loading from iris and printing the NcData object."""
    print("")
    print("==============")
    print("TEMPORARY: iris save-to-ncdata test")
    iris.FUTURE.datum_support = True
    filepath = testdata_dir / "toa_brightness_temperature.nc"
    cubes = iris.load(filepath)
    ds = from_iris(cubes)
    # show result
    print(ds)


if __name__ == "__main__":
    # TODO: save only for now :  TBD add a load case.
    example_ncdata_from_iris()
