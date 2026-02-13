"""
Tests for :func:`ncdata.iris.to_iris`.

Most practical behaviours relating to different types of dataset content are tested in
:mod:`tests/integration/test_roundtrips_iris`.  For example, different datatypes,
the use of groups, passing of data arrays and attributes.

This module only tests some specific API and behaviours of the top-level function, not
covered by the generic 'roundtrip' testcases.
"""

import dask.array as da
from iris._constraints import NameConstraint
from iris.cube import CubeList
import numpy as np

from ncdata import NcData, NcDimension, NcVariable
from ncdata.iris import to_iris

from tests import MonitoredArray


def test_lazy_nocompute():
    """Check that converting to iris preserves lazy data without fetching it."""
    np_data = np.array([1.23, 4, 7.3, 4.1], dtype=np.float32)
    monitored_array = MonitoredArray(np_data)
    dask_array = da.from_array(monitored_array, chunks=(2,), meta=np.ndarray)
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(name="var_x", dimensions=["x"], data=dask_array)
        ],
    )

    # convert to cubes
    (cube,) = to_iris(ncdata)

    # Check that no data was read, then fetch it and check again
    assert cube.has_lazy_data()
    cube_data = cube.core_data()
    assert monitored_array._accesses == []
    real_data = cube_data.compute()
    # should have fetched in 2 chunks
    # NOTE: order of access is not guaranteed, hence 'sorted'.
    assert sorted(monitored_array._accesses) == [
        (slice(0, 2),),
        (slice(2, 4),),
    ]
    assert np.all(real_data == np_data)


def test_real_nocopy():
    """Check that converting to iris does not copy real data."""
    real_array = np.array([1.23, 4, 7.3, 4.1], dtype=np.float32)
    monitored_array = MonitoredArray(real_array)
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(name="var_x", dimensions=["x"], data=monitored_array)
        ],
    )

    # convert to cubes
    (cube,) = to_iris(ncdata)

    # Check that the cube data has not been fetched
    # - but N.B. a zero-length slice is fetched when making a lazy wrapper.
    assert monitored_array._accesses == [(slice(0, 0, None),)]
    # Check value equivalence, and check accesses again
    monitored_array._accesses = []
    assert np.all(cube.data == real_array)
    # It has now been fetched, in a single chunk.
    assert monitored_array._accesses == [(slice(0, 4, None),)]


def sample_2vars_ncdata():
    """Construct a test ncdata that loads as 2 cubes with a common dimension."""
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(
                name="var1",
                dimensions=["x"],
                data=np.arange(3, dtype=np.int16),
            ),
            NcVariable(
                name="var2",
                dimensions=["x"],
                data=np.arange(3.0, dtype=np.float32),
            ),
        ],
    )
    return ncdata


def test_multiple_cubes():
    """Check a case where we get multiple cubes."""
    ncdata = sample_2vars_ncdata()

    # convert to cubes
    cubes = to_iris(ncdata)

    # check structure of result
    assert isinstance(cubes, CubeList)
    assert len(cubes) == 2
    # N.B. order of cubes can vary !
    assert sorted(cube.name() for cube in cubes) == ["var1", "var2"]


def test_kwargs__load_by_name():
    """Check kwargs usage : select one cube by name."""
    ncdata = sample_2vars_ncdata()

    # convert to cubes, but select only one.
    cubes = to_iris(ncdata, constraints=NameConstraint(var_name="var2"))

    # check structure of result
    assert isinstance(cubes, CubeList)
    assert len(cubes) == 1
    assert cubes[0].name() == "var2"


def test_iris_loadchain():
    """Check that standard iris load-chain processing is applied, including a merge."""

    # Create a pair of datasets with mergeable variables.
    # Start by making one, with a scalar 'z' coord.
    ncdata = NcData(
        dimensions=[NcDimension("x", 3)],
        variables=[
            NcVariable(
                name="v_data",
                dimensions=["x"],
                data=[1, 2, 3],
                attributes={"long_name": "data", "coordinates": "v_z"},
            ),
            NcVariable(
                name="v_z",
                dimensions=[],
                data=[1],
                attributes={"long_name": "z"},
            ),
        ],
    )

    # Duplicate to get a second dataset, then change the z value.
    # N.B. we need 2 datasets, as Iris won't match coords with different var-names.
    ncdata2 = ncdata.copy()

    # N.B. must **copy data array** before modifying, as var.copy() doesn't do so.
    data = ncdata2.variables["v_z"].data
    data = data.copy()
    data[0] = 2
    ncdata2.variables["v_z"].data = data

    # Loading should now merge these 2 into one cube.
    cubes = to_iris([ncdata, ncdata2])

    assert isinstance(cubes, CubeList)
    assert len(cubes) == 1
    (cube,) = cubes
    assert cube.long_name == "data"
    assert cube.shape == (2, 3)
    assert cube.coords("z", dim_coords=True)
