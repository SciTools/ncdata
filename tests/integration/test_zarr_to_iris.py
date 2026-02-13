"""Test conversion of remote and local Zarr store to iris Cube."""

from importlib.resources import files as importlib_files
from pathlib import Path

import fsspec
import iris
import pytest
import xarray as xr
import zarr

from ncdata.iris_xarray import cubes_from_xarray as conversion_func

zarr_major_version = int(zarr.__version__.split(".")[0])


def _return_kwargs():
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    return {
        "consolidated": True,
        "decode_times": time_coder,
        "engine": "zarr",
        "chunks": {},
        "backend_kwargs": {},
    }


def _run_checks(cube):
    """Run some standard checks."""
    assert cube.var_name == "q"
    assert cube.standard_name == "specific_humidity"
    assert cube.long_name is None
    coords = cube.coords()
    coord_names = [coord.standard_name for coord in coords]
    assert "longitude" in coord_names
    assert "latitude" in coord_names


def test_load_zarr2_local():
    """Test loading a Zarr2 store from local FS."""
    zarr_path = (
        Path(importlib_files("tests"))
        / "testdata"
        / "zarr-sample-data"
        / "example_field_0.zarr2"
    )

    xr_kwargs = _return_kwargs()
    zarr_xr = xr.open_dataset(zarr_path, **xr_kwargs)

    cubes = conversion_func(zarr_xr)

    assert len(cubes) == 1
    cube = cubes[0]
    _run_checks(cube)


@pytest.mark.skipif(
    zarr_major_version < 3, reason="Zarr3 file not supported by zarr v2."
)
def test_load_zarr3_local():
    """Test loading a Zarr3 store from local FS."""
    zarr_path = (
        Path(importlib_files("tests"))
        / "testdata"
        / "zarr-sample-data"
        / "example_field_0.zarr3"
    )

    xr_kwargs = _return_kwargs()
    zarr_xr = xr.open_dataset(zarr_path, **xr_kwargs)

    cubes = conversion_func(zarr_xr)

    assert len(cubes) == 1
    cube = cubes[0]
    _run_checks(cube)


def _is_url_ok(url):
    fs = fsspec.filesystem("http")
    valid_zarr = True
    try:
        fs.open(str(url) + "/zarr.json", "rb")  # Zarr3
    except Exception:  # noqa: BLE001
        try:
            fs.open(str(url) + "/.zmetadata", "rb")  # Zarr2
        except Exception:  # noqa: BLE001
            valid_zarr = False

    return valid_zarr


S3_TEST_PATH = (
    "https://uor-aces-o.s3-ext.jc.rl.ac.uk/"
    "esmvaltool-zarr/pr_Amon_CNRM-ESM2-1_02Kpd-11_r1i1p2f2_gr_200601-220112.zarr3"
)

# Check 3 files, as we apparently sometimes get 'partial success'
_S3_accessible = all(
    _is_url_ok(S3_TEST_PATH + subpath)
    for subpath in ["", "/.zattrs", "/.zmetadata"]
)


@pytest.mark.skipif(
    zarr_major_version < 3, reason="Zarr3 file not supported by zarr v2."
)
@pytest.mark.skipif(not _S3_accessible, reason="S3 url not accessible")
def test_load_remote_zarr():
    """Test loading a remote Zarr store.

    This is a ~250MB compressed Zarr in an S3 bucket.
    Conversion is done fully lazily, by passing chunks={}
    to Xarray loader. Test takes ~3-4s and needs ~400MB res mem.
    """
    zarr_path = S3_TEST_PATH

    xr_kwargs = _return_kwargs()
    zarr_xr = xr.open_dataset(zarr_path, **xr_kwargs)

    cubes = conversion_func(zarr_xr)

    assert isinstance(cubes, iris.cube.CubeList)
    assert len(cubes) == 1
    assert cubes[0].has_lazy_data()
