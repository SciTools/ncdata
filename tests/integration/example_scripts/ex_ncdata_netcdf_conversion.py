"""
A proof-of-concept example workflow for :mod:`ncdata.netcdf4`.

Showing loading and saving ncdata to/from netcdf files.
"""

import tempfile
from pathlib import Path
from shutil import rmtree

import netCDF4 as nc
import numpy as np

from ncdata import NcData, NcDimension, NcVariable
from ncdata.netcdf4 import from_nc4, to_nc4
from ncdata.utils import dataset_differences

from tests import testdata_dir


def example_nc4_load_save_roundtrip():  # noqa: D103
    """Load an existing file ; save to netcdf4 ; check they are the same."""
    print("\n----\nNetcdf4 load-save example.")

    filepath = testdata_dir / "toa_brightness_temperature.nc"
    ncdata = from_nc4(filepath)

    tempdir_path = Path(tempfile.mkdtemp())
    try:
        filepath2 = tempdir_path / "temp_nc_output.nc"
        to_nc4(ncdata, filepath2)

        result = dataset_differences(filepath, filepath2)
        equals_result = result == []
        print("\nFiles compare? :", equals_result)
        assert equals_result

    finally:
        rmtree(tempdir_path)

    print("\n== Netcdf4 load-save roundtrip finished OK.\n")


def example_nc4_save_reload_unlimited_roundtrip():
    """Create arbitrary ncdata ; save to netcdf4 ; re-load and check similarities."""
    print("\n----\nNetcdf4 save-load example.")

    ncdata = NcData()
    len_x, len_y = 4, 2
    ncdata.dimensions["dim_x"] = NcDimension("dim_x", len_x, unlimited=True)
    ncdata.dimensions["dim_y"] = NcDimension("dim_y", len_y)
    ncdata.variables["var_x"] = NcVariable(
        "var_x",
        dimensions=["dim_x"],
        dtype=np.float32,
        data=np.arange(4),
        # Just an an attribute for the sake of it.
        attributes={"varattr1": 1},
    )
    ncdata.avals["globalattr1"] = "one"
    print("Source ncdata object:")
    print(ncdata)

    tempdir_path = Path(tempfile.mkdtemp())
    try:
        print("")
        filepath = tempdir_path / "temp_nc_2_unlim.nc"
        print("Saving to file ", filepath)
        to_nc4(ncdata, filepath)
        import os

        print("\n\nCreated output:")
        os.system(f"ncdump -h {filepath!s}")
        print("\n")

        ds = nc.Dataset(filepath)
        assert list(ds.dimensions.keys()) == ["dim_x", "dim_y"]
        assert ds.dimensions["dim_x"].isunlimited()

        assert list(ds.variables.keys()) == ["var_x"]
        assert ds.variables["var_x"].ncattrs() == ["varattr1"]

        assert ds.ncattrs() == ["globalattr1"]
        assert ds.getncattr("globalattr1") == "one"
        ds.close()

        # Now readback
        ds_back = from_nc4(filepath)
        print("\nRead-back ncdata from file:")
        print(ds_back)

        assert list(ds_back.dimensions.keys()) == ["dim_x", "dim_y"]
        dimx, dimy = [ds_back.dimensions[name] for name in ("dim_x", "dim_y")]
        assert dimx.size == len_x and dimx.unlimited
        assert dimy.size == len_y and not dimy.unlimited

        assert list(ds_back.variables.keys()) == ["var_x"]
        varx = ds_back.variables["var_x"]

        assert list(varx.avals.keys()) == ["varattr1"]
        assert varx.avals["varattr1"] == 1

        assert list(ds_back.avals.keys()) == ["globalattr1"]
        assert ds_back.avals["globalattr1"] == "one"

    finally:
        rmtree(tempdir_path)

    print("\n== Netcdf4 save-load roundtrip finished OK.\n")


if __name__ == "__main__":
    example_nc4_load_save_roundtrip()
    example_nc4_save_reload_unlimited_roundtrip()
