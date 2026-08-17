r"""
An adaptor layer for :mod:`ncdata` to emulate :mod:`netCDF4`.

Primarily, allows an :class:`ncdata.NcData` to masquerade as a
:class:`netCDF4.Dataset` object.

Note:
    This is a low-level interface, exposed publicly for extended experimental uses.
    If you only want to convert **Iris** data to + from :class:`~ncdata.NcData`,
    please use the functions in :mod:`ncdata.iris` instead.

----

These classes contain :class:`~ncdata.NcData`, :class:`~ncdata.NcDimension`,  and
:class:`~ncdata.NcVariable` objects,  but emulate the access APIs of
:class:`netCDF4.Dataset` :class:`netCDF4.Dimension` and :class:`netCDF4.Variable`.

This is provided primarily to support the reuse of the :mod:`iris.fileformats.netcdf`
file format load + save, to convert cubes to + from ncdata objects (and hence,
especially, to convert Iris :class:`~iris.cube.Cube`\s to + from an Xarray
:class:`~xarray.Dataset`
).

Notes
-----
Currently only supports what is required for Iris load/save capability.
In principle, it *should* be possible to use these objects with other packages
expecting a :class:`netCDF4.Dataset` object.  However the API simulation is far from
complete, so this module may need to be extended, in future, to support other such uses.

"""

from typing import Any, Dict, List

import dask.array as da
import netCDF4
import numpy as np

from . import NcData, NcDimension, NcVariable


class _Nc4DatalikeWithNcattrs:
    # A mixin, shared by Nc4DatasetLike and Nc4VariableLike, which adds netcdf-like
    #  attribute operations 'ncattrs / setncattr / getncattr', *AND* extends the local
    #  objects attribute to those things also
    # N.B. "self._ncdata" is the underlying NcData object : either an NcData or
    #  NcVariable object.

    def ncattrs(self) -> List[str]:
        return list(self._ncdata.attributes.keys())

    def getncattr(self, attr: str):
        attrs = self._ncdata.avals
        if attr in attrs:
            result = attrs[attr]
        else:
            # Don't allow it to issue a KeyError, as this upsets 'getattr' usage.
            # Raise an AttributeError instead.
            raise AttributeError(attr)
        return result

    def setncattr(self, attr: str, value):
        # TODO: are we sure we need this translation ??
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        # N.B. using the NcAttribute class for storage also ensures/requires that all
        #  attributes are cast as numpy arrays (so have shape, dtype etc).
        self._ncdata.avals[attr] = value

    # Extend local object attribute access to the ncattrs of the stored data item
    #  (Unpleasant, but I think the Iris load code requires it).

    def __getattr__(self, attr):
        return self.getncattr(attr)

    def __setattr__(self, attr, value):
        if attr in self._local_instance_props:
            # N.B. use _local_instance_props to define standard instance attributes, to
            # avoid a possible endless loop here.
            super().__setattr__(attr, value)
        else:
            self.setncattr(attr, value)

    def set_auto_chartostring(self, value):
        if bool(value):
            msg = "Cannot enable 'auto_chartostring' for Nc4Dataselike or Nc4VariableLike."
            raise ValueError()


class Nc4DatasetLike(_Nc4DatalikeWithNcattrs):
    """
    An object which emulates a :class:`netCDF4.Dataset`.

    It can be both read and written (modified) via its emulated
    :class:`netCDF4.Dataset`-like API.

    The core, contained content object, ``self._ncdata``, is a :class:`ncdata.NcData`.
    This completely defines the parent object state.
    If not provided on init, a new, empty dataset is created.

    """

    _local_instance_props = ("_ncdata", "variables", "dimensions")

    # Needed for Iris to recognise the dataset format.
    file_format = "NETCDF4"

    def __init__(self, ncdata: NcData = None):
        """Create an Nc4DatasetLike, wrapping an :class:`~ncdata.NcData`."""
        if ncdata is None:
            ncdata = NcData()  # an empty dataset
        #: the contained dataset.  If not provided, a new, empty dataset is created.
        self._ncdata = ncdata
        # N.B. we need to create + store our OWN variables, as they are wrappers for
        #  the underlying NcVariable objects, with different properties.
        self.variables = {
            name: Nc4VariableLike._from_ncvariable(ncvar)
            for name, ncvar in self._ncdata.variables.items()
        }
        self.dimensions = {
            name: Nc4DimensionLike(dim)
            for name, dim in self._ncdata.dimensions.items()
        }

    @property
    def groups(self):  # noqa: D102
        return None  # not supported

    def createDimension(self, dimname: str, size: int):  # noqa: D102
        if dimname in self.dimensions:
            msg = f'creating duplicate dimension "{dimname}".'
            raise ValueError(msg)

        # Create a new actual NcDimension in the contained dataset.
        dim = NcDimension(dimname, size)
        self._ncdata.dimensions[dimname] = dim
        # Create a wrapper for it, install that into self, and return it.
        nc4dim = Nc4DimensionLike(dim)
        self.dimensions[dimname] = nc4dim
        return nc4dim

    def createVariable(
        self,
        varname: str,
        datatype: np.dtype,
        dimensions: List[str] = (),
        fill_value=None,
        **kwargs: Dict[str, Any],
    ):  # noqa: D102
        if varname in self.variables:
            msg = f'creating duplicate variable "{varname}".'
            raise ValueError(msg)
        # Add a variable into the underlying NcData object.

        # N.B. to correctly mirror netCDF4, a variable should be created with all-masked
        # content.  For this we need to decode the dims + work out the shape.
        # NOTE: simplistic version here, as we don't support groups.
        shape = tuple(
            self._ncdata.dimensions[dim_name].size for dim_name in dimensions
        )
        # Note: initial content is an all-masked lazy array, to avoid allocating space
        # in memory.  This is not assignable, so we cannot mimic partial writes, but
        # that's okay as we don't currently provide any Nc4VariableLike.__setitem__ :
        # The user will instead read and write Nc4VariableLike._data_array.
        initial_allmasked_data = da.ma.masked_array(
            da.zeros(shape, dtype=datatype), mask=True
        )

        ncvar = NcVariable(
            name=varname,
            dimensions=dimensions,
            data=initial_allmasked_data,
            group=self._ncdata,
        )
        if fill_value is not None:
            ncvar.avals["_FillValue"] = fill_value
        # Note: no valid data is initially assigned, since that is how the netCDF4 API
        # does it.
        self._ncdata.variables[varname] = ncvar
        # Create a netCDF4-like "wrapper" variable + install that here.
        nc4var = Nc4VariableLike._from_ncvariable(ncvar, dtype=datatype)
        self.variables[varname] = nc4var
        return nc4var

    def sync(self):  # noqa: D102
        pass

    def close(self):  # noqa: D102
        self.sync()

    @staticmethod
    def filepath() -> str:  # noqa: D102
        # Note: for now, let's just not care about this.
        # we *might* need this to be an optional defined item on an NcData ??
        # .. or, we might need to store an xarray "encoding" somewhere ?
        # TODO: more thought here ?
        # return self.ncdata.encoding.get("source", "")
        return "<Nc4DatasetLike>"

    def __lt__(self, other):
        """Support comparison operations.

        Not a real operation: we provide "comparison", just so that Iris can load a list
        of them.
        """
        return False


class Nc4VariableLike(_Nc4DatalikeWithNcattrs):
    """
    An object which contains a :class:`ncdata.NcVariable` and emulates a :class:`netCDF4.Variable`.

    The core, contained content object, ``self._ncdata``, is a :class:`~ncdata.NcVariable`.
    This completely defines the parent object state.

    The property ``._data_array`` is detected by Iris to do direct data transfer
    (copy-free and lazy-preserving).

    At present, this object emulates only the *default* read/write behaviour of a
    :class:`netCDF4.Variable`, i.e. :

    *  the underlying NcVariable contains a 'raw' data array, which may be real
       (i.e. numpy) or lazy (i.e. dask).
    *  The ``._data_array`` property read/write interface then applies/removes any
       scaling and masking as it is to be "seen" from the outside.

    That suits how *Iris* reads netCDF4 data, but it won't work if the user wants to
    control the masking/saving behaviour, as you can do in netCDF4.
    Thus, at present, we do *not* provide any of the
    ``set_auto_mask/scale/maskandscale()`` methods.

    """

    _local_instance_props = ("_ncdata", "name", "datatype", "_data_array")

    def __init__(self, ncvar: NcVariable, datatype: np.dtype):  # noqa: D107
        self._ncdata = ncvar
        self.name = ncvar.name
        # Note: datatype must be known at creation, which may be before an actual data
        #  array is assigned on the ncvar.
        self.datatype = np.dtype(datatype)

        # Fix the direct array content.
        array = ncvar.data
        if array is None:
            # temporary empty data (to correctly support never-written content)
            # NOTE: significantly, does *not* allocate an actual full array in memory
            array = da.ma.masked_array(
                da.zeros(self.shape, self.datatype), mask=True
            )

        # Convert from the "inner" raw form to masked-and-scaled, and then back again
        # (by assigning self._data_array).
        # This ensures that our inner data is stored with the correct fill-value, as
        # determined from our attributes and the netCDF4 default fill-values.
        array = self._maskandscale_inner_to_apparent(array)
        self._data_array = array

    @classmethod
    def _from_ncvariable(cls, ncvar: NcVariable, dtype: np.dtype = None):
        if dtype is None:
            dtype = ncvar.dtype
        self = cls(
            ncvar=ncvar,
            datatype=dtype,
        )
        return self

    def _get_scaling(self):
        """
        Determine any scaling settings from attributes.

        Notes
        -----
        * this must be checked dynamically, as the attributes could change.
        """
        scale_factor, add_offset = (
            self.getncattr(attr) if attr in self.ncattrs() else None
            for attr in ("scale_factor", "add_offset")
        )
        is_scaled = scale_factor is not None or add_offset is not None
        if is_scaled:
            if scale_factor is None:
                scale_factor = np.array(1, dtype=self.dtype)
            if add_offset is None:
                add_offset = np.array(0, dtype=self.dtype)
        return is_scaled, scale_factor, add_offset

    def _get_fillvalue(self):
        """
        Calculate any applicable fill-value from attributes and netCDF4 defaults.

        Notes
        -----
        * this must be checked dynamically, as the attributes could change.
        * for byte data, there is no netCDF default fill, so the result can be None.
        """
        fv = self._ncdata.avals.get("_FillValue", None)
        if fv is None:
            if self.dtype.itemsize != 1:
                # NOTE: single-byte types have NO default fill-value
                dtype_code = self.dtype.str[1:]
                fv = netCDF4.default_fillvals[dtype_code]
        return fv

    def _maskandscale_inner_to_apparent(self, array):
        """
        Convert raw data values to masked+scaled.

        This replicates the netCDF4 default auto-scaling procedure.
        """
        # N.B. fill-value matches the internal raw (unscaled) values and dtype
        fv = self._get_fillvalue()
        if fv is not None:
            array = da.ma.masked_equal(array, fv)

        is_scaled, scale_factor, add_offset = self._get_scaling()
        if is_scaled:
            # Scale the array, which may result in a type change.
            array = array * scale_factor + add_offset

        return array

    @staticmethod
    def _fill_masked(array, fv):
        """
        Fill any masked numpy data with a fill-value.

        This is applied lazily so that it can test whether actual data values are
        masked or not, since ordinary ndarrays do not provide a ".filled()" method.
        """
        if np.ma.isMaskedArray(array):
            array = array.filled(fv)
        return array

    def _maskandscale_apparent_to_inner(self, array):
        """
        Convert masked+scaled data to raw values.

        This replicates the netCDF4 default auto-scaling procedure.
        """
        is_scaled, scale_factor, add_offset = self._get_scaling()
        if is_scaled:
            # "Un-scale" the array, which may include a type change.
            array = (array - add_offset) / scale_factor
            if self.datatype.kind in "iu":
                array = np.round(array)
            array = array.astype(self.datatype)

        # N.B. fill-value matches the internal raw (unscaled) values and dtype
        fv = self._get_fillvalue()
        if fv is not None:
            # convert masked data to a filled array, but in a lazy fashion
            if not isinstance(array, da.Array):
                # To apply map_blocks, we must first have an actual Dask array.
                # NOTE: da.from_array() performs a trial access meta=array[:0, :0, ...]
                # to determine the 'meta'.  Normally we might avoid that by passing a
                # 'meta' argument, but in this case we don't know what 'array' might be,
                # so we have no (other) generic way to construct a suitable 'meta'.
                array = da.from_array(array, chunks="auto")
            array = da.map_blocks(
                self._fill_masked, array, fv, meta=array._meta
            )

        return array

    # Providing a "_data_array" property labels this as an 'emulated' netCDF4.Variable,
    # containing an actual (possibly lazy) array, which can be directly read/written.
    @property
    def _data_array(self):
        """
        Fetch the data from the underlying NcVariable.

        We also convert the 'raw' values into a masked-and-scaled form, possibly with
        a promoted dtype, to emulate the *default* read behaviour of a netCFD4 Variable.
        """
        array = self._ncdata.data
        array = self._maskandscale_inner_to_apparent(array)
        return array

    @_data_array.setter
    def _data_array(self, data):
        """
        Save data to the underlying NcVariable.

        We also convert the given values from a masked-and-scaled form into 'raw' data
        values, possibly of  a different dtype, to emulate the *default* write
        behaviour of a
        netCFD4 Variable.
        """
        data = self._maskandscale_apparent_to_inner(data)
        self._ncdata.data = data

    def group(self):  # noqa: D102
        return self._ncdata.group

    @property
    def dimensions(self) -> List[str]:  # noqa: D102
        return self._ncdata.dimensions

    #
    # "Normal" data access is via indexing.
    # N.B. we must still support this, e.g. for Iris coordinate loading.
    #
    def __getitem__(self, keys):  # noqa: D105
        # For now, only support whole-array fetches.
        # This restriction is probably unnecessary, but it is all that Iris requires.
        if keys != slice(None):
            raise IndexError(keys)

        array = self._data_array
        if hasattr(array, "compute"):
            # When directly accessed as a data variable, realise any lazy content.
            array = array.compute()
        return array

    # The __setitem__ is not required for normal saving.
    # The saver will assign ._data_array instead
    # TODO: might need to support this for future non-Iris usage ?
    # def __setitem__(self, keys, data):
    #     if keys != slice(None):
    #         raise IndexError(keys)
    #     if not hasattr(data, "dtype"):
    #         raise ValueError(f"nonarray assigned as data : {data}")
    #     if not data.shape == self.shape:
    #         msg = (
    #             f"assigned data has wrong shape : "
    #             f"{data.shape} instead of {self.shape}"
    #         )
    #         raise ValueError(msg)
    #     self._data_array = data

    @property
    def dtype(self):  # noqa: D102
        return self.datatype

    @property
    def dims(self):  # noqa: D102
        return self.dimensions

    @property
    def ndim(self):  # noqa: D102
        return len(self.dimensions)

    @property
    def shape(self):  # noqa: D102
        dims = self.group().dimensions
        return tuple(dims[n].size for n in self.dimensions)

    @property
    def size(self):  # noqa: D102
        return np.prod(self.shape)

    def chunking(self):  # noqa: D102
        """
        Return chunk sizes.

        Actual datasets return a list of sizes by dimension, or 'contiguous'.

        For now, simply returns ``None``.  Could be replaced when required.
        """
        return None


class Nc4DimensionLike:
    """
    An object which emulates a :class:`netCDF4.Dimension` object.

    The core, contained content object, ``self._ncdata``, is a :class:`ncdata.NcDimension`.
    This completely defines the parent object state.

    """

    def __init__(self, ncdim: NcDimension):  # noqa: D107
        self._ncdata = ncdim

    @property
    def name(self):  # noqa: D102
        return self._ncdata.name

    @property
    def size(self):  # noqa: D102
        return self._ncdata.size

    def isunlimited(self):  # noqa: D102
        return self._ncdata.unlimited

    def group(self):  # noqa: D102
        # Not properly supported ?
        return None
