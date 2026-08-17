"""
Tests for :func:`ncdata.netcdf.fix_varlen_string_chunks`.

"""

import dask.config
import pytest

from ncdata.netcdf4 import fix_varlen_string_chunks


class TestFixVarlenStringChunks:
    @staticmethod
    def run(chunks, shape):
        chunks_list = list(chunks)
        fix_varlen_string_chunks(chunks=chunks_list, shape=shape)
        return chunks_list

    def test_noauto(self):
        shape = (1, 3, 5, 4)
        chunks = (1, 2, 3, 4)
        result = self.run(chunks, shape)
        assert result == list(chunks)

    def test_fullshape_auto(self):
        shape = (1, 3, 5, 4)
        chunks = ["auto"] * 4
        result = self.run(chunks, shape)
        assert result == list(shape)

    def test_scalar(self):
        shape = ()
        chunks = []
        result = self.run(chunks, shape)
        assert result == list(shape)

    def test_auto_last(self):
        shape = (1, 3, 5, 14)
        chunks = (1, 2, 3, "auto")
        result = self.run(chunks, shape)
        assert result == [1, 2, 3, 14]

    def test_auto_first(self):
        shape = (21, 33, 45, 14)
        chunks = ("auto", 2, 3, 4)
        result = self.run(chunks, shape)
        assert result == [21, 2, 3, 4]

    def test_auto_penult(self):
        shape = (1, 3, 5, 14)
        chunks = (1, 2, "auto", 4)
        result = self.run(chunks, shape)
        assert result == [1, 2, 5, 4]

    @pytest.fixture()
    def varstr_len_100(self, mocker):
        # Allow 100 characters(bytes) in variable-length strings
        mocker.patch("ncdata.netcdf4.ASSUMED_TYPICAL_STRINGLENGTH", 100)
        yield

    def test_sizelimit_fullauto(self, varstr_len_100):
        shape = (10, 12)
        chunks = ("auto", "auto")

        # set chunksize to match 40 strings
        with dask.config.set({"array.chunk-size": "4000b"}):
            result = self.run(chunks, shape)
        assert result == [3, 12]

    def test_sizelimit_multispread(self, varstr_len_100):
        shape = (10, 12, 3, 3)
        chunks = ("auto", "auto", "auto", "auto")
        # set chunksize to match 40 strings
        with dask.config.set({"array.chunk-size": "4000b"}):
            result = self.run(chunks, shape)
        assert result == [1, 4, 3, 3]

    def test_sizelimit_mixed(self, varstr_len_100):
        shape = (5, 10, 4, 6)
        chunks = ("auto", "auto", 2, "auto")

        # set chunksize to match 40 strings
        with dask.config.set({"array.chunk-size": "4000b"}):
            result = self.run(chunks, shape)
        assert result == [1, 3, 2, 6]
