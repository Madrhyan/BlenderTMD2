"""File reading helpers (decompression aware) for the ChatGPT TMD2 add-on."""

import os

from ..ctamlib.pzze import readPZZE
from ..ctamlib.tmd import TMD
from ..ctamlib.tmd2 import TMD2
from ..ctamlib.tmd0 import TMD0
from ..ctamlib.tmdv import TMDV
from ..ctamlib.lds import LDS
from ..ctamlib.utils.PyBinaryReader.binary_reader import BinaryReader, Endian


def _decompress_if_needed(path, magic_expected=None):
    with open(path, "rb") as handle:
        head = handle.read(4)
        handle.seek(0)
        if head == b"PZZE":
            pzze = readPZZE(path)
            data = pzze.decompress()
            if data is None:
                raise ValueError(f"Failed to decompress {os.path.basename(path)}")
            return data
        return handle.read()


def readTMD2(file):
    data = _decompress_if_needed(file)
    if not data.startswith(b"tmd0"):
        raise ValueError(f"Invalid TMD2 magic in {os.path.basename(file)}")
    br = BinaryReader(data, Endian.LITTLE)
    name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMD2, None, name)


def readTMD(file, texture_names=None):
    data = _decompress_if_needed(file)
    br = BinaryReader(data, Endian.LITTLE)
    name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMD, None, name, texture_names or {})


def readTMD0(file, texture_names=None):
    data = _decompress_if_needed(file)
    br = BinaryReader(data, Endian.LITTLE)
    name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMD0, None, name, texture_names or {})


def readTMDV(file):
    data = _decompress_if_needed(file)
    br = BinaryReader(data, Endian.LITTLE)
    name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMDV, None, name)


def readLDS(file):
    data = _decompress_if_needed(file)
    br = BinaryReader(data, Endian.LITTLE)
    name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(LDS, None, name)
