"""Thin wrappers around the binary writers of ``ctamlib``."""

from ..ctamlib.lds import LDS
from ..ctamlib.pzze import PZZEFile
from ..ctamlib.utils.PyBinaryReader.binary_reader import BinaryReader


def write_tmd2(tmd2, path, compress=False):
    br = BinaryReader()
    br.write_struct(tmd2)
    data = bytes(br.buffer())
    if compress:
        pzze = PZZEFile()
        pzze.fileFormat = 'tmd2'
        pzze.decompressedData = data
        br = BinaryReader()
        br.write_struct(pzze)
        data = bytes(br.buffer())
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def write_lds(textures, path, compress=False):
    lds = LDS()
    lds.textures = list(textures)
    br = BinaryReader()
    br.write_struct(lds)
    data = bytes(br.buffer())
    if compress:
        pzze = PZZEFile()
        pzze.fileFormat = 'lds'
        pzze.decompressedData = data
        br = BinaryReader()
        br.write_struct(pzze)
        data = bytes(br.buffer())
    with open(path, "wb") as handle:
        handle.write(data)
    return path
