from .tamLib.pzze import *
from .tamLib.utils.PyBinaryReader.binary_reader import *
from .tamLib.tmd import *
from .tamLib.tmd2 import *
from .tamLib.lds import *
from .tamLib.cats import *
from .tamLib.tmo import *
from .tamLib.tmd0 import *
from .tamLib.tmdv import *
from .tamLib.vfxt import *
from .tamLib.tactpkg import *
import os


def _read_decompressed_bytes(file: str) -> bytes:
    with open(file, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic == b"PZZE":
            pzze = readPZZE(file)
            data = pzze.decompress()
            if data is None:
                raise ValueError(f"Failed to decompress PZZE archive: {file}")
            return data
        return f.read()


def readTMD2(file: str) -> TMD2:
    tmd_data = _read_decompressed_bytes(file)
    if not tmd_data.startswith(b"tmd0"):
        got = tmd_data[:4].decode("latin-1", errors="replace")
        raise ValueError(f"Invalid TMD2 magic. Expected 'tmd0', got: {got!r}")

    br = BinaryReader(tmd_data, Endian.LITTLE)
    file_name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMD2, None, file_name)


def readTMD(file: str, texture_names=None):
    if texture_names is None:
        texture_names = {}
    tmd_data = _read_decompressed_bytes(file)
    if not tmd_data.startswith(b"tmd0"):
        got = tmd_data[:4].decode("latin-1", errors="replace")
        raise ValueError(f"Invalid TMD magic. Expected 'tmd0', got: {got!r}")

    br = BinaryReader(tmd_data, Endian.LITTLE)
    file_name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMD, None, file_name, texture_names)


def readTMD0(file: str, texture_names=None):
    if texture_names is None:
        texture_names = {}
    tmd_data = _read_decompressed_bytes(file)
    if not (tmd_data.startswith(b"tmd0") or tmd_data.startswith(b"0dmt")):
        got = tmd_data[:4].decode("latin-1", errors="replace")
        raise ValueError(f"Invalid TMD0 magic. Expected 'tmd0' or '0dmt', got: {got!r}")

    br = BinaryReader(tmd_data, Endian.LITTLE)
    file_name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMD0, None, file_name, texture_names)


def readTMDV(file: str):
    tmd_data = _read_decompressed_bytes(file)
    if not (tmd_data.startswith(b"tmdv") or tmd_data.startswith(b"vdmt")):
        got = tmd_data[:4].decode("latin-1", errors="replace")
        raise ValueError(f"Invalid TMDV magic. Expected 'tmdv' or 'vdmt', got: {got!r}")

    br = BinaryReader(tmd_data, Endian.LITTLE)
    file_name = os.path.splitext(os.path.basename(file))[0]
    return br.read_struct(TMDV, None, file_name)


def readLDS(file: str) -> LDS:
    lds_data = _read_decompressed_bytes(file)
    file_name = os.path.splitext(os.path.basename(file))[0]
    br = BinaryReader(lds_data, Endian.LITTLE)
    return br.read_struct(LDS, None, file_name)


def readVFXT(file: str) -> VFXT:
    vfx_data = _read_decompressed_bytes(file)
    file_name = os.path.splitext(os.path.basename(file))[0]
    br = BinaryReader(vfx_data, Endian.LITTLE)
    return br.read_struct(VFXT, None, file_name)


def readCATS(file: str) -> CATS:
    cats_data = _read_decompressed_bytes(file)
    if not cats_data.startswith(b"CATS"):
        got = cats_data[:4].decode("latin-1", errors="replace")
        raise ValueError(f"Invalid CATS magic. Expected 'CATS', got: {got!r}")
    br = BinaryReader(cats_data, Endian.LITTLE)
    return br.read_struct(CATS)


def readActorPackage(file: str):
    ap_data = _read_decompressed_bytes(file)
    if not ap_data.startswith(b"actmng_pkg"):
        raise ValueError("Invalid ActorPackage magic.")
    br = BinaryReader(ap_data, Endian.LITTLE)
    return br.read_struct(TactPkg)


def readTMO(file: str) -> TMO:
    tmo_data = _read_decompressed_bytes(file)
    if not tmo_data.startswith(b"tmo1"):
        raise ValueError("Invalid TMO magic.")
    file_name = os.path.splitext(os.path.basename(file))[0]
    br = BinaryReader(tmo_data, Endian.LITTLE)
    tmo = br.read_struct(TMO)
    tmo.name = file_name
    return tmo


def writeTMD2(tmd2, output, compress=False):
    br = BinaryReader()
    br.write_struct(tmd2)
    filebytes = bytes(br.buffer())

    if compress:
        pzze = PZZEFile()
        pzze.fileFormat = "tmd2"
        pzze.decompressedData = filebytes

        br = BinaryReader()
        br.write_struct(pzze)
        filebytes = bytes(br.buffer())

    with open(output, "wb") as f:
        f.write(filebytes)


def writeTMD(tmd, output):
    br = BinaryReader()
    br.write_struct(tmd)
    filebytes = bytes(br.buffer())
    with open(output, "wb") as f:
        f.write(filebytes)


def writeLDS(lds, output, compress=False):
    br = BinaryReader()
    br.write_struct(lds)
    filebytes = bytes(br.buffer())
    if compress:
        pzze = PZZEFile()
        pzze.fileFormat = "lds"
        pzze.decompressedData = filebytes

        br = BinaryReader()
        br.write_struct(pzze)
        filebytes = bytes(br.buffer())

    with open(output, "wb") as f:
        f.write(filebytes)
