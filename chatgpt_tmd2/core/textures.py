"""Texture loading and the TMD2 texture-index resolver.

The original importer indexes the loaded image list with
``tm_texture.index`` (``TMD2Texture.index``), which is *usually* the position in
the texture table but is not guaranteed to be.  This module builds an explicit
resolver that maps

* TMD2 texture table position,
* ``TMD2Texture.index`` field,
* texture hash,

to a Blender image, so a material texture reference can always be resolved to the
correct image even when the file stores a different index order.

Images are named ``CTMD2_<file>_<n>`` so they can never clash with images created
by the original BlenderTMD2 add-on.
"""

import os
import tempfile

import bpy

from ..ctamlib.lds import LDS
from ..ctamlib.utils.PyBinaryReader.binary_reader import BinaryReader, Endian
from ..ctamlib.pzze import readPZZE


IMAGE_PREFIX = "CTMD2_"
_NONCOLOR_SUFFIX = "_NC"


class TextureResolver:
    """Resolve TMD2 material texture references to Blender images."""

    def __init__(self, images=None, texture_table=None):
        self.images = list(images or [])
        self.by_position = {}
        self.by_index = {}
        self.by_hash = {}
        for position, image in enumerate(self.images):
            if image is not None:
                self.by_position[position] = image
        for tex in texture_table or []:
            position = _table_position(texture_table, tex)
            image = self.by_position.get(position)
            if image is None:
                continue
            index = getattr(tex, "index", None)
            if isinstance(index, int):
                self.by_index.setdefault(index, image)
            tex_hash = getattr(tex, "hash", None)
            if tex_hash:
                self.by_hash.setdefault(tex_hash, image)

    # -- public ---------------------------------------------------------
    def resolve(self, mat_texture):
        """Resolve a ``TMD2MatTexture`` (or an index/hash pair)."""
        index = getattr(mat_texture, "textureIndex", None)
        if index is None:
            index = getattr(mat_texture, "index", None)
        tex = getattr(mat_texture, "texture", None)
        tex_index = getattr(tex, "index", None) if tex is not None else None
        tex_hash = getattr(tex, "hash", None) if tex is not None else None
        if tex_hash is None:
            tex_hash = getattr(mat_texture, "textureHash", None)

        if isinstance(index, int) and index >= 0:
            if index in self.by_index:
                return self.by_index[index]
            if index in self.by_position:
                return self.by_position[index]
        if isinstance(tex_index, int) and tex_index >= 0:
            if tex_index in self.by_index:
                return self.by_index[tex_index]
            if tex_index in self.by_position:
                return self.by_position[tex_index]
        if tex_hash and tex_hash in self.by_hash:
            return self.by_hash[tex_hash]
        return None

    def image_at(self, position):
        return self.by_position.get(position)

    def __bool__(self):
        return bool(self.images)

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        return iter(self.images)


def _table_position(table, item):
    try:
        return table.index(item)
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Image creation
# ---------------------------------------------------------------------------
def _unique_name(base):
    name = base
    counter = 1
    while bpy.data.images.get(name) is not None:
        name = f"{base}.{counter:03d}"
        counter += 1
    return name


def load_image_from_bytes(dds_bytes, name, source_path=None):
    """Create (or re-use) a packed Blender image from raw DDS data."""
    existing = bpy.data.images.get(name)
    if existing is not None and existing.get("ctmd2_source") == (source_path or name) \
            and existing.size[0] > 0:
        return existing

    # Resolve to a real file first: Blender loads image data lazily, so the file
    # has to stay on disk until the image is packed.
    path = source_path if (source_path and os.path.exists(source_path)) else None
    if path is None:
        path = _persistent_texture_file(dds_bytes, name)
    if path is None:
        return None

    try:
        image = bpy.data.images.load(path, check_existing=False)
    except Exception as exc:
        print(f"[ChatGPT TMD2] Failed to load texture {name}: {exc}")
        return None

    if image.size[0] == 0 or image.size[1] == 0:
        # Blender cannot decode every DDS variant (uncompressed formats in
        # particular).  Drop the broken datablock so the caller can substitute a
        # placeholder instead of leaving an empty image behind.
        print(f"[ChatGPT TMD2] {name}: image data could not be decoded "
              f"({image.size[0]}x{image.size[1]})")
        try:
            bpy.data.images.remove(image)
        except Exception:
            pass
        return None

    image.name = name if bpy.data.images.get(name) is None else _unique_name(name)
    try:
        image.colorspace_settings.name = 'sRGB'
    except Exception:
        pass
    try:
        image.alpha_mode = 'CHANNEL_PACKED'
    except Exception:
        pass
    if not image.packed_file:
        try:
            image.pack()
        except Exception as exc:
            print(f"[ChatGPT TMD2] Could not pack {name}: {exc}")
    image.use_fake_user = True
    image["ctmd2_source"] = source_path or name
    return image


def _persistent_texture_file(dds_bytes, name):
    """Write texture data to a persistent cache file and return its path."""
    directory = os.path.join(tempfile.gettempdir(), "chatgpt_tmd2_textures")
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.dds")
        # Always rewrite: the cache may still hold the previous import of a file
        # that happened to use the same image name.
        with open(path, "wb") as handle:
            handle.write(dds_bytes)
        return path
    except OSError as exc:
        print(f"[ChatGPT TMD2] Could not cache {name}: {exc}")
        return None


def non_color_copy(image):
    """Return a Non-Color copy of ``image`` for data textures (normal maps)."""
    if image is None:
        return None
    try:
        if image.colorspace_settings.name == 'Non-Color':
            return image
    except Exception:
        return image
    name = image.name + _NONCOLOR_SUFFIX
    cached = bpy.data.images.get(name)
    if cached is not None:
        return cached
    copy = image.copy()
    copy.name = _unique_name(name)
    try:
        copy.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    copy.use_fake_user = True
    return copy


def placeholder_image(name, width=64, height=64):
    image = bpy.data.images.get(name)
    if image:
        return image
    image = bpy.data.images.new(name, width=max(1, width), height=max(1, height))
    image.use_fake_user = True
    return image


# ---------------------------------------------------------------------------
# LDS / file-system loading
# ---------------------------------------------------------------------------
def read_lds_bytes(path):
    """Read a (possibly PZZE compressed) .lds file and return an ``LDS``."""
    with open(path, "rb") as handle:
        magic = handle.read(4)
        handle.seek(0)
        if magic == b"PZZE":
            pzze = readPZZE(path)
            data = pzze.decompress()
            if data is None:
                raise ValueError("Failed to decompress LDS data")
        else:
            data = handle.read()
    br = BinaryReader(data, Endian.LITTLE)
    name = os.path.splitext(os.path.basename(path))[0]
    return br.read_struct(LDS, None, name)


def load_lds_images(path, verbose=False):
    """Load every texture of an .lds file into Blender, in table order."""
    lds = read_lds_bytes(path)
    stem = IMAGE_PREFIX + os.path.splitext(os.path.basename(path))[0]
    images = []
    for i, texture_bytes in enumerate(lds.textures):
        name = f"{stem}_{i}"
        image = load_image_from_bytes(texture_bytes, name, source_path=None)
        if image is None:
            image = placeholder_image(name)
        images.append(image)
    if verbose:
        print(f"[ChatGPT TMD2] Loaded {len(images)} textures from {path}")
    return images


def load_dds_folder(directory, stem):
    """Load loose .dds files from ``directory`` for a model called ``stem``."""
    images = []
    if not os.path.isdir(directory):
        return images
    candidates = sorted(
        f for f in os.listdir(directory) if f.lower().endswith(".dds")
    )
    for i, file_name in enumerate(candidates):
        full = os.path.join(directory, file_name)
        name = f"{IMAGE_PREFIX}{stem}_{i}"
        image = load_image_from_bytes(b"", name, source_path=full)
        if image is not None:
            images.append(image)
    return images


def find_texture_file(tmd2_path, dds_dir=None):
    """Find the most likely texture container for a .tmd2 file."""
    directory = os.path.dirname(os.path.abspath(tmd2_path))
    stem = os.path.splitext(os.path.basename(tmd2_path))[0]
    candidates = []
    if dds_dir:
        candidates.append(os.path.join(dds_dir, stem + ".lds"))
    candidates.append(os.path.join(directory, stem + ".lds"))
    candidates.append(os.path.join(directory, stem.upper() + ".lds"))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None
