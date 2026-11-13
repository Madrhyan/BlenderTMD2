import json
import math
import os
import struct
import tempfile
from collections import defaultdict
from math import radians
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy
import numpy as np
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Euler, Matrix, Quaternion, Vector

from .materials.analyzer import analyze_tmd_material
from .materials.shaders import build_material_from_analysis
from .reader import (
    readActorPackage,
    readCATS,
    readLDS,
    readTMD,
    readTMD0,
    readTMD2,
    readTMDV,
    readTMO,
    readVFXT,
)
from .tamLib.cat0 import GXT, cat
from .tamLib.cat1 import GNF, cat1
from .tamLib.lds import LDS
from .tamLib.tactpkg import TactPkg
from .tamLib.tmd import TMD, TMDBone, TMDMaterial, TMDModel, TMDSubmesh
from .tamLib.tmd0 import TMD0
from .tamLib.tmd2 import (
    TMD2,
    TMD2Bone,
    TMD2Material,
    TMD2MatTexture,
    TMD2Model,
    TMD2Submesh,
    TMD2Texture,
    tamCRC32,
)
from .tamLib.tmdv import TMDV, TMDVSubmesh
from .tamLib.tmo import TMO
from .tamLib.utils.PyBinaryReader.binary_reader import BinaryReader, Endian

with open(os.path.join(os.path.dirname(__file__), "hashes.json"), "r", encoding="utf-8") as _hf:
    HASHES: Dict[str, str] = json.load(_hf)


# =====================================================================
# TEXTURE CONTAINER HELPERS (LDS, VFXT, LOOSE DDS)
# =====================================================================

def _load_dds_bytes_into_blender(tex_name: str, raw_dds: bytes) -> Optional[bpy.types.Image]:
    if not raw_dds:
        return None
    existing = bpy.data.images.get(tex_name)
    if existing is not None:
        try:
            existing.pack(data=raw_dds, data_len=len(raw_dds))
            existing.source = "FILE"
            existing.use_fake_user = True
            return existing
        except Exception:
            pass

    fd, tmp_path = tempfile.mkstemp(suffix=".dds", prefix=f"{tex_name}_")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(raw_dds)
        img = bpy.data.images.load(tmp_path)
        img.name = tex_name
        img.pack()
        img.use_fake_user = True
        return img
    except Exception as exc:
        print(f"[ChatGPT TMD2] Failed to load DDS '{tex_name}': {exc}")
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def importLDS(file_path: str, return_tex: bool = False):
    lds = readLDS(file_path)
    if lds is None:
        return None
    images_list: List[Optional[bpy.types.Image]] = []
    raw_list: List[bytes] = []
    for i, texture in enumerate(lds.textures):
        tex_name = f"{lds.name}_{i}"
        img = _load_dds_bytes_into_blender(tex_name, texture)
        images_list.append(img)
        raw_list.append(texture if isinstance(texture, (bytes, bytearray)) else b"")
    lds.images = images_list
    lds.raw_dds = raw_list
    if return_tex:
        return lds
    return None


def importVFXT(file_path: str) -> Tuple[List[bpy.types.Image], Dict[int, bpy.types.Image], Dict[str, bpy.types.Image]]:
    vfxt = readVFXT(file_path)
    imgs: List[bpy.types.Image] = []
    by_hash: Dict[int, bpy.types.Image] = {}
    by_name: Dict[str, bpy.types.Image] = {}
    if vfxt is None:
        return imgs, by_hash, by_name

    base = os.path.splitext(os.path.basename(file_path))[0]
    for i, vt in enumerate(vfxt.textures):
        raw_h = getattr(vt, "hash", b"")
        label = f"{base}_vfx_{i}"
        crc_val = None
        if isinstance(raw_h, (bytes, bytearray)):
            if len(raw_h) == 4:
                crc_val = struct.unpack("<i", raw_h)[0]
                label = HASHES.get(str(crc_val), f"{base}_{crc_val}")
            elif len(raw_h) > 0:
                decoded = raw_h.rstrip(b"\x00").decode("utf-8", errors="ignore").strip()
                if decoded:
                    label = decoded
                    crc_val = tamCRC32(decoded)
        img = _load_dds_bytes_into_blender(label, vt.textureData)
        if img is not None:
            imgs.append(img)
            by_name[label] = img
            by_name[label.lower()] = img
            if crc_val is not None:
                by_hash[int(crc_val)] = img
    return imgs, by_hash, by_name


def scan_directory_textures(directory: str) -> Tuple[Dict[int, bpy.types.Image], Dict[str, bpy.types.Image], Dict[str, str]]:
    """Scans directory (and optional sibling 'tex'/'textures' subfolders) for .dds and .vfxt textures."""
    dds_by_hash: Dict[int, bpy.types.Image] = {}
    dds_by_name: Dict[str, bpy.types.Image] = {}
    dds_paths: Dict[str, str] = {}

    if not directory or not os.path.isdir(directory):
        return dds_by_hash, dds_by_name, dds_paths

    search_dirs = [directory]
    for sub in ("tex", "textures", "Texture", "Textures"):
        cand = os.path.join(directory, sub)
        if os.path.isdir(cand):
            search_dirs.append(cand)

    for sdir in search_dirs:
        try:
            entries = os.listdir(sdir)
        except OSError:
            continue
        for fname in entries:
            full = os.path.join(sdir, fname)
            low = fname.lower()
            if low.endswith(".dds") and os.path.isfile(full):
                stem = os.path.splitext(fname)[0]
                dds_paths[stem] = full
                dds_paths[stem.lower()] = full
            elif low.endswith(".vfxt") and os.path.isfile(full):
                _, vh, vn = importVFXT(full)
                dds_by_hash.update(vh)
                dds_by_name.update(vn)

    return dds_by_hash, dds_by_name, dds_paths


def _get_or_load_loose_dds(stem: str, dds_paths: Dict[str, str]) -> Optional[bpy.types.Image]:
    if not stem:
        return None
    img = bpy.data.images.get(stem)
    if img is not None:
        return img
    path = dds_paths.get(stem) or dds_paths.get(stem.lower())
    if path and os.path.isfile(path):
        try:
            img = bpy.data.images.load(path)
            img.name = stem
            img.pack()
            img.use_fake_user = True
            return img
        except Exception:
            return None
    return None


def _find_matching_lds(directory: str, base_name: str, explicit_lds: Dict[str, str]) -> str:
    if base_name in explicit_lds:
        return explicit_lds[base_name]
    if not directory:
        return ""
    direct = os.path.join(directory, base_name + ".lds")
    if os.path.isfile(direct):
        return direct
    # Case-insensitive fallback + common strip of '_mdl' -> '_tex' conventions
    candidates = {base_name.lower() + ".lds"}
    if base_name.lower().endswith("_mdl"):
        candidates.add(base_name[:-4].lower() + ".lds")
        candidates.add(base_name[:-4].lower() + "_tex.lds")
    try:
        for fn in os.listdir(directory):
            if fn.lower() in candidates:
                return os.path.join(directory, fn)
    except OSError:
        pass
    return ""


# =====================================================================
# MATERIAL PROPERTY STORAGE HELPER
# =====================================================================

def _populate_blender_mat_props(
    blender_mat: bpy.types.Material,
    tm_mat,
    analysis,
    fallback_options: dict,
) -> None:
    bmat_props = blender_mat.chatgpt_tmd2_material
    bmat_props.material_hash = str(getattr(tm_mat, "hash", 0))
    bmat_props.shader_id = getattr(tm_mat, "shaderID", "") or "Default"
    bmat_props.unk = int(getattr(tm_mat, "unk", 0))
    bmat_props.detected_family = analysis.shader_family
    bmat_props.detected_blend = analysis.blend_mode

    bmat_props.reconstruction_profile = fallback_options.get("fallback_profile", "AUTO")
    bmat_props.transparency_mode = fallback_options.get("transparency_mode", "AUTO")
    bmat_props.vertex_color_mode = fallback_options.get("use_vertex_colors", "AUTO")
    bmat_props.grayscale_mode = fallback_options.get("grayscale_mode", "AUTO")
    bmat_props.backface_culling_mode = fallback_options.get("backface_culling", "AUTO")
    bmat_props.emission_multiplier = float(fallback_options.get("emission_multiplier", 1.0))

    bmat_props.param_values.clear()
    raw_params = list(getattr(tm_mat, "shaderParams", []) or [])
    raw_metas = list(getattr(tm_mat, "shaderParamMetas", []) or [])
    for idx_p, val in enumerate(raw_params):
        p = bmat_props.param_values.add()
        p.value = float(val)
        p.meta = int(raw_metas[idx_p]) if idx_p < len(raw_metas) else 0

    bmat_props.textures.clear()
    tm_textures = list(getattr(tm_mat, "textures", []) or [])
    for idx_t, tmat_tex in enumerate(tm_textures):
        t = bmat_props.textures.add()
        tex_obj = getattr(tmat_tex, "texture", None)
        thash = int(getattr(tmat_tex, "textureHash", getattr(tex_obj, "hash", 0)))
        t.texture_hash = str(thash)
        t.resolved_name = HASHES.get(str(thash), getattr(tex_obj, "name", "") or "")
        t.value1 = int(getattr(tmat_tex, "unk1", 0))
        t.value2 = int(getattr(tmat_tex, "unk2", 0))
        t.value3 = int(getattr(tmat_tex, "unk3", 0)) & 0xFF
        t.slot = int(getattr(tmat_tex, "slot", idx_t)) & 0xFF
        is_ext = bool(getattr(tmat_tex, "isExternal", False)) or int(getattr(tmat_tex, "textureIndex", 0)) < 0
        t.is_external = is_ext
        if idx_t < len(analysis.bindings) and analysis.bindings[idx_t].image is not None:
            t.image = analysis.bindings[idx_t].image


# =====================================================================
# MAIN UNIFIED IMPORTER OPERATOR (WITH CONFIGURABLE FALLBACK UI)
# =====================================================================

class CGTMD2_IMPORTER_OT_IMPORT(Operator, ImportHelper):
    bl_label = "Import ChatGPT TMD2"
    bl_idname = "import_scene.chatgpt_tmd2"
    bl_options = {"REGISTER", "UNDO"}

    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    filter_glob: StringProperty(
        default="*.tmd2;*.lds;*.tmd;*.tmd0;*.tmdv;*.vfxt;*.tmo;*.cat;*.tactpkg",
        options={"HIDDEN"},
    )
    filename_ext = ".tmd2"
    filepath: StringProperty(subtype="FILE_PATH")

    auto_find_textures: BoolProperty(
        name="Auto Find Textures (.lds / .vfxt / .dds)",
        description="Automatically locate companion .lds, .vfxt, and loose .dds textures",
        default=True,
    )
    preserve_double_sided_faces: BoolProperty(
        name="Preserve Double-Sided Mesh Faces",
        description="Keep opposite-winding backfaces in VFX/cloth/spirit meshes without dropping geometry or desyncing UVs",
        default=True,
    )
    fallback_profile: EnumProperty(
        name="Shader Reconstruction",
        description="Fallback shader mode for unknown or game-specific shader IDs",
        items=[
            ("AUTO", "Auto (Data-Driven)", "Infer colors, glow, alpha, ramp, and spirit/VFX shaders from actual file data"),
            ("TOON_CHAR", "Anime / Cel Character", "TamEngine cel/toon shading"),
            ("VFX_ADDITIVE", "Glowing VFX / Aura", "Additive glowing shot/tiger/energy aura shader"),
            ("GHOST_SPIRIT", "Ghost / Spirit (See-Through)", "Semi-transparent spirit shader with Fresnel rim glow"),
            ("METALLIC_GLOW", "Metallic / Gold Sheen", "High-metallic reflective gold/armor shader"),
            ("PBR_TINT", "Standard Lit Tint", "Principled BSDF with extracted parameter colors"),
        ],
        default="AUTO",
    )
    transparency_mode: EnumProperty(
        name="Transparency",
        items=[
            ("AUTO", "Auto", "Detect Opaque, Cutout, Alpha Blend, Additive, or Ghost transparency from data"),
            ("OPAQUE", "Opaque", "Force opaque rendering"),
            ("ALPHA_CLIP", "Alpha Clip", "Force cutout transparency"),
            ("ALPHA_BLEND", "Alpha Blend", "Force smooth see-through blending"),
            ("ADDITIVE", "Additive Glow", "Force additive energy blending"),
            ("GHOST", "Ghost Translucent", "Force Fresnel ghost translucency"),
        ],
        default="AUTO",
    )
    use_vertex_colors: EnumProperty(
        name="Vertex Colors",
        items=[
            ("AUTO", "Auto", "Use mesh vertex RGB & Alpha when non-trivial vertex colors exist"),
            ("ON", "Force On", "Always multiply vertex Color & Alpha"),
            ("OFF", "Off", "Do not connect vertex colors in shader"),
        ],
        default="AUTO",
    )
    grayscale_mode: EnumProperty(
        name="Grayscale Tinting",
        items=[
            ("AUTO", "Auto", "Colorize grayscale textures automatically using extracted shaderParams or Ramp LUT"),
            ("COLORIZE", "2-Tone Colorize", "Force two-tone colorization"),
            ("MULTIPLY", "Simple Multiply", "Multiply grayscale texture by Base Tint"),
            ("OFF", "Disable Tint", "Leave textures untinted"),
        ],
        default="AUTO",
    )
    backface_culling: EnumProperty(
        name="Backface Culling",
        items=[
            ("AUTO", "Auto", "Show both sides on VFX/Ghost/double-sided meshes; cull on solid meshes"),
            ("OFF", "Double-Sided", "Show backfaces on all materials"),
            ("ON", "Single-Sided", "Enable backface culling on all materials"),
        ],
        default="AUTO",
    )
    emission_multiplier: FloatProperty(
        name="Glow Multiplier",
        description="Scale factor for reconstructed emission/bloom strength",
        default=1.0,
        min=0.0,
        max=25.0,
    )

    def draw(self, context):
        layout = self.layout
        box_io = layout.box()
        box_io.label(text="Import & Mesh Options", icon="MESH_DATA")
        box_io.prop(self, "auto_find_textures")
        box_io.prop(self, "preserve_double_sided_faces")

        box_sh = layout.box()
        box_sh.label(text="Shader & Fallback Options", icon="SHADING_RENDERED")
        box_sh.prop(self, "fallback_profile")
        box_sh.prop(self, "transparency_mode")
        box_sh.prop(self, "use_vertex_colors")
        box_sh.prop(self, "grayscale_mode")
        box_sh.prop(self, "backface_culling")
        box_sh.prop(self, "emission_multiplier")

    def _fallback_dict(self) -> dict:
        return {
            "fallback_profile": self.fallback_profile,
            "transparency_mode": self.transparency_mode,
            "use_vertex_colors": self.use_vertex_colors,
            "grayscale_mode": self.grayscale_mode,
            "backface_culling": self.backface_culling,
            "emission_multiplier": self.emission_multiplier,
        }

    def execute(self, context):
        selected_files = list(self.files)
        base_dir = self.directory
        if not selected_files and self.filepath:
            base_dir = os.path.dirname(self.filepath)
            fname = os.path.basename(self.filepath)

            class _SingleFile:
                name = fname

            selected_files = [_SingleFile()]

        tmd2_files: Dict[str, str] = {}
        tmd_files: Dict[str, str] = {}
        tmd0_files: List[str] = []
        tmdv_files: List[str] = []
        lds_files: Dict[str, str] = {}
        vfxt_files: List[str] = []
        tmo_files: List[str] = []
        cat_files: List[str] = []
        tactpkg_files: List[str] = []

        for f in selected_files:
            full_path = os.path.join(base_dir, f.name)
            bname, ext = os.path.splitext(f.name)
            bkey = bname.lower()
            ext_l = ext.lower()
            if ext_l == ".tmd2":
                tmd2_files[bkey] = full_path
            elif ext_l == ".lds":
                lds_files[bkey] = full_path
            elif ext_l == ".tmd":
                tmd_files[bkey] = full_path
            elif ext_l == ".tmd0":
                tmd0_files.append(full_path)
            elif ext_l == ".tmdv":
                tmdv_files.append(full_path)
            elif ext_l == ".vfxt":
                vfxt_files.append(full_path)
            elif ext_l == ".tmo":
                tmo_files.append(full_path)
            elif ext_l == ".cat":
                cat_files.append(full_path)
            elif ext_l == ".tactpkg":
                tactpkg_files.append(full_path)

        dds_by_hash, dds_by_name, dds_paths = (
            scan_directory_textures(base_dir) if self.auto_find_textures else ({}, {}, {})
        )

        # Load explicitly selected standalone VFXT or LDS files if no TMD2 consumed them
        for vf_path in vfxt_files:
            _, vh, vn = importVFXT(vf_path)
            dds_by_hash.update(vh)
            dds_by_name.update(vn)

        if lds_files and not tmd2_files:
            for lpath in lds_files.values():
                importLDS(lpath, return_tex=False)

        kw = self.as_keywords(ignore=("filter_glob", "files", "directory", "filepath"))
        fb_opts = self._fallback_dict()

        # Import TMD2 files
        for bkey, tmd2_path in tmd2_files.items():
            tex_path = _find_matching_lds(base_dir, bkey, lds_files) if self.auto_find_textures else lds_files.get(bkey, "")
            tmd2 = readTMD2(tmd2_path)
            if tmd2 is None:
                self.report({"ERROR"}, f"Failed to parse TMD2 file: {tmd2_path}")
                continue
            imp = importTMD2(
                self,
                tmd2_path,
                kw,
                tmd2,
                dds_paths=dds_paths,
                dds_by_hash=dds_by_hash,
                dds_by_name=dds_by_name,
                fallback_options=fb_opts,
            )
            imp.texture_path = tex_path
            imp.read(context)

        # Import legacy TMD files
        if tmd_files:
            hashed_names = {tamCRC32(k): k for k in dds_paths.keys()}
            for _, tmd_path in tmd_files.items():
                tmd = readTMD(tmd_path, hashed_names)
                if tmd is None:
                    self.report({"ERROR"}, f"Failed to parse TMD file: {tmd_path}")
                    continue
                imp = importTMD(
                    self,
                    tmd_path,
                    kw,
                    tmd,
                    dds_paths=dds_paths,
                    dds_by_hash=dds_by_hash,
                    dds_by_name=dds_by_name,
                    fallback_options=fb_opts,
                )
                imp.read(context)

        # Import TMD0 / TMDV / TMO / CAT / TACTPKG if selected
        if tmd0_files:
            hashed_names = {tamCRC32(k): k for k in dds_paths.keys()}
            for p in tmd0_files:
                t0 = readTMD0(p, hashed_names)
                if t0 is not None:
                    importTMD0(self, kw, t0, [], fallback_options=fb_opts).read(context)

        for p in tmdv_files:
            tv = readTMDV(p)
            if tv is not None:
                importTMDV(self, kw, tv, [], fallback_options=fb_opts).read(context)

        for p in tmo_files:
            tm = readTMO(p)
            if tm is not None:
                importTMO(self, kw, tm).read(context)

        for p in cat_files:
            _import_cats_archive(self, context, p, kw, fb_opts)

        for p in tactpkg_files:
            importActorPackage(p, self)

        return {"FINISHED"}


# =====================================================================
# CAT IMPORTER OPERATOR (WITH BRS / CATS SUPPORT FIXED!)
# =====================================================================

def _import_cats_archive(operator, context, cat_filepath: str, kw: dict, fb_opts: dict):
    link_path = os.path.splitext(cat_filepath)[0] + "_link.txt"
    if os.path.isfile(link_path):
        with open(link_path, "r", encoding="utf-8", errors="ignore") as lf:
            for line in lf:
                linked = line.strip()
                if linked:
                    full_l = os.path.join(os.path.dirname(cat_filepath), linked)
                    if full_l.lower().endswith(".lds") and os.path.isfile(full_l):
                        importLDS(full_l, return_tex=False)

    cat_obj = readCATS(cat_filepath)
    if cat_obj is None:
        return
    for c in cat_obj.subCATS:
        if c.name == "mdl.cat":
            for i in range(c.catCount):
                mdl = c.subData[i]
                name = c.subNames[i]
                br = BinaryReader(mdl, Endian.LITTLE)
                tmd2 = br.read_struct(TMD2)
                tmd2.name = name
                imp = importTMD2(operator, cat_filepath, kw, tmd2, fallback_options=fb_opts)
                imp.read(context)


class CGTMD2_CAT_OT_IMPORT(Operator, ImportHelper):
    bl_label = "Import ChatGPT CAT"
    bl_idname = "import_scene.chatgpt_cat"
    bl_options = {"REGISTER", "UNDO"}

    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    filter_glob: StringProperty(default="*.cat", options={"HIDDEN"})
    filename_ext = ".cat"
    filepath: StringProperty(subtype="FILE_PATH")

    target_game: EnumProperty(
        name="CAT Format",
        items=[
            ("AUTO", "Auto Detect", "Detect CATS (Modern/Bleach/Tsubasa) vs CAT0/CAT1 automatically"),
            ("BRS", "CATS Container (Bleach / Modern TamEngine)", ""),
            ("NPU", "Neptunia U: Action Unleashed (CAT0)", ""),
            ("NPP", "Hyperdimension Neptunia PP (CAT0)", ""),
            ("MTB", "MegaTagmension Blanc (CAT0)", ""),
            ("TLR", "To LOVE-Ru Darkness (CAT0)", ""),
            ("NHS", "Natsuiro High School (CAT0)", ""),
            ("SKVS", "Senran Kagura: Shinovi Versus (CAT1)", ""),
        ],
        default="AUTO",
    )

    def draw(self, context):
        self.layout.prop(self, "target_game")

    def execute(self, context):
        kw = self.as_keywords(ignore=("filter_glob", "files", "directory", "filepath"))
        if self.target_game in ("NPU", "NPP", "MTB", "TLR", "NHS"):
            importCat0(self, context)
        elif self.target_game == "SKVS":
            importCat1(self, context)
        elif self.target_game == "BRS":
            for f in self.files:
                _import_cats_archive(self, context, os.path.join(self.directory, f.name), kw, {})
        else:
            for f in self.files:
                full = os.path.join(self.directory, f.name)
                with open(full, "rb") as fp:
                    head = fp.read(8)
                if head.startswith(b"CATS") or (head.startswith(b"PZZE") and head[4:8] == b"CATS"):
                    _import_cats_archive(self, context, full, kw, {})
                else:
                    try:
                        importCat0(self, context)
                    except Exception:
                        importCat1(self, context)
        return {"FINISHED"}


def importCat0(operator, context):
    for file in operator.files:
        filepath = os.path.join(operator.directory, file.name)
        cat_name = os.path.splitext(file.name)[0]
        with open(filepath, "rb") as f:
            br = BinaryReader(f.read())
            cat_file = br.read_struct(cat)

        for name, tmd0 in cat_file.models.items():
            tmd0.name = f"{cat_name}_{name}"
            tex_container = cat_file.textures.get(name, [])
            dds_list = []
            if isinstance(tex_container, GXT):
                dds_list = [tx.to_dds() for tx in tex_container.textures]
            elif isinstance(tex_container, LDS):
                dds_list = tex_container.textures

            images_list = [
                _load_dds_bytes_into_blender(f"{tmd0.name}_{i}", dds)
                for i, dds in enumerate(dds_list)
            ]
            importTMD0(operator, {}, tmd0, images_list).read(context)

        for name, tmo in cat_file.animations.items():
            tmo.name = f"{cat_name}_{name}"
            importTMO(operator, {}, tmo).read(context)


def importCat1(operator, context):
    for file in operator.files:
        filepath = os.path.join(operator.directory, file.name)
        with open(filepath, "rb") as f:
            br = BinaryReader(f.read())
            cat_file: cat1 = br.read_struct(cat1)

        loaded_imgs: List[bpy.types.Image] = []
        for content in cat_file.content:
            if content.type == "gxt":
                raw_map = {}
                if isinstance(content.obj, (GXT, GNF)):
                    for tx in content.obj.textures:
                        raw_map[tx.name] = tx.to_dds()
                elif isinstance(content.obj, LDS):
                    for idx_l, tx in enumerate(content.obj.textures):
                        raw_map[f"lds_{idx_l}"] = tx
                for tname, dds in raw_map.items():
                    if isinstance(dds, (bytes, bytearray)):
                        img = _load_dds_bytes_into_blender(tname, dds)
                        if img is not None:
                            loaded_imgs.append(img)
            elif content.type == "tmdv":
                importTMDV(operator, {}, content.obj, loaded_imgs).read(context)
            elif content.type == "tmd0":
                importTMD0(operator, {}, content.obj, loaded_imgs).read(context)


# =====================================================================
# TMD2 CORE IMPORTER (FULL SHADER RECONSTRUCTION + MESH PRESERVATION)
# =====================================================================

class importTMD2:
    def __init__(
        self,
        operator: Operator,
        filepath: str,
        import_settings: dict,
        tmdfile: TMD2,
        dds_paths: Optional[Dict[str, str]] = None,
        dds_by_hash: Optional[Dict[int, bpy.types.Image]] = None,
        dds_by_name: Optional[Dict[str, bpy.types.Image]] = None,
        fallback_options: Optional[dict] = None,
    ):
        self.operator = operator
        self.filepath = filepath
        self.texture_path = ""
        self.preserve_double_sided_faces = True
        for key, value in import_settings.items():
            setattr(self, key, value)

        self.tmd2: TMD2 = tmdfile
        self.dds_paths = dds_paths or {}
        self.dds_by_hash = dict(dds_by_hash or {})
        self.dds_by_name = dict(dds_by_name or {})
        self.fallback_options = fallback_options or {}

    def _resolve_images(self) -> Tuple[List[Optional[bpy.types.Image]], List[bytes]]:
        images_list: List[Optional[bpy.types.Image]] = []
        raw_dds_list: List[bytes] = []

        if self.texture_path and os.path.isfile(self.texture_path):
            lds = importLDS(self.texture_path, return_tex=True)
            if lds is not None:
                images_list = list(lds.images)
                raw_dds_list = list(getattr(lds, "raw_dds", []))

        # Populate hash/name maps for any resolved textures in TMD2 header or loose directory
        for i, tex in enumerate(self.tmd2.textures):
            resolved_name = HASHES.get(str(tex.hash), tex.name or "")
            if resolved_name:
                tex.name = resolved_name
            img = None
            if i < len(images_list) and images_list[i] is not None:
                img = images_list[i]
            else:
                img = (
                    self.dds_by_hash.get(int(tex.hash))
                    or _get_or_load_loose_dds(resolved_name, self.dds_paths)
                    or bpy.data.images.get(f"{self.tmd2.name}_{i}")
                    or bpy.data.images.get(f"{self.tmd2.name}_{i}.dds")
                )
                while len(images_list) <= i:
                    images_list.append(None)
                images_list[i] = img

            if img is not None:
                self.dds_by_hash[int(tex.hash)] = img
                if resolved_name:
                    self.dds_by_name[resolved_name] = img

        # Also resolve any external (textureIndex == -1) material textures from loose DDS / hashes.json
        for tm_mat in self.tmd2.materials:
            for mt in tm_mat.textures:
                th = int(mt.textureHash)
                if th not in self.dds_by_hash:
                    rname = HASHES.get(str(th), "")
                    if rname:
                        ext_img = _get_or_load_loose_dds(rname, self.dds_paths) or bpy.data.images.get(rname)
                        if ext_img is not None:
                            self.dds_by_hash[th] = ext_img
                            self.dds_by_name[rname] = ext_img

        return images_list, raw_dds_list

    def read(self, context):
        collection = bpy.data.collections.new(f"{self.tmd2.name}")
        context.collection.children.link(collection)

        # Store TMD2 container metadata on uniquely-named ChatGPT TMD2 Empty
        empty = bpy.data.objects.new(f"#CGTMD2 PROPERTIES [{self.tmd2.name}]", None)
        empty.empty_display_size = 0.01
        empty.chatgpt_tmd2_props.format = "TMD2"
        ver_str = str(self.tmd2.version)
        for enum_val in ("521", "520", "519", "518", "517", "516", "515", "514", "513"):
            if ver_str == enum_val:
                empty.chatgpt_tmd2_props.version = enum_val
                break
        empty.chatgpt_tmd2_props.flag1 = int(self.tmd2.flag1)
        empty.chatgpt_tmd2_props.flag2 = int(self.tmd2.flag2)
        empty.chatgpt_tmd2_props.flag3 = int(self.tmd2.animFlag)
        empty.chatgpt_tmd2_props.after_image = int(self.tmd2.afterImageValue)
        collection.objects.link(empty)

        images_list, raw_dds_list = self._resolve_images()

        combined_dds = dict(self.dds_paths)
        combined_dds.update(self.dds_by_name)
        for h_k, img_v in self.dds_by_hash.items():
            combined_dds[str(h_k)] = img_v

        # Reconstruct every material from real game binary parameters, textures & vertex attributes
        materials_dict = {}
        for i, tm_mat in enumerate(self.tmd2.materials):
            tm_mat: TMD2Material
            mat_name = HASHES.get(str(tm_mat.hash), f"{self.tmd2.name}_mat_{i}")

            analysis = analyze_tmd_material(
                tm_mat,
                self.tmd2,
                images_list,
                combined_dds,
                HASHES,
                mat_name=mat_name,
                fallback_options=self.fallback_options,
            )
            blender_mat = build_material_from_analysis(mat_name, analysis)
            _populate_blender_mat_props(blender_mat, tm_mat, analysis, self.fallback_options)
            materials_dict[tm_mat] = blender_mat

        YUP_TO_ZUP = Matrix.Rotation(radians(90), 4, "X")
        bones_to_pose = {}
        armature = None
        armature_obj = None

        if self.tmd2.modelFlags & 0x2000:
            armature = bpy.data.armatures.new(self.tmd2.name)
            armature_obj = bpy.data.objects.new(self.tmd2.name, armature)
            collection.objects.link(armature_obj)
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode="EDIT")
            armature_obj.data.display_type = "STICK"

            conv_mat = axis_conversion(from_forward="-Z", from_up="Y", to_forward="Y", to_up="Z").to_4x4()
            for tmbone in self.tmd2.bones:
                tmbone: TMD2Bone
                if str(tmbone.hash) in HASHES:
                    tmbone.name = HASHES[str(tmbone.hash)]
                bbone = armature_obj.data.edit_bones.new(tmbone.name or f"bone_{tmbone.hash}")

                matrix = Matrix(tmbone.matrix).transposed().inverted()
                matrix_up = conv_mat @ matrix
                bbone.head = matrix_up @ Vector((0, 0, 0))
                bbone.tail = matrix_up @ Vector((0, 0.05, 0))
                bbone.matrix = matrix_up

                if 0 <= tmbone.parentIndex < len(armature_obj.data.edit_bones):
                    bbone.parent = armature_obj.data.edit_bones[tmbone.parentIndex]

                if tmbone.extra > -1:
                    bones_to_pose[bbone.name] = tmbone.offset

                bbone["hash"] = str(tmbone.hash)
                bbone["extra"] = tmbone.extra
                bbone["offset"] = tmbone.offset
                bbone["posed_loc"] = tmbone.posedLocation
                bbone["unk1"] = tmbone.unk1
                bbone["matrix"] = matrix
                bbone["matrix_up"] = matrix_up

            bpy.ops.object.mode_set(mode="OBJECT")

            if armature and bones_to_pose:
                for bname, rotation in bones_to_pose.items():
                    p = armature_obj.pose.bones.get(bname)
                    if not p or not p.parent:
                        continue
                    rot_q = Euler(rotation, "XYZ").to_quaternion()
                    p.rotation_mode = "QUATERNION"
                    p.rotation_quaternion = rot_q.inverted()

        for model_idx, tmd_model in enumerate(self.tmd2.models):
            tmd_model: TMD2Model
            m_name = tmd_model.name or f"{self.tmd2.name}_mesh_{model_idx}"
            mesh = bpy.data.meshes.new(m_name)
            mesh_obj = bpy.data.objects.new(m_name, mesh)
            collection.objects.link(mesh_obj)

            mesh_props = mesh_obj.chatgpt_tmd2_mesh
            if tmd_model.hashFlag:
                mesh_props.has_hash = True
                mesh_props.name_hash = str(tmd_model.hash)
            if tmd_model.nameFlag & 4:
                mesh_props.has_name = True
                mesh_props.name = tmd_model.name
            if tmd_model.nameFlag == 3:
                mesh_props.has_extra = True
                mesh_props.unk1 = int(getattr(tmd_model, "unk1", 0))
                mesh_props.unk2 = int(getattr(tmd_model, "unk2", 0))
                mesh_props.unk3 = int(getattr(tmd_model, "unk3", 0))
            mesh_props.lhs_flag = bool(getattr(tmd_model, "lhsFlag", False))

            model_mats = {}
            for i, mesh_mat in enumerate(tmd_model.materials):
                bmat = materials_dict.get(mesh_mat)
                if bmat is not None:
                    mesh.materials.append(bmat)
                    model_mats[mesh_mat] = i

            bm = bmesh.new()
            custom_normals: List[List[float]] = []

            has_skin = bool(self.tmd2.modelFlags & 0x400) and armature is not None
            vgroup_layer = None
            if has_skin:
                vgroup_layer = bm.verts.layers.deform.new("Weights")
                for bone in armature.bones:
                    mesh_obj.vertex_groups.new(name=bone.name)
                armature_modifier = mesh_obj.modifiers.new(name=armature.name, type="ARMATURE")
                armature_modifier.object = armature_obj
                mesh_obj.parent = armature_obj

            has_uv0 = bool(self.tmd2.modelFlags & 0x10)
            has_uv1 = bool(self.tmd2.modelFlags & 0x20)
            has_uv2 = bool(self.tmd2.modelFlags & 0x40)
            has_col0 = bool(self.tmd2.modelFlags & 0x80)
            has_col1 = bool(self.tmd2.modelFlags & 0x200)
            has_nrm = bool(self.tmd2.modelFlags & 0x4)

            if has_uv0:
                bm.loops.layers.uv.new("UVMap")
            if has_uv1:
                bm.loops.layers.uv.new("UVMap1")
            if has_uv2:
                bm.loops.layers.uv.new("UVMap2")
            if has_col0:
                bm.loops.layers.color.new("Color")
            if has_col1:
                bm.loops.layers.color.new("Color2")

            uv0_list: List[float] = []
            uv1_list: List[float] = []
            uv2_list: List[float] = []
            color1_list: List[float] = []
            color2_list: List[float] = []

            for tmd_mesh in tmd_model.meshes:
                tmd_mesh: TMD2Submesh
                if len(tmd_mesh.vertexIndices) == 0 or len(tmd_mesh.triangles) == 0:
                    continue

                mesh_vertices = self.tmd2.vertices[tmd_mesh.vertexIndices]

                bone_ids = None
                bone_weights = None
                if has_skin:
                    if self.tmd2.modelFlags & 0x8000:
                        bone_ids = np.concatenate([mesh_vertices["boneIDs"], mesh_vertices["boneIDs2"]], axis=1)
                        bone_weights = np.concatenate([mesh_vertices["boneWeights"], mesh_vertices["boneWeights2"]], axis=1)
                    else:
                        bone_ids = mesh_vertices["boneIDs"]
                        bone_weights = mesh_vertices["boneWeights"]

                bm_verts = [bm.verts.new(vertex["position"]) for vertex in mesh_vertices]
                bm.verts.ensure_lookup_table()

                if has_nrm:
                    custom_normals.extend(mesh_vertices["normal"][:, :3].tolist())

                if has_skin and vgroup_layer is not None:
                    idx_tbl_len = len(tmd_mesh.indexTable)
                    for i, v in enumerate(bm_verts):
                        for b_id, weight in zip(bone_ids[i], bone_weights[i]):
                            if weight > 0.0 and int(b_id) < idx_tbl_len:
                                v[vgroup_layer][int(tmd_mesh.indexTable[int(b_id)])] = float(weight)

                mat_slot_idx = model_mats.get(tmd_mesh.material, 0)
                accepted_tris: List[Tuple[int, int, int]] = []

                for tri in tmd_mesh.triangles:
                    i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
                    # Skip degenerate zero-area triangles cleanly without desyncing UVs/colors!
                    if i0 == i1 or i1 == i2 or i0 == i2:
                        continue
                    try:
                        face = bm.faces.new((bm_verts[i0], bm_verts[i1], bm_verts[i2]))
                        face.smooth = True
                        face.material_index = mat_slot_idx
                        accepted_tris.append((i0, i1, i2))
                    except ValueError:
                        # Opposite-winding double-sided face or exact duplicate triangle
                        if not self.preserve_double_sided_faces:
                            continue
                        existing = bm.faces.get((bm_verts[i0], bm_verts[i1], bm_verts[i2]))
                        if existing is not None:
                            # Check if winding is opposite to existing face (authored double-sided backface!)
                            ex_verts = tuple(existing.verts)
                            req_verts = (bm_verts[i0], bm_verts[i1], bm_verts[i2])
                            is_same_winding = ex_verts in (
                                req_verts,
                                (req_verts[1], req_verts[2], req_verts[0]),
                                (req_verts[2], req_verts[0], req_verts[1]),
                            )
                            if not is_same_winding:
                                dup_vs = []
                                for orig_idx in (i0, i1, i2):
                                    dv = bm.verts.new(mesh_vertices[orig_idx]["position"])
                                    if has_skin and vgroup_layer is not None:
                                        for b_id, weight in zip(bone_ids[orig_idx], bone_weights[orig_idx]):
                                            if weight > 0.0 and int(b_id) < len(tmd_mesh.indexTable):
                                                dv[vgroup_layer][int(tmd_mesh.indexTable[int(b_id)])] = float(weight)
                                    if has_nrm:
                                        n_orig = mesh_vertices[orig_idx]["normal"][:3]
                                        custom_normals.append([-float(n_orig[0]), -float(n_orig[1]), -float(n_orig[2])])
                                    dup_vs.append(dv)
                                try:
                                    bface = bm.faces.new(dup_vs)
                                    bface.smooth = True
                                    bface.material_index = mat_slot_idx
                                    accepted_tris.append((i0, i1, i2))
                                except ValueError:
                                    pass

                if accepted_tris:
                    acc_arr = np.asarray(accepted_tris, dtype=np.int32)
                    if has_uv0:
                        uv_flat = mesh_vertices["uv"][acc_arr].copy()
                        uv_flat[..., 1] = 1.0 - uv_flat[..., 1]
                        uv0_list.extend(uv_flat.ravel().tolist())
                    if has_uv1:
                        uv_flat = mesh_vertices["uv2"][acc_arr].copy()
                        uv_flat[..., 1] = 1.0 - uv_flat[..., 1]
                        uv1_list.extend(uv_flat.ravel().tolist())
                    if has_uv2:
                        uv_flat = mesh_vertices["uv3"][acc_arr].copy()
                        uv_flat[..., 1] = 1.0 - uv_flat[..., 1]
                        uv2_list.extend(uv_flat.ravel().tolist())
                    if has_col0:
                        color1_list.extend(mesh_vertices["color"][acc_arr].ravel().tolist())
                    if has_col1:
                        color2_list.extend(mesh_vertices["color2"][acc_arr].ravel().tolist())

            bm.to_mesh(mesh)
            bm.free()

            if has_nrm and len(custom_normals) == len(mesh.vertices):
                mesh.normals_split_custom_set_from_vertices(custom_normals)

            if uv0_list and len(mesh.uv_layers) > 0:
                mesh.uv_layers[0].data.foreach_set("uv", uv0_list)
            if uv1_list and len(mesh.uv_layers) > 1:
                mesh.uv_layers[1].data.foreach_set("uv", uv1_list)
            if uv2_list and len(mesh.uv_layers) > 2:
                mesh.uv_layers[2].data.foreach_set("uv", uv2_list)

            if color1_list and "Color" in mesh.color_attributes:
                mesh.color_attributes["Color"].data.foreach_set("color_srgb", color1_list)
            if color2_list and "Color2" in mesh.color_attributes:
                mesh.color_attributes["Color2"].data.foreach_set("color_srgb", color2_list)

            mesh.update()
            if len(mesh.color_attributes) > 0:
                mesh.color_attributes.render_color_index = 0
                mesh.color_attributes.active_color_index = 0

            mesh.transform(YUP_TO_ZUP)


# =====================================================================
# LEGACY TMD, TMD0, TMDV, TMO, TACTPKG IMPORTERS
# =====================================================================

class importTMD:
    def __init__(
        self,
        operator: Operator,
        filepath: str,
        import_settings: dict,
        tmdfile: TMD,
        dds_paths: Optional[Dict[str, str]] = None,
        dds_by_hash: Optional[Dict[int, bpy.types.Image]] = None,
        dds_by_name: Optional[Dict[str, bpy.types.Image]] = None,
        fallback_options: Optional[dict] = None,
    ):
        self.operator = operator
        self.filepath = filepath
        for key, value in import_settings.items():
            setattr(self, key, value)
        self.tmd: TMD = tmdfile
        self.dds_paths = dds_paths or {}
        self.dds_by_hash = dict(dds_by_hash or {})
        self.dds_by_name = dict(dds_by_name or {})
        self.fallback_options = fallback_options or {}

    def read(self, context):
        collection = bpy.data.collections.new(f"{self.tmd.name}")
        context.collection.children.link(collection)

        empty = bpy.data.objects.new(f"#CGTMD2 PROPERTIES [{self.tmd.name}]", None)
        empty.empty_display_size = 0.01
        empty.chatgpt_tmd2_props.format = "TMD"
        collection.objects.link(empty)

        images_list: List[Optional[bpy.types.Image]] = []
        for i, texture in enumerate(self.tmd.textures):
            rname = HASHES.get(str(texture.hash), texture.name or "")
            if rname:
                texture.name = rname
            img = (
                self.dds_by_hash.get(int(texture.hash))
                or _get_or_load_loose_dds(rname, self.dds_paths)
                or bpy.data.images.get(f"{self.tmd.name}_{i}.dds")
                or bpy.data.images.get(f"{self.tmd.name}_{i}")
            )
            images_list.append(img)
            if img is not None:
                self.dds_by_hash[int(texture.hash)] = img

        combined_dds = dict(self.dds_paths)
        combined_dds.update(self.dds_by_name)
        for h_k, img_v in self.dds_by_hash.items():
            combined_dds[str(h_k)] = img_v

        materials_dict = {}
        for i, tm_mat in enumerate(self.tmd.materials):
            mat_name = HASHES.get(str(tm_mat.hash), f"{self.tmd.name}_mat_{i}")
            analysis = analyze_tmd_material(
                tm_mat,
                self.tmd,
                images_list,
                combined_dds,
                HASHES,
                mat_name=mat_name,
                fallback_options=self.fallback_options,
            )
            blender_mat = build_material_from_analysis(mat_name, analysis)
            _populate_blender_mat_props(blender_mat, tm_mat, analysis, self.fallback_options)
            materials_dict[tm_mat] = blender_mat

        YUP_TO_ZUP = Matrix.Rotation(radians(90), 4, "X")
        bones_to_pose = {}
        armature = None
        armature_obj = None

        if self.tmd.modelFlags & 0x2000:
            armature = bpy.data.armatures.new(self.tmd.name)
            armature_obj = bpy.data.objects.new(self.tmd.name, armature)
            collection.objects.link(armature_obj)
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode="EDIT")
            armature_obj.data.display_type = "STICK"

            for tmbone in self.tmd.bones:
                if str(tmbone.hash) in HASHES:
                    tmbone.name = HASHES[str(tmbone.hash)]
                bbone = armature_obj.data.edit_bones.new(tmbone.name or f"bone_{tmbone.hash}")
                matrix = Matrix(tmbone.matrix).transposed().inverted()
                matrix_up = YUP_TO_ZUP @ matrix
                bbone.head = matrix_up @ Vector((0, 0, 0))
                bbone.tail = matrix_up @ Vector((0, 0.05, 0))
                bbone.matrix = matrix_up
                if 0 <= tmbone.parentIndex < len(armature_obj.data.edit_bones):
                    bbone.parent = armature_obj.data.edit_bones[tmbone.parentIndex]
                if tmbone.extra > -1:
                    bones_to_pose[bbone.name] = tmbone.offset
                bbone["hash"] = str(tmbone.hash)
                bbone["extra"] = tmbone.extra
                bbone["offset"] = tmbone.offset
                bbone["posed_loc"] = tmbone.posedLocation
                bbone["unk1"] = tmbone.unk1
                bbone["matrix"] = matrix
                bbone["matrix_up"] = matrix_up

            bpy.ops.object.mode_set(mode="OBJECT")
            for bname, rotation in bones_to_pose.items():
                p = armature_obj.pose.bones.get(bname)
                if p and p.parent:
                    p.rotation_mode = "QUATERNION"
                    p.rotation_quaternion = Quaternion((0, 0, 1), -rotation[2])

        for idx_m, tmd_model in enumerate(self.tmd.models):
            m_name = tmd_model.name or f"{self.tmd.name}_mesh_{idx_m}"
            mesh = bpy.data.meshes.new(m_name)
            mesh_obj = bpy.data.objects.new(m_name, mesh)
            collection.objects.link(mesh_obj)

            mesh_props = mesh_obj.chatgpt_tmd2_mesh
            if tmd_model.hashFlag:
                mesh_props.has_hash = True
                mesh_props.name_hash = str(tmd_model.hash)
            if tmd_model.nameFlag & 4:
                mesh_props.has_name = True
                mesh_props.name = tmd_model.name

            model_mats = {}
            for i, mesh_mat in enumerate(tmd_model.materials):
                bmat = materials_dict.get(mesh_mat)
                if bmat is not None:
                    mesh.materials.append(bmat)
                    model_mats[mesh_mat] = i

            bm = bmesh.new()
            custom_normals = []
            vgroup_layer = None
            if (self.tmd.modelFlags & 0x400) and armature is not None:
                vgroup_layer = bm.verts.layers.deform.new("Weights")
                for bone in armature.bones:
                    mesh_obj.vertex_groups.new(name=bone.name)
                mod = mesh_obj.modifiers.new(name=armature.name, type="ARMATURE")
                mod.object = armature_obj
                mesh_obj.parent = armature_obj

            uv_layers = []
            if self.tmd.modelFlags & 0x10:
                uv_layers.append(bm.loops.layers.uv.new("UVMap"))
            if self.tmd.modelFlags & 0x20:
                uv_layers.append(bm.loops.layers.uv.new("UVMap1"))
            if self.tmd.modelFlags & 0x40:
                uv_layers.append(bm.loops.layers.uv.new("UVMap2"))

            color_layers = []
            if self.tmd.modelFlags & 0x80:
                color_layers.append(bm.loops.layers.color.new("Color"))
            if self.tmd.modelFlags & 0x200:
                color_layers.append(bm.loops.layers.color.new("Color2"))

            for tmd_mesh in tmd_model.meshes:
                bm_verts = []
                for idx_v, vertex in enumerate(tmd_mesh.vertices):
                    v = bm.verts.new(vertex.position)
                    if vgroup_layer is not None:
                        for j in range(4):
                            w = vertex.boneWeights[j]
                            if w > 0.0:
                                v[vgroup_layer][tmd_mesh.indexTable[vertex.boneIDs[j]]] = w
                    bm_verts.append(v)
                    if self.tmd.modelFlags & 0x4:
                        custom_normals.append(list(vertex.normal[:3]))
                bm.verts.ensure_lookup_table()

                for tri in tmd_mesh.triangles:
                    if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                        continue
                    try:
                        face = bm.faces.new([bm_verts[k] for k in tri])
                    except ValueError:
                        continue
                    face.smooth = True
                    face.material_index = model_mats.get(tmd_mesh.material, 0)
                    for u_i, uv_l in enumerate(uv_layers):
                        for lp_i in range(3):
                            vt = tmd_mesh.vertices[tri[lp_i]]
                            src_uv = vt.uv if u_i == 0 else (vt.uv2 if u_i == 1 else vt.uv3)
                            face.loops[lp_i][uv_l].uv = (src_uv[0], 1.0 - src_uv[1])
                    for c_i, col_l in enumerate(color_layers):
                        for lp_i in range(3):
                            vt = tmd_mesh.vertices[tri[lp_i]]
                            src_c = vt.color if c_i == 0 else vt.color2
                            face.loops[lp_i][col_l] = [x / 255.0 if x > 1.0 else x for x in src_c]

            bm.to_mesh(mesh)
            bm.free()
            if (self.tmd.modelFlags & 0x4) and len(custom_normals) == len(mesh.vertices):
                mesh.normals_split_custom_set_from_vertices(custom_normals)
            if len(mesh.color_attributes) > 0:
                mesh.color_attributes.render_color_index = 0
                mesh.color_attributes.active_color_index = 0
            mesh.transform(YUP_TO_ZUP)


class importTMD0:
    def __init__(self, operator: Operator, import_settings: dict, tmdfile: TMD0, images_list=None, fallback_options=None):
        self.operator = operator
        self.tmd: TMD0 = tmdfile
        self.images_list = images_list or []
        self.fallback_options = fallback_options or {}

    def read(self, context):
        collection = bpy.data.collections.new(f"{self.tmd.name}")
        context.collection.children.link(collection)

        materials_dict = {}
        for i, tm_mat in enumerate(self.tmd.materials):
            mat_name = HASHES.get(str(tm_mat.hash), f"{self.tmd.name}_mat_{i}")
            analysis = analyze_tmd_material(
                tm_mat, self.tmd, self.images_list, {}, HASHES, mat_name=mat_name, fallback_options=self.fallback_options
            )
            blender_mat = build_material_from_analysis(mat_name, analysis)
            _populate_blender_mat_props(blender_mat, tm_mat, analysis, self.fallback_options)
            materials_dict[tm_mat] = blender_mat

        YUP_TO_ZUP = Matrix.Rotation(radians(90), 4, "X")
        armature = bpy.data.armatures.new(self.tmd.name)
        armature_obj = bpy.data.objects.new(self.tmd.name, armature)
        collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")
        armature_obj.data.display_type = "STICK"

        for tmbone in self.tmd.bones:
            if str(tmbone.hash) in HASHES:
                tmbone.name = HASHES[str(tmbone.hash)]
            bbone = armature_obj.data.edit_bones.new(tmbone.name or f"bone_{tmbone.hash}")
            matrix = Matrix(tmbone.matrix).transposed().inverted()
            matrix_up = YUP_TO_ZUP @ matrix
            bbone.head = matrix_up @ Vector((0, 0, 0))
            bbone.tail = matrix_up @ Vector((0, 0.05, 0))
            bbone.matrix = matrix_up
            if 0 <= tmbone.parentIndex < len(armature_obj.data.edit_bones):
                bbone.parent = armature_obj.data.edit_bones[tmbone.parentIndex]
            bbone["hash"] = str(tmbone.hash)
            bbone["posed_loc"] = tmbone.posedLocation
            bbone["matrix"] = matrix
            bbone["matrix_up"] = matrix_up
        bpy.ops.object.mode_set(mode="OBJECT")

        for tmd_model in self.tmd.models:
            mesh = bpy.data.meshes.new(tmd_model.name)
            mesh_obj = bpy.data.objects.new(tmd_model.name, mesh)
            collection.objects.link(mesh_obj)
            model_mats = {}
            for i, mesh_mat in enumerate(tmd_model.materials):
                mesh.materials.append(materials_dict[mesh_mat])
                model_mats[mesh_mat] = i

            bm = bmesh.new()
            custom_normals = []
            vgroup_layer = None
            if self.tmd.modelFlags & 0x800:
                vgroup_layer = bm.verts.layers.deform.new("Weights")
                for bone in armature.bones:
                    mesh_obj.vertex_groups.new(name=bone.name)
                mod = mesh_obj.modifiers.new(name=armature.name, type="ARMATURE")
                mod.object = armature_obj
                mesh_obj.parent = armature_obj

            has_uv = bool(self.tmd.modelFlags & 0x40)
            has_color = bool(self.tmd.modelFlags & 0x20)
            if has_uv:
                bm.loops.layers.uv.new("UVMap")
            if has_color:
                bm.loops.layers.color.new("Color")

            uv0_list, color0_list = [], []
            for tmd_mesh in tmd_model.meshes:
                unique_indices, inverse = np.unique(tmd_mesh.triangles, return_inverse=True)
                local_tris = inverse.reshape(-1, 3)
                mesh_vertices = self.tmd.vertices[unique_indices]
                bm_verts = [bm.verts.new(v["position"]) for v in mesh_vertices]
                bm.verts.ensure_lookup_table()
                if vgroup_layer is not None:
                    for i, v in enumerate(bm_verts):
                        for bid, bw in zip(mesh_vertices["boneIDs"][i], mesh_vertices["boneWeights"][i]):
                            if bw > 0.0:
                                v[vgroup_layer][tmd_mesh.indexTable[int(bid)]] = float(bw)
                if self.tmd.modelFlags & 0x4:
                    custom_normals.extend(mesh_vertices["normal"][:, :3].tolist())
                accepted = []
                for tri in local_tris:
                    try:
                        face = bm.faces.new([bm_verts[i] for i in tri])
                    except ValueError:
                        continue
                    face.smooth = True
                    face.material_index = model_mats.get(tmd_mesh.material, 0)
                    accepted.append(tri)
                if accepted:
                    acc = np.array(accepted)
                    if has_uv:
                        uvs = mesh_vertices["uv"][acc].copy()
                        uvs[..., 1] = 1.0 - uvs[..., 1]
                        uv0_list.extend(uvs.ravel().tolist())
                    if has_color:
                        color0_list.extend(mesh_vertices["color"][acc].ravel().tolist())

            bm.to_mesh(mesh)
            bm.free()
            if (self.tmd.modelFlags & 0x4) and len(custom_normals) == len(mesh.vertices):
                mesh.normals_split_custom_set_from_vertices(custom_normals)
            if uv0_list:
                mesh.uv_layers[0].data.foreach_set("uv", uv0_list)
            if color0_list and "Color" in mesh.color_attributes:
                mesh.color_attributes["Color"].data.foreach_set("color_srgb", color0_list)
            mesh.update()
            mesh.transform(YUP_TO_ZUP)


class importTMDV:
    def __init__(self, operator: Operator, import_settings: dict, tmdfile: TMDV, images_list=None, fallback_options=None):
        self.operator = operator
        self.tmd: TMDV = tmdfile
        self.images_list = images_list or []
        self.fallback_options = fallback_options or {}

    def read(self, context):
        collection = bpy.data.collections.new(f"{self.tmd.name}")
        context.collection.children.link(collection)

        materials_dict = {}
        for i, tm_mat in enumerate(self.tmd.materials):
            mat_name = HASHES.get(str(tm_mat.hash), f"{self.tmd.name}_mat_{i}")
            analysis = analyze_tmd_material(
                tm_mat, self.tmd, self.images_list, {}, HASHES, mat_name=mat_name, fallback_options=self.fallback_options
            )
            blender_mat = build_material_from_analysis(mat_name, analysis)
            _populate_blender_mat_props(blender_mat, tm_mat, analysis, self.fallback_options)
            materials_dict[tm_mat] = blender_mat

        YUP_TO_ZUP = Matrix.Rotation(radians(90), 4, "X")
        armature = bpy.data.armatures.new(self.tmd.name)
        armature_obj = bpy.data.objects.new(self.tmd.name, armature)
        collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode="EDIT")
        armature_obj.data.display_type = "STICK"
        for tmbone in self.tmd.bones:
            if str(tmbone.hash) in HASHES:
                tmbone.name = HASHES[str(tmbone.hash)]
            bbone = armature_obj.data.edit_bones.new(tmbone.name or f"bone_{tmbone.hash}")
            matrix = Matrix(tmbone.matrix).transposed().inverted()
            matrix_up = YUP_TO_ZUP @ matrix
            bbone.head = matrix_up @ Vector((0, 0, 0))
            bbone.tail = matrix_up @ Vector((0, 0.05, 0))
            bbone.matrix = matrix_up
            if 0 <= tmbone.parentIndex < len(armature_obj.data.edit_bones):
                bbone.parent = armature_obj.data.edit_bones[tmbone.parentIndex]
            bbone["hash"] = str(tmbone.hash)
            bbone["posed_loc"] = tmbone.posedLocation
            bbone["matrix"] = matrix
            bbone["matrix_up"] = matrix_up
        bpy.ops.object.mode_set(mode="OBJECT")

        for tmd_model in self.tmd.models:
            mesh = bpy.data.meshes.new(tmd_model.name)
            mesh_obj = bpy.data.objects.new(tmd_model.name, mesh)
            collection.objects.link(mesh_obj)
            model_mats = {}
            for i, mesh_mat in enumerate(tmd_model.materials):
                mesh.materials.append(materials_dict[mesh_mat])
                model_mats[mesh_mat] = i

            bm = bmesh.new()
            custom_normals = []
            vgroup_layer = None
            if self.tmd.modelFlags & 0x800:
                vgroup_layer = bm.verts.layers.deform.new("Weights")
                for bone in armature.bones:
                    mesh_obj.vertex_groups.new(name=bone.name)
                mod = mesh_obj.modifiers.new(name=armature.name, type="ARMATURE")
                mod.object = armature_obj
                mesh_obj.parent = armature_obj

            has_uv = bool(self.tmd.modelFlags & 0x40)
            has_uv2 = bool(self.tmd.modelFlags & 0x80)
            has_color = bool(self.tmd.modelFlags & 0x20)
            if has_uv:
                bm.loops.layers.uv.new("UVMap")
            if has_uv2:
                bm.loops.layers.uv.new("UVMap1")
            if has_color:
                bm.loops.layers.color.new("Color")

            uv0_list, uv1_list, color0_list = [], [], []
            for tmd_mesh in tmd_model.meshes:
                unique_indices, inverse = np.unique(tmd_mesh.triangles, return_inverse=True)
                local_tris = inverse.reshape(-1, 3)
                mesh_vertices = self.tmd.vertices[unique_indices]
                bm_verts = [bm.verts.new(v["position"]) for v in mesh_vertices]
                bm.verts.ensure_lookup_table()
                if vgroup_layer is not None:
                    for i, v in enumerate(bm_verts):
                        for bid, bw in zip(mesh_vertices["boneIDs"][i], mesh_vertices["boneWeights"][i]):
                            if bw > 0.0:
                                v[vgroup_layer][tmd_mesh.indexTable[int(bid)]] = float(bw)
                if self.tmd.modelFlags & 0x4:
                    custom_normals.extend(mesh_vertices["normal"][:, :3].tolist())
                accepted = []
                for tri in local_tris:
                    try:
                        face = bm.faces.new([bm_verts[i] for i in tri])
                    except ValueError:
                        continue
                    face.smooth = True
                    face.material_index = model_mats.get(tmd_mesh.material, 0)
                    accepted.append(tri)
                if accepted:
                    acc = np.array(accepted)
                    if has_uv:
                        uvs = mesh_vertices["uv"][acc].copy()
                        uvs[..., 1] = 1.0 - uvs[..., 1]
                        uv0_list.extend(uvs.ravel().tolist())
                    if has_uv2:
                        uvs = mesh_vertices["uv1"][acc].copy()
                        uvs[..., 1] = 1.0 - uvs[..., 1]
                        uv1_list.extend(uvs.ravel().tolist())
                    if has_color:
                        color0_list.extend(mesh_vertices["color"][acc].ravel().tolist())

            bm.to_mesh(mesh)
            bm.free()
            if (self.tmd.modelFlags & 0x4) and len(custom_normals) == len(mesh.vertices):
                mesh.normals_split_custom_set_from_vertices(custom_normals)
            if uv0_list:
                mesh.uv_layers[0].data.foreach_set("uv", uv0_list)
            if uv1_list:
                mesh.uv_layers[1].data.foreach_set("uv", uv1_list)
            if color0_list and "Color" in mesh.color_attributes:
                mesh.color_attributes["Color"].data.foreach_set("color_srgb", color0_list)
            mesh.update()
            mesh.transform(YUP_TO_ZUP)


class importTMO:
    def __init__(self, operator: Operator, import_settings: dict, tmofile: TMO):
        self.operator = operator
        self.tmo: TMO = tmofile

    def read(self, context):
        target = context.object
        if not target or target.type != "ARMATURE" or not target.data:
            self.operator.report({"ERROR"}, "Select an Armature before importing a .tmo animation.")
            return {"CANCELLED"}

        action = bpy.data.actions.new(name=f"{self.tmo.name}_action")
        target.animation_data_create()
        target.animation_data.action = action

        bone_hashes = {}
        for bone in target.data.bones:
            if "hash" in bone.keys():
                bone_hashes[str(bone["hash"])] = bone.name
            else:
                bone_hashes[str(tamCRC32(bone.name))] = bone.name

        for bidx, tmbone in enumerate(self.tmo.hashes):
            bname = bone_hashes.get(str(tmbone), "")
            bone = target.data.bones.get(bname)
            if not bone:
                continue
            target.pose.bones[bone.name].rotation_mode = "XYZ"

            for i in range(3):
                fcurve = action.fcurves.new(data_path=f'pose.bones["{bone.name}"].location', index=i)
                offset = self.tmo.fCurves[i + bidx * 9]
                start_frame = int(offset["startFrame"])
                frame_count = int(offset["frameCount"])
                if frame_count == 0:
                    continue
                kf_slice = self.tmo.keyframes[start_frame : start_frame + frame_count]
                co_flat = np.empty(frame_count * 2, dtype=np.float64)
                co_flat[0::2] = kf_slice["frame"]
                co_flat[1::2] = kf_slice["value"]
                fcurve.keyframe_points.add(frame_count)
                fcurve.keyframe_points.foreach_set("co", co_flat.tolist())
                fcurve.update()

            for ri in range(3):
                offset = self.tmo.fCurves[(3 + ri) + bidx * 9]
                start = int(offset["startFrame"])
                count = int(offset["frameCount"])
                if count == 0:
                    continue
                kf = self.tmo.keyframes[start : start + count]
                fcurve = action.fcurves.new(data_path=f'pose.bones["{bone.name}"].rotation_euler', index=ri)
                co_flat = np.empty(count * 2, dtype=np.float64)
                co_flat[0::2] = kf["frame"]
                co_flat[1::2] = kf["value"]
                fcurve.keyframe_points.add(count)
                fcurve.keyframe_points.foreach_set("co", co_flat.tolist())
                fcurve.update()
        return {"FINISHED"}


def importActorPackage(file_path: str, operator: Operator):
    ap: TactPkg = readActorPackage(file_path)
    if not ap or not ap.tmoPkg:
        return
    for tmo_name, tmo in ap.tmoPkg.tmoFiles.items():
        tmo.name = tmo_name
        importTMO(operator, {}, tmo).read(bpy.context)


# =====================================================================
# VIEWPORT DRAG-AND-DROP OPERATORS & FILE HANDLERS (UNIQUE CGTMD2 IDs)
# =====================================================================

class CGTMD2_OT_DropImport(Operator):
    """Import dropped TamEngine files using ChatGPT TMD2"""
    bl_idname = "import_scene.chatgpt_drop_any"
    bl_label = "Import with ChatGPT TMD2"
    bl_options = {"REGISTER", "UNDO"}

    files: CollectionProperty(type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    filepath: StringProperty(subtype="FILE_PATH")

    def execute(self, context):
        t0 = perf_counter()
        dds_by_hash, dds_by_name, dds_paths = scan_directory_textures(self.directory)
        lds_map: Dict[str, str] = {}
        if self.directory and os.path.isdir(self.directory):
            for fn in os.listdir(self.directory):
                if fn.lower().endswith(".lds"):
                    lds_map[os.path.splitext(fn)[0].lower()] = os.path.join(self.directory, fn)

        for f in self.files:
            full = os.path.join(self.directory, f.name)
            bkey, ext = os.path.splitext(f.name)
            ext_l = ext.lower()
            if ext_l == ".tmd2":
                tmd2 = readTMD2(full)
                if tmd2 is not None:
                    imp = importTMD2(self, full, {}, tmd2, dds_paths, dds_by_hash, dds_by_name)
                    imp.texture_path = _find_matching_lds(self.directory, bkey.lower(), lds_map)
                    imp.read(context)
            elif ext_l == ".lds":
                importLDS(full, return_tex=False)
            elif ext_l == ".vfxt":
                importVFXT(full)
            elif ext_l == ".tmd":
                hnames = {tamCRC32(k): k for k in dds_paths.keys()}
                tmd = readTMD(full, hnames)
                if tmd is not None:
                    importTMD(self, full, {}, tmd, dds_paths, dds_by_hash, dds_by_name).read(context)
            elif ext_l == ".tmd0":
                hnames = {tamCRC32(k): k for k in dds_paths.keys()}
                tmd0 = readTMD0(full, hnames)
                if tmd0 is not None:
                    importTMD0(self, {}, tmd0, []).read(context)
            elif ext_l == ".tmdv":
                tmdv = readTMDV(full)
                if tmdv is not None:
                    importTMDV(self, {}, tmdv, []).read(context)
            elif ext_l == ".cat":
                _import_cats_archive(self, context, full, {}, {})
            elif ext_l == ".tmo":
                tmo = readTMO(full)
                if tmo is not None:
                    importTMO(self, {}, tmo).read(context)
            elif ext_l == ".tactpkg":
                importActorPackage(full, self)

        elapsed = perf_counter() - t0
        self.report({"INFO"}, f"ChatGPT TMD2 imported {len(self.files)} file(s) in {elapsed:.2f}s.")
        return {"FINISHED"}


class CGTMD2_FH_Import(bpy.types.FileHandler):
    bl_idname = "CGTMD2_FH_import"
    bl_label = "ChatGPT TMD2 (.tmd2, .lds, .vfxt, .tmd, .cat, .tmo)"
    bl_import_operator = "import_scene.chatgpt_drop_any"
    bl_file_extensions = ".tmd2;.lds;.vfxt;.tmd;.tmd0;.tmdv;.cat;.tmo;.tactpkg"

    @classmethod
    def poll_drop(cls, context):
        return bool(context.area and context.area.type == "VIEW_3D")


# =====================================================================
# MENU & REGISTRATION
# =====================================================================

def menu_func_import(self, context):
    self.layout.operator(
        CGTMD2_IMPORTER_OT_IMPORT.bl_idname,
        text="ChatGPT TMD2 (.tmd2, .tmd, .lds, .vfxt)",
        icon="IMPORT",
    )
    self.layout.operator(
        CGTMD2_CAT_OT_IMPORT.bl_idname,
        text="ChatGPT TMD2 CAT Archive (.cat)",
        icon="IMPORT",
    )


IMPORTER_CLASSES = (
    CGTMD2_IMPORTER_OT_IMPORT,
    CGTMD2_CAT_OT_IMPORT,
    CGTMD2_OT_DropImport,
    CGTMD2_FH_Import,
)


def register():
    for cls in IMPORTER_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(IMPORTER_CLASSES):
        bpy.utils.unregister_class(cls)
