import json
import math
import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import bpy
import numpy as np


# ---------------------------------------------------------------------------
# Known TamEngine Shader IDs & Semantic Families
# ---------------------------------------------------------------------------

KNOWN_CHARACTER_SHADERS = {"BSS1", "HSK1"}
KNOWN_HAIR_SHADERS = {"BHG1", "BHA1"}
KNOWN_EYE_SHADERS = {"BEI0", "BED0", "BES0", "BEH0"}
KNOWN_GHOST_SHADERS = {"BGH0", "BGH1"}
KNOWN_ALL_SHADERS = (
    KNOWN_CHARACTER_SHADERS
    | KNOWN_HAIR_SHADERS
    | KNOWN_EYE_SHADERS
    | KNOWN_GHOST_SHADERS
)

_NORMAL_NAME_RE = re.compile(r"(?:^|[_\-.])(n|nrm|norm|normal|bumpmap)(?:[_\-.\d]|$)", re.IGNORECASE)
_RAMP_NAME_RE = re.compile(r"(?:grad|grade|ramp|lut|toon_ramp|colorramp)", re.IGNORECASE)
_MATCAP_NAME_RE = re.compile(r"(?:^mr_|_mr_|matcap|spheremap|goldmetal|metalref|envmap)", re.IGNORECASE)
_EMISSIVE_NAME_RE = re.compile(r"(?:glow|emis|emit|aura|fire|flame|light|beam|laser|thunder|spark|shot)", re.IGNORECASE)
_ADDITIVE_NAME_RE = re.compile(r"(?:^add_|_add(?:_|$)|additive|ef_|_eff(?:_|$)|vfx)", re.IGNORECASE)
_GHOST_NAME_RE = re.compile(r"(?:ghost|spirit|phantom|soul|hollow|holo|afterimage|mirage)", re.IGNORECASE)
_ALPHA_NAME_RE = re.compile(r"(?:alpha|transp|glass|water|smoke|cloud|mist|fog)", re.IGNORECASE)
_GOLD_METAL_NAME_RE = re.compile(r"(?:gold|metal|silver|bronze|chrome|trophy|emblem)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Persistent Custom Shader Rules Registry (for user-configured unknown shaders)
# ---------------------------------------------------------------------------

def _get_rules_filepath() -> str:
    try:
        cfg_dir = bpy.utils.user_resource("CONFIG", path="chatgpt_tmd2", create=True)
        if cfg_dir:
            return os.path.join(cfg_dir, "custom_shader_rules.json")
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "custom_shader_rules.json")


def load_custom_shader_rules() -> Dict[str, dict]:
    path = _get_rules_filepath()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def save_custom_shader_rule(shader_id: str, rule: dict) -> str:
    sid = (shader_id or "").strip()
    if not sid:
        return ""
    rules = load_custom_shader_rules()
    rules[sid] = rule
    path = _get_rules_filepath()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    return path


def delete_custom_shader_rule(shader_id: str) -> bool:
    sid = (shader_id or "").strip()
    rules = load_custom_shader_rules()
    if sid in rules:
        del rules[sid]
        path = _get_rules_filepath()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        return True
    return False


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TextureInspection:
    name: str = ""
    width: int = 0
    height: int = 0
    dds_format: str = "UNKNOWN"
    is_grayscale: bool = False
    has_alpha: bool = False
    alpha_zeroed: bool = False
    is_smooth_alpha: bool = False
    is_cutout_alpha: bool = False
    is_dark_bg_energy: bool = False
    is_normal_map: bool = False
    needs_normal_z_reconstruct: bool = False
    is_ramp_lut: bool = False
    is_matcap_sphere: bool = False
    mean_rgb: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    p95_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    mean_alpha: float = 1.0
    min_alpha: float = 1.0
    max_alpha: float = 1.0
    dark_fraction: float = 0.0


@dataclass
class TextureSlotBinding:
    role: str  # 'albedo', 'ramp', 'ilm_shadow', 'specular_emissive', 'normal', 'spec_mask', 'detail_mask', 'extra'
    list_index: int
    sampler_slot: int
    tex_hash: int
    resolved_name: str
    image: Optional[bpy.types.Image]
    inspection: TextureInspection
    uv_index: int = 0
    non_color: bool = False
    clamp_uv: bool = False
    unk1: int = 0
    unk2: int = 0
    unk3: int = 0
    is_external: bool = False


@dataclass
class SubmeshUsageInfo:
    uv_count: int = 1
    has_normals: bool = True
    has_color0: bool = False
    color0_rgb_active: bool = False
    color0_is_tinted: bool = False
    color0_is_white: bool = True
    color0_mean_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color0_alpha_active: bool = False
    color0_alpha_zeroed: bool = False
    color0_min_alpha: float = 1.0
    color0_mean_alpha: float = 1.0
    has_color1: bool = False
    color1_rgb_active: bool = False
    color1_is_tinted: bool = False
    color1_mean_rgb: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color1_alpha_active: bool = False
    has_double_sided_faces: bool = False


@dataclass
class MaterialAnalysis:
    shader_id: str = ""
    is_known_shader: bool = False
    shader_family: str = "STANDARD"  # 'TOON_CHAR', 'HAIR', 'EYE', 'SKIN', 'VFX_ADDITIVE', 'GHOST_SPIRIT', 'METALLIC_GLOW', 'STAGE', 'STANDARD'
    blend_mode: str = "OPAQUE"       # 'OPAQUE', 'ALPHA_CLIP', 'ALPHA_BLEND', 'ADDITIVE', 'GHOST'
    material_name: str = ""
    resolved_hash_name: str = ""
    model_names: List[str] = field(default_factory=list)

    # Extracted Colors & Parameters (always derived from actual TMD2 data, never random)
    base_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    secondary_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    shadow_color: Tuple[float, float, float, float] = (0.45, 0.42, 0.48, 1.0)
    shadow_intensity: float = 0.20
    specular_color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    specular_size: float = 0.5
    rim_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    rim_strength: float = 0.0
    rim_power: float = 3.0
    emission_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    emission_strength: float = 0.0
    metallic: float = 0.0
    roughness: float = 0.45
    opacity: float = 1.0
    alpha_clip_threshold: float = 0.5
    color_burn_opacity: float = 0.0
    color_burn_intensity: float = 0.0

    # Parameter index provenance (which indices in shaderParams drove each value; -1 = none/default)
    base_color_param_idx: int = -1
    secondary_color_param_idx: int = -1
    specular_color_param_idx: int = -1
    emission_color_param_idx: int = -1
    opacity_param_idx: int = -1
    alpha_clip_param_idx: int = -1
    uv_scale: Tuple[float, float] = (1.0, 1.0)
    uv_offset: Tuple[float, float] = (0.0, 0.0)

    # Feature flags
    use_vertex_color_rgb: bool = False
    vertex_color_layer: str = "Color"
    use_vertex_color_alpha: bool = False
    use_vertex_color2_fade: bool = False
    colorize_grayscale: bool = False
    use_ramp_lut: bool = False
    luminance_to_alpha: bool = False
    fresnel_translucency: bool = False
    backface_culling: bool = False
    flip_normal_green: bool = True

    # Bound textures & diagnostics
    bindings: List[TextureSlotBinding] = field(default_factory=list)
    submesh_usage: SubmeshUsageInfo = field(default_factory=SubmeshUsageInfo)
    diagnostic_notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DDS Header & Pixel Inspection
# ---------------------------------------------------------------------------

_DXGI_BC_MAP = {
    70: "BC1", 71: "BC1", 72: "BC1_SRGB",
    73: "BC2", 74: "BC2", 75: "BC2_SRGB",
    76: "BC3", 77: "BC3", 78: "BC3_SRGB",
    79: "BC4", 80: "BC4_UNORM", 81: "BC4_SNORM",
    82: "BC5", 83: "BC5_UNORM", 84: "BC5_SNORM",
    95: "BC6H", 96: "BC6H",
    97: "BC7", 98: "BC7", 99: "BC7_SRGB",
    28: "RGBA8", 29: "RGBA8_SRGB", 87: "BGRA8", 91: "BGRA8_SRGB",
    61: "R8",
}


def _parse_dds_format(raw_bytes: bytes) -> str:
    if not raw_bytes or len(raw_bytes) < 128 or raw_bytes[:4] != b"DDS ":
        return "UNKNOWN"
    fourcc = raw_bytes[84:88]
    if fourcc == b"DXT1":
        return "BC1"
    if fourcc in (b"DXT2", b"DXT3"):
        return "BC2"
    if fourcc in (b"DXT4", b"DXT5"):
        return "BC3"
    if fourcc in (b"ATI1", b"BC4U", b"BC4S"):
        return "BC4"
    if fourcc in (b"ATI2", b"BC5U", b"BC5S"):
        return "BC5"
    if fourcc == b"DX10" and len(raw_bytes) >= 132:
        dxgi_fmt = struct.unpack_from("<I", raw_bytes, 128)[0]
        return _DXGI_BC_MAP.get(dxgi_fmt, f"DX10_{dxgi_fmt}")
    return "RGBA"


_IMAGE_INSPECTION_CACHE: Dict[Tuple[str, int, int], TextureInspection] = {}


def inspect_blender_image(image: Optional[bpy.types.Image], hint_name: str = "") -> TextureInspection:
    info = TextureInspection(name=hint_name or (image.name if image else ""))
    if image is None:
        return info

    w, h = int(image.size[0]), int(image.size[1])
    info.width = w
    info.height = h
    cache_key = (image.name, w, h)
    if cache_key in _IMAGE_INSPECTION_CACHE and not hint_name:
        return _IMAGE_INSPECTION_CACHE[cache_key]

    check_name = f"{image.name} {hint_name}".strip()
    if image.packed_file and image.packed_file.data:
        info.dds_format = _parse_dds_format(image.packed_file.data[:148])

    if (w >= 16 and 1 <= h <= 32 and w >= 4 * h) or (h >= 16 and 1 <= w <= 32 and h >= 4 * w) or bool(_RAMP_NAME_RE.search(check_name)):
        info.is_ramp_lut = True

    if bool(_MATCAP_NAME_RE.search(check_name)):
        info.is_matcap_sphere = True

    if info.dds_format.startswith("BC5") or bool(_NORMAL_NAME_RE.search(check_name)):
        info.is_normal_map = True

    if w > 0 and h > 0 and image.channels >= 3:
        try:
            total_px = w * h
            ch = image.channels
            raw = np.empty(total_px * ch, dtype=np.float32)
            image.pixels.foreach_get(raw)
            px = raw.reshape(total_px, ch)
            # Subsample up to 4096 pixels uniformly for <1ms speed
            step = max(1, total_px // 4096)
            sample = px[::step]
            rgb = sample[:, :3]
            alpha = sample[:, 3] if ch >= 4 else np.ones(len(sample), dtype=np.float32)

            mean_rgb = rgb.mean(axis=0)
            p95_rgb = np.percentile(rgb, 95, axis=0)
            info.mean_rgb = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
            info.p95_rgb = (float(p95_rgb[0]), float(p95_rgb[1]), float(p95_rgb[2]))

            a_min = float(alpha.min())
            a_max = float(alpha.max())
            a_mean = float(alpha.mean())
            info.min_alpha = a_min
            info.max_alpha = a_max
            info.mean_alpha = a_mean

            if a_max <= 0.01:
                info.alpha_zeroed = True
                info.has_alpha = False
            elif a_min < 0.98:
                info.has_alpha = True
                mid_frac = float(np.mean((alpha > 0.08) & (alpha < 0.92)))
                if mid_frac > 0.025:
                    info.is_smooth_alpha = True
                else:
                    info.is_cutout_alpha = True

            chroma = np.max(rgb, axis=1) - np.min(rgb, axis=1)
            p95_chroma = float(np.percentile(chroma, 95))
            if info.dds_format.startswith("BC4") or p95_chroma < 0.035:
                info.is_grayscale = True

            lum = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
            info.dark_fraction = float(np.mean(lum < 0.06))
            p95_lum = float(np.percentile(lum, 95))
            if not info.has_alpha and info.dark_fraction > 0.18 and p95_lum > 0.28:
                info.is_dark_bg_energy = True

            # Normal map heuristic from pixel distribution (if not ramp/grayscale)
            if not info.is_grayscale and not info.is_ramp_lut:
                mr, mg, mb = info.mean_rgb
                std_b = float(np.std(rgb[:, 2]))
                if 0.38 <= mr <= 0.62 and 0.38 <= mg <= 0.62:
                    if mb >= 0.74 and std_b < 0.20:
                        info.is_normal_map = True
                    elif mb <= 0.06 and std_b < 0.04 and (float(np.std(rgb[:, 0])) > 0.015 or (0.45 <= mr <= 0.55 and 0.45 <= mg <= 0.55)):
                        info.is_normal_map = True
                        info.needs_normal_z_reconstruct = True

            if info.is_normal_map and info.mean_rgb[2] < 0.20:
                info.needs_normal_z_reconstruct = True
        except Exception:
            pass

    if not hint_name:
        _IMAGE_INSPECTION_CACHE[cache_key] = info
    return info


# ---------------------------------------------------------------------------
# Submesh Vertex Color & UV Usage Inspection
# ---------------------------------------------------------------------------

def inspect_submesh_usage(tmd_obj, target_material) -> SubmeshUsageInfo:
    usage = SubmeshUsageInfo()
    flags = getattr(tmd_obj, "modelFlags", 0)
    is_legacy_tmd0_or_v = type(tmd_obj).__name__ in ("TMD0", "TMDV")

    if is_legacy_tmd0_or_v:
        uv_c = (1 if (flags & 0x40) else 0) + (1 if (flags & 0x80 and type(tmd_obj).__name__ == "TMDV") else 0)
        usage.uv_count = max(1, uv_c)
        usage.has_normals = bool(flags & 0x4)
        has_c0 = bool(flags & 0x20)
        has_c1 = False
    else:
        uv_c = (1 if (flags & 0x10) else 0) + (1 if (flags & 0x20) else 0) + (1 if (flags & 0x40) else 0)
        usage.uv_count = max(1, uv_c)
        usage.has_normals = bool(flags & 0x4)
        has_c0 = bool(flags & 0x80)
        has_c1 = bool(flags & 0x200)

    usage.has_color0 = has_c0
    usage.has_color1 = has_c1

    # Gather vertex colors across all submeshes assigned to target_material
    c0_chunks = []
    c1_chunks = []
    double_sided = False

    for model in getattr(tmd_obj, "models", []):
        for sub in getattr(model, "meshes", []):
            if getattr(sub, "material", None) is not target_material:
                continue
            tris = getattr(sub, "triangles", None)
            if tris is not None and len(tris) > 0:
                tri_arr = np.asarray(tris)
                if tri_arr.ndim == 2 and tri_arr.shape[1] == 3:
                    sorted_tris = np.sort(tri_arr, axis=1)
                    if len(np.unique(sorted_tris, axis=0)) < len(sorted_tris):
                        double_sided = True

            verts_global = getattr(tmd_obj, "vertices", None)
            if isinstance(verts_global, np.ndarray) and verts_global.dtype.names:
                v_indices = getattr(sub, "vertexIndices", None)
                if v_indices is None and tris is not None:
                    v_indices = np.unique(np.asarray(tris))
                if v_indices is not None and len(v_indices) > 0:
                    sub_v = verts_global[v_indices]
                    if has_c0 and "color" in sub_v.dtype.names:
                        c0_chunks.append(sub_v["color"])
                    if has_c1 and "color2" in sub_v.dtype.names:
                        c1_chunks.append(sub_v["color2"])
            elif hasattr(sub, "vertices") and sub.vertices:
                if has_c0:
                    c0_chunks.append(np.array([v.color for v in sub.vertices], dtype=np.float32))
                if has_c1:
                    c1_chunks.append(np.array([v.color2 for v in sub.vertices], dtype=np.float32))

    usage.has_double_sided_faces = double_sided

    if c0_chunks:
        c0 = np.concatenate(c0_chunks, axis=0)
        if len(c0) > 0 and c0.shape[-1] >= 3:
            rgb0 = c0[:, :3]
            max_rgb = float(rgb0.max())
            min_rgb = float(rgb0.min())
            mean_rgb = rgb0.mean(axis=0)
            usage.color0_mean_rgb = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
            usage.color0_rgb_active = max_rgb > 0.015
            usage.color0_is_white = min_rgb > 0.965
            chroma = float(np.percentile(np.max(rgb0, axis=1) - np.min(rgb0, axis=1), 90))
            if usage.color0_rgb_active and (chroma > 0.03 or not usage.color0_is_white):
                usage.color0_is_tinted = True

            if c0.shape[-1] >= 4:
                a0 = c0[:, 3]
                a_min = float(a0.min())
                a_max = float(a0.max())
                usage.color0_min_alpha = a_min
                usage.color0_mean_alpha = float(a0.mean())
                if a_max <= 0.01:
                    usage.color0_alpha_zeroed = True
                elif a_min < 0.975:
                    usage.color0_alpha_active = True

    if c1_chunks:
        c1 = np.concatenate(c1_chunks, axis=0)
        if len(c1) > 0 and c1.shape[-1] >= 3:
            rgb1 = c1[:, :3]
            mean_rgb1 = rgb1.mean(axis=0)
            usage.color1_mean_rgb = (float(mean_rgb1[0]), float(mean_rgb1[1]), float(mean_rgb1[2]))
            usage.color1_rgb_active = float(rgb1.max()) > 0.015
            chroma1 = float(np.percentile(np.max(rgb1, axis=1) - np.min(rgb1, axis=1), 90))
            if usage.color1_rgb_active and (chroma1 > 0.03 or float(rgb1.min()) < 0.965):
                usage.color1_is_tinted = True
            if c1.shape[-1] >= 4:
                a1 = c1[:, 3]
                if float(a1.max()) > 0.01 and float(a1.min()) < 0.975:
                    usage.color1_alpha_active = True

    return usage


# ---------------------------------------------------------------------------
# Shader Parameter Constant-Buffer Register Analyzer
# ---------------------------------------------------------------------------

def _is_finite_nonneg(vals) -> bool:
    return all(math.isfinite(v) and v >= -1e-4 for v in vals)


def _is_uv_transform_vec4(v0: float, v1: float, v2: float, v3: float) -> bool:
    """Detects common HLSL UV scale/offset vec4s like (1, 1, 0, 0) or (2, 2, 0, 0)
    so they are never misidentified as yellow RGB=(1, 1, 0)."""
    if abs(v2) < 1e-5 and abs(v3) < 1e-5 and v0 > 0.0 and abs(v0 - v1) < 1e-5:
        if v0 in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0):
            return True
    return False


def _normalize_hdr_color(r: float, g: float, b: float) -> Tuple[Tuple[float, float, float, float], float]:
    peak = max(r, g, b)
    if peak <= 1e-6:
        return (0.0, 0.0, 0.0, 1.0), 0.0
    if peak > 1.0:
        return (max(0.0, r / peak), max(0.0, g / peak), max(0.0, b / peak), 1.0), float(peak)
    return (max(0.0, r), max(0.0, g), max(0.0, b), 1.0), 1.0


def _classify_shader_family(
    shader_id: str,
    mat_name: str,
    model_names: List[str],
    tex_bindings: List[TextureSlotBinding],
    usage: SubmeshUsageInfo,
    params: List[float],
) -> str:
    sid = (shader_id or "").upper().strip()
    combined_names = " ".join([mat_name, *model_names, *(b.resolved_name for b in tex_bindings)])

    # 1. Exact known TamEngine IDs
    if sid in KNOWN_CHARACTER_SHADERS:
        return "SKIN" if sid == "HSK1" else "TOON_CHAR"
    if sid in KNOWN_HAIR_SHADERS:
        return "HAIR"
    if sid in KNOWN_EYE_SHADERS:
        return "EYE"
    if sid in KNOWN_GHOST_SHADERS:
        return "TOON_CHAR"

    # 2. Name/hash-based explicit cues from actual game strings
    if _GHOST_NAME_RE.search(combined_names):
        return "GHOST_SPIRIT"
    if _ADDITIVE_NAME_RE.search(combined_names) or _EMISSIVE_NAME_RE.search(combined_names):
        if _GOLD_METAL_NAME_RE.search(combined_names):
            return "METALLIC_GLOW"
        return "VFX_ADDITIVE"
    if _GOLD_METAL_NAME_RE.search(combined_names) or any(b.inspection.is_matcap_sphere for b in tex_bindings):
        return "METALLIC_GLOW"

    # 3. 4-char TamEngine Shader ID family patterns
    if len(sid) >= 3:
        mid = sid[1:3]
        prefix2 = sid[:2]
        if mid in ("HG", "HA", "HR") or prefix2 in ("HR", "HA"):
            return "HAIR"
        if mid in ("EI", "ED", "ES", "EH", "EY") or prefix2 == "EY":
            return "EYE"
        if mid in ("SK", "FC") or prefix2 == "SK":
            return "SKIN"
        if prefix2 in ("EF", "VF", "FX", "PT", "AD", "EM", "GL", "AU") or mid in ("EF", "FX", "AD", "EM", "GL", "AU", "PT"):
            return "VFX_ADDITIVE"
        if mid in ("GH", "SP", "TR") or prefix2 in ("GH", "TR"):
            return "GHOST_SPIRIT"
        if mid in ("MT", "GD", "MR"):
            return "METALLIC_GLOW"
        if mid in ("SS", "SD", "TO", "TN", "CH", "CS", "BD") or prefix2 in ("TO", "TN", "CH", "CS"):
            return "TOON_CHAR"
        if mid in ("BG", "MP", "ST", "LM") or prefix2 in ("BG", "MP", "ST", "LM"):
            return "STAGE"

    # 4. Data-driven structural cues (unlit VFX meshes without normals, dark-bg energy maps, HDR params)
    albedo_binding = next((b for b in tex_bindings if b.role == "albedo" and b.image is not None), None)
    has_hdr_color = False
    for k in range(0, len(params) - 2, 4):
        chunk = params[k:k + 4]
        if len(chunk) >= 3 and _is_finite_nonneg(chunk[:3]):
            r, g, b = chunk[0], chunk[1], chunk[2]
            w = chunk[3] if len(chunk) == 4 else 1.0
            if not _is_uv_transform_vec4(r, g, b, w):
                if 1.08 < max(r, g, b) <= 30.0 and (min(r, g, b) > 0.0 or sum(x > 0.05 for x in (r, g, b)) >= 2 or (0.0 < w <= 1.0)):
                    has_hdr_color = True
                    break

    if albedo_binding and albedo_binding.inspection.is_dark_bg_energy:
        if not usage.has_normals or has_hdr_color or usage.color0_is_tinted or usage.color0_alpha_active:
            return "VFX_ADDITIVE"

    if not usage.has_normals and (has_hdr_color or usage.color0_is_tinted):
        return "VFX_ADDITIVE"

    if len(params) >= 28:
        return "TOON_CHAR"

    return "STANDARD"


# ---------------------------------------------------------------------------
# Texture Resolution & Role Assignment
# ---------------------------------------------------------------------------

_KNOWN_SHADER_SLOTS = {
    "BSS1": [
        (0, "albedo", False, 0),
        (2, "ilm_shadow", True, 0),
        (3, "specular_emissive", False, 0),
        (4, "normal", True, 0),
        (5, "spec_mask", True, 0),
        (6, "detail_mask", True, 0),
    ],
    "BHG1": [
        (2, "ilm_shadow", True, 0),
        (2, "specular_emissive", True, 1),  # Rimlight mask on UV1
    ],
    "BHA1": [
        (1, "ilm_shadow", True, 0),
        (2, "specular_emissive", True, 1),
    ],
    "BEI0": [
        (0, "albedo", False, 0),
        (2, "ilm_shadow", True, 0),
        (3, "specular_emissive", False, 0),
        (4, "normal", True, 0),
        (6, "detail_mask", True, 0),
    ],
    "BED0": [
        (0, "albedo", False, 0),
        (2, "ilm_shadow", True, 0),
        (3, "specular_emissive", False, 0),
        (4, "normal", True, 0),
        (6, "detail_mask", True, 0),
    ],
    "BES0": [
        (0, "albedo", False, 0),
    ],
    "BEH0": [
        (0, "albedo", False, 0),
        (2, "ilm_shadow", True, 0),
    ],
    "BGH0": [
        (0, "albedo", False, 0),
        (2, "ilm_shadow", True, 0),
        (3, "specular_emissive", False, 0),
        (4, "normal", True, 0),
        (6, "detail_mask", True, 0),
    ],
    "BGH1": [
        (0, "albedo", False, 0),
        (2, "ilm_shadow", True, 0),
        (3, "specular_emissive", False, 0),
        (4, "normal", True, 0),
        (6, "detail_mask", True, 0),
    ],
    "HSK1": [
        (0, "albedo", False, 0),
        (1, "ramp", False, 0),
        (2, "ilm_shadow", True, 0),
        (3, "specular_emissive", False, 0),
        (4, "normal", True, 0),
    ],
}


def resolve_single_texture_image(
    tmat_tex,
    tmd_name: str,
    lds_images: List[bpy.types.Image],
    dds_by_name: Dict[str, str],
    hashes_map: Dict[str, str],
) -> Tuple[Optional[bpy.types.Image], str]:
    """Accurately resolves a TMD2MatTexture / TMDMatTexture to a Blender Image,
    strictly matching tm_texture.index against LDS images without index shift bugs."""
    tm_tex = getattr(tmat_tex, "texture", None)
    tex_hash = int(getattr(tmat_tex, "textureHash", getattr(tm_tex, "hash", 0)) or 0)
    if tex_hash == 0 and tm_tex is not None:
        tex_hash = int(getattr(tm_tex, "hash", 0) or 0)

    is_external = bool(getattr(tmat_tex, "isExternal", False)) or getattr(tmat_tex, "textureIndex", 0) == -1
    lds_idx = int(getattr(tm_tex, "index", getattr(tmat_tex, "textureIndex", -1))) if tm_tex is not None else int(getattr(tmat_tex, "textureIndex", -1))

    resolved_name = hashes_map.get(str(tex_hash), "")
    if not resolved_name and tm_tex is not None and getattr(tm_tex, "name", ""):
        resolved_name = str(tm_tex.name)

    # 1. Non-external texture with valid LDS index in lds_images
    if not is_external and lds_images and 0 <= lds_idx < len(lds_images):
        img = lds_images[lds_idx]
        if img is not None:
            return img, resolved_name or img.name

    # 2. Match by resolved name or hash in dds_by_name (external or loose DDS workflow)
    candidates = []
    if resolved_name:
        candidates.extend([resolved_name, f"{resolved_name}.dds"])
    if tex_hash:
        candidates.extend([str(tex_hash), f"{tex_hash}.dds"])
    if not is_external and lds_idx >= 0:
        candidates.extend([
            f"{tmd_name}_{lds_idx}",
            f"{tmd_name}_{lds_idx}.dds",
        ])

    for cand in candidates:
        existing = bpy.data.images.get(cand)
        if existing is not None:
            return existing, resolved_name or existing.name

    for cand in candidates:
        stem = os.path.splitext(cand)[0]
        for key in (cand, stem, cand.lower(), stem.lower()):
            if key in dds_by_name:
                val = dds_by_name[key]
                if isinstance(val, bpy.types.Image):
                    return val, resolved_name or stem
                if isinstance(val, str) and os.path.exists(val):
                    img = bpy.data.images.load(val, check_existing=True)
                    try:
                        img.pack()
                    except Exception:
                        pass
                    img.use_fake_user = True
                    return img, resolved_name or stem

    # 3. Fallback if lds_images indexed by textureIndex (for TMD0/TMDV/CAT)
    tex_table_idx = int(getattr(tmat_tex, "textureIndex", -1))
    if not is_external and lds_images and 0 <= tex_table_idx < len(lds_images):
        img = lds_images[tex_table_idx]
        if img is not None:
            return img, resolved_name or img.name

    return None, resolved_name or (str(tex_hash) if tex_hash else "")


def build_texture_bindings(
    tm_mat,
    shader_id: str,
    tmd_name: str,
    lds_images: List[bpy.types.Image],
    dds_by_name: Dict[str, str],
    hashes_map: Dict[str, str],
    uv_count: int = 1,
) -> List[TextureSlotBinding]:
    raw_entries = []
    for list_idx, tmat_tex in enumerate(getattr(tm_mat, "textures", [])):
        img, res_name = resolve_single_texture_image(tmat_tex, tmd_name, lds_images, dds_by_name, hashes_map)
        insp = inspect_blender_image(img, res_name)
        tm_tex = getattr(tmat_tex, "texture", None)
        tex_hash = int(getattr(tmat_tex, "textureHash", getattr(tm_tex, "hash", 0)) or 0)
        if tex_hash == 0 and tm_tex is not None:
            tex_hash = int(getattr(tm_tex, "hash", 0) or 0)
        slot_val = int(getattr(tmat_tex, "slot", list_idx))
        raw_entries.append({
            "list_idx": list_idx,
            "slot": slot_val,
            "tex_hash": tex_hash,
            "resolved_name": res_name,
            "image": img,
            "inspection": insp,
            "unk1": int(getattr(tmat_tex, "unk1", 0)),
            "unk2": int(getattr(tmat_tex, "unk2", 0)),
            "unk3": int(getattr(tmat_tex, "unk3", 0)),
            "is_external": bool(getattr(tmat_tex, "isExternal", False)),
        })

    bindings: List[TextureSlotBinding] = []
    sid = (shader_id or "").strip()

    # Known shader explicit slot mapping (supports both dense list index AND sparse sampler slot!)
    if sid in _KNOWN_SHADER_SLOTS:
        used_list_indices = set()
        for target_idx, role, non_col, req_uv in _KNOWN_SHADER_SLOTS[sid]:
            matched = None
            # Prefer exact list_idx when dense, else match hardware sampler slot == target_idx
            if target_idx < len(raw_entries):
                matched = raw_entries[target_idx]
            else:
                for entry in raw_entries:
                    if entry["slot"] == target_idx:
                        matched = entry
                        break
            if matched is None:
                continue
            used_list_indices.add(matched["list_idx"])
            uv_i = min(req_uv, max(0, uv_count - 1))
            clamp = matched["inspection"].is_ramp_lut or role == "ramp"
            bindings.append(TextureSlotBinding(
                role=role,
                list_index=matched["list_idx"],
                sampler_slot=matched["slot"],
                tex_hash=matched["tex_hash"],
                resolved_name=matched["resolved_name"],
                image=matched["image"],
                inspection=matched["inspection"],
                uv_index=uv_i,
                non_color=non_col or matched["inspection"].is_normal_map,
                clamp_uv=clamp,
                unk1=matched["unk1"],
                unk2=matched["unk2"],
                unk3=matched["unk3"],
                is_external=matched["is_external"],
            ))
        # Preserve any remaining textures on the material as 'extra' so nothing is lost
        for entry in raw_entries:
            if entry["list_idx"] not in used_list_indices:
                bindings.append(TextureSlotBinding(
                    role="extra",
                    list_index=entry["list_idx"],
                    sampler_slot=entry["slot"],
                    tex_hash=entry["tex_hash"],
                    resolved_name=entry["resolved_name"],
                    image=entry["image"],
                    inspection=entry["inspection"],
                    uv_index=0,
                    non_color=entry["inspection"].is_normal_map,
                    clamp_uv=entry["inspection"].is_ramp_lut,
                    unk1=entry["unk1"],
                    unk2=entry["unk2"],
                    unk3=entry["unk3"],
                    is_external=entry["is_external"],
                ))
        return bindings

    # Unknown / Custom Shader Intelligent Role Assignment
    assigned_roles = set()
    for entry in raw_entries:
        insp: TextureInspection = entry["inspection"]
        slot = entry["slot"]
        idx = entry["list_idx"]
        unk3 = entry["unk3"]
        role = "extra"
        non_col = False
        uv_i = unk3 if (unk3 in (1, 2) and uv_count > unk3) else 0

        if insp.is_normal_map and "normal" not in assigned_roles:
            role = "normal"
            non_col = True
        elif insp.is_ramp_lut and len(raw_entries) >= 2 and slot != 0 and "ramp" not in assigned_roles:
            role = "ramp"
            non_col = False
        elif insp.is_ramp_lut and "albedo" in assigned_roles and "ramp" not in assigned_roles:
            role = "ramp"
            non_col = False
        elif insp.is_matcap_sphere and "specular_emissive" not in assigned_roles:
            role = "specular_emissive"
            non_col = False
        elif "albedo" not in assigned_roles and slot == 0:
            role = "albedo"
            non_col = False
        elif "albedo" not in assigned_roles and idx == 0:
            role = "albedo"
            non_col = False
        elif slot == 4 and "normal" not in assigned_roles and len(raw_entries) >= 4:
            role = "normal"
            non_col = True
        elif slot == 2 and "ilm_shadow" not in assigned_roles and len(raw_entries) >= 3:
            role = "ilm_shadow"
            non_col = True
        elif slot == 3 and "specular_emissive" not in assigned_roles:
            role = "specular_emissive"
            non_col = False
        elif "albedo" not in assigned_roles and not insp.is_normal_map:
            role = "albedo"
            non_col = False
        elif "ramp" not in assigned_roles and insp.is_ramp_lut:
            role = "ramp"
            non_col = False
        elif "specular_emissive" not in assigned_roles and not insp.is_grayscale:
            role = "specular_emissive"
            non_col = False
        elif "ilm_shadow" not in assigned_roles and insp.is_grayscale:
            role = "ilm_shadow"
            non_col = True
            if uv_count >= 2 and idx >= 1:
                uv_i = 0
        elif "detail_mask" not in assigned_roles:
            role = "detail_mask"
            non_col = True
            if uv_count >= 2:
                uv_i = 1

        assigned_roles.add(role)
        bindings.append(TextureSlotBinding(
            role=role,
            list_index=idx,
            sampler_slot=slot,
            tex_hash=entry["tex_hash"],
            resolved_name=entry["resolved_name"],
            image=entry["image"],
            inspection=insp,
            uv_index=min(uv_i, max(0, uv_count - 1)),
            non_color=non_col,
            clamp_uv=(role == "ramp" or insp.is_ramp_lut),
            unk1=entry["unk1"],
            unk2=entry["unk2"],
            unk3=entry["unk3"],
            is_external=entry["is_external"],
        ))

    return bindings


# ---------------------------------------------------------------------------
# Full Material Analysis Entry Point
# ---------------------------------------------------------------------------

def analyze_tmd_material(
    tm_mat,
    tmd_obj,
    lds_images: List[bpy.types.Image],
    dds_by_name: Dict[str, str],
    hashes_map: Dict[str, str],
    mat_name: str = "",
    fallback_options: Optional[dict] = None,
) -> MaterialAnalysis:
    if fallback_options is None:
        fallback_options = {}

    sid = (getattr(tm_mat, "shaderID", "") or "").strip()
    mat_hash = int(getattr(tm_mat, "hash", 0) or 0)
    resolved_hash_name = hashes_map.get(str(mat_hash), "")
    params = [float(x) for x in getattr(tm_mat, "shaderParams", [])]

    model_names = []
    for m in getattr(tmd_obj, "models", []):
        if tm_mat in getattr(m, "materials", []):
            mname = getattr(m, "name", "")
            if mname:
                model_names.append(mname)

    usage = inspect_submesh_usage(tmd_obj, tm_mat)
    tmd_name = getattr(tmd_obj, "name", "tmd2")
    bindings = build_texture_bindings(
        tm_mat, sid, tmd_name, lds_images, dds_by_name, hashes_map, uv_count=usage.uv_count
    )

    analysis = MaterialAnalysis(
        shader_id=sid,
        is_known_shader=(sid in KNOWN_ALL_SHADERS),
        material_name=mat_name or resolved_hash_name or f"{tmd_name}_mat",
        resolved_hash_name=resolved_hash_name,
        model_names=model_names,
        bindings=bindings,
        submesh_usage=usage,
    )

    # Check persistent custom shader rules first
    custom_rules = load_custom_shader_rules()
    custom_rule = custom_rules.get(sid, {})

    family = _classify_shader_family(sid, analysis.material_name, model_names, bindings, usage, params)
    if custom_rule.get("profile") and custom_rule["profile"] != "AUTO":
        family = custom_rule["profile"]
    elif fallback_options.get("fallback_profile") and fallback_options["fallback_profile"] != "AUTO" and not analysis.is_known_shader:
        family = fallback_options["fallback_profile"]
    analysis.shader_family = family

    albedo_bind = next((b for b in bindings if b.role == "albedo" and b.image is not None), None)
    ramp_bind = next((b for b in bindings if b.role == "ramp" and b.image is not None), None)
    spec_bind = next((b for b in bindings if b.role == "specular_emissive" and b.image is not None), None)

    has_any_albedo_tex = albedo_bind is not None
    albedo_is_grayscale = bool(albedo_bind and albedo_bind.inspection.is_grayscale)
    albedo_is_dark_energy = bool(albedo_bind and albedo_bind.inspection.is_dark_bg_energy)

    # --- Extract Shader Parameters Based on Known Layout or Smart Register Analysis ---
    def _get_rgb(start: int) -> Optional[Tuple[float, float, float]]:
        if 0 <= start and start + 3 <= len(params):
            vals = params[start:start + 3]
            if _is_finite_nonneg(vals) and max(vals) <= 50.0:
                return (float(vals[0]), float(vals[1]), float(vals[2]))
        return None

    if sid == "BSS1":
        if len(params) >= 4:
            analysis.shadow_intensity = max(0.0, min(2.0, params[0]))
            sc = _get_rgb(1)
            if sc:
                analysis.shadow_color = (*sc, 1.0)
        if len(params) >= 12:
            analysis.color_burn_opacity = max(0.0, min(1.0, params[10]))
            analysis.color_burn_intensity = params[11]
        if len(params) >= 25 and 0.0 < params[24] < 1.0:
            analysis.alpha_clip_threshold = params[24]
            analysis.alpha_clip_param_idx = 24
        if len(params) >= 32:
            analysis.specular_size = max(0.01, params[28])
            spc = _get_rgb(29)
            if spc:
                analysis.specular_color = (*spc, 1.0)
                analysis.specular_color_param_idx = 29
        analysis.use_vertex_color2_fade = usage.has_color1 and usage.color1_rgb_active

    elif sid in ("BHG1", "BHA1") or (family == "HAIR" and len(params) >= 15):
        if len(params) >= 4 and sid == "BHA1":
            analysis.shadow_intensity = max(0.0, min(2.0, params[0]))
            sc = _get_rgb(1)
            if sc:
                analysis.shadow_color = (*sc, 1.0)
        c_base = _get_rgb(12)
        c_shine = _get_rgb(9)
        if c_base and max(c_base) > 0.005:
            norm_c, peak = _normalize_hdr_color(*c_base)
            analysis.base_color = norm_c
            analysis.base_color_param_idx = 12
            if peak > 1.05:
                analysis.emission_color = norm_c
                analysis.emission_strength = peak - 1.0
        if c_shine and max(c_shine) > 0.005:
            norm_s, _ = _normalize_hdr_color(*c_shine)
            analysis.secondary_color = norm_s
            analysis.specular_color = norm_s
            analysis.secondary_color_param_idx = 9
        if sid == "BHA1":
            analysis.use_vertex_color2_fade = usage.has_color1 and usage.color1_rgb_active

    else:
        # Smart Register Analyzer for Unknown / Captain Tsubasa / Effect / Stage / Toon Shaders
        # 1. Check if 32+ param TamEngine character register layout matches
        if len(params) >= 32 and family in ("TOON_CHAR", "SKIN"):
            if _is_finite_nonneg(params[0:4]) and params[0] <= 2.0:
                analysis.shadow_intensity = max(0.0, min(1.5, params[0]))
                sc = _get_rgb(1)
                if sc and max(sc) <= 1.5:
                    analysis.shadow_color = (*sc, 1.0)
            if 0.0 < params[24] < 1.0:
                analysis.alpha_clip_threshold = params[24]
                analysis.alpha_clip_param_idx = 24
            spc = _get_rgb(29)
            if spc and max(spc) <= 4.0:
                norm_sp, _ = _normalize_hdr_color(*spc)
                analysis.specular_color = norm_sp
                analysis.specular_size = max(0.05, min(4.0, params[28]))
                analysis.specular_color_param_idx = 29

        # 2. Scan all 4-float constant buffer registers (c0, c1, c2, ...)
        color_candidates = []  # (reg_start, r, g, b, w, is_hdr, chroma, peak)
        for start in range(0, len(params), 4):
            chunk = params[start:start + 4]
            if len(chunk) < 3:
                continue
            r, g, b = chunk[0], chunk[1], chunk[2]
            w = chunk[3] if len(chunk) >= 4 else 1.0

            if not _is_finite_nonneg((r, g, b)):
                continue
            if _is_uv_transform_vec4(r, g, b, w):
                if analysis.uv_scale == (1.0, 1.0) and analysis.uv_offset == (0.0, 0.0):
                    analysis.uv_scale = (r, g)
                    analysis.uv_offset = (b, w)
                continue

            peak = max(r, g, b)
            if peak < 0.015 or peak > 32.0:
                continue

            # Exclude single-scalar registers like (50.0, 0.0, 0.0, 0.0) where peak > 1.5 and two channels are 0 unless w looks like alpha in (0, 1]
            zero_channels = sum(1 for ch in (r, g, b) if ch < 1e-4)
            if peak > 1.5 and zero_channels >= 2 and not (0.05 <= w <= 1.0 and peak <= 12.0):
                continue

            # Skip shadow [0..3] & specular [28..31] if already consumed on 32-param toon char
            if len(params) >= 32 and family in ("TOON_CHAR", "SKIN") and start in (0, 28):
                continue

            chroma = max(r, g, b) - min(r, g, b)
            is_hdr = peak > 1.05
            color_candidates.append((start, r, g, b, w, is_hdr, chroma, peak))

        # Also check [scalar, R, G, B] pattern at start=0 when params has 4..16 floats and c0[0] is intensity & c0[1:4] is RGB
        if not color_candidates and len(params) >= 4:
            s0 = params[0]
            rgb1 = _get_rgb(1)
            if rgb1 and 0.0 <= s0 <= 10.0 and 0.02 <= max(rgb1) <= 16.0:
                peak = max(rgb1)
                chroma = peak - min(rgb1)
                color_candidates.append((1, rgb1[0], rgb1[1], rgb1[2], 1.0, peak > 1.05, chroma, peak))

        # Prioritize HDR emissive vectors & chromatic color vectors
        hdr_candidates = [c for c in color_candidates if c[5]]
        chromatic_candidates = [c for c in color_candidates if c[6] > 0.04]

        primary_cand = None
        if hdr_candidates:
            # Prefer chromatic HDR first, else brightest HDR
            hdr_chrom = [c for c in hdr_candidates if c[6] > 0.04]
            primary_cand = hdr_chrom[0] if hdr_chrom else hdr_candidates[0]
        elif chromatic_candidates:
            # Prefer register 12 (TamEngine base color convention) or register 0 / first chromatic register
            reg12 = next((c for c in chromatic_candidates if c[0] == 12), None)
            reg0 = next((c for c in chromatic_candidates if c[0] == 0), None)
            primary_cand = reg12 or reg0 or chromatic_candidates[0]
        elif color_candidates and (not has_any_albedo_tex or albedo_is_grayscale):
            primary_cand = color_candidates[0]

        if primary_cand is not None:
            p_idx, pr, pg, pb, pw, p_hdr, p_chroma, p_peak = primary_cand
            norm_col, hdr_mag = _normalize_hdr_color(pr, pg, pb)

            # Apply base tint when:
            # - Texture is grayscale/monochrome or missing (needs colorization!)
            # - Shader family is VFX / Ghost / Metallic / Hair
            # - Or register is at c0 / c3(12) with explicit alpha w in (0, 1] and chromatic color
            should_tint_base = (
                not has_any_albedo_tex
                or albedo_is_grayscale
                or family in ("VFX_ADDITIVE", "GHOST_SPIRIT", "METALLIC_GLOW", "HAIR")
                or (p_idx in (0, 12) and 0.02 <= pw <= 1.0 and p_chroma > 0.08)
            )
            if should_tint_base:
                analysis.base_color = norm_col
                analysis.base_color_param_idx = p_idx
                analysis.diagnostic_notes.append(
                    f"Extracted base color from shaderParams[{p_idx}:{p_idx+3}] = ({pr:.3f}, {pg:.3f}, {pb:.3f})"
                )

            # Alpha / Opacity in 4th component w when strictly in (0.02, 0.995)
            if 0.02 <= pw < 0.995:
                analysis.opacity = float(pw)
                analysis.opacity_param_idx = p_idx + 3
                analysis.diagnostic_notes.append(
                    f"Extracted material opacity from shaderParams[{p_idx+3}] = {pw:.3f}"
                )

            # HDR Emission from peak > 1.05
            if p_hdr:
                analysis.emission_color = norm_col
                analysis.emission_strength = min(25.0, hdr_mag)
                analysis.emission_color_param_idx = p_idx
                analysis.diagnostic_notes.append(
                    f"Detected HDR emission color at shaderParams[{p_idx}] (strength {hdr_mag:.2f})"
                )

            # Secondary color candidate (for rim/gradient/shadow)
            remaining = [c for c in color_candidates if c[0] != p_idx]
            if remaining:
                sec = next((c for c in remaining if c[6] > 0.04), remaining[0])
                norm_sec, sec_mag = _normalize_hdr_color(sec[1], sec[2], sec[3])
                analysis.secondary_color = norm_sec
                analysis.secondary_color_param_idx = sec[0]
                if sec[5] and analysis.emission_strength == 0.0:
                    analysis.emission_color = norm_sec
                    analysis.emission_strength = min(25.0, sec_mag)
                    analysis.emission_color_param_idx = sec[0]

    # --- Apply explicit custom rule parameter index overrides if defined ---
    def _apply_override_rgb(key: str) -> Optional[Tuple[Tuple[float, float, float, float], float, int]]:
        idx = int(custom_rule.get(key, -1))
        if 0 <= idx and idx + 3 <= len(params):
            r, g, b = params[idx], params[idx + 1], params[idx + 2]
            c, mag = _normalize_hdr_color(max(0.0, r), max(0.0, g), max(0.0, b))
            return c, mag, idx
        return None

    ov_base = _apply_override_rgb("base_color_idx")
    if ov_base:
        analysis.base_color, _, analysis.base_color_param_idx = ov_base
    ov_sec = _apply_override_rgb("secondary_color_idx")
    if ov_sec:
        analysis.secondary_color, _, analysis.secondary_color_param_idx = ov_sec
    ov_spec = _apply_override_rgb("specular_color_idx")
    if ov_spec:
        analysis.specular_color, _, analysis.specular_color_param_idx = ov_spec
    ov_emi = _apply_override_rgb("emission_color_idx")
    if ov_emi:
        analysis.emission_color, mag, analysis.emission_color_param_idx = ov_emi
        if analysis.emission_strength <= 0.0:
            analysis.emission_strength = max(1.0, mag)
    emi_s_idx = int(custom_rule.get("emission_strength_idx", -1))
    if 0 <= emi_s_idx < len(params):
        analysis.emission_strength = max(0.0, float(params[emi_s_idx]))
    op_idx = int(custom_rule.get("opacity_idx", -1))
    if 0 <= op_idx < len(params) and 0.0 <= params[op_idx] <= 1.0:
        analysis.opacity = float(params[op_idx])
        analysis.opacity_param_idx = op_idx
    ac_idx = int(custom_rule.get("alpha_clip_idx", -1))
    if 0 <= ac_idx < len(params) and 0.0 <= params[ac_idx] <= 1.0:
        analysis.alpha_clip_threshold = float(params[ac_idx])
        analysis.alpha_clip_param_idx = ac_idx

    # --- Vertex Color & Vertex Alpha Integration ---
    vcol_opt = custom_rule.get("use_vertex_color") or fallback_options.get("use_vertex_colors", "AUTO")
    if vcol_opt == "OFF":
        analysis.use_vertex_color_rgb = False
        analysis.use_vertex_color_alpha = False
    elif vcol_opt in ("TINT", "TINT_ALPHA"):
        analysis.use_vertex_color_rgb = usage.has_color0 and usage.color0_rgb_active
        analysis.use_vertex_color_alpha = (vcol_opt == "TINT_ALPHA") and usage.has_color0 and not usage.color0_alpha_zeroed
    else:  # AUTO
        # In known character shaders (BED0/BEI0/BGH0/BGH1), Color0 is shadow tint, handled in toon shading;
        # For VFX/Ghost/Metallic/Stage/Unknown shaders, active non-zero Color0 RGB tints the mesh
        if not analysis.is_known_shader:
            if usage.has_color0 and usage.color0_rgb_active and not usage.color0_is_white:
                # Guard: in character toon shaders with full-color skin/costume textures, dark neutral vertex color can be AO/outline mask;
                # apply full RGB multiply when vertex color is chromatic OR texture is grayscale/missing OR family is VFX/Ghost/Metallic/Stage
                v_r, v_g, v_b = usage.color0_mean_rgb
                v_chroma = max(v_r, v_g, v_b) - min(v_r, v_g, v_b)
                if (
                    v_chroma > 0.035
                    or albedo_is_grayscale
                    or not has_any_albedo_tex
                    or family in ("VFX_ADDITIVE", "GHOST_SPIRIT", "METALLIC_GLOW", "STAGE")
                ):
                    analysis.use_vertex_color_rgb = True
                    analysis.diagnostic_notes.append(
                        f"Enabled Vertex Color ('Color') RGB tint: mean=({v_r:.2f}, {v_g:.2f}, {v_b:.2f})"
                    )
        if usage.has_color0 and usage.color0_alpha_active:
            if not analysis.is_known_shader or family in ("VFX_ADDITIVE", "GHOST_SPIRIT"):
                analysis.use_vertex_color_alpha = True
                analysis.diagnostic_notes.append(
                    f"Enabled Vertex Color ('Color') Alpha fade: min={usage.color0_min_alpha:.2f}, mean={usage.color0_mean_alpha:.2f}"
                )

    # --- Grayscale Colorization & Ramp LUT ---
    gs_opt = custom_rule.get("grayscale_mode") or fallback_options.get("grayscale_mode", "AUTO")
    if ramp_bind is not None and gs_opt in ("AUTO", "RAMP"):
        if albedo_is_grayscale or sid == "HSK1" or gs_opt == "RAMP":
            analysis.use_ramp_lut = True
            analysis.diagnostic_notes.append(f"Enabled Ramp/Gradient LUT texture: {ramp_bind.resolved_name}")

    if albedo_is_grayscale and gs_opt != "RAW":
        analysis.colorize_grayscale = True
        analysis.diagnostic_notes.append("Detected grayscale primary texture; enabled material/vertex colorization")

    # --- Family-Specific Material Appearance Refinement (Metallic, Rim, Emission, Transparency) ---
    if family == "METALLIC_GLOW":
        analysis.metallic = 0.82
        analysis.roughness = 0.22
        if analysis.emission_strength == 0.0 and (albedo_is_dark_energy or _EMISSIVE_NAME_RE.search(analysis.material_name)):
            analysis.emission_color = analysis.base_color
            analysis.emission_strength = 1.6
        analysis.rim_color = analysis.secondary_color if analysis.secondary_color_param_idx >= 0 else analysis.base_color
        analysis.rim_strength = max(analysis.rim_strength, 0.65)
        analysis.rim_power = 2.6

    elif family == "VFX_ADDITIVE":
        analysis.roughness = 0.5
        if analysis.emission_strength <= 0.0:
            # Derive emission color from extracted base color, vertex color, or texture highlights
            if analysis.base_color_param_idx >= 0:
                analysis.emission_color = analysis.base_color
            elif analysis.use_vertex_color_rgb:
                analysis.emission_color = (1.0, 1.0, 1.0, 1.0)
            else:
                analysis.emission_color = (1.0, 1.0, 1.0, 1.0)
            analysis.emission_strength = float(fallback_options.get("emission_multiplier", 1.8))
        else:
            analysis.emission_strength *= float(fallback_options.get("emission_multiplier", 1.0))
        analysis.rim_color = analysis.secondary_color if analysis.secondary_color_param_idx >= 0 else analysis.base_color
        analysis.rim_strength = max(analysis.rim_strength, 0.45)

    elif family == "GHOST_SPIRIT":
        analysis.fresnel_translucency = True
        if analysis.opacity >= 0.99:
            analysis.opacity = 0.72
        if analysis.emission_strength <= 0.0:
            analysis.emission_color = analysis.base_color
            analysis.emission_strength = float(fallback_options.get("emission_multiplier", 1.4))
        analysis.rim_color = analysis.secondary_color if analysis.secondary_color_param_idx >= 0 else analysis.base_color
        analysis.rim_strength = max(analysis.rim_strength, 1.1)
        analysis.rim_power = 2.2

    elif spec_bind is not None and spec_bind.inspection.is_matcap_sphere:
        analysis.metallic = max(analysis.metallic, 0.7)
        analysis.roughness = min(analysis.roughness, 0.28)

    # --- Determine Final Transparency / Blend Mode ---
    forced_blend = custom_rule.get("blend_mode") or fallback_options.get("transparency_mode", "AUTO")
    if forced_blend and forced_blend != "AUTO":
        analysis.blend_mode = forced_blend
        if forced_blend == "ADDITIVE":
            analysis.luminance_to_alpha = True
        elif forced_blend == "GHOST":
            analysis.fresnel_translucency = True
    else:
        tex_has_smooth_a = bool(albedo_bind and albedo_bind.inspection.is_smooth_alpha)
        tex_has_cutout_a = bool(albedo_bind and albedo_bind.inspection.is_cutout_alpha)

        if sid == "BES0" or (family == "EYE" and sid.endswith("S0")):
            analysis.blend_mode = "ALPHA_BLEND"
        elif family == "GHOST_SPIRIT":
            analysis.blend_mode = "GHOST"
            if albedo_is_dark_energy and not (tex_has_smooth_a or tex_has_cutout_a):
                analysis.luminance_to_alpha = True
        elif family == "VFX_ADDITIVE":
            if tex_has_smooth_a:
                analysis.blend_mode = "ALPHA_BLEND"
            elif albedo_is_dark_energy or not has_any_albedo_tex:
                analysis.blend_mode = "ADDITIVE"
                analysis.luminance_to_alpha = True
            elif analysis.opacity < 0.98 or analysis.use_vertex_color_alpha:
                analysis.blend_mode = "ALPHA_BLEND"
            else:
                analysis.blend_mode = "ADDITIVE"
                analysis.luminance_to_alpha = True
        elif tex_has_smooth_a or analysis.opacity < 0.98 or analysis.use_vertex_color_alpha or _ALPHA_NAME_RE.search(analysis.material_name):
            analysis.blend_mode = "ALPHA_BLEND"
        elif tex_has_cutout_a or analysis.alpha_clip_param_idx >= 0:
            analysis.blend_mode = "ALPHA_CLIP"
        elif gs_opt == "LUMINANCE_ALPHA" and albedo_is_grayscale:
            analysis.blend_mode = "ADDITIVE"
            analysis.luminance_to_alpha = True
        else:
            analysis.blend_mode = "OPAQUE"

    # Backface culling: keep False for see-through / double-sided / hair / VFX / Ghost so back parts show through!
    cull_opt = fallback_options.get("backface_culling", "AUTO")
    if cull_opt == "ON":
        analysis.backface_culling = True
    elif cull_opt == "OFF":
        analysis.backface_culling = False
    else:
        if analysis.blend_mode in ("ALPHA_BLEND", "ADDITIVE", "GHOST", "ALPHA_CLIP") or usage.has_double_sided_faces or family in ("HAIR", "VFX_ADDITIVE", "GHOST_SPIRIT"):
            analysis.backface_culling = False
        else:
            analysis.backface_culling = False

    return analysis
