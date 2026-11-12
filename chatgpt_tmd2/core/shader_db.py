"""Shader knowledge base for TMD2 materials.

Everything here is derived from the material definitions that ship with the
Tamsoft material library (``materials.blend``) plus the layout of the
``shaderParams`` array as it is stored in .tmd2/.tmd files.

Terminology
-----------
* **shader ID**      - the 4 character ASCII tag stored in ``TMD2Material.shaderID``
* **param index**    - the index inside ``TMD2Material.shaderParams`` (a float array)
* **texture slot**   - the index inside ``TMD2Material.textures``
* **library material** - the Blender material of the same name inside materials.blend

The original add-on only mapped five shader IDs (BSS1, BHG1, BHA1, BSH/HSK1 and
friends) and did not scale colours, alpha or emission from the parameter block.
This module adds a description for every shader that is present in the shipped
material library, plus a heuristic layer that interprets the *unknown* ones
without inventing data.

Parameter maps
--------------
Colours in `shaderParams` are stored as **separate float components** (e.g. three
consecutive floats forming RGB) and are, in the material library, declared as raw
values (not colour managed).  The generic convention used by the game is:

    index 0..N  - shader dependent, see `SHADERS`
    alpha       - usually index 24 for the "BSS1" style body shader

Rather than hard-coding one layout for every shader we describe what is known and
fall back to detection (``is_spread_color``) for the rest.
"""

from collections import OrderedDict


class ShaderTextureSlot:
    """Description of one texture slot used by a shader."""

    __slots__ = ("name", "tex_index", "node", "non_color", "uv", "role")

    def __init__(self, name, tex_index, node=None, non_color=False, uv=None, role=""):
        self.name = name
        self.tex_index = tex_index          # index inside TMD2Material.textures
        self.node = node or name            # image node name inside the library material
        self.non_color = non_color          # data texture (normal / mask / roughness)
        self.uv = uv                        # uv layer name to drive the node with
        self.role = role                    # human readable purpose

    def __repr__(self):  # pragma: no cover - debug helper
        return f"<ShaderTextureSlot {self.name} idx={self.tex_index} non_color={self.non_color}>"


class ShaderColorParam:
    """A colour that lives inside ``shaderParams``."""

    __slots__ = ("name", "indices", "scale", "fallback", "is_strength", "emissive")

    def __init__(self, name, indices=None, scale="srgb", fallback=None,
                 is_strength=False, emissive=False):
        self.name = name
        self.indices = indices or ()
        self.scale = scale        # 'srgb' (0..1 -> linear), 'linear', 'hdr'
        self.fallback = fallback  # used when the param block is too short
        self.is_strength = is_strength
        self.emissive = emissive  # HDR colours that represent a glow


class ShaderScalarParam:
    __slots__ = ("name", "index", "default", "min_value", "max_value", "meaning")

    def __init__(self, name, index, default=0.0, min_value=None, max_value=None, meaning=""):
        self.name = name
        self.index = index
        self.default = default
        self.min_value = min_value
        self.max_value = max_value
        self.meaning = meaning


class ShaderDescription:
    """Everything the importer knows about one family of game shaders."""

    def __init__(self, shader_id, library=None, description="", family="opaque",
                 textures=None, colors=None, scalars=None, blend="OPAQUE",
                 vertex_color=False, alpha_param=None, alpha_texture=None,
                 normal_texture=None, emission=None):
        self.shader_id = shader_id
        self.library = library or shader_id     # material name inside materials.blend
        self.description = description
        self.family = family
        self.textures = textures or []
        self.colors = colors or []
        self.scalars = scalars or []
        self.blend = blend
        self.vertex_color = vertex_color
        self.alpha_param = alpha_param          # index in shaderParams, or None
        self.alpha_texture = alpha_texture      # index in material.textures, or None
        self.normal_texture = normal_texture    # index in material.textures, or None
        self.emission = emission                # ShaderColorParam describing emission

    # -- convenience -----------------------------------------------------
    @property
    def color_indices(self):
        """All parameter indices that behave like colours."""
        out = []
        for c in self.colors:
            out.extend(c.indices)
        if self.emission:
            out.extend(self.emission.indices)
        return out

    def __repr__(self):  # pragma: no cover
        return f"<ShaderDescription {self.shader_id} lib={self.library}>"


# ---------------------------------------------------------------------------
# Helpers used while building the table
# ---------------------------------------------------------------------------
def _tex(name, idx, non_color=False, uv=None, role=""):
    return ShaderTextureSlot(name, idx, node=name, non_color=non_color, uv=uv, role=role)


#: Texture layouts for every shader present in the shipped material library.
#: ``tex_index`` refers to the position inside ``TMD2Material.textures``.
_TEXTURE_LAYOUTS = {
    # Body / cloth shader.  Used for the vast majority of Captain Tsubasa models.
    "BSS1": [
        _tex("Texture 1", 0, role="base color + alpha"),
        _tex("Texture 3", 2, non_color=True, role="shadow ramp / mask"),
        _tex("Texture 4", 3, role="sub surface / detail"),
        _tex("Texture 5", 4, non_color=True, role="normal map"),
        _tex("Texture 6", 5, non_color=True, role="extra mask"),
        _tex("Texture 7", 6, non_color=True, role="scratches / blood mask"),
    ],
    "BEI0": [
        _tex("Texture 1", 0, role="base color + alpha"),
        _tex("Texture 3", 2, non_color=True, role="shadow ramp"),
        _tex("Texture 4", 3, role="detail"),
        _tex("Texture 5", 4, non_color=True, role="normal map"),
        _tex("Texture 7", 6, non_color=True, role="scratches mask"),
    ],
    "BED0": [
        _tex("Texture 1", 0, role="base color + alpha"),
        _tex("Texture 3", 2, non_color=True, role="shadow ramp"),
        _tex("Texture 4", 3, role="detail"),
        _tex("Texture 5", 4, non_color=True, role="normal map"),
        _tex("Texture 7", 6, non_color=True, role="scratches mask"),
    ],
    "BGH0": [
        _tex("Texture 1", 0, role="base color + alpha"),
        _tex("Texture 3", 2, non_color=True, role="shadow ramp"),
        _tex("Texture 4", 3, role="detail"),
        _tex("Texture 5", 4, non_color=True, role="normal map"),
        _tex("Texture 7", 6, non_color=True, role="scratches mask"),
    ],
    "BGH1": [
        _tex("Texture 1", 0, role="base color + alpha"),
        _tex("Texture 3", 2, non_color=True, role="shadow ramp"),
        _tex("Texture 4", 3, role="detail"),
        _tex("Texture 5", 4, non_color=True, role="normal map"),
        _tex("Texture 7", 6, non_color=True, role="scratches mask"),
    ],
    "BEH0": [
        _tex("Texture 1", 0, role="base color + alpha"),
        _tex("Texture 3", 2, non_color=True, role="shadow ramp"),
    ],
    "BES0": [
        _tex("Texture 1", 0, role="unlit / cutout"),
    ],
    # Hair shaders
    "BHA1": [
        _tex("Texture 2", 1, non_color=True, role="shadow mask"),
        _tex("Texture 3", 2, non_color=True, uv="UVMap1", role="hair shine mask"),
    ],
    "BHG1": [
        _tex("Texture 3", 2, non_color=True, role="shadow mask"),
        _tex("Texture 3_UV1", 2, non_color=True, uv="UVMap1", role="rimlight mask"),
    ],
    # Skin shader (Bleach / Tsubasa faces)
    "HSK1": [
        _tex("Texture 1", 0, role="diffuse"),
        _tex("Texture 2", 1, role="ramp texture"),
        _tex("Texture 3", 2, non_color=True, role="shadows"),
        _tex("Texture 4", 3, role="specular"),
        _tex("Texture 5", 4, non_color=True, role="normals"),
    ],
    "Default": [
        _tex("Texture 1", 0, role="base color"),
    ],
}


#: Colour parameters per shader.  Indices were verified against the material
#: library node inputs and cross-checked with the parameter block layout.
_COLOR_PARAMS = {
    "BSS1": [
        ShaderColorParam("Shadow Color", (1, 2, 3)),
        ShaderColorParam("Specular Color", (29, 30, 31)),
        ShaderColorParam("Blood Color", (16, 17, 18)),
    ],
    "BEI0": [
        ShaderColorParam("Shadow Color", (1, 2, 3)),
        ShaderColorParam("Specular Color", (29, 30, 31)),
    ],
    "BED0": [
        ShaderColorParam("Shadow Color", (1, 2, 3)),
        ShaderColorParam("Specular Color", (29, 30, 31)),
    ],
    "BGH0": [
        ShaderColorParam("Shadow Color", (1, 2, 3)),
        ShaderColorParam("Specular Color", (29, 30, 31)),
    ],
    "BGH1": [
        ShaderColorParam("Shadow Color", (1, 2, 3)),
        ShaderColorParam("Specular Color", (29, 30, 31)),
    ],
    "BHA1": [
        ShaderColorParam("Hair Base Color", (12, 13, 14)),
        ShaderColorParam("Hair Shine Color", (9, 10, 11)),
    ],
    "BHG1": [
        ShaderColorParam("Hair Color 1", (12, 13, 14)),
        ShaderColorParam("Hair Color 2", (9, 10, 11)),
        ShaderColorParam("Shadow Color", (1, 2, 3)),
    ],
    "HSK1": [
        ShaderColorParam("Specular", (9, 10, 11)),
    ],
}


#: Scalar parameters per shader.
_SCALAR_PARAMS = {
    "BSS1": [
        ShaderScalarParam("Shadow Intensity", 0, 0.15, 0.0, 1.0, "strength of the toon shadow band"),
        ShaderScalarParam("Color Burn Opacity", 10, 0.15, 0.0, 1.0),
        ShaderScalarParam("Color Burn Intensity", 11, 0.0, 0.0, 1.0),
        ShaderScalarParam("Alpha Clip Threshold", 24, 0.0, 0.0, 1.0),
        ShaderScalarParam("Specular Size", 28, 0.5, 0.0, 1.0),
    ],
    "BEI0": [
        ShaderScalarParam("Shadow Intensity", 0, 0.15, 0.0, 1.0),
        ShaderScalarParam("Specular Size", 28, 0.5, 0.0, 1.0),
    ],
    "BED0": [
        ShaderScalarParam("Shadow Intensity", 0, 0.15, 0.0, 1.0),
        ShaderScalarParam("Specular Size", 28, 0.5, 0.0, 1.0),
    ],
    "BGH0": [
        ShaderScalarParam("Shadow Intensity", 0, 0.15, 0.0, 1.0),
        ShaderScalarParam("Specular Size", 28, 0.5, 0.0, 1.0),
    ],
    "BGH1": [
        ShaderScalarParam("Shadow Intensity", 0, 0.15, 0.0, 1.0),
        ShaderScalarParam("Specular Size", 28, 0.5, 0.0, 1.0),
    ],
    "BHA1": [
        ShaderScalarParam("Shadow Intensity", 0, 0.0, 0.0, 1.0),
    ],
    "BHG1": [],
    "HSK1": [
        ShaderScalarParam("Shadows", 0, 50.0, 0.0, 100.0),
    ],
}


#: Alpha handling.  ``alpha_param`` is a shaderParams index holding the opacity.
_ALPHA = {
    "BSS1": (24, 0),        # (param index used as clip threshold, alpha texture slot)
    "BES0": (None, 0),
    "BEH0": (None, 0),
    "BED0": (None, 0),
}


_SHADER_DEFS = OrderedDict()


def _add(desc):
    _SHADER_DEFS[desc.shader_id] = desc


_add(ShaderDescription(
    "BSS1", library="BSS1", family="body",
    description="Standard toon body/cloth shader with shadow ramp, specular and "
                "vertex colour burn support.",
    textures=_TEXTURE_LAYOUTS["BSS1"], colors=_COLOR_PARAMS["BSS1"],
    scalars=_SCALAR_PARAMS["BSS1"], blend="HASHED", vertex_color=True,
    alpha_param=24, alpha_texture=0, normal_texture=4,
))
_add(ShaderDescription(
    "BEI0", library="BEI0", family="body",
    description="Body shader variant used for inner layers (BEI).",
    textures=_TEXTURE_LAYOUTS["BEI0"], colors=_COLOR_PARAMS["BEI0"],
    scalars=_SCALAR_PARAMS["BEI0"], blend="HASHED", vertex_color=True,
    alpha_texture=0, normal_texture=4,
))
_add(ShaderDescription(
    "BED0", library="BED0", family="body",
    description="Body shader variant (BED) with detail texture.",
    textures=_TEXTURE_LAYOUTS["BED0"], colors=_COLOR_PARAMS["BED0"],
    scalars=_SCALAR_PARAMS["BED0"], blend="HASHED", vertex_color=True,
    alpha_texture=0, normal_texture=4,
))
_add(ShaderDescription(
    "BGH0", library="BGH0", family="body",
    description="Body shader variant (BGH0).",
    textures=_TEXTURE_LAYOUTS["BGH0"], colors=_COLOR_PARAMS["BGH0"],
    scalars=_SCALAR_PARAMS["BGH0"], blend="HASHED", vertex_color=True,
    alpha_texture=0, normal_texture=4,
))
_add(ShaderDescription(
    "BGH1", library="BGH1", family="body",
    description="Body shader variant (BGH1), usually used on low-poly props.",
    textures=_TEXTURE_LAYOUTS["BGH1"], colors=_COLOR_PARAMS["BGH1"],
    scalars=_SCALAR_PARAMS["BGH1"], blend="HASHED", vertex_color=True,
    alpha_texture=0, normal_texture=4,
))
_add(ShaderDescription(
    "BEH0", library="BEH0", family="effect",
    description="Two texture effect shader with strong shadow ramping, used for "
                "glowing / aura style effects.",
    textures=_TEXTURE_LAYOUTS["BEH0"], colors=_COLOR_PARAMS.get("BEH0", []),
    blend="HASHED", vertex_color=False, alpha_texture=0,
))
_add(ShaderDescription(
    "BES0", library="BES0", family="unlit",
    description="Unlit additive/transparent shader driven only by the base texture.",
    textures=_TEXTURE_LAYOUTS["BES0"], blend="BLEND", alpha_texture=0,
))
_add(ShaderDescription(
    "BHA1", library="BHA1", family="hair",
    description="Anisotropic hair shader with base and shine colours.",
    textures=_TEXTURE_LAYOUTS["BHA1"], colors=_COLOR_PARAMS["BHA1"],
    scalars=_SCALAR_PARAMS["BHA1"], blend="HASHED", vertex_color=True,
))
_add(ShaderDescription(
    "BHG1", library="BHG1", family="hair",
    description="Hair glow / rimlight shader driven by colour parameters only.",
    textures=_TEXTURE_LAYOUTS["BHG1"], colors=_COLOR_PARAMS["BHG1"],
    scalars=_SCALAR_PARAMS["BHG1"], blend="HASHED", vertex_color=True,
))
_add(ShaderDescription(
    "HSK1", library="HSK1", family="skin",
    description="Skin/face toon shader with ramp, specular and normal texture.",
    textures=_TEXTURE_LAYOUTS["HSK1"], colors=_COLOR_PARAMS["HSK1"],
    scalars=_SCALAR_PARAMS["HSK1"], blend="BLEND", vertex_color=False,
    normal_texture=4,
))
_add(ShaderDescription(
    "Default", library="Default", family="fallback",
    description="Plain diffuse material, used when nothing else is known.",
    textures=_TEXTURE_LAYOUTS["Default"], blend="OPAQUE", alpha_texture=None,
))


# ---------------------------------------------------------------------------
# Public lookups
# ---------------------------------------------------------------------------
SHADERS = _SHADER_DEFS

#: Every material name that exists inside ``materials.blend``.
LIBRARY_MATERIALS = tuple(_SHADER_DEFS.keys()) + ("TMD Outline",)

#: Shader IDs that are simply aliases of another shader.
SHADER_ALIASES = {
    "BSS0": "BSS1",
    "BES1": "BES0",
    "HSK0": "HSK1",
    "BHA0": "BHA1",
    "BAH1": "BHA1",
    "BGH2": "BGH1",
}


def normalize_shader_id(shader_id):
    """Clean up trailing NULs / whitespace from a 4 byte shader tag."""
    if not shader_id:
        return ""
    sid = shader_id.replace("\x00", "").strip()
    return sid


def get_shader(shader_id):
    """Return the :class:`ShaderDescription` for a shader ID (or ``None``)."""
    sid = normalize_shader_id(shader_id)
    if sid in SHADERS:
        return SHADERS[sid]
    alias = SHADER_ALIASES.get(sid)
    if alias:
        return SHADERS.get(alias)
    return None


def is_known_shader(shader_id):
    return get_shader(shader_id) is not None


def known_shader_ids():
    return tuple(SHADERS.keys())


# ---------------------------------------------------------------------------
# Heuristic helpers for unknown shaders
# ---------------------------------------------------------------------------
def looks_like_color(values, start, count=3, min_components=2):
    """Heuristic: are ``values[start:start+count]`` a plausible colour?

    Game colours are stored as floats in 0..1.  Colours tend to be "spread" (not
    all equal) while most other parameters are scalars such as 0, 1 or 100.
    """
    if len(values) < start + min_components:
        return False
    chunk = list(values[start:start + count])
    if not chunk:
        return False
    for v in chunk:
        if v < -0.001 or v > 1.001:
            return False
    # alpha-only or all zero/one blocks are usually not a colour
    if all(abs(v - chunk[0]) < 1e-4 for v in chunk):
        return False
    return True


def _color_score(values, start):
    """Score how likely ``values[start:start+3]`` is an authored colour.

    Game colours are stored as ``r, g, b, 0`` (or ``r, g, b, 1``).  The trailing
    padding float is the strongest signal, the saturation is used to order the
    remaining candidates so that vivid blocks win over pale single channels.
    """
    if len(values) < start + 3:
        return 0.0
    rgb = [float(v) for v in values[start:start + 3]]
    if any(v < -0.001 or v > 1.001 for v in rgb):
        return 0.0
    spread = max(rgb) - min(rgb)
    if max(rgb) <= 0.01:
        return 0.0            # black is far too common to be a colour block
    if start + 3 >= len(values):
        return 0.0            # needs a padding component after the colour
    padding = float(values[start + 3])
    if abs(padding) < 1e-3:
        score = 1.0
    elif abs(padding - 1.0) < 1e-3:
        score = 0.5
    else:
        return 0.0            # not a float4 aligned colour block
    # A colour block always follows a scalar / the padding of the previous block,
    # so a candidate that starts in the middle of three floats is rejected.
    if start > 0:
        previous = float(values[start - 1])
        if abs(previous) > 1e-3 and abs(previous - 1.0) > 1e-3:
            return 0.0

    score += spread * 2.0
    if min(rgb) > 0.01:
        score += 0.5          # actual colours, not a single bright channel
    # Sliding windows starting inside a real colour block look like bright
    # single-channel colours (0.1, 0.0, 1.0 ...); prefer centred, saturated
    # blocks and give a small bonus to the first colour in the block.
    score -= start * 0.01
    return score


def guess_color_params(values, max_colors=6):
    """Find colour-like slices inside an unknown parameter block.

    Returns a list of ``(start_index, [r, g, b])`` tuples ordered by confidence,
    which is the best we can do without knowing the shader.
    """
    candidates = []
    n = len(values)
    for i in range(n - 2):
        score = _color_score(values, i)
        if score > 0:
            candidates.append((score, i, [float(v) for v in values[i:i + 3]]))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    chosen = []
    used = set()
    for score, start, rgb in candidates:
        if len(chosen) >= max_colors:
            break
        if any(abs(start - other) < 3 for other in used):
            continue
        used.add(start)
        chosen.append((start, rgb))
    return chosen


def best_color(values):
    """Return the most plausible colour in a parameter block (or ``None``)."""
    found = guess_color_params(values, max_colors=1)
    return found[0] if found else None


def guess_alpha_param(values, texture_count=0):
    """Guess which parameter holds alpha/clip information for unknown shaders."""
    n = len(values)
    for idx in (24, 23, 25, 7, 8):
        if idx < n and 0.0 < values[idx] < 1.0:
            return idx
    return None


def guess_blend_mode(texture_count=0, has_alpha_param=False, family=""):
    """Best-effort blend mode for unknown shaders."""
    if family in ("unlit",):
        return "BLEND"
    if has_alpha_param:
        return "HASHED"
    if texture_count <= 1:
        return "OPAQUE"
    return "HASHED"
