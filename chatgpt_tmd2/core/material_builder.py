"""Material reconstruction for TMD2 models.

Two paths exist:

``library`` path
    Used when the shader ID is known.  The material prototype of the same name is
    loaded from ``materials.blend`` (the game shader reconstruction that ships with
    the add-on), copied, then wired up with the real textures, the real colour /
    scalar parameters, vertex colours, alpha and emission.

``fallback`` path
    Used when the shader ID is unknown.  A clean Principled BSDF graph is generated
    and the *real* values from the file (texture slots, colour parameters, alpha)
    are applied.  A configurable fallback mode decides how much interpretation is
    allowed.

Loading the library
-------------------
The original ``BlenderTMD2`` add-on also loads ``BSS1``, ``BHA1`` ... from its own
copy of materials.blend.  If both add-ons are enabled, Blender would rename the
second copy (``BSS1.001``) and its node group, which makes the "shared" node group
inside the copied material point at whichever library was loaded last.  To stay
out of the way we load with a dedicated prefix and rename both the prototype
material and its node groups so they can never be confused with the originals.
"""

import os

import bpy

from . import shader_db


# ---------------------------------------------------------------------------
# Small colour helpers
# ---------------------------------------------------------------------------
def srgb_to_linear(c):
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c):
    if c <= 0.0031308:
        return c * 12.92
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def to_linear_rgb(values):
    return tuple(srgb_to_linear(_clamp(float(v))) for v in values)


def to_linear_rgba(values, alpha=1.0):
    rgb = to_linear_rgb(values[:3]) if len(values) >= 3 else (1.0, 1.0, 1.0)
    return (rgb[0], rgb[1], rgb[2], alpha)


def is_black(values, tol=1e-3):
    return all(abs(float(v)) <= tol for v in values)


def is_white(values, tol=1e-3):
    return all(abs(float(v) - 1.0) <= tol for v in values)


# ---------------------------------------------------------------------------
# Material library management
# ---------------------------------------------------------------------------
LIB_PREFIX = "CTMD2_LIB_"
_prototype_cache = {}


def _library_path():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "materials", "materials.blend")


def _ensure_prototype(shader_id):
    """Load and rename the library material for ``shader_id``.

    Returns the prototype material (never used directly, always copied) or
    ``None`` when the material is missing from materials.blend.
    """
    desc = shader_db.get_shader(shader_id)
    if not desc:
        return None
    lib_name = desc.library

    cached = _prototype_cache.get(lib_name)
    if cached and cached.name in bpy.data.materials:
        return cached

    proto_name = LIB_PREFIX + lib_name
    proto = bpy.data.materials.get(proto_name)
    if proto is not None:
        _prototype_cache[lib_name] = proto
        return proto

    path = _library_path()
    if not os.path.exists(path):
        return None

    try:
        with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
            if lib_name in data_from.materials:
                data_to.materials = [lib_name]
            else:
                data_to.materials = []
    except Exception as e:  # pragma: no cover - corrupt library
        print(f"[ChatGPT TMD2] Failed to load material library: {e}")
        return None

    loaded = data_to.materials[0] if data_to.materials else None
    if loaded is None:
        return None

    # Rename the material and every node group it uses so that the original
    # add-on (which loads the very same names) can never interfere.
    loaded.name = proto_name
    loaded.use_fake_user = True
    nt = loaded.node_tree
    if nt:
        for node in nt.nodes:
            if node.type == 'GROUP' and node.node_tree:
                group = node.node_tree
                if not group.name.startswith(LIB_PREFIX):
                    group.name = LIB_PREFIX + lib_name + "_NT"
                    group.use_fake_user = True
    _prototype_cache[lib_name] = loaded
    return loaded


def library_material_names():
    """List of shader library materials that can be instantiated."""
    available = []
    path = _library_path()
    if not os.path.exists(path):
        return available
    try:
        with bpy.data.libraries.load(path, link=False) as (data_from, _):
            available = list(data_from.materials)
    except Exception:
        return available
    return available


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------
def _nodes(material):
    return material.node_tree if material.use_nodes else None


def _image_is_grayscale(image, samples=256):
    """Cheap check whether a texture is effectively black and white."""
    if image is None or image.size[0] == 0:
        return False
    try:
        pixels = image.pixels[:]
    except Exception:
        return False
    if not pixels:
        return False
    stride = max(4, (len(pixels) // (samples * 4)) * 4 or 4)
    max_delta = 0.0
    count = 0
    for i in range(0, len(pixels) - 3, stride):
        r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
        a = pixels[i + 3]
        if a <= 0.004:
            continue
        max_delta = max(max_delta, abs(r - g), abs(g - b), abs(r - b))
        count += 1
        if count >= samples:
            break
    return count > 0 and max_delta < 0.02


def _ensure_uv_node(nt, uv_name, image_node):
    """Make ``image_node`` sample ``uv_name`` (creates a UV Map node if needed)."""
    if not uv_name:
        return
    existing = None
    for link in nt.links:
        if link.to_node == image_node and link.to_socket.name == 'Vector':
            existing = link.from_node
            break
    if existing and existing.type == 'UVMAP':
        existing.uv_map = uv_name
        return
    uv_node = nt.nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = uv_name
    uv_node.label = uv_name
    uv_node.location = (image_node.location.x - 250, image_node.location.y - 150)
    nt.links.new(uv_node.outputs['UV'], image_node.inputs['Vector'])


def _find_output(nt):
    for node in nt.nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            return node
    for node in nt.nodes:
        if node.type == 'OUTPUT_MATERIAL':
            return node
    return None


def _set_blend(material, mode):
    """Apply a blend mode on both Blender 4.2+ and older versions."""
    mode = (mode or "OPAQUE").upper()
    try:
        if mode == 'BLEND':
            material.surface_render_method = 'BLENDED'
        elif mode in {'HASHED', 'CLIP', 'ADD'}:
            material.surface_render_method = 'DITHERED'
        else:
            material.surface_render_method = 'DITHERED'
    except Exception:
        pass
    # Blender < 4.2 (EEVEE legacy)
    legacy = {'BLEND': 'BLEND', 'HASHED': 'HASHED', 'CLIP': 'CLIP', 'OPAQUE': 'OPAQUE'}
    try:
        material.blend_method = legacy.get(mode, 'HASHED')
    except Exception:
        pass
    try:
        material.shadow_method = 'HASHED' if mode != 'OPAQUE' else 'OPAQUE'
    except Exception:
        pass
    material.use_backface_culling = False
    try:
        material.show_transparent_back = mode == 'BLEND'
    except Exception:
        pass


def _get_principled(nt):
    for node in nt.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def _get_emission_node(nt):
    for node in nt.nodes:
        if node.type == 'EMISSION':
            return node
    return None


# ---------------------------------------------------------------------------
# Interpretation of one TMD2 material
# ---------------------------------------------------------------------------
class Interpretation:
    """Result of analysing a raw TMD2 material."""

    def __init__(self):
        self.shader_id = ""
        self.desc = None
        self.known = False
        self.blend = "OPAQUE"
        self.base_color = (1.0, 1.0, 1.0, 1.0)
        self.alpha = 1.0
        self.alpha_clip = 0.0
        self.emission_color = (0.0, 0.0, 0.0, 1.0)
        self.emission_strength = 0.0
        self.uses_vertex_color = False
        self.has_alpha_texture = False
        self.colors = []          # [(name, (r,g,b))]
        self.scalars = []         # [(name, value)]
        self.notes = []
        self.unmapped_colors = []
        self.base_image = None


def _param(tm_material, index, default=0.0):
    params = getattr(tm_material, "shaderParams", []) or []
    if 0 <= index < len(params):
        return float(params[index])
    return default


def _rgb(tm_material, indices, default=(0.0, 0.0, 0.0)):
    params = getattr(tm_material, "shaderParams", []) or []
    out = []
    for i in indices:
        if i < len(params):
            out.append(float(params[i]))
        else:
            out.append(default[len(out)])
    while len(out) < 3:
        out.append(1.0)
    return tuple(out[:3])


def _texture_images(tm_material, images):
    """Return the list of Blender images referenced by a TMD2 material.

    ``images`` may be a plain list (indexed by table position) or a
    :class:`..core.textures.TextureResolver`.
    """
    resolver = images if hasattr(images, "resolve") else None
    result = []
    for mat_tex in getattr(tm_material, "textures", []) or []:
        image = None
        tex = getattr(mat_tex, "texture", None)
        index = getattr(mat_tex, "textureIndex", getattr(tex, "index", -1) if tex else -1)
        if resolver is not None:
            image = resolver.resolve(mat_tex)
            if image is None and isinstance(index, int):
                image = resolver.image_at(index)
        elif images and isinstance(index, int) and 0 <= index < len(images):
            image = images[index]
        result.append({
            "image": image,
            "index": index if isinstance(index, int) else -1,
            "hash": getattr(tex, "hash", 0) if tex else 0,
            "unk1": getattr(mat_tex, "unk1", 0),
            "unk2": getattr(mat_tex, "unk2", 0),
            "slot": getattr(mat_tex, "slot", 0),
        })
    return result


def analyse(tm_material, images=None, settings=None):
    """Turn the raw TMD2 material into an :class:`Interpretation`."""
    interp = Interpretation()
    shader_id = shader_db.normalize_shader_id(getattr(tm_material, "shaderID", ""))
    interp.shader_id = shader_id
    desc = shader_db.get_shader(shader_id)
    interp.desc = desc
    interp.known = desc is not None

    textures = _texture_images(tm_material, images)
    params = list(getattr(tm_material, "shaderParams", []) or [])

    # --- known shader ---------------------------------------------------
    if desc is not None:
        interp.blend = desc.blend
        interp.uses_vertex_color = desc.vertex_color
        for color in desc.colors:
            if not color.indices:
                continue
            values = _rgb(tm_material, color.indices)
            interp.colors.append((color.name, values))
        for scalar in desc.scalars:
            interp.scalars.append((scalar.name, _param(tm_material, scalar.index, scalar.default)))

        if desc.alpha_param is not None and desc.alpha_param < len(params):
            # The value stored here is a *clip threshold* (the game discards
            # pixels whose texture alpha is below it), not a global opacity.
            value = _param(tm_material, desc.alpha_param, 0.0)
            if 0.0 <= value < 1.0:
                interp.alpha_clip = value
                if value > 0.0:
                    interp.blend = "CLIP"
                    interp.notes.append(f"alpha clip threshold {value:.2f}")
            interp.alpha = 1.0
        if desc.alpha_texture is not None and desc.alpha_texture < len(textures):
            interp.has_alpha_texture = True

        # Hair shaders carry their full colour in the parameter block: this is
        # what makes hair look black instead of blonde when ignored.
        if desc.family == "hair" and interp.colors:
            name, values = interp.colors[0]
            interp.base_color = to_linear_rgba(values, 1.0)
            interp.notes.append(f"{name} from shaderParams")

        # Unlit shaders have no texture driven colour of their own, the tint is
        # entirely decided by the parameters (spirit / aura materials).
        if desc.family == "unlit" and interp.colors:
            name, values = interp.colors[0]
            interp.base_color = to_linear_rgba(values, 1.0)
            interp.notes.append(f"unlit tint from {name}")

        # "view" gives the user a quick idea of the coloured look in the
        # solid viewport even for shaders whose colour comes from a texture.
        if desc.family == "body" and not is_black(interp.base_color[:3]):
            interp.notes.append("base colour from texture")

        # Effect shaders whose textures are grayscale need their parameter
        # colour to reproduce the in-game tint (golden tiger aura, red spirit...).
        if desc.family in ("effect", "unlit"):
            tint = None
            tint_source = None
            for name, values in interp.colors:
                if not is_black(values):
                    tint, tint_source = values, name
                    break
            if tint is None:
                # The library has no dedicated colour slot for these shaders, so
                # the tint has to come from the parameter block.  ``best_color``
                # only reports float4 aligned, saturated blocks - nothing is
                # invented, an all-zero parameter block simply yields no tint.
                best = shader_db.best_color(params)
                if best is not None:
                    start, values = best
                    tint = values
                    tint_source = f"param[{start}..{start + 2}]"
                    interp.colors.append((tint_source, values))
            if tint is not None:
                rgba = to_linear_rgba(tint, 1.0)
                interp.base_color = rgba
                interp.emission_color = rgba
                interp.notes.append(f"{desc.family} tint from {tint_source}")
            if interp.emission_strength <= 0.0:
                # Unlit shaders always glow: that is what makes the golden /
                # spirit style effects look right in EEVEE and Cycles.
                interp.emission_color = interp.base_color
                interp.emission_strength = 1.0
                interp.notes.append(f"{desc.family} shader emits its base colour")

        # Emission only comes from parameters that are *authored* as emissive
        # (HDR colours on unlit shaders).  Specular colours are never treated as
        # emission, that was one of the bugs of the original importer.
        for color in desc.colors:
            if not color.emissive:
                continue
            values = _rgb(tm_material, color.indices)
            strength = max(values)
            if strength > 1.0:
                interp.emission_color = to_linear_rgba(
                    [v / strength for v in values], 1.0)
                interp.emission_strength = strength
                interp.notes.append(f"emission from {color.name}")

        # Unmapped colour-like slices are reported but not applied blindly.
        mapped = set(desc.color_indices)
        for start, values in shader_db.guess_color_params(params):
            if any(i in mapped for i in range(start, start + 3)):
                continue
            interp.unmapped_colors.append((start, values))

    # --- unknown shader -------------------------------------------------
    else:
        interp.blend = "OPAQUE"
        for start, values in shader_db.guess_color_params(params):
            interp.colors.append((f"param[{start}..{start + 2}]", values))
        best = shader_db.best_color(params)
        if best is not None:
            start, values = best
            interp.base_color = to_linear_rgba(values, 1.0)
            interp.notes.append(f"unknown shader, base color from param[{start}]")
        alpha_index = shader_db.guess_alpha_param(params, len(textures))
        if alpha_index is not None:
            value = _param(tm_material, alpha_index, 1.0)
            interp.alpha_clip = value
            interp.notes.append(f"unknown shader, alpha from param[{alpha_index}]")
        interp.uses_vertex_color = True
        interp.notes.append("unknown shader - using fallback material")

    # Vertex colours are only meaningful for shaders that read them.  Multiplying
    # a coloured base texture with the (often dark) vertex colours would destroy
    # the in-game look, which is a common complaint about naive importers.
    # remember the base texture: unlit shaders re-use it for their emission
    if textures and textures[0]["image"] is not None:
        interp.base_image = textures[0]["image"]

    if interp.known and desc is not None and desc.family == "body":
        base_image = interp.base_image
        if base_image is not None and not _image_is_grayscale(base_image):
            interp.uses_vertex_color = False
            interp.notes.append("vertex colour skipped: base texture is coloured")

    if settings is not None:
        mode = getattr(settings, "alpha_mode", 'AUTO')
        # The fallback PBR material has no alpha wiring, so "auto" keeps unknown
        # shaders opaque unless forcing a mode below.
        if not interp.known and mode == 'AUTO':
            interp.blend = "OPAQUE"
        if mode != 'AUTO':
            interp.blend = mode
        if getattr(settings, "vertex_color_mode", 'AUTO') == 'NEVER':
            interp.uses_vertex_color = False
        elif getattr(settings, "vertex_color_mode", 'AUTO') == 'ALWAYS':
            interp.uses_vertex_color = True
        boost = getattr(settings, "emission_boost", 1.0)
        interp.emission_strength *= float(boost)

    return interp


# ---------------------------------------------------------------------------
# Texture assignment
# ---------------------------------------------------------------------------
def _set_viewport_color(material, interp):
    """Colour used by the solid view and the material preview thumbnail.

    The real colour lives inside the shader node group, but the solid view only
    reads ``Material.diffuse_color``.  White is used for known shaders driven by
    a texture so the model never shows up as a black blob in the viewport.
    """
    colour = tuple(interp.base_color)
    if interp.known and not interp.colors:
        colour = (1.0, 1.0, 1.0, 1.0)
    try:
        material.diffuse_color = colour
    except Exception:
        pass
    try:
        material.metallic = 0.0
        material.roughness = 0.6
    except Exception:
        pass


def _disconnect_node(material, node):
    """Remove every link of an image node so its default colour is used."""
    nt = _nodes(material)
    if nt is None:
        return
    for socket in list(node.outputs):
        for link in list(socket.links):
            nt.links.remove(link)


def _placeholder_image(name="CTMD2_missing_texture"):
    """Light grey stand-in for a texture that was never loaded."""
    from .textures import placeholder_image
    return placeholder_image(name)


def _assign_library_textures(material, desc, tm_material, images, notes):
    """Assign the images to the image nodes of the game shader graph.

    Textures that cannot be resolved are *wired to nothing*: a Blender image node
    without an image outputs black, which made every model whose ``.lds`` was
    missing render as a silhouette.  Unconnected group inputs keep the neutral
    values of the material library instead (and the user is told about it).
    """
    nt = _nodes(material)
    if nt is None:
        return
    textures = _texture_images(tm_material, images)
    missing = []
    for slot in desc.textures:
        node = nt.nodes.get(slot.node)
        if node is None or node.type != 'TEX_IMAGE':
            continue
        image = None
        if slot.tex_index < len(textures):
            image = textures[slot.tex_index]["image"]
        if image is None:
            _disconnect_node(material, node)
            if slot.role:
                node.label = f"{slot.name}: MISSING ({slot.role})"
            else:
                node.label = f"{slot.name}: MISSING"
            missing.append(slot.name)
            continue
        node.image = image
        try:
            node.image.colorspace_settings.name = 'Non-Color' if slot.non_color else 'sRGB'
        except Exception:
            pass
        node.label = f"{slot.name}: {slot.role}" if slot.role else slot.name
        if slot.uv:
            _ensure_uv_node(nt, slot.uv, node)
    if missing:
        notes.append("missing textures: " + ", ".join(missing))


def _apply_library_params(material, desc, tm_material, interp):
    """Push the real parameter values into the game shader node group."""
    nt = _nodes(material)
    if nt is None or desc is None:
        return
    group_node = None
    wanted = LIB_PREFIX + desc.library
    for node in nt.nodes:
        if node.type != 'GROUP' or node.node_tree is None:
            continue
        # The prototype's node group was renamed to CTMD2_LIB_<shader>_NT when it
        # was loaded, so both the plain and the renamed variant are accepted.
        name = node.node_tree.name
        if name == wanted or name.startswith(wanted) or name == desc.library:
            group_node = node
            break
    if group_node is None:
        for node in nt.nodes:
            if node.type == 'GROUP':
                group_node = node
                break
    if group_node is None:
        return

    inputs = group_node.inputs

    def safe_set(name, value):
        if name in inputs:
            try:
                inputs[name].default_value = value
                return True
            except Exception:
                return False
        return False

    for color in desc.colors:
        if not color.indices:
            continue
        values = _rgb(tm_material, color.indices, color.fallback or (0, 0, 0))
        if color.scale == "srgb":
            rgba = to_linear_rgba(values, 1.0)
        else:
            rgba = (values[0], values[1], values[2], 1.0)
        if not safe_set(color.name, rgba):
            continue
        # The library graph reads most colour inputs as raw values, so keep a
        # copy of the untouched numbers as a comment on the node label.
        interp.colors = [c for c in interp.colors]

    for scalar in desc.scalars:
        value = _param(tm_material, scalar.index, scalar.default)
        if scalar.min_value is not None:
            value = max(scalar.min_value, value)
        if scalar.max_value is not None:
            value = min(scalar.max_value, value)
        safe_set(scalar.name, value)

    # Generic sweep: any remaining colour input of the group that has a known
    # counterpart in the unmapped parameters is left untouched on purpose - we
    # never invent colours, they are only reported to the user.


def _iter_all_nodes(nt, _seen=None):
    """Yield every node of a node tree, descending into node groups once."""
    _seen = _seen if _seen is not None else set()
    for node in nt.nodes:
        yield node
        if node.type == 'GROUP' and node.node_tree and node.node_tree.name not in _seen:
            _seen.add(node.node_tree.name)
            for child in _iter_all_nodes(node.node_tree, _seen):
                yield child


def _bypass_node(nt, node):
    """Rewire ``node``'s consumers to a neutral value and drop its links.

    A ``Vertex Color`` node that points at a layer the mesh does not have
    evaluates to *black*, and because the game shaders multiply their base
    texture with it the whole model turns black.  When there is no usable colour
    attribute the node is unlinked instead, so the group falls back to its
    neutral constants.
    """
    for output in list(node.outputs):
        for link in list(output.links):
            target = link.to_socket
            nt.links.remove(link)
            if target.type == 'RGBA':
                target.default_value = (1.0, 1.0, 1.0, 1.0)
            elif target.type == 'VALUE':
                target.default_value = 1.0
    node.label = "Vertex Color: MISSING (set to white)"
    node.mute = True


def _apply_vertex_color_nodes(material, color_layer_names, interp):
    """Point every Vertex Color node at a colour layer that actually exists."""
    nt = _nodes(material)
    if nt is None:
        return
    names = [name for name in (color_layer_names or []) if name]
    first = names[0] if names else None
    found = 0
    for node in list(_iter_all_nodes(nt)):
        if node.type != 'VERTEX_COLOR':
            continue
        found += 1
        current = node.layer_name
        if current in names:
            continue
        # "Color2" nodes should stay on the second layer when there is one,
        # otherwise they are left untouched instead of silently reading the
        # first layer twice.
        if current and current.lower().endswith("2"):
            if len(names) > 1:
                node.layer_name = names[1]
            elif first:
                node.layer_name = first
            else:
                _bypass_node(nt, node)
        elif current in ("", "Color", "Col") or current not in names:
            if first:
                node.layer_name = first
            else:
                _bypass_node(nt, node)
    if found and first:
        interp.uses_vertex_color = True


def _add_emission_overlay(material, interp, base_color_linked=False):
    """Add the emission term that the game shader would add.

    Two different situations have to be covered:

    * the shader already contributes its own colour through the group result
      (``BSS1`` and friends) - the glow is simply *added* on top;
    * the shader is unlit (``BES0``, ``BEH0``) - its colour comes from the
      parameter block, so the emission source is that colour, optionally
      modulated by the base texture.  Without this the unlit materials render as
      flat grey, which is the opposite of the golden / red effect look.
    """
    nt = _nodes(material)
    if nt is None:
        return
    output = _find_output(nt)
    if output is None:
        return
    surface_link = output.inputs['Surface'].links
    if not surface_link:
        return
    source_socket = surface_link[0].from_socket
    nt.links.remove(surface_link[0])

    emission = nt.nodes.new('ShaderNodeEmission')
    emission.name = "CTMD2 Emission"
    emission.label = "CTMD2 Emission"
    emission.inputs['Color'].default_value = interp.emission_color
    # Emission strength is what turns a dark model into the glowing game look,
    # a value of exactly 1.0 adds almost nothing on its own.
    emission.inputs['Strength'].default_value = max(interp.emission_strength, 1.0)
    emission.location = (output.location.x - 200, output.location.y - 300)

    color_source = None
    if not base_color_linked:
        base_image = getattr(interp, "base_image", None)
        if base_image is not None:
            tex = nt.nodes.new('ShaderNodeTexImage')
            tex.name = "CTMD2 Emission Texture"
            tex.label = "CTMD2 Emission (base texture)"
            tex.image = base_image
            tex.location = (output.location.x - 500, output.location.y - 300)
            try:
                tex.image.colorspace_settings.name = 'sRGB'
            except Exception:
                pass
            color_source = tex.outputs['Color']
        if color_source is not None:
            # tint the texture with the colour taken from the parameter block
            mix = nt.nodes.new('ShaderNodeMix')
            mix.data_type = 'RGBA'
            mix.blend_type = 'MULTIPLY'
            mix.inputs['Factor'].default_value = 1.0
            mix.location = (output.location.x - 350, output.location.y - 300)
            nt.links.new(color_source, mix.inputs[6])
            mix.inputs[7].default_value = interp.emission_color
            color_source = mix.outputs[2]
        else:
            color_source = None

    add = nt.nodes.new('ShaderNodeAddShader')
    add.name = "CTMD2 Emission Add"
    add.label = "CTMD2 Emission Add"
    add.location = (output.location.x - 100, output.location.y)

    nt.links.new(source_socket, add.inputs[0])
    nt.links.new(emission.outputs['Emission'], add.inputs[1])
    nt.links.new(add.outputs['Shader'], output.inputs['Surface'])


# ---------------------------------------------------------------------------
# Fallback (unknown shader) material
# ---------------------------------------------------------------------------
def _add_vertex_color_mix(nt, color_socket, principled, layer_name, strength=1.0):
    """Multiply ``color_socket`` with the mesh colour attribute."""
    attribute = nt.nodes.new('ShaderNodeVertexColor')
    attribute.layer_name = layer_name
    attribute.label = f"Vertex Color ({layer_name})"
    attribute.location = (-450, -300)
    mix = nt.nodes.new('ShaderNodeMix')
    mix.data_type = 'RGBA'
    mix.blend_type = 'MULTIPLY'
    mix.inputs['Factor'].default_value = strength
    mix.location = (30, -150)
    nt.links.new(color_socket, mix.inputs[6])       # A (color)
    nt.links.new(attribute.outputs['Color'], mix.inputs[7])  # B (color)
    nt.links.new(mix.outputs[2], principled.inputs['Base Color'])
    return mix


def build_simple_material(name, interp, tm_material, images, mode="SIMPLE"):
    """Create a neutral material for an unknown shader."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nt = material.node_tree
    nt.nodes.clear()

    output = nt.nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    if mode == "PLACEHOLDER":
        emission = nt.nodes.new('ShaderNodeEmission')
        emission.name = "CTMD2 Placeholder"
        emission.inputs['Color'].default_value = (1.0, 0.0, 0.6, 1.0)
        emission.inputs['Strength'].default_value = 1.0
        emission.location = (300, 0)
        nt.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        material.diffuse_color = (1.0, 0.0, 0.6, 1.0)
        return material

    textures = _texture_images(tm_material, images)
    base_image = textures[0]["image"] if textures else None

    if mode == "UNLIT":
        emission = nt.nodes.new('ShaderNodeEmission')
        emission.name = "CTMD2 Emission"
        emission.inputs['Color'].default_value = interp.base_color
        emission.inputs['Strength'].default_value = max(interp.emission_strength, 1.0)
        emission.location = (300, 0)
        if base_image is not None:
            tex = nt.nodes.new('ShaderNodeTexImage')
            tex.name = "Texture 1"
            tex.image = base_image
            tex.location = (-200, 0)
            try:
                tex.image.colorspace_settings.name = 'sRGB'
            except Exception:
                pass
            nt.links.new(tex.outputs['Color'], emission.inputs['Color'])
            if interp.uses_vertex_color:
                attribute = nt.nodes.new('ShaderNodeVertexColor')
                attribute.layer_name = "Color"
                attribute.location = (-400, -200)
                mix = nt.nodes.new('ShaderNodeMix')
                mix.data_type = 'RGBA'
                mix.blend_type = 'MULTIPLY'
                mix.inputs['Factor'].default_value = 1.0
                mix.location = (0, 0)
                nt.links.new(tex.outputs['Color'], mix.inputs[6])
                nt.links.new(attribute.outputs['Color'], mix.inputs[7])
                nt.links.new(mix.outputs[2], emission.inputs['Color'])
        nt.links.new(emission.outputs['Emission'], output.inputs['Surface'])
        return material

    principled = nt.nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    principled.inputs['Base Color'].default_value = interp.base_color
    principled.inputs['Roughness'].default_value = 0.6
    if 'Metallic' in principled.inputs:
        # Golden / metallic looking materials usually have a bright, saturated
        # colour parameter - be conservative and stay non metallic unless the
        # caller asks otherwise.
        principled.inputs['Metallic'].default_value = 0.0
    nt.links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    if mode == "DIFFUSE":
        diffuse = nt.nodes.new('ShaderNodeBsdfDiffuse')
        diffuse.location = (300, -250)
        nt.links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])

    if base_image is not None:
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.name = "Texture 1"
        tex.label = "Base Color"
        tex.image = base_image
        tex.location = (-200, 0)
        try:
            tex.image.colorspace_settings.name = 'sRGB'
        except Exception:
            pass
        if interp.uses_vertex_color:
            _add_vertex_color_mix(nt, tex.outputs['Color'], principled, "Color")
        else:
            nt.links.new(tex.outputs['Color'], principled.inputs['Base Color'])

        target = principled.inputs['Alpha']
        if interp.blend in {'HASHED', 'CLIP', 'BLEND'}:
            nt.links.new(tex.outputs['Alpha'], target)
        if interp.blend == 'CLIP':
            try:
                principled.inputs['Alpha'].default_value = 1.0
            except Exception:
                pass

    elif interp.uses_vertex_color:
        attribute = nt.nodes.new('ShaderNodeVertexColor')
        attribute.layer_name = "Color"
        attribute.location = (-200, -100)
        mix = nt.nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        mix.blend_type = 'MULTIPLY'
        mix.inputs['Factor'].default_value = 1.0
        mix.location = (50, 0)
        mix.inputs[6].default_value = interp.base_color
        nt.links.new(attribute.outputs['Color'], mix.inputs[7])
        nt.links.new(mix.outputs[2], principled.inputs['Base Color'])

    if interp.emission_strength > 0.0 and 'Emission Color' in principled.inputs:
        principled.inputs['Emission Color'].default_value = interp.emission_color
        principled.inputs['Emission Strength'].default_value = interp.emission_strength

    return material


def _auto_library_for_unknown(tm_material, images):
    """Pick the most fitting known shader family for an unknown shader ID."""
    textures = _texture_images(tm_material, images)
    count = len(textures)
    if count >= 5:
        return "BSS1"
    if count == 2:
        return "BEH0"
    return "Default"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_material(name, tm_material, images, settings=None, color_layers=None,
                   force_blend=None):
    """Build a Blender material for ``tm_material``.

    Parameters
    ----------
    name        : material name
    tm_material : raw TMD2/TMD material (or a PropertyGroup mirror)
    images      : list of Blender images indexed like the TMD2 texture table
    settings    : ``scene.ctmd2_settings`` (optional)
    color_layers: names of the colour attributes on the target mesh
    """
    interp = analyse(tm_material, images, settings)

    material = None
    desc = interp.desc

    used_library = False
    if desc is not None:
        proto = _ensure_prototype(interp.shader_id)
        if proto is not None:
            material = proto.copy()
            material.name = name
            used_library = True

    if material is None:
        # Not implemented in the shipped library: honour the fallback setting.
        fallback = getattr(settings, "fallback_shader", 'SIMPLE') if settings else 'SIMPLE'
        if fallback == 'AUTO':
            auto_id = _auto_library_for_unknown(tm_material, images)
            proto = _ensure_prototype(auto_id)
            if proto is not None:
                material = proto.copy()
                material.name = name
                used_library = True
                desc = shader_db.get_shader(auto_id)
                interp.notes.append(f"auto fallback to {auto_id} shader")
        if material is None:
            material = build_simple_material(
                name, interp, tm_material, images,
                mode='SIMPLE' if fallback == 'AUTO' else fallback,
            )
            interp.notes.append(f"fallback material ({fallback})")

    if used_library and desc is not None:
        _assign_library_textures(material, desc, tm_material, images, interp.notes)
        _apply_library_params(material, desc, tm_material, interp)

    if color_layers:
        _apply_vertex_color_nodes(material, color_layers, interp)

    if interp.emission_strength > 0.0:
        if _get_principled(_nodes(material)) is not None:
            principled = _get_principled(_nodes(material))
            try:
                principled.inputs['Emission Color'].default_value = interp.emission_color
                principled.inputs['Emission Strength'].default_value = interp.emission_strength
            except Exception:
                pass
        else:
            # unlit / effect shaders have no colour of their own in the graph
            unlit = bool(desc is not None and desc.family in ("unlit", "effect"))
            _add_emission_overlay(material, interp, base_color_linked=not unlit)

    _set_blend(material, force_blend or interp.blend)
    _set_viewport_color(material, interp)

    store_interpretation(material, tm_material, interp, images)
    return material


def store_interpretation(material, tm_material, interp, images):
    """Write everything interesting into the material property group."""
    if not hasattr(material, "ctmd2_material"):
        return
    props = material.ctmd2_material
    props.shader_id = interp.shader_id
    props.shader_known = interp.known
    props.shader_detected = interp.desc.library if interp.desc else "fallback"
    props.detected_blend = interp.blend if interp.blend in {
        'OPAQUE', 'CLIP', 'HASHED', 'BLEND', 'ADD'} else 'HASHED'
    props.uses_vertex_color = interp.uses_vertex_color
    props.uses_emission = interp.emission_strength > 0.0
    props.base_color = interp.base_color
    props.emission_color = interp.emission_color
    props.emission_strength = interp.emission_strength
    props.alpha_value = interp.alpha
    props.alpha_clip_threshold = interp.alpha_clip
    props.notes = "; ".join(interp.notes)[:1024]


def describe_material(material):
    """Human readable dump used by the "Report" button."""
    props = material.ctmd2_material
    lines = [
        f"Material     : {material.name}",
        f"Shader ID    : {props.shader_id or '(empty)'}",
        f"Known shader : {props.shader_known} ({props.shader_detected})",
        f"Blend        : {props.detected_blend}",
        f"Base color   : {tuple(round(c, 4) for c in props.base_color)}",
        f"Alpha        : {props.alpha_value:.4f}  clip={props.alpha_clip_threshold:.4f}",
        f"Emission     : {tuple(round(c, 4) for c in props.emission_color)} "
        f"x{props.emission_strength:.2f}",
        f"Vertex color : {props.uses_vertex_color}",
        f"Notes        : {props.notes}",
        "Textures     :",
    ]
    for i, tex in enumerate(props.textures):
        lines.append(
            f"  [{i}] hash={tex.texture_hash} index={tex.texture_index} "
            f"slot={tex.slot} image={tex.image.name if tex.image else None}"
        )
    lines.append("Parameters   :")
    for i, param in enumerate(props.param_values):
        lines.append(f"  [{i}] {param.value:.5f}")
    return "\n".join(lines)


def rebuild_material_from_props(material, settings=None):
    """Rebuild the node tree of an already imported material from stored props."""
    props = material.ctmd2_material
    shader_id = shader_db.normalize_shader_id(props.shader_id)
    desc = shader_db.get_shader(shader_id)
    if desc is not None:
        proto = _ensure_prototype(shader_id)
        if proto is None:
            return False
        images = [t.image for t in props.textures]
        # A tiny shim so the shared helpers can be reused for property groups.
        shim = _PropsShim(tm_material=props, images=images)
        interp = _interpret_from_props(props, desc)
        new_mat = proto.copy()
        new_mat.name = material.name
        _assign_library_textures(new_mat, desc, shim, images, interp.notes)
        _apply_library_params(new_mat, desc, shim, interp)
        mesh = _mesh_for_material(material)
        if mesh is not None:
            _apply_vertex_color_nodes(
                new_mat, [a.name for a in mesh.color_attributes], interp)
        if interp.emission_strength > 0.0:
            _add_emission_overlay(new_mat, interp)
        _set_blend(new_mat, interp.blend)
        _set_viewport_color(new_mat, interp)
        _swap_node_tree(material, new_mat)
        store_interpretation(material, shim, interp, images)
        bpy.data.materials.remove(new_mat)
        return True
    return False


class _PropsShim:
    """Adapter that makes a material PropertyGroup look like a raw TMD material."""

    def __init__(self, tm_material=None, images=None):
        self._props = tm_material
        self.shaderID = tm_material.shader_id
        self.hash = tm_material.material_hash
        self.shaderParams = [p.value for p in tm_material.param_values]
        self.textures = [_PropTexture(t) for t in tm_material.textures]


class _PropTexture:
    def __init__(self, tex):
        self.textureIndex = tex.texture_index
        self.unk1 = tex.value1
        self.unk2 = tex.value2
        self.slot = tex.slot
        self.texture = _PropTextureRef(tex.texture_hash, tex.texture_index)


class _PropTextureRef:
    def __init__(self, tex_hash, index):
        try:
            self.hash = int(tex_hash)
        except (TypeError, ValueError):
            self.hash = 0
        self.index = index


def _interpret_from_props(props, desc):
    interp = Interpretation()
    interp.shader_id = desc.shader_id
    interp.desc = desc
    interp.known = True
    interp.blend = props.detected_blend
    interp.uses_vertex_color = props.uses_vertex_color
    interp.base_color = tuple(props.base_color)
    interp.emission_color = tuple(props.emission_color)
    interp.emission_strength = props.emission_strength
    interp.alpha = props.alpha_value
    interp.alpha_clip = props.alpha_clip_threshold
    return interp


def _mesh_for_material(material):
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.data:
            continue
        for slot in obj.material_slots:
            if slot.material == material:
                return obj.data
    return None


def _swap_node_tree(target, source):
    """Replace ``target``'s node tree with a copy of ``source``'s."""
    if target.node_tree is not None:
        bpy.data.node_groups  # noqa: B018 - keep bpy loaded
        for node in list(target.node_tree.nodes):
            target.node_tree.nodes.remove(node)
        # Copy nodes/links manually (materials cannot share node trees).
        mapping = {}
        for node in source.node_tree.nodes:
            new_node = target.node_tree.nodes.new(node.bl_idname)
            new_node.name = node.name
            new_node.label = node.label
            new_node.location = node.location
            for attr in ("image", "layer_name", "uv_map", "operation", "blend_type",
                         "data_type", "clamp_factor", "node_tree", "interpolation",
                         "extension", "mute"):
                if hasattr(node, attr):
                    try:
                        setattr(new_node, attr, getattr(node, attr))
                    except Exception:
                        pass
            if hasattr(node, "inputs"):
                for src_in, dst_in in zip(node.inputs, new_node.inputs):
                    if hasattr(src_in, "default_value") and not src_in.is_linked:
                        try:
                            dst_in.default_value = src_in.default_value
                        except Exception:
                            pass
            mapping[node.name] = new_node
        for link in source.node_tree.links:
            try:
                target.node_tree.links.new(
                    mapping[link.from_node.name].outputs[link.from_socket.name],
                    mapping[link.to_node.name].inputs[link.to_socket.name],
                )
            except Exception:
                continue
    try:
        target.surface_render_method = source.surface_render_method
    except Exception:
        pass
    try:
        target.blend_method = source.blend_method
    except Exception:
        pass
