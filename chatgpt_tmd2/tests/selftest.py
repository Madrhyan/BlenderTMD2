"""Self tests for the ChatGPT TMD2 add-on.

Run inside Blender:

    blender --background --python chatgpt_tmd2/tests/selftest.py

or with the ``bpy`` Python module:

    python chatgpt_tmd2/tests/selftest.py -- --addon-dir <parent dir of chatgpt_tmd2>

The tests build a synthetic TMD2 file (valid magic, real header, real material
block) and exercise the importer and exporter end to end.
"""

import inspect
import os
import re
import sys
import traceback

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.dirname(TEST_DIR)
PARENT = os.path.dirname(ADDON_DIR)

if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

import chatgpt_tmd2  # noqa: E402
from chatgpt_tmd2.ctamlib.tmd2 import (  # noqa: E402
    TMD2,
    TMD2BoundingBox,
    TMD2Material,
    TMD2MatTexture,
    TMD2Model,
    TMD2Submesh,
    TMD2Texture,
)
from chatgpt_tmd2.exporters.writer import write_tmd2, write_lds  # noqa: E402

FAILURES = []
CHECKS = [0]


def check(condition, message):
    CHECKS[0] += 1
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL: {message}")
    return condition


# ---------------------------------------------------------------------------
# Sample file construction
# ---------------------------------------------------------------------------
def _params(shader):
    params = [0.0] * 40
    if shader in ("BSS1", "BEI0", "BED0", "BGH0", "BGH1"):
        params[0] = 0.35
        params[1:4] = [0.9, 0.75, 0.2]
        params[10:12] = [0.5, 0.3]
        params[16:19] = [0.55, 0.0, 0.0]
        params[24] = 0.42
        params[28] = 0.65
        params[29:32] = [0.7, 0.6, 0.3]
    elif shader in ("BHA1", "BHG1"):
        params[9:12] = [0.99, 0.76, 0.33]
        params[12:15] = [0.88, 0.58, 0.22]
    elif shader in ("BEH0", "BES0"):
        # unlit effects carry their tint as a float4 block after a scalar
        params[0] = 0.75
        params[3:6] = [1.0, 0.72, 0.05]
    elif shader == "ZZZZ":
        # trailing padding is 1.0 -> the block before it must stay at index 3
        params[0] = 0.0
        params[3:6] = [0.8, 0.1, 0.1]
        params[6] = 1.0
    return params


def _dds(w=8, h=8, color=0xF800):
    import struct
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 8, 0x1007)
    struct.pack_into("<I", header, 12, h)
    struct.pack_into("<I", header, 16, w)
    struct.pack_into("<I", header, 20, max(4, w // 2))
    struct.pack_into("<I", header, 28, 1)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 4)
    struct.pack_into("<I", header, 84, struct.unpack("<I", b"DXT1")[0])
    struct.pack_into("<I", header, 108, 0x1000)
    blocks = bytearray()
    for _ in range(max(1, (w // 4) * (h // 4))):
        blocks += struct.pack("<HHI", color, 0, 0)
    return bytes(header) + bytes(blocks)


def make_sample(path, shader="BSS1", with_lds=True):
    tmd = TMD2()
    tmd.name = os.path.splitext(os.path.basename(path))[0]
    tmd.flag1 = 0
    tmd.flag2 = 0
    tmd.animFlag = 0
    tmd.afterImageValue = 0
    tmd.bones = []
    tmd.boundingBox = [-1, -1, -1, 1, 1, 1]
    tmd.unkBoneInfo = []
    tmd.indexTables = []
    tmd.materials = []
    tmd.textures = []
    tmd.models = []
    tmd.modelFlags = 0x1 | 0x2 | 0x4 | 0x10 | 0x20 | 0x80 | 0x200 | 0x1000

    for i in range(5):
        texture = TMD2Texture()
        texture.hash = 1000 + i
        texture.index = i
        texture.width = 8
        texture.height = 8
        tmd.textures.append(texture)

    for m, (shader_id, params) in enumerate([
        (shader, _params(shader)),
        ("ZZZZ", [0.0, 0.0, 0.0, 0.8, 0.1, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    ]):
        material = TMD2Material()
        material.hash = 5000 + m
        material.shaderID = shader_id
        material.shaderParams = params
        material.unk = -1
        for k in range(5):
            mat_tex = TMD2MatTexture()
            mat_tex.texture = tmd.textures[k]
            mat_tex.textureIndex = k
            mat_tex.textureHash = tmd.textures[k].hash
            mat_tex.unk1 = 0
            mat_tex.unk2 = 0
            mat_tex.slot = k
            material.textures.append(mat_tex)
        tmd.materials.append(material)

    dtype = np.dtype([
        ("position", "f4", (3,)), ("custom_normal", "f4", (3,)),
        ("uv0", "f4", (2,)), ("uv1", "f4", (2,)),
        ("color0", "f4", (4,)), ("color1", "f4", (4,)),
    ])
    vertices = np.zeros(8, dtype=dtype)
    vertices["position"] = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0),
    ]
    vertices["custom_normal"] = [(0, 0, 1)] * 8
    vertices["uv0"] = [(0, 0), (1, 0), (1, 1), (0, 1)] * 2
    vertices["uv1"] = vertices["uv0"]
    vertices["color0"] = [(1, 1, 1, 1)] * 8
    vertices["color1"] = [(0.5, 0.25, 0.75, 1)] * 8
    tmd.vertices = vertices
    tmd.triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.uint32)
    tmd.trianglesCount = 4

    for m, material in enumerate(tmd.materials):
        model = TMD2Model()
        model.name = f"mesh_{m}"
        model.hash = 7000 + m
        submesh = TMD2Submesh()
        submesh.material = material
        submesh.triangles = tmd.triangles[m * 2:m * 2 + 2]
        submesh.vertexIndices = np.unique(submesh.triangles)
        model.meshes = [submesh]
        model.materials = [material]
        tmd.models.append(model)

    bbox = TMD2BoundingBox()
    bbox.corners = [[0, 0, 0]] * 8
    tmd.BBoxCorners = [bbox] * (len(tmd.models) + 1)
    write_tmd2(tmd, path, compress=False)

    if with_lds:
        lds_path = os.path.splitext(path)[0] + ".lds"
        colors = [0xF800, 0x07E0, 0x001F, 0xFFE0, 0x2104]
        write_lds([_dds(color=c) for c in colors], lds_path, compress=False)
    return tmd


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_registration():
    print("[1] registration / identifiers")
    chatgpt_tmd2.register()
    check(hasattr(bpy.types.Material, "ctmd2_material"), "Material.ctmd2_material missing")
    check(hasattr(bpy.types.Object, "ctmd2_mesh"), "Object.ctmd2_mesh missing")
    check(hasattr(bpy.types.Object, "ctmd2_props"), "Object.ctmd2_props missing")
    check(hasattr(bpy.types.Scene, "ctmd2_settings"), "Scene.ctmd2_settings missing")
    check(not hasattr(bpy.types.Material, "tmd2_material"),
          "add-on registered the original add-on's material property")
    check(not hasattr(bpy.types.Scene, "tmd2_props"),
          "add-on registered an original add-on identifier")

    for operator in ("import_tmd2", "export_tmd2", "drop_tmd2", "rebuild_material"):
        check(hasattr(bpy.ops.ctmd2, operator), f"operator ctmd2.{operator} missing")

    check("ctmd2.import_tmd2" != "import_scene.tmd2", "import operator id collision")
    ids = [getattr(cls, "bl_idname", None) for cls in chatgpt_tmd2._CLASSES]
    ids = [identifier for identifier in ids if identifier]
    check(len(ids) == len(set(ids)), "duplicate bl_idname inside the add-on")
    for identifier in ids:
        check(identifier.startswith("CTMD2") or identifier.startswith("ctmd2."),
              f"identifier does not use the add-on prefix: {identifier}")
    # the original add-on's identifiers must not be registered by this add-on
    original_ids = {
        "import_scene.tmd2", "import_scene.drop_tmd2", "TMD2_FH_import",
        "MATERIAL_PT_tmd2_material", "OBJECT_PT_tmd2_mesh", "OBJECT_PT_tmd2_panel",
        "TMD2_UL_Textures", "TMD2_UL_Params", "tmd2.copy_material_data",
        "export_scene.tmd2",
    }
    check(not (set(ids) & original_ids), f"identifier collision: {set(ids) & original_ids}")


def test_import(tmpdir):
    print("[2] import")
    path = os.path.join(tmpdir, "selftest_bss1.tmd2")
    make_sample(path, "BSS1")
    result = bpy.ops.ctmd2.import_tmd2(
        filepath=path, directory=tmpdir,
        files=[{"name": os.path.basename(path)}])
    check(result == {'FINISHED'}, "import operator did not finish")

    collection = bpy.data.collections.get("selftest_bss1")
    check(collection is not None, "import collection missing")
    if not collection:
        return None
    meshes = [o for o in collection.objects if o.type == 'MESH']
    check(len(meshes) == 2, f"expected 2 meshes, got {len(meshes)}")
    check(any(o.name.startswith("#CTMD2 PROPERTIES") for o in collection.objects),
          "file property object missing")
    check(not any(o.name.startswith("#TMD PROPERTIES") for o in collection.objects),
          "add-on created an object named like the original add-on")

    for obj in meshes:
        mesh = obj.data
        check(len(mesh.vertices) == 4, f"{obj.name}: unexpected vertex count")
        check(len(mesh.polygons) == 2, f"{obj.name}: unexpected polygon count")
        check([u.name for u in mesh.uv_layers][:2] == ["UVMap", "UVMap1"],
              f"{obj.name}: UV layers wrong: {[u.name for u in mesh.uv_layers]}")
        check([c.name for c in mesh.color_attributes] == ["Color", "Color2"],
              f"{obj.name}: colour attributes wrong")

    # material checks
    known = [m for m in bpy.data.materials
             if getattr(m, "ctmd2_material", None) and m.ctmd2_material.shader_id == "BSS1"]
    check(len(known) == 1, "known shader material missing")
    if known:
        mat = known[0]
        props = mat.ctmd2_material
        check(props.shader_known, "BSS1 not detected as known shader")
        check(props.detected_blend in {"CLIP", "HASHED", "OPAQUE"},
              f"unexpected blend mode {props.detected_blend}")
        check(abs(props.alpha_clip_threshold - 0.42) < 1e-4,
              f"alpha clip threshold not imported: {props.alpha_clip_threshold}")
        check(props.uses_vertex_color, "vertex colour not enabled for BSS1")
        images = [n.image for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE']
        check(any(image is not None and image.size[0] == 8 for image in images),
              "textures were not assigned to the material")
        check(any(image is not None and image.colorspace_settings.name == 'Non-Color'
                  for image in images), "normal map not set to Non-Color")
        # shadow colour from shaderParams must land in the node group
        group = None
        for node in mat.node_tree.nodes:
            if node.type == 'GROUP':
                group = node
                break
        check(group is not None, "shader node group missing")
        if group is not None:
            shadow = group.inputs.get("Shadow Color")
            check(shadow is not None and abs(shadow.default_value[0] - 0.787) < 0.02,
                  f"shadow colour not applied: {shadow.default_value[:] if shadow else None}")
            vertex = group.inputs.get("Vertex Color 2")
            check(vertex is not None and vertex.is_linked,
                  "vertex colour not connected to the shader group")

    unknown = [m for m in bpy.data.materials
               if getattr(m, "ctmd2_material", None) and m.ctmd2_material.shader_id == "ZZZZ"]
    check(len(unknown) == 1, "fallback material for the unknown shader missing")
    if unknown:
        props = unknown[0].ctmd2_material
        check(not props.shader_known, "unknown shader reported as known")
        check(props.notes, "fallback material has no notes")
        check(props.base_color[0] > 0.5 and props.base_color[1] < 0.2,
              f"unknown shader colour not interpreted: {props.base_color[:]}")
    return collection


def test_export(collection, tmpdir):
    print("[3] export")
    if collection is None:
        check(False, "export skipped, import failed")
        return
    target = os.path.join(tmpdir, "selftest_export")
    result = bpy.ops.ctmd2.export_tmd2(collection=collection.name, filepath=target)
    check(result == {'FINISHED'}, "export operator did not finish")
    exported = target + ".tmd2"
    check(os.path.exists(exported), "exported file missing")
    if not os.path.exists(exported):
        return

    from chatgpt_tmd2.importers.reader import readTMD2
    tmd = readTMD2(exported)
    check(len(tmd.models) == 2, f"exported model count wrong: {len(tmd.models)}")
    check(len(tmd.materials) == 2, f"exported material count wrong: {len(tmd.materials)}")
    check({m.shaderID for m in tmd.materials} == {"BSS1", "ZZZZ"},
          f"shader IDs not preserved: {[m.shaderID for m in tmd.materials]}")
    for material in tmd.materials:
        check(len(material.shaderParams) > 0, "shader parameters lost on export")
    check(len(tmd.vertices) == 8, f"vertex count changed on export: {len(tmd.vertices)}")
    check(len(tmd.triangles) == 4, f"triangle count changed on export: {len(tmd.triangles)}")

    # texture table, hashes and the .lds archive
    check(len(tmd.textures) == 5, f"texture count changed on export: {len(tmd.textures)}")
    lds_path = target + ".lds"
    if os.path.exists(lds_path):
        from chatgpt_tmd2.core.textures import read_lds_bytes
        archive = read_lds_bytes(lds_path)
        check(len(archive.textures) == len(tmd.textures),
              f"lds texture count {len(archive.textures)} != {len(tmd.textures)}")
        check(all(len(data) > 0 for data in archive.textures),
              "lds contains empty texture entries")
        check(all(data[:4] == b"DDS " for data in archive.textures),
              "lds entries are not DDS files")

    # the material -> texture hashes must survive, the game looks them up by hash
    for material in tmd.materials:
        for index, mat_tex in enumerate(material.textures):
            check(mat_tex.textureHash == tmd.textures[index].hash,
                  f"texture hash {index} not preserved: "
                  f"{mat_tex.textureHash} vs {tmd.textures[index].hash}")

    # round trip: import the exported file again and compare the essentials
    reimport = os.path.join(tmpdir, "selftest_reimport")
    os.makedirs(reimport, exist_ok=True)
    before = set(bpy.data.objects.keys())
    result = bpy.ops.ctmd2.import_tmd2(
        filepath=exported, directory=tmpdir,
        files=[{"name": os.path.basename(exported)}])
    check(result == {'FINISHED'}, "re-import of the exported file failed")
    new_objects = [bpy.data.objects[name] for name in set(bpy.data.objects.keys()) - before]
    new_meshes = [obj for obj in new_objects if obj.type == 'MESH']
    check(len(new_meshes) == 2, f"re-import produced {len(new_meshes)} meshes")
    for obj in new_meshes:
        check(len(obj.data.uv_layers) == 2,
              f"re-import lost UV layers: {[u.name for u in obj.data.uv_layers]}")
        check(len(obj.data.color_attributes) == 2,
              f"re-import lost colour attributes: "
              f"{[c.name for c in obj.data.color_attributes]}")
    reimported = [m for m in bpy.data.materials
                  if getattr(m, "ctmd2_material", None)
                  and m.ctmd2_material.shader_id == "BSS1"
                  and m.name.startswith("selftest_export")]
    check(bool(reimported), "re-import did not create a BSS1 material")


def test_negative():
    print("[4] edge cases")
    from chatgpt_tmd2.core import shader_db

    check(shader_db.get_shader("BSS1") is not None, "shader_db lookup failed")
    check(shader_db.get_shader("nope") is None, "shader_db returned a shader for garbage")
    check(shader_db.normalize_shader_id("BSS1\x00") == "BSS1", "shader id normalisation failed")
    check(shader_db.get_shader("BSS0") is shader_db.get_shader("BSS1"), "alias failed")

    best = shader_db.best_color([0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.1, 0.0, 1.0])
    check(best is not None and best[1][0] > 0.8, f"colour guessing failed: {best}")
    check(shader_db.best_color([0.0, 0.0, 0.0, 0.0]) is None,
          "colour guessing invented a colour from black")

    from chatgpt_tmd2.core.material_builder import srgb_to_linear, linear_to_srgb
    check(abs(linear_to_srgb(srgb_to_linear(0.5)) - 0.5) < 1e-4, "colour conversion roundtrip")


def test_missing_textures(tmpdir):
    """Regression: a model without its .lds used to render pitch black.

    A Blender image node without an image outputs black, and a Vertex Color node
    pointing at a non existent layer does the same.  Both are now disconnected so
    the material falls back to the neutral library constants.
    """
    print("[5] missing textures / black model regression")
    from chatgpt_tmd2.core.textures import placeholder_image

    grey = placeholder_image("CTMD2_selftest_placeholder", 8, 8)
    check(tuple(grey.pixels[:3]) != (0.0, 0.0, 0.0), "placeholder image is black")
    check(max(grey.pixels[:3]) > 0.5, f"placeholder image too dark: {grey.pixels[:3]}")

    # a file without a matching .lds
    lonely_dir = os.path.join(tmpdir, "lonely")
    os.makedirs(lonely_dir, exist_ok=True)
    lonely = os.path.join(lonely_dir, "lonely_bss1.tmd2")
    make_sample(lonely, "BSS1", with_lds=False)
    if os.path.exists(os.path.join(lonely_dir, "lonely_bss1.lds")):
        os.remove(os.path.join(lonely_dir, "lonely_bss1.lds"))

    result = bpy.ops.ctmd2.import_tmd2(
        filepath=lonely, directory=lonely_dir,
        files=[{"name": os.path.basename(lonely)}])
    check(result == {'FINISHED'}, "import without .lds failed")

    materials = [m for m in bpy.data.materials
                 if getattr(m, "ctmd2_material", None)
                 and m.ctmd2_material.shader_id == "BSS1"
                 and m.name.startswith("lonely_bss1")]
    if not check(bool(materials), "material without textures missing"):
        return
    material = materials[0]

    # every image node must either hold a real image or be disconnected
    for node in material.node_tree.nodes:
        if node.type != 'TEX_IMAGE':
            continue
        if node.image is None:
            check(all(not socket.links for socket in node.outputs),
                  f"{node.name} has no image but is still linked (feeds black)")
            check("MISSING" in node.label, f"{node.name} not flagged as missing")

    # no vertex colour node may point at a layer the mesh does not have
    for node in material.node_tree.nodes:
        if node.type != 'VERTEX_COLOR':
            continue
        layers = []
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and any(slot.material == material for slot in obj.material_slots):
                layers += [attribute.name for attribute in obj.data.color_attributes]
        check(node.layer_name in layers or node.mute,
              f"vertex colour node points at a missing layer: {node.layer_name}")

    check(material.diffuse_color[0] > 0.5 and material.diffuse_color[1] > 0.5,
          f"viewport colour is too dark: {material.diffuse_color[:]}")
    check("missing textures" in material.ctmd2_material.notes,
          f"missing textures not reported: {material.ctmd2_material.notes!r}")


def test_effect_colour(tmpdir):
    """Regression: unlit/effect shaders must keep their parameter tint."""
    print("[6] effect (unlit) shader tint")
    path = os.path.join(tmpdir, "selftest_effect.tmd2")
    make_sample(path, "BEH0")
    result = bpy.ops.ctmd2.import_tmd2(
        filepath=path, directory=tmpdir,
        files=[{"name": os.path.basename(path)}])
    check(result == {'FINISHED'}, "effect shader import failed")
    materials = [m for m in bpy.data.materials
                 if getattr(m, "ctmd2_material", None)
                 and m.ctmd2_material.shader_id == "BEH0"]
    if not check(bool(materials), "effect material missing"):
        return
    props = materials[0].ctmd2_material
    check(props.emission_strength > 0.0, "effect shader does not glow")
    check(props.emission_color[0] > 0.5 and props.emission_color[2] < 0.5,
          f"effect tint not taken from shaderParams: {props.emission_color[:]}")
    emission = [n for n in materials[0].node_tree.nodes if n.type == 'EMISSION']
    check(bool(emission), "effect material has no emission node")
    if emission:
        strength = emission[0].inputs['Strength'].default_value
        check(strength >= 1.0, f"emission strength too weak to be visible: {strength}")


def _fake_layout(log):
    class Fake:
        def __init__(self):
            pass

        def __getattr__(self, name):
            def method(*args, **kwargs):
                log.append((name, args[:1]))
                return Fake()
            return method

    return Fake()


def test_ui(collection):
    print("[5] user interface")
    obj = None
    if collection is not None:
        obj = next((o for o in collection.objects if o.type == 'MESH'), None)
    if obj is not None:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

    drawn = 0
    for cls in chatgpt_tmd2._CLASSES:
        identifier = getattr(cls, "bl_idname", "")
        if not (identifier.startswith("CTMD2_PT_") or identifier.startswith("CTMD2_MT_")):
            continue
        log = []
        panel = type("PanelUnderTest", (), {})()
        panel.layout = _fake_layout(log)
        try:
            if hasattr(cls, "poll") and not cls.poll(bpy.context):
                continue
            cls.draw(panel, bpy.context)
        except Exception:
            traceback.print_exc()
            check(False, f"{identifier} failed to draw")
            continue
        drawn += 1
        check(bool(log), f"{identifier} drew nothing")
    check(drawn >= 4, f"only {drawn} interface elements could be drawn")

    # every operator the panels reference must exist, otherwise the panel dies
    operators = set()
    for cls in chatgpt_tmd2._CLASSES:
        if not getattr(cls, "bl_idname", "").startswith("CTMD2_PT_"):
            continue
        source = getattr(cls.draw, "__code__", None)
        if source is None:
            continue
        try:
            text = inspect.getsource(cls.draw)
        except OSError:
            continue
        operators.update(re.findall(r'"(ctmd2\.[a-z_0-9]+)"', text))
    check(bool(operators), "no operators referenced from the panels")
    for identifier in sorted(operators):
        check(hasattr(bpy.ops.ctmd2, identifier.rsplit(".", 1)[1]),
              f"panel references a missing operator: {identifier}")


def test_unregister():
    print("[5] unregister")
    chatgpt_tmd2.unregister()
    check(not hasattr(bpy.types.Material, "ctmd2_material"), "material property not removed")
    check(not hasattr(bpy.types.Scene, "ctmd2_settings"), "scene property not removed")
    check("ctmd2.import_tmd2" not in dir(bpy.types),
          "operator still registered after unregister")
    chatgpt_tmd2.register()
    check(hasattr(bpy.types.Material, "ctmd2_material"), "re-register failed")
    chatgpt_tmd2.unregister()


def test_rebuild(collection):
    print("[4] material rebuild")
    material = None
    for mat in bpy.data.materials:
        props = getattr(mat, "ctmd2_material", None)
        if props is not None and props.shader_id == "BSS1":
            material = mat
            break
    if not check(material is not None, "no BSS1 material to rebuild"):
        return

    if collection is not None:
        obj = next((o for o in collection.objects
                    if o.type == 'MESH'
                    and material.name in [m.name for m in o.data.materials if m]), None)
        if obj is not None:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)

    for node in list(material.node_tree.nodes):
        material.node_tree.nodes.remove(node)
    check(len(material.node_tree.nodes) == 0, "node wipe failed")

    result = bpy.ops.ctmd2.rebuild_material()
    check(result == {'FINISHED'}, "rebuild operator did not finish")
    check(len(material.node_tree.nodes) > 5, "rebuild did not recreate nodes")
    outputs = [link for link in material.node_tree.links if link.to_node.type == 'OUTPUT_MATERIAL']
    check(bool(outputs), "rebuild did not connect the material output")
    images = [n.image for n in material.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image]
    check(len(images) >= 3, f"rebuild restored only {len(images)} textures")
    vertex_colors = [n for n in material.node_tree.nodes if n.type == 'VERTEX_COLOR']
    check(bool(vertex_colors), "rebuild lost the vertex colour node")

    before = material.node_tree.nodes[0].name
    result = bpy.ops.ctmd2.rebuild_all_materials()
    check(result == {'FINISHED'}, "rebuild-all operator did not finish")
    check(material.node_tree.nodes[0].name == before, "rebuild-all changed the material")


def main():
    import tempfile
    argv = sys.argv
    separator = argv.index("--") if "--" in argv else len(argv)
    extra = argv[separator + 1:]

    print("ChatGPT TMD2 self test")
    tmpdir = tempfile.mkdtemp(prefix="ctmd2_selftest_")
    print(f"  work dir: {tmpdir}")

    for test in (test_registration, test_negative):
        try:
            test()
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"{test.__name__} raised")

    collection = None
    try:
        collection = test_import(tmpdir)
        test_export(collection, tmpdir)
    except Exception:
        traceback.print_exc()
        FAILURES.append("import/export raised")

    try:
        test_rebuild(collection)
    except Exception:
        traceback.print_exc()
        FAILURES.append("material rebuild raised")

    try:
        test_missing_textures(tmpdir)
    except Exception:
        traceback.print_exc()
        FAILURES.append("missing texture handling raised")

    try:
        test_effect_colour(tmpdir)
    except Exception:
        traceback.print_exc()
        FAILURES.append("effect shader colour raised")

    try:
        test_ui(collection)
    except Exception:
        traceback.print_exc()
        FAILURES.append("user interface raised")

    try:
        test_unregister()
    except Exception:
        traceback.print_exc()
        FAILURES.append("unregister raised")

    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} of {CHECKS[0]} checks)")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"OK - {CHECKS[0]} checks passed")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if "bpy" in sys.modules and bpy.app.background:
        sys.exit(exit_code)
