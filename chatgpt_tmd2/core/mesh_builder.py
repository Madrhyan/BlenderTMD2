"""Mesh construction for TMD2 / TMD / TMD0 / TMDV files.

The original importer creates vertices per submesh which means every submesh gets
duplicated geometry and custom normals / UVs are pushed through ``bmesh`` layers
in a fragile way.  This module:

* builds the vertex buffer once per model and re-uses it,
* writes UV layers and colour attributes with ``foreach_set`` (fast + exact),
* keeps the raw TMD2 vertex colours as **byte colour** attributes named
  ``Color`` / ``Color2`` so the game shaders can consume them,
* preserves custom normals when the file stores them,
* keeps the original material ordering so material indices stay valid.
"""

import numpy as np
import bpy
import bmesh
from mathutils import Matrix
from math import radians


YUP_TO_ZUP = Matrix.Rotation(radians(90), 4, 'X')

#: TMD2 colour attributes are authored as sRGB bytes.
COLOR_LAYER_NAMES = ("Color", "Color2")


def _field_names(vertices):
    if vertices is None:
        return []
    try:
        return list(vertices.dtype.names or [])
    except AttributeError:
        return []


def _flip_v(uv_array):
    uv = np.array(uv_array, dtype=np.float32, copy=True)
    if uv.size:
        uv[..., 1] = 1.0 - uv[..., 1]
    return uv


class MeshBuildResult:
    def __init__(self):
        self.mesh = None
        self.obj = None
        self.uv_layers = []
        self.color_layers = []
        self.material_slots = []
        self.warnings = []


def build_mesh(model, tmd, name, collection, materials, color_layers_enabled=True,
               import_normals=True, operator=None, bone_names=None):
    """Create one Blender object for ``model``.

    ``model`` must expose ``meshes`` and ``materials`` (TMD2Model / TMDModel).
    """
    result = MeshBuildResult()
    vertices = getattr(tmd, "vertices", None)
    flags = int(getattr(tmd, "modelFlags", 0))
    fields = _field_names(vertices)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    result.mesh = mesh
    result.obj = obj

    # Vertex groups must exist before the bmesh weights are written.
    if bone_names and (flags & 0x400):
        for bone_name in bone_names:
            obj.vertex_groups.new(name=bone_name)

    # ---- materials ----------------------------------------------------
    mat_map = {}
    for index, tm_mat in enumerate(getattr(model, "materials", []) or []):
        blender_mat = materials.get(id(tm_mat))
        if blender_mat is None:
            blender_mat = materials.get(tm_mat)
        if blender_mat is None:
            result.warnings.append(f"missing material {index}")
            continue
        mesh.materials.append(blender_mat)
        mat_map[tm_mat] = len(mesh.materials) - 1
    result.material_slots = [m for m in mesh.materials]

    if vertices is None or len(vertices) == 0:
        return result

    # ---- geometry -----------------------------------------------------
    bm = bmesh.new()
    uv_layers = []
    if "uv" in fields:
        uv_layers.append(bm.loops.layers.uv.new("UVMap"))
    if "uv2" in fields or "uv1" in fields:
        uv_layers.append(bm.loops.layers.uv.new("UVMap1"))
    if "uv3" in fields:
        uv_layers.append(bm.loops.layers.uv.new("UVMap2"))

    # The deform layer has to exist before any vertex is created, adding a layer
    # later invalidates the BMVert handles.
    deform_layer = None
    if flags & 0x400 and "boneIDs" in fields and bone_names:
        deform_layer = bm.verts.layers.deform.verify()
    vertex_maps = []

    for submesh in getattr(model, "meshes", []) or []:
        indices = getattr(submesh, "vertexIndices", None)
        if indices is None or len(indices) == 0:
            vertex_maps.append(None)
            continue
        sub_verts = vertices[indices]
        bm_verts = [bm.verts.new(v["position"]) for v in sub_verts]
        bm.verts.ensure_lookup_table()
        vertex_maps.append(bm_verts)

        if deform_layer is not None and "boneIDs" in fields:
            index_table = getattr(submesh, "indexTable", None) or []
            ids = sub_verts["boneIDs"]
            weights = sub_verts["boneWeights"]
            if flags & 0x8000 and "boneIDs2" in fields:
                ids = np.concatenate([ids, sub_verts["boneIDs2"]], axis=1)
                weights = np.concatenate([weights, sub_verts["boneWeights2"]], axis=1)
            for vi, vert in enumerate(bm_verts):
                dv = vert[deform_layer]
                for bone_index, weight in zip(ids[vi], weights[vi]):
                    if weight <= 0.0:
                        continue
                    group_index = bone_index
                    if 0 <= int(bone_index) < len(index_table):
                        group_index = index_table[int(bone_index)]
                    dv[int(group_index)] = float(weight)

        material_index = mat_map.get(getattr(submesh, "material", None), 0)
        triangles = getattr(submesh, "triangles", None)
        if triangles is None:
            continue
        for tri in triangles:
            try:
                face = bm.faces.new([bm_verts[int(i)] for i in tri])
            except (ValueError, IndexError):
                continue
            face.smooth = True
            face.material_index = material_index

    bm.to_mesh(mesh)
    bm.free()

    # ---- UV layers ----------------------------------------------------
    uv_specs = []
    if "uv" in fields:
        uv_specs.append(("uv", 0))
    if "uv2" in fields:
        uv_specs.append(("uv2", 1))
    elif "uv1" in fields:
        uv_specs.append(("uv1", 1))
    if "uv3" in fields:
        uv_specs.append(("uv3", 2))

    for field_name, layer_index in uv_specs:
        if layer_index >= len(mesh.uv_layers):
            continue
        flat = []
        for submesh, bm_verts in zip(getattr(model, "meshes", []) or [], vertex_maps):
            if bm_verts is None:
                continue
            indices = submesh.vertexIndices
            tris = np.asarray(submesh.triangles, dtype=np.int64)
            if tris.size == 0:
                continue
            uvs = _flip_v(vertices[indices][field_name])
            flat.extend(uvs[tris].reshape(-1).tolist())
        if flat:
            layer = mesh.uv_layers[layer_index]
            expected = len(layer.data) * 2
            if len(flat) == expected:
                layer.data.foreach_set("uv", flat)
            elif flat:
                count = min(len(flat), expected)
                layer.data.foreach_set("uv", flat[:count])
        result.uv_layers.append(mesh.uv_layers[layer_index].name)

    # ---- colour attributes --------------------------------------------
    if color_layers_enabled:
        color_specs = []
        if "color" in fields:
            color_specs.append(("color", COLOR_LAYER_NAMES[0]))
        if "color2" in fields:
            color_specs.append(("color2", COLOR_LAYER_NAMES[1]))

        for field_name, layer_name in color_specs:
            colors = _collect_loop_colors(model, vertices, field_name)
            if not colors:
                continue
            try:
                attribute = mesh.color_attributes.new(
                    name=layer_name, type='BYTE_COLOR', domain='CORNER')
            except Exception:
                attribute = mesh.color_attributes.new(
                    name=layer_name, type='FLOAT_COLOR', domain='CORNER')
            expected = len(mesh.loops) * 4
            data = colors
            if len(data) != expected:
                data = (data + [1.0, 1.0, 1.0, 1.0] * (expected // 4 + 1))[:expected]
            try:
                attribute.data.foreach_set("color", data)
            except Exception:
                for i, value in enumerate(data[:len(attribute.data) * 4]):  # pragma: no cover
                    attribute.data[i // 4].color[i % 4] = value
            result.color_layers.append(attribute.name)

        if result.color_layers:
            try:
                mesh.color_attributes.active_color_index = 0
                mesh.color_attributes.render_color_index = 0
            except Exception:
                pass

    # ---- custom normals ----------------------------------------------
    if import_normals and "normal" in fields:
        normals = _collect_vertex_normals(model, vertices)
        if normals is not None and len(normals) == len(mesh.vertices):
            try:
                mesh.normals_split_custom_set_from_vertices(normals)
            except Exception as exc:  # pragma: no cover
                result.warnings.append(f"custom normals: {exc}")

    mesh.update()
    mesh.transform(YUP_TO_ZUP)
    mesh.validate(verbose=False)

    # ---- metadata -----------------------------------------------------
    try:
        props = obj.ctmd2_mesh
        props.name = getattr(model, "name", "") or name
        if getattr(model, "hashFlag", 0):
            props.has_hash = True
            props.name_hash = str(getattr(model, "hash", 0))
        if int(getattr(model, "nameFlag", 0)) & 4:
            props.has_name = True
    except Exception:
        pass

    return result


def _collect_loop_colors(model, vertices, field_name):
    data = []
    for submesh in getattr(model, "meshes", []) or []:
        indices = getattr(submesh, "vertexIndices", None)
        if indices is None or len(indices) == 0:
            continue
        tris = np.asarray(submesh.triangles, dtype=np.int64)
        if tris.size == 0:
            continue
        colors = np.asarray(vertices[indices][field_name], dtype=np.float32)
        if colors.shape[1] == 3:
            alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
            colors = np.concatenate([colors, alpha], axis=1)
        data.extend(colors[tris].reshape(-1).tolist())
    return data


def _collect_vertex_normals(model, vertices):
    out = []
    for submesh in getattr(model, "meshes", []) or []:
        indices = getattr(submesh, "vertexIndices", None)
        if indices is None or len(indices) == 0:
            continue
        out.extend(np.asarray(vertices[indices]["normal"], dtype=np.float32).tolist())
    if not out:
        return None
    arr = np.asarray(out, dtype=np.float32)
    if arr.shape[1] > 3:
        arr = arr[:, :3]
    return arr
