"""Helpers shared by the exporter modules."""

import json
import os

import numpy as np
from mathutils import Matrix, Vector
from math import radians

from ..ctamlib.tmd2 import tamCRC32


ZUP_TO_YUP = Matrix.Rotation(radians(-90), 4, 'X')

_HASHES = None


def hashes():
    global _HASHES
    if _HASHES is None:
        _HASHES = {}
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "hashes.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _HASHES = json.load(handle)
        except Exception:
            _HASHES = {}
    return _HASHES


def name_for_hash(value):
    return hashes().get(str(value))


def hash_for_name(name):
    return tamCRC32(name)


def collection_bbox(collection):
    """World space bounding box of every mesh in ``collection``."""
    corners = []
    for obj in collection.objects:
        if obj.type != 'MESH':
            continue
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))
    if not corners:
        zero = Vector((0.0, 0.0, 0.0))
        return zero, zero, [zero] * 8
    min_corner = Vector((min(v[i] for v in corners) for i in range(3)))
    max_corner = Vector((max(v[i] for v in corners) for i in range(3)))
    box = [
        (min_corner.x, min_corner.y, min_corner.z),
        (max_corner.x, min_corner.y, min_corner.z),
        (max_corner.x, max_corner.y, min_corner.z),
        (min_corner.x, max_corner.y, min_corner.z),
        (min_corner.x, min_corner.y, max_corner.z),
        (max_corner.x, min_corner.y, max_corner.z),
        (max_corner.x, max_corner.y, max_corner.z),
        (min_corner.x, max_corner.y, max_corner.z),
    ]
    return min_corner, max_corner, box


def material_props(material):
    """Return the ChatGPT TMD2 property group of ``material`` (or None)."""
    if material is None:
        return None
    return getattr(material, "ctmd2_material", None)


def mesh_props(obj):
    return getattr(obj, "ctmd2_mesh", None)


def props_object(collection):
    for obj in collection.objects:
        if obj.get("ctmd2_props_object"):
            return obj
    for obj in collection.objects:
        if obj.name.startswith("#CTMD2 PROPERTIES"):
            return obj
    return None


# ---------------------------------------------------------------------------
# Loop data extraction
# ---------------------------------------------------------------------------
def extract_loop_data(mesh_obj, bone_name_to_index, max_influences=8):
    """Collect everything the TMD2 writer needs from a Blender mesh."""
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()

    try:
        if not mesh.uv_layers:
            mesh.uv_layers.new(name="UVMap")
        mesh.calc_tangents()
    except Exception:
        pass

    num_loops = len(mesh.loops)
    num_tris = len(mesh.loop_triangles)

    vertex_indices = np.empty(num_loops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", vertex_indices)

    positions = np.empty((num_loops, 3), dtype=np.float32)
    vert_positions = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", vert_positions.ravel())
    positions[:] = vert_positions[vertex_indices]

    normals = np.empty((num_loops, 3), dtype=np.float32)
    mesh.loops.foreach_get("normal", normals.ravel())

    data = {
        "position": positions,
        "custom_normal": normals,
        "real_normal": normals.copy(),
        "loop_tri_indices": np.empty((num_tris, 3), dtype=np.uint32),
        "material_indices": np.empty(num_tris, dtype=np.uint8),
        "uv_count": min(3, len(mesh.uv_layers)),
        "color_count": min(2, len(mesh.color_attributes)),
    }
    mesh.loop_triangles.foreach_get("loops", data["loop_tri_indices"].ravel())
    mesh.loop_triangles.foreach_get("material_index", data["material_indices"])

    # Tangents (may fail on meshes without UVs)
    try:
        tangents = np.empty((num_loops, 3), dtype=np.float32)
        mesh.loops.foreach_get("tangent", tangents.ravel())
        data["tangent"] = tangents
        bitangents = np.empty((num_loops, 3), dtype=np.float32)
        mesh.loops.foreach_get("bitangent", bitangents.ravel())
        data["bitangent"] = bitangents
        signs = np.ones(num_loops, dtype=np.float32)
        data["tangent_sign"] = signs
        data["bitangent_sign"] = signs.copy()
    except Exception:
        pass

    for i in range(data["uv_count"]):
        uv = np.zeros((num_loops, 2), dtype=np.float32)
        mesh.uv_layers[i].data.foreach_get("uv", uv.ravel())
        uv[:, 1] = 1.0 - uv[:, 1]
        data[f"uv{i}"] = uv

    for i in range(data["color_count"]):
        colors = np.zeros((num_loops, 4), dtype=np.float32)
        try:
            mesh.color_attributes[i].data.foreach_get("color_srgb", colors.ravel())
        except Exception:
            mesh.color_attributes[i].data.foreach_get("color", colors.ravel())
        data[f"color{i}"] = colors

    # Bone weights
    bone_ids = np.zeros((num_loops, max_influences), dtype=np.uint16)
    weights = np.zeros((num_loops, max_influences), dtype=np.float32)
    if bone_name_to_index:
        v_bone_ids = np.zeros((len(mesh.vertices), max_influences), dtype=np.uint16)
        v_weights = np.zeros((len(mesh.vertices), max_influences), dtype=np.float32)
        groups = mesh_obj.vertex_groups
        for vertex in mesh.vertices:
            collected = []
            for group in vertex.groups:
                try:
                    name = groups[group.group].name
                except (IndexError, TypeError):
                    continue
                if name in bone_name_to_index:
                    collected.append((bone_name_to_index[name], group.weight))
            if not collected:
                v_weights[vertex.index, 0] = 1.0
                continue
            collected.sort(key=lambda item: item[1], reverse=True)
            for slot, (bone_id, weight) in enumerate(collected[:max_influences]):
                v_bone_ids[vertex.index, slot] = bone_id
                v_weights[vertex.index, slot] = weight
            total = v_weights[vertex.index].sum()
            if total > 0:
                v_weights[vertex.index] /= total
            else:
                v_weights[vertex.index, 0] = 1.0
        bone_ids[:] = v_bone_ids[vertex_indices]
        weights[:] = v_weights[vertex_indices]
    else:
        weights[:, 0] = 1.0

    data["boneIDs"] = bone_ids
    data["weights"] = weights
    return data


def dedupe_vertices(vertices, tri_indices, material_indices):
    """Merge identical loop vertices and remap the triangles."""
    raw = vertices.view(np.void)
    unique_raw, unique_index, inverse = np.unique(raw, return_index=True, return_inverse=True)
    unique_vertices = vertices[unique_index]
    remapped = inverse[tri_indices.ravel()].reshape(tri_indices.shape).astype(np.uint32)
    return unique_vertices, remapped, material_indices
