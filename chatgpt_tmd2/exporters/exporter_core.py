"""TMD2 collection exporter.

Writes a collection back to a .tmd2 (+ .lds) using the raw TMD2 structs from
``ctamlib.tmd2``.  Everything is read from the ``ctmd2_*`` property groups.
"""

import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler, Quaternion
from math import radians

from ..ctamlib.tmd2 import (
    TMD2,
    TMD2Bone,
    TMD2BoundingBox,
    TMD2Material,
    TMD2MatTexture,
    TMD2Model,
    TMD2Submesh,
    TMD2Texture,
    tamCRC32,
)
from .common import (
    collection_bbox,
    dedupe_vertices,
    extract_loop_data,
    hash_for_name,
    material_props,
    mesh_props,
    props_object,
)
from .writer import write_lds, write_tmd2


YUP_UP = Matrix.Rotation(radians(90), 4, 'X')


def _flip_positions(vertices):
    out = vertices.copy()
    out[:, [1, 2]] = out[:, [2, 1]]
    out[:, 2] *= -1.0
    return out


def _flip_normals(vectors):
    out = vectors.copy()
    out[:, [1, 2]] = out[:, [2, 1]]
    out[:, 2] *= -1.0
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return out / norm


def export_collection(collection, filepath, version=0x209, export_textures=True,
                      compress=True, original_bones=True):
    tmd2 = TMD2()
    tmd2.version = version
    tmd2.name = collection.name

    props = props_object(collection)
    if props is not None and hasattr(props, "ctmd2_props"):
        tmd2.flag1 = int(props.ctmd2_props.flag1)
        tmd2.flag2 = int(props.ctmd2_props.flag2)
        tmd2.animFlag = int(props.ctmd2_props.flag3)
        tmd2.afterImageValue = int(props.ctmd2_props.after_image)

    min_corner, max_corner, corners = collection_bbox(collection)
    tmd2.boundingBox = list(min_corner) + list(max_corner)

    armature = None
    meshes = []
    for obj in collection.objects:
        if obj.type == 'ARMATURE' and armature is None:
            armature = obj
        elif obj.type == 'MESH':
            meshes.append(obj)
    print(f"[ChatGPT TMD2] export: {len(meshes)} mesh(es) in {collection.name}")

    if armature is not None:
        _export_bones(tmd2, armature, original=original_bones)

    material_cache = {}
    texture_cache = {}

    image_by_hash = {}

    def get_material(material):
        key = material.name
        if key in material_cache:
            return material_cache[key]
        props = material_props(material)
        tm_mat = TMD2Material()
        if props is not None:
            try:
                tm_mat.hash = int(props.material_hash)
            except (TypeError, ValueError):
                tm_mat.hash = hash_for_name(material.name)
            tm_mat.shaderID = props.shader_id or "BSS1"
            tm_mat.shaderParams = [float(p.value) for p in props.param_values]
            source_textures = list(props.textures)
        else:
            tm_mat.hash = hash_for_name(material.name)
            tm_mat.shaderID = "BSS1"
            tm_mat.shaderParams = []
            source_textures = []
        if tm_mat.hash == 0:
            tm_mat.hash = hash_for_name(material.name)

        tm_mat.textures = []
        for index, tex in enumerate(source_textures):
            image = tex.image
            # Preserve the hash that was stored in the original file: the game
            # looks textures up by hash, not by name.
            image_hash = 0
            try:
                image_hash = int(tex.texture_hash)
            except (TypeError, ValueError):
                image_hash = 0
            if not image_hash and image is not None:
                image_hash = hash_for_name(image.name)
            texture = texture_cache.get(image_hash)
            if texture is None:
                texture = TMD2Texture()
                texture.hash = image_hash
                texture.width, texture.height = image.size if image is not None else (0, 0)
                texture.index = len(texture_cache)
                texture.format = 0x5252
                texture_cache[image_hash] = texture
            mat_texture = TMD2MatTexture()
            mat_texture.texture = texture
            mat_texture.textureHash = image_hash
            mat_texture.textureIndex = texture.index
            mat_texture.unk1 = tex.value1
            mat_texture.unk2 = tex.value2
            mat_texture.slot = tex.slot or index
            tm_mat.textures.append(mat_texture)
            if image is not None:
                image_by_hash[image_hash] = image
        material_cache[key] = tm_mat
        return tm_mat

    vertex_buffers = []
    triangle_buffer = []
    vertex_offset = 0

    for mesh_obj in meshes:
        mesh_data = mesh_obj.data
        if not len(mesh_data.polygons):
            print(f"[ChatGPT TMD2] export: skipping empty mesh {mesh_obj.name}")
            continue
        mesh_props_group = mesh_props(mesh_obj)
        model = TMD2Model()
        model.name = (mesh_props_group.name if mesh_props_group and mesh_props_group.name
                      else mesh_obj.name.split(".")[0])

        if mesh_props_group is not None and mesh_props_group.has_hash:
            model.hashFlag = 1
            try:
                model.hash = int(mesh_props_group.name_hash)
            except (TypeError, ValueError):
                model.hash = hash_for_name(model.name)
        if mesh_props_group is not None and mesh_props_group.has_name:
            model.nameFlag |= 4

        bone_index = {}
        if armature is not None:
            bone_index = {b.name: i for i, b in enumerate(armature.data.bones)}

        loop_data = extract_loop_data(mesh_obj, bone_index, max_influences=8)

        positions = _flip_positions(loop_data["position"])
        normals = _flip_normals(loop_data["custom_normal"])

        # Field names must match what ctamlib.tmd2.TMD2.__br_write__ consumes.
        fields = [("position", "f4", (3,))]
        arrays = {"position": positions}
        flags = 0x1 | 0x2 | 0x1000

        if armature is not None:
            fields += [("weights", "f4", (8,)), ("boneIDs", "u4", (8,))]
            arrays["weights"] = loop_data["weights"][:, :8]
            arrays["boneIDs"] = loop_data["boneIDs"][:, :8].astype(np.uint32)
            flags |= 0x400 | 0x8000

        fields.append(("custom_normal", "f4", (3,)))
        arrays["custom_normal"] = normals
        flags |= 0x4

        if "tangent" in loop_data and "bitangent" in loop_data:
            fields += [("tangent", "f4", (3,)), ("bitangent", "f4", (3,))]
            arrays["tangent"] = _flip_normals(loop_data["tangent"])
            arrays["bitangent"] = _flip_normals(loop_data["bitangent"])
            flags |= 0x8

        fields.append(("real_normal", "f4", (3,)))
        arrays["real_normal"] = normals
        flags |= 0x100

        if loop_data["color_count"] >= 1:
            fields.append(("color0", "f4", (4,)))
            arrays["color0"] = loop_data["color0"]
            flags |= 0x80
        if loop_data["color_count"] >= 2:
            fields.append(("color1", "f4", (4,)))
            arrays["color1"] = loop_data["color1"]
            flags |= 0x200

        if loop_data["uv_count"] >= 1:
            flags |= 0x10
        if loop_data["uv_count"] >= 2:
            flags |= 0x20
        if loop_data["uv_count"] >= 3:
            flags |= 0x40
        for i in range(loop_data["uv_count"]):
            fields.append((f"uv{i}", "f4", (2,)))
            arrays[f"uv{i}"] = loop_data[f"uv{i}"]

        tmd2.modelFlags |= flags

        dtype = np.dtype(fields)
        vertices = np.empty(len(positions), dtype=dtype)
        for key, value in arrays.items():
            vertices[key] = value

        triangles = loop_data["loop_tri_indices"].astype(np.uint32)
        material_indices = loop_data["material_indices"].astype(np.int64)
        vertices, triangles, material_indices = dedupe_vertices(
            vertices, triangles, material_indices)

        order = np.argsort(material_indices, kind="stable")
        tri_materials = material_indices[order]
        triangles = triangles[order]

        model.meshes = []
        boundaries = np.where(np.diff(tri_materials) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        for group, tri_group in enumerate(np.split(triangles, boundaries)):
            if len(tri_group) == 0:
                continue
            slot_index = int(tri_materials[starts[group]]) if group < len(starts) else 0
            if slot_index >= len(mesh_obj.material_slots):
                slot_index = 0
            material = (mesh_obj.material_slots[slot_index].material
                        if len(mesh_obj.material_slots) else None)
            submesh = TMD2Submesh()
            submesh.material = get_material(material) if material else _default_material(get_material)
            submesh.triangles = tri_group.astype(np.uint32)
            submesh.vertexIndices = np.unique(tri_group)
            model.meshes.append(submesh)
            if submesh.material not in model.materials:
                model.materials.append(submesh.material)

        vertex_buffers.append(vertices)
        triangle_buffer.append(triangles)
        vertex_offset += len(vertices)
        tmd2.models.append(model)

    if vertex_buffers:
        tmd2.vertices = np.concatenate(vertex_buffers)
        tmd2.triangles = np.concatenate(triangle_buffer).astype(np.uint32)
        if len(tmd2.vertices) > 0xFFFF:
            tmd2.modelFlags |= 0x800
        # The writer expects a flat triangle list, the submeshes store slices of it
        # through their ``trianglesCount``/``trianglesStart`` fields.
        _prepare_submeshes(tmd2)
    else:
        tmd2.vertices = np.zeros(0, dtype=np.dtype([("position", "f4", (3,))]))
        tmd2.triangles = np.zeros((0, 3), dtype=np.uint32)

    tmd2.materials = list(material_cache.values())
    tmd2.textures = list(texture_cache.values())
    # The reader always parses modelCount + 1 bounding boxes, so the writer has
    # to emit exactly that many.
    for _ in range(len(tmd2.models) + 1):
        bbox = TMD2BoundingBox()
        bbox.corners = [list(c) for c in corners]
        tmd2.BBoxCorners.append(bbox)

    write_tmd2(tmd2, f"{filepath}.tmd2", compress)

    if export_textures and tmd2.textures:
        write_textures_for(tmd2, image_by_hash, f"{filepath}.lds", compress)
    return tmd2


def _default_material(get_material):
    fallback = bpy.data.materials.get("CTMD2_Default")
    if fallback is None:
        fallback = bpy.data.materials.new("CTMD2_Default")
        fallback.use_nodes = True
    material = get_material(fallback)
    if not material.shaderID:
        material.shaderID = "Default"
    return material


def _prepare_submeshes(tmd2):
    """Convert absolute triangle indices in submeshes to slices of the flat list."""
    cursor = 0
    for model in tmd2.models:
        for submesh in model.meshes:
            count = len(submesh.triangles)
            submesh.trianglesCount = count
            submesh.trianglesStart = cursor
            cursor += count
    total = sum(len(s.triangles) for m in tmd2.models for s in m.meshes)
    if total != len(tmd2.triangles):
        print(f"[ChatGPT TMD2] triangle mismatch: {total} vs {len(tmd2.triangles)}")


# ---------------------------------------------------------------------------
# Bones
# ---------------------------------------------------------------------------
def _export_bones(tmd2, armature, original=True):
    tmd2.modelFlags |= 0x2000
    tmd2.bones = []
    bone_indices = {bone.name: i for i, bone in enumerate(armature.data.bones)}
    up_matrix = Matrix.Rotation(radians(90), 4, 'X')
    pose_offset = 0
    for bone in armature.data.bones:
        tmbone = TMD2Bone()
        tmbone.name = bone.name
        tmbone.hash = int(bone.get("hash", tamCRC32(bone.name)))
        stored = bone.get("matrix")
        if original and stored is not None:
            tmbone.matrix = Matrix(stored).inverted().transposed()
        else:
            tmbone.matrix = (up_matrix @ bone.matrix_local).inverted().transposed()

        pose_bone = armature.pose.bones.get(bone.name)
        offset = bone.get("offset")
        extra = bone.get("extra", None)
        if extra is not None and int(extra) > -1:
            tmbone.offset = list(offset) if offset else [0.0, 0.0, 0.0]
            tmbone.extra = pose_offset
            pose_offset += 1
        elif pose_bone is not None and _pose_rotated(pose_bone):
            rotation = _pose_euler(pose_bone)
            tmbone.offset = [rotation.x, rotation.y, rotation.z]
            tmbone.extra = pose_offset
            pose_offset += 1
        else:
            tmbone.extra = -1

        if bone.parent is not None:
            tmbone.parentIndex = bone_indices.get(bone.parent.name, -1)
            local = pose_bone.head - pose_bone.parent.head if pose_bone and pose_bone.parent else Vector()
        else:
            tmbone.parentIndex = -1
            local = pose_bone.head if pose_bone else Vector()
        tmbone.posedLocation = list(up_matrix @ local if local else [0.0, 0.0, 0.0])
        tmd2.bones.append(tmbone)


def _pose_rotated(pose_bone):
    try:
        if pose_bone.rotation_mode == 'QUATERNION':
            return pose_bone.rotation_quaternion != Quaternion()
        return pose_bone.rotation_euler != Euler()
    except Exception:
        return False


def _pose_euler(pose_bone):
    try:
        if pose_bone.rotation_mode == 'QUATERNION':
            return pose_bone.rotation_quaternion.inverted().to_euler()
        return pose_bone.rotation_euler
    except Exception:
        return Euler()


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------
def write_textures_for(tmd2, image_by_hash, path, compress):
    """Write an .lds with the DDS data of the textures, in table order."""
    datas = []
    for texture in tmd2.textures:
        image = image_by_hash.get(texture.hash)
        if image is None:
            datas.append(b"")
            continue
        try:
            if image.packed_file is None:
                image.pack()
            data = bytes(image.packed_file.data)
        except Exception as exc:
            print(f"[ChatGPT TMD2] Could not pack {image.name}: {exc}")
            data = b""
        datas.append(data)
    if not any(datas):
        print("[ChatGPT TMD2] no texture data to write")
        return
    write_lds(datas, path, compress)
    print(f"[ChatGPT TMD2] wrote {len(datas)} textures to {path}")
