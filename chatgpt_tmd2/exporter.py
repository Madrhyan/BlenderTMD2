import os
import time
from math import copysign, radians
from typing import Optional

import bmesh
import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, axis_conversion
from mathutils import Euler, Matrix, Quaternion, Vector

from .reader import readTMD, writeLDS, writeTMD, writeTMD2
from .tamLib.lds import LDS
from .tamLib.tmd import TMD, TMDBone, TMDMaterial, TMDMatTexture, TMDModel, TMDSubmesh, TMDTexture
from .tamLib.tmd2 import (
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


def _strip_known_ext(filepath: str) -> str:
    low = filepath.lower()
    for ext in (".tmd2", ".tmd", ".lds"):
        if low.endswith(ext):
            return filepath[: -len(ext)]
    return filepath


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


class CGTMD2_EXPORTER_OT_EXPORT(Operator, ExportHelper):
    bl_idname = "export_scene.chatgpt_tmd2"
    bl_label = "Export ChatGPT TMD2"
    filename_ext = ""

    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})
    filepath: StringProperty(subtype="FILE_PATH")

    collection: StringProperty(
        name="Collection",
        description="The collection to export as TMD2 / TMD",
    )

    tmd_version: EnumProperty(
        name="Version",
        items=[
            ("0x208", "TMD2 0x208 (Captain Tsubasa / Modern)", "TMD2 v2.08"),
            ("0x209", "TMD2 0x209 (Bleach / Latest)", "TMD2 v2.09"),
            ("0x207", "TMD2 0x207", "TMD2 v2.07"),
            ("0x206", "TMD2 0x206", "TMD2 v2.06"),
            ("0x205", "TMD2 0x205", "TMD2 v2.05"),
            ("0x204", "TMD2 0x204", "TMD2 v2.04"),
            ("0x201", "TMD 0x201 (Legacy TMD)", "TMD v2.01"),
        ],
        default="0x208",
        description="Target TamEngine model version",
    )

    export_textures: BoolProperty(
        name="Export Companion .lds Textures",
        default=True,
        description="Export packed DDS textures into a companion .lds file",
    )

    export_original_bone_data: BoolProperty(
        name="Use Original Bone Data",
        default=True,
        description="Export exact stored bone matrices and posed locations whenever available",
    )

    compress_files: BoolProperty(
        name="PZZE Compress Output",
        default=True,
        description="Compress exported .tmd2 and .lds files in a PZZE (zlib) container",
    )

    inject: BoolProperty(
        name="Inject into Existing Legacy TMD",
        default=False,
        description="Inject exported geometry into an existing legacy .tmd file",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "collection", bpy.data, "collections")
        layout.prop(self, "tmd_version")
        if self.tmd_version == "0x201":
            layout.prop(self, "inject")
        layout.prop(self, "export_textures")
        layout.prop(self, "export_original_bone_data")
        layout.prop(self, "compress_files")

    def invoke(self, context, event):
        if not self.collection:
            if context.collection and context.collection.name in bpy.data.collections:
                self.collection = context.collection.name
            elif len(bpy.data.collections) > 0:
                self.collection = bpy.data.collections[0].name

        # Auto-select version from #CGTMD2 PROPERTIES if present in collection
        coll = bpy.data.collections.get(self.collection)
        if coll is not None:
            for obj in coll.objects:
                if obj.name.startswith("#CGTMD2 PROPERTIES") and hasattr(obj, "chatgpt_tmd2_props"):
                    v_int = _safe_int(obj.chatgpt_tmd2_props.version, 0x208)
                    hex_v = hex(v_int)
                    if hex_v in {"0x209", "0x208", "0x207", "0x206", "0x205", "0x204", "0x201"}:
                        self.tmd_version = hex_v
                    break

        if not self.filepath:
            self.filepath = self.collection if self.collection else "untitled"

        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        start_time = time.time()
        collection = bpy.data.collections.get(self.collection)
        if collection is None:
            self.report({"ERROR"}, f"Collection '{self.collection}' not found.")
            return {"CANCELLED"}

        base_out = _strip_known_ext(self.filepath)
        ver = int(self.tmd_version, 16)

        if ver > 0x201:
            self.tmd = TMD2()
            self.tmd.version = ver
            self.export_tmd2(collection, base_out)
        else:
            self.tmd = TMD()
            self.tmd.version = ver
            if self.inject and os.path.isfile(base_out + ".tmd"):
                existing_tmd = readTMD(base_out + ".tmd")
                self.tmd = existing_tmd
                self.tmd.models = []
                self.export_tmd(collection, base_out, self.tmd)
            else:
                self.tmd.TransformationFramesCount = 0
                self.export_tmd(collection, base_out)

        self.report({"INFO"}, f"ChatGPT TMD2 export completed in {time.time() - start_time:.2f}s")
        return {"FINISHED"}

    def export_tmd2(self, collection: bpy.types.Collection, base_out: str):
        ZUP_TO_YUP = Matrix.Rotation(radians(-90), 4, "X")

        props_obj = None
        legacy_props_obj = None
        for obj in collection.objects:
            if obj.name.startswith("#CGTMD2 PROPERTIES"):
                props_obj = obj
                break
            if "#TMD PROPERTIES" in obj.name:
                legacy_props_obj = obj

        if props_obj is not None and hasattr(props_obj, "chatgpt_tmd2_props"):
            p = props_obj.chatgpt_tmd2_props
            self.tmd.flag1 = int(p.flag1)
            self.tmd.flag2 = int(p.flag2)
            self.tmd.animFlag = int(p.flag3)
            self.tmd.afterImageValue = int(p.after_image)
        elif legacy_props_obj is not None and hasattr(legacy_props_obj, "tmd2_props"):
            lp = legacy_props_obj.tmd2_props
            self.tmd.flag1 = int(lp.flag1)
            self.tmd.flag2 = int(lp.flag2)
            self.tmd.animFlag = int(lp.flag3)
            self.tmd.afterImageValue = int(lp.after_image)
        else:
            self.tmd.flag1 = 0
            self.tmd.flag2 = 0
            self.tmd.animFlag = 0
            self.tmd.afterImageValue = 0

        bbox_res = get_collection_bbox(collection)
        if bbox_res is None:
            raise RuntimeError(f"Collection '{collection.name}' contains no mesh geometry to export.")
        bbmin, bbmax, bbcorners = bbox_res
        self.tmd.boundingBox = list(bbmin) + list(bbmax)
        bboxcorners = TMD2BoundingBox()
        bboxcorners.corners = bbcorners
        self.tmd.BBoxCorners.append(bboxcorners)

        armature = None
        meshes = []
        for obj in collection.objects:
            if obj.type == "ARMATURE":
                if armature is not None:
                    raise RuntimeError("A TMD2 collection can only contain 1 Armature.")
                armature = obj
            elif obj.type == "MESH" and len(obj.data.polygons) > 0:
                meshes.append(obj)

        if armature is not None:
            if self.export_original_bone_data and all("matrix" in b for b in armature.data.bones):
                self.make_bones_original(armature, ZUP_TO_YUP)
            else:
                self.make_bones(armature, ZUP_TO_YUP)

        materials = {}
        textures_by_key = {}
        textures_list = []

        def get_or_create_material(material: Optional[bpy.types.Material]) -> TMD2Material:
            mat_key = material.name if material is not None else "__default__"
            if mat_key in materials:
                return materials[mat_key]

            tmd2_mat = TMD2Material()
            if material is None:
                tmd2_mat.hash = tamCRC32("Default")
                tmd2_mat.shaderID = "BSS1"
                tmd2_mat.shaderParams = [1.0, 1.0, 1.0, 1.0]
                tmd2_mat.shaderParamMetas = [0, 0, 0, 0]
                materials[mat_key] = tmd2_mat
                return tmd2_mat

            mat_props = getattr(material, "chatgpt_tmd2_material", None)
            if mat_props is not None and (len(mat_props.textures) > 0 or len(mat_props.param_values) > 0 or mat_props.material_hash != "0"):
                mhash = _safe_int(mat_props.material_hash, 0)
                tmd2_mat.hash = mhash if mhash != 0 else tamCRC32(material.name.split(".")[0])
                sid = (mat_props.shader_id or "BSS1")[:4]
                tmd2_mat.shaderID = sid if sid != "Defa" else "BSS1"
                tmd2_mat.shaderParams = [float(p.value) for p in mat_props.param_values]
                tmd2_mat.shaderParamMetas = [int(p.meta) for p in mat_props.param_values]
                tmd2_mat.unk = int(mat_props.unk)
                tmd2_mat.textures = []

                for i, tex in enumerate(mat_props.textures):
                    thash = _safe_int(tex.texture_hash, 0)
                    if thash == 0 and tex.image is not None:
                        thash = tamCRC32(tex.image.name.rsplit(".", 1)[0])

                    t2mattex = TMD2MatTexture()
                    t2mattex.textureHash = thash
                    t2mattex.unk1 = int(tex.value1)
                    t2mattex.unk2 = int(tex.value2)
                    t2mattex.unk3 = int(tex.value3) & 0xFF
                    t2mattex.slot = int(tex.slot if tex.slot >= 0 else i) & 0xFF
                    t2mattex.isExternal = bool(tex.is_external)

                    if tex.is_external:
                        ext_tex = TMD2Texture()
                        ext_tex.hash = thash
                        ext_tex.index = -1
                        t2mattex.texture = ext_tex
                        t2mattex.textureIndex = -1
                    else:
                        dedup_key = (thash, tex.image.name if tex.image else f"none_{i}")
                        t2tex = textures_by_key.get(dedup_key)
                        if t2tex is None:
                            t2tex = TMD2Texture()
                            t2tex.hash = thash
                            t2tex.width, t2tex.height = tex.image.size if tex.image else (0, 0)
                            t2tex.data = b""
                            if tex.image is not None and self.export_textures:
                                try:
                                    if tex.image.packed_file is None:
                                        tex.image.pack()
                                    if tex.image.packed_file is not None:
                                        t2tex.data = tex.image.packed_file.data
                                except Exception:
                                    t2tex.data = b""
                            t2tex.index = len(textures_list)
                            t2tex.format = 0x5252
                            textures_by_key[dedup_key] = t2tex
                            textures_list.append(t2tex)
                        t2mattex.texture = t2tex
                        t2mattex.textureIndex = t2tex.index

                    tmd2_mat.textures.append(t2mattex)
            else:
                # Fallback for freshly-created Blender materials without ChatGPT TMD2 metadata
                tmd2_mat.hash = tamCRC32(material.name.split(".")[0])
                tmd2_mat.shaderID = "BSS1"
                tmd2_mat.shaderParams = [1.0, 1.0, 1.0, 1.0]
                tmd2_mat.shaderParamMetas = [0, 0, 0, 0]
                if material.use_nodes and material.node_tree:
                    img_nodes = [n for n in material.node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None]
                    for slot_i, inode in enumerate(img_nodes):
                        img = inode.image
                        thash = tamCRC32(img.name.rsplit(".", 1)[0])
                        dedup_key = (thash, img.name)
                        t2tex = textures_by_key.get(dedup_key)
                        if t2tex is None:
                            t2tex = TMD2Texture()
                            t2tex.hash = thash
                            t2tex.width, t2tex.height = img.size
                            t2tex.data = b""
                            if self.export_textures:
                                try:
                                    if img.packed_file is None:
                                        img.pack()
                                    if img.packed_file is not None:
                                        t2tex.data = img.packed_file.data
                                except Exception:
                                    pass
                            t2tex.index = len(textures_list)
                            t2tex.format = 0x5252
                            textures_by_key[dedup_key] = t2tex
                            textures_list.append(t2tex)
                        mt = TMD2MatTexture()
                        mt.texture = t2tex
                        mt.textureIndex = t2tex.index
                        mt.textureHash = thash
                        mt.slot = slot_i
                        tmd2_mat.textures.append(mt)

            materials[mat_key] = tmd2_mat
            return tmd2_mat

        global_vertex_buffer = []
        global_face_buffer = []
        global_vertex_count = 0

        # Pre-scan max UVs & colors across all meshes so global vertex struct matches every mesh!
        max_uv_layers = max((min(3, len(m.data.uv_layers)) for m in meshes), default=0)
        max_col_layers = max((min(2, len(m.data.color_attributes)) for m in meshes), default=0)
        has_any_vgroups = any(len(m.vertex_groups) > 0 for m in meshes) and armature is not None

        self.tmd.modelFlags |= 0x1 | 0x2 | 0x4 | 0x8 | 0x100 | 0x1000
        if max_uv_layers >= 1:
            self.tmd.modelFlags |= 0x10
        if max_uv_layers >= 2:
            self.tmd.modelFlags |= 0x20
        if max_uv_layers >= 3:
            self.tmd.modelFlags |= 0x40
        if max_col_layers >= 1:
            self.tmd.modelFlags |= 0x80
        if max_col_layers >= 2:
            self.tmd.modelFlags |= 0x200
        if has_any_vgroups:
            self.tmd.modelFlags |= 0x400 | 0x8000

        bone_map = {b.name: i for i, b in enumerate(armature.data.bones)} if armature is not None else {}

        for mesh_obj in meshes:
            model = TMD2Model()
            clean_obj_name = mesh_obj.name.split(".")[0]
            model.name = clean_obj_name
            mesh_props = getattr(mesh_obj, "chatgpt_tmd2_mesh", None)

            if mesh_props is not None:
                if mesh_props.has_name:
                    model.nameFlag = 4
                    if mesh_props.name:
                        model.name = mesh_props.name
                if mesh_props.has_extra:
                    model.nameFlag = 3
                    model.unk1 = int(mesh_props.unk1)
                    model.unk2 = int(mesh_props.unk2)
                    model.unk3 = int(mesh_props.unk3)
                if mesh_props.has_hash:
                    model.hashFlag = 1
                    stored_h = _safe_int(mesh_props.name_hash, 0)
                    model.hash = stored_h if stored_h != 0 else tamCRC32(clean_obj_name)
                else:
                    model.hashFlag = 0
                    model.hash = 0
                model.lhsFlag = 1 if mesh_props.lhs_flag else 0

            world_bbox = [ZUP_TO_YUP @ (mesh_obj.matrix_world @ Vector(corner)) for corner in mesh_obj.bound_box]
            min_corner = Vector((min(v[i] for v in world_bbox) for i in range(3)))
            max_corner = Vector((max(v[i] for v in world_bbox) for i in range(3)))
            model.boundingBox = list(min_corner) + list(max_corner)
            bboxcorners = TMD2BoundingBox()
            bboxcorners.corners = [list(world_bbox[i]) for i in [6, 2, 7, 3, 4, 0, 5, 1]]
            self.tmd.BBoxCorners.append(bboxcorners)

            loop_data = extract_loop_data_numpy(
                mesh_obj,
                bone_map,
                max_influences=8,
                target_uv_layers=max_uv_layers,
                target_col_layers=max_col_layers,
            )

            vertex_data = [loop_data["position"]]
            vertex_attributes = ["position"]
            if self.tmd.modelFlags & 0x400:
                vertex_data.append(loop_data["weights"])
                vertex_attributes.append("weights")
                vertex_data.append(loop_data["boneIDs"])
                vertex_attributes.append("boneIDs")
            vertex_data.append(loop_data["custom_normal"])
            vertex_attributes.append("custom_normal")
            vertex_data.extend([loop_data["tangent"], loop_data["tangent_sign"]])
            vertex_attributes.extend(["tangent", "tangent_sign"])
            vertex_data.extend([loop_data["bitangent"], loop_data["bitangent_sign"]])
            vertex_attributes.extend(["bitangent", "bitangent_sign"])
            if max_col_layers >= 1:
                vertex_data.append(loop_data["color0"])
                vertex_attributes.append("color0")
            vertex_data.append(loop_data["real_normal"])
            vertex_attributes.append("real_normal")
            if max_col_layers >= 2:
                vertex_data.append(loop_data["color1"])
                vertex_attributes.append("color1")
            for u_idx in range(max_uv_layers):
                vertex_data.append(loop_data[f"uv{u_idx}"])
                vertex_attributes.append(f"uv{u_idx}")

            fields = [(name, data.dtype, data.shape[1:]) for name, data in zip(vertex_attributes, vertex_data)]
            vertex_array = np.empty(len(loop_data["position"]), dtype=np.dtype(fields))
            for name, data in zip(vertex_attributes, vertex_data):
                vertex_array[name] = data

            vertex_array, triangles, material_indices = dedupe_vertices_structured(
                vertex_array, loop_data["loop_tri_indices"], loop_data["material_indices"]
            )

            sorted_order = np.argsort(material_indices, kind="stable")
            sorted_materials_indices = material_indices[sorted_order]
            sorted_triangles = triangles[sorted_order] + global_vertex_count

            global_vertex_buffer.append(vertex_array)
            global_vertex_count += len(vertex_array)
            global_face_buffer.extend(sorted_triangles)

            change_indices = np.where(np.diff(sorted_materials_indices) != 0)[0] + 1
            material_boundaries = np.split(sorted_triangles, change_indices)
            slot_groups = np.split(sorted_materials_indices, change_indices)

            submeshes = []
            for slot_grp, tri_group in zip(slot_groups, material_boundaries):
                if len(tri_group) == 0:
                    continue
                submesh = TMD2Submesh()
                slot_idx = int(slot_grp[0])
                bmat = (
                    mesh_obj.material_slots[slot_idx].material
                    if 0 <= slot_idx < len(mesh_obj.material_slots)
                    else None
                )
                submesh.material = get_or_create_material(bmat)
                submesh.triangles = tri_group
                submeshes.append(submesh)

            model.meshes = submeshes
            self.tmd.models.append(model)

        self.tmd.materials = list(materials.values())
        self.tmd.textures = textures_list
        self.tmd.vertices = np.concatenate(global_vertex_buffer)
        if len(self.tmd.vertices) > 0xFFFF:
            self.tmd.modelFlags |= 0x800
        self.tmd.triangles = np.array(global_face_buffer, dtype=np.uint32)

        writeTMD2(self.tmd, f"{base_out}.tmd2", self.compress_files)

        if self.export_textures and textures_list:
            lds = LDS()
            lds.textures = [tex.data for tex in textures_list]
            writeLDS(lds, f"{base_out}.lds", self.compress_files)

    def make_bones(self, armature, ZUP_TO_YUP):
        pose_values = []
        self.tmd.modelFlags |= 0x2000
        bone_indices = {b.name: i for i, b in enumerate(armature.data.bones)}
        up_matrix = axis_conversion(from_forward="Y", from_up="Z", to_forward="-Z", to_up="Y").to_4x4()

        for bone in armature.data.bones:
            tmbone = TMD2Bone()
            tmbone.name = bone.name
            tmbone.hash = _safe_int(bone.get("hash"), tamCRC32(bone.name))
            tmbone.matrix = (up_matrix @ bone.matrix_local).inverted().transposed()

            pbone = armature.pose.bones[bone.name]
            if pbone.rotation_quaternion != Quaternion():
                rot = pbone.rotation_quaternion.inverted().to_euler()
                tmbone.offset = [rot.x, rot.y, rot.z]
                tmbone.extra = len(pose_values)
                pose_values.append(tmbone.offset)
            elif pbone.rotation_euler != Euler():
                rot = pbone.rotation_euler
                tmbone.offset = [rot.x, rot.y, rot.z]
                tmbone.extra = len(pose_values)
                pose_values.append(tmbone.offset)
            else:
                tmbone.extra = -1

            if bone.parent:
                tmbone.parentIndex = bone_indices[bone.parent.name]
                local_pos = pbone.head - pbone.parent.head
            else:
                tmbone.parentIndex = -1
                local_pos = pbone.head

            tmbone.posedLocation = list(up_matrix @ local_pos)
            self.tmd.bones.append(tmbone)

    def make_bones_original(self, armature, ZUP_TO_YUP):
        pose_index = 0
        self.tmd.modelFlags |= 0x2000
        bone_indices = {b.name: i for i, b in enumerate(armature.data.bones)}
        for bone in armature.data.bones:
            tmbone = TMD2Bone()
            tmbone.name = bone.name
            tmbone.hash = _safe_int(bone.get("hash"), tamCRC32(bone.name))
            tmbone.matrix = Matrix(bone["matrix"]).inverted().transposed()

            pbone = armature.pose.bones[bone.name]
            if bone.get("extra") is not None and int(bone.get("extra")) > -1:
                tmbone.offset = list(bone["offset"])
                tmbone.extra = pose_index
                pose_index += 1
            else:
                tmbone.extra = -1

            if bone.parent:
                tmbone.parentIndex = bone_indices[bone.parent.name]
                local_pos = pbone.head - pbone.parent.head
            else:
                tmbone.parentIndex = -1
                local_pos = pbone.head

            posed = bone.get("posed_loc", bone.get("posedLoc"))
            if posed is not None:
                tmbone.posedLocation = list(posed)
            else:
                tmbone.posedLocation = list(ZUP_TO_YUP @ local_pos)

            if bone.get("unk1") is not None:
                tmbone.unk1 = int(bone["unk1"])
            self.tmd.bones.append(tmbone)

    def make_bones_original_TMD(self, armature, ZUP_TO_YUP):
        pose_index = 0
        self.tmd.modelFlags |= 0x2000
        bone_indices = {b.name: i for i, b in enumerate(armature.data.bones)}
        for bone in armature.data.bones:
            tmbone = TMDBone()
            tmbone.name = bone.name
            tmbone.hash = _safe_int(bone.get("hash"), tamCRC32(bone.name))
            if "matrix" in bone:
                tmbone.matrix = Matrix(bone["matrix"]).inverted().transposed()
            else:
                tmbone.matrix = (ZUP_TO_YUP @ bone.matrix_local).inverted().transposed()

            pbone = armature.pose.bones[bone.name]
            if bone.get("extra") is not None and int(bone.get("extra")) > -1:
                tmbone.offset = list(bone["offset"])
                tmbone.extra = pose_index
                pose_index += 1
            else:
                tmbone.extra = -1

            if bone.parent:
                tmbone.parentIndex = bone_indices[bone.parent.name]
                local_pos = pbone.head - pbone.parent.head
            else:
                tmbone.parentIndex = -1
                local_pos = pbone.head

            posed = bone.get("posed_loc", bone.get("posedLoc"))
            tmbone.posedLocation = list(posed) if posed is not None else list(ZUP_TO_YUP @ local_pos)
            self.tmd.bones.append(tmbone)

    def export_tmd(self, collection, base_out: str, tmd=None):
        ZUP_TO_YUP = Matrix.Rotation(radians(-90), 4, "X")
        bbox_res = get_collection_bbox(collection)
        if bbox_res is None:
            raise RuntimeError(f"Collection '{collection.name}' contains no mesh geometry.")
        bbmin, bbmax, _ = bbox_res
        self.tmd.boundingBox = list(bbmin) + list(bbmax)

        if not tmd:
            self.tmd.animFlag = 0
            self.tmd.TransformationFramesCount = 0
            self.tmd.TransformationFrames = []

        armature = None
        meshes = []
        for obj in collection.objects:
            if obj.type == "ARMATURE":
                armature = obj
            elif obj.type == "MESH" and len(obj.data.polygons) > 0:
                meshes.append(obj)

        if armature and not tmd:
            self.make_bones_original_TMD(armature, ZUP_TO_YUP)

        materials = {}
        textures = {}
        material_textures = list(tmd.materialTextures) if tmd else []

        def get_or_create_material(material):
            key = material.name if material else "__default__"
            if key in materials:
                return materials[key]
            tmd_mat = TMDMaterial()
            if material is None:
                tmd_mat.hash = tamCRC32("Default")
                tmd_mat.shaderID = "BSS1"
                tmd_mat.shaderParams = [1.0, 1.0, 1.0, 1.0]
                materials[key] = tmd_mat
                return tmd_mat

            mp = material.chatgpt_tmd2_material
            tmd_mat.hash = _safe_int(mp.material_hash, tamCRC32(material.name.split(".")[0]))
            tmd_mat.shaderID = (mp.shader_id or "BSS1")[:4]
            tmd_mat.shaderParams = [float(p.value) for p in mp.param_values]
            tmd_mat.textures = []
            for i, tex in enumerate(mp.textures):
                thash = _safe_int(tex.texture_hash, tamCRC32(tex.image.name.rsplit(".", 1)[0]) if tex.image else 0)
                mt = TMDMatTexture()
                mt.textureHash = thash
                mt.unk1 = int(tex.value1)
                mt.unk2 = int(tex.value2)
                mt.unk3 = int(tex.value3) & 0xFF
                mt.slot = int(tex.slot if tex.slot >= 0 else i) & 0xFF
                mt.isExternal = bool(tex.is_external)
                if tex.is_external:
                    et = TMDTexture()
                    et.hash = thash
                    et.index = -1
                    mt.texture = et
                else:
                    t2tex = textures.get(thash)
                    if t2tex is None:
                        t2tex = TMDTexture()
                        t2tex.hash = thash
                        t2tex.width, t2tex.height = tex.image.size if tex.image else (0, 0)
                        t2tex.index = len(textures)
                        t2tex.format = 0x5252
                        textures[thash] = t2tex
                    mt.texture = t2tex
                tmd_mat.textures.append(mt)
                material_textures.append(mt)
            materials[key] = tmd_mat
            return tmd_mat

        max_uv_layers = max((min(3, len(m.data.uv_layers)) for m in meshes), default=0)
        max_col_layers = max((min(2, len(m.data.color_attributes)) for m in meshes), default=0)
        has_any_vgroups = any(len(m.vertex_groups) > 0 for m in meshes) and armature is not None

        self.tmd.modelFlags |= 0x1 | 0x2 | 0x4 | 0x8 | 0x1000 | 0x4000
        if max_uv_layers >= 1:
            self.tmd.modelFlags |= 0x10
        if max_uv_layers >= 2:
            self.tmd.modelFlags |= 0x20
        if max_uv_layers >= 3:
            self.tmd.modelFlags |= 0x40
        if max_col_layers >= 1:
            self.tmd.modelFlags |= 0x80
        if max_col_layers >= 2:
            self.tmd.modelFlags |= 0x200
        if has_any_vgroups:
            self.tmd.modelFlags |= 0x400

        bone_map = {b.name: i for i, b in enumerate(armature.data.bones)} if armature else {}
        global_vertex_buffer = []
        global_face_buffer = []
        global_vertex_count = 0

        for mesh_obj in meshes:
            model = TMDModel()
            mp = mesh_obj.chatgpt_tmd2_mesh
            model.name = mp.name if (mp.has_name and mp.name) else mesh_obj.name.split(".")[0]
            if mp.has_hash:
                model.hashFlag = 1
                model.hash = _safe_int(mp.name_hash, tamCRC32(model.name))
            if mp.has_name:
                model.nameFlag = 4
            if mp.has_extra:
                model.nameFlag = 3
                model.unk1, model.unk2, model.unk3 = int(mp.unk1), int(mp.unk2), int(mp.unk3)

            world_bbox = [ZUP_TO_YUP @ (mesh_obj.matrix_world @ Vector(corner)) for corner in mesh_obj.bound_box]
            model.boundingBox = [min(v[i] for v in world_bbox) for i in range(3)] + [
                max(v[i] for v in world_bbox) for i in range(3)
            ]

            loop_data = extract_loop_data_numpy(
                mesh_obj, bone_map, max_influences=4, target_uv_layers=max_uv_layers, target_col_layers=max_col_layers
            )
            vertex_data = [loop_data["position"]]
            vertex_attributes = ["position"]
            if self.tmd.modelFlags & 0x400:
                vertex_data.extend([loop_data["weights"], loop_data["boneIDs"]])
                vertex_attributes.extend(["weights", "boneIDs"])
            vertex_data.extend(
                [
                    loop_data["custom_normal"],
                    loop_data["tangent"],
                    loop_data["tangent_sign"],
                    loop_data["bitangent"],
                    loop_data["bitangent_sign"],
                ]
            )
            vertex_attributes.extend(["custom_normal", "tangent", "tangent_sign", "bitangent", "bitangent_sign"])
            if max_col_layers >= 1:
                vertex_data.append(loop_data["color0"])
                vertex_attributes.append("color0")
            vertex_data.append(loop_data["real_normal"])
            vertex_attributes.append("real_normal")
            if max_col_layers >= 2:
                vertex_data.append(loop_data["color1"])
                vertex_attributes.append("color1")
            for u_idx in range(max_uv_layers):
                vertex_data.append(loop_data[f"uv{u_idx}"])
                vertex_attributes.append(f"uv{u_idx}")

            fields = [(n, d.dtype, d.shape[1:]) for n, d in zip(vertex_attributes, vertex_data)]
            vertex_array = np.empty(len(loop_data["position"]), dtype=np.dtype(fields))
            for n, d in zip(vertex_attributes, vertex_data):
                vertex_array[n] = d

            vertex_array, triangles, material_indices = dedupe_vertices_structured(
                vertex_array, loop_data["loop_tri_indices"], loop_data["material_indices"]
            )
            sorted_order = np.argsort(material_indices, kind="stable")
            sorted_mat_idx = material_indices[sorted_order]
            sorted_tris = triangles[sorted_order] + global_vertex_count

            global_vertex_buffer.append(vertex_array)
            global_vertex_count += len(vertex_array)
            global_face_buffer.extend(sorted_tris)

            changes = np.where(np.diff(sorted_mat_idx) != 0)[0] + 1
            tri_groups = np.split(sorted_tris, changes)
            slot_groups = np.split(sorted_mat_idx, changes)

            submeshes = []
            for sgrp, tgrp in zip(slot_groups, tri_groups):
                if len(tgrp) == 0:
                    continue
                sub = TMDSubmesh()
                sidx = int(sgrp[0])
                bmat = mesh_obj.material_slots[sidx].material if 0 <= sidx < len(mesh_obj.material_slots) else None
                sub.material = get_or_create_material(bmat)
                sub.triangles = tgrp
                submeshes.append(sub)
            model.meshes = submeshes
            self.tmd.models.append(model)

        if not self.inject:
            self.tmd.materials = list(materials.values())
            self.tmd.textures = list(textures.values())
            self.tmd.materialTextures = material_textures

        self.tmd.vertices = np.concatenate(global_vertex_buffer)
        if len(self.tmd.vertices) > 0xFFFF:
            self.tmd.modelFlags |= 0x800
        self.tmd.triangles = np.array(global_face_buffer, dtype=np.uint32)
        writeTMD(self.tmd, f"{base_out}.tmd")


def extract_loop_data_numpy(
    mesh_obj,
    bone_name_to_index,
    max_influences: int = 4,
    target_uv_layers: int = 0,
    target_col_layers: int = 0,
):
    src_mesh = mesh_obj.data
    work_mesh = src_mesh.copy()

    # Triangulate if any polygon has > 3 vertices so loop count, normals, and tangents stay 100% consistent
    has_non_tris = any(len(p.vertices) != 3 for p in work_mesh.polygons)
    if has_non_tris:
        bm = bmesh.new()
        bm.from_mesh(work_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(work_mesh)
        bm.free()

    work_mesh.calc_loop_triangles()

    # Create a clean copy without custom split normals to read geometric ("real") normals
    real_mesh = work_mesh.copy()
    if real_mesh.attributes.get("custom_normal"):
        real_mesh.attributes.remove(real_mesh.attributes["custom_normal"])

    num_loops = len(work_mesh.loops)
    num_verts = len(work_mesh.vertices)
    num_tris = len(work_mesh.loop_triangles)

    vertex_indices = np.empty(num_loops, dtype=np.int32)
    work_mesh.loops.foreach_get("vertex_index", vertex_indices)

    # UV Layers (padded to target_uv_layers if another mesh in collection had more UV maps)
    uv_layers = work_mesh.uv_layers
    uv_arrays = []
    for i in range(target_uv_layers):
        uv_array = np.zeros((num_loops, 2), dtype=np.float32)
        if i < len(uv_layers):
            uv_layers[i].data.foreach_get("uv", uv_array.ravel())
            uv_array[:, 1] = 1.0 - uv_array[:, 1]
        uv_arrays.append(uv_array)

    # Color Layers (supports both CORNER and POINT domains!)
    color_layers = work_mesh.color_attributes
    color_arrays = []
    for i in range(target_col_layers):
        c_arr = np.ones((num_loops, 4), dtype=np.float32)
        if i < len(color_layers):
            ca = color_layers[i]
            prop_name = "color_srgb" if hasattr(ca.data[0], "color_srgb") else "color"
            if ca.domain == "CORNER":
                ca.data.foreach_get(prop_name, c_arr.ravel())
            elif ca.domain == "POINT":
                pt_arr = np.ones((num_verts, 4), dtype=np.float32)
                ca.data.foreach_get(prop_name, pt_arr.ravel())
                c_arr = pt_arr[vertex_indices]
        color_arrays.append(c_arr)

    # Positions (Z-up -> TamEngine Y-up)
    vert_positions = np.empty((num_verts, 3), dtype=np.float32)
    work_mesh.vertices.foreach_get("co", vert_positions.ravel())
    vert_positions = vert_positions[:, [0, 2, 1]]
    vert_positions[:, 2] *= -1.0
    positions = vert_positions[vertex_indices]

    # Custom Normals & Geometric Real Normals
    custom_normals = np.empty((num_loops, 3), dtype=np.float32)
    work_mesh.corner_normals.foreach_get("vector", custom_normals.ravel())
    custom_normals = custom_normals[:, [0, 2, 1]]
    custom_normals[:, 2] *= -1.0

    real_normals = np.empty((num_loops, 3), dtype=np.float32)
    real_mesh.corner_normals.foreach_get("vector", real_normals.ravel())
    real_normals = real_normals[:, [0, 2, 1]]
    real_normals[:, 2] *= -1.0

    # Tangents & Bitangents (safe even when mesh has 0 UV maps!)
    tangents = np.zeros((num_loops, 3), dtype=np.float32)
    tangent_signs = np.ones(num_loops, dtype=np.float32)
    bitangents = np.zeros((num_loops, 3), dtype=np.float32)
    bitangent_signs = np.ones(num_loops, dtype=np.float32)

    if len(work_mesh.uv_layers) > 0:
        try:
            work_mesh.calc_tangents()
            work_mesh.loops.foreach_get("tangent", tangents.ravel())
            tangents = tangents[:, [0, 2, 1]]
            tangents[:, 2] *= -1.0
            work_mesh.loops.foreach_get("bitangent_sign", tangent_signs)

            work_mesh.loops.foreach_get("bitangent", bitangents.ravel())
            bitangents = bitangents[:, [0, 2, 1]]
            bitangents[:, 2] *= -1.0
            bitangent_signs[:] = tangent_signs
        except RuntimeError:
            tangents[:, 0] = 1.0
            bitangents[:, 1] = 1.0
    else:
        tangents[:, 0] = 1.0
        bitangents[:, 1] = 1.0

    # Bone Weights
    vertex_groups = mesh_obj.vertex_groups
    bone_ids = np.zeros((num_loops, max_influences), dtype=np.uint16)
    weights = np.zeros((num_loops, max_influences), dtype=np.float32)

    if bone_name_to_index:
        v_bone_ids = np.zeros((num_verts, max_influences), dtype=np.uint16)
        v_weights = np.zeros((num_verts, max_influences), dtype=np.float32)
        for v in work_mesh.vertices:
            vw = []
            for g in v.groups:
                if g.group < len(vertex_groups):
                    gname = vertex_groups[g.group].name
                    if gname in bone_name_to_index and g.weight > 0.0:
                        vw.append((bone_name_to_index[gname], g.weight))
            if vw:
                vw.sort(key=lambda x: x[1], reverse=True)
                vw = vw[:max_influences]
                for idx_i, (bid, wval) in enumerate(vw):
                    v_bone_ids[v.index, idx_i] = bid
                    v_weights[v.index, idx_i] = wval
                total = v_weights[v.index].sum()
                if total > 0.0:
                    v_weights[v.index] /= total
                else:
                    v_weights[v.index, 0] = 1.0
            else:
                v_weights[v.index, 0] = 1.0
        bone_ids[:] = v_bone_ids[vertex_indices]
        weights[:] = v_weights[vertex_indices]
    else:
        weights[:, 0] = 1.0

    loop_tri_indices = np.empty((num_tris, 3), dtype=np.uint32)
    material_indices = np.empty(num_tris, dtype=np.uint8)
    work_mesh.loop_triangles.foreach_get("loops", loop_tri_indices.ravel())
    work_mesh.loop_triangles.foreach_get("material_index", material_indices)

    bpy.data.meshes.remove(real_mesh)
    bpy.data.meshes.remove(work_mesh)

    mesh_dict = {
        "position": positions,
        "custom_normal": custom_normals,
        "real_normal": real_normals,
        "tangent": tangents,
        "tangent_sign": tangent_signs,
        "bitangent": bitangents,
        "bitangent_sign": bitangent_signs,
        "boneIDs": bone_ids,
        "weights": weights,
        "loop_tri_indices": loop_tri_indices,
        "material_indices": material_indices,
    }
    for i, uv_arr in enumerate(uv_arrays):
        mesh_dict[f"uv{i}"] = uv_arr
    for i, col_arr in enumerate(color_arrays):
        mesh_dict[f"color{i}"] = col_arr
    return mesh_dict


def dedupe_vertices_structured(verts, tri_indices, material_indices):
    raw = verts.view(np.void)
    _, unique_index, inverse_map = np.unique(raw, return_index=True, return_inverse=True)
    return verts[unique_index], inverse_map[tri_indices], material_indices


def get_collection_bbox(collection):
    ZUP_TO_YUP = Matrix.Rotation(radians(-90), 4, "X")
    all_world_points = []
    for obj in collection.objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            all_world_points.append(ZUP_TO_YUP @ (obj.matrix_world @ Vector(corner)))
    if not all_world_points:
        return None
    min_c = Vector((min(v[i] for v in all_world_points) for i in range(3)))
    max_c = Vector((max(v[i] for v in all_world_points) for i in range(3)))
    corners = [
        (max_c.x, max_c.y, min_c.z),
        (min_c.x, max_c.y, min_c.z),
        (max_c.x, max_c.y, max_c.z),
        (min_c.x, max_c.y, max_c.z),
        (max_c.x, min_c.y, min_c.z),
        (min_c.x, min_c.y, min_c.z),
        (max_c.x, min_c.y, max_c.z),
        (min_c.x, min_c.y, max_c.z),
    ]
    return min_c, max_c, corners


def menu_func_export(self, context):
    self.layout.operator(
        CGTMD2_EXPORTER_OT_EXPORT.bl_idname,
        text="ChatGPT TMD2 (.tmd2, .tmd)",
        icon="EXPORT",
    )


EXPORTER_CLASSES = (CGTMD2_EXPORTER_OT_EXPORT,)


def register():
    for cls in EXPORTER_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(EXPORTER_CLASSES):
        bpy.utils.unregister_class(cls)
