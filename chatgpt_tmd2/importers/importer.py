"""Importers for ChatGPT TMD2.

All operators, collections and helper objects are prefixed with ``ctmd2`` /
``#CTMD2`` so this add-on can be enabled side by side with the original
``BlenderTMD2`` add-on.
"""

import json
import os
import time

import bpy
import numpy as np
from bpy.props import BoolProperty, CollectionProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector, Euler
from math import radians

from ..ctamlib.tmd2 import tamCRC32
from ..core import material_builder
from ..core.mesh_builder import build_mesh
from ..core.textures import (
    IMAGE_PREFIX,
    TextureResolver,
    find_texture_file,
    load_lds_images,
    placeholder_image,
)
from .reader import readTMD2, readTMD, readTMD0, readTMDV


_HASHES = {}


def hashes():
    """Lazy loader for the shared name<->hash table."""
    if not _HASHES:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "hashes.json")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                _HASHES.update(json.load(handle))
        except Exception as exc:  # pragma: no cover
            print(f"[ChatGPT TMD2] Could not read hashes.json: {exc}")
    return _HASHES


def _material_name(tm_mat, fallback):
    try:
        name = hashes().get(str(getattr(tm_mat, "hash", 0)))
    except Exception:
        name = None
    return name or fallback


# ---------------------------------------------------------------------------
# Shared material creation
# ---------------------------------------------------------------------------
def build_materials(tmd, images, settings, color_layers, report=False):
    """Create a Blender material for every material of ``tmd``.

    Returns a dict keyed by ``id(raw_material)``.
    """
    texture_table = list(getattr(tmd, "textures", []) or [])
    resolver = images if isinstance(images, TextureResolver) else TextureResolver(images, texture_table)

    materials = {}
    used_names = set()
    for i, tm_mat in enumerate(getattr(tmd, "materials", []) or []):
        base = _material_name(tm_mat, f"{getattr(tmd, 'name', 'TMD2')}_mat_{i}")
        # Always qualify with the model name so repeated imports of different
        # files never silently merge materials in existing meshes.
        base = f"{base}[{getattr(tmd, 'name', 'TMD2')}]"
        name = base
        counter = 1
        while name in used_names:
            name = f"{base}.{counter:03d}"
            counter += 1
        used_names.add(name)

        force_blend = None
        if settings is not None and getattr(settings, "alpha_mode", 'AUTO') == 'BLEND':
            force_blend = 'BLEND'
        try:
            blender_mat = material_builder.build_material(
                name, tm_mat, resolver, settings, color_layers, force_blend=force_blend)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ChatGPT TMD2] Material {name} failed: {exc}")
            blender_mat = bpy.data.materials.new(name)
            blender_mat.use_nodes = True

        props = blender_mat.ctmd2_material
        props.material_hash = str(getattr(tm_mat, "hash", 0))
        props.shader_id = getattr(tm_mat, "shaderID", "") or ""
        props.unk = int(getattr(tm_mat, "unk", -1) or 0)

        params = list(getattr(tm_mat, "shaderParams", []) or [])
        props.param_values.clear()
        for value in params:
            props.param_values.add().value = float(value)

        props.textures.clear()
        for mat_tex in getattr(tm_mat, "textures", []) or []:
            tex = getattr(mat_tex, "texture", None)
            entry = props.textures.add()
            entry.texture_hash = str(getattr(tex, "hash", 0) if tex is not None else 0)
            entry.texture_index = int(getattr(mat_tex, "textureIndex", -1))
            entry.slot = int(getattr(mat_tex, "slot", 0))
            entry.width = int(getattr(tex, "width", 0) if tex is not None else 0)
            entry.height = int(getattr(tex, "height", 0) if tex is not None else 0)
            entry.value1 = int(getattr(mat_tex, "unk1", 0))
            entry.value2 = int(getattr(mat_tex, "unk2", 0))
            image = resolver.resolve(mat_tex)
            if image is not None:
                entry.image = image

        materials[id(tm_mat)] = blender_mat
        if report:
            print("\n" + material_builder.describe_material(blender_mat))
    return materials


def _color_layer_names(mesh):
    try:
        return [attribute.name for attribute in mesh.color_attributes]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# TMD2 importer
# ---------------------------------------------------------------------------
class ImportOptions:
    """Plain container so the settings can be passed around without bpy."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _scene_settings(context):
    scene = getattr(context, "scene", None)
    settings = getattr(scene, "ctmd2_settings", None) if scene else None
    return settings


def import_tmd2_data(context, tmd, base_name, texture_path=None, dds_dir=None,
                     settings=None, operator=None):
    """Create the Blender objects for an already parsed TMD2 file."""
    settings = settings or _scene_settings(context)
    collection = bpy.data.collections.new(base_name)
    context.collection.children.link(collection)

    # --- textures ------------------------------------------------------
    images = []
    if texture_path and os.path.exists(texture_path):
        try:
            images = load_lds_images(texture_path)
        except Exception as exc:
            print(f"[ChatGPT TMD2] Could not read {texture_path}: {exc}")
            if operator:
                operator.report({'WARNING'}, f"Texture file unreadable: {exc}")
    if not images:
        images = _images_from_table(tmd, dds_dir)

    resolver = TextureResolver(images, getattr(tmd, "textures", []))

    # --- file properties object ---------------------------------------
    empty = bpy.data.objects.new(f"#CTMD2 PROPERTIES [{base_name}]", None)
    empty.empty_display_size = 0.01
    try:
        props = empty.ctmd2_props
        props.flag1 = int(getattr(tmd, "flag1", 0))
        props.flag2 = int(getattr(tmd, "flag2", 0))
        props.flag3 = int(getattr(tmd, "animFlag", 0))
        props.after_image = int(getattr(tmd, "afterImageValue", 0))
        props.model_flags = int(getattr(tmd, "modelFlags", 0))
        props.source_file = getattr(tmd, "filepath", "") or ""
    except Exception as exc:  # pragma: no cover
        print(f"[ChatGPT TMD2] Could not store file properties: {exc}")
    collection.objects.link(empty)

    # --- armature ------------------------------------------------------
    armature_obj = None
    bones_to_pose = {}
    flags = int(getattr(tmd, "modelFlags", 0))
    if flags & 0x2000 and getattr(tmd, "bones", None):
        armature_obj, bones_to_pose = _build_armature(context, tmd, base_name, collection)

    # --- materials -----------------------------------------------------
    color_layers = ["Color", "Color2"]
    report = bool(settings and getattr(settings, "shader_report", False))
    materials = build_materials(tmd, resolver, settings, color_layers, report=report)

    bone_names = [bone.name for bone in armature_obj.data.bones] if armature_obj else None

    # --- meshes --------------------------------------------------------
    created = []
    for model in getattr(tmd, "models", []) or []:
        name = getattr(model, "name", "") or f"{base_name}_mesh"
        result = build_mesh(
            model, tmd, name, collection, materials,
            color_layers_enabled=bool(getattr(settings, "import_vertex_colors", True)),
            import_normals=bool(getattr(settings, "auto_smooth", True)),
            operator=operator,
            bone_names=bone_names,
        )
        obj = result.obj
        created.append(obj)
        if result.warnings:
            for warning in result.warnings:
                print(f"[ChatGPT TMD2] {name}: {warning}")
        if armature_obj is not None:
            _bind_to_armature(obj, armature_obj, tmd, bone_names)

    if armature_obj is not None and bones_to_pose:
        _pose_armature(armature_obj, bones_to_pose)

    return collection, created


def _images_from_table(tmd, dds_dir=None):
    """Build images for a TMD2 texture table without an .lds file."""
    images = []
    for i, texture in enumerate(getattr(tmd, "textures", []) or []):
        name = f"{IMAGE_PREFIX}{getattr(tmd, 'name', 'TMD2')}_{i}"
        image = bpy.data.images.get(name)
        if image is None and dds_dir:
            stem = getattr(tmd, "name", "TMD2")
            image = bpy.data.images.get(f"{IMAGE_PREFIX}{stem}_{i}")
        if image is None:
            width = getattr(texture, "width", 0) or 64
            height = getattr(texture, "height", 0) or 64
            image = placeholder_image(name, width, height)
        images.append(image)
    return images


def _build_armature(context, tmd, base_name, collection):
    armature_name = f"{base_name}_armature"
    armature = bpy.data.armatures.new(armature_name)
    armature_obj = bpy.data.objects.new(armature_name, armature)
    collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.data.display_type = 'STICK'

    bones_to_pose = {}
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        table = hashes()
        for tmbone in getattr(tmd, "bones", []):
            name = getattr(tmbone, "name", "") or str(getattr(tmbone, "hash", 0))
            if str(getattr(tmbone, "hash", "")) in table:
                name = table[str(tmbone.hash)]
            edit_bone = armature_obj.data.edit_bones.new(name)
            matrix = Matrix(getattr(tmbone, "matrix", np.eye(4))).transposed().inverted()
            matrix_up = _axis_matrix() @ matrix
            edit_bone.head = matrix_up @ Vector((0, 0, 0))
            edit_bone.tail = matrix_up @ Vector((0, 0.05, 0))
            parent_index = getattr(tmbone, "parentIndex", -1)
            if 0 <= parent_index < len(armature_obj.data.edit_bones) - 1:
                edit_bone.parent = armature_obj.data.edit_bones[parent_index]
            tries = 0
            while edit_bone.name in [b.name for b in armature_obj.data.edit_bones if b != edit_bone]:
                tries += 1
                edit_bone.name = f"{name}_{tries}"
            edit_bone["hash"] = str(getattr(tmbone, "hash", 0))
            edit_bone["extra"] = int(getattr(tmbone, "extra", -1))
            edit_bone["offset"] = list(getattr(tmbone, "offset", [0, 0, 0]))
            if int(getattr(tmbone, "extra", -1)) > -1:
                bones_to_pose[edit_bone.name] = list(getattr(tmbone, "offset", [0, 0, 0]))
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')
    return armature_obj, bones_to_pose


def _axis_matrix():
    return Matrix.Rotation(radians(-90), 4, 'X')


def _pose_armature(armature_obj, bones_to_pose):
    for bone_name, rotation in bones_to_pose.items():
        bone = armature_obj.pose.bones.get(bone_name)
        if bone is None or bone.parent is None:
            continue
        try:
            quaternion = Euler(rotation, 'XYZ').to_quaternion()
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = quaternion.inverted()
        except Exception:
            continue


def _bind_to_armature(obj, armature_obj, tmd, bone_names=None):
    """Parent the mesh to the armature and add an armature modifier."""
    if not getattr(tmd, "modelFlags", 0) & 0x400:
        return
    if len(obj.vertex_groups) == 0 and bone_names:
        for bone_name in bone_names:
            obj.vertex_groups.new(name=bone_name)
    modifier = obj.modifiers.new(name=armature_obj.name, type='ARMATURE')
    modifier.object = armature_obj
    modifier.use_vertex_groups = True
    obj.parent = armature_obj


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
_IMPORT_OPTIONS = (
    "auto_find_textures",
    "texture_path",
    "dds_dir",
    "import_vertex_colors",
    "alpha_mode",
)


class CTMD2_OT_ImportTMD2(Operator, ImportHelper):
    """Import a Tamsoft TMD2 model (ChatGPT TMD2)"""

    bl_idname = "ctmd2.import_tmd2"
    bl_label = "Import TMD2 (ChatGPT TMD2)"
    bl_options = {'REGISTER', 'UNDO'}

    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={'HIDDEN', 'SKIP_SAVE'})  # type: ignore
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})  # type: ignore
    filter_glob: StringProperty(default="*.tmd2;*.tmd;*.tmd0;*.tmdv;*.lds",
                                options={'HIDDEN'})  # type: ignore
    filename_ext = ".tmd2"

    auto_find_textures: BoolProperty(
        name="Auto Find Textures", default=True,
        description="Look for a matching .lds texture container next to the model")  # type: ignore
    texture_path: StringProperty(
        name="Texture File", subtype='FILE_PATH', default="",
        description="Explicit .lds file to use (leave empty to auto detect)")  # type: ignore
    dds_dir: StringProperty(
        name="DDS Folder", subtype='DIR_PATH', default="",
        description="Optional folder with loose .dds textures")  # type: ignore
    import_vertex_colors: BoolProperty(
        name="Import Vertex Colors", default=True)  # type: ignore
    auto_smooth: BoolProperty(
        name="Import Custom Normals", default=True)  # type: ignore
    shader_report: BoolProperty(
        name="Print Material Report", default=False,
        description="Print the interpreted material data to the console")  # type: ignore
    as_single_collection: BoolProperty(
        name="Merge Into One Collection", default=False)  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "auto_find_textures")
        layout.prop(self, "texture_path")
        layout.prop(self, "dds_dir")
        layout.separator()
        layout.prop(self, "import_vertex_colors")
        layout.prop(self, "auto_smooth")
        layout.prop(self, "shader_report")

    # -- helpers --------------------------------------------------------
    def _options(self):
        return ImportOptions(
            import_vertex_colors=self.import_vertex_colors,
            auto_smooth=self.auto_smooth,
            shader_report=self.shader_report,
        )

    def execute(self, context):
        start = time.perf_counter()
        files = list(self.files) or []
        if not files and self.filepath:
            files = [type("F", (), {"name": os.path.basename(self.filepath)})()]
        count = 0
        for file_entry in files:
            path = self.filepath if os.path.isabs(self.filepath) and len(files) == 1 \
                else os.path.join(self.directory or os.path.dirname(self.filepath), file_entry.name)
            if not os.path.exists(path):
                continue
            extension = os.path.splitext(path)[1].lower()
            try:
                if extension == ".lds":
                    load_lds_images(path, verbose=True)
                    continue
                if extension == ".tmd2":
                    self._import_tmd2(context, path)
                elif extension in {".tmd", ".tmd0"}:
                    self._import_tmd(context, path, extension)
                elif extension == ".tmdv":
                    self._import_tmdv(context, path)
                count += 1
            except Exception as exc:
                print(f"[ChatGPT TMD2] Import failed for {path}: {exc}")
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, f"{os.path.basename(path)}: {exc}")
        elapsed = time.perf_counter() - start
        self.report({'INFO'}, f"ChatGPT TMD2 imported {count} file(s) in {elapsed:.2f}s")
        return {'FINISHED'}

    def _texture_for(self, path):
        if self.texture_path and os.path.exists(self.texture_path):
            return self.texture_path
        if not self.auto_find_textures:
            return None
        return find_texture_file(path, self.dds_dir or None)

    def _import_tmd2(self, context, path):
        tmd = readTMD2(path)
        tmd.name = os.path.splitext(os.path.basename(path))[0]
        try:
            tmd.filepath = path
        except Exception:
            pass
        settings = _merge_settings(_scene_settings(context), self._options())
        import_tmd2_data(
            context, tmd, tmd.name,
            texture_path=self._texture_for(path),
            dds_dir=self.dds_dir or None,
            settings=settings,
            operator=self,
        )

    def _import_tmd(self, context, path, extension):
        names = {}
        dds = {}
        directory = os.path.dirname(path)
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(".dds"):
                stem = os.path.splitext(file_name)[0]
                dds[stem] = os.path.join(directory, file_name)
                names[tamCRC32(stem)] = stem
        if extension == ".tmd0":
            tmd = readTMD0(path, names)
        else:
            tmd = readTMD(path, names)
        tmd.name = os.path.splitext(os.path.basename(path))[0]
        images = []
        for i, texture in enumerate(getattr(tmd, "textures", []) or []):
            image = None
            for key, value in dds.items():
                if tamCRC32(key) == getattr(texture, "hash", None):
                    image = bpy.data.images.get(IMAGE_PREFIX + key)
                    if image is None:
                        image = bpy.data.images.load(value, check_existing=True)
                    break
            if image is None:
                image = placeholder_image(f"{IMAGE_PREFIX}{tmd.name}_{i}")
            images.append(image)
        settings = _merge_settings(_scene_settings(context), self._options())
        _import_generic(context, tmd, tmd.name, images, settings, self)

    def _import_tmdv(self, context, path):
        tmd = readTMDV(path)
        tmd.name = os.path.splitext(os.path.basename(path))[0]
        settings = _merge_settings(_scene_settings(context), self._options())
        _import_generic(context, tmd, tmd.name, [], settings, self)


class Options:
    """Lightweight settings object (avoids instantiating bpy PropertyGroups)."""

    _DEFAULTS = {
        "fallback_shader": 'SIMPLE',
        "alpha_mode": 'AUTO',
        "vertex_color_mode": 'AUTO',
        "emission_boost": 1.0,
        "import_vertex_colors": True,
        "auto_smooth": True,
        "import_outline": False,
        "shader_report": False,
    }

    def __init__(self, **kwargs):
        for key, value in self._DEFAULTS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_scene(cls, scene_settings, **overrides):
        values = dict(cls._DEFAULTS)
        if scene_settings is not None:
            for key in cls._DEFAULTS:
                if hasattr(scene_settings, key):
                    values[key] = getattr(scene_settings, key)
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
        return cls(**values)


def _merge_settings(scene_settings, options):
    """Build the effective settings from the scene panel + operator options."""
    overrides = {}
    for key in ("import_vertex_colors", "auto_smooth", "shader_report"):
        if hasattr(options, key):
            overrides[key] = getattr(options, key)
    for key in ("fallback_shader", "alpha_mode", "vertex_color_mode", "emission_boost"):
        if hasattr(options, key) and getattr(options, key) not in (None, ""):
            overrides[key] = getattr(options, key)
    return Options.from_scene(scene_settings, **overrides)


def _import_generic(context, tmd, base_name, images, settings, operator):
    """Import a TMD / TMD0 / TMDV file (legacy vertex layout)."""
    collection = bpy.data.collections.new(base_name)
    context.collection.children.link(collection)

    empty = bpy.data.objects.new(f"#CTMD2 PROPERTIES [{base_name}]", None)
    empty.empty_display_size = 0.01
    try:
        empty.ctmd2_props.model_flags = int(getattr(tmd, "modelFlags", 0))
        empty.ctmd2_props.flag1 = int(getattr(tmd, "flag1", 0) or 0)
        empty.ctmd2_props.flag2 = int(getattr(tmd, "flag2", 0) or 0)
    except Exception:
        pass
    collection.objects.link(empty)

    resolver = TextureResolver(images, getattr(tmd, "textures", []))
    materials = build_materials(tmd, resolver, settings, ["Color"],
                                report=bool(getattr(settings, "shader_report", False)))

    for model in getattr(tmd, "models", []) or []:
        name = getattr(model, "name", "") or f"{base_name}_mesh"
        build_mesh(
            model, tmd, name, collection, materials,
            color_layers_enabled=bool(getattr(settings, "import_vertex_colors", True)),
            import_normals=True,
            operator=operator,
        )
    return collection


class CTMD2_OT_DropTMD2(Operator):
    """Import a TMD2 file dropped into the 3D viewport (ChatGPT TMD2)"""

    bl_idname = "ctmd2.drop_tmd2"
    bl_label = "Import TMD2 (Drag and Drop)"

    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={'HIDDEN', 'SKIP_SAVE'})  # type: ignore
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})  # type: ignore
    filepath: StringProperty(subtype='FILE_PATH')  # type: ignore
    filter_glob: StringProperty(default="*.tmd2;*.tmd;*.tmd0;*.tmdv;*.lds",
                                options={'HIDDEN'})  # type: ignore
    filename_ext = ".tmd2"

    def execute(self, context):
        directory = self.directory or os.path.dirname(self.filepath)
        paths = [os.path.join(directory, f.name) for f in self.files] if self.files else [self.filepath]
        scene_settings = _scene_settings(context)
        imported = 0
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            extension = os.path.splitext(path)[1].lower()
            settings = _merge_settings(scene_settings, Options())
            try:
                if extension == ".tmd2":
                    tmd = readTMD2(path)
                    tmd.name = os.path.splitext(os.path.basename(path))[0]
                    import_tmd2_data(
                        context, tmd, tmd.name,
                        texture_path=find_texture_file(path),
                        settings=settings, operator=self)
                elif extension == ".lds":
                    load_lds_images(path, verbose=True)
                else:
                    continue
                imported += 1
            except Exception as exc:
                print(f"[ChatGPT TMD2] Drop import failed for {path}: {exc}")
                import traceback
                traceback.print_exc()
                self.report({'ERROR'}, f"{os.path.basename(path)}: {exc}")
        self.report({'INFO'}, f"ChatGPT TMD2 imported {imported} file(s)")
        return {'FINISHED'}


class CTMD2_FH_import(bpy.types.FileHandler):
    """File handler for ChatGPT TMD2"""

    bl_idname = "CTMD2_FH_import"
    bl_label = "ChatGPT TMD2 file handler"
    bl_import_operator = "ctmd2.drop_tmd2"
    bl_file_extensions = ".tmd2"

    @classmethod
    def poll_drop(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'


class CTMD2_FH_import_tmd(bpy.types.FileHandler):
    """File handler for ChatGPT TMD"""

    bl_idname = "CTMD2_FH_import_tmd"
    bl_label = "ChatGPT TMD file handler"
    bl_import_operator = "ctmd2.drop_tmd2"
    bl_file_extensions = ".tmd;.tmd0;.tmdv;.lds"

    @classmethod
    def poll_drop(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'


def menu_func_import(self, context):
    self.layout.operator(CTMD2_OT_ImportTMD2.bl_idname,
                         text='Tamsoft TMD2 (ChatGPT TMD2)',
                         icon='IMPORT')
