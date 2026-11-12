"""Property groups, operators and panels for the ChatGPT TMD2 add-on.

Every identifier in here is prefixed with ``ctmd2`` so the add-on can live next to
the original ``BlenderTMD2`` add-on without a single collision:

* PropertyGroup names   -> ``CTMD2*``
* Material/Object props -> ``material.ctmd2_*`` / ``object.ctmd2_*``
* Operator bl_idnames   -> ``ctmd2.*``
* Panel bl_idnames      -> ``CTMD2_PT_*``
* UIList bl_idnames     -> ``CTMD2_UL_*``
"""

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Panel, PropertyGroup, UIList


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------
class CTMD2ShaderParam(PropertyGroup):
    value: FloatProperty(name="Value", precision=4)  # type: ignore


class CTMD2MaterialTexture(PropertyGroup):
    texture_hash: StringProperty(name="Texture Hash", default="0")  # type: ignore
    texture_index: IntProperty(name="TMD2 Texture Index", default=-1)  # type: ignore
    slot: IntProperty(name="Material Texture Slot", default=0)  # type: ignore
    width: IntProperty(name="Width", default=0)  # type: ignore
    height: IntProperty(name="Height", default=0)  # type: ignore
    value1: IntProperty(name="Unknown 1", default=0)  # type: ignore
    value2: IntProperty(name="Unknown 2", default=0)  # type: ignore
    value3: IntProperty(name="Unknown 3", default=0)  # type: ignore

    image: PointerProperty(
        name="DDS Texture",
        type=bpy.types.Image,
        description="Image used for this texture slot",
    )  # type: ignore


class CTMD2MaterialProperties(PropertyGroup):
    """Stored on ``bpy.types.Material`` - mirrors the raw TMD2 material."""

    material_hash: StringProperty(name="Material Hash", default="0")  # type: ignore
    shader_id: StringProperty(name="Shader ID", default="")  # type: ignore
    shader_known: BoolProperty(name="Known Shader", default=False)  # type: ignore
    shader_detected: StringProperty(name="Detected Shader", default="")  # type: ignore
    unk: IntProperty(name="Unknown", default=-1)  # type: ignore

    param_values: CollectionProperty(type=CTMD2ShaderParam)  # type: ignore
    param_index: IntProperty(default=0)  # type: ignore

    textures: CollectionProperty(type=CTMD2MaterialTexture)  # type: ignore
    texture_index: IntProperty(default=0)  # type: ignore

    # --- interpretation results (filled in by the material builder) --------
    detected_blend: EnumProperty(
        name="Detected Blend Mode",
        items=[
            ('OPAQUE', "Opaque", ""),
            ('CLIP', "Alpha Clip", ""),
            ('HASHED', "Alpha Hashed", ""),
            ('BLEND', "Alpha Blend", ""),
            ('ADD', "Additive", ""),
        ],
        default='OPAQUE',
    )  # type: ignore
    uses_vertex_color: BoolProperty(name="Uses Vertex Color", default=False)  # type: ignore
    uses_emission: BoolProperty(name="Uses Emission", default=False)  # type: ignore
    base_color: FloatVectorProperty(
        name="Base Color", subtype='COLOR', size=4, min=0.0, max=1.0, default=(1, 1, 1, 1)
    )  # type: ignore
    emission_color: FloatVectorProperty(
        name="Emission Color", subtype='COLOR', size=4, min=0.0, max=1.0, default=(0, 0, 0, 1)
    )  # type: ignore
    emission_strength: FloatProperty(name="Emission Strength", default=0.0, min=0.0, soft_max=50.0)  # type: ignore
    alpha_value: FloatProperty(name="Alpha", default=1.0, min=0.0, max=1.0)  # type: ignore
    alpha_clip_threshold: FloatProperty(name="Alpha Clip Threshold", default=0.0, min=0.0, max=1.0)  # type: ignore
    notes: StringProperty(name="Interpretation Notes", default="")  # type: ignore


class CTMD2MeshProperties(PropertyGroup):
    has_name: BoolProperty(name="Has Name", default=False)  # type: ignore
    has_hash: BoolProperty(name="Has Hash", default=False)  # type: ignore
    has_extra: BoolProperty(name="Has Extra Values", default=False)  # type: ignore
    name: StringProperty(name="Name", default="")  # type: ignore
    name_hash: StringProperty(name="Name Hash", default="0")  # type: ignore
    unk: IntProperty(default=0)  # type: ignore
    unk1: IntProperty(default=0)  # type: ignore
    unk2: IntProperty(default=0)  # type: ignore
    unk3: IntProperty(default=0)  # type: ignore


class CTMD2Properties(PropertyGroup):
    flag1: IntProperty(name="Flag 1", default=0)  # type: ignore
    flag2: IntProperty(name="Flag 2", default=0)  # type: ignore
    flag3: IntProperty(name="Flag 3", default=0)  # type: ignore
    after_image: IntProperty(name="After Image Value", default=0)  # type: ignore
    transform_animation: PointerProperty(type=bpy.types.Action)  # type: ignore
    model_flags: IntProperty(name="Model Flags", default=0)  # type: ignore
    source_file: StringProperty(name="Source File", default="")  # type: ignore


# ---------------------------------------------------------------------------
# Scene level settings (fallback system, global options)
# ---------------------------------------------------------------------------
class CTMD2Settings(PropertyGroup):
    fallback_shader: EnumProperty(
        name="Unknown Shader Fallback",
        description="Material definition used when a shader ID is not implemented",
        items=[
            ('AUTO', "Auto (Detect)", "Guess the best existing shader library material from the texture layout"),
            ('SIMPLE', "Simple PBR", "Neutral Principled BSDF with the first texture and vertex colours"),
            ('DIFFUSE', "Diffuse Only", "Plain diffuse material driven by the first texture"),
            ('UNLIT', "Unlit / Emission", "Emissive unlit material, useful for glowing effects"),
            ('PLACEHOLDER', "Placeholder", "Bright magenta placeholder so unknown shaders are obvious"),
        ],
        default='SIMPLE',
    )  # type: ignore

    alpha_mode: EnumProperty(
        name="Alpha Handling",
        description="How alpha information found in the material data is applied",
        items=[
            ('AUTO', "Auto", "Derive the blend mode from the material data and shader"),
            ('OPAQUE', "Force Opaque", ""),
            ('CLIP', "Force Alpha Clip", ""),
            ('HASHED', "Force Alpha Hashed", ""),
            ('BLEND', "Force Alpha Blend", ""),
        ],
        default='AUTO',
    )  # type: ignore

    vertex_color_mode: EnumProperty(
        name="Vertex Color",
        description="Multiply the base colour with the mesh vertex colours",
        items=[
            ('AUTO', "Auto", "Use vertex colour when the shader expects it"),
            ('ALWAYS', "Always", "Always multiply base colour with the first colour layer"),
            ('NEVER', "Never", "Ignore vertex colours"),
        ],
        default='AUTO',
    )  # type: ignore

    emission_boost: FloatProperty(
        name="Emission Boost",
        description="Multiplier applied to emission found in the material data",
        default=1.0, min=0.0, soft_max=20.0,
    )  # type: ignore

    import_vertex_colors: BoolProperty(
        name="Import Vertex Colors", default=True,
        description="Import the TMD2 colour layers as mesh colour attributes",
    )  # type: ignore

    import_outline: BoolProperty(
        name="Create Outline Material", default=False,
        description="Create an inverted hull outline material for characters",
    )  # type: ignore

    auto_smooth: BoolProperty(
        name="Import Custom Normals", default=True,
        description="Use the normals stored in the TMD2 vertices",
    )  # type: ignore

    shader_report: BoolProperty(name="Print Material Report", default=False)  # type: ignore


def active_material(context):
    """Return the material the user is working on.

    ``context.material`` only exists in the material properties editor, so fall
    back to the active object's active material slot.
    """
    material = getattr(context, "material", None)
    if material is not None:
        return material
    obj = getattr(context, "object", None) or getattr(context, "active_object", None)
    if obj is not None and getattr(obj, "type", "") == 'MESH' and obj.material_slots:
        return obj.material_slots[obj.active_material_index].material
    return None


# ---------------------------------------------------------------------------
# Clipboard helpers (independent from the original add-on)
# ---------------------------------------------------------------------------
class CTMD2MaterialClipboard:
    material_hash = "0"
    shader_id = ""
    unk = -1
    param_values = []
    textures = []

    @classmethod
    def copy(cls, src):
        cls.material_hash = src.material_hash
        cls.shader_id = src.shader_id
        cls.unk = src.unk
        cls.param_values = [p.value for p in src.param_values]
        cls.textures = [{
            "texture_hash": t.texture_hash,
            "texture_index": t.texture_index,
            "slot": t.slot,
            "width": t.width,
            "height": t.height,
            "value1": t.value1,
            "value2": t.value2,
            "value3": t.value3,
            "image": t.image,
        } for t in src.textures]

    @classmethod
    def paste(cls, dst):
        dst.material_hash = cls.material_hash
        dst.shader_id = cls.shader_id
        dst.unk = cls.unk

        dst.param_values.clear()
        for value in cls.param_values:
            dst.param_values.add().value = value

        dst.textures.clear()
        for t in cls.textures:
            tex = dst.textures.add()
            tex.texture_hash = t["texture_hash"]
            tex.texture_index = t["texture_index"]
            tex.slot = t["slot"]
            tex.width = t["width"]
            tex.height = t["height"]
            tex.value1 = t["value1"]
            tex.value2 = t["value2"]
            tex.value3 = t["value3"]
            tex.image = t["image"]


class CTMD2ParamClipboard:
    param_values = []

    @classmethod
    def copy(cls, params):
        cls.param_values = [p.value for p in params]

    @classmethod
    def paste(cls, params):
        params.clear()
        for value in cls.param_values:
            params.add().value = value


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
class CTMD2_OT_CopyMaterialData(bpy.types.Operator):
    bl_idname = "ctmd2.copy_material_data"
    bl_label = "Copy TMD2 Material Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        mat = active_material(context)
        if not mat or not hasattr(mat, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material to copy.")
            return {'CANCELLED'}
        CTMD2MaterialClipboard.copy(mat.ctmd2_material)
        return {'FINISHED'}


class CTMD2_OT_PasteMaterialData(bpy.types.Operator):
    bl_idname = "ctmd2.paste_material_data"
    bl_label = "Paste TMD2 Material Data"
    bl_options = {'REGISTER'}

    def execute(self, context):
        mat = active_material(context)
        if not mat or not hasattr(mat, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material to paste into.")
            return {'CANCELLED'}
        CTMD2MaterialClipboard.paste(mat.ctmd2_material)
        return {'FINISHED'}


class CTMD2_OT_CopyParams(bpy.types.Operator):
    bl_idname = "ctmd2.copy_params"
    bl_label = "Copy All Shader Parameters"

    def execute(self, context):
        mat = active_material(context)
        if not mat or not hasattr(mat, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        CTMD2ParamClipboard.copy(mat.ctmd2_material.param_values)
        self.report({'INFO'}, f"Copied {len(CTMD2ParamClipboard.param_values)} parameters.")
        return {'FINISHED'}


class CTMD2_OT_PasteParams(bpy.types.Operator):
    bl_idname = "ctmd2.paste_params"
    bl_label = "Paste All Shader Parameters"

    def execute(self, context):
        mat = active_material(context)
        if not mat or not hasattr(mat, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        CTMD2ParamClipboard.paste(mat.ctmd2_material.param_values)
        return {'FINISHED'}


class CTMD2_OT_TextureAdd(bpy.types.Operator):
    bl_idname = "ctmd2.texture_add"
    bl_label = "Add Texture"

    def execute(self, context):
        material = active_material(context)
        if material is None or not hasattr(material, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        props = material.ctmd2_material
        props.textures.add()
        props.texture_index = len(props.textures) - 1
        return {'FINISHED'}


class CTMD2_OT_TextureRemove(bpy.types.Operator):
    bl_idname = "ctmd2.texture_remove"
    bl_label = "Remove Texture"

    def execute(self, context):
        material = active_material(context)
        if material is None or not hasattr(material, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        props = material.ctmd2_material
        if props.textures:
            props.textures.remove(props.texture_index)
            props.texture_index = max(0, props.texture_index - 1)
        return {'FINISHED'}


class CTMD2_OT_TextureMove(bpy.types.Operator):
    bl_idname = "ctmd2.texture_move"
    bl_label = "Move Texture"

    direction: EnumProperty(items=[('UP', "Up", ""), ('DOWN', "Down", "")], name="Direction")  # type: ignore

    def execute(self, context):
        material = active_material(context)
        if material is None or not hasattr(material, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        props = material.ctmd2_material
        idx = props.texture_index
        new_idx = idx - 1 if self.direction == 'UP' else idx + 1
        if 0 <= new_idx < len(props.textures):
            props.textures.move(idx, new_idx)
            props.texture_index = new_idx
        return {'FINISHED'}


class CTMD2_OT_ParamAdd(bpy.types.Operator):
    bl_idname = "ctmd2.param_add"
    bl_label = "Add Param"

    def execute(self, context):
        material = active_material(context)
        if material is None or not hasattr(material, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        props = material.ctmd2_material
        props.param_values.add()
        props.param_index = len(props.param_values) - 1
        return {'FINISHED'}


class CTMD2_OT_ParamRemove(bpy.types.Operator):
    bl_idname = "ctmd2.param_remove"
    bl_label = "Remove Param"

    def execute(self, context):
        material = active_material(context)
        if material is None or not hasattr(material, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        props = material.ctmd2_material
        if props.param_values:
            props.param_values.remove(props.param_index)
            props.param_index = max(0, props.param_index - 1)
        return {'FINISHED'}


class CTMD2_OT_TextureOpenDDS(bpy.types.Operator):
    bl_idname = "ctmd2.open_dds"
    bl_label = "Open DDS Texture"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH')  # type: ignore
    filter_glob: StringProperty(default="*.dds;*.png;*.tga;*.tif;*.tiff", options={'HIDDEN'})  # type: ignore

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        material = active_material(context)
        if material is None or not hasattr(material, "ctmd2_material"):
            self.report({'WARNING'}, "No ChatGPT TMD2 material selected")
            return {'CANCELLED'}
        props = material.ctmd2_material
        if not props.textures:
            props.textures.add()
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to load image: {e}")
            return {'CANCELLED'}
        props.textures[props.texture_index].image = img
        return {'FINISHED'}


def _rebuild_materials(context, all_materials):
    """Shared implementation for the two rebuild operators."""
    from . import material_builder as builder
    from ..importers.importer import Options

    settings = Options.from_scene(getattr(context.scene, "ctmd2_settings", None))

    if all_materials:
        targets = [mat for mat in bpy.data.materials
                   if getattr(mat, "ctmd2_material", None) and mat.ctmd2_material.shader_id]
    else:
        targets = []
        active = active_material(context)
        if active is not None and getattr(active, "ctmd2_material", None):
            targets.append(active)
        for obj in getattr(context, "selected_objects", []):
            if obj.type != 'MESH':
                continue
            for slot in obj.material_slots:
                mat = slot.material
                if mat and getattr(mat, "ctmd2_material", None) and mat not in targets:
                    targets.append(mat)

    rebuilt = 0
    failed = 0
    for mat in targets:
        try:
            if builder.rebuild_material_from_props(mat, settings):
                rebuilt += 1
        except Exception as exc:  # pragma: no cover - defensive
            failed += 1
            print(f"[ChatGPT TMD2] rebuild failed for {mat.name}: {exc}")
    return rebuilt, failed


class CTMD2_OT_RebuildMaterial(bpy.types.Operator):
    """Re-run shader interpretation on the active or selected material(s)"""

    bl_idname = "ctmd2.rebuild_material"
    bl_label = "Rebuild TMD2 Material"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rebuilt, failed = _rebuild_materials(context, False)
        if failed:
            self.report({'WARNING'}, f"Rebuilt {rebuilt} material(s), {failed} failed.")
        elif not rebuilt:
            self.report({'INFO'}, "No ChatGPT TMD2 material selected.")
        else:
            self.report({'INFO'}, f"Rebuilt {rebuilt} material(s).")
        return {'FINISHED'}


class CTMD2_OT_RebuildAllMaterials(bpy.types.Operator):
    """Rebuild every ChatGPT TMD2 material in the file"""

    bl_idname = "ctmd2.rebuild_all_materials"
    bl_label = "Rebuild All TMD2 Materials"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rebuilt, failed = _rebuild_materials(context, True)
        if failed:
            self.report({'WARNING'}, f"Rebuilt {rebuilt} material(s), {failed} failed.")
        else:
            self.report({'INFO'}, f"Rebuilt {rebuilt} material(s).")
        return {'FINISHED'}


class CTMD2_OT_PrintMaterialReport(bpy.types.Operator):
    """Print the raw TMD2 material data of the active material to the console"""

    bl_idname = "ctmd2.print_material_report"
    bl_label = "Print Material Report"

    def execute(self, context):
        from . import material_builder as shader_builder
        mat = active_material(context)
        if not mat or not getattr(mat, "ctmd2_material", None):
            self.report({'WARNING'}, "No TMD2 material selected.")
            return {'CANCELLED'}
        print("\n" + shader_builder.describe_material(mat) + "\n")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# UI lists
# ---------------------------------------------------------------------------
class CTMD2_UL_Textures(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            row.label(text=f"[{index}]")
            row.prop(item, "texture_hash", text="", emboss=False)
            row.prop_search(item, "image", bpy.data, "images", text="")
            row.operator("ctmd2.open_dds", text="", icon='FILE_FOLDER')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='TEXTURE')


class CTMD2_UL_Params(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            layout.label(text=f"Param {index}")
            layout.prop(item, "value", text="Value")
        elif self.layout_type == 'GRID':
            layout.label(text=str(index))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
class CTMD2_PT_MaterialPanel(Panel):
    bl_idname = 'CTMD2_PT_material'
    bl_label = 'ChatGPT TMD2 Material'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    @classmethod
    def poll(cls, context):
        material = getattr(context, "material", None) or active_material(context)
        return material is not None and hasattr(material, "ctmd2_material")

    def draw(self, context):
        layout = self.layout
        mat = active_material(context)
        props = mat.ctmd2_material

        row = layout.row(align=True)
        row.operator("ctmd2.copy_material_data", icon="COPYDOWN", text="Copy")
        row.operator("ctmd2.paste_material_data", icon="PASTEDOWN", text="Paste")
        row.operator("ctmd2.print_material_report", icon="CONSOLE", text="Report")

        box = layout.box()
        box.label(text="Shader", icon='NODETREE')
        row = box.row()
        row.prop(props, "shader_id", text="Shader ID")
        row.prop(props, "material_hash", text="Hash")
        if props.shader_known:
            box.label(text=f"Interpreted: {props.shader_detected}", icon='CHECKMARK')
        else:
            box.label(text="Unknown shader - fallback material used", icon='ERROR')
        row = box.row(align=True)
        row.operator("ctmd2.rebuild_material", icon='FILE_REFRESH', text="Rebuild")
        row.operator("ctmd2.rebuild_all_materials", icon='FILE_REFRESH', text="Rebuild All")

        box = layout.box()
        box.label(text="Interpreted Surface", icon='SHADING_RENDERED')
        col = box.column(align=True)
        col.prop(props, "detected_blend", text="Blend")
        col.prop(props, "base_color", text="Base Color")
        col.prop(props, "alpha_value", text="Alpha")
        col.prop(props, "alpha_clip_threshold", text="Clip Threshold")
        col.prop(props, "emission_color", text="Emission")
        col.prop(props, "emission_strength", text="Strength")
        row = col.row(align=True)
        row.prop(props, "uses_vertex_color", text="Vertex Color")
        row.prop(props, "uses_emission", text="Emission")
        if props.notes:
            for line in props.notes.split(";"):
                if line.strip():
                    box.label(text=line.strip(), icon='DOT')

        row = layout.row()
        col1 = row.column()
        col2 = row.column()

        box1 = col1.box()
        box1.label(text="Raw Material")
        row = box1.row()
        row.label(text=f"Generated hash: {props.material_hash}")
        row = box1.row()
        row.prop(props, "unk", text="Unknown")

        box = col2.box()
        box.label(text="Shader Parameters")
        row = box.row()
        row.template_list("CTMD2_UL_Params", "", props, "param_values", props, "param_index")
        col = row.column(align=True)
        col.operator("ctmd2.param_add", icon="ADD", text="")
        col.operator("ctmd2.param_remove", icon="REMOVE", text="")
        row = box.row(align=True)
        row.operator("ctmd2.copy_params", text="Copy Params")
        row.operator("ctmd2.paste_params", text="Paste Params")

        box2 = layout.box()
        box2.label(text="Textures")
        row = box2.row()
        row.template_list("CTMD2_UL_Textures", "", props, "textures", props, "texture_index")
        col = row.column(align=True)
        col.operator("ctmd2.texture_add", icon="ADD", text="")
        col.operator("ctmd2.texture_remove", icon="REMOVE", text="")
        if props.textures:
            tex = props.textures[min(props.texture_index, len(props.textures) - 1)]
            row = box2.row(align=True)
            row.prop(tex, "slot", text="Slot")
            row.prop(tex, "texture_index", text="Index")
            row = box2.row(align=True)
            row.prop(tex, "value1", text="Unknown 1")
            row.prop(tex, "value2", text="Unknown 2")


class CTMD2_PT_MeshPanel(Panel):
    bl_idname = 'CTMD2_PT_mesh'
    bl_label = 'ChatGPT TMD2 Mesh'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    @classmethod
    def poll(cls, context):
        return context.object is not None and hasattr(context.object, "ctmd2_mesh")

    def draw(self, context):
        layout = self.layout
        props = context.object.ctmd2_mesh
        row = layout.row()
        row.prop(props, "has_name", text="Has Name")
        row.prop(props, "has_hash", text="Has Hash")
        row = layout.row()
        row.prop(props, "name", text="Name")
        row.prop(props, "name_hash", text="Hash")


class CTMD2_PT_FilePanel(Panel):
    bl_idname = 'CTMD2_PT_file'
    bl_label = 'ChatGPT TMD2 File'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    @classmethod
    def poll(cls, context):
        return context.object is not None and hasattr(context.object, "ctmd2_props") \
            and context.object.name.startswith("#CTMD2 PROPERTIES")

    def draw(self, context):
        layout = self.layout
        props = context.object.ctmd2_props
        col = layout.column(align=True)
        col.prop(props, "source_file", text="File")
        col.prop(props, "model_flags", text="Model Flags")
        row = layout.row()
        row.prop(props, "flag1", text="Flag 1")
        row.prop(props, "flag2", text="Flag 2")
        row.prop(props, "flag3", text="Flag 3")
        layout.prop(props, "after_image", text="After Image Value")


class CTMD2_PT_ScenePanel(Panel):
    bl_idname = 'CTMD2_PT_settings'
    bl_label = 'ChatGPT TMD2'
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'scene'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.ctmd2_settings
        box = layout.box()
        box.label(text="Material Interpretation", icon='MATERIAL')
        box.prop(settings, "fallback_shader")
        box.prop(settings, "alpha_mode")
        box.prop(settings, "vertex_color_mode")
        box.prop(settings, "emission_boost")
        box = layout.box()
        box.label(text="Import Options", icon='IMPORT')
        box.prop(settings, "import_vertex_colors")
        box.prop(settings, "auto_smooth")
        box.prop(settings, "import_outline")
        box.prop(settings, "shader_report")


property_groups = [
    CTMD2ShaderParam,
    CTMD2MaterialTexture,
    CTMD2MaterialProperties,
    CTMD2MeshProperties,
    CTMD2Properties,
    CTMD2Settings,
]

operators = [
    CTMD2_OT_CopyMaterialData,
    CTMD2_OT_PasteMaterialData,
    CTMD2_OT_CopyParams,
    CTMD2_OT_PasteParams,
    CTMD2_OT_TextureAdd,
    CTMD2_OT_TextureRemove,
    CTMD2_OT_TextureMove,
    CTMD2_OT_ParamAdd,
    CTMD2_OT_ParamRemove,
    CTMD2_OT_TextureOpenDDS,
    CTMD2_OT_RebuildMaterial,
    CTMD2_OT_RebuildAllMaterials,
    CTMD2_OT_PrintMaterialReport,
]

ui_lists = [
    CTMD2_UL_Textures,
    CTMD2_UL_Params,
]

panels = [
    CTMD2_PT_ScenePanel,
    CTMD2_PT_MaterialPanel,
    CTMD2_PT_MeshPanel,
    CTMD2_PT_FilePanel,
]
