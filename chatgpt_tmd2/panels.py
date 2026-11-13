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

from .materials.analyzer import (
    delete_custom_shader_rule,
    load_custom_shader_rules,
    save_custom_shader_rule,
)
from .materials.shaders import rebuild_blender_material
from .tamLib.tmd2 import tamCRC32


# =====================================================================
# TEXTURE & SHADER PARAM ITEMS
# =====================================================================

class CGTMD2_TextureItem(bpy.types.PropertyGroup):
    image: PointerProperty(name="Texture", type=bpy.types.Image)
    texture_hash: StringProperty(name="Texture Hash", default="0")
    resolved_name: StringProperty(name="Resolved Name", default="")
    value1: IntProperty(name="Unk 1", default=0)
    value2: IntProperty(name="Unk 2", default=0)
    value3: IntProperty(name="Unk 3 / UV Flags", default=0, min=0, max=255)
    slot: IntProperty(name="Sampler Slot", default=0, min=0, max=255)
    is_external: BoolProperty(
        name="External / Global Texture",
        description="True when textureIndex == -1 in TMD2 (stage/environment/shared texture outside local LDS)",
        default=False,
    )


class CGTMD2_ParamItem(bpy.types.PropertyGroup):
    value: FloatProperty(name="Value", default=0.0, precision=4)
    meta: IntProperty(
        name="Param Meta",
        description="Raw 32-bit metadata prefix stored before float value in TMD2 shaderParams",
        default=0,
    )


# =====================================================================
# MATERIAL PROPERTIES (with Reconstruction & Configurable Fallback)
# =====================================================================

class CGTMD2_MaterialProperties(bpy.types.PropertyGroup):
    material_hash: StringProperty(
        name="Material Hash",
        description="Signed 32-bit CRC hash of the material",
        default="0",
    )
    shader_id: StringProperty(
        name="Shader ID",
        description="4-character TamEngine shader identifier (e.g. BSS1, HSK1, or Captain Tsubasa shader code)",
        default="Default",
        maxlen=8,
    )
    unk: IntProperty(name="Unk Flags", default=0)

    textures: CollectionProperty(type=CGTMD2_TextureItem)
    active_texture_index: IntProperty(default=0)

    param_values: CollectionProperty(type=CGTMD2_ParamItem)
    active_param_index: IntProperty(default=0)

    # Diagnostic / Auto-Detected Info
    detected_family: StringProperty(name="Detected Family", default="AUTO")
    detected_blend: StringProperty(name="Detected Blend", default="OPAQUE")

    # Configurable Reconstruction & Fallback Settings
    reconstruction_profile: EnumProperty(
        name="Shader Profile",
        description="Shader reconstruction profile for known or unknown game shaders",
        items=[
            ("AUTO", "Auto (Data-Driven)", "Automatically infer shader behavior from shaderParams, textures, and mesh attributes"),
            ("TOON_CHAR", "Anime / Cel Character", "TamEngine cel/toon character & kit shading"),
            ("VFX_ADDITIVE", "Glowing VFX / Aura (Additive)", "Emissive additive shot/aura effects (e.g. golden glowing tiger, fire, lightning)"),
            ("GHOST_SPIRIT", "Ghost / Spirit (Translucent Rim)", "See-through spirit/phantom model with glowing Fresnel silhouette"),
            ("METALLIC_GLOW", "Metallic / Gold Sheen", "High-metallic reflective gold/armor surface with rim & glow support"),
            ("PBR_TINT", "Standard Tinted Lit", "Clean Principled BSDF surface with parameter tint & alpha support"),
        ],
        default="AUTO",
    )
    transparency_mode: EnumProperty(
        name="Transparency Mode",
        items=[
            ("AUTO", "Auto", "Infer transparency from texture alpha, luminance, vertex alpha, and material parameters"),
            ("OPAQUE", "Opaque", "Fully solid surface"),
            ("ALPHA_CLIP", "Alpha Clip (Cutout)", "Sharp cutout transparency"),
            ("ALPHA_BLEND", "Alpha Blend (See-Through)", "Smooth semi-transparent surface"),
            ("ADDITIVE", "Additive Glow", "Additive light blending (black pixels become fully transparent)"),
            ("GHOST", "Ghost / Fresnel See-Through", "Translucent core with brighter glowing edges"),
        ],
        default="AUTO",
    )
    vertex_color_mode: EnumProperty(
        name="Vertex Colors",
        items=[
            ("AUTO", "Auto", "Apply vertex RGB/Alpha only when submesh vertices carry non-trivial color/alpha data"),
            ("ON", "Force Enable", "Always multiply Color & Alpha vertex attributes"),
            ("OFF", "Disable", "Ignore mesh vertex colors"),
        ],
        default="AUTO",
    )
    grayscale_mode: EnumProperty(
        name="Grayscale Tinting",
        items=[
            ("AUTO", "Auto", "Automatically colorize grayscale textures using extracted shaderParams or Ramp LUT"),
            ("COLORIZE", "Force 2-Tone Colorize", "Map grayscale luminance through Secondary -> Base Tint colors"),
            ("MULTIPLY", "Simple Multiply", "Multiply texture directly by Base Tint color"),
            ("OFF", "No Tint", "Use raw texture colors without tint"),
        ],
        default="AUTO",
    )
    backface_culling_mode: EnumProperty(
        name="Backface Culling",
        items=[
            ("AUTO", "Auto", "Disable backface culling on VFX/Ghost/double-sided meshes; enable on solid meshes"),
            ("OFF", "Double-Sided (Show Back)", "Render both sides of faces"),
            ("ON", "Single-Sided (Cull Back)", "Cull backfaces"),
        ],
        default="AUTO",
    )
    emission_multiplier: FloatProperty(
        name="Glow / Emission Scale",
        description="Multiplier applied to reconstructed emission intensity",
        default=1.0,
        min=0.0,
        max=25.0,
    )
    base_color_param_idx: IntProperty(
        name="Tint Param Start Index",
        description="Index in Shader Params where RGB tint starts (-1 = automatic detection)",
        default=-1,
        min=-1,
        max=256,
    )
    use_custom_tint: BoolProperty(
        name="Override Tint Color",
        description="Manually override the Base Tint Color instead of reading from Shader Params",
        default=False,
    )
    custom_tint_color: FloatVectorProperty(
        name="Custom Tint",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
    )
    use_custom_emission: BoolProperty(
        name="Override Emission",
        description="Manually override Emission color and strength",
        default=False,
    )
    custom_emission_color: FloatVectorProperty(
        name="Emission Color",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(1.0, 0.72, 0.15),
    )
    custom_emission_strength: FloatProperty(
        name="Emission Strength",
        default=2.0,
        min=0.0,
        max=50.0,
    )
    opacity_override: FloatProperty(
        name="Opacity Override",
        description="Set >= 0.0 to override material opacity (-1.0 = automatic)",
        default=-1.0,
        min=-1.0,
        max=1.0,
    )
    flip_normal_green: BoolProperty(
        name="Flip Normal Y (DirectX -> OpenGL)",
        description="Invert green channel of DirectX normal maps for Blender",
        default=True,
    )


# =====================================================================
# MESH & CONTAINER PROPERTIES
# =====================================================================

class CGTMD2_ExtraHashItem(bpy.types.PropertyGroup):
    value: StringProperty(name="Extra Hash", default="0")


class CGTMD2_MeshProperties(bpy.types.PropertyGroup):
    has_name: BoolProperty(name="Has Name", default=False)
    has_hash: BoolProperty(name="Has Hash", default=False)
    has_extra: BoolProperty(name="Has Extra", default=False)
    name: StringProperty(name="Internal Name", default="")
    name_hash: StringProperty(name="Name Hash", default="0")
    unk: IntProperty(name="Unk", default=0)
    unk1: IntProperty(name="Unk 1", default=0)
    unk2: IntProperty(name="Unk 2", default=0)
    unk3: IntProperty(name="Unk 3", default=0)
    model_flag: IntProperty(name="Model Flag", default=0)
    lhs_flag: BoolProperty(name="LHS Flag", default=False)
    extra_hashes: CollectionProperty(type=CGTMD2_ExtraHashItem)
    active_extra_hash_index: IntProperty(default=0)


class CGTMD2_ObjectProperties(bpy.types.PropertyGroup):
    format: EnumProperty(
        name="Format",
        items=[("TMD2", "TMD2", "TamEngine TMD2"), ("TMD", "TMD", "Legacy TamEngine TMD")],
        default="TMD2",
    )
    version: EnumProperty(
        name="Version",
        items=[
            ("520", "2.08 (Captain Tsubasa / Modern)", "TMD2 v2.08 (0x208)"),
            ("521", "2.09 (Bleach / Latest)", "TMD2 v2.09 (0x209)"),
            ("519", "2.07", "TMD2 v2.07 (0x207)"),
            ("518", "2.06", "TMD2 v2.06 (0x206)"),
            ("517", "2.05", "TMD2 v2.05 (0x205)"),
            ("516", "2.04", "TMD2 v2.04 (0x204)"),
            ("515", "2.03", "TMD2 v2.03 (0x203)"),
            ("514", "2.02", "TMD2 v2.02 (0x202)"),
            ("513", "2.01", "TMD2 v2.01 (0x201)"),
        ],
        default="520",
    )
    flag1: IntProperty(name="Flag 1", default=0)
    flag2: IntProperty(name="Flag 2", default=0)
    flag3: IntProperty(name="Anim Flag (Flag 3)", default=0)
    after_image: IntProperty(name="After-Image Value", default=0)
    transform_animation: PointerProperty(name="Transform Animation", type=bpy.types.Action)


# =====================================================================
# UI LISTS
# =====================================================================

class CGTMD2_UL_Textures(bpy.types.UIList):
    bl_idname = "CGTMD2_UL_Textures"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "image", text="")
        row.label(text=f"S:{item.slot}")
        if item.is_external:
            row.label(text="EXT", icon="LINKED")


class CGTMD2_UL_Params(bpy.types.UIList):
    bl_idname = "CGTMD2_UL_Params"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"[{index:02d}]")
        row.prop(item, "value", text="")


# =====================================================================
# OPERATORS
# =====================================================================

class CGTMD2_OT_TexAdd(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.tex_add"
    bl_label = "Add Texture Slot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        props = mat.chatgpt_tmd2_material
        item = props.textures.add()
        item.slot = max(0, len(props.textures) - 1)
        props.active_texture_index = len(props.textures) - 1
        return {"FINISHED"}


class CGTMD2_OT_TexRemove(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.tex_remove"
    bl_label = "Remove Texture Slot"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        props = mat.chatgpt_tmd2_material
        idx = props.active_texture_index
        if 0 <= idx < len(props.textures):
            props.textures.remove(idx)
            props.active_texture_index = max(0, idx - 1)
        return {"FINISHED"}


class CGTMD2_OT_TexGenerateHash(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.tex_generate_hash"
    bl_label = "Hash Image Name"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        props = mat.chatgpt_tmd2_material
        idx = props.active_texture_index
        if not (0 <= idx < len(props.textures)):
            return {"CANCELLED"}
        tex_item = props.textures[idx]
        if not tex_item.image:
            self.report({"WARNING"}, "No image selected in this slot")
            return {"CANCELLED"}
        clean_name = tex_item.image.name.rsplit(".", 1)[0]
        tex_item.texture_hash = str(tamCRC32(clean_name))
        tex_item.resolved_name = clean_name
        return {"FINISHED"}


class CGTMD2_OT_ParamAdd(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.param_add"
    bl_label = "Add Shader Param"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        props = mat.chatgpt_tmd2_material
        props.param_values.add()
        props.active_param_index = len(props.param_values) - 1
        return {"FINISHED"}


class CGTMD2_OT_ParamRemove(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.param_remove"
    bl_label = "Remove Shader Param"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        props = mat.chatgpt_tmd2_material
        idx = props.active_param_index
        if 0 <= idx < len(props.param_values):
            props.param_values.remove(idx)
            props.active_param_index = max(0, idx - 1)
        return {"FINISHED"}


class CGTMD2_OT_MatGenerateHash(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.mat_generate_hash"
    bl_label = "Hash Material Name"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        clean_name = mat.name.split(".")[0]
        mat.chatgpt_tmd2_material.material_hash = str(tamCRC32(clean_name))
        return {"FINISHED"}


class CGTMD2_OT_RebuildMaterial(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.rebuild_material"
    bl_label = "Rebuild Shader Nodes"
    bl_description = "Reconstruct this material's Blender shader graph from its TMD2 data and fallback settings"
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(default="")

    def execute(self, context):
        mat = context.material
        if mat is None and context.active_object and context.active_object.active_material:
            mat = context.active_object.active_material
        if mat is None:
            self.report({"WARNING"}, "No active material found")
            return {"CANCELLED"}
        if self.preset:
            mat.chatgpt_tmd2_material.reconstruction_profile = self.preset
        analysis = rebuild_blender_material(mat, preset_override=self.preset or None)
        self.report(
            {"INFO"},
            f"Rebuilt '{mat.name}' -> Family: {analysis.shader_family}, Blend: {analysis.blend_mode}",
        )
        return {"FINISHED"}


class CGTMD2_OT_RebuildAllMaterials(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.rebuild_all_materials"
    bl_label = "Rebuild Selected / Scene Materials"
    bl_description = "Rebuild ChatGPT TMD2 shader nodes on selected meshes (or all TMD2 materials in scene)"
    bl_options = {"REGISTER", "UNDO"}

    preset: StringProperty(default="")
    only_selected: BoolProperty(name="Selected Objects Only", default=True)

    def execute(self, context):
        mats = set()
        objs = context.selected_objects if (self.only_selected and context.selected_objects) else context.scene.objects
        for obj in objs:
            if obj.type == "MESH":
                for slot in obj.material_slots:
                    if slot.material is not None:
                        mats.add(slot.material)
        if not mats:
            for m in bpy.data.materials:
                if hasattr(m, "chatgpt_tmd2_material") and (len(m.chatgpt_tmd2_material.textures) > 0 or len(m.chatgpt_tmd2_material.param_values) > 0):
                    mats.add(m)

        count = 0
        for mat in sorted(mats, key=lambda x: x.name):
            if self.preset:
                mat.chatgpt_tmd2_material.reconstruction_profile = self.preset
            rebuild_blender_material(mat, preset_override=self.preset or None)
            count += 1

        self.report({"INFO"}, f"Rebuilt {count} material(s)")
        return {"FINISHED"}


class CGTMD2_OT_SaveShaderRule(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.save_shader_rule"
    bl_label = "Save Default Rule for Shader ID"
    bl_description = "Save these reconstruction settings for this 4-char Shader ID so future imports apply them automatically"
    bl_options = {"REGISTER"}

    def execute(self, context):
        mat = context.material
        if mat is None and context.active_object:
            mat = context.active_object.active_material
        if mat is None:
            return {"CANCELLED"}
        p = mat.chatgpt_tmd2_material
        sid = (p.shader_id or "").strip()
        if not sid:
            self.report({"WARNING"}, "Shader ID is empty")
            return {"CANCELLED"}

        rule = {
            "family": p.reconstruction_profile,
            "blend_mode": p.transparency_mode,
            "base_color_idx": p.base_color_param_idx,
            "use_vertex_color": p.vertex_color_mode == "ON",
            "colorize_grayscale": p.grayscale_mode in ("AUTO", "COLORIZE"),
            "emission_multiplier": p.emission_multiplier,
        }
        save_custom_shader_rule(sid, rule)
        self.report({"INFO"}, f"Saved persistent shader rule for '{sid}'")
        return {"FINISHED"}


class CGTMD2_OT_DeleteShaderRule(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.delete_shader_rule"
    bl_label = "Reset Saved Shader Rule"
    bl_description = "Remove custom saved rule for this Shader ID and return to automatic data-driven detection"
    bl_options = {"REGISTER"}

    def execute(self, context):
        mat = context.material
        if mat is None and context.active_object:
            mat = context.active_object.active_material
        if mat is None:
            return {"CANCELLED"}
        sid = (mat.chatgpt_tmd2_material.shader_id or "").strip()
        if delete_custom_shader_rule(sid):
            self.report({"INFO"}, f"Removed custom rule for '{sid}'")
        else:
            self.report({"INFO"}, f"No saved custom rule existed for '{sid}'")
        return {"FINISHED"}


class _CGTMD2_MatClipboard:
    data = None


class CGTMD2_OT_CopyMaterial(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.copy_material"
    bl_label = "Copy Material Data"

    def execute(self, context):
        mat = context.material
        if not mat:
            return {"CANCELLED"}
        p = mat.chatgpt_tmd2_material
        _CGTMD2_MatClipboard.data = {
            "material_hash": p.material_hash,
            "shader_id": p.shader_id,
            "unk": p.unk,
            "reconstruction_profile": p.reconstruction_profile,
            "transparency_mode": p.transparency_mode,
            "vertex_color_mode": p.vertex_color_mode,
            "grayscale_mode": p.grayscale_mode,
            "emission_multiplier": p.emission_multiplier,
            "params": [(x.value, x.meta) for x in p.param_values],
            "textures": [
                {
                    "image": t.image,
                    "texture_hash": t.texture_hash,
                    "resolved_name": t.resolved_name,
                    "value1": t.value1,
                    "value2": t.value2,
                    "value3": t.value3,
                    "slot": t.slot,
                    "is_external": t.is_external,
                }
                for t in p.textures
            ],
        }
        self.report({"INFO"}, f"Copied TMD2 material data from '{mat.name}'")
        return {"FINISHED"}


class CGTMD2_OT_PasteMaterial(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.paste_material"
    bl_label = "Paste Material Data"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        mat = context.material
        d = _CGTMD2_MatClipboard.data
        if not mat or not d:
            self.report({"WARNING"}, "Clipboard is empty")
            return {"CANCELLED"}
        p = mat.chatgpt_tmd2_material
        p.material_hash = d["material_hash"]
        p.shader_id = d["shader_id"]
        p.unk = d["unk"]
        p.reconstruction_profile = d["reconstruction_profile"]
        p.transparency_mode = d["transparency_mode"]
        p.vertex_color_mode = d["vertex_color_mode"]
        p.grayscale_mode = d["grayscale_mode"]
        p.emission_multiplier = d["emission_multiplier"]
        p.param_values.clear()
        for val, meta in d["params"]:
            item = p.param_values.add()
            item.value = val
            item.meta = meta
        p.textures.clear()
        for t in d["textures"]:
            item = p.textures.add()
            item.image = t["image"]
            item.texture_hash = t["texture_hash"]
            item.resolved_name = t["resolved_name"]
            item.value1 = t["value1"]
            item.value2 = t["value2"]
            item.value3 = t["value3"]
            item.slot = t["slot"]
            item.is_external = t["is_external"]
        rebuild_blender_material(mat)
        self.report({"INFO"}, f"Pasted TMD2 material data & rebuilt '{mat.name}'")
        return {"FINISHED"}


class CGTMD2_OT_MeshGenerateHash(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.mesh_generate_hash"
    bl_label = "Hash Mesh Name"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.object
        if not obj:
            return {"CANCELLED"}
        clean_name = obj.name.split(".")[0]
        obj.chatgpt_tmd2_mesh.name_hash = str(tamCRC32(clean_name))
        return {"FINISHED"}


class CGTMD2_OT_ExtraHashAdd(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.extra_hash_add"
    bl_label = "Add Extra Hash"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.object
        if not obj:
            return {"CANCELLED"}
        props = obj.chatgpt_tmd2_mesh
        props.extra_hashes.add()
        props.active_extra_hash_index = len(props.extra_hashes) - 1
        return {"FINISHED"}


class CGTMD2_OT_ExtraHashRemove(bpy.types.Operator):
    bl_idname = "chatgpt_tmd2.extra_hash_remove"
    bl_label = "Remove Extra Hash"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.object
        if not obj:
            return {"CANCELLED"}
        props = obj.chatgpt_tmd2_mesh
        idx = props.active_extra_hash_index
        if 0 <= idx < len(props.extra_hashes):
            props.extra_hashes.remove(idx)
            props.active_extra_hash_index = max(0, idx - 1)
        return {"FINISHED"}


# =====================================================================
# UI PANELS
# =====================================================================

class MATERIAL_PT_chatgpt_tmd2_material(bpy.types.Panel):
    bl_label = "ChatGPT TMD2 Material"
    bl_idname = "MATERIAL_PT_chatgpt_tmd2_material"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def draw(self, context):
        layout = self.layout
        mat = context.material
        props = mat.chatgpt_tmd2_material

        # Header & Identity
        box_id = layout.box()
        row_cp = box_id.row(align=True)
        row_cp.operator("chatgpt_tmd2.copy_material", text="Copy", icon="COPYDOWN")
        row_cp.operator("chatgpt_tmd2.paste_material", text="Paste", icon="PASTEDOWN")
        row = box_id.row(align=True)
        row.prop(props, "material_hash")
        row.operator("chatgpt_tmd2.mat_generate_hash", text="", icon="FILE_REFRESH")
        row2 = box_id.row(align=True)
        row2.prop(props, "shader_id")
        row2.prop(props, "unk")
        status = box_id.row()
        status.label(text=f"Detected: {props.detected_family}  |  Blend: {props.detected_blend}", icon="SHADING_RENDERED")

        # Shader Reconstruction & Configurable Fallback Box
        box_rec = layout.box()
        box_rec.label(text="Shader Reconstruction & Fallback", icon="NODE_MATERIAL")
        box_rec.prop(props, "reconstruction_profile")
        box_rec.prop(props, "transparency_mode")
        row_vc = box_rec.row(align=True)
        row_vc.prop(props, "vertex_color_mode", text="Vertex Color")
        row_vc.prop(props, "grayscale_mode", text="Grayscale")
        row_bf = box_rec.row(align=True)
        row_bf.prop(props, "backface_culling_mode", text="Culling")
        row_bf.prop(props, "flip_normal_green", text="DX Normal Fix")

        row_num = box_rec.row(align=True)
        row_num.prop(props, "emission_multiplier")
        row_num.prop(props, "base_color_param_idx")
        box_rec.prop(props, "opacity_override")

        row_ct = box_rec.row(align=True)
        row_ct.prop(props, "use_custom_tint")
        if props.use_custom_tint:
            row_ct.prop(props, "custom_tint_color", text="")

        row_ce = box_rec.row(align=True)
        row_ce.prop(props, "use_custom_emission")
        if props.use_custom_emission:
            row_ce.prop(props, "custom_emission_color", text="")
            box_rec.prop(props, "custom_emission_strength")

        row_act = box_rec.row(align=True)
        op_reb = row_act.operator("chatgpt_tmd2.rebuild_material", text="Rebuild Material Nodes", icon="FILE_REFRESH")
        op_reb.preset = ""

        rules = load_custom_shader_rules()
        has_rule = (props.shader_id or "").strip() in rules
        row_rule = box_rec.row(align=True)
        row_rule.operator("chatgpt_tmd2.save_shader_rule", text="Save Rule for Shader ID", icon="PINNED")
        if has_rule:
            row_rule.operator("chatgpt_tmd2.delete_shader_rule", text="", icon="X")

        # Textures List
        box_tex = layout.box()
        box_tex.label(text=f"TMD2 Textures ({len(props.textures)})", icon="TEXTURE")
        row = box_tex.row()
        row.template_list("CGTMD2_UL_Textures", "", props, "textures", props, "active_texture_index", rows=3)
        col = row.column(align=True)
        col.operator("chatgpt_tmd2.tex_add", icon="ADD", text="")
        col.operator("chatgpt_tmd2.tex_remove", icon="REMOVE", text="")

        if 0 <= props.active_texture_index < len(props.textures):
            tex = props.textures[props.active_texture_index]
            sub = box_tex.column(align=True)
            r_h = sub.row(align=True)
            r_h.prop(tex, "texture_hash")
            r_h.operator("chatgpt_tmd2.tex_generate_hash", text="", icon="FILE_REFRESH")
            if tex.resolved_name:
                sub.label(text=f"Resolved Name: {tex.resolved_name}", icon="CHECKMARK")
            r_s = sub.row(align=True)
            r_s.prop(tex, "slot")
            r_s.prop(tex, "is_external")
            r_u = sub.row(align=True)
            r_u.prop(tex, "value1")
            r_u.prop(tex, "value2")
            r_u.prop(tex, "value3")

        # Shader Params List
        box_prm = layout.box()
        box_prm.label(text=f"Shader Constant Params ({len(props.param_values)})", icon="PROPERTIES")
        row = box_prm.row()
        row.template_list("CGTMD2_UL_Params", "", props, "param_values", props, "active_param_index", rows=4)
        col = row.column(align=True)
        col.operator("chatgpt_tmd2.param_add", icon="ADD", text="")
        col.operator("chatgpt_tmd2.param_remove", icon="REMOVE", text="")


class OBJECT_PT_chatgpt_tmd2_mesh(bpy.types.Panel):
    bl_label = "ChatGPT TMD2 Mesh"
    bl_idname = "OBJECT_PT_chatgpt_tmd2_mesh"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "MESH"

    def draw(self, context):
        layout = self.layout
        props = context.object.chatgpt_tmd2_mesh

        layout.prop(props, "has_name")
        if props.has_name:
            layout.prop(props, "name")

        layout.prop(props, "has_hash")
        if props.has_hash:
            row = layout.row(align=True)
            row.prop(props, "name_hash")
            row.operator("chatgpt_tmd2.mesh_generate_hash", text="", icon="FILE_REFRESH")

        layout.prop(props, "has_extra")
        if props.has_extra:
            col_e = layout.column(align=True)
            col_e.prop(props, "unk1")
            col_e.prop(props, "unk2")
            col_e.prop(props, "unk3")

        layout.prop(props, "model_flag")
        layout.prop(props, "lhs_flag")

        box = layout.box()
        box.label(text="Extra Hashes")
        row = box.row()
        col = row.column()
        for i, item in enumerate(props.extra_hashes):
            col.prop(item, "value", text=f"Hash {i + 1}")
        col_btn = row.column(align=True)
        col_btn.operator("chatgpt_tmd2.extra_hash_add", icon="ADD", text="")
        col_btn.operator("chatgpt_tmd2.extra_hash_remove", icon="REMOVE", text="")


class OBJECT_PT_chatgpt_tmd2_panel(bpy.types.Panel):
    bl_label = "ChatGPT TMD2 Container"
    bl_idname = "OBJECT_PT_chatgpt_tmd2_panel"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "object"

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "EMPTY" and obj.name.startswith("#CGTMD2 PROPERTIES")

    def draw(self, context):
        layout = self.layout
        props = context.object.chatgpt_tmd2_props
        layout.prop(props, "format")
        if props.format == "TMD2":
            layout.prop(props, "version")
            layout.prop(props, "flag1")
            layout.prop(props, "flag2")
        layout.prop(props, "flag3")
        layout.prop(props, "after_image")
        layout.prop(props, "transform_animation")


class VIEW3D_PT_chatgpt_tmd2_tools(bpy.types.Panel):
    bl_label = "ChatGPT TMD2"
    bl_idname = "VIEW3D_PT_chatgpt_tmd2_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "ChatGPT TMD2"

    def draw(self, context):
        layout = self.layout

        box_io = layout.box()
        box_io.label(text="Model & Container I/O", icon="IMPORT")
        box_io.operator("import_scene.chatgpt_tmd2", text="Import TMD2 / LDS / CAT / TMO", icon="IMPORT")
        box_io.operator("export_scene.chatgpt_tmd2", text="Export TMD2 / TMD", icon="EXPORT")

        box_sh = layout.box()
        box_sh.label(text="Quick Shader Presets (Selected)", icon="SHADING_RENDERED")
        col = box_sh.column(align=True)

        op = col.operator("chatgpt_tmd2.rebuild_all_materials", text="Auto Reconstruct (Data-Driven)", icon="FILE_REFRESH")
        op.preset = "AUTO"
        op.only_selected = True

        op = col.operator("chatgpt_tmd2.rebuild_all_materials", text="Glowing VFX / Golden Aura", icon="LIGHT_SUN")
        op.preset = "VFX_ADDITIVE"
        op.only_selected = True

        op = col.operator("chatgpt_tmd2.rebuild_all_materials", text="Ghost / Spirit (See-Through)", icon="GHOST_ENABLED")
        op.preset = "GHOST_SPIRIT"
        op.only_selected = True

        op = col.operator("chatgpt_tmd2.rebuild_all_materials", text="Metallic / Gold Sheen", icon="NODE_MATERIAL")
        op.preset = "METALLIC_GLOW"
        op.only_selected = True

        op = col.operator("chatgpt_tmd2.rebuild_all_materials", text="Anime / Cel Character", icon="OUTLINER_OB_ARMATURE")
        op.preset = "TOON_CHAR"
        op.only_selected = True


PANEL_CLASSES = (
    CGTMD2_TextureItem,
    CGTMD2_ParamItem,
    CGTMD2_MaterialProperties,
    CGTMD2_ExtraHashItem,
    CGTMD2_MeshProperties,
    CGTMD2_ObjectProperties,
    CGTMD2_UL_Textures,
    CGTMD2_UL_Params,
    CGTMD2_OT_TexAdd,
    CGTMD2_OT_TexRemove,
    CGTMD2_OT_TexGenerateHash,
    CGTMD2_OT_ParamAdd,
    CGTMD2_OT_ParamRemove,
    CGTMD2_OT_MatGenerateHash,
    CGTMD2_OT_RebuildMaterial,
    CGTMD2_OT_RebuildAllMaterials,
    CGTMD2_OT_SaveShaderRule,
    CGTMD2_OT_DeleteShaderRule,
    CGTMD2_OT_CopyMaterial,
    CGTMD2_OT_PasteMaterial,
    CGTMD2_OT_MeshGenerateHash,
    CGTMD2_OT_ExtraHashAdd,
    CGTMD2_OT_ExtraHashRemove,
    MATERIAL_PT_chatgpt_tmd2_material,
    OBJECT_PT_chatgpt_tmd2_mesh,
    OBJECT_PT_chatgpt_tmd2_panel,
    VIEW3D_PT_chatgpt_tmd2_tools,
)


def register():
    for cls in PANEL_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Material.chatgpt_tmd2_material = PointerProperty(type=CGTMD2_MaterialProperties)
    bpy.types.Object.chatgpt_tmd2_mesh = PointerProperty(type=CGTMD2_MeshProperties)
    bpy.types.Object.chatgpt_tmd2_props = PointerProperty(type=CGTMD2_ObjectProperties)


def unregister():
    if hasattr(bpy.types.Object, "chatgpt_tmd2_props"):
        del bpy.types.Object.chatgpt_tmd2_props
    if hasattr(bpy.types.Object, "chatgpt_tmd2_mesh"):
        del bpy.types.Object.chatgpt_tmd2_mesh
    if hasattr(bpy.types.Material, "chatgpt_tmd2_material"):
        del bpy.types.Material.chatgpt_tmd2_material
    for cls in reversed(PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
