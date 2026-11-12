"""Menus and panels for ChatGPT TMD2."""

from bpy.types import Panel

from ..core.props import active_material


class CTMD2_PT_MainPanel(Panel):
    bl_idname = "CTMD2_PT_main"
    bl_label = "ChatGPT TMD2"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ChatGPT TMD2"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        settings = getattr(scene, "ctmd2_settings", None)

        col = layout.column(align=True)
        col.operator("ctmd2.import_tmd2", text="Import TMD2", icon='IMPORT')
        col.operator("ctmd2.export_tmd2", text="Export Collection", icon='EXPORT')

        if settings is not None:
            box = layout.box()
            box.label(text="Material Shading", icon='MATERIAL')
            box.prop(settings, "fallback_shader")
            box.prop(settings, "alpha_mode")
            box.prop(settings, "vertex_color_mode")
            box.prop(settings, "emission_boost")

            box = layout.box()
            box.label(text="Import", icon='OPTIONS')
            box.prop(settings, "import_vertex_colors")
            box.prop(settings, "auto_smooth")
            box.prop(settings, "shader_report")

        box = layout.box()
        box.label(text="Selected Material", icon='SHADING_TEXTURE')
        mat = active_material(context)
        props = getattr(mat, "ctmd2_material", None) if mat else None
        if props is None:
            box.label(text="No TMD2 material selected", icon='INFO')
        else:
            col = box.column(align=True)
            col.label(text=f"Shader: {props.shader_id or '?'}  ({props.shader_detected})")
            col.label(text=f"Blend: {props.detected_blend}")
            col.label(text=f"Alpha: {props.alpha_value:.2f}  clip {props.alpha_clip_threshold:.2f}")
            if props.uses_emission:
                col.label(text=f"Emission x{props.emission_strength:.2f}")
            col.label(text="Vertex Color: " + ("yes" if props.uses_vertex_color else "no"))
            row = box.row(align=True)
            row.operator("ctmd2.rebuild_material", icon='FILE_REFRESH', text="Rebuild")
            row.operator("ctmd2.print_material_report", icon='CONSOLE', text="Report")


classes = [CTMD2_PT_MainPanel]
