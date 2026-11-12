"""Exporter for ChatGPT TMD2.

The export path re-uses the original, battle tested writer from ``ctamlib.tmd2``
but reads its data from the ``ctmd2_*`` property groups, so it never touches the
properties owned by the original add-on.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper


class CTMD2_OT_ExportTMD2(Operator, ExportHelper):
    """Export the selected collection as a Tamsoft TMD2 file (ChatGPT TMD2)"""

    bl_idname = "ctmd2.export_tmd2"
    bl_label = "Export TMD2 (ChatGPT TMD2)"
    bl_options = {'REGISTER'}

    filename_ext = ""
    directory: StringProperty(subtype='DIR_PATH', options={'HIDDEN', 'SKIP_SAVE'})  # type: ignore
    filepath: StringProperty(subtype='FILE_PATH')  # type: ignore

    collection: StringProperty(name="Collection", description="Collection to export")  # type: ignore
    tmd_version: EnumProperty(
        name="Version",
        items=[
            ('0x209', "TMD2 0x209", ""),
            ('0x208', "TMD2 0x208", ""),
            ('0x207', "TMD2 0x207", ""),
            ('0x206', "TMD2 0x206", ""),
            ('0x205', "TMD2 0x205", ""),
            ('0x204', "TMD2 0x204", ""),
        ],
        default='0x209',
    )  # type: ignore
    export_textures: BoolProperty(name="Export Textures", default=True)  # type: ignore
    compress_files: BoolProperty(name="Compress (PZZE)", default=True)  # type: ignore
    export_original_bone_data: BoolProperty(name="Use Original Bone Data", default=True)  # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.prop_search(self, "collection", bpy.data, "collections")
        layout.prop(self, "tmd_version")
        layout.prop(self, "export_textures")
        layout.prop(self, "export_original_bone_data")
        layout.prop(self, "compress_files")

    def invoke(self, context, event):
        if not self.collection:
            self.collection = bpy.context.collection.name
        self.filepath = self.collection or "untitled"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        collection = bpy.data.collections.get(self.collection)
        if collection is None:
            self.report({'ERROR'}, f"Collection '{self.collection}' not found")
            return {'CANCELLED'}
        try:
            from .exporter_core import export_collection
            export_collection(
                collection, self.filepath,
                version=int(self.tmd_version, 16),
                export_textures=self.export_textures,
                compress=self.compress_files,
                original_bones=self.export_original_bone_data,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Export failed: {exc}")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported {collection.name} to {self.filepath}.tmd2")
        return {'FINISHED'}


def menu_func_export(self, context):
    self.layout.operator(CTMD2_OT_ExportTMD2.bl_idname,
                         text='Tamsoft TMD2 (ChatGPT TMD2)',
                         icon='EXPORT')
