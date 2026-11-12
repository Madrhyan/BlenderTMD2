"""ChatGPT TMD2 - an independent, more faithful TMD2 importer/exporter.

This add-on is a *separate* package from the original ``BlenderTMD2`` add-on and
is designed to be installed and enabled at the same time.  Nothing in here
registers an identifier that the original add-on also registers:

============================  ==============================
item                          identifier
============================  ==============================
package                       ``chatgpt_tmd2``
extension id                  ``chatgpt_tmd2``
scene settings                ``scene.ctmd2_settings``
material properties           ``Material.ctmd2_material``
object (mesh) properties      ``Object.ctmd2_mesh``
object (file) properties      ``Object.ctmd2_props``
import operator               ``ctmd2.import_tmd2``
export operator               ``ctmd2.export_tmd2``
drop operator                 ``ctmd2.drop_tmd2``
file handlers                 ``CTMD2_FH_import`` / ``CTMD2_FH_import_tmd``
panels                        ``CTMD2_PT_*``
============================  ==============================
"""

bl_info = {
    "name": "ChatGPT TMD2",
    "author": "ChatGPT TMD2 (based on BlenderTMD2 by Al-Hydra)",
    "description": "Improved importer/exporter for Tamsoft TMD2 models with "
                   "faithful material, alpha, emission and vertex colour support",
    "blender": (4, 2, 0),
    "version": (1, 0, 0),
    "location": "File > Import / View3D > Sidebar > ChatGPT TMD2",
    "warning": "",
    "category": "Import-Export",
}

import importlib
import sys

import bpy
from bpy.props import PointerProperty

from .core import material_builder, mesh_builder, shader_db, textures  # noqa: F401
from .core import props as ctmd2_props
from .importers import importer as ctmd2_importer
from .exporters import exporter as ctmd2_exporter
from .ui import menus as ctmd2_menus


MODULE_VERSION = (1, 0, 0)

_CLASSES = (
    ctmd2_props.property_groups
    + ctmd2_props.operators
    + ctmd2_props.ui_lists
    + ctmd2_props.panels
    + [
        ctmd2_importer.CTMD2_OT_ImportTMD2,
        ctmd2_importer.CTMD2_OT_DropTMD2,
        ctmd2_importer.CTMD2_FH_import,
        ctmd2_importer.CTMD2_FH_import_tmd,
        ctmd2_exporter.CTMD2_OT_ExportTMD2,
    ]
    + list(ctmd2_menus.classes)
)


def _reload_submodules():
    """Support Blender's "Reload Scripts" while keeping import order sane."""
    package = __name__
    order = [
        "ctamlib",
        "ctamlib.utils",
        "ctamlib.utils.PyBinaryReader",
        "ctamlib.utils.PyBinaryReader.binary_reader",
        "ctamlib.pzze",
        "ctamlib.tmd",
        "ctamlib.tmd2",
        "ctamlib.tmd0",
        "ctamlib.tmdv",
        "ctamlib.lds",
        "core.shader_db",
        "core.textures",
        "core.material_builder",
        "core.mesh_builder",
        "core.props",
        "importers.reader",
        "importers.importer",
        "exporters.writer",
        "exporters.common",
        "exporters.exporter_core",
        "exporters.exporter",
        "ui.menus",
    ]
    for suffix in order:
        name = f"{package}.{suffix}"
        module = sys.modules.get(name)
        if module is not None:
            try:
                importlib.reload(module)
            except Exception as exc:  # pragma: no cover
                print(f"[ChatGPT TMD2] reload failed for {name}: {exc}")


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Material.ctmd2_material = PointerProperty(type=ctmd2_props.CTMD2MaterialProperties)
    bpy.types.Object.ctmd2_mesh = PointerProperty(type=ctmd2_props.CTMD2MeshProperties)
    bpy.types.Object.ctmd2_props = PointerProperty(type=ctmd2_props.CTMD2Properties)
    bpy.types.Scene.ctmd2_settings = PointerProperty(type=ctmd2_props.CTMD2Settings)

    bpy.types.TOPBAR_MT_file_import.append(ctmd2_importer.menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(ctmd2_exporter.menu_func_export)

    print(f"[ChatGPT TMD2] registered {len(_CLASSES)} classes")


def unregister():
    try:
        bpy.types.TOPBAR_MT_file_import.remove(ctmd2_importer.menu_func_import)
    except Exception:
        pass
    try:
        bpy.types.TOPBAR_MT_file_export.remove(ctmd2_exporter.menu_func_export)
    except Exception:
        pass

    for attr, owner in (
        ("ctmd2_settings", bpy.types.Scene),
        ("ctmd2_props", bpy.types.Object),
        ("ctmd2_mesh", bpy.types.Object),
        ("ctmd2_material", bpy.types.Material),
    ):
        if hasattr(owner, attr):
            delattr(owner, attr)

    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as exc:  # pragma: no cover
            print(f"[ChatGPT TMD2] unregister failed for {cls}: {exc}")


if __name__ == "__main__":
    register()
