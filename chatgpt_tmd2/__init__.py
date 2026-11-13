bl_info = {
    "name": "ChatGPT TMD2",
    "author": "ChatGPT TMD2 Contributors (derived from BlenderTMD2 by HydraBladeZ)",
    "description": "Standalone enhanced TamEngine TMD2/TMD/LDS/VFXT importer, exporter, and data-driven shader reconstructor for Captain Tsubasa: Rise of New Champions & TamEngine games",
    "blender": (4, 2, 0),
    "version": (2, 1, 0),
    "location": "File > Import-Export > ChatGPT TMD2 | View3D > N-Panel > ChatGPT TMD2",
    "warning": "",
    "category": "Import-Export",
}

import importlib

from . import exporter, importer, panels


def register():
    panels.register()
    importer.register()
    exporter.register()


def unregister():
    exporter.unregister()
    importer.unregister()
    panels.unregister()


if __name__ == "__main__":
    register()
