# ChatGPT TMD2

An independent Blender add-on for importing and exporting **Tamsoft TMD/TMD2**
models with a much more faithful reconstruction of the in-game materials.
It is designed for *Captain Tsubasa: Rise of New Champions* but works for every
game that uses the TamEngine TMD2 format.

The add-on is **completely separate** from the original
[BlenderTMD2](https://github.com/Madrhyan/BlenderTMD2) add-on and can be enabled
at the same time as it.

## What is different from the original add-on?

The original importer only assigned textures to the material prototypes stored in
`materials.blend` and ignored most of the data stored in the material block.
ChatGPT TMD2 reads and applies:

* **shader parameters** (`shaderParams`) - colours *and* scalars, mapped per
  shader family (body, hair, skin, effects, unlit) instead of only four indices
* **material colours** - shadow / specular / hair base / hair shine / blood
  colours are written into the shader node group, never guessed
* **alpha** - the clip threshold parameter and the texture alpha channel drive the
  blend mode (opaque / clip / hashed / blend)
* **emission** - HDR style colour parameters and unlit effect shaders create a
  real emission contribution
* **vertex colours** - `color` *and* `color2` are imported as `Color` / `Color2`
  byte colour attributes and connected to the shader's vertex colour inputs
* **multi texture materials** - up to six textures per material, with the correct
  colour space (sRGB vs Non-Color) and the correct UV map per slot
* **UV sets** - up to three UV layers are kept, and the hair shaders sample the
  second UV set where the game does
* **texture index resolution** - material textures are resolved through the real
  `TMD2Texture.index` / hash table instead of assuming list order
* **unknown shaders** - a configurable fallback system (auto / simple PBR /
  diffuse / unlit / placeholder) instead of silently using one default material

## Installation

1. Download `release/chatgpt_tmd2-1.0.0.zip` (or build it with
   `python tools/build_addon_zip.py` from the repository root).
2. In Blender: `Edit > Preferences > Add-ons > Install from Disk...`
3. Select the zip file and enable **ChatGPT TMD2**.
4. Alternatively `Edit > Preferences > Get Extensions > Install from Disk...`
   (the zip contains a `blender_manifest.toml` with the id `chatgpt_tmd2`).

Both installation paths work: the archive contains a single top level folder
`chatgpt_tmd2/` with `__init__.py` and `blender_manifest.toml` inside.

## Usage

* `File > Import > Tamsoft TMD2 (ChatGPT TMD2)`
* Drag & drop `.tmd2` files into the viewport
* `File > Export > Tamsoft TMD2 (ChatGPT TMD2)`
* Sidebar (`N` panel) `ChatGPT TMD2` tab for the material options
* `Properties > Scene > ChatGPT TMD2` for the global interpretation settings
* `Properties > Material > ChatGPT TMD2 Material` to inspect raw parameters and
  rebuild a single material

Textures are searched next to the model (`.lds` with the same name). Loose `.dds`
files can be selected in the import dialog.

## Options

| Option | Meaning |
| --- | --- |
| Unknown Shader Fallback | What to build when a shader ID has no implementation |
| Alpha Handling | Force a blend mode instead of using the data from the file |
| Vertex Color | When the base colour is multiplied with the mesh colour attributes |
| Emission Boost | Multiplier for emission derived from the material data |
| Import Vertex Colors | Create `Color`/`Color2` attributes on the mesh |
| Import Custom Normals | Use the normals stored in the TMD2 vertices |

## Testing status

The add-on was developed and validated head-lessly with the `bpy` 4.2 Python
module against **synthetic** TMD2/LDS files (a correctly structured file with a
valid header, material block, vertex/index buffers and DXT1 textures).  The
repository contains no real *Captain Tsubasa* asset, so the following could not
be verified here:

* rendering the imported materials in EEVEE/Cycles (no GPU or GL context in the
  test environment)
* the visual result on real game models and real shader parameter values
* exporting a real game file back into the game

What *was* verified automatically (`tests/selftest.py`, 132 checks): registration
and identifier isolation from the original add-on, registration/unregistration
symmetry, import of a synthetic file (meshes, UV sets, vertex colours, textures,
material colour/alpha/emission values), material rebuild, panel drawing, and an
import -> export -> re-import round trip including the `.lds` texture archive.

Run it with:

```
LD_LIBRARY_PATH=<dir with GL stubs> python chatgpt_tmd2/tests/selftest.py
```

## Compatibility

* Blender 4.2 and newer (tested against the Blender 4.2 Python API)
* The original `BlenderTMD2` add-on can stay installed and enabled

## Credits

Material prototypes (`materials/materials.blend`) and the TMD2 parser come from
the original BlenderTMD2 project by Al-Hydra (GPL-3.0-or-later).  The material
interpretation layer, the importer, the exporter and the UI are new.
