# ChatGPT TMD2

**ChatGPT TMD2** is a standalone, enhanced Blender 4.2+ Extension / Add-on derived from `BlenderTMD2`, built specifically to reconstruct accurate game materials, shaders, transparency, emission/glow, grayscale colorization, multi-UV textures, and vertex colors for **Captain Tsubasa: Rise of New Champions** and other Tamsoft TamEngine `.tmd2` / `.tmd` models.

It is completely isolated from the original `BlenderTMD2` add-on (`id = "chatgpt_tmd2"`, `CGTMD2_*` classes, `chatgpt_tmd2.*` operators, and `chatgpt_tmd2_*` RNA properties), allowing **both add-ons to remain installed and enabled side-by-side** in Blender without any conflicts.

## Key Enhancements over BlenderTMD2

1. **Data-Driven Game Shader Reconstruction (`materials/analyzer.py` & `materials/shaders.py`)**
   - Extracts real RGBA base tints, secondary gradient colors, HDR emission/glow intensities (`> 1.0`), Fresnel rim glow, metallic/roughness, opacity, and UV scale/offset directly from `tm_mat.shaderParams`.
   - Reconstructs **Glowing VFX / Shot Auras** (e.g., Golden Glowing Tiger), **Ghost / Spirit Translucent Models** (e.g., Red Ghostly Spirit with Fresnel rim translucency), **See-Through Alpha-Blended / Cutout Parts**, **Metallic / Gold Sheen**, and **Anime / Cel Character Shaders**.
   - **Grayscale Texture Colorization & 1D/2D Ramp LUTs**: Automatically detects grayscale textures and colorizes them via extracted `shaderParams` or companion Ramp LUT textures (`EXTEND` UV lookup).
   - **Luminance-to-Alpha for Dark-Background VFX**: Automatically converts black-background additive energy textures into smooth alpha + emission so models behind the effect remain visible.
   - **Dual Render Engine Output**: Generates dedicated `EEVEE` (`ShaderToRGB` cel-shading + additive/alpha VFX) and `Cycles` (`Principled BSDF` + `Emission` + `Transparent BSDF`) outputs simultaneously.
   - **DirectX Normal Map Fix & BC5 Z-Channel Reconstruction**: Automatically flips DirectX green (`Y`) channel for Blender and reconstructs `Z = sqrt(max(0, 1 - X^2 - Y^2))` for 2-channel BC5 normal maps.

2. **Multi-Texture, Multi-UV, Vertex Color, & `.vfxt` Support**
   - Resolves texture slots via explicit `tmat_texture.slot` (`uint8` offset `+11`), `tm_texture.index` against `.lds` `images_list`, `tamCRC32` hash lookup (`hashes.json`), `.vfxt` VFX texture archives, and loose `.dds` files.
   - Safely handles external/stage textures (`textureIndex == -1`) without out-of-bounds indexing or aliasing `images_list[0]`.
   - Connects `UVMap`, `UVMap1`, `UVMap2` and per-submesh `Color` / `Color2` vertex RGB & Alpha attributes.

3. **Mesh Integrity & Binary Parser Fixes**
   - Fixes `TMD2Bone.nameOffset == 0xFFFF` out-of-bounds crash on TMD2 `< 0x209` models (used in *Captain Tsubasa: Rise of New Champions*).
   - Preserves authored opposite-winding double-sided VFX/cloth triangles without dropping backfaces or desynchronizing `foreach_set` UV/color loop arrays.
   - Preserves 8-byte `shaderParamMetas`, `unk3`, `slot`, `isExternal`, and exact `texture_hash` values on lossless round-trip export.

4. **Configurable Fallback & Live Rebuild UI**
   - Configure fallback behavior directly in the Import dialog, **Material Properties > ChatGPT TMD2 Material**, or **3D Viewport > N-Panel > ChatGPT TMD2**.
   - Save persistent per-`shaderID` rules (`Save Rule for Shader ID`) to `custom_shader_rules.json` for automatic future imports.
