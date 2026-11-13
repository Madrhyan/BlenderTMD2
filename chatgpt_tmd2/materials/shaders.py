import os
from typing import Dict, List, Optional, Tuple

import bpy

from .analyzer import (
    MaterialAnalysis,
    SubmeshUsageInfo,
    TextureSlotBinding,
    analyze_tmd_material,
    inspect_blender_image,
)

MASTER_GROUP_NAME = "CGTMD2_Universal_Shader_v1"
NORMAL_FIX_GROUP_NAME = "CGTMD2_DX_Normal_Fix_v1"


# ---------------------------------------------------------------------------
# Helper: DirectX Normal Map Fix + Optional BC5 Z Reconstruction Node Group
# ---------------------------------------------------------------------------

def ensure_normal_fix_node_group() -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.get(NORMAL_FIX_GROUP_NAME)
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(NORMAL_FIX_GROUP_NAME, "ShaderNodeTree")
    iface = ng.interface
    s_col = iface.new_socket("Color", in_out="INPUT", socket_type="NodeSocketColor")
    s_col.default_value = (0.5, 0.5, 1.0, 1.0)
    s_flip = iface.new_socket("Flip Green (DX)", in_out="INPUT", socket_type="NodeSocketFloat")
    s_flip.default_value = 1.0
    s_flip.min_value = 0.0
    s_flip.max_value = 1.0
    s_rec = iface.new_socket("Reconstruct Z (BC5)", in_out="INPUT", socket_type="NodeSocketFloat")
    s_rec.default_value = 0.0
    s_rec.min_value = 0.0
    s_rec.max_value = 1.0
    iface.new_socket("Color", in_out="OUTPUT", socket_type="NodeSocketColor")

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-800, 0)
    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (650, 0)

    sep = nodes.new("ShaderNodeSeparateColor")
    sep.location = (-580, 0)
    links.new(n_in.outputs["Color"], sep.inputs["Color"])

    # Flip Green: mix(G, 1 - G, Flip Green)
    inv_g = nodes.new("ShaderNodeMath")
    inv_g.operation = "SUBTRACT"
    inv_g.inputs[0].default_value = 1.0
    inv_g.location = (-360, 80)
    links.new(sep.outputs["Green"], inv_g.inputs[1])

    mix_g = nodes.new("ShaderNodeMix")
    mix_g.data_type = "FLOAT"
    mix_g.location = (-160, 120)
    links.new(n_in.outputs["Flip Green (DX)"], mix_g.inputs[0])
    links.new(sep.outputs["Green"], mix_g.inputs[2])
    links.new(inv_g.outputs[0], mix_g.inputs[3])

    # Reconstruct Z from X=2R-1, Y=2G'-1 -> Z = sqrt(max(0, 1 - X^2 - Y^2)) -> B = 0.5*Z + 0.5
    nx = nodes.new("ShaderNodeMath")
    nx.operation = "MULTIPLY_ADD"
    nx.inputs[1].default_value = 2.0
    nx.inputs[2].default_value = -1.0
    nx.location = (-360, -120)
    links.new(sep.outputs["Red"], nx.inputs[0])

    ny = nodes.new("ShaderNodeMath")
    ny.operation = "MULTIPLY_ADD"
    ny.inputs[1].default_value = 2.0
    ny.inputs[2].default_value = -1.0
    ny.location = (20, -40)
    links.new(mix_g.outputs[0], ny.inputs[0])

    nx2 = nodes.new("ShaderNodeMath")
    nx2.operation = "MULTIPLY"
    nx2.location = (-160, -160)
    links.new(nx.outputs[0], nx2.inputs[0])
    links.new(nx.outputs[0], nx2.inputs[1])

    ny2 = nodes.new("ShaderNodeMath")
    ny2.operation = "MULTIPLY"
    ny2.location = (180, -80)
    links.new(ny.outputs[0], ny2.inputs[0])
    links.new(ny.outputs[0], ny2.inputs[1])

    sum_xy = nodes.new("ShaderNodeMath")
    sum_xy.operation = "ADD"
    sum_xy.location = (180, -220)
    links.new(nx2.outputs[0], sum_xy.inputs[0])
    links.new(ny2.outputs[0], sum_xy.inputs[1])

    one_minus = nodes.new("ShaderNodeMath")
    one_minus.operation = "SUBTRACT"
    one_minus.inputs[0].default_value = 1.0
    one_minus.use_clamp = True
    one_minus.location = (340, -220)
    links.new(sum_xy.outputs[0], one_minus.inputs[1])

    nz = nodes.new("ShaderNodeMath")
    nz.operation = "SQRT"
    nz.location = (340, -80)
    links.new(one_minus.outputs[0], nz.inputs[0])

    b_rec = nodes.new("ShaderNodeMath")
    b_rec.operation = "MULTIPLY_ADD"
    b_rec.inputs[1].default_value = 0.5
    b_rec.inputs[2].default_value = 0.5
    b_rec.location = (340, 60)
    links.new(nz.outputs[0], b_rec.inputs[0])

    mix_b = nodes.new("ShaderNodeMix")
    mix_b.data_type = "FLOAT"
    mix_b.location = (480, -60)
    links.new(n_in.outputs["Reconstruct Z (BC5)"], mix_b.inputs[0])
    links.new(sep.outputs["Blue"], mix_b.inputs[2])
    links.new(b_rec.outputs[0], mix_b.inputs[3])

    comb = nodes.new("ShaderNodeCombineColor")
    comb.location = (500, 120)
    links.new(sep.outputs["Red"], comb.inputs["Red"])
    links.new(mix_g.outputs[0], comb.inputs["Green"])
    links.new(mix_b.outputs[0], comb.inputs["Blue"])

    links.new(comb.outputs["Color"], n_out.inputs["Color"])
    return ng


# ---------------------------------------------------------------------------
# Master Dual-Engine (EEVEE Toon/VFX + Cycles PBR/VFX) Shader Node Group
# ---------------------------------------------------------------------------

def ensure_master_shader_node_group() -> bpy.types.NodeTree:
    ng = bpy.data.node_groups.get(MASTER_GROUP_NAME)
    if ng is not None:
        return ng

    ng = bpy.data.node_groups.new(MASTER_GROUP_NAME, "ShaderNodeTree")
    iface = ng.interface

    def add_col(name: str, default: Tuple[float, float, float, float]):
        s = iface.new_socket(name, in_out="INPUT", socket_type="NodeSocketColor")
        s.default_value = default
        return s

    def add_flt(name: str, default: float, min_v: float = 0.0, max_v: float = 1.0):
        s = iface.new_socket(name, in_out="INPUT", socket_type="NodeSocketFloat")
        s.default_value = default
        s.min_value = min_v
        s.max_value = max_v
        return s

    # Inputs
    add_col("Base Texture", (1.0, 1.0, 1.0, 1.0))
    add_flt("Base Texture Alpha", 1.0, 0.0, 1.0)
    add_col("Base Tint Color", (1.0, 1.0, 1.0, 1.0))
    add_col("Secondary Color", (0.5, 0.5, 0.5, 1.0))
    add_flt("Grayscale Colorize", 0.0, 0.0, 1.0)
    add_col("Ramp / Sub Texture", (1.0, 1.0, 1.0, 1.0))
    add_flt("Ramp Mix", 0.0, 0.0, 1.0)
    add_col("Vertex Color", (1.0, 1.0, 1.0, 1.0))
    add_flt("Vertex Color Mix", 0.0, 0.0, 1.0)
    add_flt("Vertex Alpha", 1.0, 0.0, 1.0)
    add_flt("Vertex Alpha Mix", 0.0, 0.0, 1.0)
    add_col("ILM / Shadow Mask", (0.5, 0.0, 1.0, 1.0))
    add_flt("Use ILM Mask", 0.0, 0.0, 1.0)
    add_col("Specular / Emissive Tex", (0.0, 0.0, 0.0, 1.0))
    add_flt("Use Specular Tex", 0.0, 0.0, 1.0)
    add_col("Detail Mask Tex", (1.0, 1.0, 1.0, 1.0))
    add_flt("Use Detail Mask", 0.0, 0.0, 1.0)
    add_flt("Toon Shading Mix", 0.65, 0.0, 1.0)
    add_flt("Shadow Intensity", 0.25, 0.0, 1.5)
    add_col("Shadow Color", (0.45, 0.42, 0.48, 1.0))
    add_col("Specular Color", (0.5, 0.5, 0.5, 1.0))
    add_flt("Specular Strength", 0.25, 0.0, 5.0)
    add_flt("Metallic", 0.0, 0.0, 1.0)
    add_flt("Roughness", 0.45, 0.0, 1.0)
    add_col("Rim Color", (1.0, 1.0, 1.0, 1.0))
    add_flt("Rim Strength", 0.0, 0.0, 10.0)
    add_flt("Rim Power", 3.0, 0.2, 16.0)
    add_col("Emission Color", (0.0, 0.0, 0.0, 1.0))
    add_flt("Emission Strength", 0.0, 0.0, 50.0)
    add_flt("Emission From Albedo", 0.85, 0.0, 1.0)
    add_flt("Material Opacity", 1.0, 0.0, 1.0)
    add_flt("Alpha Clip Threshold", 0.5, 0.0, 1.0)
    add_flt("Use Alpha Clip", 0.0, 0.0, 1.0)
    add_flt("Luminance To Alpha", 0.0, 0.0, 1.0)
    add_flt("Additive Blend", 0.0, 0.0, 1.0)
    add_flt("Ghost Rim Translucency", 0.0, 0.0, 1.0)
    s_nrm = iface.new_socket("Normal", in_out="INPUT", socket_type="NodeSocketVector")
    s_nrm.hide_value = True

    # Outputs (Dual Engine: EEVEE + Cycles)
    iface.new_socket("EEVEE Surface", in_out="OUTPUT", socket_type="NodeSocketShader")
    iface.new_socket("Cycles Surface", in_out="OUTPUT", socket_type="NodeSocketShader")

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    n_in = nodes.new("NodeGroupInput")
    n_in.location = (-1800, 0)
    n_out = nodes.new("NodeGroupOutput")
    n_out.location = (1650, 0)

    # 1. Grayscale Luminance & Colorization:
    # Standard multiply: Base Texture * Base Tint Color
    mul_tint = nodes.new("ShaderNodeMix")
    mul_tint.data_type = "RGBA"
    mul_tint.blend_type = "MULTIPLY"
    mul_tint.inputs[0].default_value = 1.0
    mul_tint.location = (-1450, 420)
    links.new(n_in.outputs["Base Texture"], mul_tint.inputs[6])
    links.new(n_in.outputs["Base Tint Color"], mul_tint.inputs[7])

    # Rich 2-tone colorization for grayscale textures:
    # Luminance L -> mix(Secondary Color * 0.35, Base Tint Color, L) * clamp(L * 1.35, 0, 1) + highlight boost on L^3
    bw_base = nodes.new("ShaderNodeRGBToBW")
    bw_base.location = (-1500, 160)
    links.new(n_in.outputs["Base Texture"], bw_base.inputs["Color"])

    sec_dark = nodes.new("ShaderNodeMix")
    sec_dark.data_type = "RGBA"
    sec_dark.blend_type = "MULTIPLY"
    sec_dark.inputs[0].default_value = 1.0
    sec_dark.inputs[7].default_value = (0.35, 0.35, 0.35, 1.0)
    sec_dark.location = (-1300, 220)
    links.new(n_in.outputs["Secondary Color"], sec_dark.inputs[6])

    grad_two_tone = nodes.new("ShaderNodeMix")
    grad_two_tone.data_type = "RGBA"
    grad_two_tone.blend_type = "MIX"
    grad_two_tone.location = (-1120, 240)
    links.new(bw_base.outputs["Val"], grad_two_tone.inputs[0])
    links.new(sec_dark.outputs[2], grad_two_tone.inputs[6])
    links.new(n_in.outputs["Base Tint Color"], grad_two_tone.inputs[7])

    # Scale by luminance so black (0) stays black (0) for dark-bg / mask textures!
    lum_boost = nodes.new("ShaderNodeMath")
    lum_boost.operation = "MULTIPLY"
    lum_boost.inputs[1].default_value = 1.25
    lum_boost.use_clamp = True
    lum_boost.location = (-1120, 40)
    links.new(bw_base.outputs["Val"], lum_boost.inputs[0])

    grad_masked = nodes.new("ShaderNodeMix")
    grad_masked.data_type = "RGBA"
    grad_masked.blend_type = "MULTIPLY"
    grad_masked.inputs[0].default_value = 1.0
    grad_masked.location = (-940, 220)
    links.new(grad_two_tone.outputs[2], grad_masked.inputs[6])
    links.new(lum_boost.outputs[0], grad_masked.inputs[7])

    # Choose between standard tint multiply and grayscale two-tone colorizer
    col_after_gs = nodes.new("ShaderNodeMix")
    col_after_gs.data_type = "RGBA"
    col_after_gs.blend_type = "MIX"
    col_after_gs.location = (-760, 360)
    links.new(n_in.outputs["Grayscale Colorize"], col_after_gs.inputs[0])
    links.new(mul_tint.outputs[2], col_after_gs.inputs[6])
    links.new(grad_masked.outputs[2], col_after_gs.inputs[7])

    # 2. Ramp / Sub Texture Mix (if Ramp LUT is active, Ramp / Sub Texture * Base Tint Color)
    ramp_tinted = nodes.new("ShaderNodeMix")
    ramp_tinted.data_type = "RGBA"
    ramp_tinted.blend_type = "MULTIPLY"
    ramp_tinted.inputs[0].default_value = 1.0
    ramp_tinted.location = (-760, 120)
    links.new(n_in.outputs["Ramp / Sub Texture"], ramp_tinted.inputs[6])
    links.new(n_in.outputs["Base Tint Color"], ramp_tinted.inputs[7])

    col_after_ramp = nodes.new("ShaderNodeMix")
    col_after_ramp.data_type = "RGBA"
    col_after_ramp.blend_type = "MIX"
    col_after_ramp.location = (-560, 320)
    links.new(n_in.outputs["Ramp Mix"], col_after_ramp.inputs[0])
    links.new(col_after_gs.outputs[2], col_after_ramp.inputs[6])
    links.new(ramp_tinted.outputs[2], col_after_ramp.inputs[7])

    # 3. Vertex Color Multiply (controlled by Vertex Color Mix)
    col_after_vc = nodes.new("ShaderNodeMix")
    col_after_vc.data_type = "RGBA"
    col_after_vc.blend_type = "MULTIPLY"
    col_after_vc.location = (-360, 320)
    links.new(n_in.outputs["Vertex Color Mix"], col_after_vc.inputs[0])
    links.new(col_after_ramp.outputs[2], col_after_vc.inputs[6])
    links.new(n_in.outputs["Vertex Color"], col_after_vc.inputs[7])

    # 4. Detail Mask (Texture 7 Red channel AO/scratches multiply)
    sep_det = nodes.new("ShaderNodeSeparateColor")
    sep_det.location = (-560, 40)
    links.new(n_in.outputs["Detail Mask Tex"], sep_det.inputs["Color"])

    col_after_det = nodes.new("ShaderNodeMix")
    col_after_det.data_type = "RGBA"
    col_after_det.blend_type = "MULTIPLY"
    col_after_det.location = (-160, 320)
    links.new(n_in.outputs["Use Detail Mask"], col_after_det.inputs[0])
    links.new(col_after_vc.outputs[2], col_after_det.inputs[6])
    links.new(sep_det.outputs["Red"], col_after_det.inputs[7])

    # 5. Fresnel / Rim Glow Calculation (View-dependent edge glow + Ghost translucency)
    layer_w = nodes.new("ShaderNodeLayerWeight")
    layer_w.inputs["Blend"].default_value = 0.5
    layer_w.location = (-760, -320)
    links.new(n_in.outputs["Normal"], layer_w.inputs["Normal"])

    rim_pow = nodes.new("ShaderNodeMath")
    rim_pow.operation = "POWER"
    rim_pow.use_clamp = True
    rim_pow.location = (-540, -320)
    links.new(layer_w.outputs["Facing"], rim_pow.inputs[0])
    links.new(n_in.outputs["Rim Power"], rim_pow.inputs[1])

    rim_fac = nodes.new("ShaderNodeMath")
    rim_fac.operation = "MULTIPLY"
    rim_fac.location = (-340, -320)
    links.new(rim_pow.outputs[0], rim_fac.inputs[0])
    links.new(n_in.outputs["Rim Strength"], rim_fac.inputs[1])

    rim_rgb = nodes.new("ShaderNodeMix")
    rim_rgb.data_type = "RGBA"
    rim_rgb.blend_type = "MULTIPLY"
    rim_rgb.inputs[0].default_value = 1.0
    rim_rgb.location = (-140, -320)
    links.new(n_in.outputs["Rim Color"], rim_rgb.inputs[6])
    links.new(rim_fac.outputs[0], rim_rgb.inputs[7])

    # 6. EEVEE Cel / Toon Shading Branch (Diffuse BSDF -> Shader to RGB + ILM Texture 3)
    diff_probe = nodes.new("ShaderNodeBsdfDiffuse")
    diff_probe.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    diff_probe.location = (-760, -60)
    links.new(n_in.outputs["Normal"], diff_probe.inputs["Normal"])

    s2rgb = nodes.new("ShaderNodeShaderToRGB")
    s2rgb.location = (-560, -60)
    links.new(diff_probe.outputs["BSDF"], s2rgb.inputs["Shader"])

    lit_bw = nodes.new("ShaderNodeRGBToBW")
    lit_bw.location = (-380, -60)
    links.new(s2rgb.outputs["Color"], lit_bw.inputs["Color"])

    sep_ilm = nodes.new("ShaderNodeSeparateColor")
    sep_ilm.location = (-560, -180)
    links.new(n_in.outputs["ILM / Shadow Mask"], sep_ilm.inputs["Color"])

    # ILM bias: (Red - 0.5) - Green * 0.5 weighted by Use ILM Mask
    ilm_bias_raw = nodes.new("ShaderNodeMath")
    ilm_bias_raw.operation = "SUBTRACT"
    ilm_bias_raw.location = (-380, -180)
    links.new(sep_ilm.outputs["Red"], ilm_bias_raw.inputs[0])
    links.new(sep_ilm.outputs["Green"], ilm_bias_raw.inputs[1])

    ilm_centered = nodes.new("ShaderNodeMath")
    ilm_centered.operation = "SUBTRACT"
    ilm_centered.inputs[1].default_value = 0.5
    ilm_centered.location = (-220, -180)
    links.new(ilm_bias_raw.outputs[0], ilm_centered.inputs[0])

    ilm_weighted = nodes.new("ShaderNodeMath")
    ilm_weighted.operation = "MULTIPLY"
    ilm_weighted.location = (-60, -180)
    links.new(ilm_centered.outputs[0], ilm_weighted.inputs[0])
    links.new(n_in.outputs["Use ILM Mask"], ilm_weighted.inputs[1])

    toon_val = nodes.new("ShaderNodeMath")
    toon_val.operation = "ADD"
    toon_val.location = (-60, -60)
    links.new(lit_bw.outputs["Val"], toon_val.inputs[0])
    links.new(ilm_weighted.outputs[0], toon_val.inputs[1])

    # Soft cel step around 0.42 so shadows look clean without jagged aliasing
    shadow_mask = nodes.new("ShaderNodeMapRange")
    shadow_mask.inputs["From Min"].default_value = 0.45
    shadow_mask.inputs["From Max"].default_value = 0.35
    shadow_mask.clamp = True
    shadow_mask.location = (110, -80)
    links.new(toon_val.outputs[0], shadow_mask.inputs["Value"])

    shadow_amt = nodes.new("ShaderNodeMath")
    shadow_amt.operation = "MULTIPLY"
    shadow_amt.use_clamp = True
    shadow_amt.location = (300, -80)
    links.new(shadow_mask.outputs["Result"], shadow_amt.inputs[0])
    links.new(n_in.outputs["Shadow Intensity"], shadow_amt.inputs[1])

    # Shaded Toon Color: multiply Base Color toward Shadow Color in shadowed regions
    shadow_tinted = nodes.new("ShaderNodeMix")
    shadow_tinted.data_type = "RGBA"
    shadow_tinted.blend_type = "MULTIPLY"
    shadow_tinted.location = (480, 220)
    links.new(shadow_amt.outputs[0], shadow_tinted.inputs[0])
    links.new(col_after_det.outputs[2], shadow_tinted.inputs[6])
    links.new(n_in.outputs["Shadow Color"], shadow_tinted.inputs[7])

    # Add Specular / Matcap Texture contribution if present
    spec_tex_contrib = nodes.new("ShaderNodeMix")
    spec_tex_contrib.data_type = "RGBA"
    spec_tex_contrib.blend_type = "ADD"
    spec_tex_contrib.location = (660, 220)
    links.new(n_in.outputs["Use Specular Tex"], spec_tex_contrib.inputs[0])
    links.new(shadow_tinted.outputs[2], spec_tex_contrib.inputs[6])
    links.new(n_in.outputs["Specular / Emissive Tex"], spec_tex_contrib.inputs[7])

    # 7. Emission Color Signal (preserves intricate stripe/energy texture detail!)
    albedo_emissive = nodes.new("ShaderNodeMix")
    albedo_emissive.data_type = "RGBA"
    albedo_emissive.blend_type = "MULTIPLY"
    albedo_emissive.inputs[0].default_value = 1.0
    albedo_emissive.location = (120, -320)
    links.new(col_after_det.outputs[2], albedo_emissive.inputs[6])
    links.new(n_in.outputs["Emission Color"], albedo_emissive.inputs[7])

    emi_mixed = nodes.new("ShaderNodeMix")
    emi_mixed.data_type = "RGBA"
    emi_mixed.blend_type = "MIX"
    emi_mixed.location = (300, -320)
    links.new(n_in.outputs["Emission From Albedo"], emi_mixed.inputs[0])
    links.new(n_in.outputs["Emission Color"], emi_mixed.inputs[6])
    links.new(albedo_emissive.outputs[2], emi_mixed.inputs[7])

    emi_scaled = nodes.new("ShaderNodeMix")
    emi_scaled.data_type = "RGBA"
    emi_scaled.blend_type = "MULTIPLY"
    emi_scaled.inputs[0].default_value = 1.0
    emi_scaled.location = (480, -320)
    links.new(emi_mixed.outputs[2], emi_scaled.inputs[6])
    links.new(n_in.outputs["Emission Strength"], emi_scaled.inputs[7])

    total_glow_rgb = nodes.new("ShaderNodeMix")
    total_glow_rgb.data_type = "RGBA"
    total_glow_rgb.blend_type = "ADD"
    total_glow_rgb.inputs[0].default_value = 1.0
    total_glow_rgb.location = (660, -320)
    links.new(emi_scaled.outputs[2], total_glow_rgb.inputs[6])
    links.new(rim_rgb.outputs[2], total_glow_rgb.inputs[7])

    # 8. Effective Alpha / Opacity Calculation:
    # Effective Vertex Alpha = mix(1.0, Vertex Alpha, Vertex Alpha Mix)
    eff_va = nodes.new("ShaderNodeMix")
    eff_va.data_type = "FLOAT"
    eff_va.inputs[2].default_value = 1.0
    eff_va.location = (-360, -560)
    links.new(n_in.outputs["Vertex Alpha Mix"], eff_va.inputs[0])
    links.new(n_in.outputs["Vertex Alpha"], eff_va.inputs[3])

    # Luminance alpha for dark-background energy/additive textures
    bw_final = nodes.new("ShaderNodeRGBToBW")
    bw_final.location = (-360, -700)
    links.new(col_after_det.outputs[2], bw_final.inputs["Color"])

    lum_alpha_boost = nodes.new("ShaderNodeMath")
    lum_alpha_boost.operation = "MULTIPLY"
    lum_alpha_boost.inputs[1].default_value = 1.6
    lum_alpha_boost.use_clamp = True
    lum_alpha_boost.location = (-180, -700)
    links.new(bw_final.outputs["Val"], lum_alpha_boost.inputs[0])

    eff_tex_alpha = nodes.new("ShaderNodeMix")
    eff_tex_alpha.data_type = "FLOAT"
    eff_tex_alpha.location = (0, -620)
    links.new(n_in.outputs["Luminance To Alpha"], eff_tex_alpha.inputs[0])
    links.new(n_in.outputs["Base Texture Alpha"], eff_tex_alpha.inputs[2])
    links.new(lum_alpha_boost.outputs[0], eff_tex_alpha.inputs[3])

    a_mul1 = nodes.new("ShaderNodeMath")
    a_mul1.operation = "MULTIPLY"
    a_mul1.location = (180, -600)
    links.new(eff_tex_alpha.outputs[0], a_mul1.inputs[0])
    links.new(eff_va.outputs[0], a_mul1.inputs[1])

    a_mul2 = nodes.new("ShaderNodeMath")
    a_mul2.operation = "MULTIPLY"
    a_mul2.use_clamp = True
    a_mul2.location = (350, -600)
    links.new(a_mul1.outputs[0], a_mul2.inputs[0])
    links.new(n_in.outputs["Material Opacity"], a_mul2.inputs[1])

    # Ghost Rim Translucency: interior more see-through (0.35 * alpha), silhouette edges brighter/opaque
    ghost_core = nodes.new("ShaderNodeMath")
    ghost_core.operation = "MULTIPLY"
    ghost_core.inputs[1].default_value = 0.38
    ghost_core.location = (350, -760)
    links.new(a_mul2.outputs[0], ghost_core.inputs[0])

    ghost_edge = nodes.new("ShaderNodeMath")
    ghost_edge.operation = "ADD"
    ghost_edge.use_clamp = True
    ghost_edge.location = (520, -760)
    links.new(ghost_core.outputs[0], ghost_edge.inputs[0])
    links.new(rim_pow.outputs[0], ghost_edge.inputs[1])

    ghost_modulated = nodes.new("ShaderNodeMath")
    ghost_modulated.operation = "MULTIPLY"
    ghost_modulated.use_clamp = True
    ghost_modulated.location = (690, -760)
    links.new(ghost_edge.outputs[0], ghost_modulated.inputs[0])
    links.new(eff_tex_alpha.outputs[0], ghost_modulated.inputs[1])

    a_after_ghost = nodes.new("ShaderNodeMix")
    a_after_ghost.data_type = "FLOAT"
    a_after_ghost.location = (860, -620)
    links.new(n_in.outputs["Ghost Rim Translucency"], a_after_ghost.inputs[0])
    links.new(a_mul2.outputs[0], a_after_ghost.inputs[2])
    links.new(ghost_modulated.outputs[0], a_after_ghost.inputs[3])

    # Alpha Clip Threshold (binary cutout when Use Alpha Clip = 1)
    clip_cmp = nodes.new("ShaderNodeMath")
    clip_cmp.operation = "GREATER_THAN"
    clip_cmp.location = (860, -460)
    links.new(a_after_ghost.outputs[0], clip_cmp.inputs[0])
    links.new(n_in.outputs["Alpha Clip Threshold"], clip_cmp.inputs[1])

    final_alpha = nodes.new("ShaderNodeMix")
    final_alpha.data_type = "FLOAT"
    final_alpha.location = (1040, -560)
    links.new(n_in.outputs["Use Alpha Clip"], final_alpha.inputs[0])
    links.new(a_after_ghost.outputs[0], final_alpha.inputs[2])
    links.new(clip_cmp.outputs[0], final_alpha.inputs[3])

    # 9. Principled BSDF (shared PBR/Metallic/Cycles foundation) & Toon Emission Shader (EEVEE)
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (880, 120)
    links.new(col_after_det.outputs[2], principled.inputs["Base Color"])
    links.new(n_in.outputs["Metallic"], principled.inputs["Metallic"])
    links.new(n_in.outputs["Roughness"], principled.inputs["Roughness"])
    links.new(n_in.outputs["Specular Color"], principled.inputs["Specular Tint"])
    links.new(n_in.outputs["Normal"], principled.inputs["Normal"])
    links.new(total_glow_rgb.outputs[2], principled.inputs["Emission Color"])
    principled.inputs["Emission Strength"].default_value = 1.0
    links.new(final_alpha.outputs[0], principled.inputs["Alpha"])

    # EEVEE Toon Surface = Emission(spec_tex_contrib + total_glow_rgb) mixed withTransparent BSDF by final_alpha
    toon_plus_glow = nodes.new("ShaderNodeMix")
    toon_plus_glow.data_type = "RGBA"
    toon_plus_glow.blend_type = "ADD"
    toon_plus_glow.inputs[0].default_value = 1.0
    toon_plus_glow.location = (880, 360)
    links.new(spec_tex_contrib.outputs[2], toon_plus_glow.inputs[6])
    links.new(total_glow_rgb.outputs[2], toon_plus_glow.inputs[7])

    toon_emi = nodes.new("ShaderNodeEmission")
    toon_emi.inputs["Strength"].default_value = 1.0
    toon_emi.location = (1060, 360)
    links.new(toon_plus_glow.outputs[2], toon_emi.inputs["Color"])

    transp = nodes.new("ShaderNodeBsdfTransparent")
    transp.location = (1060, -180)

    toon_alpha_shader = nodes.new("ShaderNodeMixShader")
    toon_alpha_shader.location = (1240, 280)
    links.new(final_alpha.outputs[0], toon_alpha_shader.inputs[0])
    links.new(transp.outputs["BSDF"], toon_alpha_shader.inputs[1])
    links.new(toon_emi.outputs["Emission"], toon_alpha_shader.inputs[2])

    eevee_lit_mix = nodes.new("ShaderNodeMixShader")
    eevee_lit_mix.location = (1380, 160)
    links.new(n_in.outputs["Toon Shading Mix"], eevee_lit_mix.inputs[0])
    links.new(principled.outputs["BSDF"], eevee_lit_mix.inputs[1])
    links.new(toon_alpha_shader.outputs["Shader"], eevee_lit_mix.inputs[2])

    # True Additive VFX Shader: Transparent BSDF + Emission((col_after_det + total_glow_rgb) * final_alpha)
    add_vfx_col = nodes.new("ShaderNodeMix")
    add_vfx_col.data_type = "RGBA"
    add_vfx_col.blend_type = "ADD"
    add_vfx_col.inputs[0].default_value = 1.0
    add_vfx_col.location = (880, -240)
    links.new(col_after_det.outputs[2], add_vfx_col.inputs[6])
    links.new(total_glow_rgb.outputs[2], add_vfx_col.inputs[7])

    add_vfx_weighted = nodes.new("ShaderNodeMix")
    add_vfx_weighted.data_type = "RGBA"
    add_vfx_weighted.blend_type = "MULTIPLY"
    add_vfx_weighted.inputs[0].default_value = 1.0
    add_vfx_weighted.location = (1060, -320)
    links.new(add_vfx_col.outputs[2], add_vfx_weighted.inputs[6])
    links.new(final_alpha.outputs[0], add_vfx_weighted.inputs[7])

    add_emi = nodes.new("ShaderNodeEmission")
    add_emi.inputs["Strength"].default_value = 1.0
    add_emi.location = (1240, -320)
    links.new(add_vfx_weighted.outputs[2], add_emi.inputs["Color"])

    add_shader = nodes.new("ShaderNodeAddShader")
    add_shader.location = (1380, -240)
    links.new(transp.outputs["BSDF"], add_shader.inputs[0])
    links.new(add_emi.outputs["Emission"], add_shader.inputs[1])

    # Final EEVEE & Cycles Mix with Additive Blend factor
    eevee_final = nodes.new("ShaderNodeMixShader")
    eevee_final.location = (1520, 120)
    links.new(n_in.outputs["Additive Blend"], eevee_final.inputs[0])
    links.new(eevee_lit_mix.outputs["Shader"], eevee_final.inputs[1])
    links.new(add_shader.outputs["Shader"], eevee_final.inputs[2])

    cycles_final = nodes.new("ShaderNodeMixShader")
    cycles_final.location = (1520, -80)
    links.new(n_in.outputs["Additive Blend"], cycles_final.inputs[0])
    links.new(principled.outputs["BSDF"], cycles_final.inputs[1])
    links.new(add_shader.outputs["Shader"], cycles_final.inputs[2])

    links.new(eevee_final.outputs["Shader"], n_out.inputs["EEVEE Surface"])
    links.new(cycles_final.outputs["Shader"], n_out.inputs["Cycles Surface"])

    return ng


# ---------------------------------------------------------------------------
# Material Render State & Node Tree Construction
# ---------------------------------------------------------------------------

def _apply_material_render_settings(material: bpy.types.Material, analysis: MaterialAnalysis) -> None:
    mode = analysis.blend_mode
    material.use_backface_culling = bool(analysis.backface_culling)

    if hasattr(material, "use_backface_culling_shadow"):
        material.use_backface_culling_shadow = False

    if mode in ("ALPHA_BLEND", "ADDITIVE", "GHOST"):
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "BLENDED"
        if hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = True
        if hasattr(material, "show_transparent_back"):
            material.show_transparent_back = True
        if hasattr(material, "use_transparent_shadow"):
            material.use_transparent_shadow = True
    elif mode == "ALPHA_CLIP":
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        if hasattr(material, "blend_method"):
            material.blend_method = "CLIP"
        if hasattr(material, "alpha_threshold"):
            material.alpha_threshold = float(analysis.alpha_clip_threshold)
    else:  # OPAQUE
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        if hasattr(material, "blend_method"):
            material.blend_method = "OPAQUE"

    # Set Solid-Viewport preview color/metallic/roughness/alpha so Solid View also reflects material color!
    r, g, b, _ = analysis.base_color
    vp_alpha = analysis.opacity if mode in ("ALPHA_BLEND", "GHOST") else (0.75 if mode == "ADDITIVE" else 1.0)
    material.diffuse_color = (r, g, b, max(0.15, min(1.0, vp_alpha)))
    material.metallic = float(analysis.metallic)
    material.roughness = float(analysis.roughness)


def _uv_layer_name(uv_index: int) -> str:
    if uv_index <= 0:
        return "UVMap"
    return f"UVMap{uv_index}"


def populate_material_nodes(material: bpy.types.Material, analysis: MaterialAnalysis) -> bpy.types.Material:
    material.use_nodes = True
    nt = material.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    master_ng = ensure_master_shader_node_group()
    nrm_fix_ng = ensure_normal_fix_node_group()

    grp = nodes.new("ShaderNodeGroup")
    grp.name = "CGTMD2 Shader"
    grp.label = f"CGTMD2 Shader [{analysis.shader_id or 'Default'}]"
    grp.node_tree = master_ng
    grp.location = (220, 120)
    grp.width = 290

    out_eevee = nodes.new("ShaderNodeOutputMaterial")
    out_eevee.name = "Material Output EEVEE"
    out_eevee.target = "EEVEE"
    out_eevee.location = (620, 180)
    links.new(grp.outputs["EEVEE Surface"], out_eevee.inputs["Surface"])

    out_cycles = nodes.new("ShaderNodeOutputMaterial")
    out_cycles.name = "Material Output Cycles"
    out_cycles.target = "CYCLES"
    out_cycles.location = (620, -20)
    links.new(grp.outputs["Cycles Surface"], out_cycles.inputs["Surface"])

    # Create UV Map (+ optional UV Mapping transform) nodes
    uv_nodes: Dict[int, bpy.types.NodeSocket] = {}
    has_uv_transform = (analysis.uv_scale != (1.0, 1.0)) or (analysis.uv_offset != (0.0, 0.0))
    max_uv = max([0] + [b.uv_index for b in analysis.bindings] + [analysis.submesh_usage.uv_count - 1])
    for uv_i in range(max(1, max_uv + 1)):
        u_node = nodes.new("ShaderNodeUVMap")
        u_node.name = f"UV_{uv_i}"
        u_node.label = _uv_layer_name(uv_i)
        u_node.uv_map = _uv_layer_name(uv_i)
        u_node.location = (-1050, 320 - uv_i * 180)
        if has_uv_transform and uv_i == 0:
            map_node = nodes.new("ShaderNodeMapping")
            map_node.location = (-860, 320)
            map_node.inputs["Scale"].default_value = (analysis.uv_scale[0], analysis.uv_scale[1], 1.0)
            map_node.inputs["Location"].default_value = (analysis.uv_offset[0], analysis.uv_offset[1], 0.0)
            links.new(u_node.outputs["UV"], map_node.inputs["Vector"])
            uv_nodes[uv_i] = map_node.outputs["Vector"]
        else:
            uv_nodes[uv_i] = u_node.outputs["UV"]

    # Configure Scalar & Color Inputs on CGTMD2 Shader Group
    inp = grp.inputs
    inp["Base Tint Color"].default_value = analysis.base_color
    inp["Secondary Color"].default_value = analysis.secondary_color
    inp["Grayscale Colorize"].default_value = 1.0 if (analysis.colorize_grayscale and not analysis.use_ramp_lut and analysis.secondary_color_param_idx >= 0) else 0.0
    inp["Ramp Mix"].default_value = 1.0 if analysis.use_ramp_lut else 0.0
    inp["Shadow Intensity"].default_value = analysis.shadow_intensity
    inp["Shadow Color"].default_value = analysis.shadow_color
    inp["Specular Color"].default_value = analysis.specular_color
    inp["Metallic"].default_value = analysis.metallic
    inp["Roughness"].default_value = analysis.roughness
    inp["Rim Color"].default_value = analysis.rim_color
    inp["Rim Strength"].default_value = analysis.rim_strength
    inp["Rim Power"].default_value = analysis.rim_power
    inp["Emission Color"].default_value = analysis.emission_color
    inp["Emission Strength"].default_value = analysis.emission_strength
    inp["Emission From Albedo"].default_value = 0.85 if any(b.role == "albedo" and b.image for b in analysis.bindings) else 0.0
    inp["Material Opacity"].default_value = analysis.opacity
    inp["Alpha Clip Threshold"].default_value = analysis.alpha_clip_threshold
    inp["Use Alpha Clip"].default_value = 1.0 if analysis.blend_mode == "ALPHA_CLIP" else 0.0
    inp["Luminance To Alpha"].default_value = 1.0 if analysis.luminance_to_alpha else 0.0
    inp["Additive Blend"].default_value = 1.0 if analysis.blend_mode == "ADDITIVE" else 0.0
    inp["Ghost Rim Translucency"].default_value = 1.0 if (analysis.blend_mode == "GHOST" or analysis.fresnel_translucency) else 0.0

    # Toon vs PBR/VFX shading weight
    if analysis.shader_family in ("VFX_ADDITIVE", "GHOST_SPIRIT"):
        inp["Toon Shading Mix"].default_value = 0.0
    elif analysis.shader_family == "METALLIC_GLOW":
        inp["Toon Shading Mix"].default_value = 0.25
    elif analysis.shader_family in ("TOON_CHAR", "HAIR", "EYE", "SKIN"):
        inp["Toon Shading Mix"].default_value = 0.85
    else:
        inp["Toon Shading Mix"].default_value = 0.35

    # Vertex Colors (Color & Color2)
    if analysis.submesh_usage.has_color0 or analysis.use_vertex_color_rgb or analysis.use_vertex_color_alpha:
        vc0 = nodes.new("ShaderNodeVertexColor")
        vc0.name = "Vertex Color"
        vc0.label = "Vertex Color (Color)"
        vc0.layer_name = analysis.vertex_color_layer or "Color"
        vc0.location = (-420, -340)
        links.new(vc0.outputs["Color"], inp["Vertex Color"])
        links.new(vc0.outputs["Alpha"], inp["Vertex Alpha"])
        inp["Vertex Color Mix"].default_value = 1.0 if analysis.use_vertex_color_rgb else 0.0
        inp["Vertex Alpha Mix"].default_value = 1.0 if analysis.use_vertex_color_alpha else 0.0

    if analysis.submesh_usage.has_color1:
        vc1 = nodes.new("ShaderNodeVertexColor")
        vc1.name = "Vertex Color 2"
        vc1.label = "Vertex Color 2 (Color2)"
        vc1.layer_name = "Color2"
        vc1.location = (-420, -500)

    # Create & Connect All Texture Nodes
    albedo_tex_node = None
    ramp_tex_node = None
    y_cursor = 520
    for bind in analysis.bindings:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.name = f"Texture_{bind.list_index}_{bind.role}"
        tex_node.label = f"Slot {bind.sampler_slot} ({bind.role}): {bind.resolved_name or bind.tex_hash}"
        tex_node.location = (-620, y_cursor)
        y_cursor -= 260

        if bind.image is not None:
            tex_node.image = bind.image
            if bind.non_color:
                try:
                    tex_node.image.colorspace_settings.name = "Non-Color"
                except Exception:
                    pass
        if bind.clamp_uv:
            tex_node.extension = "EXTEND"

        # Default UV connection
        uv_sock = uv_nodes.get(bind.uv_index, uv_nodes[0])
        links.new(uv_sock, tex_node.inputs["Vector"])

        if bind.role == "albedo" and bind.image is not None:
            if albedo_tex_node is None:
                albedo_tex_node = tex_node
                links.new(tex_node.outputs["Color"], inp["Base Texture"])
                if not bind.inspection.alpha_zeroed:
                    links.new(tex_node.outputs["Alpha"], inp["Base Texture Alpha"])
            elif ramp_tex_node is None:
                # Secondary multi-texture layer on another UV or sub-slot
                ramp_tex_node = tex_node
                links.new(tex_node.outputs["Color"], inp["Ramp / Sub Texture"])
                inp["Ramp Mix"].default_value = 0.45

        elif bind.role == "ramp" and bind.image is not None:
            ramp_tex_node = tex_node
            links.new(tex_node.outputs["Color"], inp["Ramp / Sub Texture"])
            if not analysis.use_ramp_lut:
                inp["Ramp Mix"].default_value = 0.35

        elif bind.role == "ilm_shadow" and bind.image is not None:
            links.new(tex_node.outputs["Color"], inp["ILM / Shadow Mask"])
            inp["Use ILM Mask"].default_value = 1.0

        elif bind.role == "specular_emissive" and bind.image is not None:
            if bind.inspection.is_matcap_sphere:
                # View-space spherical matcap UV coordinates
                geom_mc = nodes.new("ShaderNodeNewGeometry")
                geom_mc.location = (-1220, tex_node.location.y)
                vtrans = nodes.new("ShaderNodeVectorTransform")
                vtrans.vector_type = "NORMAL"
                vtrans.convert_from = "WORLD"
                vtrans.convert_to = "CAMERA"
                vtrans.location = (-1040, tex_node.location.y)
                mc_map = nodes.new("ShaderNodeMapping")
                mc_map.inputs["Location"].default_value = (0.5, 0.5, 0.0)
                mc_map.inputs["Scale"].default_value = (0.5, 0.5, 1.0)
                mc_map.location = (-860, tex_node.location.y)
                links.new(geom_mc.outputs["Normal"], vtrans.inputs["Vector"])
                links.new(vtrans.outputs["Vector"], mc_map.inputs["Vector"])
                links.new(mc_map.outputs["Vector"], tex_node.inputs["Vector"])
            links.new(tex_node.outputs["Color"], inp["Specular / Emissive Tex"])
            inp["Use Specular Tex"].default_value = 0.65

        elif bind.role == "normal" and bind.image is not None:
            fix_node = nodes.new("ShaderNodeGroup")
            fix_node.name = "DX Normal Fix"
            fix_node.node_tree = nrm_fix_ng
            fix_node.location = (-300, tex_node.location.y)
            fix_node.inputs["Flip Green (DX)"].default_value = 1.0 if analysis.flip_normal_green else 0.0
            fix_node.inputs["Reconstruct Z (BC5)"].default_value = 1.0 if bind.inspection.needs_normal_z_reconstruct else 0.0
            links.new(tex_node.outputs["Color"], fix_node.inputs["Color"])

            nmap = nodes.new("ShaderNodeNormalMap")
            nmap.uv_map = _uv_layer_name(bind.uv_index)
            nmap.location = (-80, tex_node.location.y)
            links.new(fix_node.outputs["Color"], nmap.inputs["Color"])
            links.new(nmap.outputs["Normal"], inp["Normal"])

        elif bind.role == "detail_mask" and bind.image is not None:
            links.new(tex_node.outputs["Color"], inp["Detail Mask Tex"])
            inp["Use Detail Mask"].default_value = 0.5

    # Wire 1D/2D Ramp LUT lookup after loop (order-independent of texture slots!)
    if analysis.use_ramp_lut and albedo_tex_node is not None and ramp_tex_node is not None:
        bw_lut = nodes.new("ShaderNodeRGBToBW")
        bw_lut.location = (-440, ramp_tex_node.location.y + 90)
        links.new(albedo_tex_node.outputs["Color"], bw_lut.inputs["Color"])

        comb_lut = nodes.new("ShaderNodeCombineXYZ")
        comb_lut.inputs["Y"].default_value = 0.5
        comb_lut.location = (-820, ramp_tex_node.location.y)
        links.new(bw_lut.outputs["Val"], comb_lut.inputs["X"])
        links.new(comb_lut.outputs["Vector"], ramp_tex_node.inputs["Vector"])
        inp["Ramp Mix"].default_value = 1.0

    # Special Hair Shader handling (BHG1/BHA1 have no Texture 1 albedo; Base Tint Color IS the hair color!)
    if analysis.shader_id in ("BHG1", "BHA1") and albedo_tex_node is None:
        inp["Base Texture"].default_value = (1.0, 1.0, 1.0, 1.0)

    _apply_material_render_settings(material, analysis)
    return material


def build_material_from_analysis(mat_name: str, analysis: MaterialAnalysis) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=mat_name)
    populate_material_nodes(mat, analysis)
    return mat


# ---------------------------------------------------------------------------
# Live Material Rebuilder (from stored ChatGPT TMD2 Material Properties!)
# ---------------------------------------------------------------------------

def rebuild_blender_material(material: bpy.types.Material, preset_override: Optional[str] = None) -> MaterialAnalysis:
    """Reconstructs an existing Blender material's shader graph in-place using its stored
    `chatgpt_tmd2_material` properties (shader_id, param_values, textures, and user overrides)."""
    props = getattr(material, "chatgpt_tmd2_material", None)
    if props is None:
        return MaterialAnalysis(material_name=material.name)

    # Build lightweight proxy objects matching TMD2Material / TMD2MatTexture for the analyzer
    class _TexProxy:
        def __init__(self, h, idx):
            self.hash = h
            self.index = idx
            self.name = ""

    class _MatTexProxy:
        def __init__(self, t_item, i):
            h = int(t_item.texture_hash) if str(t_item.texture_hash).lstrip("-").isdigit() else 0
            self.textureHash = h
            self.textureIndex = -1 if t_item.is_external else i
            self.isExternal = bool(t_item.is_external)
            self.texture = _TexProxy(h, -1 if t_item.is_external else i)
            self.unk1 = t_item.value1
            self.unk2 = t_item.value2
            self.unk3 = t_item.value3
            self.slot = t_item.slot if t_item.slot >= 0 else i

    class _MatProxy:
        def __init__(self):
            self.hash = int(props.material_hash) if str(props.material_hash).lstrip("-").isdigit() else 0
            self.shaderID = props.shader_id
            self.shaderParams = [p.value for p in props.param_values]
            self.textures = [_MatTexProxy(t, i) for i, t in enumerate(props.textures)]
            self.unk = props.unk

    class _ModelDummy:
        modelFlags = 0x10 | 0x20 | 0x80 | 0x200 | 0x4
        name = material.name
        models = []
        vertices = None

    lds_imgs = [t.image for t in props.textures]
    tm_proxy = _MatProxy()

    profile = preset_override or props.reconstruction_profile
    fb_opts = {
        "fallback_profile": profile if profile != "AUTO" else "AUTO",
        "transparency_mode": props.transparency_mode,
        "use_vertex_colors": props.vertex_color_mode,
        "grayscale_mode": props.grayscale_mode,
        "emission_multiplier": props.emission_multiplier,
        "backface_culling": props.backface_culling_mode,
    }

    analysis = analyze_tmd_material(
        tm_proxy,
        _ModelDummy(),
        lds_imgs,
        {},
        {},
        mat_name=material.name,
        fallback_options=fb_opts,
    )
    if profile and profile != "AUTO":
        analysis.shader_family = profile
        if profile == "GHOST_SPIRIT":
            analysis.blend_mode = "GHOST" if props.transparency_mode == "AUTO" else props.transparency_mode
            analysis.fresnel_translucency = True
            analysis.rim_strength = max(analysis.rim_strength, 1.2)
            analysis.opacity = min(analysis.opacity, 0.72)
        elif profile == "VFX_ADDITIVE":
            analysis.blend_mode = "ADDITIVE" if props.transparency_mode == "AUTO" else props.transparency_mode
            analysis.luminance_to_alpha = (analysis.blend_mode == "ADDITIVE")
            if analysis.emission_strength <= 0.0:
                analysis.emission_color = analysis.base_color
                analysis.emission_strength = max(1.5, props.emission_multiplier)
        elif profile == "METALLIC_GLOW":
            analysis.metallic = 0.85
            analysis.roughness = 0.20
            analysis.rim_strength = max(analysis.rim_strength, 0.7)
            if analysis.emission_strength <= 0.0:
                analysis.emission_color = analysis.base_color
                analysis.emission_strength = max(0.8, props.emission_multiplier * 0.6)

    # Apply per-material parameter index or manual color overrides from UI
    params = tm_proxy.shaderParams
    if props.base_color_param_idx >= 0 and props.base_color_param_idx + 3 <= len(params):
        i = props.base_color_param_idx
        r, g, b = max(0.0, params[i]), max(0.0, params[i + 1]), max(0.0, params[i + 2])
        pk = max(1.0, r, g, b)
        analysis.base_color = (r / pk, g / pk, b / pk, 1.0)
        analysis.base_color_param_idx = i
    if props.use_custom_tint:
        analysis.base_color = tuple(props.custom_tint_color)
    if props.use_custom_emission:
        analysis.emission_color = (*props.custom_emission_color[:3], 1.0)
        analysis.emission_strength = props.custom_emission_strength
    if props.opacity_override >= 0.0:
        analysis.opacity = props.opacity_override
        if analysis.opacity < 0.99 and analysis.blend_mode == "OPAQUE":
            analysis.blend_mode = "ALPHA_BLEND"

    analysis.flip_normal_green = bool(props.flip_normal_green)
    populate_material_nodes(material, analysis)

    props.detected_family = analysis.shader_family
    props.detected_blend = analysis.blend_mode
    return analysis


# Backwards-compatible shaders_dict helper
def _legacy_entry(shader_name: str):
    def _creator(mat_name=f"{shader_name}_Material", tm_material=None, textures=None):
        if textures is None:
            textures = []

        class _DummyTMD:
            name = mat_name
            modelFlags = 0x10 | 0x4
            models = []
            vertices = None

        analysis = analyze_tmd_material(tm_material, _DummyTMD(), textures, {}, {}, mat_name=mat_name)
        return build_material_from_analysis(mat_name, analysis)

    return _creator


shaders_dict = {
    name: _legacy_entry(name)
    for name in ("BSS1", "BHG1", "BHA1", "BEI0", "BED0", "BES0", "BEH0", "BGH0", "BGH1", "HSK1", "Default")
}
