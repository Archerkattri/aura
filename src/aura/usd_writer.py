"""USD writer(s) for AURA gaussian carrier scenes.

Two export paths:

1. ``write_usda`` — a dependency-free USD ASCII (.usda) *preview* writer. It emits
   gaussian carriers as a ``Points`` prim with ``displayColor`` so any text editor
   or USD-capable DCC (Blender, Houdini, Omniverse) can open it. Carrier metadata
   rides along as ``custom:aura:*`` attributes. This path requires no pxr library
   and is unchanged.

2. ``write_usd_gaussian_splat`` — a **schema-conformant** writer targeting the
   official OpenUSD 26.03 ``UsdVolParticleField3DGaussianSplat`` schema (USD type
   name ``ParticleField3DGaussianSplat``). It writes the applied-schema attributes
   (``positions``, ``orientations``, ``scales``, ``opacities``,
   ``radiance:sphericalHarmonicsCoefficients`` / ``…Degree``, ``primvars:displayColor``,
   ``extent``) so Omniverse and other 26.03+ tools consume the splats natively
   instead of seeing an opaque point cloud. It requires ``usd-core`` (pxr); the
   attribute names and prim type are validated against the installed schema
   registry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.scene import AuraScene

#: SH band-0 basis constant Y_0^0; rendered rgb = 0.5 + _C0 * f_dc, so the
#: degree-0 radiance coefficient is f_dc = (rgb - 0.5) / _C0 (matches
#: gltf_splat / gsplat_renderer).
_C0 = 0.28209479177387814

#: The USD type name of the concrete OpenUSD 26.03 gaussian-splat schema.
PARTICLEFIELD_SPLAT_TYPE = "ParticleField3DGaussianSplat"


def write_usda(scene: "AuraScene", output_path: "str | Path") -> Path:
    """Write scene's gaussian carriers to USD ASCII (.usda) format.

    Parameters
    ----------
    scene : AuraScene
        The loaded scene to export.
    output_path : str or Path
        Output .usda file path.

    Returns
    -------
    Path to the written .usda file.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".usda", ".usd"):
        output_path = output_path.with_suffix(".usda")

    # Collect gaussian elements
    positions = []
    colors = []
    scales = []
    carrier_ids = []

    for element in scene.elements:
        if element.carrier_id != "gaussian":
            continue
        pos = _get_position(element)
        if pos is None:
            continue
        positions.append(pos)
        colors.append(_get_color(element))
        scales.append(_get_scale(element))
        carrier_ids.append(element.id)

    lines = _build_usda(positions, colors, scales, carrier_ids, scene)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _get_position(element) -> tuple[float, float, float] | None:
    for attr in ("mean", "position"):
        val = getattr(element, attr, None)
        if val is not None:
            try:
                return (float(val[0]), float(val[1]), float(val[2]))
            except (TypeError, IndexError):
                pass
    payload = getattr(element, "payload", None)
    if isinstance(payload, dict):
        for key in ("mean", "position", "pos"):
            val = payload.get(key)
            if val is not None:
                try:
                    return (float(val[0]), float(val[1]), float(val[2]))
                except (TypeError, IndexError):
                    pass
    return None


def _get_color(element) -> tuple[float, float, float]:
    col = getattr(element, "color", None)
    if col is not None:
        try:
            return (
                min(1.0, max(0.0, float(col[0]))),
                min(1.0, max(0.0, float(col[1]))),
                min(1.0, max(0.0, float(col[2]))),
            )
        except (TypeError, IndexError):
            pass
    return (0.8, 0.8, 0.8)


def _get_scale(element) -> float:
    for attr in ("scale", "radius", "opacity"):
        val = getattr(element, attr, None)
        if val is not None:
            try:
                s = float(val)
                if s > 0:
                    return s
            except (TypeError, ValueError):
                pass
    return 0.01


def _fmt_vec3f(v: tuple) -> str:
    return f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})"


def _build_usda(
    positions: list,
    colors: list,
    scales: list,
    carrier_ids: list,
    scene,
) -> list[str]:
    n = len(positions)
    lines = [
        '#usda 1.0',
        '(',
        '    defaultPrim = "AURAScene"',
        '    doc = """AURA gaussian carrier scene exported by aura.usd_writer"""',
        '    metersPerUnit = 1.0',
        '    upAxis = "Y"',
        ')',
        '',
        'def Xform "AURAScene"',
        '{',
        '    def Points "GaussianCarriers"',
        '    {',
        f'        int[] primvars:displayOpacity = []',
        f'        point3f[] points = [',
    ]

    # positions
    for i, pos in enumerate(positions):
        comma = "," if i < n - 1 else ""
        lines.append(f'            {_fmt_vec3f(pos)}{comma}')
    lines.append('        ]')
    lines.append('')

    # widths (scales)
    lines.append('        float[] widths = [')
    for i, s in enumerate(scales):
        comma = "," if i < n - 1 else ""
        lines.append(f'            {s:.6f}{comma}')
    lines.append('        ]')
    lines.append('        uniform token widths:interpolation = "vertex"')
    lines.append('')

    # colors as displayColor
    lines.append('        color3f[] primvars:displayColor = [')
    for i, col in enumerate(colors):
        comma = "," if i < n - 1 else ""
        lines.append(f'            {_fmt_vec3f(col)}{comma}')
    lines.append('        ]')
    lines.append('        uniform token primvars:displayColor:interpolation = "vertex"')
    lines.append('')

    # Custom AURA metadata
    lines.append(f'        int custom:aura:carrierCount = {n}')
    lines.append(f'        string custom:aura:carrierType = "gaussian"')
    scene_name = getattr(getattr(scene, "asset", None), "name", "unknown")
    lines.append(f'        string custom:aura:sceneName = "{scene_name}"')
    lines.append('    }')
    lines.append('}')

    return lines


# ---------------------------------------------------------------------------
# Schema-conformant writer (OpenUSD 26.03 UsdVolParticleField3DGaussianSplat)
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert a torch tensor / list / numpy array to a contiguous numpy array."""
    import numpy as np

    if x is None:
        return None
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(x))


def write_usd_gaussian_splat(
    carriers: "dict",
    output_path: "str | Path",
    *,
    scene_name: str | None = None,
    include_confidence: bool = True,
    prim_path: str = "/AURAScene/GaussianSplat",
) -> Path:
    """Write gaussian carriers under the official OpenUSD 26.03 splat schema.

    Emits a ``ParticleField3DGaussianSplat`` (``UsdVolParticleField3DGaussianSplat``)
    prim with the applied-schema attributes so a 26.03+ path-tracer (Omniverse)
    renders the splats natively rather than as an opaque point cloud.

    Parameters
    ----------
    carriers : dict
        Carrier tensors as produced by ``aura.carrier_io`` (numpy or torch):
        ``means`` [N,3], ``scales`` [N,3] (linear), ``quats`` [N,4] wxyz,
        ``opacity`` [N] in [0,1]. Optional ``colors`` [N,3] flat linear rgb,
        ``sh`` [N,K,3] SH coefficients, ``sh_degree`` int, ``confidence`` [N].
    output_path : str or Path
        Output ``.usd``/``.usda``/``.usdc`` path. Suffix picks the layer format;
        default (no known splat suffix) → ``.usda`` (ASCII).
    scene_name : str, optional
        Recorded as ``custom:aura:sceneName``.
    include_confidence : bool
        If a ``confidence`` array is present, also write it as the AURA vendor
        channel ``custom:aura:confidence`` (schema-preserved, not interpreted).
    prim_path : str
        Prim path for the splat field.

    Returns
    -------
    Path to the written USD file.

    Raises
    ------
    ImportError
        If ``usd-core`` (pxr) is not importable.
    """
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdVol, Vt  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "write_usd_gaussian_splat requires the 'usd-core' package (pxr) for the "
            "OpenUSD 26.03 UsdVolParticleField3DGaussianSplat schema. Install it "
            "(pip install usd-core) or use write_usda for the dependency-free "
            "preview writer."
        ) from exc

    import numpy as np

    # Validate the schema is actually registered in the installed USD build so we
    # bind to the real applied-schema attribute names, not documented guesses.
    if Tf.Type.FindByName("UsdVolParticleField3DGaussianSplat") == Tf.Type.Unknown:
        raise ImportError(
            "The installed usd-core does not register "
            "UsdVolParticleField3DGaussianSplat (needs OpenUSD >= 26.03)."
        )

    means = _to_numpy(carriers["means"]).astype("float32")
    n = int(means.shape[0])
    scales = _to_numpy(carriers["scales"]).astype("float32")
    quats = _to_numpy(carriers["quats"]).astype("float32")
    opacity = np.clip(_to_numpy(carriers["opacity"]).astype("float32"), 0.0, 1.0)
    sh_degree = int(carriers.get("sh_degree", 0)) if hasattr(carriers, "get") else 0

    colors = carriers.get("colors") if hasattr(carriers, "get") else None
    sh = carriers.get("sh") if hasattr(carriers, "get") else None
    colors = _to_numpy(colors)
    sh = _to_numpy(sh)

    # Display colour (evaluated band-0 rgb) for previews and DC-from-colour fallback.
    if colors is not None:
        display_rgb = np.clip(colors.astype("float32"), 0.0, 1.0)
    elif sh is not None:
        display_rgb = np.clip(0.5 + _C0 * sh[:, 0, :].astype("float32"), 0.0, 1.0)
    else:
        display_rgb = np.full((n, 3), 0.8, dtype="float32")

    # Radiance SH coefficients in the schema's flat float3[] layout (particle-major:
    # all coefficients of particle 0, then particle 1, ...). Degree-0 = 1 coeff each.
    if sh is not None:
        sh_coeffs = sh.astype("float32").reshape(-1, 3)
    else:
        sh_coeffs = ((display_rgb - 0.5) / _C0).astype("float32")
        sh_degree = 0

    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".usda", ".usd", ".usdc"):
        output_path = output_path.with_suffix(".usda")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()  # CreateNew errors on an existing layer
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("metersPerUnit", 1.0)

    # Parent Xform (default prim) + typed splat prim beneath it.
    parent_path = Sdf.Path(prim_path).GetParentPath()
    if parent_path.pathString != "/":
        root_prim = stage.DefinePrim(parent_path, "Xform")
        stage.SetDefaultPrim(stage.GetPrimAtPath(Sdf.Path("/" + prim_path.strip("/").split("/")[0])))
    field = UsdVol.ParticleField3DGaussianSplat.Define(stage, Sdf.Path(prim_path))
    prim = field.GetPrim()

    field.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(means))
    field.CreateOrientationsAttr(
        Vt.QuatfArray([Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                       for q in quats])
    )
    field.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales))
    field.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(opacity))
    field.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(display_rgb))
    field.CreateRadianceSphericalHarmonicsDegreeAttr(int(sh_degree))
    field.CreateRadianceSphericalHarmonicsCoefficientsAttr(
        Vt.Vec3fArray.FromNumpy(sh_coeffs))
    if n:
        lo = means.min(axis=0).astype("float32")
        hi = means.max(axis=0).astype("float32")
        field.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.stack([lo, hi])))

    # AURA vendor channel: the calibrated per-carrier confidence, preserved (not
    # interpreted) by conformant tools — this is the property a bare splat lacks.
    conf = carriers.get("confidence") if hasattr(carriers, "get") else None
    conf = _to_numpy(conf)
    if include_confidence and conf is not None:
        a = prim.CreateAttribute("custom:aura:confidence", Sdf.ValueTypeNames.FloatArray)
        a.Set(Vt.FloatArray.FromNumpy(conf.astype("float32")))

    prim.CreateAttribute("custom:aura:carrierCount", Sdf.ValueTypeNames.Int).Set(n)
    prim.CreateAttribute("custom:aura:schema", Sdf.ValueTypeNames.String).Set(
        "UsdVolParticleField3DGaussianSplat (OpenUSD 26.03)")
    if scene_name:
        prim.CreateAttribute("custom:aura:sceneName", Sdf.ValueTypeNames.String).Set(
            str(scene_name))

    stage.GetRootLayer().Save()
    return output_path
