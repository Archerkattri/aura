"""USD ASCII (.usda) writer for AURA gaussian carrier scenes.

Exports gaussian splat positions and properties as USD PointInstancer or
Points primitives in a valid .usda file. No pxr library required —
USD ASCII is a text format parseable by any text editor and loadable
by Blender, Houdini, Omniverse, and USD-capable DCC tools.

The exported file documents carrier metadata as USD custom attributes
(arbitrary:aura:*) that native USD tools preserve but do not interpret.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aura.scene import AuraScene


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
