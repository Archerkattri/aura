from __future__ import annotations

import argparse
from pathlib import Path

from aura.elements import AuraChunk, AuraElement, Bounds
from aura.package import package_scene
from aura.ray import Ray
from aura.scene import AuraScene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aura")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("write-demo-package", help="Write a tiny CPU-only .aura package scaffold")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/demo.aura"))

    query = sub.add_parser("query-demo", help="Run a CPU reference ray query against the fixture scene")
    query.add_argument("--x", type=float, default=0.0)
    query.add_argument("--y", type=float, default=0.0)

    args = parser.parse_args(argv)
    scene = demo_scene()
    if args.command == "write-demo-package":
        print(package_scene(scene, fallbacks={"mesh": "fallback/preview.glb", "splat": "fallback/preview.splat"}).write(args.output_dir))
        return 0
    if args.command == "query-demo":
        result = scene.ray_query(Ray(origin=(args.x, args.y, -2.0), direction=(0.0, 0.0, 1.0)))
        print(result)
        return 0
    raise ValueError(args.command)


def demo_scene() -> AuraScene:
    bounds = Bounds(min_corner=(-0.5, -0.5, 0.0), max_corner=(0.5, 0.5, 0.1))
    element = AuraElement(
        id="wall_patch",
        carrier_id="surface",
        bounds=bounds,
        color=(0.8, 0.7, 0.6),
        opacity=0.9,
        confidence=0.85,
        normal=(0.0, 0.0, -1.0),
        material_id="mat_wall",
        semantic_id="wall",
    )
    chunk = AuraChunk(id="root", bounds=bounds, element_ids=("wall_patch",), lod=0)
    return AuraScene(name="demo", elements=(element,), chunks=(chunk,))


if __name__ == "__main__":
    raise SystemExit(main())

