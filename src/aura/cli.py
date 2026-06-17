from __future__ import annotations

import argparse
import json
from pathlib import Path

from aura.baselines import package_3dgs_export
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.package import load_package, package_scene
from aura.ray import Ray
from aura.render import compare_images, read_ppm, render_orthographic
from aura.scene import AuraScene
from aura.splats import load_3dgs_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aura")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("write-demo-package", help="Write a tiny GPU-ready .aura package scaffold")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/demo.aura"))

    splat_demo = sub.add_parser(
        "write-splat-demo-package",
        help="Convert a tiny JSON or ASCII/binary little-endian PLY 3DGS export to .aura",
    )
    splat_demo.add_argument("--input", type=Path, default=Path("tests/fixtures/tiny_3dgs_export.json"))
    splat_demo.add_argument("--output-dir", type=Path, default=Path("outputs/splat-demo.aura"))
    splat_demo.add_argument("--radius-sigma", type=float, default=2.0)

    import_3dgs = sub.add_parser("import-3dgs", help="Import a 3DGS PLY/JSON export or common 3DGS output directory")
    import_3dgs.add_argument("input", type=Path)
    import_3dgs.add_argument("--output-dir", type=Path, default=Path("outputs/imported-3dgs.aura"))
    import_3dgs.add_argument("--name", default=None)
    import_3dgs.add_argument("--radius-sigma", type=float, default=2.0)

    query = sub.add_parser("query-demo", help="Run a reference ray query against the fixture scene")
    query.add_argument("--x", type=float, default=0.0)
    query.add_argument("--y", type=float, default=0.0)

    validate = sub.add_parser("validate-package", help="Load and validate a .aura package directory")
    validate.add_argument("package_dir", type=Path)

    inspect = sub.add_parser("inspect-package", help="Load, validate, and print a .aura package summary as JSON")
    inspect.add_argument("package_dir", type=Path)

    render = sub.add_parser("render-package", help="Render a deterministic orthographic PPM preview from a .aura package")
    render.add_argument("package_dir", type=Path)
    render.add_argument("--output", type=Path, default=Path("outputs/preview.ppm"))
    render.add_argument("--width", type=int, default=64)
    render.add_argument("--height", type=int, default=64)

    compare = sub.add_parser("compare-renders", help="Compare two PPM previews and print JSON MSE/PSNR metrics")
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--min-psnr", type=float, default=None)

    args = parser.parse_args(argv)
    scene = demo_scene()
    if args.command == "write-demo-package":
        print(package_scene(scene, fallbacks={"mesh": "fallback/preview.glb", "splat": "fallback/preview.splat"}).write(args.output_dir))
        return 0
    if args.command == "write-splat-demo-package":
        splat_scene = load_3dgs_scene(args.input, radius_sigma=args.radius_sigma)
        print(package_scene(splat_scene, fallbacks={"splat": str(args.input)}).write(args.output_dir))
        return 0
    if args.command == "import-3dgs":
        package = package_3dgs_export(args.input, name=args.name, radius_sigma=args.radius_sigma)
        print(package.write(args.output_dir))
        return 0
    if args.command == "query-demo":
        result = scene.ray_query(Ray(origin=(args.x, args.y, -2.0), direction=(0.0, 0.0, 1.0)))
        print(result)
        return 0
    if args.command == "validate-package":
        package = load_package(args.package_dir)
        summary = package.summary()
        print(
            "valid AURA package: "
            f"{summary['name']} "
            f"(version {summary['version']}, "
            f"{summary['elementCount']} elements, "
            f"{summary['chunkCount']} chunks)"
        )
        return 0
    if args.command == "inspect-package":
        package = load_package(args.package_dir)
        print(json.dumps(package.summary(), indent=2, sort_keys=True))
        return 0
    if args.command == "render-package":
        package = load_package(args.package_dir)
        image = render_orthographic(package.scene, width=args.width, height=args.height)
        print(image.write_ppm(args.output))
        return 0
    if args.command == "compare-renders":
        metrics = compare_images(read_ppm(args.expected), read_ppm(args.actual), min_psnr=args.min_psnr)
        print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
        return 0 if metrics["passed"] else 1
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
