from __future__ import annotations

import argparse
import json
from pathlib import Path

from aura.assignment import RegionEvidence
from aura.benchmark import default_benchmark_suite, run_reference_benchmark
from aura.decomposition import EvidenceSample, decompose_evidence
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.ingest import load_3dgs_scene, package_3dgs_export, supported_ingest_adapters
from aura.inspection import inspect_scene_rays, native_demo_interaction_probes
from aura.migration import migration_report
from aura.carrier_payloads import SurfaceCellPayload
from aura.package import load_package, package_scene
from aura.ray import Ray
from aura.render import compare_images, read_ppm, render_orthographic
from aura.scene import AuraScene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aura")
    sub = parser.add_subparsers(dest="command", required=True)

    native_demo = sub.add_parser(
        "write-native-demo-package",
        help="Write a mixed-carrier native AURA package from evidence decomposition",
    )
    native_demo.add_argument("--output-dir", type=Path, default=Path("outputs/native-demo.aura"))

    build_native = sub.add_parser(
        "build-native-demo",
        help="Build the mixed-carrier native AURA demo package without 3DGS input",
    )
    build_native.add_argument("--output-dir", type=Path, default=Path("outputs/native-demo.aura"))

    demo = sub.add_parser("write-demo-package", help="Write a tiny single-surface .aura package scaffold")
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

    query = sub.add_parser("query-demo", help="Run a reference ray query against the native mixed-carrier demo")
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

    benchmark = sub.add_parser("benchmark-plan", help="Print the reproducible AURA benchmark and ablation plan as JSON")

    reference_benchmark = sub.add_parser("benchmark-reference", help="Run the CPU reference benchmark for a .aura package")
    reference_benchmark.add_argument("package_dir", type=Path)
    reference_benchmark.add_argument("--width", type=int, default=16)
    reference_benchmark.add_argument("--height", type=int, default=16)

    ingest = sub.add_parser("ingest-adapters", help="Print AURA-Ingest adapters and their EvidenceSample contracts as JSON")

    inspect_rays = sub.add_parser("inspect-rays", help="Inspect reference ray-query outputs for a .aura package")
    inspect_rays.add_argument("package_dir", type=Path)
    inspect_rays.add_argument("--native-demo-probes", action="store_true")

    migrate = sub.add_parser("migration-plan", help="Print package schema migration status as JSON")
    migrate.add_argument("package_dir", type=Path)

    args = parser.parse_args(argv)
    native_scene = native_demo_scene()
    if args.command in {"write-native-demo-package", "build-native-demo"}:
        print(package_scene(native_scene, fallbacks={"mesh": "fallback/native-preview.glb"}).write(args.output_dir))
        return 0
    if args.command == "write-demo-package":
        print(package_scene(demo_scene(), fallbacks={"mesh": "fallback/preview.glb", "splat": "fallback/preview.splat"}).write(args.output_dir))
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
        result = native_scene.ray_query(Ray(origin=(args.x, args.y, -2.0), direction=(0.0, 0.0, 1.0)))
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
    if args.command == "benchmark-plan":
        print(json.dumps(default_benchmark_suite().to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-reference":
        package = load_package(args.package_dir)
        print(json.dumps(run_reference_benchmark(package, package_dir=args.package_dir, render_width=args.width, render_height=args.height), indent=2, sort_keys=True))
        return 0
    if args.command == "ingest-adapters":
        print(json.dumps([adapter.to_dict() for adapter in supported_ingest_adapters()], indent=2, sort_keys=True))
        return 0
    if args.command == "inspect-rays":
        package = load_package(args.package_dir)
        inspections = native_demo_interaction_probes(package.scene) if args.native_demo_probes else inspect_scene_rays(package.scene)
        print(json.dumps([inspection.to_dict() for inspection in inspections], indent=2, sort_keys=True))
        return 0
    if args.command == "migration-plan":
        package = load_package(args.package_dir)
        print(json.dumps(migration_report(package.asset.version).to_dict(), indent=2, sort_keys=True))
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
        payload=SurfaceCellPayload(normal=(0.0, 0.0, -1.0), thickness=0.1, roughness=0.65).to_dict(),
    )
    chunk = AuraChunk(id="root", bounds=bounds, element_ids=("wall_patch",), lod=0)
    return AuraScene(name="demo", elements=(element,), chunks=(chunk,))


def native_demo_scene() -> AuraScene:
    return decompose_evidence(
        (
            EvidenceSample(
                id="surface_wall",
                bounds=Bounds((-0.75, -0.75, 0.0), (-0.25, -0.25, 0.1)),
                evidence=RegionEvidence(geometry_confidence=0.9, material_confidence=0.75, edit_need=0.7),
                color=(0.8, 0.72, 0.62),
                opacity=0.9,
                normal=(0.0, 0.0, -1.0),
                material_id="mat_wall_plaster",
                semantic_label="wall",
                confidence_map={"geometry": 0.9, "material": 0.75},
                edit={"editable": True, "group": "wall"},
            ),
            EvidenceSample(
                id="soft_volume",
                bounds=Bounds((-0.15, -0.7, 0.0), (0.35, -0.2, 0.8)),
                evidence=RegionEvidence(fuzzy_confidence=0.85, geometry_confidence=0.25),
                color=(0.55, 0.68, 0.9),
                opacity=0.35,
                confidence=0.72,
                material_id="mat_soft_volume",
            ),
            EvidenceSample(
                id="woven_frequency",
                bounds=Bounds((0.45, -0.7, 0.0), (0.95, -0.2, 0.15)),
                evidence=RegionEvidence(high_frequency=0.92, geometry_confidence=0.65),
                color=(0.95, 0.85, 0.35),
                opacity=0.75,
                material_id="mat_woven_detail",
            ),
            EvidenceSample(
                id="view_residual",
                bounds=Bounds((-0.75, 0.05, 0.0), (-0.25, 0.55, 0.2)),
                evidence=RegionEvidence(view_dependent=0.88, material_confidence=0.25, image_error=0.55),
                color=(0.4, 0.75, 0.7),
                opacity=0.6,
            ),
            EvidenceSample(
                id="semantic_object",
                bounds=Bounds((-0.1, 0.05, 0.0), (0.35, 0.5, 0.2)),
                evidence=RegionEvidence(semantic_confidence=0.95),
                color=(0.75, 0.55, 0.95),
                opacity=0.45,
                semantic_label="fixture_object",
                edit={"selectable": True},
            ),
            EvidenceSample(
                id="compact_detail",
                bounds=Bounds((0.5, 0.05, 0.0), (0.8, 0.35, 0.15)),
                evidence=RegionEvidence(compact_detail=0.9, image_error=0.2),
                color=(0.95, 0.45, 0.35),
                opacity=0.85,
            ),
            EvidenceSample(
                id="gaussian_fallback",
                bounds=Bounds((0.85, 0.3, 0.0), (1.05, 0.5, 0.2)),
                evidence=RegionEvidence(image_error=0.05, geometry_confidence=0.3, edit_need=0.1),
                color=(0.65, 0.65, 0.65),
                opacity=0.5,
                confidence=0.6,
            ),
        ),
        name="native_demo",
    )


if __name__ == "__main__":
    raise SystemExit(main())
