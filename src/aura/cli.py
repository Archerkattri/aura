from __future__ import annotations

import argparse
import json
from pathlib import Path

from aura.assignment import RegionEvidence
from aura.benchmark import (
    native_demo_ray_query_expectations,
    run_ablation_benchmarks,
    run_core_reconstruction_benchmark,
    run_production_gate_report,
    run_ray_query_correctness_benchmark,
    run_reference_benchmark,
    run_visual_quality_benchmark,
    default_benchmark_suite,
)
from aura.core import ReconstructionConfig, load_training_dataset, reconstruct_demo_scene, write_synthetic_training_frames
from aura.cuda_kernels import cuda_kernel_extension_report, cuda_renderer_report
from aura.decomposition import EvidenceSample, decompose_evidence
from aura.elements import AuraChunk, AuraElement, Bounds
from aura.evolution import CarrierEvolutionPolicy
from aura.ingest import (
    load_3dgs_scene,
    load_capture_asset_tensors,
    load_capture_assets,
    load_capture_manifest,
    package_3dgs_export,
    capture_tensors_to_training_dataset,
    supported_ingest_adapters,
    write_capture_manifest_template,
    write_colmap_capture_manifest,
)
from aura.inspection import inspect_scene_rays, native_demo_interaction_probes
from aura.migration import migration_report
from aura.carrier_payloads import SurfaceCellPayload
from aura.package import load_package, package_scene
from aura.ray import Ray
from aura.readiness import production_readiness_report
from aura.render import compare_images, read_ppm, render_orthographic
from aura.runtime_export import runtime_export_report
from aura.scene import AuraScene
from aura.semantic import SemanticEdge, SemanticGraph, SemanticNode
from aura.torch_optimizer import TorchOptimizationConfig, torch_optimize_capture_batches
from aura.torch_renderer import torch_renderer_status
from aura.torch_kernels import torch_carrier_kernel_report
from aura.training_targets import (
    capture_tensors_to_packed_render_batches,
    capture_tensors_to_render_targets,
    plan_capture_tensor_sampling,
)

NATIVE_DEMO_FALLBACKS = {
    "mesh": "fallback/native-preview.glb",
    "usd": "fallback/native-scene.usda",
    "preview": "fallback/native-demo.ppm",
}


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

    reconstruct_demo = sub.add_parser(
        "reconstruct-demo",
        help="Run the native AURA-Core posed fixture reconstruction path without 3DGS input",
    )
    reconstruct_demo.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruct-demo.aura"))
    reconstruct_demo.add_argument("--iterations", type=int, default=4)
    reconstruct_demo.add_argument("--frames", type=Path, default=None, help="JSON posed training frames for AURA-Core")
    _add_reconstruction_config_args(reconstruct_demo)

    frames_demo = sub.add_parser("write-training-frames-demo", help="Write a JSON posed-frame fixture for AURA-Core")
    frames_demo.add_argument("--output", type=Path, default=Path("outputs/training-frames.json"))

    capture_template = sub.add_parser(
        "write-capture-manifest-template",
        help="Write a real-capture manifest template for AURA-Core ingest",
    )
    capture_template.add_argument("--output", type=Path, default=Path("outputs/capture-manifest.json"))

    capture_to_training = sub.add_parser(
        "capture-manifest-to-training",
        help="Validate an AURA capture manifest and write the equivalent training dataset JSON",
    )
    capture_to_training.add_argument("manifest", type=Path)
    capture_to_training.add_argument("--output", type=Path, default=Path("outputs/training-from-capture.json"))
    capture_to_training.add_argument(
        "--load-assets",
        action="store_true",
        help="Read PNG, PPM/PGM, or COLMAP depth/normal-map assets and replace target summaries",
    )

    reconstruct_capture = sub.add_parser(
        "reconstruct-capture-manifest",
        help="Run the current AURA-Core reference reconstruction path from a capture manifest",
    )
    reconstruct_capture.add_argument("manifest", type=Path)
    reconstruct_capture.add_argument("--output-dir", type=Path, default=Path("outputs/reconstruct-capture.aura"))
    reconstruct_capture.add_argument("--iterations", type=int, default=4)
    reconstruct_capture.add_argument(
        "--load-assets",
        action="store_true",
        help="Read capture tensors and train from per-pixel image/depth/mask/normal targets",
    )
    reconstruct_capture.add_argument("--pixel-stride", type=int, default=1)
    reconstruct_capture.add_argument("--max-targets-per-frame", type=int, default=256)
    reconstruct_capture.add_argument("--tile-size", type=int, default=256)
    _add_reconstruction_config_args(reconstruct_capture)

    torch_optimize_capture = sub.add_parser(
        "torch-optimize-capture-manifest",
        help="Run the native torch AURA-Core optimization scaffold from capture tensors",
    )
    torch_optimize_capture.add_argument("manifest", type=Path)
    torch_optimize_capture.add_argument("--output-dir", type=Path, default=Path("outputs/torch-optimize-capture.aura"))
    torch_optimize_capture.add_argument("--iterations", type=int, default=4)
    torch_optimize_capture.add_argument("--pixel-stride", type=int, default=1)
    torch_optimize_capture.add_argument("--max-targets-per-frame", type=int, default=256)
    torch_optimize_capture.add_argument("--tile-size", type=int, default=256)
    torch_optimize_capture.add_argument(
        "--max-targets-per-batch",
        type=int,
        default=None,
        help="Maximum packed capture targets per torch optimizer step",
    )
    torch_optimize_capture.add_argument("--device", default=None, help="Torch device such as cpu or cuda")
    torch_optimize_capture.add_argument("--color-learning-rate", type=float, default=0.25)

    train = sub.add_parser("train", help="Train native AURA carriers from a capture manifest")
    train.add_argument("manifest", type=Path)
    train.add_argument("--output", type=Path, default=Path("outputs/scene.aura"))
    train.add_argument("--iterations", type=int, default=8)
    train.add_argument("--pixel-stride", type=int, default=1)
    train.add_argument("--max-targets-per-frame", type=int, default=4096)
    train.add_argument("--tile-size", type=int, default=256)
    train.add_argument("--max-targets-per-batch", type=int, default=1024)
    train.add_argument("--device", default=None, help="Torch device such as cuda or cpu")
    train.add_argument("--color-learning-rate", type=float, default=0.25)
    train.add_argument("--disable-evolution", action="store_true")
    train.add_argument("--split-image-loss-threshold", type=float, default=0.03)
    train.add_argument("--depth-anchor-loss-threshold", type=float, default=0.10)
    train.add_argument("--merge-image-loss-threshold", type=float, default=0.025)
    train.add_argument("--merge-depth-loss-threshold", type=float, default=0.04)
    train.add_argument("--demote-after-iteration", type=int, default=3)
    train.add_argument("--demote-image-loss-threshold", type=float, default=0.045)
    train.add_argument("--demote-depth-loss-threshold", type=float, default=0.02)

    inspect_capture_assets = sub.add_parser(
        "inspect-capture-assets",
        help="Load capture-manifest PNG, PPM/PGM, or COLMAP depth/normal-map assets and print deterministic summaries as JSON",
    )
    inspect_capture_assets.add_argument("manifest", type=Path)

    inspect_capture_tensors = sub.add_parser(
        "inspect-capture-tensors",
        help="Load capture-manifest image/depth/mask/normal assets and print tensor shape/sample metadata as JSON",
    )
    inspect_capture_tensors.add_argument("manifest", type=Path)

    plan_capture_sampling = sub.add_parser(
        "plan-capture-sampling",
        help="Plan tiled capture tensor sampling for CPU reference or future GPU loaders",
    )
    plan_capture_sampling.add_argument("manifest", type=Path)
    plan_capture_sampling.add_argument("--pixel-stride", type=int, default=1)
    plan_capture_sampling.add_argument("--max-targets-per-frame", type=int, default=256)
    plan_capture_sampling.add_argument("--tile-size", type=int, default=256)

    colmap_to_manifest = sub.add_parser(
        "colmap-to-capture-manifest",
        help="Convert a COLMAP binary or text model directory to an AURA capture manifest",
    )
    colmap_to_manifest.add_argument("colmap_dir", type=Path)
    colmap_to_manifest.add_argument("--output", type=Path, default=Path("outputs/capture-from-colmap.json"))
    colmap_to_manifest.add_argument("--root", default="data/custom-captures/colmap-scene")
    colmap_to_manifest.add_argument("--image-dir", default="images")

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

    export_report = sub.add_parser("export-report", help="Print native/glTF/USD runtime export readiness as JSON")
    export_report.add_argument("package_dir", type=Path)

    sub.add_parser("readiness-report", help="Print AURA production-readiness audit as JSON")

    render = sub.add_parser("render-package", help="Render a deterministic orthographic PPM preview from a .aura package")
    render.add_argument("package_dir", type=Path)
    render.add_argument("--output", type=Path, default=Path("outputs/preview.ppm"))
    render.add_argument("--width", type=int, default=64)
    render.add_argument("--height", type=int, default=64)

    render_native = sub.add_parser("render", help="Render a .aura package to a PPM image")
    render_native.add_argument("package_dir", type=Path)
    render_native.add_argument("--output", type=Path, default=Path("outputs/render.ppm"))
    render_native.add_argument("--width", type=int, default=64)
    render_native.add_argument("--height", type=int, default=64)

    compare = sub.add_parser("compare-renders", help="Compare two PPM previews and print JSON MSE/PSNR metrics")
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--min-psnr", type=float, default=None)

    benchmark = sub.add_parser("benchmark-plan", help="Print the reproducible AURA benchmark and ablation plan as JSON")

    core_benchmark = sub.add_parser("benchmark-core", help="Run the native AURA-Core reconstruction benchmark")
    core_benchmark.add_argument("--iterations", type=int, default=6)

    reference_benchmark = sub.add_parser("benchmark-reference", help="Run the CPU reference benchmark for a .aura package")
    reference_benchmark.add_argument("package_dir", type=Path)
    reference_benchmark.add_argument("--width", type=int, default=16)
    reference_benchmark.add_argument("--height", type=int, default=16)
    reference_benchmark.add_argument("--include-ablations", action="store_true")

    visual_benchmark = sub.add_parser("benchmark-visual", help="Compare a rendered .aura package preview against a teacher/reference PPM")
    visual_benchmark.add_argument("package_dir", type=Path)
    visual_benchmark.add_argument("reference", type=Path)
    visual_benchmark.add_argument("--baseline-label", default="teacher")
    visual_benchmark.add_argument("--width", type=int, default=None)
    visual_benchmark.add_argument("--height", type=int, default=None)
    visual_benchmark.add_argument("--min-psnr", type=float, default=None)

    production_gate = sub.add_parser(
        "production-gate-report",
        help="Print the native AURA production-claim gate for a .aura package as JSON",
    )
    production_gate.add_argument("package_dir", type=Path)
    production_gate.add_argument("--visual-baseline-label", default="reference_preview_self")
    production_gate.add_argument(
        "--external-visual-reference",
        action="store_true",
        help="Mark the visual gate context as external; run benchmark-visual separately for actual metrics",
    )

    ray_benchmark = sub.add_parser("benchmark-ray-query", help="Score ray-query correctness for a .aura package")
    ray_benchmark.add_argument("package_dir", type=Path)
    ray_benchmark.add_argument(
        "--native-demo-expectations",
        action="store_true",
        help="Use the native demo first-hit/depth/transmittance/semantic/material expectation set",
    )

    ingest = sub.add_parser("ingest-adapters", help="Print AURA-Ingest adapters and their EvidenceSample contracts as JSON")

    torch_status = sub.add_parser("torch-renderer-status", help="Print optional PyTorch/CUDA renderer availability as JSON")

    sub.add_parser("torch-kernel-report", help="Print native carrier torch/CUDA kernel readiness as JSON")
    cuda_build = sub.add_parser("cuda-kernel-build-report", help="Probe native CUDA carrier extension build/load readiness as JSON")
    cuda_build.add_argument("--build", action="store_true", help="Attempt to compile and load the packaged CUDA source")
    cuda_build.add_argument("--verbose", action="store_true", help="Print verbose torch extension build output when --build is used")
    sub.add_parser("cuda-renderer-report", help="Print CPU-safe CUDA renderer callable API readiness as JSON")

    inspect_rays = sub.add_parser("inspect-rays", help="Inspect reference ray-query outputs for a .aura package")
    inspect_rays.add_argument("package_dir", type=Path)
    inspect_rays.add_argument("--native-demo-probes", action="store_true")

    migrate = sub.add_parser("migration-plan", help="Print package schema migration status as JSON")
    migrate.add_argument("package_dir", type=Path)

    args = parser.parse_args(argv)
    native_scene = native_demo_scene()
    if args.command in {"write-native-demo-package", "build-native-demo"}:
        print(package_scene(native_scene, fallbacks=NATIVE_DEMO_FALLBACKS).write(args.output_dir))
        return 0
    if args.command == "reconstruct-demo":
        dataset = load_training_dataset(args.frames) if args.frames is not None else None
        result = reconstruct_demo_scene(
            _reconstruction_config_from_args(args),
            frames=dataset.frames if dataset is not None else None,
            regions=dataset.regions if dataset is not None else None,
        )
        package_dir = package_scene(result.scene, fallbacks={"mesh": "fallback/reconstruct-preview.glb"}).write(args.output_dir)
        report_path = package_dir / "training_report.json"
        report_path.write_text(json.dumps(result.report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(package_dir)
        return 0
    if args.command == "write-training-frames-demo":
        print(write_synthetic_training_frames(args.output))
        return 0
    if args.command == "write-capture-manifest-template":
        print(write_capture_manifest_template(args.output))
        return 0
    if args.command == "capture-manifest-to-training":
        manifest = load_capture_manifest(args.manifest)
        out = args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(manifest.to_training_dataset(load_assets=args.load_assets).to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(out)
        return 0
    if args.command == "reconstruct-capture-manifest":
        manifest = load_capture_manifest(args.manifest)
        tensors = load_capture_asset_tensors(manifest) if args.load_assets else None
        dataset = (
            capture_tensors_to_training_dataset(manifest, tensors)
            if tensors is not None
            else manifest.to_training_dataset(load_assets=False)
        )
        render_targets = None
        sampling_plan = None
        if tensors is not None:
            sampling_plan = plan_capture_tensor_sampling(
                dataset.frames,
                tensors,
                pixel_stride=args.pixel_stride,
                max_targets_per_frame=args.max_targets_per_frame,
                tile_size=args.tile_size,
            )
            pixel_targets = capture_tensors_to_render_targets(
                dataset.frames,
                tensors,
                pixel_stride=args.pixel_stride,
                max_targets_per_frame=args.max_targets_per_frame,
            )
            render_targets = tuple(target.render_target for target in pixel_targets)
        result = reconstruct_demo_scene(
            _reconstruction_config_from_args(args),
            frames=dataset.frames,
            regions=dataset.regions,
            render_targets=render_targets,
            name="reconstruct_capture",
        )
        package_dir = package_scene(result.scene, fallbacks={"mesh": "fallback/reconstruct-capture-preview.glb"}).write(args.output_dir)
        report_path = package_dir / "training_report.json"
        report = result.report.to_dict()
        if sampling_plan is not None:
            report["captureSamplingPlan"] = sampling_plan.to_dict()
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(package_dir)
        return 0
    if args.command == "torch-optimize-capture-manifest":
        manifest = load_capture_manifest(args.manifest)
        tensors = load_capture_asset_tensors(manifest)
        dataset = capture_tensors_to_training_dataset(manifest, tensors)
        sampling_plan = plan_capture_tensor_sampling(
            dataset.frames,
            tensors,
            pixel_stride=args.pixel_stride,
            max_targets_per_frame=args.max_targets_per_frame,
            tile_size=args.tile_size,
            max_targets_per_batch=args.max_targets_per_batch,
        )
        scene = _scene_from_training_dataset(dataset, name="torch_optimize_capture")
        packed_batches = capture_tensors_to_packed_render_batches(
            dataset.frames,
            tensors,
            pixel_stride=args.pixel_stride,
            max_targets_per_frame=args.max_targets_per_frame,
            tile_size=args.tile_size,
            max_targets_per_batch=args.max_targets_per_batch,
        )
        result = torch_optimize_capture_batches(
            scene,
            packed_batches,
            TorchOptimizationConfig(
                iterations=args.iterations,
                color_learning_rate=args.color_learning_rate,
                max_samples_per_batch=sampling_plan.max_targets_per_batch,
            ),
            device=args.device,
        )
        package_dir = package_scene(result.scene, fallbacks={"mesh": "fallback/torch-optimize-capture-preview.glb"}).write(args.output_dir)
        report = {
            "format": "AURA_CORE_TORCH_OPTIMIZATION_REPORT",
            "name": result.scene.name,
            "stages": [
                "capture_manifest_assets",
                "native_evidence_initialization",
                "torch_packed_capture_batches",
                "torch_reference_optimization",
                "aura_package_export_ready",
            ],
            "sources": ["capture_tensor_pixels", "training_regions", "depth_targets", "normal_targets"],
            "captureSamplingPlan": sampling_plan.to_dict(),
            "packedBatchCount": len(packed_batches),
            "packedTargetCount": sum(batch.target_count for batch in packed_batches),
            "torch": torch_renderer_status().to_dict(),
            **result.to_dict(),
        }
        report_path = package_dir / "torch_training_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(package_dir)
        return 0
    if args.command == "train":
        package_dir = _train_capture_manifest_command(args)
        print(package_dir)
        return 0
    if args.command == "inspect-capture-assets":
        manifest = load_capture_manifest(args.manifest)
        print(json.dumps([item.to_dict() for item in load_capture_assets(manifest)], indent=2, sort_keys=True))
        return 0
    if args.command == "inspect-capture-tensors":
        manifest = load_capture_manifest(args.manifest)
        print(json.dumps([item.to_dict() for item in load_capture_asset_tensors(manifest)], indent=2, sort_keys=True))
        return 0
    if args.command == "plan-capture-sampling":
        manifest = load_capture_manifest(args.manifest)
        tensors = load_capture_asset_tensors(manifest)
        dataset = capture_tensors_to_training_dataset(manifest, tensors)
        plan = plan_capture_tensor_sampling(
            dataset.frames,
            tensors,
            pixel_stride=args.pixel_stride,
            max_targets_per_frame=args.max_targets_per_frame,
            tile_size=args.tile_size,
        )
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "colmap-to-capture-manifest":
        print(write_colmap_capture_manifest(args.colmap_dir, args.output, root=args.root, image_dir=args.image_dir))
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
    if args.command == "export-report":
        package = load_package(args.package_dir)
        print(json.dumps(runtime_export_report(package).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "readiness-report":
        print(json.dumps(production_readiness_report().to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command in {"render-package", "render"}:
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
    if args.command == "torch-renderer-status":
        print(json.dumps(torch_renderer_status().to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "torch-kernel-report":
        print(json.dumps(torch_carrier_kernel_report(), indent=2, sort_keys=True))
        return 0
    if args.command == "cuda-kernel-build-report":
        print(json.dumps(cuda_kernel_extension_report(build=args.build, verbose=args.verbose), indent=2, sort_keys=True))
        return 0
    if args.command == "cuda-renderer-report":
        print(json.dumps(cuda_renderer_report(), indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-core":
        print(json.dumps(run_core_reconstruction_benchmark(iterations=args.iterations), indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-reference":
        package = load_package(args.package_dir)
        if args.include_ablations:
            payload = run_ablation_benchmarks(package, package_dir=args.package_dir, render_width=args.width, render_height=args.height)
        else:
            payload = run_reference_benchmark(package, package_dir=args.package_dir, render_width=args.width, render_height=args.height)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-visual":
        package = load_package(args.package_dir)
        payload = run_visual_quality_benchmark(
            package,
            read_ppm(args.reference),
            baseline_label=args.baseline_label,
            render_width=args.width,
            render_height=args.height,
            min_psnr=args.min_psnr,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 1
    if args.command == "production-gate-report":
        package = load_package(args.package_dir)
        payload = run_production_gate_report(
            package,
            visual_baseline_label=args.visual_baseline_label,
            visual_self_reference=not args.external_visual_reference,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark-ray-query":
        package = load_package(args.package_dir)
        if not args.native_demo_expectations:
            raise ValueError("benchmark-ray-query currently requires --native-demo-expectations")
        payload = run_ray_query_correctness_benchmark(package.scene, native_demo_ray_query_expectations())
        print(json.dumps(payload, indent=2, sort_keys=True))
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


def _add_reconstruction_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--color-learning-rate", type=float, default=0.35)
    parser.add_argument(
        "--render-backend",
        choices=("cpu", "torch", "auto"),
        default="cpu",
        help="Renderer used by reconstruction iterations; torch uses the native tensor ray-query path",
    )
    parser.add_argument("--device", default=None, help="Torch device for --render-backend torch/auto, such as cpu or cuda")
    parser.add_argument("--require-cuda", action="store_true", help="Fail unless reconstruction resolves to a CUDA torch device")
    parser.add_argument("--disable-adaptive-evolution", action="store_true")
    parser.add_argument("--split-image-loss-threshold", type=float, default=0.03)
    parser.add_argument("--depth-anchor-loss-threshold", type=float, default=0.10)
    parser.add_argument("--merge-image-loss-threshold", type=float, default=0.025)
    parser.add_argument("--merge-depth-loss-threshold", type=float, default=0.04)
    parser.add_argument("--demote-after-iteration", type=int, default=3)
    parser.add_argument("--demote-image-loss-threshold", type=float, default=0.045)
    parser.add_argument("--demote-depth-loss-threshold", type=float, default=0.02)


def _reconstruction_config_from_args(args: argparse.Namespace) -> ReconstructionConfig:
    return ReconstructionConfig(
        iterations=args.iterations,
        color_learning_rate=args.color_learning_rate,
        render_backend=args.render_backend,
        torch_device=args.device,
        require_cuda=args.require_cuda,
        enable_adaptive_evolution=not args.disable_adaptive_evolution,
        split_image_loss_threshold=args.split_image_loss_threshold,
        depth_anchor_loss_threshold=args.depth_anchor_loss_threshold,
        merge_image_loss_threshold=args.merge_image_loss_threshold,
        merge_depth_loss_threshold=args.merge_depth_loss_threshold,
        demote_after_iteration=args.demote_after_iteration,
        demote_image_loss_threshold=args.demote_image_loss_threshold,
        demote_depth_loss_threshold=args.demote_depth_loss_threshold,
    )


def _train_capture_manifest_command(args: argparse.Namespace) -> Path:
    manifest = load_capture_manifest(args.manifest)
    tensors = load_capture_asset_tensors(manifest)
    dataset = capture_tensors_to_training_dataset(manifest, tensors)
    sampling_plan = plan_capture_tensor_sampling(
        dataset.frames,
        tensors,
        pixel_stride=args.pixel_stride,
        max_targets_per_frame=args.max_targets_per_frame,
        tile_size=args.tile_size,
        max_targets_per_batch=args.max_targets_per_batch,
    )
    packed_batches = capture_tensors_to_packed_render_batches(
        dataset.frames,
        tensors,
        pixel_stride=args.pixel_stride,
        max_targets_per_frame=args.max_targets_per_frame,
        tile_size=args.tile_size,
        max_targets_per_batch=args.max_targets_per_batch,
    )
    scene = _scene_from_training_dataset(dataset, name="aura_train")
    result = torch_optimize_capture_batches(
        scene,
        packed_batches,
        TorchOptimizationConfig(
            iterations=args.iterations,
            color_learning_rate=args.color_learning_rate,
            max_samples_per_batch=sampling_plan.max_targets_per_batch,
            evolution_policy=None
            if args.disable_evolution
            else CarrierEvolutionPolicy(
                split_image_loss_threshold=args.split_image_loss_threshold,
                depth_anchor_loss_threshold=args.depth_anchor_loss_threshold,
                merge_image_loss_threshold=args.merge_image_loss_threshold,
                merge_depth_loss_threshold=args.merge_depth_loss_threshold,
                demote_after_iteration=args.demote_after_iteration,
                demote_image_loss_threshold=args.demote_image_loss_threshold,
                demote_depth_loss_threshold=args.demote_depth_loss_threshold,
            ),
        ),
        device=args.device,
    )
    package_dir = package_scene(result.scene, fallbacks={"mesh": "fallback/aura-train-preview.glb"}).write(args.output)
    report = {
        "format": "AURA_TRAINING_REPORT",
        "name": result.scene.name,
        "manifest": str(args.manifest),
        "device": args.device,
        "stages": [
            "capture_manifest_assets",
            "native_evidence_initialization",
            "packed_capture_batches",
            "torch_native_differentiable_render_train",
            "adaptive_carrier_evolution" if not args.disable_evolution else "adaptive_carrier_evolution_disabled",
            "aura_package_export",
        ],
        "captureSamplingPlan": sampling_plan.to_dict(),
        "packedBatchCount": len(packed_batches),
        "packedTargetCount": sum(batch.target_count for batch in packed_batches),
        "torch": torch_renderer_status().to_dict(),
        "adaptiveEvolutionEnabled": not args.disable_evolution,
        **result.to_dict(),
    }
    report_path = package_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package_dir


def _scene_from_training_dataset(dataset, *, name: str) -> AuraScene:
    by_frame = {frame.id: frame for frame in dataset.frames}
    evidence = []
    for region in dataset.regions:
        frame = by_frame.get(region.frame_id)
        if frame is None:
            raise ValueError(f"training region {region.id} references unknown frame {region.frame_id}")
        evidence.append(region.to_evidence_sample(frame))
    if not evidence:
        raise ValueError("torch capture optimization requires at least one training region")
    return decompose_evidence(tuple(evidence), name=name)


def native_demo_scene() -> AuraScene:
    scene = decompose_evidence(
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
                confidence_map={"geometry": 0.9, "material": 0.75, "collision": 0.88, "shadow": 0.82},
                edit={"editable": True, "group": "room_shell", "operation": "insert_object_occlusion_probe"},
                metadata={
                    "demo_role": "opaque_surface_occluder",
                    "interaction_probe": "inserted_object_occlusion",
                    "export_proxy": "mesh",
                    "query_contract": "first_hit_normal_material_collision",
                },
            ),
            EvidenceSample(
                id="soft_volume",
                bounds=Bounds((-0.15, -0.7, 0.0), (0.35, -0.2, 0.8)),
                evidence=RegionEvidence(fuzzy_confidence=0.85, geometry_confidence=0.25),
                color=(0.55, 0.68, 0.9),
                opacity=0.35,
                confidence=0.72,
                material_id="mat_soft_volume",
                confidence_map={"density": 0.85, "geometry": 0.25, "transmittance": 0.78},
                edit={"editable": True, "group": "atmosphere", "operation": "attenuation_tuning"},
                metadata={
                    "demo_role": "translucent_volume",
                    "interaction_probe": "shadow_transmittance",
                    "export_proxy": "usd_volume_metadata",
                    "query_contract": "path_transmittance",
                },
            ),
            EvidenceSample(
                id="woven_frequency",
                bounds=Bounds((0.45, -0.7, 0.0), (0.95, -0.2, 0.15)),
                evidence=RegionEvidence(high_frequency=0.92, geometry_confidence=0.65),
                color=(0.95, 0.85, 0.35),
                opacity=0.75,
                material_id="mat_woven_detail",
                confidence_map={"frequency": 0.92, "geometry": 0.65, "alias_control": 0.74},
                edit={"editable": True, "group": "surface_detail", "operation": "texture_frequency_edit"},
                metadata={
                    "demo_role": "high_frequency_texture",
                    "interaction_probe": "ordered_detail_trace",
                    "export_proxy": "texture_metadata",
                    "query_contract": "carrier_color_modulation",
                },
            ),
            EvidenceSample(
                id="view_residual",
                bounds=Bounds((-0.75, 0.05, 0.0), (-0.25, 0.55, 0.2)),
                evidence=RegionEvidence(view_dependent=0.88, material_confidence=0.25, image_error=0.55),
                color=(0.4, 0.75, 0.7),
                opacity=0.6,
                confidence=0.68,
                material_id="mat_view_dependent_glaze",
                confidence_map={"view": 0.88, "material": 0.25, "residual": 0.55},
                edit={"editable": False, "group": "residuals", "operation": "bake_or_keep_native"},
                metadata={
                    "demo_role": "view_dependent_residual",
                    "interaction_probe": "reflection_ready_surface",
                    "export_proxy": "native_residual_only",
                    "query_contract": "residual_flag_confidence",
                },
            ),
            EvidenceSample(
                id="semantic_object",
                bounds=Bounds((-0.1, 0.05, 0.0), (0.35, 0.5, 0.2)),
                evidence=RegionEvidence(semantic_confidence=0.95),
                color=(0.75, 0.55, 0.95),
                opacity=0.45,
                confidence=0.95,
                material_id="mat_fixture_marker",
                semantic_label="fixture_object",
                confidence_map={"semantic": 0.95, "object": 0.93, "edit": 0.9},
                edit={"selectable": True, "editable": True, "group": "inserted_fixture", "operation": "object_level_edit"},
                metadata={
                    "demo_role": "semantic_object_handle",
                    "interaction_probe": "semantic_object_query",
                    "export_proxy": "usd_object_metadata",
                    "query_contract": "semantic_id_object_selection",
                },
            ),
            EvidenceSample(
                id="compact_detail",
                bounds=Bounds((0.5, 0.05, 0.0), (0.8, 0.35, 0.15)),
                evidence=RegionEvidence(compact_detail=0.9, image_error=0.2),
                color=(0.95, 0.45, 0.35),
                opacity=0.85,
                confidence=0.81,
                material_id="mat_compact_chip",
                confidence_map={"compact_support": 0.9, "image_residual": 0.2, "lod": 0.8},
                edit={"editable": True, "group": "detail_kernels", "operation": "local_support_move"},
                metadata={
                    "demo_role": "compact_bounded_kernel",
                    "interaction_probe": "local_detail_pick",
                    "export_proxy": "native_kernel_metadata",
                    "query_contract": "bounded_support_opacity",
                },
            ),
            EvidenceSample(
                id="gaussian_fallback",
                bounds=Bounds((0.85, 0.3, 0.0), (1.05, 0.5, 0.2)),
                evidence=RegionEvidence(image_error=0.05, geometry_confidence=0.3, edit_need=0.1),
                color=(0.65, 0.65, 0.65),
                opacity=0.5,
                confidence=0.6,
                material_id="mat_low_structure_fallback",
                gaussian_mean=(0.95, 0.4, 0.1),
                gaussian_covariance=((0.0064, 0.0, 0.0), (0.0, 0.0064, 0.0), (0.0, 0.0, 0.0049)),
                fallback_source="native-demo-low-structure-evidence",
                confidence_map={"structure": 0.3, "image_residual": 0.05, "fallback": 0.6},
                edit={"editable": False, "group": "fallbacks", "operation": "promote_when_evidence_improves"},
                metadata={
                    "demo_role": "explicit_gaussian_fallback",
                    "interaction_probe": "fallback_trace",
                    "export_proxy": "splat_fallback",
                    "query_contract": "covariance_weighted_fallback",
                },
            ),
        ),
        name="native_demo",
    )
    graph = scene.semantic_graph
    if {node.id for node in graph.nodes} >= {"object:wall", "object:fixture_object"}:
        graph = SemanticGraph(
            nodes=tuple(
                SemanticNode(
                    id=node.id,
                    label=node.label,
                    element_ids=node.element_ids,
                    confidence=node.confidence,
                    attributes={
                        **dict(node.attributes),
                        **(
                            {
                                "demoRole": "occluder_and_collision_proxy",
                                "exportTarget": "mesh",
                                "editGroup": "room_shell",
                            }
                            if node.label == "wall"
                            else {
                                "demoRole": "selectable_inserted_object",
                                "exportTarget": "usd_object_metadata",
                                "editGroup": "inserted_fixture",
                            }
                        ),
                    },
                )
                for node in graph.nodes
            ),
            edges=(
                SemanticEdge(
                    source="object:fixture_object",
                    target="object:wall",
                    relation="occluded_by",
                    confidence=0.9,
                ),
            ),
        )
    return AuraScene(name=scene.name, elements=scene.elements, chunks=scene.chunks, semantic_graph=graph)


if __name__ == "__main__":
    raise SystemExit(main())
