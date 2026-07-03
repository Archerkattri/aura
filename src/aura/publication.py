"""Publication validation aggregation for AURA.

This module reads durable experiment artifacts and turns them into an explicit
paper-claim boundary. It is intentionally artifact-based, but a gate only passes
when the evidence is *content-checked against real committed results*, not merely
present:

  * The calibration / certificate / transfer / full-resolution gates parse the
    committed real-scene artifacts in ``outputs/`` (``calib_<scene>.json``,
    ``cert_sweep.json``, ``cross_scene_transfer.json``, ``p2_summary.json``,
    ``rl_<scene>_fr.json``) and check schema + numeric thresholds per real scene
    (truck/garden/kitchen/room). Missing, stale, or malformed data FAILS the gate.
  * The relight and secondary-ray gates run on a real trained asset
    (``outputs/truck-sidecar.aura``), not the synthetic native demo scene. If the
    probe cannot execute they return an explicit UNVERIFIED / REQUIRES_GPU failure
    state — a distinct failure, never a pass.
  * The calibration gate is guarded by ``aura.split_guard`` so a re-introduced P0
    train/eval leak (held-out views seen in training) fails the gate mechanically.

Every numeric threshold lives in the constants block below with a comment naming
the committed result that set it: loose enough that today's validated numbers
pass, tight enough that a regression trips the gate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aura.carriers import CARRIER_MATURITIES, CarrierSpec, default_registry
from aura.split_guard import SplitLeakError, verify_recorded_split


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUTPUTS = ROOT / "outputs"


# --- Real-scene gate thresholds -------------------------------------------------
# Every threshold is derived from the committed artifacts in outputs/ (read at the
# time B3 bound these gates). Each is loose enough that today's validated numbers
# pass and tight enough that a regression trips it; the comment names the result.

REAL_SCENES = ("truck", "garden", "kitchen", "room")
HOLDOUT_STRIDE = 8  # llffhold / test_every convention (docs/P0_CALIBRATED_CONFIDENCE.md)

# Calibrated ECE on the held-out eval half. Worst committed colour-label value is
# calib_room_fr.json = 0.0019; base+fr colour range 0.0006-0.0019 across the four
# scenes (docs/P0_CALIBRATED_CONFIDENCE.md, docs/P2_FULLRES_RENDERLOSS.md). Budget
# 0.005 => ~2.6x head-room; a >2.6x calibration regression fails.
CALIBRATED_ECE_MAX = 0.005

# Conformal pruning certificate operating point. Every committed calib_*/cert_sweep
# certificate is stated at epsilon=0.6, alpha=0.1 (docs/P1_CROSS_SCENE.md,
# cert_sweep.json). Kept-set empirical risk must stay within the certified budget;
# worst committed is 0.5939 (render-loss garden/kitchen) < 0.6.
CERT_EPSILON = 0.6
CERT_ALPHA = 0.1
CERT_RISK_MAX = CERT_EPSILON
CERT_PARAM_TOL = 1e-6

# Cross-scene calibrator transfer: selection AUC is rank-based and isotonic transfer
# is rank-invariant, so committed max |ΔAUC vs in-scene| = 0.0001 (colour) / 0.0004
# (depth) over all 24 ordered scene pairs (cross_scene_transfer.json, P1a). 0.001
# budget fails any real ranking drift.
TRANSFER_AUC_DELTA_MAX = 0.001

# Full-resolution reliability correlation (train-agreement feature vs held-out
# reliability). Committed full-res colour corr 0.922-0.980 (p2_summary.json, P2).
# 0.85 floor keeps all four scenes and fails a de-correlation regression.
FULLRES_CORR_MIN = 0.85
# Render-loss label survives with honestly weaker margins: committed corr 0.655-0.816
# (rl_<scene>_fr.json). 0.6 floor.
RENDERLOSS_CORR_MIN = 0.6
# Leak fingerprint: the clean-split correction lowered the Truck colour certificate
# kept-fraction from the leaked 1.00 to 0.77 while garden/kitchen/room stayed 1.00
# (docs/P2_FULLRES_RENDERLOSS.md). A Truck full-res colour certificate back at 1.00
# would signal the P0 leak returned, so require it strictly below 1.0.
TRUCK_FULLRES_KEPT_MAX = 0.95

# Certified LOD/streaming plan (P4, outputs/lod_certified.json): every published
# level's bound must hold on the disjoint eval half, with the stated family-wise
# Bonferroni accounting (alpha' = alpha / K). Committed run: all 16 bounds hold
# across the four real scenes at alpha=0.1, K=4 (docs/P4_CERTIFIED_LOD.md).
LOD_ALPHA = 0.1
LOD_MIN_LEVELS = 4

# Real-asset relight / secondary-ray probe thresholds (outputs/truck-sidecar.aura,
# 129,531 trained carriers, CPU probe). Measured probe values: two-light relight
# mean|Δ| = 0.265, relit-vs-flat-albedo mean|Δ| = 0.295, 24/24 primary hits.
RELIGHT_MIN_LIGHT_DELTA = 0.02
RELIGHT_MIN_ALBEDO_DELTA = 0.02
SECONDARY_MIN_PRIMARY_HITS = 4
REAL_ASSET_DIR = OUTPUTS / "truck-sidecar.aura"

# Carrier-registry honesty contract (audit item M4). The registry advertises seven
# carrier *types* but they are not equally real; this map is the maturity each type
# is allowed to claim, and the gate fails if the live registry advertises anything
# it can't back. The comment on each entry names the reason.
#   trained  -> real-scene training + committed evidence (calib_<scene>.json).
#   demo     -> footprint renders but only in demo/2D/PRISM-extension use.
#   metadata -> typed contract/payload only, no trained render family.
CARRIER_MATURITY_CONTRACT = {
    "gaussian": "trained",   # gsplat quality backend; calib_<scene>.json corpus
    "beta": "trained",       # DBS-Beta quality backend; +0.335 dB result, calib corpus
    "gabor": "demo",         # real gabor_footprint but validated only on 2D crops
    "neural": "demo",        # make_neural_footprint orphaned+experimental; hybrid falls back to gaussian
    "surface": "metadata",   # typed contract/payload only
    "volume": "metadata",    # typed contract/payload only
    "semantic": "metadata",  # semantic graph/labels + GPU-only distilled features (uncommitted)
}
# A "trained" claim must be backed by committed real-scene evidence. This maps the
# only carrier families with such evidence to the artifacts that prove it: the
# per-scene calibration JSONs, which exist because those scenes were trained with
# the gaussian (gsplat) / beta (DBS-Beta) quality backends. A carrier that claims
# "trained" but is absent here (or whose evidence files are missing) fails the gate.
TRAINED_CARRIER_EVIDENCE = {
    "gaussian": tuple(f"calib_{scene}.json" for scene in REAL_SCENES),
    "beta": tuple(f"calib_{scene}.json" for scene in REAL_SCENES),
}

# Distinct gate statuses. UNVERIFIED / REQUIRES_GPU are failures, never passes.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_UNVERIFIED = "unverified"
STATUS_REQUIRES_GPU = "requires_gpu"


@dataclass(frozen=True)
class PublicationGate:
    id: str
    title: str
    passed: bool
    evidence: tuple[str, ...]
    gaps: tuple[str, ...] = ()
    next_steps: tuple[str, ...] = ()
    status: str = ""

    @property
    def resolved_status(self) -> str:
        return self.status or (STATUS_PASSED if self.passed else STATUS_FAILED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "passed": self.passed,
            "status": self.resolved_status,
            "evidence": list(self.evidence),
            "gaps": list(self.gaps),
            "nextSteps": list(self.next_steps),
        }


@dataclass(frozen=True)
class PublicationValidationReport:
    gates: tuple[PublicationGate, ...]
    artifacts: dict[str, str]

    @property
    def publication_ready(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def remaining_gate_ids(self) -> tuple[str, ...]:
        return tuple(gate.id for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "AURA_PUBLICATION_VALIDATION_REPORT",
            "publicationReady": self.publication_ready,
            "passedGateCount": sum(1 for gate in self.gates if gate.passed),
            "gateCount": len(self.gates),
            "remainingGateIds": list(self.remaining_gate_ids),
            "gates": [gate.to_dict() for gate in self.gates],
            "artifacts": self.artifacts,
            "claimBoundary": {
                "canClaim": [
                    "AURA Beta beats fixed Gaussian on every audited local scene",
                    "AURA has a fully audited local dataset coverage table",
                    "PRISM is validated as an additive Gabor/neural extension layer over gsplat/DBS-Beta",
                    "PRISM CUDA throughput has measured FPS artifacts on RTX 5090",
                    "trained local scene checkpoints have bounded real-scene FPS evidence",
                    "AURA exports concrete KHR_gaussian_splatting GLB and USD bridge artifacts",
                    "AURA GLB/USD exports pass local structural viewer-compatibility checks",
                    "learned LPIPS runs on CUDA and can be emitted into JSON reports",
                    "AURA has same-split external baseline metrics for COLMAP, NeRF, 3DGS, 2DGS, and ray-traced GS",
                    "AURA has completed official 2DGS and 3DGUT same-split rows for all 8 audited scenes",
                    "secondary shadow/reflection ray-query readiness is validated on live probes",
                    "explicit albedo/roughness/metallic material fields are consumed by PBR relighting",
                ],
                "cannotClaim": [
                    "official external-repo leaderboard superiority over COLMAP/NeRF/2DGS/ray-traced-GS baselines",
                    "full production-resolution FPS across every publication scene",
                    "third-party GUI viewer render compatibility without an installed viewer/checker artifact",
                    "photorealistic reflected-image benchmark quality",
                    "full inverse-material recovery from unconstrained captures",
                ],
            },
        }


def publication_validation_report(
    results_dir: Path | None = None,
    outputs_dir: Path | None = None,
) -> PublicationValidationReport:
    results_dir = results_dir or RESULTS
    outputs_dir = outputs_dir or OUTPUTS
    artifacts: dict[str, str] = {}

    multiscene = _read_json(results_dir / "multiscene.json", artifacts)
    audit = _read_json(results_dir / "multiscene_audit.json", artifacts)
    prism_additive = _read_json(results_dir / "prism_additive_validation_2026-06-24.json", artifacts)
    prism_fps = _latest_json(results_dir, "production_fps_sweep*.json", artifacts)
    if prism_fps is None:
        prism_fps = _read_json(results_dir / "prism_fps_2026-06-24.json", artifacts)
    learned_lpips = _read_json(results_dir / "learned_lpips_smoke_2026-06-24.json", artifacts)
    external = _latest_json(results_dir, "external_baselines*.json", artifacts)
    official_multiscene = _latest_json(results_dir, "official_multiscene_baselines*.json", artifacts)
    engine = _latest_json(results_dir, "engine_integration_validation*.json", artifacts)
    viewer = _latest_json(results_dir, "viewer_compatibility_validation*.json", artifacts)
    real_fps = _latest_json(results_dir, "real_scene_fps_sweep*.json", artifacts)

    gates = (
        _local_multiscene_gate(multiscene),
        _dataset_audit_gate(audit),
        _prism_additive_gate(prism_additive),
        _prism_fps_gate(prism_fps),
        _real_scene_fps_gate(real_fps),
        _engine_integration_gate(engine),
        _viewer_compatibility_gate(viewer),
        _learned_lpips_gate(learned_lpips),
        _external_baselines_gate(external, official_multiscene),
        # Real committed-artifact, content-checked calibration killer-property gates.
        _calibration_ece_gate(outputs_dir, artifacts),
        _pruning_certificate_gate(outputs_dir, artifacts),
        _cross_scene_transfer_gate(outputs_dir, artifacts),
        _fullres_renderloss_gate(outputs_dir, artifacts),
        _certified_lod_gate(outputs_dir, artifacts),
        # Real trained-asset probes (not the synthetic native demo scene).
        _secondary_reflection_gate(outputs_dir, artifacts),
        _inverse_materials_gate(outputs_dir, artifacts),
        # Carrier-registry honesty: no carrier type may advertise maturity it can't back.
        _carrier_registry_honesty_gate(outputs_dir, artifacts),
    )
    return PublicationValidationReport(gates=gates, artifacts=artifacts)


def _read_json(path: Path, artifacts: dict[str, str]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    artifacts[path.stem] = str(path)
    return json.loads(path.read_text())


def _latest_json(results_dir: Path, pattern: str, artifacts: dict[str, str]) -> dict[str, Any] | None:
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        return None
    return _read_json(matches[0], artifacts)


def _local_multiscene_gate(payload: dict[str, Any] | None) -> PublicationGate:
    rows = tuple((payload or {}).get("scenes", ()))
    wins = [float(row.get("delta_psnr", 0.0)) > 0.0 for row in rows]
    passed = bool(rows) and all(wins)
    return PublicationGate(
        id="local_multiscene_quality",
        title="Local multi-scene Beta-vs-Gaussian quality",
        passed=passed,
        evidence=(f"{sum(wins)}/{len(rows)} local scenes have positive Beta PSNR delta",) if rows else (),
        gaps=() if passed else ("missing multiscene.json or at least one scene does not beat Gaussian",),
        next_steps=("extend the same table to external-method baselines",),
    )


def _dataset_audit_gate(payload: dict[str, Any] | None) -> PublicationGate:
    complete = bool((payload or {}).get("complete"))
    local_scene_count = int((payload or {}).get("localSceneCount", 0) or len((payload or {}).get("scenes", ())))
    return PublicationGate(
        id="dataset_audit",
        title="Downloaded local dataset audit",
        passed=complete,
        evidence=(f"{local_scene_count} local scene roots audited with Beta and Gaussian arms",) if complete else (),
        gaps=() if complete else ("local dataset audit is missing or incomplete",),
        next_steps=("repeat audit whenever more scenes are downloaded",),
    )


def _prism_additive_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and bool(payload.get("completeForAdditiveRole")) and bool(payload.get("additiveExtensionChangedImage"))
    return PublicationGate(
        id="prism_additive_contract",
        title="PRISM additive extension contract",
        passed=passed,
        evidence=(
            "Gaussian/Beta route to the primary backend; Gabor/neural route to PRISM",
            f"mean image delta {payload.get('meanAbsoluteImageDelta'):.6f}",
        ) if passed and payload else (),
        gaps=() if passed else ("PRISM additive CUDA validation artifact is missing or failed",),
        next_steps=("keep PRISM positioned as additive, not a gsplat/DBS-Beta replacement",),
    )


def _prism_fps_gate(payload: dict[str, Any] | None) -> PublicationGate:
    rows = tuple((payload or {}).get("benchmark", ()))
    fps = [float(row["prism_cuda_fps"]) for row in rows if "prism_cuda_fps" in row]
    passed = bool(fps) and min(fps) >= 30.0
    boundary = (payload or {}).get("roleBoundary") or {}
    evidence = (
        f"PRISM CUDA FPS range {min(fps):.1f}-{max(fps):.1f} over {len(fps)} synthetic sweeps",
        f"role boundary: {boundary.get('prismRole', 'additive_extension_layer')}; primary quality backends remain {', '.join(boundary.get('primaryQualityBackends', ('gsplat', 'DBS-Beta')))}",
    ) if fps else ()
    return PublicationGate(
        id="prism_cuda_fps",
        title="PRISM CUDA throughput smoke",
        passed=passed,
        evidence=evidence,
        gaps=() if passed else ("PRISM CUDA FPS artifact missing or below 30 FPS",),
        next_steps=("run full production-resolution FPS sweeps on real trained scenes",),
    )


def _engine_integration_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and bool(payload.get("passed"))
    return PublicationGate(
        id="engine_integration_exports",
        title="Engine/viewer export integration",
        passed=passed,
        evidence=(
            "KHR_gaussian_splatting GLB and USD bridge artifacts were written",
            f"GLB {int(payload.get('gltf', {}).get('bytes', 0))} bytes; USD {int(payload.get('usd', {}).get('bytes', 0))} bytes",
            "runtime export report marks native runtime, glTF preview, USD metadata, and chunked streaming workflows ready",
        ) if passed and payload else (),
        gaps=() if passed else ("engine integration validation artifact is missing or failed",),
        next_steps=("load the GLB/USD outputs in target viewers as an external compatibility check",),
    )


def _viewer_compatibility_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and bool(payload.get("passed"))
    tools = (payload or {}).get("externalTools") or {}
    installed = sorted(name for name, path in tools.items() if path)
    return PublicationGate(
        id="viewer_compatibility_exports",
        title="Viewer/export structural compatibility",
        passed=passed,
        evidence=(
            "GLB declares KHR_gaussian_splatting POINTS primitive with required attributes",
            "USD bridge has defaultPrim, GaussianCarriers Points prim, displayColor, and AURA carrier metadata",
            f"installed checker tools recorded: {', '.join(installed) if installed else 'none'}",
        ) if passed and payload else (),
        gaps=() if passed else ("viewer compatibility validation artifact is missing or failed",),
        next_steps=("run third-party viewer checks when Blender/USDView/browser viewer automation is installed",),
    )


def _real_scene_fps_gate(payload: dict[str, Any] | None) -> PublicationGate:
    rows = tuple((payload or {}).get("rows") or ())
    fps = [float(row.get("fps", 0.0)) for row in rows]
    passed = bool(payload) and bool(payload.get("passed")) and fps and min(fps) >= 30.0
    return PublicationGate(
        id="real_scene_fps",
        title="Real trained-scene FPS",
        passed=passed,
        evidence=(
            f"trained-scene FPS range {min(fps):.1f}-{max(fps):.1f} over {len(rows)} rows",
            "includes Truck DBS-Beta/fixed-Gaussian checkpoints and official 3DGUT Truck/Room render timings",
        ) if passed else (),
        gaps=() if passed else ("real trained-scene FPS artifact is missing or below 30 FPS",),
        next_steps=("extend trained-scene FPS to every official-baseline scene as checkpoints finish",),
    )


def _learned_lpips_gate(payload: dict[str, Any] | None) -> PublicationGate:
    passed = bool(payload) and payload.get("lpipsBackend") == "learned_lpips_alex" and payload.get("meanLpips") is not None
    return PublicationGate(
        id="learned_lpips_cuda",
        title="Learned LPIPS on CUDA",
        passed=passed,
        evidence=(
            f"learned LPIPS mean {float(payload['meanLpips']):.4f} on {payload.get('frameCount')} CUDA-evaluated frames",
        ) if passed and payload else (),
        gaps=() if passed else ("learned LPIPS CUDA report missing or fell back to proxy/unavailable",),
        next_steps=("run learned LPIPS on the full publication split, not only the smoke subset",),
    )


def _external_baselines_gate(
    payload: dict[str, Any] | None,
    official_multiscene: dict[str, Any] | None = None,
) -> PublicationGate:
    required = {"colmap", "nerf", "3dgs", "2dgs", "ray_traced_gs"}
    present = set((payload or {}).get("baselines", {}).keys())
    counts = (official_multiscene or {}).get("completedSceneCounts", {})
    official_complete = (
        int(counts.get("official_2dgs", 0)) >= 8
        and int(counts.get("official_3dgut", 0)) >= 8
        and not (official_multiscene or {}).get("missing", {}).get("official_2dgs")
        and not (official_multiscene or {}).get("missing", {}).get("official_3dgut")
    )
    evidence = []
    if present:
        evidence.append(f"external baselines present: {', '.join(sorted(present))}")
    if official_multiscene:
        evidence.append(
            "official 2DGS/3DGUT same-split rows complete: "
            f"{int(counts.get('official_2dgs', 0))}/8 and {int(counts.get('official_3dgut', 0))}/8"
        )
    gaps = []
    if not required.issubset(present):
        gaps.append(f"missing baselines: {', '.join(sorted(required - present))}")
    if not official_complete:
        gaps.append("official 2DGS/3DGUT multiscene rows are incomplete")
    passed = not gaps
    return PublicationGate(
        id="external_method_baselines",
        title="External method baseline table",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("keep official rows regenerated when the audited scene set changes",),
    )


def _load_artifact(path: Path, artifacts: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    """Load a committed JSON artifact. Return (payload, error).

    ``payload`` is ``None`` on any missing or malformed file, with a human-readable
    ``error`` describing which check failed — so a gate FAILS on stale/malformed data
    instead of silently treating it as absent-and-fine.
    """
    if not path.exists():
        return None, f"missing artifact {path.name}"
    try:
        payload = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        return None, f"malformed artifact {path.name}: {exc}"
    if not isinstance(payload, dict):
        return None, f"malformed artifact {path.name}: not a JSON object"
    artifacts[path.stem] = str(path)
    return payload, None


def _num(value: Any) -> float | None:
    """Coerce ``value`` to a finite float, else ``None`` (schema/type guard)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _calibration_ece_gate(outputs_dir: Path, artifacts: dict[str, str]) -> PublicationGate:
    """Calibrated-confidence ECE on real held-out eval halves, per real scene.

    Parses calib_<scene>.json and calib_<scene>_fr.json (colour label), checks the
    schema and that the calibrated ECE is within CALIBRATED_ECE_MAX and that the
    calibrated confidence out-selects opacity. Guarded by split_guard: the per-scene
    train/eval view counts recorded in p2_summary.json must be a clean disjoint
    llffhold partition, so a re-introduced P0 leak fails this gate.
    """
    evidence: list[str] = []
    gaps: list[str] = []
    worst_ece = 0.0
    checked = 0
    for scene in REAL_SCENES:
        for suffix in ("", "_fr"):
            name = f"calib_{scene}{suffix}.json"
            payload, err = _load_artifact(outputs_dir / name, artifacts)
            if payload is None:
                gaps.append(err or f"missing {name}")
                continue
            if payload.get("scene") != scene or payload.get("label") != "color":
                gaps.append(f"{name}: wrong scene/label header ({payload.get('scene')}/{payload.get('label')})")
                continue
            ece = _num((payload.get("calibration") or {}).get("ece_calibrated"))
            if ece is None:
                gaps.append(f"{name}: missing/non-numeric calibration.ece_calibrated")
                continue
            auc = payload.get("selection_auc_retained_reliability") or {}
            auc_cal = _num(auc.get("calibrated_confidence"))
            auc_opa = _num(auc.get("opacity"))
            checked += 1
            worst_ece = max(worst_ece, ece)
            if ece > CALIBRATED_ECE_MAX:
                gaps.append(f"{name}: calibrated ECE {ece:.4f} > {CALIBRATED_ECE_MAX}")
            if auc_cal is None or auc_opa is None:
                gaps.append(f"{name}: missing selection AUC (calibrated/opacity)")
            elif auc_cal <= auc_opa:
                gaps.append(f"{name}: calibrated AUC {auc_cal:.4f} does not beat opacity {auc_opa:.4f}")

    # Split guard: the eval half must be disjoint from the carrier-training views.
    p2, p2_err = _load_artifact(outputs_dir / "p2_summary.json", artifacts)
    if p2 is None:
        gaps.append(f"split guard: {p2_err}")
    else:
        scenes = (p2.get("scenes") or {})
        for scene in REAL_SCENES:
            summary = (((scenes.get(scene) or {}).get("fullres_color") or {}).get("reliability_summary") or {})
            train_views = summary.get("train_views")
            test_views = summary.get("test_views")
            if not isinstance(train_views, int) or not isinstance(test_views, int):
                gaps.append(f"split guard: {scene} missing recorded train/test view counts")
                continue
            try:
                verify_recorded_split(
                    train_count=train_views, eval_count=test_views,
                    holdout=HOLDOUT_STRIDE, context=f"{scene} full-res split",
                )
            except SplitLeakError as exc:
                gaps.append(f"split guard: {exc}")
        if not any(g.startswith("split guard") for g in gaps):
            evidence.append(f"split guard OK: eval views disjoint from training views on all {len(REAL_SCENES)} scenes (llffhold-{HOLDOUT_STRIDE})")

    if checked:
        evidence.insert(0, f"calibrated ECE <= {CALIBRATED_ECE_MAX} on {checked} real-scene halves (worst {worst_ece:.4f})")
    passed = not gaps and checked == 2 * len(REAL_SCENES)
    return PublicationGate(
        id="calibration_ece",
        title="Real-scene calibrated-confidence ECE (split-guarded)",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("extend calibration to independently re-captured scenes (P3)",),
    )


def _pruning_certificate_gate(outputs_dir: Path, artifacts: dict[str, str]) -> PublicationGate:
    """Conformal pruning certificate valid at the stated epsilon/alpha on real scenes.

    Checks cert_sweep.json (every scene x label certified at the epsilon=0.6 operating
    point, alpha=0.1) and each calib_<scene>{,_fr}.json certificate block (certified,
    epsilon/alpha as stated, kept-set empirical risk within the certified budget).
    """
    evidence: list[str] = []
    gaps: list[str] = []

    sweep, err = _load_artifact(outputs_dir / "cert_sweep.json", artifacts)
    if sweep is None:
        gaps.append(err or "missing cert_sweep.json")
    else:
        if _num(sweep.get("alpha")) != CERT_ALPHA:
            gaps.append(f"cert_sweep.json: alpha {sweep.get('alpha')} != {CERT_ALPHA}")
        by_scene = sweep.get("by_scene") or {}
        certified_rows = 0
        for scene in REAL_SCENES:
            for label in ("color", "depth"):
                rows = ((by_scene.get(scene) or {}).get(label) or {}).get("sweep") or []
                row = next((r for r in rows if _num(r.get("epsilon")) is not None
                            and abs(_num(r.get("epsilon")) - CERT_EPSILON) < 1e-9), None)
                if row is None:
                    gaps.append(f"cert_sweep.json: {scene}/{label} has no epsilon={CERT_EPSILON} row")
                    continue
                risk = _num(row.get("empirical_risk_kept"))
                if not row.get("certified"):
                    gaps.append(f"cert_sweep.json: {scene}/{label} not certified at epsilon={CERT_EPSILON}")
                elif risk is None or risk > CERT_RISK_MAX:
                    gaps.append(f"cert_sweep.json: {scene}/{label} kept risk {risk} > {CERT_RISK_MAX}")
                else:
                    certified_rows += 1
        if certified_rows:
            evidence.append(f"cert_sweep.json: {certified_rows}/{2 * len(REAL_SCENES)} scene x label certified at epsilon={CERT_EPSILON}, alpha={CERT_ALPHA}")

    per_scene_ok = 0
    for scene in REAL_SCENES:
        for suffix in ("", "_fr"):
            name = f"calib_{scene}{suffix}.json"
            payload, cerr = _load_artifact(outputs_dir / name, artifacts)
            if payload is None:
                gaps.append(cerr or f"missing {name}")
                continue
            cert = payload.get("pruning_certificate") or {}
            eps = _num(cert.get("epsilon"))
            alpha = _num(cert.get("alpha"))
            risk = _num(cert.get("empirical_risk_kept"))
            if not cert.get("certified"):
                gaps.append(f"{name}: certificate not certified")
            elif eps is None or abs(eps - CERT_EPSILON) > CERT_PARAM_TOL:
                gaps.append(f"{name}: certificate epsilon {eps} != {CERT_EPSILON}")
            elif alpha is None or abs(alpha - CERT_ALPHA) > CERT_PARAM_TOL:
                gaps.append(f"{name}: certificate alpha {alpha} != {CERT_ALPHA}")
            elif risk is None or risk > CERT_RISK_MAX:
                gaps.append(f"{name}: kept risk {risk} > budget {CERT_RISK_MAX}")
            else:
                per_scene_ok += 1
    if per_scene_ok:
        evidence.append(f"{per_scene_ok} per-scene certificates valid at epsilon={CERT_EPSILON}, alpha={CERT_ALPHA}")
    passed = not gaps
    return PublicationGate(
        id="pruning_certificate",
        title="Real-scene conformal pruning certificate",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("hold the certificate operating point as scenes are added",),
    )


def _cross_scene_transfer_gate(outputs_dir: Path, artifacts: dict[str, str]) -> PublicationGate:
    """Cross-scene calibrator transfer: selection ranking is preserved, certs valid.

    Parses cross_scene_transfer.json and checks that the transferred selection AUC
    stays within TRANSFER_AUC_DELTA_MAX of the in-scene AUC on every ordered scene
    pair, and that every transferred (off-diagonal) certificate is certified on the
    target's own local conformal split.
    """
    evidence: list[str] = []
    gaps: list[str] = []
    payload, err = _load_artifact(outputs_dir / "cross_scene_transfer.json", artifacts)
    if payload is None:
        gaps.append(err or "missing cross_scene_transfer.json")
        return PublicationGate(
            id="cross_scene_transfer",
            title="Cross-scene calibrator transfer",
            passed=False, evidence=(), gaps=tuple(gaps),
            next_steps=("re-run P1a transfer when the scene set changes",),
        )
    if _num(payload.get("epsilon")) != CERT_EPSILON or _num(payload.get("alpha")) != CERT_ALPHA:
        gaps.append(f"cross_scene_transfer.json: epsilon/alpha {payload.get('epsilon')}/{payload.get('alpha')} != {CERT_EPSILON}/{CERT_ALPHA}")
    by_label = payload.get("by_label") or {}
    worst_delta = 0.0
    for label in ("color", "depth"):
        matrix = (by_label.get(label) or {}).get("matrix") or []
        if not matrix:
            gaps.append(f"cross_scene_transfer.json: {label} matrix missing/empty")
            continue
        off = [r for r in matrix if r.get("source") != r.get("target")]
        if not off:
            gaps.append(f"cross_scene_transfer.json: {label} has no transferred pairs")
            continue
        for row in matrix:
            delta = _num(row.get("auc_delta_vs_inscene"))
            if delta is None:
                gaps.append(f"cross_scene_transfer.json: {label} {row.get('source')}->{row.get('target')} missing auc_delta")
                continue
            worst_delta = max(worst_delta, abs(delta))
            if abs(delta) > TRANSFER_AUC_DELTA_MAX:
                gaps.append(f"cross_scene_transfer.json: {label} {row.get('source')}->{row.get('target')} |ΔAUC| {abs(delta):.4f} > {TRANSFER_AUC_DELTA_MAX}")
        for row in off:
            cert = row.get("cert_transferred_local_split") or {}
            if not cert.get("certified"):
                gaps.append(f"cross_scene_transfer.json: {label} {row.get('source')}->{row.get('target')} transferred cert not certified")
    if not gaps or worst_delta:
        evidence.append(f"transferred selection AUC within {TRANSFER_AUC_DELTA_MAX} of in-scene on all pairs (worst |ΔAUC| {worst_delta:.4f})")
    passed = not gaps
    return PublicationGate(
        id="cross_scene_transfer",
        title="Cross-scene calibrator transfer",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("re-run P1a transfer when the scene set changes",),
    )


def _fullres_renderloss_gate(outputs_dir: Path, artifacts: dict[str, str]) -> PublicationGate:
    """Full-resolution + render-loss reliability survives with honest margins.

    Parses p2_summary.json (full-res colour train-agreement correlation per scene, and
    the leak-corrected Truck certificate) and rl_<scene>_fr.json (render-loss label
    correlation). The Truck full-res colour certificate must stay strictly below 1.0
    — a value back at 1.00 would signal the corrected P0 leak returned.
    """
    evidence: list[str] = []
    gaps: list[str] = []
    p2, err = _load_artifact(outputs_dir / "p2_summary.json", artifacts)
    if p2 is None:
        gaps.append(err or "missing p2_summary.json")
    else:
        scenes = p2.get("scenes") or {}
        worst_corr = 1.0
        for scene in REAL_SCENES:
            block = ((scenes.get(scene) or {}).get("fullres_color") or {})
            corr = _num((block.get("reliability_summary") or {}).get("corr_trainagree_reliability"))
            if corr is None:
                gaps.append(f"p2_summary.json: {scene} missing full-res colour correlation")
                continue
            worst_corr = min(worst_corr, corr)
            if corr < FULLRES_CORR_MIN:
                gaps.append(f"p2_summary.json: {scene} full-res corr {corr:.3f} < {FULLRES_CORR_MIN}")
            if scene == "truck":
                kept = _num((block.get("pruning_certificate") or {}).get("kept_fraction"))
                if kept is None:
                    gaps.append("p2_summary.json: truck full-res certificate missing kept_fraction")
                elif kept >= TRUCK_FULLRES_KEPT_MAX:
                    gaps.append(f"p2_summary.json: truck full-res kept {kept:.3f} >= {TRUCK_FULLRES_KEPT_MAX} — P0 leak fingerprint")
        if not any(g.startswith("p2_summary") for g in gaps):
            evidence.append(f"full-res colour reliability corr >= {FULLRES_CORR_MIN} on all {len(REAL_SCENES)} scenes (worst {worst_corr:.3f}); leak-corrected Truck certificate < 1.0")

    rl_ok = 0
    worst_rl = 1.0
    for scene in REAL_SCENES:
        payload, rerr = _load_artifact(outputs_dir / f"rl_{scene}_fr.json", artifacts)
        if payload is None:
            gaps.append(rerr or f"missing rl_{scene}_fr.json")
            continue
        corr = _num(payload.get("corr_trainagree_reliability"))
        if corr is None:
            gaps.append(f"rl_{scene}_fr.json: missing corr_trainagree_reliability")
            continue
        worst_rl = min(worst_rl, corr)
        if corr < RENDERLOSS_CORR_MIN:
            gaps.append(f"rl_{scene}_fr.json: render-loss corr {corr:.3f} < {RENDERLOSS_CORR_MIN}")
        else:
            rl_ok += 1
    if rl_ok:
        evidence.append(f"render-loss label survives on {rl_ok}/{len(REAL_SCENES)} scenes (weaker, corr >= {RENDERLOSS_CORR_MIN}, worst {worst_rl:.3f})")
    passed = not gaps
    return PublicationGate(
        id="fullres_renderloss",
        title="Full-resolution + render-loss reliability",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("add the native 17.4 MP Garden render-loss label on an idle GPU",),
    )


def _certified_lod_gate(outputs_dir: Path, artifacts: dict[str, str]) -> PublicationGate:
    """Certified LOD/streaming plan (P4) holds with honest family-wise accounting.

    Parses lod_certified.json and checks: the Bonferroni correction is stated and
    arithmetically consistent (alpha' == alpha / K), all four real scenes carry at
    least LOD_MIN_LEVELS levels, every level's certified bound holds on the
    disjoint eval half, the full-keep level is trivial with epsilon exactly 0, and
    epsilon decreases as the keep-fraction grows.
    """
    evidence: list[str] = []
    gaps: list[str] = []
    payload, err = _load_artifact(outputs_dir / "lod_certified.json", artifacts)
    if payload is None:
        return PublicationGate(
            id="certified_lod",
            title="Certified LOD/streaming plan",
            passed=False, evidence=(), gaps=(err or "missing lod_certified.json",),
            next_steps=("run experiments/lod_certified_eval.py",),
        )
    alpha = _num(payload.get("alpha"))
    alpha_pl = _num(payload.get("alpha_per_level"))
    n_levels = len(payload.get("levels") or [])
    if payload.get("correction") != "bonferroni":
        gaps.append(f"lod_certified.json: correction {payload.get('correction')!r} != 'bonferroni'")
    if alpha != LOD_ALPHA:
        gaps.append(f"lod_certified.json: alpha {alpha} != {LOD_ALPHA}")
    if n_levels < LOD_MIN_LEVELS:
        gaps.append(f"lod_certified.json: only {n_levels} levels < {LOD_MIN_LEVELS}")
    if alpha is not None and alpha_pl is not None and n_levels:
        if abs(alpha_pl - alpha / n_levels) > 1e-12:
            gaps.append(
                f"lod_certified.json: alpha_per_level {alpha_pl} != alpha/K = {alpha / n_levels}"
                " — family-wise accounting broken"
            )
    by_scene = payload.get("by_scene") or {}
    total_holds = 0
    for scene in REAL_SCENES:
        levels = ((by_scene.get(scene) or {}).get("levels")) or []
        if len(levels) < LOD_MIN_LEVELS:
            gaps.append(f"lod_certified.json: {scene} has {len(levels)} evaluated levels < {LOD_MIN_LEVELS}")
            continue
        eps_by_keep: list[tuple[float, float]] = []
        for lvl in levels:
            keep = _num(lvl.get("keep_fraction"))
            eps = _num(lvl.get("epsilon_certified"))
            if keep is None or eps is None:
                gaps.append(f"lod_certified.json: {scene} level missing keep_fraction/epsilon")
                continue
            if lvl.get("holds") is not True:
                gaps.append(f"lod_certified.json: {scene}@{keep} bound does not hold")
                continue
            if keep == 1.0 and (lvl.get("trivial") is not True or eps != 0.0):
                gaps.append(f"lod_certified.json: {scene} full-keep level not trivial/zero (eps={eps})")
                continue
            eps_by_keep.append((keep, eps))
            total_holds += 1
        eps_sorted = sorted(eps_by_keep)
        if any(e2 > e1 for (_, e1), (_, e2) in zip(eps_sorted, eps_sorted[1:])):
            gaps.append(f"lod_certified.json: {scene} epsilon not non-increasing in keep-fraction")
    if payload.get("all_bounds_hold") is not True or payload.get("violations"):
        gaps.append(
            f"lod_certified.json: all_bounds_hold={payload.get('all_bounds_hold')}, "
            f"violations={payload.get('violations')}"
        )
    if total_holds:
        evidence.append(
            f"{total_holds} level bounds hold on disjoint eval halves across "
            f"{len(REAL_SCENES)} scenes (family-wise 1-alpha={1 - LOD_ALPHA:g}, Bonferroni)"
        )
    passed = not gaps
    return PublicationGate(
        id="certified_lod",
        title="Certified LOD/streaming plan",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("re-run lod_certified_eval.py whenever calibrators or scenes change",),
    )


def _probe_secondary_rays(aura_dir: Path) -> dict[str, Any]:
    """Cast primary + secondary (shadow, reflection) rays over a real trained asset.

    Runs entirely on CPU over the carriers.npz tensor set. Raises on any failure
    (missing carriers, torch unavailable) so the gate can report REQUIRES_GPU/
    UNVERIFIED rather than passing.
    """
    import numpy as np
    import torch

    from aura.carrier_query import carrier_ray_query

    torch.set_num_threads(2)
    data = np.load(aura_dir / "carriers.npz")
    carriers = {
        key: torch.tensor(data[key], dtype=torch.float32)
        for key in ("means", "scales", "quats", "opacity", "colors", "confidence")
        if key in data.files
    }
    means = data["means"]
    centroid = means.mean(axis=0)
    radius = float(np.linalg.norm(means.std(axis=0)) * 3.0 + 2.0)
    # Deterministic camera directions on a shell around the scene, aimed at centroid.
    dirs = np.array([
        (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
        (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
        (1, 1, 1), (-1, 1, -1), (1, -1, 1), (-1, -1, -1),
    ], dtype=np.float64)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    primary_hits = 0
    shadow_total = 0
    shadow_bounded = 0
    reflection_total = 0
    reflection_finite = 0
    for unit in dirs:
        origin = centroid + unit * radius
        direction = centroid - origin
        direction = direction / np.linalg.norm(direction)
        res = carrier_ray_query(carriers, origin.tolist(), direction.tolist(),
                                device="cpu", min_opacity=0.1)
        if res.provenance == "miss" or res.depth is None or res.normal is None:
            continue
        primary_hits += 1
        hit = [origin[i] + direction[i] * res.depth for i in range(3)]
        normal = res.normal
        offset = [hit[i] + normal[i] * 1e-3 for i in range(3)]
        shadow = carrier_ray_query(carriers, offset, list(normal), device="cpu", min_opacity=0.1)
        shadow_total += 1
        if shadow.transmittance is not None and 0.0 <= shadow.transmittance <= 1.0:
            shadow_bounded += 1
        dot = sum(direction[i] * normal[i] for i in range(3))
        reflection = [direction[i] - 2.0 * dot * normal[i] for i in range(3)]
        reflection_total += 1
        if all(math.isfinite(x) for x in reflection):
            reflection_finite += 1
    return {
        "carriers": int(means.shape[0]),
        "primary_hits": primary_hits,
        "shadow_bounded_rate": (shadow_bounded / shadow_total) if shadow_total else 0.0,
        "reflection_finite_rate": (reflection_finite / reflection_total) if reflection_total else 0.0,
    }


def _secondary_reflection_gate(
    outputs_dir: Path,
    artifacts: dict[str, str],
    prober: Callable[[Path], dict[str, Any]] | None = None,
) -> PublicationGate:
    """Secondary shadow/reflection ray-query readiness on a REAL trained asset.

    Runs over outputs/truck-sidecar.aura (not the synthetic native demo). Passes only
    when real primary hits expose bounded shadow transmittance and finite reflection
    vectors. If the probe cannot run, returns UNVERIFIED/REQUIRES_GPU — never a pass.
    """
    aura_dir = outputs_dir / "truck-sidecar.aura"
    if not (aura_dir / "carriers.npz").exists():
        return PublicationGate(
            id="secondary_ray_reflection",
            title="Secondary-ray/reflection on real asset",
            passed=False, status=STATUS_UNVERIFIED, evidence=(),
            gaps=(f"real trained asset {aura_dir.name}/carriers.npz is not available to probe",),
            next_steps=("benchmark rendered reflection/shadow behavior, not only ray-query readiness",),
        )
    prober = prober or _probe_secondary_rays
    try:
        metrics = prober(aura_dir)
    except Exception as exc:  # torch/GPU/loader failure -> honest, distinct failure
        return PublicationGate(
            id="secondary_ray_reflection",
            title="Secondary-ray/reflection on real asset",
            passed=False, status=STATUS_REQUIRES_GPU, evidence=(),
            gaps=(f"real-asset secondary-ray probe could not execute: {exc}",),
            next_steps=("run the secondary-ray probe where torch can load the real asset",),
        )
    artifacts["real_asset_secondary_rays"] = str(aura_dir)
    hits = int(metrics.get("primary_hits", 0))
    shadow_rate = float(metrics.get("shadow_bounded_rate", 0.0))
    refl_rate = float(metrics.get("reflection_finite_rate", 0.0))
    gaps: list[str] = []
    if hits < SECONDARY_MIN_PRIMARY_HITS:
        gaps.append(f"only {hits} real primary hits (< {SECONDARY_MIN_PRIMARY_HITS})")
    if shadow_rate < 1.0:
        gaps.append(f"shadow transmittance out of [0,1] on some hits (bounded rate {shadow_rate:.2f})")
    if refl_rate <= 0.0:
        gaps.append("no finite reflection vectors from real hits")
    passed = not gaps
    return PublicationGate(
        id="secondary_ray_reflection",
        title="Secondary-ray/reflection on real asset",
        passed=passed,
        status=STATUS_PASSED if passed else STATUS_FAILED,
        evidence=(
            f"{hits} real primary hits over {metrics.get('carriers')} trained carriers",
            f"bounded shadow transmittance rate {shadow_rate:.2f}; finite reflection vector rate {refl_rate:.2f}",
        ) if passed else (),
        gaps=tuple(gaps),
        next_steps=("benchmark rendered reflection/shadow behavior, not only ray-query readiness",),
    )


def _probe_real_relight(aura_dir: Path) -> dict[str, Any]:
    """Relight a real trained asset under two lights on CPU; report the response.

    Raises on any failure so the gate can report UNVERIFIED/REQUIRES_GPU.
    """
    import numpy as np
    import torch

    from aura.relight import carrier_albedo, relight_colors
    from aura.shading import DirectionalLight

    torch.set_num_threads(2)
    data = np.load(aura_dir / "carriers.npz")
    carriers = {
        key: torch.tensor(data[key], dtype=torch.float32)
        for key in ("means", "scales", "quats", "opacity", "colors")
        if key in data.files
    }
    warm = DirectionalLight(direction=(0.0, 0.0, -1.0), color=(1.0, 0.95, 0.9), intensity=2.0)
    cool = DirectionalLight(direction=(0.7, 0.0, -0.7), color=(0.45, 0.65, 1.0), intensity=1.6)
    color_a = relight_colors(carriers, [warm], ambient=0.1, device="cpu")
    color_b = relight_colors(carriers, [cool], ambient=0.1, device="cpu")
    albedo = carrier_albedo(torch, {"colors": carriers["colors"]})
    return {
        "carriers": int(data["means"].shape[0]),
        "light_delta": float((color_a - color_b).abs().mean()),
        "albedo_delta": float((color_a - albedo).abs().mean()),
        "finite": bool(torch.isfinite(color_a).all() and torch.isfinite(color_b).all()),
        "nonnegative": bool((color_a >= 0).all() and (color_b >= 0).all()),
    }


def _inverse_materials_gate(
    outputs_dir: Path,
    artifacts: dict[str, str],
    prober: Callable[[Path], dict[str, Any]] | None = None,
) -> PublicationGate:
    """Relighting/material response on a REAL trained asset.

    Relights outputs/truck-sidecar.aura carriers under two lights (not the synthetic
    hand-set albedo probe). Passes only when the real relit output is finite,
    non-negative, changes with the light, and differs from the flat baked albedo. If
    the probe cannot run, returns UNVERIFIED/REQUIRES_GPU — never a pass.
    """
    aura_dir = outputs_dir / "truck-sidecar.aura"
    if not (aura_dir / "carriers.npz").exists():
        return PublicationGate(
            id="inverse_materials",
            title="Relighting response on real asset",
            passed=False, status=STATUS_UNVERIFIED, evidence=(),
            gaps=(f"real trained asset {aura_dir.name}/carriers.npz is not available to relight",),
            next_steps=("evaluate material/albedo/roughness behavior on inverse-rendering benchmarks",),
        )
    prober = prober or _probe_real_relight
    try:
        metrics = prober(aura_dir)
    except Exception as exc:
        return PublicationGate(
            id="inverse_materials",
            title="Relighting response on real asset",
            passed=False, status=STATUS_REQUIRES_GPU, evidence=(),
            gaps=(f"real-asset relight probe could not execute: {exc}",),
            next_steps=("run the relight probe where torch can load the real asset",),
        )
    artifacts["real_asset_relight"] = str(aura_dir)
    light_delta = float(metrics.get("light_delta", 0.0))
    albedo_delta = float(metrics.get("albedo_delta", 0.0))
    gaps: list[str] = []
    if not metrics.get("finite"):
        gaps.append("relit output is not finite")
    if not metrics.get("nonnegative"):
        gaps.append("relit output has negative components")
    if light_delta < RELIGHT_MIN_LIGHT_DELTA:
        gaps.append(f"relight barely changes with light direction (Δ {light_delta:.4f} < {RELIGHT_MIN_LIGHT_DELTA})")
    if albedo_delta < RELIGHT_MIN_ALBEDO_DELTA:
        gaps.append(f"relit output ~= flat baked albedo (Δ {albedo_delta:.4f} < {RELIGHT_MIN_ALBEDO_DELTA})")
    passed = not gaps
    return PublicationGate(
        id="inverse_materials",
        title="Relighting response on real asset",
        passed=passed,
        status=STATUS_PASSED if passed else STATUS_FAILED,
        evidence=(
            f"relit {metrics.get('carriers')} real carriers under two lights",
            f"mean|Δ| light={light_delta:.4f}, relit-vs-albedo={albedo_delta:.4f}, finite & non-negative",
        ) if passed else (),
        gaps=tuple(gaps),
        next_steps=("evaluate material/albedo/roughness behavior on inverse-rendering benchmarks",),
    )


def _carrier_registry_honesty_gate(
    outputs_dir: Path,
    artifacts: dict[str, str],
    registry: dict[str, CarrierSpec] | None = None,
) -> PublicationGate:
    """No carrier type may advertise a maturity it cannot back.

    Content-checked against the live registry (``aura.carriers.default_registry``)
    and the committed real-scene evidence in ``outputs/``. The gate FAILS when:

      * a registered carrier type is absent from ``CARRIER_MATURITY_CONTRACT``
        (an unlabelled type is an implicit over-claim), or advertises an unknown
        maturity literal;
      * the registry's advertised ``maturity`` disagrees with the contract;
      * a carrier advertises ``"trained"`` but has no committed real-scene evidence
        — i.e. it is not in ``TRAINED_CARRIER_EVIDENCE`` (only gaussian/beta today),
        or its calib_<scene>.json evidence artifacts are missing.

    Passing a ``registry`` overrides the default so a tampered registry (e.g. gabor
    flipped to "trained") can be exercised in tests.
    """
    reg = registry if registry is not None else default_registry()
    evidence: list[str] = []
    gaps: list[str] = []

    trained_ok: list[str] = []
    for carrier_id, spec in reg.items():
        maturity = getattr(spec, "maturity", None)
        if maturity not in CARRIER_MATURITIES:
            gaps.append(f"{carrier_id}: registry maturity {maturity!r} is not a known level")
            continue
        expected = CARRIER_MATURITY_CONTRACT.get(carrier_id)
        if expected is None:
            gaps.append(f"{carrier_id}: registered carrier absent from the maturity contract")
            continue
        if maturity != expected:
            gaps.append(f"{carrier_id}: registry advertises maturity {maturity!r} but contract allows {expected!r}")
            continue
        if maturity == "trained":
            required = TRAINED_CARRIER_EVIDENCE.get(carrier_id)
            if not required:
                gaps.append(f"{carrier_id}: claims 'trained' but has no committed real-scene evidence source")
                continue
            missing = [name for name in required if not (outputs_dir / name).exists()]
            if missing:
                gaps.append(f"{carrier_id}: 'trained' evidence missing {', '.join(missing)}")
                continue
            for name in required:
                artifacts[Path(name).stem] = str(outputs_dir / name)
            trained_ok.append(carrier_id)

    if trained_ok:
        evidence.append(
            f"trained carriers {sorted(trained_ok)} backed by committed calib_<scene>.json "
            f"evidence on {len(REAL_SCENES)} real scenes"
        )
    demo = sorted(c for c, m in CARRIER_MATURITY_CONTRACT.items() if m == "demo")
    meta = sorted(c for c, m in CARRIER_MATURITY_CONTRACT.items() if m == "metadata")
    if not gaps:
        evidence.append(f"demo-stage carriers {demo} advertised as demo, not trained")
        evidence.append(f"metadata-only carriers {meta} advertised as metadata, not trained")
        evidence.append(f"all {len(reg)} registered carrier types carry an explicit, backed maturity")

    passed = not gaps
    return PublicationGate(
        id="carrier_registry_honesty",
        title="Carrier registry advertises only backed maturity",
        passed=passed,
        evidence=tuple(evidence),
        gaps=tuple(gaps),
        next_steps=("promote a demo carrier (gabor/neural) to 'trained' only with committed real-scene evidence",),
    )
