from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any, Mapping, Sequence

from aura.assignment import RegionEvidence
from aura.core import TrainingFrame, TrainingRegion
from aura.elements import Bounds


@dataclass(frozen=True)
class CaptureProposalFeatures:
    frame_id: str
    image_detail: float
    depth_edge: float
    mask_coverage: float
    normal_present: bool
    depth: float

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "imageDetail": self.image_detail,
            "depthEdge": self.depth_edge,
            "maskCoverage": self.mask_coverage,
            "normalPresent": self.normal_present,
            "depth": self.depth,
        }


@dataclass(frozen=True)
class CaptureProposalScore:
    frame_id: str
    proposal_type: str
    score: float
    features: CaptureProposalFeatures
    threshold: float
    model_id: str

    @property
    def accepted(self) -> bool:
        return self.score >= self.threshold

    def to_dict(self) -> dict:
        return {
            "frameId": self.frame_id,
            "proposalType": self.proposal_type,
            "score": self.score,
            "threshold": self.threshold,
            "accepted": self.accepted,
            "modelId": self.model_id,
            "features": self.features.to_dict(),
        }


@dataclass(frozen=True)
class CaptureProposalModel:
    """Model-scored native region proposal contract for capture tensors.

    The default weights are deterministic reference weights. Production can
    replace them with learned weights or a neural proposal network while keeping
    the same feature/score/TrainingRegion contract.
    """

    id: str = "aura-reference-capture-proposal-v1"
    image_detail_weights: Mapping[str, float] | None = None
    depth_edge_weights: Mapping[str, float] | None = None
    threshold: float = 0.55

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("proposal threshold must be in [0, 1]")

    def score(self, features: CaptureProposalFeatures) -> tuple[CaptureProposalScore, CaptureProposalScore]:
        image_weights = self.image_detail_weights or {
            "bias": -2.2,
            "image_detail": 4.8,
            "depth_edge": 0.6,
            "mask_coverage": 0.2,
            "normal_present": 0.15,
        }
        depth_weights = self.depth_edge_weights or {
            "bias": -2.0,
            "image_detail": 0.7,
            "depth_edge": 5.0,
            "mask_coverage": 0.15,
            "normal_present": 0.2,
        }
        return (
            CaptureProposalScore(
                frame_id=features.frame_id,
                proposal_type="image_detail",
                score=_score_logistic(image_weights, features),
                features=features,
                threshold=self.threshold,
                model_id=self.id,
            ),
            CaptureProposalScore(
                frame_id=features.frame_id,
                proposal_type="depth_edge",
                score=_score_logistic(depth_weights, features),
                features=features,
                threshold=self.threshold,
                model_id=self.id,
            ),
        )


def default_capture_proposal_model() -> CaptureProposalModel:
    return CaptureProposalModel()


def score_capture_proposals(
    frames: Sequence[TrainingFrame],
    tensors: Mapping[str, Any],
    assets: Mapping[str, Any],
    *,
    model: CaptureProposalModel | None = None,
) -> tuple[CaptureProposalScore, ...]:
    model = model or default_capture_proposal_model()
    scores: list[CaptureProposalScore] = []
    for frame in frames:
        frame_tensors = tensors.get(frame.id)
        if frame_tensors is None:
            continue
        features = capture_proposal_features(frame, frame_tensors, assets.get(frame.id))
        scores.extend(model.score(features))
    return tuple(scores)


def propose_training_regions_from_tensors(
    frames: Sequence[TrainingFrame],
    tensors: Mapping[str, Any],
    assets: Mapping[str, Any],
    *,
    model: CaptureProposalModel | None = None,
) -> tuple[TrainingRegion, ...]:
    model = model or default_capture_proposal_model()
    regions: list[TrainingRegion] = []
    by_frame = {frame.id: frame for frame in frames}
    for frame in frames:
        frame_tensors = tensors.get(frame.id)
        if frame_tensors is None:
            continue
        asset = assets.get(frame.id)
        features = capture_proposal_features(frame, frame_tensors, asset)
        for score in model.score(features):
            if not score.accepted:
                continue
            if score.proposal_type == "image_detail":
                regions.append(_image_detail_region(by_frame[score.frame_id], frame_tensors, asset, score))
            elif score.proposal_type == "depth_edge":
                regions.append(_depth_edge_region(by_frame[score.frame_id], frame_tensors, asset, score))
    return tuple(regions)


def capture_proposal_features(frame: TrainingFrame, tensors: Any, asset: Any | None) -> CaptureProposalFeatures:
    depth = asset.average_depth if asset is not None and asset.average_depth is not None else frame.target_depth
    return CaptureProposalFeatures(
        frame_id=frame.id,
        image_detail=_image_detail_score(tensors.image),
        depth_edge=_depth_edge_score(tensors.depth),
        mask_coverage=0.0 if asset is None or asset.mask_coverage is None else asset.mask_coverage,
        normal_present=asset is not None and asset.average_normal is not None,
        depth=depth,
    )


def _image_detail_region(
    frame: TrainingFrame,
    tensors: Any,
    asset: Any | None,
    score: CaptureProposalScore,
) -> TrainingRegion:
    detail = score.features.image_detail
    depth = score.features.depth
    half_width = _depth_region_half_extent(frame, depth) * 0.5
    confidence = min(1.0, 0.45 + 0.55 * score.score)
    return TrainingRegion(
        id=f"{frame.id}_image_detail_proposal",
        frame_id=frame.id,
        bounds=Bounds(
            min_corner=(-half_width, -half_width, max(depth - max(depth * 0.03, 1e-3), 1e-6)),
            max_corner=(half_width, half_width, depth + max(depth * 0.03, 1e-3)),
        ),
        evidence=RegionEvidence(
            high_frequency=min(1.0, max(0.8, detail)),
            image_error=min(1.0, detail),
            material_confidence=0.45,
            ray_need=0.65,
            compact_detail=0.45,
        ),
        color=_average_tensor_rgb(tensors.image),
        opacity=0.7,
        confidence=confidence,
        normal=asset.average_normal if asset is not None else None,
        material_id="mat_image_detail_proposal",
        fallback_source="capture-feature-proposal",
    )


def _depth_edge_region(
    frame: TrainingFrame,
    tensors: Any,
    asset: Any | None,
    score: CaptureProposalScore,
) -> TrainingRegion:
    edge_score = score.features.depth_edge
    depth = score.features.depth
    half_width = _depth_region_half_extent(frame, depth) * 0.4
    thickness = max(depth * 0.02, edge_score * depth * 0.05, 1e-3)
    confidence = min(1.0, 0.45 + 0.55 * score.score)
    return TrainingRegion(
        id=f"{frame.id}_depth_edge_proposal",
        frame_id=frame.id,
        bounds=Bounds(
            min_corner=(-half_width, -half_width, max(depth - thickness, 1e-6)),
            max_corner=(half_width, half_width, depth + thickness),
        ),
        evidence=RegionEvidence(
            compact_detail=min(1.0, max(0.8, edge_score)),
            geometry_confidence=min(0.9, 0.45 + 0.45 * edge_score),
            image_error=min(1.0, edge_score),
            ray_need=0.7,
        ),
        color=_average_tensor_rgb(tensors.image),
        opacity=0.75,
        confidence=confidence,
        normal=asset.average_normal if asset is not None else None,
        material_id="mat_depth_edge_proposal",
        fallback_source="capture-feature-proposal",
    )


def _score_logistic(weights: Mapping[str, float], features: CaptureProposalFeatures) -> float:
    value = float(weights.get("bias", 0.0))
    value += float(weights.get("image_detail", 0.0)) * features.image_detail
    value += float(weights.get("depth_edge", 0.0)) * features.depth_edge
    value += float(weights.get("mask_coverage", 0.0)) * features.mask_coverage
    value += float(weights.get("normal_present", 0.0)) * (1.0 if features.normal_present else 0.0)
    return _clamp_unit(1.0 / (1.0 + exp(-value)))


def _average_tensor_rgb(image: Any) -> tuple[float, float, float]:
    if image.channels < 3:
        raise ValueError("average RGB requires at least a 3-channel image")
    totals = [0.0, 0.0, 0.0]
    for pixel_start in range(0, len(image.values), image.channels):
        for channel in range(3):
            totals[channel] += image.values[pixel_start + channel]
    pixels = image.width * image.height
    return (totals[0] / pixels, totals[1] / pixels, totals[2] / pixels)


def _image_detail_score(image: Any) -> float:
    if image.channels < 3 or image.width * image.height < 2:
        return 0.0
    scores = []
    for y in range(image.height):
        for x in range(image.width):
            if x + 1 < image.width:
                scores.append(_rgb_distance(image, x, y, x + 1, y))
            if y + 1 < image.height:
                scores.append(_rgb_distance(image, x, y, x, y + 1))
    return 0.0 if not scores else sum(scores) / len(scores)


def _rgb_distance(image: Any, ax: int, ay: int, bx: int, by: int) -> float:
    first = (ay * image.width + ax) * image.channels
    second = (by * image.width + bx) * image.channels
    return sum(abs(image.values[first + channel] - image.values[second + channel]) for channel in range(3)) / 3.0


def _depth_edge_score(depth: Any | None) -> float:
    if depth is None or depth.channels != 1 or depth.width * depth.height < 2:
        return 0.0
    scores = []
    for y in range(depth.height):
        for x in range(depth.width):
            value = depth.values[y * depth.width + x]
            if value <= 0.0:
                continue
            if x + 1 < depth.width:
                neighbor = depth.values[y * depth.width + x + 1]
                if neighbor > 0.0:
                    scores.append(abs(value - neighbor))
            if y + 1 < depth.height:
                neighbor = depth.values[(y + 1) * depth.width + x]
                if neighbor > 0.0:
                    scores.append(abs(value - neighbor))
    return 0.0 if not scores else min(1.0, sum(scores) / len(scores))


def _depth_region_half_extent(frame: TrainingFrame, depth: float) -> float:
    if frame.intrinsics is None:
        return max(0.05, depth * 0.05)
    width = frame.intrinsics.get("width", 1.0)
    height = frame.intrinsics.get("height", 1.0)
    fx = max(frame.intrinsics.get("fx", 1.0), 1e-6)
    fy = max(frame.intrinsics.get("fy", 1.0), 1e-6)
    half_x = depth * width / (2.0 * fx)
    half_y = depth * height / (2.0 * fy)
    return max(0.05, min(half_x, half_y))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
