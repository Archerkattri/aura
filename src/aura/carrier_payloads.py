from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from aura.ray import Vec3


@dataclass(frozen=True)
class SurfaceCellPayload:
    normal: Vec3
    thickness: float
    roughness: float = 0.5
    plane_point: Vec3 | None = None

    def to_dict(self) -> dict:
        _positive("thickness", self.thickness)
        _unit("roughness", self.roughness)
        payload = {"type": "surface_cell", **asdict(self)}
        if self.plane_point is None:
            payload.pop("plane_point")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SurfaceCellPayload":
        _require_type(payload, "surface_cell")
        plane_point = _vec3(payload["plane_point"]) if "plane_point" in payload else None
        return cls(
            normal=_vec3(payload["normal"]),
            thickness=float(payload["thickness"]),
            roughness=float(payload["roughness"]),
            plane_point=plane_point,
        )


@dataclass(frozen=True)
class VolumeCellPayload:
    density: float
    phase_anisotropy: float = 0.0

    def to_dict(self) -> dict:
        _unit("density", self.density)
        if not -1.0 <= float(self.phase_anisotropy) <= 1.0:
            raise ValueError("phase_anisotropy must be in [-1, 1]")
        return {"type": "volume_cell", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VolumeCellPayload":
        _require_type(payload, "volume_cell")
        return cls(density=float(payload["density"]), phase_anisotropy=float(payload["phase_anisotropy"]))


@dataclass(frozen=True)
class BetaKernelPayload:
    alpha: float
    beta: float
    support_radius: Vec3

    def to_dict(self) -> dict:
        _positive("alpha", self.alpha)
        _positive("beta", self.beta)
        for item in self.support_radius:
            _positive("support_radius", item)
        return {"type": "beta_kernel", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BetaKernelPayload":
        _require_type(payload, "beta_kernel")
        return cls(alpha=float(payload["alpha"]), beta=float(payload["beta"]), support_radius=_vec3(payload["support_radius"]))


@dataclass(frozen=True)
class GaborFrequencyPayload:
    frequency: Vec3
    bandwidth: float
    phase: float = 0.0
    plane_point: Vec3 | None = None

    def to_dict(self) -> dict:
        _positive("bandwidth", self.bandwidth)
        payload = {"type": "gabor_frequency", **asdict(self)}
        if self.plane_point is None:
            payload.pop("plane_point")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GaborFrequencyPayload":
        _require_type(payload, "gabor_frequency")
        plane_point = _vec3(payload["plane_point"]) if "plane_point" in payload else None
        return cls(
            frequency=_vec3(payload["frequency"]),
            bandwidth=float(payload["bandwidth"]),
            phase=float(payload["phase"]),
            plane_point=plane_point,
        )


@dataclass(frozen=True)
class NeuralResidualPayload:
    latent_dim: int
    residual_scale: float
    model_ref: str | None = None

    def to_dict(self) -> dict:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        _unit("residual_scale", self.residual_scale)
        return {"type": "neural_residual", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuralResidualPayload":
        _require_type(payload, "neural_residual")
        model_ref = payload.get("model_ref")
        return cls(latent_dim=int(payload["latent_dim"]), residual_scale=float(payload["residual_scale"]), model_ref=None if model_ref is None else str(model_ref))


@dataclass(frozen=True)
class GaussianFallbackPayload:
    mean: Vec3
    covariance: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    source: str = "ingest"

    def to_dict(self) -> dict:
        if len(self.covariance) != 3 or any(len(row) != 3 for row in self.covariance):
            raise ValueError("covariance must be a 3x3 matrix")
        if any(self.covariance[index][index] <= 0.0 for index in range(3)):
            raise ValueError("covariance diagonal entries must be positive")
        return {
            "type": "gaussian_fallback",
            "mean": list(self.mean),
            "covariance": [list(row) for row in self.covariance],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GaussianFallbackPayload":
        _require_type(payload, "gaussian_fallback")
        covariance = tuple(tuple(float(item) for item in row) for row in payload["covariance"])
        return cls(mean=_vec3(payload["mean"]), covariance=covariance, source=str(payload["source"]))  # type: ignore[arg-type]


@dataclass(frozen=True)
class SemanticFeaturePayload:
    label: str
    confidence: float
    feature_refs: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        if not self.label:
            raise ValueError("label is required")
        _unit("confidence", self.confidence)
        return {"type": "semantic_feature", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticFeaturePayload":
        _require_type(payload, "semantic_feature")
        return cls(
            label=str(payload["label"]),
            confidence=float(payload["confidence"]),
            feature_refs=tuple(str(item) for item in payload.get("feature_refs", ())),
        )


def _unit(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_type(payload: Mapping[str, Any], expected: str) -> None:
    actual = payload.get("type")
    if actual != expected:
        raise ValueError(f"payload type must be {expected!r}, got {actual!r}")


def _vec3(value: Any) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("vec3 payload fields must have exactly three values")
    return (float(value[0]), float(value[1]), float(value[2]))
