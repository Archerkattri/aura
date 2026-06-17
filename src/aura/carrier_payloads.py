from __future__ import annotations

from dataclasses import asdict, dataclass, field

from aura.ray import Vec3


@dataclass(frozen=True)
class SurfaceCellPayload:
    normal: Vec3
    thickness: float
    roughness: float = 0.5

    def to_dict(self) -> dict:
        _positive("thickness", self.thickness)
        _unit("roughness", self.roughness)
        return {"type": "surface_cell", **asdict(self)}


@dataclass(frozen=True)
class VolumeCellPayload:
    density: float
    phase_anisotropy: float = 0.0

    def to_dict(self) -> dict:
        _unit("density", self.density)
        if not -1.0 <= float(self.phase_anisotropy) <= 1.0:
            raise ValueError("phase_anisotropy must be in [-1, 1]")
        return {"type": "volume_cell", **asdict(self)}


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


@dataclass(frozen=True)
class GaborFrequencyPayload:
    frequency: Vec3
    bandwidth: float
    phase: float = 0.0

    def to_dict(self) -> dict:
        _positive("bandwidth", self.bandwidth)
        return {"type": "gabor_frequency", **asdict(self)}


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


def _unit(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _positive(name: str, value: float) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{name} must be positive")
