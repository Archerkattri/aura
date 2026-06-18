from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

from aura.ray import Vec3

# Relighting colour triple (RGB in [0, 1])
_AlbedoRGB = Tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceCellPayload:
    """Typed payload for a surface carrier element.

    Encodes the surface normal, slab thickness, roughness, and an optional
    anchor point on the surface plane.
    """

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
    """Typed payload for a volumetric density carrier element."""

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
    """Typed payload for a Beta-distribution kernel carrier element.

    The kernel is parameterised by its shape parameters *alpha* and *beta* and
    per-axis support radii that define the ellipsoidal region of influence.

    DBS upgrade (arXiv:2501.18630, GES arXiv:2402.10128):
    - ``adaptive_alpha``: if set, overrides alpha (learnable per-carrier shape param)
    - ``adaptive_beta``: if set, overrides beta
    - ``frequency_scale``: multiplicative frequency modulation (1.0 = no change)
    - ``appearance_shift``: additive color/appearance bias term (0.0 = no change)
    All new fields default to values that reproduce exactly the prior behavior.
    Note: new fields are only included in to_dict() when non-default, preserving
    JSON schema compatibility (schemas use additionalProperties: false).
    """

    alpha: float
    beta: float
    support_radius: Vec3
    adaptive_alpha: Optional[float] = None
    adaptive_beta: Optional[float] = None
    frequency_scale: float = 1.0
    appearance_shift: float = 0.0

    def to_dict(self) -> dict:
        _positive("alpha", self.alpha)
        _positive("beta", self.beta)
        for item in self.support_radius:
            _positive("support_radius", item)
        payload: dict = {
            "type": "beta_kernel",
            "alpha": self.alpha,
            "beta": self.beta,
            "support_radius": list(self.support_radius),
        }
        # Only include DBS extension fields when non-default
        if self.adaptive_alpha is not None:
            payload["adaptive_alpha"] = self.adaptive_alpha
        if self.adaptive_beta is not None:
            payload["adaptive_beta"] = self.adaptive_beta
        if self.frequency_scale != 1.0:
            payload["frequency_scale"] = self.frequency_scale
        if self.appearance_shift != 0.0:
            payload["appearance_shift"] = self.appearance_shift
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BetaKernelPayload":
        _require_type(payload, "beta_kernel")
        return cls(
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            support_radius=_vec3(payload["support_radius"]),
            adaptive_alpha=float(payload["adaptive_alpha"]) if "adaptive_alpha" in payload else None,
            adaptive_beta=float(payload["adaptive_beta"]) if "adaptive_beta" in payload else None,
            frequency_scale=float(payload.get("frequency_scale", 1.0)),
            appearance_shift=float(payload.get("appearance_shift", 0.0)),
        )


@dataclass(frozen=True)
class GaborFrequencyPayload:
    """Typed payload for a Gabor-frequency carrier element.

    The carrier modulates element color with a sinusoidal Gabor function
    parameterised by spatial *frequency*, *bandwidth*, and *phase*.

    Gabor Filter Bank upgrade (arXiv:2508.05343, 3DGabSplat arXiv:2504.11003):
    - ``num_filters``: number of directional Gabor kernels (1 = current behavior)
    - ``frequencies``: per-filter frequencies; if None, uses existing frequency for all
    - ``orientations``: per-filter orientation angles in radians; if None, uses 0.0 for all
    - ``phases``: per-filter phases; if None, uses existing phase for all
    - ``filter_weights``: mixing weights; if None, equal weights (1/num_filters)
    All new fields default to values that reproduce exactly the prior behavior.
    Note: new fields are only included in to_dict() when non-default, preserving
    JSON schema compatibility (schemas use additionalProperties: false).
    """

    frequency: Vec3
    bandwidth: float
    phase: float = 0.0
    plane_point: Vec3 | None = None
    num_filters: int = 1
    frequencies: Optional[List[float]] = None
    orientations: Optional[List[float]] = None
    phases: Optional[List[float]] = None
    filter_weights: Optional[List[float]] = None

    def to_dict(self) -> dict:
        _positive("bandwidth", self.bandwidth)
        payload: dict = {
            "type": "gabor_frequency",
            "frequency": list(self.frequency),
            "bandwidth": self.bandwidth,
            "phase": self.phase,
        }
        if self.plane_point is not None:
            payload["plane_point"] = list(self.plane_point)
        # Only include filter bank fields when non-default
        if self.num_filters != 1:
            payload["num_filters"] = self.num_filters
        if self.frequencies is not None:
            payload["frequencies"] = list(self.frequencies)
        if self.orientations is not None:
            payload["orientations"] = list(self.orientations)
        if self.phases is not None:
            payload["phases"] = list(self.phases)
        if self.filter_weights is not None:
            payload["filter_weights"] = list(self.filter_weights)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GaborFrequencyPayload":
        _require_type(payload, "gabor_frequency")
        plane_point = _vec3(payload["plane_point"]) if "plane_point" in payload else None
        frequencies = list(payload["frequencies"]) if "frequencies" in payload else None
        orientations = list(payload["orientations"]) if "orientations" in payload else None
        phases = list(payload["phases"]) if "phases" in payload else None
        filter_weights = list(payload["filter_weights"]) if "filter_weights" in payload else None
        return cls(
            frequency=_vec3(payload["frequency"]),
            bandwidth=float(payload["bandwidth"]),
            phase=float(payload["phase"]),
            plane_point=plane_point,
            num_filters=int(payload.get("num_filters", 1)),
            frequencies=frequencies,
            orientations=orientations,
            phases=phases,
            filter_weights=filter_weights,
        )


@dataclass(frozen=True)
class NeuralResidualPayload:
    """Typed payload for a neural residual carrier element.

    Stores the latent dimension and residual scale used by the neural
    renderer path. ``model_ref`` optionally references an external model
    checkpoint for production inference.

    Scaffold-GS upgrade (arXiv:2312.00109):
    - ``anchor_feature_dim``: dimension of anchor latent features; if None, uses latent_dim
    - ``mlp_hidden_dim``: hidden dim for the small decode MLP (default 64)
    - ``num_mlp_layers``: number of MLP layers (default 2)
    - ``use_anchor_conditioning``: condition on neighboring anchor features (default False)
    All new fields default to values that reproduce exactly the prior behavior.
    Note: new fields are only included in to_dict() when non-default, preserving
    JSON schema compatibility (schemas use additionalProperties: false).
    """

    latent_dim: int
    residual_scale: float
    model_ref: str | None = None
    anchor_feature_dim: Optional[int] = None
    mlp_hidden_dim: int = 64
    num_mlp_layers: int = 2
    use_anchor_conditioning: bool = False

    def to_dict(self) -> dict:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        _unit("residual_scale", self.residual_scale)
        payload: dict = {
            "type": "neural_residual",
            "latent_dim": self.latent_dim,
            "residual_scale": self.residual_scale,
            "model_ref": self.model_ref,
        }
        # Only include Scaffold-GS extension fields when non-default
        if self.anchor_feature_dim is not None:
            payload["anchor_feature_dim"] = self.anchor_feature_dim
        if self.mlp_hidden_dim != 64:
            payload["mlp_hidden_dim"] = self.mlp_hidden_dim
        if self.num_mlp_layers != 2:
            payload["num_mlp_layers"] = self.num_mlp_layers
        if self.use_anchor_conditioning:
            payload["use_anchor_conditioning"] = self.use_anchor_conditioning
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NeuralResidualPayload":
        _require_type(payload, "neural_residual")
        model_ref = payload.get("model_ref")
        anchor_feature_dim = payload.get("anchor_feature_dim")
        return cls(
            latent_dim=int(payload["latent_dim"]),
            residual_scale=float(payload["residual_scale"]),
            model_ref=None if model_ref is None else str(model_ref),
            anchor_feature_dim=int(anchor_feature_dim) if anchor_feature_dim is not None else None,
            mlp_hidden_dim=int(payload.get("mlp_hidden_dim", 64)),
            num_mlp_layers=int(payload.get("num_mlp_layers", 2)),
            use_anchor_conditioning=bool(payload.get("use_anchor_conditioning", False)),
        )


@dataclass(frozen=True)
class GaussianFallbackPayload:
    """Typed payload for a Gaussian-fallback carrier element.

    Provides a full 3-D Gaussian parameterised by its *mean* and a 3x3
    *covariance* matrix. Used as a compatibility path for conventional
    3-D Gaussian Splatting primitives.
    """

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
    """Typed payload for a semantic-feature carrier element.

    Associates the element with a human-readable *label* and an optional set
    of feature references for language or object embedding look-ups.

    LangSplatV2 upgrade (arXiv:2507.07136):
    - ``use_sparse_codebook``: if False (default), existing dense path unchanged
    - ``codebook_size``: number of dictionary atoms (default 256)
    - ``codebook_dim``: dimension of each atom (default 64)
    - ``sparse_indices``: indices of active atoms; if None, dense path
    - ``sparse_weights``: weights for each active atom; if None, dense path
    All new fields default to values that reproduce exactly the prior behavior.
    Note: new fields are only included in to_dict() when non-default, preserving
    JSON schema compatibility (schemas use additionalProperties: false).
    """

    label: str
    confidence: float
    feature_refs: tuple[str, ...] = field(default_factory=tuple)
    use_sparse_codebook: bool = False
    codebook_size: int = 256
    codebook_dim: int = 64
    sparse_indices: Optional[List[int]] = None
    sparse_weights: Optional[List[float]] = None

    def to_dict(self) -> dict:
        if not self.label:
            raise ValueError("label is required")
        _unit("confidence", self.confidence)
        payload: dict = {
            "type": "semantic_feature",
            "label": self.label,
            "confidence": self.confidence,
            "feature_refs": list(self.feature_refs),
        }
        # Only include sparse codebook fields when non-default
        if self.use_sparse_codebook:
            payload["use_sparse_codebook"] = self.use_sparse_codebook
        if self.codebook_size != 256:
            payload["codebook_size"] = self.codebook_size
        if self.codebook_dim != 64:
            payload["codebook_dim"] = self.codebook_dim
        if self.sparse_indices is not None:
            payload["sparse_indices"] = list(self.sparse_indices)
        if self.sparse_weights is not None:
            payload["sparse_weights"] = list(self.sparse_weights)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticFeaturePayload":
        _require_type(payload, "semantic_feature")
        sparse_indices = [int(i) for i in payload["sparse_indices"]] if "sparse_indices" in payload else None
        sparse_weights = [float(w) for w in payload["sparse_weights"]] if "sparse_weights" in payload else None
        return cls(
            label=str(payload["label"]),
            confidence=float(payload["confidence"]),
            feature_refs=tuple(str(item) for item in payload.get("feature_refs", ())),
            use_sparse_codebook=bool(payload.get("use_sparse_codebook", False)),
            codebook_size=int(payload.get("codebook_size", 256)),
            codebook_dim=int(payload.get("codebook_dim", 64)),
            sparse_indices=sparse_indices,
            sparse_weights=sparse_weights,
        )


@dataclass(frozen=True)
class RelightingPayload:
    """Per-carrier PBR material parameters for the relighting shading pipeline.

    This payload is **not** serialised to JSON (schema files are not owned and
    use ``additionalProperties: false``).  Attach it to a scene element's
    payload at runtime or training time; the shading module reads
    ``payload.get("albedo")``, ``payload.get("shading_roughness")``, and
    ``payload.get("shading_metallic")``.

    Relighting fields (GS-IR arXiv:2311.16473):
    - ``albedo``: physical albedo RGB in [0, 1]³ (default: reproduce emissive)
    - ``shading_roughness``: PBR roughness in [0, 1] (default 0.5)
    - ``shading_metallic``: metallic factor in [0, 1] (default 0.0 = dielectric)

    All fields default to values that leave the emissive output unchanged when
    shading is OFF (the renderer's default).  The ``albedo`` field defaults to
    ``None``, meaning the shading module falls back to the element's emissive
    color, thus preserving bit-identical output with shading disabled.

    Note: new fields are only included in to_dict() when non-default so that
    existing JSON schemas (which use additionalProperties: false) stay valid.
    Because full payload type keys are owned by existing payloads this
    dataclass serialises as ``type: "relighting"`` for any *new* standalone
    usage, but the fields can also be embedded into existing payload dicts at
    runtime (the shading module reads them by key, not by type).
    """

    albedo: Optional[_AlbedoRGB] = None          # None → use emissive color
    shading_roughness: float = 0.5
    shading_metallic: float = 0.0

    def to_dict(self) -> dict:
        for name, value in (("shading_roughness", self.shading_roughness), ("shading_metallic", self.shading_metallic)):
            _unit(name, value)
        payload: dict = {"type": "relighting"}
        # Only emit non-default fields so additionalProperties schemas stay valid
        if self.albedo is not None:
            if not (isinstance(self.albedo, (list, tuple)) and len(self.albedo) == 3):
                raise ValueError("albedo must be an RGB triple")
            payload["albedo"] = [float(self.albedo[0]), float(self.albedo[1]), float(self.albedo[2])]
        if self.shading_roughness != 0.5:
            payload["shading_roughness"] = self.shading_roughness
        if self.shading_metallic != 0.0:
            payload["shading_metallic"] = self.shading_metallic
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RelightingPayload":
        _require_type(payload, "relighting")
        albedo = _vec3(payload["albedo"]) if "albedo" in payload else None
        return cls(
            albedo=albedo,
            shading_roughness=float(payload.get("shading_roughness", 0.5)),
            shading_metallic=float(payload.get("shading_metallic", 0.0)),
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
