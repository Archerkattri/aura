from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SemanticNode:
    """One labelled node in the scene semantic graph.

    A node groups zero or more :class:`~aura.elements.AuraElement` ids under a
    human-readable label and an optional confidence score.
    """

    id: str
    label: str
    element_ids: tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("semantic node id is required")
        if not self.label:
            raise ValueError("semantic node label is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("semantic node confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["element_ids"] = list(self.element_ids)
        payload["attributes"] = dict(self.attributes)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticNode":
        return cls(
            id=str(payload["id"]),
            label=str(payload["label"]),
            element_ids=tuple(str(item) for item in payload.get("element_ids", ())),
            confidence=float(payload.get("confidence", 1.0)),
            attributes=dict(payload.get("attributes", {})),
        )


@dataclass(frozen=True)
class SemanticEdge:
    """Directed relation between two :class:`SemanticNode` ids."""

    source: str
    target: str
    relation: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("semantic edge source and target are required")
        if not self.relation:
            raise ValueError("semantic edge relation is required")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("semantic edge confidence must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticEdge":
        return cls(
            source=str(payload["source"]),
            target=str(payload["target"]),
            relation=str(payload["relation"]),
            confidence=float(payload.get("confidence", 1.0)),
        )


@dataclass(frozen=True)
class SemanticGraph:
    """Immutable graph of semantic nodes and directed relations.

    Node ids must be unique within the graph; edge endpoints must reference
    existing node ids.
    """

    nodes: Sequence[SemanticNode] = field(default_factory=tuple)
    edges: Sequence[SemanticEdge] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        node_ids = {node.id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("semantic graph contains duplicate node ids")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"semantic edge references unknown node: {edge.source}->{edge.target}")

    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticGraph":
        return cls(
            nodes=tuple(SemanticNode.from_dict(item) for item in payload.get("nodes", ())),
            edges=tuple(SemanticEdge.from_dict(item) for item in payload.get("edges", ())),
        )


def decode_semantic_feature(
    payload: Mapping[str, Any],
    codebook: Optional[List[List[float]]] = None,
) -> Optional[List[float]]:
    """Decode a semantic feature vector from a SemanticFeaturePayload dict.

    LangSplatV2 sparse codebook decode (arXiv:2507.07136):
    - When ``use_sparse_codebook=False`` (default): returns None (dense path, no decode)
    - When ``use_sparse_codebook=True`` and codebook is provided: reconstructs feature
      vector as a weighted sum of active codebook atoms indexed by ``sparse_indices``
      with ``sparse_weights``.

    Args:
        payload: A dict from SemanticFeaturePayload.to_dict().
        codebook: Optional list of atom vectors (shape: codebook_size x codebook_dim).
                  Required when use_sparse_codebook=True.

    Returns:
        Reconstructed feature vector as a list of floats, or None for dense path.
    """
    use_sparse = bool(payload.get("use_sparse_codebook", False))
    if not use_sparse:
        # Dense path: existing behavior unchanged
        return None

    sparse_indices: Optional[List[int]] = payload.get("sparse_indices")
    sparse_weights: Optional[List[float]] = payload.get("sparse_weights")
    codebook_dim = int(payload.get("codebook_dim", 64))

    if sparse_indices is None or sparse_weights is None:
        # No active atoms: return zero vector
        return [0.0] * codebook_dim

    if codebook is None:
        # No codebook provided: return zero vector of codebook_dim
        return [0.0] * codebook_dim

    if len(sparse_indices) != len(sparse_weights):
        raise ValueError(
            f"sparse_indices length ({len(sparse_indices)}) must match "
            f"sparse_weights length ({len(sparse_weights)})"
        )

    # Reconstruct feature vector: sum of weight_i * atom_i
    result = [0.0] * codebook_dim
    for idx, weight in zip(sparse_indices, sparse_weights):
        atom = codebook[idx]
        for dim_i in range(codebook_dim):
            result[dim_i] += weight * atom[dim_i]
    return result
