from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


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
