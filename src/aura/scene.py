from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable, Sequence

from aura.elements import AuraChunk, AuraElement, Bounds
from aura.ray import Ray, RayQueryResult, Vec3
from aura.semantic import SemanticGraph


BVH_CHUNK_THRESHOLD = 3


@dataclass(frozen=True)
class AuraScene:
    """Reference scene for the AURA ray-query contract."""

    name: str
    elements: Sequence[AuraElement]
    chunks: Sequence[AuraChunk] = field(default_factory=tuple)
    semantic_graph: SemanticGraph = field(default_factory=SemanticGraph)

    def ray_query(self, ray: Ray) -> RayQueryResult:
        return self.traverse_ray(ray).result

    def traverse_ray(self, ray: Ray) -> "RayTraversal":
        element_by_id = {element.id: element for element in self.elements}
        chunks = tuple(self.chunks)
        candidate_chunks, traversal_mode, tested_bvh_node_count = _candidate_chunks(ray, chunks, self._chunk_bvh)
        if chunks:
            chunked_ids = {element_id for chunk in chunks for element_id in chunk.element_ids}
            candidate_ids = []
            for chunk in candidate_chunks:
                candidate_ids.extend(chunk.element_ids)
            chunk_candidates = tuple(element_by_id[element_id] for element_id in candidate_ids if element_id in element_by_id)
            orphan_candidates = tuple(element for element in self.elements if element.id not in chunked_ids)
            candidates = (*chunk_candidates, *orphan_candidates)
        else:
            candidates = tuple(self.elements)
        hits = tuple(
            RayHitTrace(element_id=element.id, carrier_id=element.carrier_id, result=hit)
            for element in candidates
            if (hit := element.ray_query(ray)) is not None
        )
        if not hits:
            result = RayQueryResult(color=(0.0, 0.0, 0.0), transmittance=1.0, confidence=0.0, provenance="miss")
            return RayTraversal(
                result=result,
                ordered_hits=(),
                tested_chunk_ids=tuple(chunk.id for chunk in candidate_chunks),
                tested_element_ids=tuple(element.id for element in candidates),
                total_element_count=len(self.elements),
                hit_count=0,
                traversal_mode=traversal_mode,
                tested_bvh_node_count=tested_bvh_node_count,
            )
        ordered_hits = tuple(sorted(hits, key=lambda item: item.depth if item.depth is not None else float("inf")))
        return RayTraversal(
            result=composite_front_to_back(hit.result for hit in ordered_hits),
            ordered_hits=ordered_hits,
            tested_chunk_ids=tuple(chunk.id for chunk in candidate_chunks),
            tested_element_ids=tuple(element.id for element in candidates),
            total_element_count=len(self.elements),
            hit_count=len(hits),
            traversal_mode=traversal_mode,
            tested_bvh_node_count=tested_bvh_node_count,
        )

    def carrier_ids(self) -> list[str]:
        return sorted({element.carrier_id for element in self.elements})

    def chunk_ids(self) -> list[str]:
        return sorted({element.chunk_id for element in self.elements})

    @cached_property
    def _chunk_bvh(self) -> "_BvhNode | None":
        chunks = tuple(self.chunks)
        if len(chunks) < BVH_CHUNK_THRESHOLD:
            return None
        return _build_bvh(chunks)


@dataclass(frozen=True)
class RayHitTrace:
    """One carrier contribution in front-to-back ray order."""

    element_id: str
    carrier_id: str
    result: RayQueryResult

    @property
    def depth(self) -> float | None:
        return self.result.depth

    def to_dict(self) -> dict:
        return {
            "elementId": self.element_id,
            "carrierId": self.carrier_id,
            "depth": self.result.depth,
            "color": list(self.result.color),
            "transmittance": self.result.transmittance,
            "opacity": self.result.opacity,
            "confidence": self.result.confidence,
            "normal": list(self.result.normal) if self.result.normal is not None else None,
            "materialId": self.result.material_id,
            "semanticId": self.result.semantic_id,
            "residual": self.result.residual,
            "provenance": self.result.provenance,
        }


@dataclass(frozen=True)
class RayTraversal:
    result: RayQueryResult
    ordered_hits: tuple[RayHitTrace, ...]
    tested_chunk_ids: tuple[str, ...]
    tested_element_ids: tuple[str, ...]
    total_element_count: int
    hit_count: int
    traversal_mode: str = "linear"
    tested_bvh_node_count: int = 0

    @property
    def skipped_element_count(self) -> int:
        return max(0, self.total_element_count - len(set(self.tested_element_ids)))

    def to_dict(self) -> dict:
        return {
            "result": {
                "color": list(self.result.color),
                "transmittance": self.result.transmittance,
                "opacity": self.result.opacity,
                "confidence": self.result.confidence,
                "depth": self.result.depth,
                "normal": list(self.result.normal) if self.result.normal is not None else None,
                "materialId": self.result.material_id,
                "semanticId": self.result.semantic_id,
                "residual": self.result.residual,
                "provenance": self.result.provenance,
            },
            "orderedHits": [hit.to_dict() for hit in self.ordered_hits],
            "compositing": {
                "mode": "front_to_back",
                "orderedHitCount": len(self.ordered_hits),
                "provenanceOrder": [hit.result.provenance for hit in self.ordered_hits],
            },
            "testedChunkIds": list(self.tested_chunk_ids),
            "testedElementIds": list(self.tested_element_ids),
            "totalElementCount": self.total_element_count,
            "testedElementCount": len(self.tested_element_ids),
            "skippedElementCount": self.skipped_element_count,
            "hitCount": self.hit_count,
            "traversalMode": self.traversal_mode,
            "testedBvhNodeCount": self.tested_bvh_node_count,
        }


def composite_front_to_back(hits: Iterable[RayQueryResult]) -> RayQueryResult:
    color: Vec3 = (0.0, 0.0, 0.0)
    transmittance = 1.0
    confidence_num = 0.0
    confidence_den = 0.0
    first = None
    provenance: list[str] = []
    residual = False
    for hit in hits:
        if first is None:
            first = hit
        alpha = 1.0 - hit.transmittance
        weight = transmittance * alpha
        color = (
            color[0] + weight * hit.color[0],
            color[1] + weight * hit.color[1],
            color[2] + weight * hit.color[2],
        )
        confidence_num += weight * hit.confidence
        confidence_den += weight
        transmittance *= hit.transmittance
        provenance.append(hit.provenance or "unknown")
        residual = residual or hit.residual
    confidence = 0.0 if confidence_den == 0.0 else confidence_num / confidence_den
    return RayQueryResult(
        color=color,
        transmittance=transmittance,
        confidence=confidence,
        depth=first.depth if first else None,
        normal=first.normal if first else None,
        material_id=first.material_id if first else None,
        semantic_id=first.semantic_id if first else None,
        residual=residual,
        provenance=",".join(provenance),
    )


@dataclass(frozen=True)
class _BvhNode:
    bounds: Bounds
    chunks: tuple[AuraChunk, ...]
    left: "_BvhNode | None" = None
    right: "_BvhNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def _candidate_chunks(
    ray: Ray,
    chunks: Sequence[AuraChunk],
    bvh_root: "_BvhNode | None" = None,
) -> tuple[tuple[AuraChunk, ...], str, int]:
    if len(chunks) >= BVH_CHUNK_THRESHOLD:
        root = bvh_root if bvh_root is not None else _build_bvh(tuple(chunks))
        candidates, tested_nodes = _candidate_chunks_bvh(ray, root)
        return candidates, "bvh", tested_nodes
    candidates = []
    for chunk in chunks:
        hit = chunk.bounds.intersect_ray(ray)
        if hit is not None:
            candidates.append((hit[0], chunk))
    candidates.sort(key=lambda item: item[0])
    return tuple(chunk for _depth, chunk in candidates), "chunk_linear", 0


def _candidate_chunks_bvh(ray: Ray, root: _BvhNode) -> tuple[tuple[AuraChunk, ...], int]:
    stack = [(0.0, root)]
    candidates: list[tuple[float, AuraChunk]] = []
    tested_nodes = 0
    while stack:
        _entry, node = stack.pop()
        tested_nodes += 1
        node_hit = node.bounds.intersect_ray(ray)
        if node_hit is None:
            continue
        if node.is_leaf:
            for chunk in node.chunks:
                chunk_hit = chunk.bounds.intersect_ray(ray)
                if chunk_hit is not None:
                    candidates.append((chunk_hit[0], chunk))
            continue
        children = []
        for child in (node.left, node.right):
            if child is None:
                continue
            child_hit = child.bounds.intersect_ray(ray)
            if child_hit is not None:
                children.append((child_hit[0], child))
        children.sort(key=lambda item: item[0], reverse=True)
        stack.extend(children)
    candidates.sort(key=lambda item: item[0])
    return tuple(chunk for _depth, chunk in candidates), tested_nodes


def _build_bvh(chunks: tuple[AuraChunk, ...]) -> _BvhNode:
    if len(chunks) <= 1:
        return _BvhNode(bounds=_union_chunk_bounds(chunks), chunks=chunks)
    axis = _longest_axis(_union_chunk_bounds(chunks))
    ordered = tuple(sorted(chunks, key=lambda chunk: _chunk_center(chunk, axis)))
    midpoint = len(ordered) // 2
    left = _build_bvh(ordered[:midpoint])
    right = _build_bvh(ordered[midpoint:])
    return _BvhNode(bounds=_union_chunk_bounds(chunks), chunks=tuple(), left=left, right=right)


def _union_chunk_bounds(chunks: Sequence[AuraChunk]) -> Bounds:
    if not chunks:
        raise ValueError("BVH requires at least one chunk")
    mins = tuple(min(chunk.bounds.min_corner[index] for chunk in chunks) for index in range(3))
    maxs = tuple(max(chunk.bounds.max_corner[index] for chunk in chunks) for index in range(3))
    return type(chunks[0].bounds)(mins, maxs)


def _longest_axis(bounds: Bounds) -> int:
    extents = tuple(bounds.max_corner[index] - bounds.min_corner[index] for index in range(3))
    return max(range(3), key=lambda index: extents[index])


def _chunk_center(chunk: AuraChunk, axis: int) -> float:
    return (chunk.bounds.min_corner[axis] + chunk.bounds.max_corner[axis]) * 0.5
