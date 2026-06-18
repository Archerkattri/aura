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
        traversal_index = self._traversal_index
        candidate_chunks, traversal_mode, bvh_stats = _candidate_chunks(ray, traversal_index.chunks, traversal_index.bvh_root)
        if traversal_index.chunks:
            candidate_ids: list[str] = []
            seen_candidate_ids: set[str] = set()
            for chunk in candidate_chunks:
                for element_id in chunk.element_ids:
                    if element_id in seen_candidate_ids:
                        continue
                    seen_candidate_ids.add(element_id)
                    candidate_ids.append(element_id)
            chunk_candidates = tuple(
                traversal_index.element_by_id[element_id]
                for element_id in candidate_ids
                if element_id in traversal_index.element_by_id
            )
            orphan_candidates = tuple(
                traversal_index.element_by_id[element_id]
                for element_id in traversal_index.orphan_element_ids
                if element_id in traversal_index.element_by_id
            )
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
                tested_bvh_node_count=bvh_stats.tested_node_count,
                tested_bvh_leaf_count=bvh_stats.tested_leaf_count,
                tested_bvh_chunk_bounds_count=bvh_stats.tested_chunk_bounds_count,
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
            tested_bvh_node_count=bvh_stats.tested_node_count,
            tested_bvh_leaf_count=bvh_stats.tested_leaf_count,
            tested_bvh_chunk_bounds_count=bvh_stats.tested_chunk_bounds_count,
        )

    def carrier_ids(self) -> list[str]:
        return sorted({element.carrier_id for element in self.elements})

    def chunk_ids(self) -> list[str]:
        return sorted({element.chunk_id for element in self.elements})

    @cached_property
    def _chunk_bvh(self) -> "_BvhNode | None":
        return self._traversal_index.bvh_root

    @cached_property
    def _traversal_index(self) -> "_SceneTraversalIndex":
        return _prepare_traversal_index(self)

    def traversal_acceleration(self) -> "SceneAccelerationMetadata":
        """Return cached, serializable acceleration metadata for runtime reports."""

        return self._traversal_index.metadata


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
    """Complete diagnostic record for one ray traversal through an :class:`AuraScene`.

    ``result`` holds the composited front-to-back output; ``ordered_hits``
    contains the per-element traces in depth order. The BVH and chunk
    counters are populated when spatial acceleration is active.
    """

    result: RayQueryResult
    ordered_hits: tuple[RayHitTrace, ...]
    tested_chunk_ids: tuple[str, ...]
    tested_element_ids: tuple[str, ...]
    total_element_count: int
    hit_count: int
    traversal_mode: str = "linear"
    tested_bvh_node_count: int = 0
    tested_bvh_leaf_count: int = 0
    tested_bvh_chunk_bounds_count: int = 0

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
            "testedBvhLeafCount": self.tested_bvh_leaf_count,
            "testedBvhChunkBoundsCount": self.tested_bvh_chunk_bounds_count,
            "acceleration": {
                "mode": self.traversal_mode,
                "candidateChunkCount": len(self.tested_chunk_ids),
                "candidateElementCount": len(self.tested_element_ids),
                "skippedElementCount": self.skipped_element_count,
                "bvh": {
                    "testedNodeCount": self.tested_bvh_node_count,
                    "testedLeafCount": self.tested_bvh_leaf_count,
                    "testedChunkBoundsCount": self.tested_bvh_chunk_bounds_count,
                },
            },
        }


@dataclass(frozen=True)
class SceneAccelerationMetadata:
    """Serializable snapshot of scene spatial-acceleration statistics.

    Returned by :meth:`AuraScene.traversal_acceleration`. Useful for
    diagnostic reports and renderer readiness checks.
    """

    element_count: int
    chunk_count: int
    chunked_element_count: int
    orphan_element_count: int
    active_traversal_mode: str
    bvh_chunk_threshold: int
    bvh_node_count: int
    bvh_leaf_count: int
    bvh_max_depth: int
    bvh_leaf_chunk_counts: tuple[int, ...]

    @property
    def chunked_element_coverage_rate(self) -> float:
        if self.element_count == 0:
            return 1.0
        return self.chunked_element_count / self.element_count

    def to_dict(self) -> dict:
        return {
            "elementCount": self.element_count,
            "chunkCount": self.chunk_count,
            "chunkedElementCount": self.chunked_element_count,
            "orphanElementCount": self.orphan_element_count,
            "chunkedElementCoverageRate": self.chunked_element_coverage_rate,
            "activeTraversalMode": self.active_traversal_mode,
            "bvhChunkThreshold": self.bvh_chunk_threshold,
            "supportsChunkCulling": self.chunk_count > 0,
            "supportsCachedBvh": self.bvh_node_count > 0,
            "bvhNodeCount": self.bvh_node_count,
            "bvhLeafCount": self.bvh_leaf_count,
            "bvhMaxDepth": self.bvh_max_depth,
            "bvhLeafChunkCounts": list(self.bvh_leaf_chunk_counts),
            "supportsOrderedFrontToBackCandidates": True,
            "supportsUnchunkedElementFallback": True,
            "candidateOrdering": "front_to_back_chunks_then_unchunked_elements",
        }


def composite_front_to_back(hits: Iterable[RayQueryResult]) -> RayQueryResult:
    """Composite an ordered sequence of ray hits front-to-back into a single result.

    Blending uses the standard over-operator: each hit's transmittance
    attenuates remaining light for all subsequent hits.
    """
    color: Vec3 = (0.0, 0.0, 0.0)
    transmittance = 1.0
    confidence_num = 0.0
    confidence_den = 0.0
    depth_num = 0.0
    depth_den = 0.0
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
        if hit.depth is not None:
            depth_num += weight * hit.depth
            depth_den += weight
        transmittance *= hit.transmittance
        provenance.append(hit.provenance or "unknown")
        residual = residual or hit.residual
    confidence = 0.0 if confidence_den == 0.0 else confidence_num / confidence_den
    # Contribution-weighted expected depth (matches the torch renderer); falls
    # back to the first hit's depth when nothing contributes weight.
    if depth_den > 1e-8:
        weighted_depth = depth_num / depth_den
    else:
        weighted_depth = first.depth if first else None
    return RayQueryResult(
        color=color,
        transmittance=transmittance,
        confidence=confidence,
        depth=weighted_depth,
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


@dataclass(frozen=True)
class _BvhTraversalStats:
    tested_node_count: int = 0
    tested_leaf_count: int = 0
    tested_chunk_bounds_count: int = 0


@dataclass(frozen=True)
class _SceneTraversalIndex:
    chunks: tuple[AuraChunk, ...]
    element_by_id: dict[str, AuraElement]
    orphan_element_ids: tuple[str, ...]
    bvh_root: _BvhNode | None
    metadata: SceneAccelerationMetadata


def _prepare_traversal_index(scene: AuraScene) -> _SceneTraversalIndex:
    chunks = tuple(scene.chunks)
    element_by_id = {element.id: element for element in scene.elements}
    chunked_element_ids = {element_id for chunk in chunks for element_id in chunk.element_ids if element_id in element_by_id}
    orphan_element_ids = tuple(element.id for element in scene.elements if element.id not in chunked_element_ids)
    bvh_root = _build_bvh(chunks) if len(chunks) >= BVH_CHUNK_THRESHOLD else None
    bvh_node_count, bvh_leaf_count, bvh_max_depth, bvh_leaf_chunk_counts = _bvh_metadata(bvh_root)
    metadata = SceneAccelerationMetadata(
        element_count=len(scene.elements),
        chunk_count=len(chunks),
        chunked_element_count=len(chunked_element_ids),
        orphan_element_count=len(orphan_element_ids),
        active_traversal_mode=_active_traversal_mode(chunks),
        bvh_chunk_threshold=BVH_CHUNK_THRESHOLD,
        bvh_node_count=bvh_node_count,
        bvh_leaf_count=bvh_leaf_count,
        bvh_max_depth=bvh_max_depth,
        bvh_leaf_chunk_counts=bvh_leaf_chunk_counts,
    )
    return _SceneTraversalIndex(
        chunks=chunks,
        element_by_id=element_by_id,
        orphan_element_ids=orphan_element_ids,
        bvh_root=bvh_root,
        metadata=metadata,
    )


def _active_traversal_mode(chunks: Sequence[AuraChunk]) -> str:
    if not chunks:
        return "element_linear"
    if len(chunks) >= BVH_CHUNK_THRESHOLD:
        return "bvh"
    return "chunk_linear"


def _bvh_metadata(root: _BvhNode | None) -> tuple[int, int, int, tuple[int, ...]]:
    if root is None:
        return 0, 0, 0, tuple()
    node_count = 0
    leaf_count = 0
    max_depth = 0
    leaf_chunk_counts = []
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        node_count += 1
        max_depth = max(max_depth, depth)
        if node.is_leaf:
            leaf_count += 1
            leaf_chunk_counts.append(len(node.chunks))
            continue
        for child in (node.left, node.right):
            if child is not None:
                stack.append((child, depth + 1))
    return node_count, leaf_count, max_depth, tuple(leaf_chunk_counts)


def _candidate_chunks(
    ray: Ray,
    chunks: Sequence[AuraChunk],
    bvh_root: "_BvhNode | None" = None,
) -> tuple[tuple[AuraChunk, ...], str, _BvhTraversalStats]:
    if len(chunks) >= BVH_CHUNK_THRESHOLD:
        root = bvh_root if bvh_root is not None else _build_bvh(tuple(chunks))
        candidates, stats = _candidate_chunks_bvh(ray, root)
        return candidates, "bvh", stats
    candidates = []
    for chunk in chunks:
        hit = chunk.bounds.intersect_ray(ray)
        if hit is not None:
            candidates.append((hit[0], chunk))
    candidates.sort(key=lambda item: item[0])
    mode = "chunk_linear" if chunks else "element_linear"
    return (
        tuple(chunk for _depth, chunk in candidates),
        mode,
        _BvhTraversalStats(tested_chunk_bounds_count=len(chunks)),
    )


def _candidate_chunks_bvh(ray: Ray, root: _BvhNode) -> tuple[tuple[AuraChunk, ...], _BvhTraversalStats]:
    root_hit = root.bounds.intersect_ray(ray)
    tested_nodes = 1
    if root_hit is None:
        return tuple(), _BvhTraversalStats(tested_node_count=tested_nodes)
    stack = [(root_hit[0], root)]
    candidates: list[tuple[float, AuraChunk]] = []
    tested_leaf_count = 0
    tested_chunk_bounds_count = 0
    while stack:
        _entry, node = stack.pop()
        if node.is_leaf:
            tested_leaf_count += 1
            for chunk in node.chunks:
                tested_chunk_bounds_count += 1
                chunk_hit = chunk.bounds.intersect_ray(ray)
                if chunk_hit is not None:
                    candidates.append((chunk_hit[0], chunk))
            continue
        children = []
        for child in (node.left, node.right):
            if child is None:
                continue
            tested_nodes += 1
            child_hit = child.bounds.intersect_ray(ray)
            if child_hit is not None:
                children.append((child_hit[0], child))
        children.sort(key=lambda item: item[0], reverse=True)
        stack.extend(children)
    candidates.sort(key=lambda item: item[0])
    return (
        tuple(chunk for _depth, chunk in candidates),
        _BvhTraversalStats(
            tested_node_count=tested_nodes,
            tested_leaf_count=tested_leaf_count,
            tested_chunk_bounds_count=tested_chunk_bounds_count,
        ),
    )


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
