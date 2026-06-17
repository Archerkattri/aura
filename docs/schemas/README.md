# AURA JSON Schemas

These schemas describe the JSON files inside a native `.aura` package.

Current schema version: `0.1`

- `manifest.schema.json`: package manifest and capability metadata.
- `elements.schema.json`: bounded carrier element records, typed carrier
  payloads, confidence maps, and edit metadata.
- `chunks.schema.json`: chunk and LOD records.
- `semantic_graph.schema.json`: semantic/object nodes, element bindings, and
  relationships.
- `exchange.schema.json`: native AURA, glTF fallback, and USD bridge metadata.
- `capture_manifest.schema.json`: real capture manifest used to point AURA-Core
  at image/depth/mask/normal files, camera intrinsics, frame poses, and seed regions.
- `training_dataset.schema.json`: AURA-Core posed frame, target, and native
  evidence-region inputs used by reconstruction fixtures.

The Python loader validates these schemas at runtime, then performs cross-file
checks such as manifest chunk IDs matching `chunks.json`, unique chunk IDs,
chunk element references resolving to records in `elements.json`, and each
element `chunk_id` agreeing with the chunk that lists it. It also checks that
chunk bounds contain every listed element so reference chunk/BVH culling cannot
drop valid hits. Any non-empty element payload type must match the element
carrier. The JSON Schema validates the payload shape for surface, volume, beta,
gabor, neural, gaussian fallback, and semantic carriers.

The AURA-Core training loader also validates `training_dataset.schema.json`
before constructing native frames and evidence regions, then checks that every
region references a known frame. The capture-manifest loader validates
`capture_manifest.schema.json` and converts it to the same training dataset
contract without reading image pixels; GPU-side loaders should replace the
summary color/depth/normal targets with real sampled image/depth/normal losses
while keeping the manifest identifiers stable.
