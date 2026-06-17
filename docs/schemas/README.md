# AURA JSON Schemas

These schemas describe the JSON files inside a native `.aura` package.

Current schema version: `0.1`

- `manifest.schema.json`: package manifest and capability metadata.
- `elements.schema.json`: bounded carrier element records, typed carrier
  payloads, confidence maps, and edit metadata.
- `chunks.schema.json`: chunk and LOD records.
- `semantic_graph.schema.json`: semantic/object nodes, element bindings, and
  relationships.

The Python loader validates these schemas at runtime, then performs cross-file
checks such as manifest chunk IDs matching `chunks.json` and chunk element
references resolving to records in `elements.json`. It also checks that any
non-empty element payload type matches the element carrier. The JSON Schema
validates the payload shape for surface, volume, beta, gabor, neural, gaussian
fallback, and semantic carriers.
