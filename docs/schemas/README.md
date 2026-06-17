# AURA JSON Schemas

These schemas describe the JSON files inside a native `.aura` package.

Current schema version: `0.1`

- `manifest.schema.json`: package manifest and capability metadata.
- `elements.schema.json`: bounded carrier element records.
- `chunks.schema.json`: chunk and LOD records.

The Python loader validates these schemas at runtime, then performs cross-file
checks such as manifest chunk IDs matching `chunks.json` and chunk element
references resolving to records in `elements.json`.
