# ChatGPT Handoff - AURA

Use this prompt when opening the private repo on another machine:

```text
You are helping me develop AURA, Adaptive Unified Radiance Asset.

Read README.md, AGENTS.md, docs/CPU_MVP.md, docs/DATASETS.md, src/aura/cli.py,
and the tests before changing code.

Goal: turn this CPU-only contract scaffold into the first AURA MVP:
3DGS bootstrap -> adaptive local carriers -> ray-query API -> .aura package ->
glTF/USD fallback metadata.

Rules:
- Do not commit datasets, third-party repos, checkpoints, outputs, .env,
  LOCAL_SECRETS.md, or tokens.
- Ask before GPU work unless I explicitly say this is the GPU machine.
- AURA is not a single splat variant. Keep the carrier registry and asset
  contract central.
- Add tests for every file parser, package schema, and query contract.

First useful tasks:
1. run tests;
2. add a tiny 3DGS export fixture;
3. implement a splat-to-AURA scaffold reader;
4. add first-hit/depth/transmittance query tests;
5. create one .aura fixture package;
6. only then connect to a real 3DGS baseline scene.
```

