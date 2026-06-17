# AURA CPU MVP

This package now contains the non-GPU skeleton for AURA:

- adaptive carrier registry;
- evidence-to-carrier assignment;
- bounded local elements;
- CPU reference ray queries;
- front-to-back compositing;
- chunk and LOD metadata;
- native `.aura` package writer;
- glTF/USD exchange-plan metadata;
- CLI fixtures.

It is not a renderer, trainer, CUDA kernel, or benchmark result. It is the stable
contract layer that later GPU/rendering work should implement against.

