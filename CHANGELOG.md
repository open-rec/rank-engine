# Changelog

## Unreleased

- Separate HTTP handling, model lifecycle, feature loading and inference orchestration.
- Add a model registry and LR/FM serving forward entry points while reusing rec-algorithm
  checkpoint definitions; retain CPU projections and batched GPU inference.
- Keep LightGBM CPU/GPU prediction and conversion together under `models/lightgbm/`;
  reserve `backends/` for reusable execution frameworks such as Torch.
- Publish model, feature snapshot and executor as one runtime version, preserving
  in-flight requests during feature refresh and model replacement.
- Preserve existing HTTP and artifact contracts; add model-extension and HTTP lifecycle
  regression coverage. See [architecture and extension guidance](docs/architecture.md).

## 0.1.0 — 2026-10-05

First coordinated release. See [release notes](RELEASE_NOTES.md) for features, compatibility and known limitations.
