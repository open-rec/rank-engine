# rank-engine v0.1.0

Released: 2026-10-05

Online model inference. First coordinated OpenRec source release.

## Features

- FastAPI scoring for source users and item or user candidates, using LR, FM and LightGBM models.
- Shared encoders and feature contracts from rec-algorithm, Redis feature snapshots, and refreshable time-window features.
- Versioned model loading, retained activation state, startup recovery and readiness checks.
- Explicit feature-sidecar validation, scoring metrics and CPU/CUDA device configuration.
- Session and request feature transport where supported by the selected model contract.

## Installation and compatibility

Build with rec-algorithm `v0.1.0`. Model weights, fitted feature sidecars and selected definitions must be compatible. Default bootstrap models live under `rank/item` and `rank/user`; trained releases and activation state use separate runtime storage.

The service loads and scores models; offline training belongs to rec-algorithm. Standalone normally bypasses ranking. A nonempty rec-server response alone does not establish that ranking succeeded; cluster acceptance checks rank scores.

## Validation and known boundaries

See this repository's README for build/test commands and deployment requirements. The coordinated release's [validation record](https://github.com/open-rec/openrec/blob/v0.1.0/release/VALIDATION.md) distinguishes checks executed for this release from historical integration evidence.

This initial release establishes a versioned source baseline. Source archives and checksums are published; external package registries and container registries are not populated by the source-release workflow. Upgrade the complete compatible distribution, retain data/checkpoints/artifacts, and preserve prior component refs for rollback.
