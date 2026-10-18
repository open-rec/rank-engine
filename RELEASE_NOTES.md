# rank-engine v0.2.0

Released: 2026-10-18

Part of the coordinated OpenRec v0.2.0 source release.

## Changes since v0.1.0

- Separate HTTP handling, model lifecycle, feature loading and inference orchestration.
- Add model registry, batched inference and cached CPU projections, with CPU/CUDA selection and LightGBM inference support.
- Publish model, feature snapshot and executor together to preserve in-flight requests during replacement.

## Installation and compatibility

Build with rec-algorithm v0.2.0. Keep model weights, fitted feature sidecars and selected definitions together. Bootstrap artifacts now live in rank/{item,user}/{lr,fm,lightgbm}/. Trained releases retain separate runtime storage.

See the [distribution upgrade and rollback instructions](https://github.com/open-rec/openrec/blob/v0.2.0/RELEASE_NOTES.md) for the complete compatible component set and persisted configuration migrations.

## Validation and publication

The [validation record](https://github.com/open-rec/openrec/blob/v0.2.0/release/VALIDATION.md) distinguishes release checks from maintainer-reported prior CI and integration results. GitHub Actions publishes source archives and SHA-256 checksums; Maven Central, PyPI and versioned container registries are not populated by this workflow.
