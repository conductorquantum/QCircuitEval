# CI / maintainer scripts

Maintainer utilities used by workflows and local checks. Run them through the
project environment:

```bash
uv run python ci/<name>.py ...
```

- `export_public_subset.py` — optional utility that writes a 28-instance
  category sample to `public_subset/` (not checked in). The public dataset is
  the full 280-instance task set in `src/qceval/assets`.
- `generate_semantic_assets.py` — regenerate or `--check` packaged core
  contracts and targets.
- `generate_qec_semantic_assets.py` — regenerate or `--check` packaged QEC
  contracts, targets, and framework task assets.
- `check_asset_consistency.py` — fail CI when task files, contracts, and
  prompt scaffolds disagree on grading-relevant fields.
- `check_packaged_assets_wheel.py` — build an isolated wheel with hatchling and
  smoke-test asset loading from a clean ``--no-deps`` install (requires the
  ``dev`` extra so ``hatchling`` is available).
