# pydra-compose-monai test plan

**Date:** 2026-05-28
**Status:** Approved design, ready for implementation plan
**Scope:** Tests for `pydra.compose.monai` — `builder.py`, `fields.py`, `spec_parser.py`, `task.py`. Includes targeted behavioural refinements driven by the tests.

## Background

`pydra-compose-monai` wraps MONAI Model Zoo bundles as Pydra task classes. Given a path to a bundle (or its `configs/metadata.json`), `monai.define(...)` returns a Pydra `Task` subclass whose input/output fields mirror the bundle's `network_data_format`. At run time, `MonaiTask._run` loads the bundle's `configs/inference.json` via `monai.bundle.ConfigParser`, populates `dataset#data` from the task's input fields, sets `output_dir`, and calls `evaluator.run()`. `MonaiOutputs._from_job` then scans the output directory and populates the output fields.

The module has only one existing test file (`tests/test_monai_spec.py`) covering a subset of `spec_parser` and `define()`. `_run`, `_from_job`, and `_resolve_bundle_dir` are uncovered. This spec defines a complete test plan and a small set of behavioural refinements the tests will encode.

## Goals

- Lock in tested contracts for `builder`, `fields`, `spec_parser`, and `task`.
- Refine a handful of brittle or implicit behaviours in `task.py` and `builder.py` via test-driven changes.
- Establish a synthetic-bundle fixture so `_run` and `_from_job` can be exercised end-to-end without external dependencies.
- Add one network-gated smoke test against a published MONAI Model Zoo bundle to catch upstream drift.

## Non-goals

- Pydra-engine performance benchmarks.
- End-to-end DICOM input flowing through `_run` (parser-level DICOM coverage only).
- Snapshot or regression testing of model outputs.
- A matrix across multiple MONAI versions.

## Test scope (tests-as-spec)

The test suite (a) covers what exists today, (b) refines small behavioural issues where doing so makes contracts cleaner, and (c) includes one opt-in subset of end-to-end integration testing using a real published bundle. Brittle behaviours are explicitly *not* pinned; they are refined first, then tested.

## Test environment

- CPU only; no GPU assumed.
- Network access is permitted in CI but gated behind a marker so default runs work offline.
- Python ≥ 3.11. `monai >= 1.3` is already a hard dependency; no extra install needed.

## File layout

```
pydra/compose/monai/tests/
    conftest.py            # shared fixtures
    test_fields.py         # arg / out construction
    test_spec_parser.py    # parse_monai_spec, _map_type, name_from_spec, _import_monai_bundle
    test_builder.py        # define() class form, spec form, name override, base attrs, path metadata
    test_task.py           # _resolve_bundle_dir, _run, _from_job with mocked evaluator
    test_integration.py    # @pytest.mark.integration / @pytest.mark.network
```

The current `tests/test_monai_spec.py` is split into `test_spec_parser.py` and `test_builder.py` along the obvious seam.

## Module-by-module coverage

### `test_fields.py`

- `arg(name=..., type=..., path=...)` stores all kwargs.
- `out(name=..., type=..., path=...)` stores all kwargs.
- `path` defaults to `None` on both.
- Type and help propagate through to the resulting field.

### `test_spec_parser.py`

- `parse_monai_spec` loads from a metadata.json path.
- `parse_monai_spec` loads from a bundle directory (reads `configs/metadata.json`).
- `_map_type` resolves:
  - `type=image, modality=MRI` → `NiftiGzX`
  - `format=segmentation` → `NiftiGzX`
  - `format=hounsfield` or `modality=CT` → `NiftiGzX`
  - `format=dicom` → `DicomSeries`
  - `type=dicom_series` → `DicomSeries`
  - unknown combinations → `ty.Any`
- Input help string contains modality and format when present.
- Output help string contains format and channel count when present.
- `name_from_spec` returns `_meta_#name` from metadata when present, transformed via `_to_class_name`.
- `name_from_spec` falls back to the directory name when metadata is missing.
- `name_from_spec` sanitises non-identifier characters (e.g. `"Whole Brain Seg UNEST"` → `"WholeBrainSegUnest"`).
- `_import_monai_bundle` resolves to the PyPI `monai.bundle` even when `pydra/compose` is on `sys.path` (regression guard).

### `test_builder.py`

- `define(klass)` (decorator form on a class with a `function` attribute) returns a Task class.
- `define(metadata_json_path)` returns a Task class with fields parsed from metadata.
- `define(bundle_dir)` likewise.
- Explicit `name=` overrides the metadata-derived name.
- Base attrs (`model_weights`) are added to the parsed inputs on every generated Task class.
- `arg.path` and `out.path` survive `build_task_class` and are reachable via `field.metadata["__PYDRA_METADATA__"].path`.
- `bases=` and `outputs_bases=` pass-through.
- `define()` accepts both `str` and `pathlib.Path` for the spec path.
- `define()` with a non-existent path raises a clear error.

### `test_task.py`

`_resolve_bundle_dir`:

- Returns the input path if it is a directory containing `configs/metadata.json`.
- Raises `ValueError` if the directory does not contain `configs/metadata.json` (R3).
- Walks up from a `.pt` / `.ts` weights file to find the bundle root.
- Raises `ValueError` if a weights file has no `configs/metadata.json` in any ancestor.
- Raises `ValueError` if `model_weights` is `None`.
- Calls `monai.bundle.load(..., source="monaihosting")` for inputs that look like a Model Zoo bundle name (no path separator, no file extension) (R4).
- Raises `ValueError` for strings that look neither like a valid path nor a valid bundle name (R4).

`_run` (uses the mocked-`ConfigParser` fixture):

- Calls `ConfigParser().read_meta(bundle_dir / "configs/metadata.json")`.
- Calls `ConfigParser().load_config_file(bundle_dir / "configs/inference.json")`.
- Sets `parser["dataset#data"]` to a list with one dict mapping each non-`BASE_ATTRS` input field name to its stringified value.
- Sets `parser["output_dir"]` to the job's output directory (which is created if absent).
- Calls `parser.get_parsed_content("evaluator", instantiate=True)`.
- Calls `evaluator.run()` exactly once.

`_from_job` (R1 + R2):

- Reads the bundle's `inference.json` postprocessing transforms.
- Constructs expected output paths from `SaveImaged` / `SaveImage` `output_postfix`, `output_ext`, and the matched key.
- Populates each output field with the matched file path.
- Leaves the field unset when no matching file exists.
- Does not touch base Output fields (`stdout`, `stderr`, `return_code`); these come from `super()._from_job(job)`.

### `test_integration.py`

- **Synthetic bundle E2E (`@pytest.mark.integration`):** Build a synthetic bundle in `tmp_path`, generate a Task class via `define()`, instantiate with a downsampled T1w patch as input, call `_run` directly with a real `ConfigParser` and `SupervisedEvaluator`. Verify the expected output NIfTI lands in `output_dir` with the postprocessing-defined postfix.
- **Submitter run (`@pytest.mark.integration`):** Same synthetic bundle, but submitted through a Pydra Submitter (rather than calling `_run` directly). Verifies the field metadata and output collection round-trip cleanly through the engine.
- **Real-bundle smoke (`@pytest.mark.integration` + `@pytest.mark.network`):** Download `spleen_ct_segmentation` via `monai.bundle.load`, pass the bundle dir to `define()`, assert the generated Task class has the expected `image` input field and `pred` output field, and a non-empty class name. Does *not* run inference (no CT data, and the synthetic bundle already covers `_run`).

## Synthetic bundle fixture

A session-scoped fixture in `tests/conftest.py` builds a real but minimal MONAI bundle in `tmp_path` once per session and returns the bundle directory.

Structure:

```
<tmp_path>/synthetic_bundle/
    configs/
        metadata.json
        inference.json
```

`metadata.json` (illustrative):

```json
{
  "name": "Synthetic Test Bundle",
  "network_data_format": {
    "inputs":  {"image": {"type": "image", "modality": "MRI", "num_channels": 1, "spatial_shape": [16, 16, 16]}},
    "outputs": {"pred":  {"type": "image", "format": "segmentation", "num_channels": 1, "spatial_shape": [16, 16, 16]}}
  }
}
```

`inference.json` (illustrative — key references use MONAI's `@name` syntax):

- `imports`: `["$import torch"]`
- `device`: `"cpu"`
- `output_dir`: `""` (overridden by `_run`)
- `network_def`: `monai.networks.nets.UNet` with `spatial_dims=3`, `in_channels=1`, `out_channels=1`, `channels=[4, 8]`, `strides=[2]`
- `preprocessing`: `Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"])])`
- `dataset`: `Dataset(data=[], transform=@preprocessing)` (data overridden by `_run`)
- `dataloader`: `DataLoader(dataset=@dataset, batch_size=1)`
- `inferer`: `SimpleInferer`
- `postprocessing`: `Compose([SaveImaged(keys=["pred"], output_dir=@output_dir, output_postfix="seg", separate_folder=False)])`
- `evaluator`: `SupervisedEvaluator(network=@network_def, inferer=@inferer, postprocessing=@postprocessing, val_data_loader=@dataloader, device=@device)`

The fixture exposes a `make_synthetic_bundle(metadata_overrides=None, inference_overrides=None)` factory so individual tests can produce variants (e.g. a DICOM-input metadata.json, a bundle with a different output postfix) without proliferating top-level fixtures.

A separate fixture (`bundle_with_weights_file`) constructs a synthetic bundle with a `models/model.pt` placeholder file to exercise the walk-up branch of `_resolve_bundle_dir`.

Inputs to the synthetic bundle come from the existing `test-data/nifti/anat/T1w.nii.gz` sample. The integration test crops it to a 16³ patch before passing to `_run` to keep CPU runtime under a few seconds.

## Mocked parser fixture

For `test_task.py` unit tests of `_run`, a `mock_config_parser` fixture patches `monai.bundle.ConfigParser` so the parser instance is a `MagicMock` whose `get_parsed_content("evaluator", ...)` returns a `MagicMock` evaluator with a `.run()` method. The patch is applied via `monkeypatch` on the `monai.bundle.ConfigParser` class. The fixture returns both the parser mock and the evaluator mock so tests can assert on `__setitem__` calls (`parser["dataset#data"] = ...`, `parser["output_dir"] = ...`) and on `evaluator.run.assert_called_once()`.

This keeps unit tests sub-millisecond and free of torch.

## Behavioural refinements (code changes driven by tests)

| ID | Location | Refinement |
|----|----------|------------|
| R1 | `MonaiOutputs._from_job` ([task.py:33](../../../pydra/compose/monai/task.py)) | Replace the loose `*{field}*` glob fallback with deterministic output-path resolution. Read `inference.json` postprocessing. For each output field, walk the postprocessing transforms (`Compose` → `SaveImaged` / `SaveImage`) and find a transform whose `keys` (or single `key`) contains the field name. From that transform's `output_postfix`, `output_ext` (default `.nii.gz`), and the input filename used in `dataset#data`, construct the expected output path. If no postprocessing transform writes a given output field, log a warning and leave the field unset. |
| R2 | `MonaiOutputs._from_job` ([task.py:35](../../../pydra/compose/monai/task.py)) | Replace the `field.name.startswith("_")` skip with an explicit list of base Output field names (`stdout`, `stderr`, `return_code`). Mirror via a `BASE_OUTPUT_ATTRS` tuple, analogous to `MonaiTask.BASE_ATTRS`. |
| R3 | `MonaiTask._resolve_bundle_dir` ([task.py:134](../../../pydra/compose/monai/task.py)) | Validate that the input directory contains `configs/metadata.json`; raise `ValueError` with a clear message otherwise. |
| R4 | `MonaiTask._resolve_bundle_dir` ([task.py:147](../../../pydra/compose/monai/task.py)) | Validate the repo-ID string before calling `monai.bundle.load`. A valid bundle name has no path separator and no recognised file extension. Strings that look like file paths or have extensions raise `ValueError` referencing the supported input forms (existing dir, existing weights file, bundle name). |
| R5 | `MonaiTask.arch` field ([task.py:67](../../../pydra/compose/monai/task.py)) | Remove. Unused anywhere in `_run` or the broader codebase. Re-add when a concrete use case appears. |
| R6 | `define()` signature ([builder.py:30](../../../pydra/compose/monai/builder.py)) | Change `wrapped: type \| Yaml \| None` to `wrapped: type \| str \| Path \| None`. Drop the unused `Yaml` import. |

## Pinned behaviours (tests written, no code change)

| ID | Behaviour |
|----|-----------|
| P1 | `_run` derives `dataset#data` keys from field **name**, not the `arg.path` attribute. A generated task's field name must equal its corresponding `network_data_format.inputs` key. This is the contract; the test documents it. |
| P2 | `_map_type` mapping rules (see `test_spec_parser.py` coverage above) are pinned. |
| P3 | `_import_monai_bundle` correctly resolves the PyPI module under sys.path shadowing. |
| P4 | `name_from_spec` non-identifier sanitisation rules pinned. |
| P5 | `arg.path` and `out.path` round-trip through `build_task_class`. |

## Known limitations (not addressed in this round)

These are recorded as `pytest.skip(reason="<short reason>; see spec L<N>")` in the test files, so they appear in the test report without polluting failures.

- **L1.** Runtime DICOM inputs through `_run`. Parser-level DICOM coverage exists; end-to-end DICOM input is out of scope here.
- **L2.** Scalar / classification outputs (`type: "scalar"`, regression heads, etc.). `_map_type` falls through to `ty.Any`; `_from_job` cannot resolve a non-file output.
- **L3.** Bundles whose `inference.json` does not expose a top-level `output_dir` key. `_run` unconditionally sets `parser["output_dir"]`; for bundles that hardcode the path elsewhere, the override is harmless but ineffective.

## Pytest markers and run modes

`pyproject.toml` `[tool.pytest.ini_options]`:

```toml
pythonpath = ["."]
markers = [
    "integration: end-to-end tests that build a synthetic bundle and run real CPU inference (~5s)",
    "network: tests that require outbound network access to MONAI hosting",
]
addopts = "-m 'not integration and not network'"
```

Run modes:

- `pytest` — unit tests only. Fast, no torch instantiation, no network. Default CI signal.
- `pytest -m integration` — synthetic-bundle E2E + submitter test. No network.
- `pytest -m network` — real-bundle smoke test only. Requires outbound network.
- `pytest -m ''` — everything.

## Coverage target

Configure `coverage` to enforce a **70% minimum on `pydra/compose/monai/`**, excluding `_version.py`. Threshold lives alongside the existing `.coveragerc`. Below threshold fails the test job.

## CI workflow (recommendation)

Not part of the test-writing PR itself, but worth establishing alongside:

- **Job 1 — `tests`:** `pip install -e .[test]`, run `pytest`. Fast, no network.
- **Job 2 — `tests-integration`:** Same install, run `pytest -m integration`. Slower; non-blocking initially.
- **Job 3 — `tests-network`:** Same install, run `pytest -m network`. Allowed to fail without blocking PRs until stable.

## Implementation order

1. Add markers + `addopts` to `pyproject.toml`; update `.coveragerc` for the 70% bar.
2. Split `test_monai_spec.py` into `test_spec_parser.py` and `test_builder.py`; broaden coverage per the matrix above.
3. Write `test_fields.py`.
4. Apply refinements R5 (remove `arch`) and R6 (drop `Yaml`) — smallest, lowest-risk.
5. Write the synthetic-bundle fixture and mocked-parser fixture in `tests/conftest.py`.
6. Apply refinements R2 and R3, with tests.
7. Apply refinement R4 (repo-ID validation), with tests.
8. Apply refinement R1 (postprocessing-driven output resolution), with tests. Largest single change.
9. Write `test_task.py` covering `_run` and `_resolve_bundle_dir` via the mocked parser.
10. Write `test_integration.py` — synthetic bundle E2E, Submitter run, real-bundle smoke (network-gated).

## Deferred to later spec sessions

These were discussed during brainstorming and explicitly deferred:

- Output-handling contract (beyond R1's postprocessing-driven resolution) — a broader design covering how outputs map back to XNAT.
- Model-weight distribution and update strategy (bake into container vs runtime download vs host cache, plus auto-update semantics).
- Preprocessing pipeline (DICOM→NIfTI conversion, optional registration) as upstream Pydra tasks.
- End-to-end XNAT integration (model selection UI, data fetch, result upload).
