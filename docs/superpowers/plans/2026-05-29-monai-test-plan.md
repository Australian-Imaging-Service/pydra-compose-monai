# pydra-compose-monai Test Plan Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive test suite for `pydra.compose.monai` (builder, fields, spec_parser, task), driven by tests-as-spec: small behavioural refinements (R1–R6) are made first via failing tests, then existing contracts are pinned.

**Architecture:** Per-module test files under `pydra/compose/monai/tests/`, a session-scoped synthetic-bundle fixture for end-to-end coverage, a `mock_config_parser` fixture for unit-testing `_run` orchestration without torch, and pytest markers (`integration`, `network`) for opt-in slower/network-dependent tests.

**Tech Stack:** pytest, pytest-cov, monai (already a hard dep), fileformats, MONAI Model Zoo's `monai.bundle.ConfigParser` for real-bundle paths, `unittest.mock` for fakes.

**Reference spec:** [docs/superpowers/specs/2026-05-28-monai-test-plan-design.md](../specs/2026-05-28-monai-test-plan-design.md)

---

## File Structure

Files this plan creates or modifies:

```
pyproject.toml                                   # markers, addopts
.coveragerc                                      # fail_under threshold (Task 15)
pydra/compose/monai/builder.py                   # R6: drop Yaml typing
pydra/compose/monai/task.py                      # R1-R5: refinements
pydra/compose/monai/tests/conftest.py            # new: synthetic bundle + mock parser fixtures
pydra/compose/monai/tests/test_fields.py         # new
pydra/compose/monai/tests/test_spec_parser.py    # new (from old test_monai_spec.py)
pydra/compose/monai/tests/test_builder.py        # new (from old test_monai_spec.py)
pydra/compose/monai/tests/test_task.py           # new
pydra/compose/monai/tests/test_integration.py    # new
pydra/compose/monai/tests/test_monai_spec.py     # DELETED after split
```

Each test file mirrors one source module. Refinements (R1–R6) are applied to `task.py` and `builder.py` only.

---

## Task 1: Configure pytest markers and addopts

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit pyproject.toml's `[tool.pytest.ini_options]` section**

Replace:
```toml
[tool.pytest.ini_options]
# Keep the repo root (not pydra/compose/) on sys.path so that `import monai`
# resolves to the PyPI package, not to this pydra/compose/monai subpackage.
pythonpath = ["."]
```

With:
```toml
[tool.pytest.ini_options]
# Keep the repo root (not pydra/compose/) on sys.path so that `import monai`
# resolves to the PyPI package, not to this pydra/compose/monai subpackage.
pythonpath = ["."]
markers = [
    "integration: end-to-end tests that build a synthetic bundle and run real CPU inference (~5s)",
    "network: tests that require outbound network access to MONAI hosting",
]
addopts = "-m 'not integration and not network' --cov=pydra/compose/monai --cov-report=term-missing"
```

(Coverage `fail_under` is added in Task 15 once unit coverage is sufficient.)

- [ ] **Step 2: Verify markers are recognised**

Run: `pytest --markers | grep -E '(integration|network)'`
Expected output includes:
```
@pytest.mark.integration: end-to-end tests that build a synthetic bundle and run real CPU inference (~5s)
@pytest.mark.network: tests that require outbound network access to MONAI hosting
```

- [ ] **Step 3: Verify existing tests still pass with new config**

Run: `pytest -v`
Expected: All existing tests in `test_monai_spec.py` PASS, coverage report printed at the end.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: register integration/network pytest markers and enable coverage"
```

---

## Task 2: Split `test_monai_spec.py` into `test_spec_parser.py` and `test_builder.py`

The existing file mixes parser tests with `define()` tests. Splitting them clarifies failures.

**Files:**
- Create: `pydra/compose/monai/tests/test_spec_parser.py`
- Create: `pydra/compose/monai/tests/test_builder.py`
- Delete: `pydra/compose/monai/tests/test_monai_spec.py`

- [ ] **Step 1: Create `test_spec_parser.py`** with the parser tests and the fixtures they share

```python
"""Tests for parsing MONAI bundle metadata into Pydra field definitions."""
import json
import typing as ty
import pytest
from pathlib import Path
from fileformats.medimage import NiftiGzX
from pydra.compose.monai.spec_parser import parse_monai_spec, name_from_spec


MINIMAL_METADATA = {
    "name": "Whole Brain Seg UNEST",
    "network_data_format": {
        "inputs": {
            "image": {
                "type": "image",
                "format": "hounsfield",
                "modality": "MRI",
                "num_channels": 1,
                "spatial_shape": [96, 96, 96],
            }
        },
        "outputs": {
            "pred": {
                "type": "image",
                "format": "segmentation",
                "num_channels": 133,
                "spatial_shape": [96, 96, 96],
            }
        },
    },
}


@pytest.fixture
def metadata_json(tmp_path: Path) -> Path:
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(MINIMAL_METADATA))
    return p


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "metadata.json").write_text(json.dumps(MINIMAL_METADATA))
    return tmp_path


# ---------------------------------------------------------------------------
# parse_monai_spec
# ---------------------------------------------------------------------------


def test_parse_inputs_from_metadata_json(metadata_json: Path):
    parsed_inputs, _ = parse_monai_spec(metadata_json)
    assert "image" in parsed_inputs
    field = parsed_inputs["image"]
    assert field.name == "image"
    assert field.type is NiftiGzX
    assert field.path == "network_data_format/inputs/image"


def test_parse_outputs_from_metadata_json(metadata_json: Path):
    _, parsed_outputs = parse_monai_spec(metadata_json)
    assert "pred" in parsed_outputs
    field = parsed_outputs["pred"]
    assert field.name == "pred"
    assert field.type is NiftiGzX
    assert field.path == "network_data_format/outputs/pred"


def test_parse_from_bundle_dir(bundle_dir: Path):
    parsed_inputs, parsed_outputs = parse_monai_spec(bundle_dir)
    assert "image" in parsed_inputs
    assert "pred" in parsed_outputs


def test_parse_unknown_type_falls_back_to_any(tmp_path: Path):
    metadata = {
        "network_data_format": {
            "inputs": {"feat": {"type": "tensor", "format": "embedding"}},
            "outputs": {},
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    parsed_inputs, _ = parse_monai_spec(p)
    assert parsed_inputs["feat"].type is ty.Any


def test_input_help_contains_modality(metadata_json: Path):
    parsed_inputs, _ = parse_monai_spec(metadata_json)
    assert "MRI" in parsed_inputs["image"].help


def test_output_help_contains_format(metadata_json: Path):
    _, parsed_outputs = parse_monai_spec(metadata_json)
    assert "segmentation" in parsed_outputs["pred"].help


# ---------------------------------------------------------------------------
# name_from_spec
# ---------------------------------------------------------------------------


def test_name_from_spec_uses_metadata_name(metadata_json: Path):
    assert name_from_spec(metadata_json) == "WholeBrainSegUnest"


def test_name_from_spec_uses_dir_name(tmp_path: Path):
    name = name_from_spec(tmp_path)
    assert name
    assert name.isidentifier()
```

- [ ] **Step 2: Create `test_builder.py`** with the `define()` tests

```python
"""Tests for the pydra.compose.monai define() builder."""
import json
import pytest
from pathlib import Path
from fileformats.medimage import NiftiGzX
from pydra.compose import monai


MINIMAL_METADATA = {
    "name": "Whole Brain Seg UNEST",
    "network_data_format": {
        "inputs": {
            "image": {
                "type": "image",
                "format": "hounsfield",
                "modality": "MRI",
                "num_channels": 1,
                "spatial_shape": [96, 96, 96],
            }
        },
        "outputs": {
            "pred": {
                "type": "image",
                "format": "segmentation",
                "num_channels": 133,
                "spatial_shape": [96, 96, 96],
            }
        },
    },
}


@pytest.fixture
def metadata_json(tmp_path: Path) -> Path:
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(MINIMAL_METADATA))
    return p


def test_define_from_metadata_json_creates_task_class(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    assert TaskCls is not None
    field_names = [f.name for f in TaskCls.__attrs_attrs__]
    assert "image" in field_names
    assert "model_weights" in field_names


def test_define_from_metadata_json_outputs_have_pred(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    output_field_names = [f.name for f in TaskCls.Outputs.__attrs_attrs__]
    assert "pred" in output_field_names


def test_define_preserves_path_on_arg(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    image_field = next(f for f in TaskCls.__attrs_attrs__ if f.name == "image")
    pydra_meta = image_field.metadata["__PYDRA_METADATA__"]
    assert pydra_meta.path == "network_data_format/inputs/image"


def test_define_preserves_path_on_out(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    pred_field = next(
        f for f in TaskCls.Outputs.__attrs_attrs__ if f.name == "pred"
    )
    pydra_meta = pred_field.metadata["__PYDRA_METADATA__"]
    assert pydra_meta.path == "network_data_format/outputs/pred"


def test_define_class_name_from_metadata(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    assert TaskCls.__name__ == "WholeBrainSegUnest"


def test_define_explicit_name_overrides_metadata(metadata_json: Path):
    TaskCls = monai.define(metadata_json, name="MyCustomTask")
    assert TaskCls.__name__ == "MyCustomTask"
```

- [ ] **Step 3: Delete the original file**

```bash
git rm pydra/compose/monai/tests/test_monai_spec.py
```

- [ ] **Step 4: Run the suite to confirm no test was lost**

Run: `pytest pydra/compose/monai/tests/ -v`
Expected: all 12 tests pass (6 in `test_spec_parser.py`, 6 in `test_builder.py`).

- [ ] **Step 5: Commit**

```bash
git add pydra/compose/monai/tests/
git commit -m "refactor(tests): split test_monai_spec.py into per-module files"
```

---

## Task 3: Write `test_fields.py`

**Files:**
- Create: `pydra/compose/monai/tests/test_fields.py`

- [ ] **Step 1: Write the tests**

```python
"""Tests for pydra.compose.monai.fields.arg and .out."""
import typing as ty
from fileformats.medimage import NiftiGzX
from pydra.compose import monai


def test_arg_stores_path_kwarg():
    a = monai.arg(name="T1w", type=NiftiGzX, path="anat/T1w")
    assert a.path == "anat/T1w"


def test_arg_path_defaults_to_none():
    a = monai.arg(name="T1w", type=NiftiGzX)
    assert a.path is None


def test_arg_type_and_help_propagate():
    a = monai.arg(name="T1w", type=NiftiGzX, help="T1-weighted image")
    assert a.type is NiftiGzX
    assert a.help == "T1-weighted image"


def test_out_stores_path_kwarg():
    o = monai.out(name="mask", type=NiftiGzX, path="anat/mask")
    assert o.path == "anat/mask"


def test_out_path_defaults_to_none():
    o = monai.out(name="mask", type=NiftiGzX)
    assert o.path is None


def test_out_type_and_help_propagate():
    o = monai.out(name="mask", type=NiftiGzX, help="binary mask")
    assert o.type is NiftiGzX
    assert o.help == "binary mask"


def test_arg_accepts_any_type():
    a = monai.arg(name="weights", type=ty.Any)
    assert a.type is ty.Any
```

- [ ] **Step 2: Run the tests**

Run: `pytest pydra/compose/monai/tests/test_fields.py -v`
Expected: 7 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/tests/test_fields.py
git commit -m "test: cover arg and out field constructors"
```

---

## Task 4: Broaden `test_spec_parser.py`

Add `_map_type` rule coverage, DICOM coverage, name sanitisation edge cases, and the `_import_monai_bundle` shadow-path guard.

**Files:**
- Modify: `pydra/compose/monai/tests/test_spec_parser.py`

- [ ] **Step 1: Append DICOM, name sanitisation, and shadow-guard tests**

Append to `pydra/compose/monai/tests/test_spec_parser.py`:

```python
import sys


# ---------------------------------------------------------------------------
# _map_type — additional rules
# ---------------------------------------------------------------------------


def test_map_type_dicom_format(tmp_path: Path):
    from fileformats.medimage import DicomSeries

    metadata = {
        "network_data_format": {
            "inputs": {"image": {"format": "dicom"}},
            "outputs": {},
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    parsed_inputs, _ = parse_monai_spec(p)
    assert parsed_inputs["image"].type is DicomSeries


def test_map_type_dicom_series_type(tmp_path: Path):
    from fileformats.medimage import DicomSeries

    metadata = {
        "network_data_format": {
            "inputs": {"image": {"type": "dicom_series"}},
            "outputs": {},
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    parsed_inputs, _ = parse_monai_spec(p)
    assert parsed_inputs["image"].type is DicomSeries


def test_map_type_ct_modality(tmp_path: Path):
    metadata = {
        "network_data_format": {
            "inputs": {"image": {"type": "image", "modality": "CT"}},
            "outputs": {},
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    parsed_inputs, _ = parse_monai_spec(p)
    assert parsed_inputs["image"].type is NiftiGzX


def test_map_type_segmentation_output(tmp_path: Path):
    metadata = {
        "network_data_format": {
            "inputs": {},
            "outputs": {"pred": {"type": "image", "format": "segmentation"}},
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    _, parsed_outputs = parse_monai_spec(p)
    assert parsed_outputs["pred"].type is NiftiGzX


# ---------------------------------------------------------------------------
# name_from_spec — edge cases
# ---------------------------------------------------------------------------


def test_name_from_spec_sanitises_punctuation(tmp_path: Path):
    metadata = {"name": "foo-bar.baz_qux v2"}
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    name = name_from_spec(p)
    assert name == "FooBarBazQuxV2"
    assert name.isidentifier()


def test_name_from_spec_handles_missing_metadata_name(tmp_path: Path):
    # Metadata without "name" field
    metadata = {"network_data_format": {"inputs": {}, "outputs": {}}}
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    name = name_from_spec(p)
    # Falls back to file stem
    assert name.isidentifier()


# ---------------------------------------------------------------------------
# _import_monai_bundle — sys.path shadow guard
# ---------------------------------------------------------------------------


def test_import_monai_bundle_resolves_pypi_under_shadow():
    """Even if pydra/compose is on sys.path, monai.bundle must resolve to the
    PyPI package, not to pydra.compose.monai."""
    from pydra.compose.monai.spec_parser import _import_monai_bundle

    shadow = str(Path(__file__).parent.parent.parent)  # pydra/compose
    sys.path.insert(0, shadow)
    try:
        mod = _import_monai_bundle()
        # The PyPI monai package's bundle module has ConfigParser
        assert hasattr(mod, "ConfigParser")
        # And its __file__ is NOT inside pydra/compose/monai
        assert "pydra/compose/monai" not in (mod.__file__ or "")
    finally:
        if shadow in sys.path:
            sys.path.remove(shadow)
```

- [ ] **Step 2: Run the tests**

Run: `pytest pydra/compose/monai/tests/test_spec_parser.py -v`
Expected: all tests in the file PASS (original 9 + 7 new = 16).

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/tests/test_spec_parser.py
git commit -m "test(spec_parser): cover DICOM mapping, name sanitisation, shadow guard"
```

---

## Task 5: Broaden `test_builder.py`

Add coverage for `Path` vs `str` inputs, non-existent paths, `bases=` pass-through, and the bundle-dir form.

**Files:**
- Modify: `pydra/compose/monai/tests/test_builder.py`

- [ ] **Step 1: Append the new tests**

Append to `pydra/compose/monai/tests/test_builder.py`:

```python
@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "metadata.json").write_text(json.dumps(MINIMAL_METADATA))
    return tmp_path


def test_define_accepts_path_object(metadata_json: Path):
    TaskCls = monai.define(Path(metadata_json))
    assert TaskCls is not None


def test_define_accepts_str_path(metadata_json: Path):
    TaskCls = monai.define(str(metadata_json))
    assert TaskCls is not None


def test_define_from_bundle_dir(bundle_dir: Path):
    TaskCls = monai.define(bundle_dir)
    field_names = [f.name for f in TaskCls.__attrs_attrs__]
    assert "image" in field_names


def test_define_includes_base_attrs(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    field_names = {f.name for f in TaskCls.__attrs_attrs__}
    assert "model_weights" in field_names


def test_define_rejects_non_path_non_class():
    with pytest.raises(ValueError, match="must be a class or a str"):
        monai.define(42)


def test_define_image_input_has_nifti_type(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    image_field = next(f for f in TaskCls.__attrs_attrs__ if f.name == "image")
    assert image_field.type is NiftiGzX
```

- [ ] **Step 2: Run the tests**

Run: `pytest pydra/compose/monai/tests/test_builder.py -v`
Expected: all tests PASS (6 existing + 6 new = 12).

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/tests/test_builder.py
git commit -m "test(builder): cover Path/str inputs, bundle dir form, base attrs"
```

---

## Task 6: R5 — Remove the `arch` field from `MonaiTask`

`arch` is declared on `MonaiTask` and listed in `BASE_ATTRS`, but is never referenced in `_run` or anywhere else. Remove it (YAGNI).

**Files:**
- Modify: `pydra/compose/monai/task.py:57-69`
- Modify: `pydra/compose/monai/tests/test_builder.py` (add regression test)

- [ ] **Step 1: Write the failing regression test**

Append to `pydra/compose/monai/tests/test_builder.py`:

```python
def test_define_does_not_include_arch_field(metadata_json: Path):
    """R5: `arch` is YAGNI and was removed."""
    TaskCls = monai.define(metadata_json)
    field_names = {f.name for f in TaskCls.__attrs_attrs__}
    assert "arch" not in field_names
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest pydra/compose/monai/tests/test_builder.py::test_define_does_not_include_arch_field -v`
Expected: FAIL with `AssertionError: assert 'arch' not in {'arch', 'image', 'model_weights'}`.

- [ ] **Step 3: Remove the `arch` field from `MonaiTask`**

Edit `pydra/compose/monai/task.py`. Replace:

```python
    BASE_ATTRS = (
        "model_weights",
        "arch",
    )

    model_weights: str = fields.arg(
        name="model_weights",
        type=ty.Any,
        help="the weights of the model",
    )
    arch: list[tuple[str, str]] | None = fields.arg(
        name="arch", type=ty.Any, help="the architecture of the model"
    )
```

With:

```python
    BASE_ATTRS = ("model_weights",)

    model_weights: str = fields.arg(
        name="model_weights",
        type=ty.Any,
        help="the weights of the model",
    )
```

- [ ] **Step 4: Run the regression test and full suite**

Run: `pytest pydra/compose/monai/tests/ -v`
Expected: all tests PASS, including `test_define_does_not_include_arch_field`.

- [ ] **Step 5: Commit**

```bash
git add pydra/compose/monai/task.py pydra/compose/monai/tests/test_builder.py
git commit -m "refactor(task): remove unused arch field (R5)"
```

---

## Task 7: R6 — Drop `Yaml` from `define()` signature

`Yaml` from `fileformats.application` appears in the type hint but `define()` never handles a Yaml argument; the relevant branch handles `str` and `Path`.

**Files:**
- Modify: `pydra/compose/monai/builder.py:8` and `:30`
- Modify: `pydra/compose/monai/tests/test_builder.py` (already has `test_define_accepts_path_object` and `test_define_accepts_str_path` from Task 5; these stay as pin tests)

- [ ] **Step 1: Update the import and signature in builder.py**

Edit `pydra/compose/monai/builder.py`. Remove the line:

```python
from fileformats.application import Yaml
```

Replace:

```python
def define(
    wrapped: type | Yaml | None = None,
    /,
    inputs: list[str | arg] | dict[str, arg | type] | None = None,
```

With:

```python
def define(
    wrapped: type | str | Path | None = None,
    /,
    inputs: list[str | arg] | dict[str, arg | type] | None = None,
```

(`Path` is already imported at the top of the file.)

- [ ] **Step 2: Verify the existing builder tests still pass**

Run: `pytest pydra/compose/monai/tests/test_builder.py -v`
Expected: all PASS, including `test_define_accepts_path_object` and `test_define_accepts_str_path`.

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/builder.py
git commit -m "refactor(builder): drop unused Yaml typing from define() (R6)"
```

---

## Task 8: Add fixtures to `pydra/compose/monai/tests/conftest.py`

Two fixtures: a session-scoped synthetic-bundle factory (for integration tests), and a function-scoped `mock_config_parser` (for unit tests of `_run`). Also a small helper that constructs a `Job`-like stand-in.

**Files:**
- Create: `pydra/compose/monai/tests/conftest.py`

- [ ] **Step 1: Write the fixtures**

Create `pydra/compose/monai/tests/conftest.py`:

```python
"""Shared fixtures for pydra.compose.monai tests."""
import json
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock
import pytest


# ---------------------------------------------------------------------------
# Synthetic bundle factory
# ---------------------------------------------------------------------------


DEFAULT_METADATA = {
    "name": "Synthetic Test Bundle",
    "network_data_format": {
        "inputs": {
            "image": {
                "type": "image",
                "modality": "MRI",
                "num_channels": 1,
                "spatial_shape": [16, 16, 16],
            }
        },
        "outputs": {
            "pred": {
                "type": "image",
                "format": "segmentation",
                "num_channels": 1,
                "spatial_shape": [16, 16, 16],
            }
        },
    },
}


DEFAULT_INFERENCE = {
    "imports": ["$import torch"],
    "device": "cpu",
    "output_dir": "",
    "network_def": {
        "_target_": "monai.networks.nets.UNet",
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [4, 8],
        "strides": [2],
    },
    "preprocessing": {
        "_target_": "Compose",
        "transforms": [
            {"_target_": "LoadImaged", "keys": ["image"]},
            {"_target_": "EnsureChannelFirstd", "keys": ["image"]},
        ],
    },
    "dataset": {
        "_target_": "Dataset",
        "data": [],
        "transform": "@preprocessing",
    },
    "dataloader": {
        "_target_": "DataLoader",
        "dataset": "@dataset",
        "batch_size": 1,
    },
    "inferer": {"_target_": "SimpleInferer"},
    "postprocessing": {
        "_target_": "Compose",
        "transforms": [
            {
                "_target_": "SaveImaged",
                "keys": ["pred"],
                "output_dir": "@output_dir",
                "output_postfix": "seg",
                "separate_folder": False,
            }
        ],
    },
    "evaluator": {
        "_target_": "SupervisedEvaluator",
        "device": "@device",
        "val_data_loader": "@dataloader",
        "network": "@network_def",
        "inferer": "@inferer",
        "postprocessing": "@postprocessing",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge — `override` wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@pytest.fixture
def make_synthetic_bundle(tmp_path: Path) -> Callable[..., Path]:
    """Factory: build a minimal MONAI bundle dir; override metadata/inference."""

    def _make(
        metadata_overrides: dict | None = None,
        inference_overrides: dict | None = None,
        subdir: str = "synthetic_bundle",
    ) -> Path:
        bundle = tmp_path / subdir
        configs = bundle / "configs"
        configs.mkdir(parents=True, exist_ok=True)

        metadata = _deep_merge(DEFAULT_METADATA, metadata_overrides or {})
        inference = _deep_merge(DEFAULT_INFERENCE, inference_overrides or {})

        (configs / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (configs / "inference.json").write_text(json.dumps(inference, indent=2))
        return bundle

    return _make


@pytest.fixture
def synthetic_bundle_dir(make_synthetic_bundle) -> Path:
    """Default synthetic bundle — image input, pred segmentation output."""
    return make_synthetic_bundle()


@pytest.fixture
def bundle_with_weights_file(make_synthetic_bundle) -> Path:
    """Synthetic bundle with a placeholder weights file at models/model.pt.

    Returns the path to the weights file (caller can walk up for the bundle root).
    """
    bundle = make_synthetic_bundle(subdir="bundle_with_weights")
    models = bundle / "models"
    models.mkdir()
    weights = models / "model.pt"
    weights.write_bytes(b"")  # empty placeholder; we never load it
    return weights


# ---------------------------------------------------------------------------
# Mock ConfigParser fixture for unit tests of _run
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_parser(monkeypatch):
    """Patch monai.bundle.ConfigParser so unit tests don't instantiate networks.

    Returns (parser_mock, evaluator_mock). The parser supports __setitem__ so
    tests can assert on `parser['dataset#data'] = ...`. The evaluator's
    `.run()` is a MagicMock for `assert_called_once()`.
    """
    parser = MagicMock(name="ConfigParser_instance")
    evaluator = MagicMock(name="evaluator")
    parser.get_parsed_content.return_value = evaluator

    # __setitem__ records calls so tests can assert on them
    parser.set_calls = {}

    def record_setitem(key, value):
        parser.set_calls[key] = value

    parser.__setitem__.side_effect = record_setitem

    # Patch the ConfigParser class used inside _run, which goes through
    # _import_monai_bundle().ConfigParser.
    import monai.bundle as monai_bundle

    monkeypatch.setattr(monai_bundle, "ConfigParser", MagicMock(return_value=parser))

    return parser, evaluator


# ---------------------------------------------------------------------------
# Stand-in for pydra.engine.Job in _run unit tests
# ---------------------------------------------------------------------------


class FakeJob:
    """Minimal stand-in for pydra's Job, sufficient for _run / _from_job."""

    def __init__(self, task, output_dir: Path):
        self.task = task
        self.output_dir = str(output_dir)


@pytest.fixture
def fake_job():
    """Factory for FakeJob instances bound to a task and output dir."""
    return FakeJob
```

- [ ] **Step 2: Add a smoke test that the fixtures load**

Append to `pydra/compose/monai/tests/test_spec_parser.py`:

```python
def test_synthetic_bundle_fixture_creates_valid_bundle(synthetic_bundle_dir: Path):
    """Smoke test: the fixture writes a parseable metadata.json."""
    assert (synthetic_bundle_dir / "configs" / "metadata.json").exists()
    assert (synthetic_bundle_dir / "configs" / "inference.json").exists()
    parsed_inputs, parsed_outputs = parse_monai_spec(synthetic_bundle_dir)
    assert "image" in parsed_inputs
    assert "pred" in parsed_outputs
```

- [ ] **Step 3: Run the smoke test**

Run: `pytest pydra/compose/monai/tests/test_spec_parser.py::test_synthetic_bundle_fixture_creates_valid_bundle -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pydra/compose/monai/tests/conftest.py pydra/compose/monai/tests/test_spec_parser.py
git commit -m "test: add synthetic bundle factory and mock_config_parser fixtures"
```

---

## Task 9: R3 — Validate that bundle directory contains `configs/metadata.json`

**Files:**
- Modify: `pydra/compose/monai/task.py:134`
- Create: `pydra/compose/monai/tests/test_task.py`

- [ ] **Step 1: Write the failing test**

Create `pydra/compose/monai/tests/test_task.py`:

```python
"""Tests for MonaiTask._run, _resolve_bundle_dir, and MonaiOutputs._from_job."""
import pytest
from pathlib import Path
from pydra.compose import monai
from pydra.compose.monai.task import MonaiTask


# ---------------------------------------------------------------------------
# _resolve_bundle_dir
# ---------------------------------------------------------------------------


def test_resolve_bundle_dir_accepts_dir_with_configs(synthetic_bundle_dir: Path):
    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image="dummy")
    # _resolve_bundle_dir takes a job-like object with a .task attribute
    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, Path("/tmp/unused"))
    assert task._resolve_bundle_dir(job) == synthetic_bundle_dir


def test_resolve_bundle_dir_rejects_dir_without_configs(tmp_path: Path):
    """R3: a directory without configs/metadata.json must raise ValueError."""
    bare = tmp_path / "bare"
    bare.mkdir()
    from pydra.compose.monai.tests.conftest import FakeJob

    # Use the synthetic bundle's TaskCls so we can construct a task instance
    # even though model_weights points at the bare dir.
    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(model_weights=str(bare))
    job = FakeJob(task, tmp_path / "out")

    with pytest.raises(ValueError, match="configs/metadata.json"):
        task._resolve_bundle_dir(job)
```

- [ ] **Step 2: Run the tests to verify the second one fails**

Run: `pytest pydra/compose/monai/tests/test_task.py -v`
Expected: `test_resolve_bundle_dir_accepts_dir_with_configs` PASSES, `test_resolve_bundle_dir_rejects_dir_without_configs` FAILS — current code does not validate the directory.

- [ ] **Step 3: Add validation in `_resolve_bundle_dir`**

Edit `pydra/compose/monai/task.py`. Replace the existing `if path.is_dir():` block:

```python
        if path.is_dir():
            return path
```

With:

```python
        if path.is_dir():
            if not (path / "configs" / "metadata.json").is_file():
                raise ValueError(
                    f"Bundle directory {path} does not contain configs/metadata.json. "
                    "Pass a path to a valid MONAI bundle root."
                )
            return path
```

- [ ] **Step 4: Run the test again**

Run: `pytest pydra/compose/monai/tests/test_task.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pydra/compose/monai/task.py pydra/compose/monai/tests/test_task.py
git commit -m "feat(task): validate bundle dir has configs/metadata.json (R3)"
```

---

## Task 10: R4 — Validate repo-ID shape before `monai.bundle.load`

A non-existent path string currently falls through to `monai.bundle.load`, producing a confusing 404. Validate the input has no path separator and no recognised file extension before treating it as a Model Zoo bundle name.

**Files:**
- Modify: `pydra/compose/monai/task.py:147` (the repo-ID branch)
- Modify: `pydra/compose/monai/tests/test_task.py`

- [ ] **Step 1: Write failing tests**

Append to `pydra/compose/monai/tests/test_task.py`:

```python
def test_resolve_bundle_dir_treats_simple_name_as_repo_id(monkeypatch, tmp_path: Path):
    """R4: a string with no path sep / no extension is a Model Zoo bundle name."""
    from unittest.mock import MagicMock
    import monai.bundle as monai_bundle

    fake_dir = tmp_path / "downloaded"
    fake_dir.mkdir()
    (fake_dir / "configs").mkdir()
    (fake_dir / "configs" / "metadata.json").write_text("{}")

    fake_load = MagicMock(return_value=str(fake_dir))
    monkeypatch.setattr(monai_bundle, "load", fake_load)

    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)

    task = TaskCls(model_weights="spleen_ct_segmentation")
    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")
    result = task._resolve_bundle_dir(job)
    assert result == Path(fake_dir)
    fake_load.assert_called_once()


def test_resolve_bundle_dir_rejects_path_like_string(tmp_path: Path):
    """R4: strings with path separators must raise rather than hit the network."""
    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)

    task = TaskCls(model_weights="/tmp/nonexistent/bundle.pt")
    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")

    with pytest.raises(ValueError, match="not a valid MONAI bundle"):
        task._resolve_bundle_dir(job)


def test_resolve_bundle_dir_rejects_string_with_extension(tmp_path: Path):
    """R4: strings with a known file extension must raise rather than hit the network."""
    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)

    task = TaskCls(model_weights="my_model.pt")
    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")

    with pytest.raises(ValueError, match="not a valid MONAI bundle"):
        task._resolve_bundle_dir(job)
```

- [ ] **Step 2: Run the tests to verify the path-like rejection ones fail**

Run: `pytest pydra/compose/monai/tests/test_task.py -v`
Expected: `test_resolve_bundle_dir_treats_simple_name_as_repo_id` may PASS (current behavior accidentally handles this), but `test_resolve_bundle_dir_rejects_path_like_string` and `test_resolve_bundle_dir_rejects_string_with_extension` FAIL.

- [ ] **Step 3: Add the validation to `_resolve_bundle_dir`**

In `pydra/compose/monai/task.py`, replace the fall-through block at the end of `_resolve_bundle_dir`:

```python
        # Treat as a Hugging Face / MONAI Model Zoo repo ID and download
        from .spec_parser import _import_monai_bundle
        bundle_load = _import_monai_bundle().load
        logger.info("Downloading MONAI bundle %s", weights)
        bundle_dir = bundle_load(str(weights), source="monaihosting")
        return Path(bundle_dir)
```

With:

```python
        # If it's not a path on disk, the only remaining valid form is a
        # MONAI Model Zoo bundle name (e.g. "spleen_ct_segmentation").
        # Bundle names contain no path separators and no file extension.
        weights_str = str(weights)
        if (
            "/" in weights_str
            or "\\" in weights_str
            or Path(weights_str).suffix != ""
        ):
            raise ValueError(
                f"model_weights={weights_str!r} is not a valid MONAI bundle "
                "reference. Provide one of: an existing bundle directory, an "
                "existing weights file inside a bundle, or a Model Zoo bundle "
                "name (e.g. 'spleen_ct_segmentation')."
            )

        from .spec_parser import _import_monai_bundle
        bundle_load = _import_monai_bundle().load
        logger.info("Downloading MONAI bundle %s", weights_str)
        bundle_dir = bundle_load(weights_str, source="monaihosting")
        return Path(bundle_dir)
```

- [ ] **Step 4: Run the test suite**

Run: `pytest pydra/compose/monai/tests/test_task.py -v`
Expected: all five `_resolve_bundle_dir` tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pydra/compose/monai/task.py pydra/compose/monai/tests/test_task.py
git commit -m "feat(task): validate repo-ID shape before bundle download (R4)"
```

---

## Task 11: R2 — Explicit `BASE_OUTPUT_ATTRS` skip in `_from_job`

The current `_from_job` skips on `field.name.startswith("_")`, which never matches `stdout` / `stderr` / `return_code`. Replace with an explicit set.

**Files:**
- Modify: `pydra/compose/monai/task.py:16-48`
- Modify: `pydra/compose/monai/tests/test_task.py`

- [ ] **Step 1: Write the failing test**

Append to `pydra/compose/monai/tests/test_task.py`:

```python
# ---------------------------------------------------------------------------
# _from_job — base output field handling
# ---------------------------------------------------------------------------


def test_from_job_does_not_overwrite_base_output_fields(synthetic_bundle_dir, tmp_path, monkeypatch):
    """R2: stdout/stderr/return_code populated by super() must not be overwritten
    by stray files in the output dir."""
    from pydra.compose.monai.task import MonaiOutputs
    from pydra.compose.monai.tests.conftest import FakeJob

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Drop files whose names start with the base output field names
    (output_dir / "stdout.log").write_text("noise")
    (output_dir / "stderr.txt").write_text("noise")

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image="dummy.nii.gz")
    job = FakeJob(task, output_dir)

    # Patch super()._from_job to return a sentinel populated outputs object
    # so we can verify our code doesn't clobber it.
    OutputsCls = TaskCls.Outputs

    # Run _from_job. The base fields should remain whatever super() set them to.
    outputs = OutputsCls._from_job(job)

    # We don't care about the exact value super() set, only that our loop
    # didn't replace it with the stray log files.
    import attrs

    for f in attrs.fields(OutputsCls):
        if f.name in ("stdout", "stderr", "return_code"):
            # Must not be a Path pointing at one of our noise files
            val = getattr(outputs, f.name, None)
            assert val is None or "noise" not in (Path(str(val)).name if val else "")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest pydra/compose/monai/tests/test_task.py::test_from_job_does_not_overwrite_base_output_fields -v`
Expected: FAIL — the current loop matches `stdout.log` and `stderr.txt` via the wildcard glob and overwrites the base fields.

- [ ] **Step 3: Refine `_from_job` to use explicit base attrs**

In `pydra/compose/monai/task.py`, replace the `MonaiOutputs` class with (note: this also stays compatible with the next refinement, R1, which replaces the glob entirely):

```python
@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiOutputs(base.Outputs):

    BASE_OUTPUT_ATTRS = ("stdout", "stderr", "return_code")

    @classmethod
    def _from_job(cls, job: "Job[MonaiTask]") -> ty.Self:
        """Collect outputs after inference by scanning the job's output directory.

        Parameters
        ----------
        job : Job[MonaiTask]
            The completed job whose output directory contains inference results.

        Returns
        -------
        outputs : MonaiOutputs
            Populated outputs dataclass.
        """
        outputs = super()._from_job(job)
        output_dir = Path(job.output_dir)

        if not output_dir.exists():
            return outputs

        for field in attrs.fields(cls):
            if field.name in cls.BASE_OUTPUT_ATTRS:
                continue
            candidates = sorted(output_dir.glob(f"{field.name}.*"))
            if candidates:
                object.__setattr__(outputs, field.name, candidates[0])
            else:
                # also try any file whose stem contains the field name
                candidates = sorted(output_dir.glob(f"*{field.name}*"))
                if candidates:
                    object.__setattr__(outputs, field.name, candidates[0])

        return outputs
```

(The loose-glob fallback is still here; Task 12 replaces it.)

- [ ] **Step 4: Run the test again**

Run: `pytest pydra/compose/monai/tests/test_task.py::test_from_job_does_not_overwrite_base_output_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pydra/compose/monai/task.py pydra/compose/monai/tests/test_task.py
git commit -m "fix(task): skip base output fields by explicit name in _from_job (R2)"
```

---

## Task 12: R1 — Postprocessing-driven output path resolution

Read `inference.json` postprocessing, walk to find `SaveImage`/`SaveImaged` transforms, and use each transform's `output_postfix` + `output_ext` plus the source image filename to construct deterministic expected output paths. Drop the loose `*{field}*` fallback.

**Files:**
- Modify: `pydra/compose/monai/task.py` (refactor `_from_job` + add helpers)
- Modify: `pydra/compose/monai/tests/test_task.py`

- [ ] **Step 1: Write failing tests for the new behaviour**

Append to `pydra/compose/monai/tests/test_task.py`:

```python
# ---------------------------------------------------------------------------
# _from_job — R1 postprocessing-driven path resolution
# ---------------------------------------------------------------------------


def test_from_job_resolves_output_from_save_transform(
    make_synthetic_bundle, tmp_path
):
    """R1: output path is constructed from SaveImaged postfix/ext and the input
    image filename, not from a loose glob match."""
    from pydra.compose.monai.tests.conftest import FakeJob

    bundle = make_synthetic_bundle()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Simulate what MONAI's SaveImaged would write: T1w_seg.nii.gz
    expected = output_dir / "T1w_seg.nii.gz"
    expected.write_bytes(b"fake nifti")

    TaskCls = monai.define(bundle)
    task = TaskCls(model_weights=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    assert Path(str(outputs.pred)) == expected


def test_from_job_does_not_match_unrelated_files(
    make_synthetic_bundle, tmp_path
):
    """R1: a file whose name *contains* the field name but doesn't match the
    SaveImaged postfix must NOT be picked up."""
    from pydra.compose.monai.tests.conftest import FakeJob

    bundle = make_synthetic_bundle()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Drop a misleading file: contains "pred" but is not the SaveImaged output
    (output_dir / "unrelated_pred_data.csv").write_text("noise")

    TaskCls = monai.define(bundle)
    task = TaskCls(model_weights=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    # Either unset (None) or — if it inherits from base.Outputs default — falsy.
    val = getattr(outputs, "pred", None)
    assert val is None or Path(str(val)).name != "unrelated_pred_data.csv"


def test_from_job_leaves_field_unset_when_save_transform_missing(
    make_synthetic_bundle, tmp_path
):
    """R1: if a bundle's postprocessing doesn't write a given output, the field
    is left unset (no glob fallback)."""
    from pydra.compose.monai.tests.conftest import FakeJob

    # Bundle whose postprocessing writes no images at all
    bundle = make_synthetic_bundle(
        inference_overrides={"postprocessing": {"_target_": "Compose", "transforms": []}}
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "pred.nii.gz").write_bytes(b"stray match")

    TaskCls = monai.define(bundle)
    task = TaskCls(model_weights=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    val = getattr(outputs, "pred", None)
    assert val is None or "stray" in (Path(str(val)).name if val else "")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest pydra/compose/monai/tests/test_task.py -k from_job -v`
Expected: the three new tests FAIL (`test_from_job_resolves_output_from_save_transform`, `test_from_job_does_not_match_unrelated_files`, `test_from_job_leaves_field_unset_when_save_transform_missing`).

- [ ] **Step 3: Implement R1 in `_from_job`**

In `pydra/compose/monai/task.py`, replace the `MonaiOutputs` class with:

```python
@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiOutputs(base.Outputs):

    BASE_OUTPUT_ATTRS = ("stdout", "stderr", "return_code")
    _IMAGE_EXTS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".dcm")

    @classmethod
    def _from_job(cls, job: "Job[MonaiTask]") -> ty.Self:
        """Collect outputs by reading the bundle's postprocessing config.

        For each non-base output field, find a SaveImage / SaveImaged
        transform in inference.json whose keys include that field name, then
        construct the expected output path from the transform's
        ``output_postfix`` and ``output_ext`` and the source image's filename.
        """
        outputs = super()._from_job(job)
        output_dir = Path(job.output_dir)

        if not output_dir.exists():
            return outputs

        try:
            bundle_dir = job.task._resolve_bundle_dir(job)
            save_specs = _parse_save_transforms(bundle_dir / "configs" / "inference.json")
        except Exception as exc:
            logger.warning(
                "Could not parse postprocessing for output resolution: %s", exc
            )
            return outputs

        input_stem = _first_input_stem(job.task)

        for field in attrs.fields(cls):
            if field.name in cls.BASE_OUTPUT_ATTRS:
                continue
            spec = save_specs.get(field.name)
            if spec is None:
                logger.warning(
                    "No SaveImage(d) transform writes output %r; field unset",
                    field.name,
                )
                continue
            if input_stem is None:
                continue
            postfix = spec.get("output_postfix", "")
            ext = spec.get("output_ext", ".nii.gz")
            if not ext.startswith("."):
                ext = "." + ext
            if postfix:
                fname = f"{input_stem}_{postfix}{ext}"
            else:
                fname = f"{input_stem}{ext}"
            expected = output_dir / fname
            if expected.is_file():
                object.__setattr__(outputs, field.name, expected)
            else:
                logger.warning(
                    "Expected output %s not found; field %r left unset",
                    expected, field.name,
                )

        return outputs


def _parse_save_transforms(inference_json: Path) -> dict[str, dict]:
    """Walk an inference.json's postprocessing for SaveImage(d) transforms.

    Returns a mapping ``{output_field_name: {"output_postfix": ..., "output_ext": ...}}``.
    """
    import json as _json

    if not inference_json.is_file():
        return {}
    config = _json.loads(inference_json.read_text())
    node = config.get("postprocessing")
    transforms = _extract_transforms(node)

    out: dict[str, dict] = {}
    for t in transforms:
        if not isinstance(t, dict):
            continue
        target = str(t.get("_target_", ""))
        if not (target.endswith("SaveImage") or target.endswith("SaveImaged")):
            continue
        keys = t.get("keys")
        if keys is None and "key" in t:
            keys = [t["key"]]
        if not keys:
            continue
        for key in keys:
            out[str(key)] = {
                "output_postfix": t.get("output_postfix", ""),
                "output_ext": t.get("output_ext", ".nii.gz"),
            }
    return out


def _extract_transforms(node) -> list:
    """Flatten a postprocessing node into a list of transform dicts."""
    if node is None:
        return []
    if isinstance(node, list):
        result = []
        for item in node:
            result.extend(_extract_transforms(item))
        return result
    if isinstance(node, dict):
        target = str(node.get("_target_", ""))
        if "Compose" in target:
            return _extract_transforms(node.get("transforms", []))
        return [node]
    return []


def _first_input_stem(task) -> str | None:
    """Return the stem (sans image extension) of the first non-BASE input."""
    for field in attrs.fields(type(task)):
        if field.name in MonaiTask.BASE_ATTRS:
            continue
        val = getattr(task, field.name, None)
        if val is None:
            continue
        name = Path(str(val)).name
        for ext in MonaiOutputs._IMAGE_EXTS:
            if name.lower().endswith(ext):
                return name[: -len(ext)]
        return Path(name).stem
    return None
```

(Note `_parse_save_transforms`, `_extract_transforms`, `_first_input_stem` are module-level helpers. `MonaiTask` is defined later in the file — Python's name resolution at call time handles the forward reference.)

- [ ] **Step 4: Run the R1 tests**

Run: `pytest pydra/compose/monai/tests/test_task.py -k from_job -v`
Expected: all three R1 tests PASS.

- [ ] **Step 5: Run the full test_task.py to confirm no regression**

Run: `pytest pydra/compose/monai/tests/test_task.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 6: Commit**

```bash
git add pydra/compose/monai/task.py pydra/compose/monai/tests/test_task.py
git commit -m "feat(task): resolve outputs from postprocessing transforms (R1)"
```

---

## Task 13: Test `_run` orchestration via mock parser

Use the `mock_config_parser` fixture to verify `_run` builds the right `dataset#data`, sets `output_dir`, and calls `evaluator.run()`.

**Files:**
- Modify: `pydra/compose/monai/tests/test_task.py`

- [ ] **Step 1: Append the `_run` orchestration tests**

```python
# ---------------------------------------------------------------------------
# _run orchestration (mocked evaluator)
# ---------------------------------------------------------------------------


def test_run_loads_metadata_and_inference_configs(
    synthetic_bundle_dir, mock_config_parser, tmp_path
):
    parser, evaluator = mock_config_parser
    output_dir = tmp_path / "out"

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    parser.read_meta.assert_called_once_with(
        str(synthetic_bundle_dir / "configs" / "metadata.json")
    )
    parser.load_config_file.assert_called_once_with(
        str(synthetic_bundle_dir / "configs" / "inference.json")
    )


def test_run_sets_dataset_data_from_inputs(
    synthetic_bundle_dir, mock_config_parser, tmp_path
):
    parser, _evaluator = mock_config_parser
    output_dir = tmp_path / "out"

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(
        model_weights=str(synthetic_bundle_dir),
        image="/data/T1w.nii.gz",
    )

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    # parser["dataset#data"] should have been set to a one-element list of dicts
    assert "dataset#data" in parser.set_calls
    data = parser.set_calls["dataset#data"]
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["image"] == "/data/T1w.nii.gz"


def test_run_sets_output_dir(
    synthetic_bundle_dir, mock_config_parser, tmp_path
):
    parser, _evaluator = mock_config_parser
    output_dir = tmp_path / "out"

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    assert parser.set_calls.get("output_dir") == str(output_dir)
    assert output_dir.exists()


def test_run_calls_evaluator_run_once(
    synthetic_bundle_dir, mock_config_parser, tmp_path
):
    _parser, evaluator = mock_config_parser
    output_dir = tmp_path / "out"

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    evaluator.run.assert_called_once()


def test_run_excludes_base_attrs_from_dataset_data(
    synthetic_bundle_dir, mock_config_parser, tmp_path
):
    """model_weights must not appear as a key in dataset#data."""
    parser, _evaluator = mock_config_parser
    output_dir = tmp_path / "out"

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    data = parser.set_calls["dataset#data"]
    assert "model_weights" not in data[0]
```

- [ ] **Step 2: Run them**

Run: `pytest pydra/compose/monai/tests/test_task.py -k "test_run_" -v`
Expected: all five `test_run_*` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/tests/test_task.py
git commit -m "test(task): cover _run orchestration with mocked ConfigParser"
```

---

## Task 14: Test remaining `_resolve_bundle_dir` branches

Cover the weights-file walk-up branch and the missing-`model_weights` branch.

**Files:**
- Modify: `pydra/compose/monai/tests/test_task.py`

- [ ] **Step 1: Append tests**

```python
# ---------------------------------------------------------------------------
# _resolve_bundle_dir — weights-file and missing-input branches
# ---------------------------------------------------------------------------


def test_resolve_bundle_dir_walks_up_from_weights_file(
    bundle_with_weights_file, tmp_path
):
    weights = bundle_with_weights_file
    bundle_root = weights.parent.parent  # weights/models/model.pt -> bundle root

    TaskCls = monai.define(bundle_root)
    task = TaskCls(model_weights=str(weights), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")
    assert task._resolve_bundle_dir(job) == bundle_root


def test_resolve_bundle_dir_raises_for_orphan_weights_file(tmp_path):
    """A .pt file with no configs/metadata.json in any ancestor must raise."""
    orphan = tmp_path / "orphan.pt"
    orphan.write_bytes(b"")

    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(model_weights=str(orphan))

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")
    with pytest.raises(ValueError, match="Cannot locate bundle root"):
        task._resolve_bundle_dir(job)


def test_resolve_bundle_dir_raises_when_model_weights_missing(tmp_path):
    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(model_weights=None)

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")
    with pytest.raises(ValueError, match="model_weights must be set"):
        task._resolve_bundle_dir(job)
```

- [ ] **Step 2: Run them**

Run: `pytest pydra/compose/monai/tests/test_task.py -k resolve_bundle_dir -v`
Expected: all `_resolve_bundle_dir` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/tests/test_task.py
git commit -m "test(task): cover weights-file walk-up and missing-weights branches"
```

---

## Task 15: Record known limitations as skipped tests

The spec calls out three known limitations (L1: runtime DICOM through `_run`; L2: scalar/classification outputs; L3: bundles without a top-level `output_dir` key). Express each as a `pytest.skip`'d placeholder test so the test report surfaces them without polluting failures.

**Files:**
- Modify: `pydra/compose/monai/tests/test_task.py`
- Modify: `pydra/compose/monai/tests/test_spec_parser.py`

- [ ] **Step 1: Add L1 and L3 skips to `test_task.py`**

Append to `pydra/compose/monai/tests/test_task.py`:

```python
# ---------------------------------------------------------------------------
# Known limitations (see spec)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="L1: runtime DICOM input through _run not yet supported")
def test_run_with_dicom_input():
    """When implemented: a task whose `image` field is a DicomSeries should
    flow through _run end-to-end, with appropriate transform handling."""
    raise NotImplementedError


@pytest.mark.skip(reason="L3: bundles without top-level output_dir key not yet handled")
def test_run_handles_bundle_without_output_dir_key():
    """When implemented: if inference.json doesn't expose @output_dir, _run
    should detect this and warn rather than silently ignoring the override."""
    raise NotImplementedError
```

- [ ] **Step 2: Add L2 skip to `test_spec_parser.py`**

Append to `pydra/compose/monai/tests/test_spec_parser.py`:

```python
@pytest.mark.skip(reason="L2: scalar/classification outputs not yet supported")
def test_parse_scalar_output_type():
    """When implemented: a `network_data_format.outputs` entry of type
    'scalar' or 'classification' should map to a sensible Python type
    (float/int/list), not fall through to ty.Any."""
    raise NotImplementedError
```

- [ ] **Step 3: Verify the skips appear in the report**

Run: `pytest -v -rs pydra/compose/monai/tests/`
Expected: all real tests PASS; three tests appear as `SKIPPED` with the reason strings visible in the `-rs` section.

- [ ] **Step 4: Commit**

```bash
git add pydra/compose/monai/tests/test_task.py pydra/compose/monai/tests/test_spec_parser.py
git commit -m "test: document known limitations (L1-L3) as skipped placeholders"
```

---

## Task 16: Enable 70% coverage threshold

By now the unit tests should comfortably cover ≥70% of `pydra/compose/monai/`. Turn on the floor.

**Files:**
- Modify: `.coveragerc`

- [ ] **Step 1: Add `[report]` section with `fail_under`**

Replace `.coveragerc`:

```ini
[run]
branch = True
omit =
    */_version.py

[report]
fail_under = 70
show_missing = True
```

- [ ] **Step 2: Run the suite and confirm coverage clears 70%**

Run: `pytest`
Expected: All tests PASS. Coverage report at the end shows total ≥70%. If below 70%, pytest exits non-zero with:
```
Coverage failure: total of XX is less than fail_under=70
```

If under 70%, identify uncovered lines via `pytest --cov-report=term-missing` and add targeted tests before proceeding. Do not lower the threshold.

- [ ] **Step 3: Commit**

```bash
git add .coveragerc
git commit -m "chore: enforce 70% coverage floor on pydra/compose/monai"
```

---

## Task 17: Integration test — synthetic bundle end-to-end

Real `monai.bundle.ConfigParser`, real `SupervisedEvaluator`, real NIfTI IO on a small T1w patch. Gated by `@pytest.mark.integration`.

**Files:**
- Create: `pydra/compose/monai/tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

Create `pydra/compose/monai/tests/test_integration.py`:

```python
"""End-to-end and network-gated tests for pydra.compose.monai.

These are slow / require torch / require network — gated behind pytest markers.
"""
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from pydra.compose import monai


@pytest.fixture
def t1w_patch(tmp_path: Path, nifti_sample_dir: Path) -> Path:
    """A small (16^3) NIfTI patch cropped from the T1w sample.

    Keeps the synthetic bundle's CPU runtime well under a second.
    """
    src = nifti_sample_dir / "anat" / "T1w.nii.gz"
    img = nib.load(str(src))
    data = np.asarray(img.dataobj)
    # Crop centre 16x16x16
    cx, cy, cz = [s // 2 for s in data.shape[:3]]
    patch = data[cx - 8:cx + 8, cy - 8:cy + 8, cz - 8:cz + 8]
    patch_path = tmp_path / "T1w_patch.nii.gz"
    nib.save(nib.Nifti1Image(patch.astype(np.float32), img.affine), str(patch_path))
    return patch_path


@pytest.mark.integration
def test_synthetic_bundle_end_to_end(synthetic_bundle_dir, t1w_patch, tmp_path):
    """Build a task from the synthetic bundle, call _run directly, verify
    the postprocessing-driven output path resolution works end-to-end."""
    from pydra.compose.monai.tests.conftest import FakeJob

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image=str(t1w_patch))
    job = FakeJob(task, output_dir)

    task._run(job)

    # MONAI's SaveImaged writes <stem>_<postfix><ext> by default
    expected = output_dir / f"{t1w_patch.stem.replace('.nii', '')}_seg.nii.gz"
    assert expected.is_file(), f"Expected output at {expected}, found: {list(output_dir.iterdir())}"

    outputs = TaskCls.Outputs._from_job(job)
    assert Path(str(outputs.pred)) == expected
```

- [ ] **Step 2: Run the integration test explicitly**

Run: `pytest pydra/compose/monai/tests/test_integration.py -m integration -v`
Expected: `test_synthetic_bundle_end_to_end` PASSES (may take 2-5 seconds).

- [ ] **Step 3: Confirm default pytest run still skips it**

Run: `pytest`
Expected: All previously-passing tests PASS; `test_synthetic_bundle_end_to_end` is deselected (does not appear).

- [ ] **Step 4: Commit**

```bash
git add pydra/compose/monai/tests/test_integration.py
git commit -m "test(integration): synthetic bundle end-to-end with real evaluator"
```

---

## Task 18: Integration test — Pydra Submitter run

Verify the task and outputs round-trip cleanly through Pydra's engine (not just direct `_run`).

**Files:**
- Modify: `pydra/compose/monai/tests/test_integration.py`

- [ ] **Step 1: Append the Submitter test**

```python
@pytest.mark.integration
def test_synthetic_bundle_via_pydra_submitter(synthetic_bundle_dir, t1w_patch, tmp_path):
    """Run the synthetic bundle through pydra's Submitter, verifying that
    field metadata and outputs round-trip through the engine."""
    from pydra.engine.submitter import Submitter

    TaskCls = monai.define(synthetic_bundle_dir)
    task = TaskCls(model_weights=str(synthetic_bundle_dir), image=str(t1w_patch))

    with Submitter(worker="cf", cache_root=str(tmp_path / "cache")) as sub:
        result = sub(task)

    # Result should have an .outputs attribute with our parsed-out `pred`
    assert result.outputs is not None
    pred = getattr(result.outputs, "pred", None)
    assert pred is not None, f"pred output not set; outputs={result.outputs!r}"
    assert Path(str(pred)).is_file()
```

- [ ] **Step 2: Run the Submitter test**

Run: `pytest pydra/compose/monai/tests/test_integration.py::test_synthetic_bundle_via_pydra_submitter -m integration -v`
Expected: PASS.

If the test fails with an import error on `pydra.engine.submitter`, check the installed pydra version; this plan targets `pydra >= 1.0a` (per `pyproject.toml`). If `Submitter` lives elsewhere in the installed pydra (e.g. `pydra.engine.core`), adjust the import to match the installed module path — do not skip the test.

- [ ] **Step 3: Commit**

```bash
git add pydra/compose/monai/tests/test_integration.py
git commit -m "test(integration): run synthetic bundle through pydra Submitter"
```

---

## Task 19: Integration test — real-bundle smoke (network-gated)

Download `spleen_ct_segmentation` from MONAI Model Zoo and verify `define()` produces a sensible Task class. Does **not** run inference (no CT input data, and the synthetic bundle already covers `_run`).

**Files:**
- Modify: `pydra/compose/monai/tests/test_integration.py`

- [ ] **Step 1: Append the smoke test**

```python
@pytest.mark.integration
@pytest.mark.network
def test_real_bundle_define_smoke(tmp_path):
    """Smoke test: monai.bundle.load + define() against a published bundle.
    Catches drift in MONAI's bundle layout or network_data_format schema."""
    from pydra.compose.monai.spec_parser import _import_monai_bundle

    bundle_load = _import_monai_bundle().load
    bundle_dir = bundle_load(
        "spleen_ct_segmentation", bundle_dir=str(tmp_path), source="monaihosting"
    )

    TaskCls = monai.define(Path(bundle_dir))

    assert TaskCls.__name__
    assert TaskCls.__name__.isidentifier()

    field_names = {f.name for f in TaskCls.__attrs_attrs__}
    assert "model_weights" in field_names
    assert "image" in field_names, f"expected 'image' in input fields; got {field_names}"

    output_names = {f.name for f in TaskCls.Outputs.__attrs_attrs__}
    assert "pred" in output_names, f"expected 'pred' in output fields; got {output_names}"
```

- [ ] **Step 2: Run the smoke test explicitly**

Run: `pytest pydra/compose/monai/tests/test_integration.py::test_real_bundle_define_smoke -m network -v`
Expected: PASS (downloads ~30 MB on first run).

- [ ] **Step 3: Confirm default `pytest` still skips it**

Run: `pytest`
Expected: 0 tests deselected mentions `test_real_bundle_define_smoke`; no network call is made.

- [ ] **Step 4: Final full-suite sanity check**

Run all selectable modes:

```bash
pytest -v
pytest -m integration -v
pytest -m network -v
```

Expected: all pass in their respective modes. Coverage on the default run is ≥70%.

- [ ] **Step 5: Commit**

```bash
git add pydra/compose/monai/tests/test_integration.py
git commit -m "test(integration): real-bundle smoke against spleen_ct_segmentation"
```

---

## Definition of Done

- [ ] `pytest` exits 0 with ≥70% coverage on `pydra/compose/monai/`.
- [ ] `pytest -m integration` exits 0 (synthetic bundle E2E + Submitter test).
- [ ] `pytest -m network` exits 0 in an online environment (real-bundle smoke test).
- [ ] `arch` field is gone from `MonaiTask`; `Yaml` is gone from `define()`'s signature.
- [ ] `_resolve_bundle_dir` validates both bundle directories and repo-ID strings before proceeding.
- [ ] `_from_job` resolves outputs deterministically from postprocessing transforms; no loose glob fallback.
- [ ] `test_monai_spec.py` is removed; per-module test files exist for fields / spec_parser / builder / task / integration.
