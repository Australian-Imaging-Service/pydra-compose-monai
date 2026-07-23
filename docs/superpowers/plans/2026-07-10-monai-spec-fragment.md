# pydra-compose-monai: spec_fragment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public `spec_fragment()` function that serializes a parsed MONAI bundle into a pipeline2app `sources`/`sinks`/`parameters` YAML fragment.

**Architecture:** Reuse the existing `parse_monai_spec()` (which already computes typed `arg`/`out` fields from a bundle's `metadata.json`) and serialize its output to plain dicts using fileformats `.mime_like` strings.

> **Revision (2026-07-24):** The originally-planned generic `BundleTask` (Task 2) was **dropped**. The raw `MonaiTask` base class has no usable fields and cannot be instantiated with `bundle=` — only a `define()`-built subclass can. Rather than ship a generic task built from a committed fixture (which would need maintaining against schema drift), the consuming repo (Plan B) generates a per-model `define()`-built class instead (Option A). This package therefore ships **only** `spec_fragment()`. The former Task 2 is removed; the former Task 3 regression check remains below.

**Tech Stack:** Python ≥3.11, attrs, pydra.compose.base, fileformats.medimage, pytest.

## Global Constraints

- `requires-python >=3.11` — use `X | Y` union syntax, not `Optional`.
- Public API is declared in `pydra/compose/monai/__init__.py:__all__` — new public names must be added there.
- Tests live in `pydra/compose/monai/tests/`, run with `pytest` (default addopts exclude `integration`/`network` markers). Reuse existing fixtures from `tests/conftest.py` (`make_synthetic_bundle`, `synthetic_bundle_dir`).
- fileformats types serialize via the `.mime_like` attribute (e.g. `NiftiGzX.mime_like == "medimage/nifti-gz-x"`). `ty.Any` has no `.mime_like`.
- Datatype strings in the fragment must be fileformats MIME-like strings.

---

### Task 1: `spec_fragment()` — serialize a parsed bundle to a spec dict

**Files:**
- Modify: `pydra/compose/monai/spec_parser.py` (add `spec_fragment` + `_datatype_str` helper at end of file)
- Modify: `pydra/compose/monai/__init__.py:16-20` (export `spec_fragment`)
- Test: `pydra/compose/monai/tests/test_spec_parser.py` (append tests)

**Interfaces:**
- Consumes: `parse_monai_spec(spec_path) -> tuple[dict[str, arg], dict[str, out]]` (existing, `spec_parser.py:41`).
- Produces:
  ```python
  def spec_fragment(spec_path: Path | str) -> dict[str, dict[str, dict]]:
      # returns {"sources": {name: {"datatype": str, "help": str, "path": str}},
      #          "sinks":   {name: {"datatype": str, "help": str, "path": str}},
      #          "parameters": {}}
  ```
  where `datatype` is the field type's `.mime_like`, or `"field/generic"` when the type is `ty.Any`.

- [ ] **Step 1: Write the failing test**

Append to `pydra/compose/monai/tests/test_spec_parser.py`:

```python
# ---------------------------------------------------------------------------
# spec_fragment
# ---------------------------------------------------------------------------


def test_spec_fragment_has_sources_and_sinks(metadata_json: Path):
    from pydra.compose.monai.spec_parser import spec_fragment

    frag = spec_fragment(metadata_json)
    assert set(frag) == {"sources", "sinks", "parameters"}
    assert "image" in frag["sources"]
    assert "pred" in frag["sinks"]


def test_spec_fragment_source_fields(metadata_json: Path):
    from pydra.compose.monai.spec_parser import spec_fragment

    frag = spec_fragment(metadata_json)
    image = frag["sources"]["image"]
    assert image["datatype"] == "medimage/nifti-gz-x"
    assert image["path"] == "network_data_format/inputs/image"
    assert "MRI" in image["help"]


def test_spec_fragment_sink_fields(metadata_json: Path):
    from pydra.compose.monai.spec_parser import spec_fragment

    frag = spec_fragment(metadata_json)
    pred = frag["sinks"]["pred"]
    assert pred["datatype"] == "medimage/nifti-gz-x"
    assert pred["path"] == "network_data_format/outputs/pred"


def test_spec_fragment_any_type_maps_to_generic(tmp_path: Path):
    from pydra.compose.monai.spec_parser import spec_fragment

    metadata = {
        "network_data_format": {
            "inputs": {"feat": {"type": "tensor", "format": "embedding"}},
            "outputs": {},
        }
    }
    p = tmp_path / "metadata.json"
    p.write_text(json.dumps(metadata))
    frag = spec_fragment(p)
    assert frag["sources"]["feat"]["datatype"] == "field/generic"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest pydra/compose/monai/tests/test_spec_parser.py -k spec_fragment -v`
Expected: FAIL with `ImportError: cannot import name 'spec_fragment'`

- [ ] **Step 3: Write minimal implementation**

Append to `pydra/compose/monai/spec_parser.py`:

```python
def spec_fragment(spec_path: Path | str) -> dict[str, dict[str, dict]]:
    """Serialize a MONAI bundle into a pipeline2app command fragment.

    Parses the bundle via :func:`parse_monai_spec` and emits a dict with
    ``sources`` (inputs), ``sinks`` (outputs) and ``parameters`` sections
    suitable for embedding in a pipeline2app / frametree image spec.

    Parameters
    ----------
    spec_path : Path | str
        Path to a MONAI bundle ``metadata.json`` or bundle root directory.

    Returns
    -------
    dict[str, dict[str, dict]]
        ``{"sources": {name: {"datatype", "help", "path"}},
           "sinks":   {name: {"datatype", "help", "path"}},
           "parameters": {}}``
    """
    parsed_inputs, parsed_outputs = parse_monai_spec(spec_path)

    sources = {
        name: {
            "datatype": _datatype_str(field.type),
            "help": field.help,
            "path": field.path,
        }
        for name, field in parsed_inputs.items()
    }
    sinks = {
        name: {
            "datatype": _datatype_str(field.type),
            "help": field.help,
            "path": field.path,
        }
        for name, field in parsed_outputs.items()
    }
    return {"sources": sources, "sinks": sinks, "parameters": {}}


def _datatype_str(field_type: type) -> str:
    """Return the fileformats MIME-like string for a field type.

    Falls back to ``"field/generic"`` when the type is ``ty.Any`` (or any
    type without a ``mime_like`` attribute).
    """
    return getattr(field_type, "mime_like", "field/generic")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest pydra/compose/monai/tests/test_spec_parser.py -k spec_fragment -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Export `spec_fragment` from the package**

In `pydra/compose/monai/__init__.py`, change:

```python
from .builder import define
from .fields import arg, out
from .task import MonaiTask as Task, MonaiOutputs as Outputs

__all__ = ["arg", "out", "define", "Task", "Outputs", "__version__"]
```

to:

```python
from .builder import define
from .fields import arg, out
from .spec_parser import spec_fragment
from .task import MonaiTask as Task, MonaiOutputs as Outputs

__all__ = ["arg", "out", "define", "spec_fragment", "Task", "Outputs", "__version__"]
```

- [ ] **Step 6: Verify the public import works**

Run: `.venv/bin/python -c "from pydra.compose.monai import spec_fragment; print(spec_fragment)"`
Expected: prints `<function spec_fragment at ...>`

- [ ] **Step 7: Commit**

```bash
git add pydra/compose/monai/spec_parser.py pydra/compose/monai/__init__.py pydra/compose/monai/tests/test_spec_parser.py
git commit -m "feat: add spec_fragment() to serialize a bundle into a pipeline2app command fragment

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Full test-suite regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the full (non-integration/network) suite**

Run: `.venv/bin/pytest`
Expected: all previously-passing tests still pass; the new `spec_fragment` tests pass; skipped/xfail counts unchanged from before this work.

- [ ] **Step 2: Confirm public API surface**

Run: `.venv/bin/python -c "import pydra.compose.monai as m; print(sorted(m.__all__))"`
Expected: list includes `spec_fragment`, alongside the pre-existing names (`Outputs`, `Task`, `arg`, `define`, `out`). Does NOT include `BundleTask`.

## Self-Review

- **Spec coverage:** Component 1 of the design, revised — pydra-compose-monai ships `spec_fragment()` only. Task 1 covers `spec_fragment`, Task 2 guards regressions. The generic `BundleTask` was dropped (see the Revision note at the top); the consuming repo generates a per-model `define()`-built class (Option A) instead. Complete.
- **Placeholder scan:** No TBD/TODO; every code step shows full code and exact commands.
- **Type consistency:** `spec_fragment` return shape (`sources`/`sinks`/`parameters`) is used identically in tests and impl; `_datatype_str` fallback string `"field/generic"` matches the test assertion.
- **Note for Plan B:** `spec_fragment` returns `sinks[name]["path"]` as the *bundle* metadata path (`network_data_format/outputs/pred`), NOT the frametree store path (e.g. `monai/pred`). Plan B's overlay/merge step is responsible for rewriting sink `path` values to store paths.
