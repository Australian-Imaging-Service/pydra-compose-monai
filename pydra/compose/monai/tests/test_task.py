"""Tests for MonaiTask._run, _resolve_bundle_dir, and MonaiOutputs._from_job."""
import json
import pytest
from pathlib import Path
from pydra.compose import monai
from pydra.compose.monai.task import MonaiTask
from pydra.utils import get_fields


# ---------------------------------------------------------------------------
# _resolve_bundle_dir
# ---------------------------------------------------------------------------


def test_resolve_bundle_dir_accepts_dir_with_configs(
    synthetic_bundle_dir: Path, tmp_path: Path
):
    # Build a TaskCls with no image input so we can instantiate it without a
    # real NiftiGzX file (this test is about _resolve_bundle_dir, not inputs).
    no_inputs_meta = tmp_path / "meta.json"
    no_inputs_meta.write_text(
        json.dumps({"name": "synthetic_bundle_dir", "network_data_format": {"inputs": {}, "outputs": {}}})
    )
    TaskCls = monai.define(no_inputs_meta)
    task = TaskCls(bundle=str(synthetic_bundle_dir))
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
    # even though bundle points at the bare dir.
    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(bundle=str(bare))
    job = FakeJob(task, tmp_path / "out")

    with pytest.raises(ValueError, match="configs/metadata.json"):
        task._resolve_bundle_dir(job)


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

    task = TaskCls(bundle="spleen_ct_segmentation")
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

    task = TaskCls(bundle="/tmp/nonexistent/bundle.pt")
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

    task = TaskCls(bundle="my_model.pt")
    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")

    with pytest.raises(ValueError, match="not a valid MONAI bundle"):
        task._resolve_bundle_dir(job)


# ---------------------------------------------------------------------------
# _from_job — base output field handling
# ---------------------------------------------------------------------------


def test_from_job_does_not_overwrite_base_output_fields(tmp_path, monkeypatch):
    """R2: fields explicitly listed in BASE_OUTPUT_ATTRS must not be overwritten
    by stray files in the output dir, even if a bundle declares an output with the
    same name (e.g. 'stdout') and a matching file exists in the output dir."""
    from pydra.compose.monai.tests.conftest import FakeJob

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Drop files whose names exactly match base output field names
    (output_dir / "stdout.log").write_text("noise")
    (output_dir / "stderr.txt").write_text("noise")

    # Build a bundle with an output field literally named 'stdout' so that the
    # glob f"{field.name}.*" == "stdout.*" would match stdout.log on unpatched code.
    bundle_meta = tmp_path / "configs" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text(
        json.dumps({
            "name": "stdout_test_bundle",
            "network_data_format": {
                "inputs": {},
                "outputs": {
                    "stdout": {"type": "image"},
                },
            },
        })
    )
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(bundle=str(tmp_path))
    job = FakeJob(task, output_dir)

    OutputsCls = TaskCls.Outputs

    # On unpatched code the field 'stdout' is an attrs field and the glob
    # 'stdout.*' matches stdout.log — so _from_job would set outputs.stdout to
    # a Path object pointing at the noise file.
    # After the fix (BASE_OUTPUT_ATTRS skip), the field must be left at its
    # default value (attrs.NOTHING or None), not overwritten by the stray file.
    outputs = OutputsCls._from_job(job)

    import attrs as _attrs
    stdout_field_names = {f.name for f in _attrs.fields(OutputsCls)} & {"stdout", "stderr", "return_code"}
    for name in stdout_field_names:
        val = getattr(outputs, name, None)
        # Must NOT be a Path (i.e. must not have been set from the stray log file)
        assert not isinstance(val, Path), (
            f"_from_job overwrote {name!r} with {val!r} from a stray file"
        )


# ---------------------------------------------------------------------------
# _from_job — R1 postprocessing-driven path resolution
# ---------------------------------------------------------------------------


def test_from_job_resolves_output_from_save_transform(
    make_synthetic_bundle, tmp_path
):
    """R1: output path is constructed from SaveImaged postfix/ext and the input
    image filename, not from a loose glob match.

    Adaptation: image field is typed as ty.Any via metadata_overrides so that
    task construction accepts a plain string path without fileformats validation.
    """
    from pydra.compose.monai.tests.conftest import FakeJob

    # Use metadata_overrides to give the image field type ty.Any, avoiding
    # NiftiGzX file-existence/format validation during task construction.
    # We must clear "modality" as well (set to ""), because _map_type maps
    # modality="MRI" to NiftiGzX regardless of "type"; deep-merge preserves it.
    bundle = make_synthetic_bundle(
        metadata_overrides={
            "network_data_format": {
                "inputs": {"image": {"type": "generic", "modality": ""}},
            }
        }
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Simulate what MONAI's SaveImaged would write: T1w_seg.nii.gz
    # (DEFAULT_INFERENCE has SaveImaged with output_postfix="seg", no output_ext -> .nii.gz)
    expected = output_dir / "T1w_seg.nii.gz"
    expected.write_bytes(b"fake nifti")

    TaskCls = monai.define(bundle)
    task = TaskCls(bundle=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    assert Path(str(outputs.pred)) == expected


def test_from_job_does_not_match_unrelated_files(
    make_synthetic_bundle, tmp_path
):
    """R1: a file whose name *contains* the field name but doesn't match the
    SaveImaged postfix must NOT be picked up.

    Adaptation: image field is typed as ty.Any via metadata_overrides.
    """
    from pydra.compose.monai.tests.conftest import FakeJob

    bundle = make_synthetic_bundle(
        metadata_overrides={
            "network_data_format": {
                "inputs": {"image": {"type": "generic", "modality": ""}},
            }
        }
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Drop a misleading file: contains "pred" but is not the SaveImaged output
    (output_dir / "unrelated_pred_data.csv").write_text("noise")

    TaskCls = monai.define(bundle)
    task = TaskCls(bundle=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    val = getattr(outputs, "pred", None)
    import attrs as _attrs
    assert val is None or val is _attrs.NOTHING, (
        f"pred should be unset when no SaveImaged output file exists in output_dir; got {val!r}"
    )


def test_from_job_leaves_field_unset_when_save_transform_missing(
    make_synthetic_bundle, tmp_path
):
    """R1: if a bundle's postprocessing doesn't write a given output, the field
    is left unset (no glob fallback).

    Adaptation: image field is typed as ty.Any via metadata_overrides.
    """
    from pydra.compose.monai.tests.conftest import FakeJob

    # Bundle whose postprocessing writes no images at all; image is ty.Any
    bundle = make_synthetic_bundle(
        metadata_overrides={
            "network_data_format": {
                "inputs": {"image": {"type": "generic", "modality": ""}},
            }
        },
        inference_overrides={"postprocessing": {"_target_": "Compose", "transforms": []}},
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "pred.nii.gz").write_bytes(b"stray match")

    TaskCls = monai.define(bundle)
    task = TaskCls(bundle=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    val = getattr(outputs, "pred", None)
    import attrs as _attrs
    assert val is None or val is _attrs.NOTHING, (
        f"pred should be unset when no SaveImaged transform writes it; got {val!r}"
    )


# ---------------------------------------------------------------------------
# _run orchestration (mocked evaluator)
# ---------------------------------------------------------------------------


def test_run_loads_metadata_and_inference_configs(
    mock_config_parser_with_task, tmp_path
):
    bundle, TaskCls, parser, evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(bundle=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    parser.read_meta.assert_called_once_with(
        str(bundle / "configs" / "metadata.json")
    )
    parser.read_config.assert_called_once_with(
        str(bundle / "configs" / "inference.json")
    )


def test_run_sets_dataset_data_from_inputs(
    mock_config_parser_with_task, tmp_path
):
    bundle, TaskCls, parser, _evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(
        bundle=str(bundle),
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
    mock_config_parser_with_task, tmp_path
):
    bundle, TaskCls, parser, _evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(bundle=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    assert parser.set_calls.get("output_dir") == str(output_dir)
    assert output_dir.exists()


def test_run_calls_evaluator_run_once(
    mock_config_parser_with_task, tmp_path
):
    bundle, TaskCls, _parser, evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(bundle=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    evaluator.run.assert_called_once()


def test_run_excludes_base_attrs_from_dataset_data(
    mock_config_parser_with_task, tmp_path
):
    """bundle must not appear as a key in dataset#data."""
    bundle, TaskCls, parser, _evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(bundle=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    data = parser.set_calls["dataset#data"]
    assert "bundle" not in data[0]


# ---------------------------------------------------------------------------
# _resolve_bundle_dir — weights-file and missing-input branches
# ---------------------------------------------------------------------------


def test_resolve_bundle_dir_walks_up_from_weights_file(
    make_synthetic_bundle, tmp_path
):
    """A .pt file inside a bundle tree: walk-up must return the bundle root.

    Adaptation: the default bundle metadata maps image to NiftiGzX (modality=MRI).
    We use metadata_overrides to make image type=generic/modality="" so the task
    can be constructed with a plain string path without fileformats validation.
    """
    from pydra.compose.monai.tests.conftest import FakeJob

    bundle = make_synthetic_bundle(
        metadata_overrides={
            "network_data_format": {
                "inputs": {"image": {"type": "generic", "modality": ""}},
            }
        },
        subdir="bundle_with_weights",
    )
    # Create a weights file inside the bundle tree (bundle/models/model.pt)
    models = bundle / "models"
    models.mkdir()
    weights = models / "model.pt"
    weights.write_bytes(b"")

    bundle_root = weights.parent.parent  # models/model.pt -> bundle root

    TaskCls = monai.define(bundle)
    task = TaskCls(bundle=str(weights), image="dummy.nii.gz")

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
    task = TaskCls(bundle=str(orphan))

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")
    with pytest.raises(ValueError, match="Cannot locate bundle root"):
        task._resolve_bundle_dir(job)


def test_resolve_bundle_dir_raises_when_bundle_missing(tmp_path):
    """bundle=None must raise with an actionable message."""
    bundle_meta = tmp_path / "tmpconfig" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text('{"name": "x", "network_data_format": {"inputs": {}, "outputs": {}}}')
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(bundle=None)

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, tmp_path / "out")
    with pytest.raises(ValueError, match="bundle must be set"):
        task._resolve_bundle_dir(job)


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


# ---------------------------------------------------------------------------
# _extensions_for / _default_ext_for helpers (unit tests)
# ---------------------------------------------------------------------------


def test_extensions_for_niftgzx():
    """_extensions_for returns the NiftiGzX canonical extension."""
    from fileformats.medimage import NiftiGzX
    from pydra.compose.monai.task import _extensions_for

    exts = _extensions_for(NiftiGzX)
    assert ".nii.gz" in exts
    assert all(isinstance(e, str) for e in exts)


def test_extensions_for_dicomseries_returns_empty():
    """DicomSeries has no single-file extension; _extensions_for returns ()."""
    from fileformats.medimage import DicomSeries
    from pydra.compose.monai.task import _extensions_for

    exts = _extensions_for(DicomSeries)
    assert exts == ()


def test_extensions_for_any_returns_empty():
    """ty.Any is not a FileSet subclass; _extensions_for returns ()."""
    import typing as ty
    from pydra.compose.monai.task import _extensions_for

    assert _extensions_for(ty.Any) == ()


def test_default_ext_for_niftigzx():
    """_default_ext_for returns '.nii.gz' for NiftiGzX."""
    from fileformats.medimage import NiftiGzX
    from pydra.compose.monai.task import _default_ext_for

    assert _default_ext_for(NiftiGzX) == ".nii.gz"


def test_default_ext_for_unknown_falls_back():
    """_default_ext_for falls back to '.nii.gz' for types with no file extension."""
    import typing as ty
    from fileformats.medimage import DicomSeries
    from pydra.compose.monai.task import _default_ext_for

    # Both an unknown type and a dir-based type fall back to .nii.gz
    assert _default_ext_for(ty.Any) == ".nii.gz"
    assert _default_ext_for(DicomSeries) == ".nii.gz"


# ---------------------------------------------------------------------------
# _first_input_stem — fileformats-driven extension stripping
# ---------------------------------------------------------------------------


def test_first_input_stem_uses_fileformats_extensions(tmp_path):
    """_first_input_stem strips the NiftiGzX extension from the input path.

    The field type is NiftiGzX (MRI modality), so _extensions_for returns
    ['.nii.gz'] and the compound extension is stripped correctly — not via
    Path.stem (which would give 'T1w.nii') but via the type-driven lookup.

    We verify this by confirming that Path.stem alone would give the *wrong*
    answer ('T1w.nii') while the type-driven code gives the correct one ('T1w').
    """
    from pydra.compose.monai.task import _first_input_stem, _extensions_for
    from pathlib import Path as _Path

    # Confirm Path.stem alone is insufficient for compound extensions.
    assert _Path("T1w.nii.gz").stem == "T1w.nii"

    # Build a TaskCls whose image field is typed NiftiGzX (the default for MRI)
    bundle_meta = tmp_path / "configs" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text(
        json.dumps({
            "name": "nifti_stem_test",
            "network_data_format": {
                "inputs": {"image": {"type": "image", "modality": "MRI"}},
                "outputs": {},
            },
        })
    )
    TaskCls = monai.define(bundle_meta)

    # Verify the field is NiftiGzX-typed and carries the right extension.
    from fileformats.medimage import NiftiGzX
    image_field = next(f for f in get_fields(TaskCls) if f.name == "image")
    assert image_field.type is NiftiGzX
    assert ".nii.gz" in _extensions_for(NiftiGzX)

    # NiftiGzX validates file existence, magic number, and requires a BIDS JSON
    # sidecar (.json).  Create both files so pydra can coerce the path to NiftiGzX.
    import gzip as _gzip, io as _io
    buf = _io.BytesIO()
    with _gzip.open(buf, "wb") as f:
        f.write(b"")
    input_file = tmp_path / "T1w.nii.gz"
    input_file.write_bytes(buf.getvalue())
    (tmp_path / "T1w.json").write_text("{}")  # empty BIDS sidecar

    task = TaskCls(bundle=str(tmp_path), image=str(input_file))

    # _extensions_for drives the lookup, so the compound extension is stripped correctly.
    assert _first_input_stem(task) == "T1w"


def test_first_input_stem_fallback_for_unknown_extension(tmp_path):
    """_first_input_stem falls back to Path.stem for fields typed ty.Any."""
    from pydra.compose.monai.task import _first_input_stem

    # Build a TaskCls with a generic (ty.Any) image field
    bundle_meta = tmp_path / "configs" / "metadata.json"
    bundle_meta.parent.mkdir(parents=True)
    bundle_meta.write_text(
        json.dumps({
            "name": "generic_stem_test",
            "network_data_format": {
                "inputs": {"image": {"type": "generic", "modality": ""}},
                "outputs": {},
            },
        })
    )
    TaskCls = monai.define(bundle_meta)
    task = TaskCls(bundle=str(tmp_path), image="path/to/T1w.unknown")

    # Path.stem of "T1w.unknown" is "T1w"
    assert _first_input_stem(task) == "T1w"


# ---------------------------------------------------------------------------
# _from_job — output extension resolved from field type
# ---------------------------------------------------------------------------


def test_from_job_uses_field_type_for_output_ext(make_synthetic_bundle, tmp_path):
    """When inference.json omits output_ext, _from_job derives it from NiftiGzX.

    The SaveImaged transform in DEFAULT_INFERENCE has no explicit output_ext,
    so the default must come from the pred field's NiftiGzX type, not a
    hardcoded string.  We verify the mechanism by checking _default_ext_for
    returns the same extension that was used to locate the output file.
    """
    from pydra.compose.monai.tests.conftest import FakeJob
    from pydra.compose.monai.task import _default_ext_for
    from fileformats.medimage import NiftiGzX

    # Default synthetic bundle: pred output is NiftiGzX (format=segmentation)
    bundle = make_synthetic_bundle()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Verify that _default_ext_for(NiftiGzX) == ".nii.gz"
    assert _default_ext_for(NiftiGzX) == ".nii.gz"

    # The SaveImaged postfix is "seg" and there's no output_ext in DEFAULT_INFERENCE
    expected = output_dir / "T1w_seg.nii.gz"
    expected.write_bytes(b"fake nifti")

    TaskCls = monai.define(bundle)

    # Confirm that the pred output field is NiftiGzX-typed
    pred_field = next(f for f in get_fields(TaskCls.Outputs) if f.name == "pred")
    assert pred_field.type is NiftiGzX

    # NiftiGzX input field validates file existence, magic number, and requires a
    # BIDS JSON sidecar.  Create both files so pydra can coerce the path to NiftiGzX.
    import gzip as _gzip, io as _io
    buf = _io.BytesIO()
    with _gzip.open(buf, "wb") as f:
        f.write(b"")
    input_file = tmp_path / "T1w.nii.gz"
    input_file.write_bytes(buf.getvalue())
    (tmp_path / "T1w.json").write_text("{}")  # empty BIDS sidecar

    task = TaskCls(bundle=str(bundle), image=str(input_file))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    assert Path(str(outputs.pred)) == expected
