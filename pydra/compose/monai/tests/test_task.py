"""Tests for MonaiTask._run, _resolve_bundle_dir, and MonaiOutputs._from_job."""
import json
import pytest
from pathlib import Path
from pydra.compose import monai
from pydra.compose.monai.task import MonaiTask


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
    task = TaskCls(model_weights=str(synthetic_bundle_dir))
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
    task = TaskCls(model_weights=str(tmp_path))
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
    task = TaskCls(model_weights=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
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
    task = TaskCls(model_weights=str(bundle), image=str(tmp_path / "T1w.nii.gz"))
    job = FakeJob(task, output_dir)

    outputs = TaskCls.Outputs._from_job(job)
    val = getattr(outputs, "pred", None)
    # val may be None, attrs.NOTHING (field unset), or a Path — all three
    # are acceptable "unset" for this test; we only reject a stray Path match.
    import attrs as _attrs
    is_unset = val is None or val is _attrs.NOTHING
    assert is_unset or "stray" in Path(str(val)).name


# ---------------------------------------------------------------------------
# _run orchestration (mocked evaluator)
# ---------------------------------------------------------------------------


def test_run_loads_metadata_and_inference_configs(
    mock_config_parser_with_task, tmp_path
):
    bundle, TaskCls, parser, evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(model_weights=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    parser.read_meta.assert_called_once_with(
        str(bundle / "configs" / "metadata.json")
    )
    parser.load_config_file.assert_called_once_with(
        str(bundle / "configs" / "inference.json")
    )


def test_run_sets_dataset_data_from_inputs(
    mock_config_parser_with_task, tmp_path
):
    bundle, TaskCls, parser, _evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(
        model_weights=str(bundle),
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

    task = TaskCls(model_weights=str(bundle), image="dummy.nii.gz")

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

    task = TaskCls(model_weights=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    evaluator.run.assert_called_once()


def test_run_excludes_base_attrs_from_dataset_data(
    mock_config_parser_with_task, tmp_path
):
    """model_weights must not appear as a key in dataset#data."""
    bundle, TaskCls, parser, _evaluator = mock_config_parser_with_task
    output_dir = tmp_path / "out"

    task = TaskCls(model_weights=str(bundle), image="dummy.nii.gz")

    from pydra.compose.monai.tests.conftest import FakeJob

    job = FakeJob(task, output_dir)
    task._run(job)

    data = parser.set_calls["dataset#data"]
    assert "model_weights" not in data[0]
