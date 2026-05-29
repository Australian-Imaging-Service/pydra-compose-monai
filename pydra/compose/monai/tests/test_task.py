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
