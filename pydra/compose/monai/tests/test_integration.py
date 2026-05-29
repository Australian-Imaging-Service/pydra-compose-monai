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
    Copies the source JSON sidecar so that NiftiGzX validation passes.
    """
    import shutil

    src = nifti_sample_dir / "anat" / "T1w.nii.gz"
    img = nib.load(str(src))
    data = np.asarray(img.dataobj)
    # Crop centre 16x16x16
    cx, cy, cz = [s // 2 for s in data.shape[:3]]
    patch = data[cx - 8:cx + 8, cy - 8:cy + 8, cz - 8:cz + 8]
    patch_path = tmp_path / "T1w_patch.nii.gz"
    nib.save(nib.Nifti1Image(patch.astype(np.float32), img.affine), str(patch_path))
    # Copy JSON sidecar so NiftiGzX (fileformats) validation passes
    src_json = nifti_sample_dir / "anat" / "T1w.json"
    if src_json.is_file():
        shutil.copy(str(src_json), str(tmp_path / "T1w_patch.json"))
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
