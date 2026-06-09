"""Tests for the pydra.compose.monai define() builder."""
import json
import pytest
from pathlib import Path
from fileformats.medimage import NiftiGzX
from pydra.compose import monai
from pydra.utils import get_fields


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
    field_names = [f.name for f in get_fields(TaskCls)]
    assert "image" in field_names
    assert "model_weights" in field_names


def test_define_from_metadata_json_outputs_have_pred(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    output_field_names = [f.name for f in get_fields(TaskCls.Outputs)]
    assert "pred" in output_field_names


def test_define_preserves_path_on_arg(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    image_field = next(f for f in get_fields(TaskCls) if f.name == "image")
    assert image_field.path == "network_data_format/inputs/image"


def test_define_preserves_path_on_out(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    pred_field = next(
        f for f in get_fields(TaskCls.Outputs) if f.name == "pred"
    )
    assert pred_field.path == "network_data_format/outputs/pred"


def test_define_class_name_from_metadata(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    assert TaskCls.__name__ == "WholeBrainSegUnest"


def test_define_explicit_name_overrides_metadata(metadata_json: Path):
    TaskCls = monai.define(metadata_json, name="MyCustomTask")
    assert TaskCls.__name__ == "MyCustomTask"


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
    field_names = [f.name for f in get_fields(TaskCls)]
    assert "image" in field_names


def test_define_includes_base_attrs(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    field_names = {f.name for f in get_fields(TaskCls)}
    assert "model_weights" in field_names


def test_define_rejects_non_path_non_class():
    with pytest.raises(ValueError, match="must be a class or a str"):
        monai.define(42)


def test_define_image_input_has_nifti_type(metadata_json: Path):
    TaskCls = monai.define(metadata_json)
    image_field = next(f for f in get_fields(TaskCls) if f.name == "image")
    assert image_field.type is NiftiGzX


def test_define_does_not_include_arch_field(metadata_json: Path):
    """R5: `arch` is YAGNI and was removed."""
    TaskCls = monai.define(metadata_json)
    field_names = {f.name for f in get_fields(TaskCls)}
    assert "arch" not in field_names
