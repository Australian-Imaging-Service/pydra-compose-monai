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
