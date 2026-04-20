"""Tests for parsing MONAI bundle metadata into Pydra field definitions."""
import json
import typing as ty
import pytest
from pathlib import Path
from fileformats.medimage import NiftiGzX
from pydra.compose import monai
from pydra.compose.monai.spec_parser import parse_monai_spec, name_from_spec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
# parse_monai_spec tests
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
# name_from_spec tests
# ---------------------------------------------------------------------------


def test_name_from_spec_uses_metadata_name(metadata_json: Path):
    assert name_from_spec(metadata_json) == "WholeBrainSegUnest"


def test_name_from_spec_uses_dir_name(tmp_path: Path):
    # No metadata, just a bare directory
    name = name_from_spec(tmp_path)
    assert name  # non-empty
    assert name.isidentifier()


# ---------------------------------------------------------------------------
# monai.define integration tests
# ---------------------------------------------------------------------------


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
    # path is stored inside the pydra field spec under __PYDRA_METADATA__
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


def test_arg_path_attribute_direct():
    """arg and out accept a path kwarg and store it."""
    a = monai.arg(name="T1w", type=NiftiGzX, path="anat/T1w")
    assert a.path == "anat/T1w"

    o = monai.out(name="mask", type=NiftiGzX, path="anat/mask")
    assert o.path == "anat/mask"
