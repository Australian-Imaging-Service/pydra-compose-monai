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
