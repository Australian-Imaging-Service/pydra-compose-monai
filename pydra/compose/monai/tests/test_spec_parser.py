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


def test_synthetic_bundle_fixture_creates_valid_bundle(synthetic_bundle_dir: Path):
    """Smoke test: the fixture writes a parseable metadata.json."""
    assert (synthetic_bundle_dir / "configs" / "metadata.json").exists()
    assert (synthetic_bundle_dir / "configs" / "inference.json").exists()
    parsed_inputs, parsed_outputs = parse_monai_spec(synthetic_bundle_dir)
    assert "image" in parsed_inputs
    assert "pred" in parsed_outputs


# ---------------------------------------------------------------------------
# Known limitations (see spec)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="L2: scalar/classification outputs not yet supported")
def test_parse_scalar_output_type():
    """When implemented: a `network_data_format.outputs` entry of type
    'scalar' or 'classification' should map to a sensible Python type
    (float/int/list), not fall through to ty.Any."""
    raise NotImplementedError
