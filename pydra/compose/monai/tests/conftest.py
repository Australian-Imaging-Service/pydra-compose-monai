"""Shared fixtures for pydra.compose.monai tests."""
import json
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock
import pytest


# ---------------------------------------------------------------------------
# Synthetic bundle factory
# ---------------------------------------------------------------------------


DEFAULT_METADATA = {
    "name": "Synthetic Test Bundle",
    "network_data_format": {
        "inputs": {
            "image": {
                "type": "image",
                "modality": "MRI",
                "num_channels": 1,
                "spatial_shape": [16, 16, 16],
            }
        },
        "outputs": {
            "pred": {
                "type": "image",
                "format": "segmentation",
                "num_channels": 1,
                "spatial_shape": [16, 16, 16],
            }
        },
    },
}


DEFAULT_INFERENCE = {
    "imports": ["$import torch"],
    "device": "cpu",
    "output_dir": "",
    "network_def": {
        "_target_": "monai.networks.nets.UNet",
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": [4, 8],
        "strides": [2],
    },
    "preprocessing": {
        "_target_": "Compose",
        "transforms": [
            {"_target_": "LoadImaged", "keys": ["image"]},
            {"_target_": "EnsureChannelFirstd", "keys": ["image"]},
        ],
    },
    "dataset": {
        "_target_": "Dataset",
        "data": [],
        "transform": "@preprocessing",
    },
    "dataloader": {
        "_target_": "DataLoader",
        "dataset": "@dataset",
        "batch_size": 1,
    },
    "inferer": {"_target_": "SimpleInferer"},
    "postprocessing": {
        "_target_": "Compose",
        "transforms": [
            {
                "_target_": "SaveImaged",
                "keys": ["pred"],
                "output_dir": "@output_dir",
                "output_postfix": "seg",
                "separate_folder": False,
            }
        ],
    },
    "evaluator": {
        "_target_": "SupervisedEvaluator",
        "device": "@device",
        "val_data_loader": "@dataloader",
        "network": "@network_def",
        "inferer": "@inferer",
        "postprocessing": "@postprocessing",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge — `override` wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


@pytest.fixture
def make_synthetic_bundle(tmp_path: Path) -> Callable[..., Path]:
    """Factory: build a minimal MONAI bundle dir; override metadata/inference."""

    def _make(
        metadata_overrides: dict | None = None,
        inference_overrides: dict | None = None,
        subdir: str = "synthetic_bundle",
    ) -> Path:
        bundle = tmp_path / subdir
        configs = bundle / "configs"
        configs.mkdir(parents=True, exist_ok=True)

        metadata = _deep_merge(DEFAULT_METADATA, metadata_overrides or {})
        inference = _deep_merge(DEFAULT_INFERENCE, inference_overrides or {})

        (configs / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (configs / "inference.json").write_text(json.dumps(inference, indent=2))
        return bundle

    return _make


@pytest.fixture
def synthetic_bundle_dir(make_synthetic_bundle) -> Path:
    """Default synthetic bundle — image input, pred segmentation output."""
    return make_synthetic_bundle()


@pytest.fixture
def bundle_with_weights_file(make_synthetic_bundle) -> Path:
    """Synthetic bundle with a placeholder weights file at models/model.pt.

    Returns the path to the weights file (caller can walk up for the bundle root).
    """
    bundle = make_synthetic_bundle(subdir="bundle_with_weights")
    models = bundle / "models"
    models.mkdir()
    weights = models / "model.pt"
    weights.write_bytes(b"")  # empty placeholder; we never load it
    return weights


# ---------------------------------------------------------------------------
# Mock ConfigParser fixture for unit tests of _run
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_parser(monkeypatch):
    """Patch monai.bundle.ConfigParser so unit tests don't instantiate networks.

    Returns (parser_mock, evaluator_mock). The parser supports __setitem__ so
    tests can assert on `parser['dataset#data'] = ...`. The evaluator's
    `.run()` is a MagicMock for `assert_called_once()`.
    """
    parser = MagicMock(name="ConfigParser_instance")
    evaluator = MagicMock(name="evaluator")
    parser.get_parsed_content.return_value = evaluator

    # __setitem__ records calls so tests can assert on them
    parser.set_calls = {}

    def record_setitem(key, value):
        parser.set_calls[key] = value

    parser.__setitem__.side_effect = record_setitem

    # Patch the ConfigParser class used inside _run, which goes through
    # _import_monai_bundle().ConfigParser.
    import monai.bundle as monai_bundle

    monkeypatch.setattr(monai_bundle, "ConfigParser", MagicMock(return_value=parser))

    return parser, evaluator


# ---------------------------------------------------------------------------
# Stand-in for pydra.engine.Job in _run unit tests
# ---------------------------------------------------------------------------


class FakeJob:
    """Minimal stand-in for pydra's Job, sufficient for _run / _from_job."""

    def __init__(self, task, output_dir: Path, cache_dir: Path | None = None):
        self.task = task
        self.output_dir = str(output_dir)
        self.cache_dir = cache_dir


@pytest.fixture
def fake_job():
    """Factory for FakeJob instances bound to a task and output dir."""
    return FakeJob
