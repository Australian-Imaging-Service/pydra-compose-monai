"""Parse MONAI bundle metadata.json into Pydra arg/out field definitions."""

import json
import typing as ty
from pathlib import Path

from .fields import arg, out


def parse_monai_spec(
    spec_path: Path | str,
) -> tuple[dict[str, arg], dict[str, out]]:
    """Parse a MONAI bundle metadata.json and return input/output field dicts.

    Parameters
    ----------
    spec_path : Path | str
        Path to a MONAI bundle ``metadata.json`` file, or to the bundle root
        directory (in which case ``configs/metadata.json`` is loaded).

    Returns
    -------
    parsed_inputs : dict[str, arg]
        Mapping of field name → ``arg`` for each entry in
        ``network_data_format.inputs``.
    parsed_outputs : dict[str, out]
        Mapping of field name → ``out`` for each entry in
        ``network_data_format.outputs``.
    """
    spec_path = Path(spec_path)
    if spec_path.is_dir():
        spec_path = spec_path / "configs" / "metadata.json"

    with open(spec_path) as f:
        metadata = json.load(f)

    ndf = metadata.get("network_data_format", {})
    raw_inputs = ndf.get("inputs", {})
    raw_outputs = ndf.get("outputs", {})

    parsed_inputs: dict[str, arg] = {}
    for key, spec in raw_inputs.items():
        parsed_inputs[key] = arg(
            name=key,
            type=_map_type(spec),
            help=_input_help(spec),
            path=f"network_data_format/inputs/{key}",
        )

    parsed_outputs: dict[str, out] = {}
    for key, spec in raw_outputs.items():
        parsed_outputs[key] = out(
            name=key,
            type=_map_type(spec),
            help=_output_help(spec),
            path=f"network_data_format/outputs/{key}",
        )

    return parsed_inputs, parsed_outputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# TODO: add modality/format-aware branching e.g. map fmt: "dicom" to a DICOM reader type instead of a generic file path
def _map_type(spec: dict) -> type:
    """Map a network_data_format entry to a Python/fileformats type."""
    fmt = spec.get("format", "")
    data_type = spec.get("type", "")
    modality = spec.get("modality", "")

    if data_type == "image" or fmt in ("hounsfield", "segmentation"):
        try:
            from fileformats.medimage import NiftiGzX

            return NiftiGzX
        except ImportError:
            pass

    return ty.Any


def _input_help(spec: dict) -> str:
    parts = []
    if modality := spec.get("modality"):
        parts.append(f"{modality} image")
    if fmt := spec.get("format"):
        parts.append(f"format: {fmt}")
    if ch := spec.get("num_channels"):
        parts.append(f"{ch} channel(s)")
    if shape := spec.get("spatial_shape"):
        parts.append(f"patch size: {shape}")
    return ", ".join(parts) if parts else "Input image"


def _output_help(spec: dict) -> str:
    parts = []
    if fmt := spec.get("format"):
        parts.append(f"format: {fmt}")
    if ch := spec.get("num_channels"):
        parts.append(f"{ch} channel(s)")
    if shape := spec.get("spatial_shape"):
        parts.append(f"patch size: {shape}")
    return ", ".join(parts) if parts else "Output image"


def name_from_spec(spec_path: Path | str) -> str:
    """Derive a valid Python class name from a MONAI bundle path or metadata.

    Parameters
    ----------
    spec_path : Path | str
        Path to the metadata.json or bundle root directory.
    """
    spec_path = Path(spec_path)
    metadata_path = (
        spec_path / "configs" / "metadata.json" if spec_path.is_dir() else spec_path
    )

    # Try to use the "name" field from metadata
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        raw_name = metadata.get("name", "")
        if raw_name:
            return _to_class_name(raw_name)
    except (OSError, json.JSONDecodeError):
        pass

    # Fall back to the directory/file stem
    stem = spec_path.stem if spec_path.is_file() else spec_path.name
    return _to_class_name(stem)


def _to_class_name(s: str) -> str:
    """Convert an arbitrary string to a valid CamelCase Python identifier."""
    import re

    # Split on non-alphanumeric characters
    words = re.split(r"[^a-zA-Z0-9]+", s)
    return "".join(w.capitalize() for w in words if w)
