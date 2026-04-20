"""Parse MONAI bundle metadata.json into Pydra arg/out field definitions."""

import importlib
import re
import sys
import types
import typing as ty
from pathlib import Path

from .fields import arg, out


def _import_monai_bundle() -> types.ModuleType:
    """Import the PyPI ``monai.bundle`` module.

    Guards against the local ``pydra.compose.monai`` subpackage shadowing the
    top-level ``monai`` name when pytest adds ``pydra/compose/`` to sys.path.
    Temporarily removes any ``pydra/compose`` entries from sys.path so that
    ``monai`` resolves to the site-packages installation.
    """
    # Find path entries that would cause pydra/compose/monai to shadow 'monai'
    _this_pkg = str(Path(__file__).parent.parent)  # .../pydra/compose
    shadow_entries = [p for p in sys.path if Path(p).resolve() == Path(_this_pkg).resolve()]

    for p in shadow_entries:
        sys.path.remove(p)

    # Also clear any stale sys.modules entries from a previous shadow import
    stale = [k for k in list(sys.modules)
             if (k == "monai" or k.startswith("monai."))
             and getattr(sys.modules[k], "__file__", "").startswith(_this_pkg)]
    for k in stale:
        del sys.modules[k]

    try:
        return importlib.import_module("monai.bundle")
    finally:
        sys.path.extend(shadow_entries)


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
    ConfigParser = _import_monai_bundle().ConfigParser

    spec_path = Path(spec_path)
    metadata_path = (
        spec_path / "configs" / "metadata.json" if spec_path.is_dir() else spec_path
    )

    parser = ConfigParser()
    parser.read_meta(str(metadata_path))

    raw_inputs = parser.get_parsed_content(
        "_meta_#network_data_format#inputs", instantiate=False
    )
    raw_outputs = parser.get_parsed_content(
        "_meta_#network_data_format#outputs", instantiate=False
    )

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


def _map_type(spec: dict) -> type:
    """Map a network_data_format entry to a fileformats type.

    Mirrors reader-selection in monai.data image readers:
    - DICOM → fileformats.medimage.DicomSeries
    - NIfTI images / segmentations / MRI / CT → fileformats.medimage.NiftiGzX
    - Unknown → ty.Any
    """
    fmt = (spec.get("format") or "").lower()
    data_type = (spec.get("type") or "").lower()
    modality = (spec.get("modality") or "").lower()

    if fmt == "dicom" or data_type == "dicom_series":
        try:
            from fileformats.medimage import DicomSeries
            return DicomSeries
        except ImportError:
            pass

    if (
        data_type == "image"
        or fmt in ("hounsfield", "segmentation", "magnitude", "mri")
        or modality in ("ct", "mri", "mr", "pt", "nm")
    ):
        try:
            from fileformats.medimage import NiftiGzX
            return NiftiGzX
        except ImportError:
            pass

    return ty.Any  # type: ignore[return-value]


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
    ConfigParser = _import_monai_bundle().ConfigParser

    spec_path = Path(spec_path)
    metadata_path = (
        spec_path / "configs" / "metadata.json" if spec_path.is_dir() else spec_path
    )

    try:
        parser = ConfigParser()
        parser.read_meta(str(metadata_path))
        raw_name = parser.get_parsed_content("_meta_#name", instantiate=False)
        if raw_name:
            return _to_class_name(str(raw_name))
    except Exception:
        pass

    stem = spec_path.stem if spec_path.is_file() else spec_path.name
    return _to_class_name(stem)


def _to_class_name(s: str) -> str:
    """Convert an arbitrary string to a valid CamelCase Python identifier."""
    words = re.split(r"[^a-zA-Z0-9]+", s)
    return "".join(w.capitalize() for w in words if w)
