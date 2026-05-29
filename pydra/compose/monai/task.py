import attrs
import typing as ty
import logging
from pathlib import Path
from pydra.compose import base
from . import fields


logger = logging.getLogger("pydra.compose.monai")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


def _job_output_dir(job) -> Path:
    """Return the output directory for a job.

    The real pydra ``Job`` exposes ``cache_dir`` (= ``cache_root / checksum``)
    as its working/output directory.  The ``FakeJob`` stub used in unit tests
    carries an explicit ``output_dir`` attribute instead.  This helper
    abstracts over both so that ``_run`` and ``_from_job`` work with either.
    """
    if hasattr(job, "output_dir"):
        return Path(job.output_dir)
    return Path(job.cache_dir)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiOutputs(base.Outputs):

    BASE_OUTPUT_ATTRS = ("stdout", "stderr", "return_code")
    _IMAGE_EXTS = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd", ".dcm")

    @classmethod
    def _from_job(cls, job: "Job[MonaiTask]") -> ty.Self:
        """Collect outputs by reading the bundle's postprocessing config.

        For each non-base output field, find a SaveImage / SaveImaged
        transform in inference.json whose keys include that field name, then
        construct the expected output path from the transform's
        ``output_postfix`` and ``output_ext`` and the source image's filename.
        """
        outputs = super()._from_job(job)
        output_dir = _job_output_dir(job)

        if not output_dir.exists():
            return outputs

        try:
            bundle_dir = job.task._resolve_bundle_dir(job)
            save_specs = _parse_save_transforms(bundle_dir / "configs" / "inference.json")
        except Exception as exc:
            logger.warning(
                "Could not parse postprocessing for output resolution: %s", exc
            )
            return outputs

        input_stem = _first_input_stem(job.task)

        for field in attrs.fields(cls):
            if field.name.startswith("_") or field.name in cls.BASE_OUTPUT_ATTRS:
                continue
            spec = save_specs.get(field.name)
            if spec is None:
                logger.warning(
                    "No SaveImage(d) transform writes output %r; field unset",
                    field.name,
                )
                continue
            if input_stem is None:
                continue
            postfix = spec.get("output_postfix", "")
            ext = spec.get("output_ext", ".nii.gz")
            if not ext.startswith("."):
                ext = "." + ext
            if postfix:
                fname = f"{input_stem}_{postfix}{ext}"
            else:
                fname = f"{input_stem}{ext}"
            expected = output_dir / fname
            if expected.is_file():
                object.__setattr__(outputs, field.name, expected)
            else:
                logger.warning(
                    "Expected output %s not found; field %r left unset",
                    expected, field.name,
                )

        return outputs


MonaiOutputsType = ty.TypeVar("MonaiOutputsType", bound=MonaiOutputs)


# ---------------------------------------------------------------------------
# Module-level helpers for R1 postprocessing-driven output path resolution
# ---------------------------------------------------------------------------


def _parse_save_transforms(inference_json: Path) -> "dict[str, dict]":
    """Walk an inference.json's postprocessing for SaveImage(d) transforms.

    Returns a mapping ``{output_field_name: {"output_postfix": ..., "output_ext": ...}}``.
    """
    import json as _json

    if not inference_json.is_file():
        return {}
    config = _json.loads(inference_json.read_text())
    node = config.get("postprocessing")
    transforms = _extract_transforms(node)

    out: "dict[str, dict]" = {}
    for t in transforms:
        if not isinstance(t, dict):
            continue
        target = str(t.get("_target_", ""))
        if not (target.endswith("SaveImage") or target.endswith("SaveImaged")):
            continue
        keys = t.get("keys")
        if keys is None and "key" in t:
            keys = [t["key"]]
        if not keys:
            continue
        for key in keys:
            out[str(key)] = {
                "output_postfix": t.get("output_postfix", ""),
                "output_ext": t.get("output_ext", ".nii.gz"),
            }
    return out


def _extract_transforms(node) -> list:
    """Flatten a postprocessing node into a list of transform dicts."""
    if node is None:
        return []
    if isinstance(node, list):
        result = []
        for item in node:
            result.extend(_extract_transforms(item))
        return result
    if isinstance(node, dict):
        target = str(node.get("_target_", ""))
        if "Compose" in target:
            return _extract_transforms(node.get("transforms", []))
        return [node]
    return []


def _first_input_stem(task) -> "str | None":
    """Return the stem (sans image extension) of the first non-BASE input.

    Skips pydra-internal fields (those whose names start with ``_``) and
    the declared BASE_ATTRS so only user-facing image / data fields are
    considered.
    """
    for field in attrs.fields(type(task)):
        if field.name.startswith("_"):
            continue
        if field.name in MonaiTask.BASE_ATTRS:
            continue
        val = getattr(task, field.name, None)
        if val is None:
            continue
        name = Path(str(val)).name
        for ext in MonaiOutputs._IMAGE_EXTS:
            if name.lower().endswith(ext):
                return name[: -len(ext)]
        return Path(name).stem
    return None


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiTask(base.Task[MonaiOutputsType]):

    BASE_ATTRS = ("model_weights",)

    model_weights: str = fields.arg(
        name="model_weights",
        type=ty.Any,
        help="the weights of the model",
    )

    def _run(self, job: "Job[MonaiTask]", rerun: bool = True) -> None:
        """Run inference using a MONAI bundle.

        Loads configs/inference.json from the bundle directory indicated by
        ``model_weights``, overrides the dataset input paths and output
        directory with values from the job, then runs the bundle evaluator.

        Parameters
        ----------
        job : Job[MonaiTask]
            The Pydra job carrying input field values and output_dir.
        rerun : bool
            Passed through from Pydra; unused here.
        """
        from .spec_parser import _import_monai_bundle
        ConfigParser = _import_monai_bundle().ConfigParser

        bundle_dir = self._resolve_bundle_dir(job)
        output_dir = _job_output_dir(job)
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = ConfigParser()
        parser.read_meta(str(bundle_dir / "configs" / "metadata.json"))
        parser.read_config(str(bundle_dir / "configs" / "inference.json"))

        # Build the dataset entries from job inputs, keyed by field name.
        # Each entry in network_data_format.inputs becomes a key in the data dict.
        data_entry: dict[str, str] = {}
        for field in attrs.fields(type(job.task)):
            if field.name in MonaiTask.BASE_ATTRS:
                continue
            val = getattr(job.task, field.name, None)
            if val is not None:
                data_entry[field.name] = str(val)

        if data_entry:
            parser["dataset#data"] = [data_entry]

        parser["output_dir"] = str(output_dir)

        logger.info("Running MONAI bundle inference from %s", bundle_dir)
        evaluator = parser.get_parsed_content("evaluator", instantiate=True)
        evaluator.run()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_bundle_dir(self, job: "Job[MonaiTask]") -> Path:
        """Return the bundle root directory.

        ``model_weights`` may be:
        - a directory (the bundle root itself)
        - a path to a ``.pt`` / ``.ts`` weights file inside the bundle
        - a Hugging Face repo ID string (``org/model_name``) — in that case
          the bundle is downloaded via ``monai.bundle.load``
        """
        weights = getattr(job.task, "model_weights", None)
        if weights is None:
            raise ValueError("model_weights must be set before running a MonaiTask")

        path = Path(str(weights))

        if path.is_dir():
            if not (path / "configs" / "metadata.json").is_file():
                raise ValueError(
                    f"Bundle directory {path} does not contain configs/metadata.json. "
                    "Pass a path to a valid MONAI bundle root."
                )
            return path

        if path.is_file():
            # weights file lives inside the bundle tree — walk up to find configs/
            for parent in path.parents:
                if (parent / "configs" / "metadata.json").exists():
                    return parent
            raise ValueError(
                f"Cannot locate bundle root from weights file {path}. "
                "Expected configs/metadata.json in a parent directory."
            )

        # If it's not a path on disk, the only remaining valid form is a
        # MONAI Model Zoo bundle name (e.g. "spleen_ct_segmentation").
        # Bundle names contain no path separators and no file extension.
        weights_str = str(weights)
        if (
            "/" in weights_str
            or "\\" in weights_str
            or Path(weights_str).suffix != ""
        ):
            raise ValueError(
                f"model_weights={weights_str!r} is not a valid MONAI bundle "
                "reference. Provide one of: an existing bundle directory, an "
                "existing weights file inside a bundle, or a Model Zoo bundle "
                "name (e.g. 'spleen_ct_segmentation')."
            )

        from .spec_parser import _import_monai_bundle
        bundle_load = _import_monai_bundle().load
        logger.info("Downloading MONAI bundle %s", weights_str)
        bundle_dir = bundle_load(weights_str, source="monaihosting")
        return Path(bundle_dir)
