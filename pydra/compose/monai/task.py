import attrs
import copy
import typing as ty
import logging
from pathlib import Path
from fileformats.core import from_paths
from fileformats.core.exceptions import FormatRecognitionError
from fileformats.medimage import MedicalImagingData
from pydra.compose import base
from pydra.utils import get_fields
from . import fields


logger = logging.getLogger("pydra.compose.monai")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job



@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiOutputs(base.Outputs):

    BASE_OUTPUT_ATTRS = ("stdout", "stderr", "return_code")

    @classmethod
    def _from_job(cls, job: "Job[MonaiTask]") -> ty.Self:
        """Collect outputs by reading the bundle's postprocessing config.

        For each non-base output field, find a SaveImage / SaveImaged
        transform in inference.json whose keys include that field name, then
        construct the expected output path from the transform's
        ``output_postfix`` and ``output_ext`` and the source image's filename.
        """
        outputs = super()._from_job(job)
        output_dir = Path(job.cache_dir)

        if not output_dir.exists():
            return outputs

        try:
            bundle_dir = job.task._resolve_bundle_dir(job)
            save_specs = _resolved_save_specs(bundle_dir, output_dir, job.task)
        except Exception as exc:
            logger.warning(
                "Could not parse postprocessing for output resolution: %s", exc
            )
            return outputs

        input_stem = _first_input_stem(job.task)

        for field in get_fields(cls):
            if field.name in cls.BASE_OUTPUT_ATTRS:
                continue
            layout = save_specs.get(field.name)
            if layout is None:
                logger.warning(
                    "No SaveImage(d) transform writes output %r; field unset",
                    field.name,
                )
                continue
            if input_stem is None:
                continue
            expected = Path(
                layout.filename(subject=Path(input_stem).name)
            )
            if not expected.suffix and field.type is not ty.Any:
                # The transform declared no output_ext; fall back to the
                # extension implied by the declared output type.
                ext = getattr(field.type, "ext", None) or ""
                if ext:
                    expected = expected.with_name(expected.name + ext)
            if expected.is_file():
                setattr(outputs, field.name, expected)
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


def _resolved_save_specs(bundle_dir: Path, output_dir: Path, task) -> "dict[str, ty.Any]":
    """Map each saved key to the ``FolderLayout`` that names its output file.

    The bundle is resolved through ``ConfigWorkflow`` so that ``@``-references
    (``"output_postfix": "@output_postfix"``) carry their real values and
    ``keys`` has been normalised to a tuple by MONAI itself. Asking the
    instantiated ``SaveImage`` for its layout means postfix, extension and
    ``separate_folder`` nesting are all applied by the same code that wrote
    the file, rather than reconstructed here.
    """
    from monai.data.folder_layout import FolderLayout
    from monai.transforms.io.array import SaveImage
    from monai.transforms.io.dictionary import SaveImaged

    workflow = task._build_workflow(bundle_dir, output_dir, task)
    # Resolve the config without running the evaluator: parse() applies the
    # overrides and expands references, which is all the layout needs.
    workflow.parser.parse(reset=True)
    postprocessing = workflow.parser.get_parsed_content(
        "postprocessing", instantiate=True
    )

    specs: dict[str, ty.Any] = {}
    for transform in getattr(postprocessing, "transforms", []) or []:
        if isinstance(transform, SaveImaged):
            keys, saver = transform.keys, transform.saver
        elif isinstance(transform, SaveImage):
            # Undecorated SaveImage carries no keys; nothing to attribute it to.
            continue
        else:
            continue

        layout = getattr(saver, "folder_layout", None)
        if not isinstance(layout, FolderLayout):
            continue
        # Querying a layout must not create directories as a side effect.
        layout = copy.copy(layout)
        layout.makedirs = False
        for key in keys:
            specs[str(key)] = layout
    return specs


def _read_inference_config(bundle_dir: Path) -> "dict":
    """Load a bundle's raw inference.json, or ``{}`` if unreadable."""
    import json as _json

    config_file = bundle_dir / "configs" / "inference.json"
    if not config_file.is_file():
        return {}
    try:
        config = _json.loads(config_file.read_text())
    except ValueError:
        return {}
    return config if isinstance(config, dict) else {}


def _config_has_key(bundle_dir: Path, key: str) -> bool:
    """Whether the bundle's inference config defines a top-level ``key``."""
    return key in _read_inference_config(bundle_dir)


def _bundle_image_key(bundle_dir: Path) -> "str | None":
    """Return the dataset key a bundle's preprocessing reads its image from.

    Prefers an explicit top-level ``image_key``; otherwise falls back to the
    sole declared input in the metadata's ``network_data_format``. Returns
    ``None`` when the bundle expects more than one input, since binding is
    then ambiguous.
    """
    import json as _json

    image_key = _read_inference_config(bundle_dir).get("image_key")
    if isinstance(image_key, str) and image_key and not image_key.startswith("@"):
        return image_key

    metadata_file = bundle_dir / "configs" / "metadata.json"
    if not metadata_file.is_file():
        return None
    try:
        metadata = _json.loads(metadata_file.read_text())
    except ValueError:
        return None
    inputs = metadata.get("network_data_format", {}).get("inputs", {})
    if isinstance(inputs, dict) and len(inputs) == 1:
        return next(iter(inputs))
    return None


def _first_input_stem(task) -> "str | None":
    """Return the stem (sans image extension) of the first non-BASE input.

    Skips pydra-internal fields (those whose names start with ``_``) and
    the declared BASE_ATTRS so only user-facing image / data fields are
    considered.

    Uses the field's fileformats type to determine which extensions to strip.
    Falls back to ``_stem_of`` when the type is unknown (e.g. ``ty.Any``),
    which handles compound extensions such as ``.nii.gz``.
    """
    for field in get_fields(task):
        if field.name in MonaiTask.BASE_ATTRS:
            continue
        val = getattr(task, field.name, None)
        if val is None or val is attrs.NOTHING:
            continue
        if not isinstance(val, MedicalImagingData):
            try:
                val = from_paths([val], *MedicalImagingData.subclasses())
            except FormatRecognitionError:
                return Path(val).parent / Path(val).name.split('.')[0]
        return val.stem
    return None


BUNDLE_HELP = (
    "Path or name of the MONAI bundle to run (a bundle directory, a "
    "weights file inside one, or a Model Zoo bundle name such as "
    "'spleen_ct_segmentation'). When the task class is created via "
    "define(bundle_path), this field defaults to that path; supply "
    "an explicit value to override."
)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiTask(base.Task[MonaiOutputsType]):

    _executor_name = "bundle"

    BASE_ATTRS = ("bundle",)

    bundle: str = fields.arg(
        name="bundle",
        type=ty.Any,
        help=BUNDLE_HELP,
    )

    def _run(self, job: "Job[MonaiTask]", rerun: bool = True) -> None:
        """Run inference using a MONAI bundle.

        Loads configs/inference.json from the bundle directory indicated by
        ``bundle``, overrides the dataset input paths and output
        directory with values from the job, then runs the bundle evaluator.

        Parameters
        ----------
        job : Job[MonaiTask]
            The Pydra job carrying input field values and output_dir.
        rerun : bool
            Passed through from Pydra; unused here.
        """
        bundle_dir = self._resolve_bundle_dir(job)
        output_dir = Path(job.cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        workflow = self._build_workflow(bundle_dir, output_dir, job.task)

        logger.info("Running MONAI bundle inference from %s", bundle_dir)

        # ConfigWorkflow runs the bundle's own initialize -> run -> finalize
        # lifecycle. "initialize" is what loads the checkpoint into the network;
        # driving the evaluator directly would skip it and leave the network
        # randomly initialised, producing well-formed but meaningless output.
        workflow.initialize()
        if "run" in workflow.parser:
            workflow.run()
        else:
            # The bundle spec mandates a "run" section, but fall back to the
            # evaluator rather than refusing a bundle that omits it. The
            # checkpoint is already loaded, since initialize() ran above.
            logger.warning(
                "Bundle %s defines no 'run' section; invoking evaluator directly",
                bundle_dir,
            )
            workflow.parser.get_parsed_content("evaluator", instantiate=True).run()
        workflow.finalize()

    def _build_workflow(
        self, bundle_dir: Path, output_dir: Path, task: "MonaiTask"
    ) -> ty.Any:
        """Construct a ``ConfigWorkflow`` for ``bundle_dir``, ready to initialize.

        Overrides are passed to the constructor rather than assigned afterwards
        because ``ConfigWorkflow.initialize()`` re-parses with ``reset=True``.
        """
        from .spec_parser import _import_monai_bundle

        ConfigWorkflow = _import_monai_bundle().ConfigWorkflow

        overrides: dict[str, ty.Any] = {
            # bundle_root defaults to "." in most bundles, so every path derived
            # from it (notably "$@bundle_root + '/models/model.pt'") would
            # otherwise resolve against the process CWD, not the bundle.
            "bundle_root": str(bundle_dir),
            "output_dir": str(output_dir),
        }

        data_entry = self._build_data_entry(bundle_dir, task)
        if data_entry:
            overrides["dataset#data"] = [data_entry]

        overrides.update(self._device_overrides(bundle_dir))

        return ConfigWorkflow(
            config_file=str(bundle_dir / "configs" / "inference.json"),
            meta_file=str(bundle_dir / "configs" / "metadata.json"),
            # Bundle logging.conf routinely reconfigures the root logger, which
            # would hijack logging for the whole pydra process.
            logging_file=False,
            workflow_type="inference",
            **overrides,
        )

    @staticmethod
    def _build_data_entry(bundle_dir: Path, task: "MonaiTask") -> "dict[str, str]":
        """Map task input fields to the dataset keys the bundle expects.

        A bundle's preprocessing keys off its own ``image_key`` (commonly
        ``"image"``), which need not match the pydra field name. Where the
        bundle declares a single input and the task supplies a single value,
        the value is bound to the bundle's key so a differently-named field
        still reaches preprocessing instead of being silently ignored.
        """
        data_entry: dict[str, str] = {}
        for field in get_fields(task):
            if field.name in MonaiTask.BASE_ATTRS:
                continue
            val = getattr(task, field.name, None)
            # Unset attrs fields hold the NOTHING sentinel, which is not None.
            if val is not None and val is not attrs.NOTHING:
                data_entry[field.name] = str(val)

        if len(data_entry) != 1:
            return data_entry

        image_key = _bundle_image_key(bundle_dir)
        if image_key is None or image_key in data_entry:
            return data_entry

        (field_name, value), = data_entry.items()
        logger.debug(
            "Binding input field %r to bundle image_key %r", field_name, image_key
        )
        return {image_key: value}

    @staticmethod
    def _device_overrides(bundle_dir: Path) -> "dict[str, str]":
        """CPU overrides for bundles whose checkpoint was saved on CUDA.

        ``CheckpointLoader`` deserialises to the device recorded in the file, so
        setting ``device`` alone is not enough -- it takes its own
        ``map_location``.
        """
        import torch

        if torch.cuda.is_available():
            return {}

        cpu = "$torch.device('cpu')"
        overrides = {"device": cpu}
        if _config_has_key(bundle_dir, "checkpointloader"):
            overrides["checkpointloader#map_location"] = cpu
        return overrides

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_bundle_dir(self, job: "Job[MonaiTask]") -> Path:
        """Return the bundle root directory.

        ``bundle`` may be:
        - a directory (the bundle root itself)
        - a path to a ``.pt`` / ``.ts`` weights file inside the bundle
        - a MONAI Model Zoo bundle name (e.g. ``"spleen_ct_segmentation"``)
          with no path separators or file extension — in that case the
          bundle is downloaded via ``monai.bundle.load(source="monaihosting")``
        """
        bundle = getattr(job.task, "bundle", None)
        if bundle is None:
            raise ValueError("bundle must be set before running a MonaiTask")

        path = Path(str(bundle))

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
        bundle_str = str(bundle)
        if (
            "/" in bundle_str
            or "\\" in bundle_str
            or Path(bundle_str).suffix != ""
        ):
            raise ValueError(
                f"bundle={bundle_str!r} is not a valid MONAI bundle "
                "reference. Provide one of: an existing bundle directory, an "
                "existing weights file inside a bundle, or a Model Zoo bundle "
                "name (e.g. 'spleen_ct_segmentation')."
            )

        from .spec_parser import _import_monai_bundle
        bundle_load = _import_monai_bundle().load
        logger.info("Downloading MONAI bundle %s", bundle_str)
        bundle_dir = bundle_load(bundle_str, source="monaihosting")
        return Path(bundle_dir)
