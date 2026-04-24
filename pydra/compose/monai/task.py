import attrs
import typing as ty
import logging
from pathlib import Path
from pydra.compose import base
from . import fields


logger = logging.getLogger("pydra.compose.monai")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiOutputs(base.Outputs):

    @classmethod
    def _from_job(cls, job: "Job[MonaiTask]") -> ty.Self:
        """Collect outputs after inference by scanning the job's output directory.

        Parameters
        ----------
        job : Job[MonaiTask]
            The completed job whose output directory contains inference results.

        Returns
        -------
        outputs : MonaiOutputs
            Populated outputs dataclass.
        """
        outputs = super()._from_job(job)
        output_dir = Path(job.output_dir)

        for field in attrs.fields(cls):
            # Skip base Outputs fields (stdout, stderr, return_code)
            if field.name.startswith("_") or not output_dir.exists():
                continue
            candidates = sorted(output_dir.glob(f"{field.name}.*"))
            if candidates:
                object.__setattr__(outputs, field.name, candidates[0])
            else:
                # also try any file whose stem contains the field name
                candidates = sorted(output_dir.glob(f"*{field.name}*"))
                if candidates:
                    object.__setattr__(outputs, field.name, candidates[0])

        return outputs


MonaiOutputsType = ty.TypeVar("MonaiOutputsType", bound=MonaiOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class MonaiTask(base.Task[MonaiOutputsType]):

    BASE_ATTRS = (
        "model_weights",
        "arch",
    )

    model_weights: str = fields.arg(
        name="model_weights",
        type=ty.Any,
        help="the weights of the model",
    )
    arch: list[tuple[str, str]] | None = fields.arg(
        name="arch", type=ty.Any, help="the architecture of the model"
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
        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parser = ConfigParser()
        parser.read_meta(str(bundle_dir / "configs" / "metadata.json"))
        parser.load_config_file(str(bundle_dir / "configs" / "inference.json"))

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

        # Treat as a Hugging Face / MONAI Model Zoo repo ID and download
        from .spec_parser import _import_monai_bundle
        bundle_load = _import_monai_bundle().load
        logger.info("Downloading MONAI bundle %s", weights)
        bundle_dir = bundle_load(str(weights), source="monaihosting")
        return Path(bundle_dir)
