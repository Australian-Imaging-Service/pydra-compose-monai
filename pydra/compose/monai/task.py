import attrs
import typing as ty
from pathlib import Path
import logging
from frametree.core import __version__
from frametree.core.frameset import FrameSet
from frametree.axes.medimage import MedImage
from frametree.bids.store import Bids
from pydra.utils import asdict, get_fields
from pydra.compose import base
from pydra.environments.docker import Docker
from pydra.environments.native import Native
from . import fields
from .app import BidsApp

logger = logging.getLogger("pydra.compose.monai")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class BidsAppOutputs(base.Outputs):

    @classmethod
    def _from_job(cls, job: "Job[BidsAppTask]") -> ty.Self:
        """Collect the outputs of a job from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        job : Job[Task]
            The job whose outputs are being collected.
        outputs_dict : dict[str, ty.Any]
            The outputs of the job, as a dictionary

        Returns
        -------
        outputs : Outputs
            The outputs of the job in dataclass
        """
        outputs = super()._from_job(job)
        frameset: FrameSet = job.return_values["frameset"]
        output_fields: ty.List[fields.out] = get_fields(cls)
        for output_field in output_fields:
            if output_field.path:
                path = output_field.path
            else:
                path = ""  # whole directory
            path += "@" + DEFAULT_DERIVATIVES_NAME
            frameset.add_sink(
                output_field.name,
                output_field.type,
                path=path,
            )
        row = frameset.row(MedImage.session, DEFAULT_BIDS_ID)
        with frameset.store.connection:
            for output_field in output_fields:
                setattr(outputs, output_field.name, row[output_field.name])
        return outputs


BidsAppOutputsType = ty.TypeVar("BidsAppOutputsType", bound=BidsAppOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class BidsAppTask(base.Task[BidsAppOutputsType]):

    _executor_name = "app"

    BASE_ATTRS = (
        "analysis_level",
        "json_edits",
        "flags",
        "app",
        "work_dir",
    )

    analysis_level: str = fields.arg(
        name="analysis_level",
        type=str,
        default="participant",
        help="Level of analysis to run the app at",
        path=None,
    )
    json_edits: list[tuple[str, str]] | None = fields.arg(
        name="json_edits",
        type=list[tuple[str, str]] | None,
        default=None,
        path=None,
    )
    flags: str | None = fields.arg(
        name="flags",
        type=str | None,
        default=None,
        help=(
            "Additional flags to pass to the app. These are passed as a single string "
            "and should be formatted as they would be on the command line "
            "(e.g. '--flag1 --flag2 value')"
        ),
        path=None,
    )
    work_dir: Path | None = fields.arg(
        type=Path | None,
        default=None,
        help="The directory where the temporary BIDS dataset will be created and Pydra cache stored.",
        path=None,
    )

    def _run(self, job: "Job[BidsAppTask]", rerun: bool = True) -> None:
        # Create a BIDS dataset and save input data into it
        job.return_values["frameset"] = frameset = self._create_dataset()
        output_dir = (
            Path(frameset.id)
            / "derivatives"
            / DEFAULT_DERIVATIVES_NAME
            / f"sub-{DEFAULT_BIDS_ID}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.work_dir:
            work_dir = self.work_dir
        else:
            work_dir = Path.cwd() / "work-dir"
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_root = work_dir / "internal-cache"
        app_work_dir = work_dir / "app-work-dir"
        cache_root.mkdir()
        app_work_dir.mkdir()

        if self.app.startswith("/"):
            executable = self.app
            image_tag = None
        else:
            if self.app.startswith("docker://"):
                image_tag = self.app.split("docker://")[-1]
            else:
                logger.info(
                    "Assuming that the wrapped string, '%s' is a docker image tag. ",
                    self.app,
                )
                image_tag = self.app
            if "::" in image_tag:
                image_tag, executable = image_tag.split("::")
            else:
                executable = None  # entrypoint of the container

        app = BidsApp(
            executable=executable,
            dataset_path=frameset.id,
            output_path=output_dir,
            analysis_level=self.analysis_level,
            participant_label=DEFAULT_BIDS_ID,
            flags=self.flags,
            work_dir=app_work_dir,
        )
        environment = Docker(image_tag) if image_tag else Native()
        app(cache_root=cache_root, environment=environment)

    def _create_dataset(self) -> FrameSet:
        # Prepare the inputs to the function
        inputs = asdict(self)
        for atr in self.BASE_ATTRS:
            del inputs[atr]

        dataset_dir = Path.cwd() / "bids-dataset"
        frameset = Bids().create_dataset(
            id=dataset_dir,
            name=DEFAULT_FRAMESET_NAME,
            leaves=[(DEFAULT_BIDS_ID,)],
            metadata={"authors": [f"Auto-generated by FrameTree {__version__}"]},
        )

        # Update the Bids store with the JSON edits requested by the user
        # je_args = shlex.split(self.json_edits) if json_edits else []
        bids_store: Bids = frameset.store
        bids_store.json_edits = self.json_edits

        # JsonEdit.attr_converter(
        #     self.json_edits + list(zip(je_args[::2], je_args[1::2]))
        # )
        input_fields: list[fields.arg] = [
            f for f in get_fields(self) if f.name not in self.BASE_ATTRS
        ]
        for inpt in input_fields:
            frameset.add_sink(inpt.name, inpt.type, path=inpt.path)
        row = frameset.row(MedImage.session, DEFAULT_BIDS_ID)
        with frameset.store.connection:
            for inpt in input_fields:
                inpt_value = inputs[inpt.name]
                if not inpt_value:
                    logger.warning("No input provided for '%s' input", inpt.name)
                    continue
                row[inpt.name] = inpt_value
        return frameset


# For running
CONTAINER_DERIV_PATH = "/frametree_bids_outputs"
CONTAINER_DATASET_PATH = "/frametree_bids_dataset"

DEFAULT_BIDS_ID = "DEFAULT"
DEFAULT_FRAMESET_NAME = "DEFAULT"
DEFAULT_DERIVATIVES_NAME = "DEFAULT"
OUTPUT_DIR_NAME = "output-dir"
